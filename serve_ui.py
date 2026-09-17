"""本机的 Agent 网页界面服务（只用标准库，不引入 Web 框架）。

    python main.py ui            # 打开 http://127.0.0.1:8760
    python serve_ui.py --port 9000 --no-browser

两个页面：
    /           主界面：左侧会话列表，右侧对话区，左下角「更多」
    /settings   设置窗口（点「更多」在新窗口打开）：API 配置 / 知识库 两个页签

接口：
    GET    /api/bootstrap                       一次性拿全：工作区 / 会话 / 知识库 / 模型
    POST   /api/workspace/folder                切换工作区文件夹 {path}
    POST   /api/fs/list                         文件夹浏览器：列出某个目录的子目录 {path}

    POST   /api/conversations                   新建会话
    GET    /api/conversations/{id}              读取会话（含全部历史消息）
    POST   /api/conversations/{id}/delete       删除会话

    GET    /api/kb                              知识库状态（文件清单 + 片段数）
    POST   /api/kb/upload                       导入文档（JSON + base64，写进工作区文件夹）
    POST   /api/kb/exclude                      把某个文件移出知识库 {name}（不删原文件）
    POST   /api/kb/include                      重新纳入 {name}
    POST   /api/kb/clear                        全部移出知识库

    POST   /api/chat                            跑一次 Agent（SSE），并写入会话历史

数据落盘在 public/appdata/：对话与索引都在应用自己的目录里，
工作区只负责提供文档，不往里写应用数据。

安全边界：默认只监听 127.0.0.1。API Key 由浏览器随请求发过来、只用于当次调用，
不写入任何文件；服务端也会回落到环境变量里的 key。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UI_DIR = ROOT / "webui"
DIST_DIR = UI_DIR / "dist"

PAGES = {
    "/": UI_DIR / "index.html",
    "/index.html": UI_DIR / "index.html",
    "/settings": UI_DIR / "settings.html",
    "/settings.html": UI_DIR / "settings.html",
}

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".woff2": "font/woff2",
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AgentEnterpriseUI/2.2"

    # ---- 基础工具 ------------------------------------------------------
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[ui] {self.address_string()} {fmt % args}\n")

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200):
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"请求体不是合法 JSON：{exc}") from exc
        return data if isinstance(data, dict) else {}

    # ---- GET -----------------------------------------------------------
    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler 的约定命名
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in PAGES:
                self._serve_file(PAGES[path])
            elif path.startswith("/dist/"):
                relative = Path(unquote(path[len("/dist/") :]))
                target = (DIST_DIR / relative).resolve()
                if not str(target).startswith(str(DIST_DIR.resolve())):
                    self._send_json({"error": "非法路径"}, status=400)
                    return
                self._serve_file(target)
            elif path == "/api/bootstrap":
                self._send_json(self.bootstrap_payload())
            elif path == "/api/providers":
                self._send_json(self.providers_payload())
            elif path == "/api/kb":
                from app.session import knowledge_summary

                self._send_json(knowledge_summary())
            elif path == "/api/fs/list":
                from app.workspace import list_directory

                query = parse_qs(parsed.query)
                self._send_json(list_directory((query.get("path") or [""])[0]))
            elif path.startswith("/api/conversations/"):
                from app import workspace as store

                self._send_json(store.get_conversation(path[len("/api/conversations/") :]))
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _serve_file(self, target: Path):
        if not target.is_file():
            hint = ""
            if target.suffix in {".js", ".css"}:
                hint = "（前端还没构建：cd webui && npm install && npm run build）"
            self._send(404, f"文件不存在：{target.name}{hint}".encode("utf-8"),
                       "text/plain; charset=utf-8")
            return
        self._send(200, target.read_bytes(),
                   CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream"))

    @staticmethod
    def providers_payload() -> dict:
        from app.config import get_settings
        from app.llm import available_providers

        return {
            "providers": available_providers(),
            "default": {"provider": "deepseek", "model": get_settings().model},
        }

    @staticmethod
    def bootstrap_payload() -> dict:
        """界面初始化一次拿全：工作区、会话、知识库、MCP、工具。"""
        from app import workspace as store
        from app.config import get_settings
        from app.mcp_client import mcp_status
        from app.session import knowledge_summary, sync_workspace
        from app.tools import all_tools

        store.ensure_layout()
        sync_workspace()

        settings = get_settings()
        try:
            mcp = mcp_status()
        except Exception as exc:
            mcp = f"MCP 不可用：{type(exc).__name__}: {exc}"

        return {
            "model": settings.model,
            "providers": Handler.providers_payload(),
            "workspace": store.workspace_info(),
            "conversations": store.list_conversations(),
            "kb": knowledge_summary(),
            "mcp": mcp,
            "tools": [tool.name for tool in all_tools()],
            "grade_enabled": settings.kb_grade_retrieval,
            "grade_score": settings.kb_grade_score,
            "top_k": settings.kb_top_k,
        }

    # ---- POST ----------------------------------------------------------
    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/chat":
                self._chat()
            elif path == "/api/kb/upload":
                self._kb_upload()
            elif path == "/api/kb/exclude":
                self._kb_exclude()
            elif path == "/api/kb/include":
                self._kb_include()
            elif path == "/api/kb/clear":
                self._kb_clear()
            elif path == "/api/workspace/folder":
                self._set_folder()
            elif path == "/api/fs/list":
                from app.workspace import list_directory

                self._send_json(list_directory(str(self._read_json_body().get("path") or "")))
            elif path == "/api/conversations":
                self._create_conversation()
            elif path.startswith("/api/conversations/") and path.endswith("/delete"):
                self._delete_conversation(path)
            else:
                self._send_json({"error": "not found"}, status=404)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    # ---- 工作区 --------------------------------------------------------
    def _set_folder(self):
        from app import workspace as store
        from app.session import knowledge_summary, sync_workspace

        body = self._read_json_body()
        info = store.set_workspace_folder(str(body.get("path") or ""))
        sync_workspace()
        self._send_json({"ok": True, "workspace": info, "kb": knowledge_summary()})

    # ---- 会话 ----------------------------------------------------------
    def _create_conversation(self):
        from app import workspace as store

        body = self._read_json_body()
        conversation = store.create_conversation(title=str(body.get("title") or "新对话"))
        self._send_json({
            "ok": True,
            "conversation": conversation,
            "conversations": store.list_conversations(),
        })

    def _delete_conversation(self, path: str):
        from app import workspace as store

        conversation_id = path[len("/api/conversations/") : -len("/delete")]
        removed = store.delete_conversation(conversation_id)
        self._send_json({"ok": removed, "conversations": store.list_conversations()})

    # ---- 知识库 --------------------------------------------------------
    def _kb_upload(self):
        """接收一批文件（JSON + base64），写进工作区文件夹并重建索引。"""
        import base64
        import binascii

        from app.documents import MAX_BATCH_BYTES
        from app.session import import_documents, knowledge_summary

        body = self._read_json_body()
        raw_files = body.get("files") or []
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError("没有收到文件")

        items: list[tuple[str, bytes]] = []
        total = 0
        for entry in raw_files:
            name = str((entry or {}).get("name") or "未命名文件")
            try:
                data = base64.b64decode((entry or {}).get("data") or "", validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"{name} 的内容不是合法的 base64") from exc
            total += len(data)
            if total > MAX_BATCH_BYTES:
                raise ValueError("一次导入的总大小超过 40MB，请分批导入")
            items.append((name, data))

        result = import_documents(items)
        result["kb"] = knowledge_summary()
        self._send_json(result, status=200 if result.get("ok") else 400)

    def _kb_exclude(self):
        from app.session import exclude_document, knowledge_summary

        body = self._read_json_body()
        name = str(body.get("name") or "").strip()
        if not name:
            raise ValueError("缺少文件名")
        result = exclude_document(name)
        result["kb"] = knowledge_summary()
        self._send_json(result, status=200 if result.get("ok") else 400)

    def _kb_include(self):
        from app.session import include_document, knowledge_summary

        body = self._read_json_body()
        name = str(body.get("name") or "").strip()
        if not name:
            raise ValueError("缺少文件名")
        result = include_document(name)
        result["kb"] = knowledge_summary()
        self._send_json(result, status=200 if result.get("ok") else 400)

    def _kb_clear(self):
        from app.session import clear_knowledge_base, knowledge_summary

        result = clear_knowledge_base()
        result["kb"] = knowledge_summary()
        self._send_json(result)

    # ---- 对话 ----------------------------------------------------------
    def _chat(self):
        from app import workspace as store
        from app.session import has_uploaded_kb, sync_workspace

        body = self._read_json_body()
        question = str(body.get("question") or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        # 每次提问前对齐一次索引：工作区文件夹里的文件可能刚被改动
        if has_uploaded_kb():
            sync_workspace()

        conversation_id = str(body.get("conversation_id") or "")
        if not conversation_id:
            conversation_id = store.create_conversation()["id"]
        # 用户消息先落盘：即使这一轮模型调用失败，提问本身也不会丢
        store.append_message(conversation_id, {"role": "user", "text": question, "at": _now()})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event: dict) -> None:
            frame = json.dumps(event, ensure_ascii=False, default=str)
            self.wfile.write(f"data: {frame}\n\n".encode("utf-8"))
            self.wfile.flush()

        emit({"type": "conversation", "id": conversation_id})

        assistant: dict = {
            "role": "assistant",
            "text": "",
            "steps": [],
            "tools": [],
            "sources": [],
            "usage": {},
            "at": _now(),
        }
        try:
            from app.llm import use_llm
            from app.runner import stream_agent_state
            from token_tracking import TokenCounter

            counter = TokenCounter()
            with use_llm(
                provider=body.get("provider"),
                model=body.get("model"),
                api_key=body.get("api_key"),
                base_url=body.get("base_url"),
            ):
                for event in stream_agent_state(question, callbacks=[counter]):
                    kind = event.get("type")
                    if kind == "node":
                        assistant["steps"].append(event.get("label") or event.get("node"))
                    elif kind == "tool":
                        assistant["tools"].append(
                            {"name": event.get("name"), "args": event.get("args")}
                        )
                    elif kind == "sources":
                        assistant["sources"].extend(event.get("items") or [])
                    elif kind == "answer":
                        assistant["text"] = event.get("text") or ""
                        assistant["status"] = event.get("status")
                    elif kind == "done":
                        assistant["seconds"] = event.get("seconds")
                    emit(event)
            assistant["usage"] = counter.snapshot()
            emit({"type": "usage", **counter.snapshot()})
        except Exception as exc:
            traceback.print_exc()
            assistant["text"] = assistant.get("text") or f"执行失败：{type(exc).__name__}: {exc}"
            assistant["error"] = f"{type(exc).__name__}: {exc}"
            emit({"type": "error", "message": assistant["error"]})
        finally:
            try:
                store.append_message(conversation_id, assistant)
            except Exception as exc:  # 落盘失败不该把响应搞崩，但必须可见
                print(f"[ui] 保存助手消息失败：{exc}")
            try:
                title = store.get_conversation(conversation_id).get("title", "")
            except Exception:
                title = ""
            emit({"type": "saved", "conversation_id": conversation_id, "title": title})
            emit({"type": "end"})


def serve(host: str, port: int, open_browser: bool) -> int:
    # 网页界面只认工作区文件夹里的文档；项目自带的公开语料不参与
    from app import workspace as store
    from app.session import set_mode, sync_workspace

    set_mode("session")
    store.ensure_layout()
    sync_workspace()

    if host not in ("127.0.0.1", "localhost"):
        print(f"[警告] 正在监听 {host}：局域网内任何人都能访问这个界面（含你的 API Key 输入）")
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print("Agent 界面已启动：" + url)
    if not (DIST_DIR / "main.js").exists():
        print("[提示] 前端还没构建，页面会报错。先执行：")
        print("       cd webui && npm install && npm run build")
    print("按 Ctrl+C 结束。")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="启动本机 Agent 网页界面")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认只允许本机访问")
    parser.add_argument("--port", type=int, default=8760, help="端口，默认 8760")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()
    return serve(args.host, args.port, not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
