"""会话知识库：**知识库来自工作区文件夹里的文档，而不是项目自带的公开数据集**。

   工作区文件夹（本机任意目录）
     ├── 手册.pdf      ┐
     ├── 说明.md       ├─→ 扫描 → 抽文本 → 语义切分 → FAISS 索引
     └── 报告.docx     ┘        （索引写在 public/appdata/kbsession/）

门控逻辑：
  - 工作区里没有可用文档 ⇒ `active_tools()` 不返回 rag_search，
    避免白调一次拿到"没找到相关内容"、再把"资料不足"当成答案依据；
  - 有文档 ⇒ 绑定 rag_search，并把"必须优先查这份文档"写进 System Prompt。

两种运行模式（`set_mode`）：
  - ``corpus``（默认，CLI/评测用）：把 public 语料库当作知识库，行为与以前一致；
  - ``session``（网页界面用）：只认工作区文件夹里用户自己的文档。
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from . import workspace as store
from .config import BASE_DIR, get_settings
from .documents import DocumentError, extract_text

_LOCK = threading.RLock()
_MODE = "corpus"
_SESSION_INDEX: Path | None = None

#: 已导入知识库时注入 System Prompt 的硬约束
UPLOAD_POLICY = (
    "本次会话用户已导入知识库文档，涉及文档内容的问题必须优先调用 rag_search，"
    "并以检索到的原文为作答依据。"
)


def set_mode(mode: str) -> None:
    """``session``：只认工作区文件夹里的文档；``corpus``：沿用项目语料库。"""
    global _MODE
    _MODE = "session" if str(mode).lower() == "session" else "corpus"


def get_mode() -> str:
    return _MODE


def mark_uploaded(files: list[str | Path] | None = None) -> None:
    """兼容旧调用：显式声明"本次会话有知识库"（评测/CLI 用）。"""
    if files is None:
        os.environ.setdefault("KB_SESSION_UPLOADED", "1")


def has_uploaded_kb() -> bool:
    """当前会话是否有可用的知识库。

    session 模式下**只认工作区文件夹里参与索引的文档**；
    corpus 模式保留旧行为，供命令行与评测使用。
    """
    if _MODE == "session":
        return bool(store.active_documents())
    if _SESSION_INDEX is not None:
        return True
    flag = os.getenv("KB_SESSION_UPLOADED", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    default_index = Path(get_settings().kb_faiss_index)
    return (default_index / "index.faiss").exists()


def knowledge_policy() -> str:
    """有知识库时追加进 System Prompt 的硬约束；没有则返回空串。"""
    return UPLOAD_POLICY if has_uploaded_kb() else ""


# ---------------------------------------------------------------- 索引构建

def _build_index(entries: list[tuple[str, str]]) -> tuple[Path, list[dict]]:
    import sys

    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    from rag_faiss import Embedder, FaissStore
    from semantic_chunking import pack_chunks, split_semantic

    embedder = Embedder(get_settings().kb_embedding_model)
    chunks: list[dict] = []
    stats: list[dict] = []
    for name, text in entries:
        raw: list = []
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                raw.extend(split_semantic(paragraph, name, name, embedder.embed))
        packed = pack_chunks(raw)
        for chunk in packed:
            chunks.append({"content": chunk.content, "metadata": chunk.metadata})
        stats.append({"name": name, "chars": len(text), "chunks": len(packed)})
    if not chunks:
        raise DocumentError("切分后没有得到任何内容")

    index_path = store.INDEX_DIR
    FaissStore.build(chunks, embedder).save(index_path)
    return index_path, stats


def _parse_documents() -> tuple[list[tuple[str, str]], list[dict]]:
    parsed: list[tuple[str, str]] = []
    failures: list[dict] = []
    for item in store.active_documents():
        try:
            path = store.document_path(item["name"])
            parsed.append((item["name"], extract_text(item["name"], path.read_bytes())))
        except DocumentError as exc:
            failures.append({"name": item["name"], "error": str(exc)})
        except Exception as exc:
            failures.append({"name": item["name"], "error": f"{type(exc).__name__}: {exc}"})
    return parsed, failures


def rebuild_index(force: bool = False) -> dict:
    """按工作区里参与索引的文档重建索引。

    用文件指纹（名字 + 大小 + 修改时间）判断是否需要重建：指纹没变就跳过，
    否则每次提问前都会白跑一遍嵌入模型。
    """
    global _SESSION_INDEX

    documents = store.active_documents()
    if not documents:
        clear_cache()
        return {"ok": True, "chunks": 0, "files": [], "failures": [], "skipped": True}

    index_path = store.INDEX_DIR
    meta_path = index_path / "meta.json"
    fingerprint = store.documents_fingerprint()
    if not force and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint and (index_path / "index.faiss").exists():
            _SESSION_INDEX = index_path
            get_settings().kb_faiss_index = str(index_path)
            return {
                "ok": True,
                "chunks": meta.get("chunks", 0),
                "files": meta.get("files", []),
                "failures": [],
                "skipped": True,
            }

    parsed, failures = _parse_documents()
    if not parsed:
        clear_cache()
        return {"ok": False, "error": "所有文档都没能解析成功", "failures": failures}

    index_path, stats = _build_index(parsed)
    index_path.mkdir(parents=True, exist_ok=True)
    total = sum(item["chunks"] for item in stats)
    (index_path / "meta.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "chunks": total,
                "files": stats,
                "folder": str(store.workspace_folder()),
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with _LOCK:
        _SESSION_INDEX = index_path
        get_settings().kb_faiss_index = str(index_path)
    reset_cache()
    return {"ok": True, "chunks": total, "files": stats, "failures": failures}


def sync_workspace() -> dict:
    """工作区变化后调用：把检索切到该文件夹的索引（必要时重建）。"""
    global _SESSION_INDEX
    _SESSION_INDEX = None
    if _MODE != "session":
        return knowledge_summary()
    if not store.active_documents():
        clear_cache()
        return knowledge_summary()
    rebuild_index()
    return knowledge_summary()


def reset_cache() -> None:
    """丢掉已缓存的向量库，但保留当前索引目录。"""
    from .knowledge import reset_faiss_cache

    reset_faiss_cache()


def clear_cache() -> None:
    """回到"没有知识库"的状态：索引目录也切回项目默认。"""
    global _SESSION_INDEX
    from .knowledge import reset_faiss_cache

    with _LOCK:
        _SESSION_INDEX = None
        get_settings().kb_faiss_index = os.getenv(
            "KB_FAISS_INDEX", str(BASE_DIR / "public" / "faiss_kb_semantic")
        )
    reset_faiss_cache()


# ---------------------------------------------------------------- 导入 / 移除

def import_documents(items: list[tuple[str, bytes]]) -> dict:
    """把文件写进工作区文件夹并重建索引。"""
    if not items:
        return {"ok": False, "error": "没有收到文件"}
    saved = []
    for name, data in items:
        clean = Path(name).name or "未命名文件"
        store.add_document(clean, data)
        saved.append(clean)
    result = rebuild_index(force=True)
    result["saved"] = saved
    return result


def exclude_document(name: str) -> dict:
    """把某个文件移出知识库（**不删磁盘上的原文件**），然后重建索引。"""
    clean = Path(name).name
    if not store.document_path(clean).exists():
        return {"ok": False, "error": f"文件不存在：{clean}"}
    store.exclude_document(clean)
    result = rebuild_index(force=True)
    result["excluded"] = clean
    return result


def include_document(name: str) -> dict:
    clean = Path(name).name
    store.include_document(clean)
    result = rebuild_index(force=True)
    result["included"] = clean
    return result


def clear_knowledge_base() -> dict:
    """把当前文件夹里的文档全部移出知识库（原文件保留）。"""
    for item in store.list_documents():
        store.exclude_document(item["name"])
    clear_cache()
    return {"ok": True, "files": [], "chunks": 0}


def knowledge_summary() -> dict:
    """给界面用的知识库状态：文件清单（含是否已排除）+ 片段数 + 就绪判定。"""
    documents = store.list_documents()
    meta = {}
    meta_path = store.INDEX_DIR / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    stats = {item["name"]: item for item in meta.get("files", [])}
    files = [
        {
            "name": item["name"],
            "size": item["size"],
            "excluded": item["excluded"],
            "chars": stats.get(item["name"], {}).get("chars", 0),
            "chunks": stats.get(item["name"], {}).get("chunks", 0),
        }
        for item in documents
    ]
    active = [item for item in files if not item["excluded"]]
    return {
        "mode": _MODE,
        "ready": bool(active) and bool(meta.get("chunks")),
        "folder": str(store.workspace_folder()),
        "folder_name": store.workspace_folder().name,
        "is_default_folder": store.workspace_info()["is_default"],
        "files": files,
        "active_files": len(active),
        "chunks": meta.get("chunks", 0),
        "built_at": meta.get("built_at", ""),
    }


def active_tools() -> list:
    """当前会话实际绑定给模型的工具集合。

    没有知识库时不给 rag_search；配置了外部 MCP server 时，远端工具会一并绑上。
    """
    from .tools import all_tools

    tools = all_tools()
    if has_uploaded_kb():
        return tools
    return [tool for tool in tools if tool.name != "rag_search"]
