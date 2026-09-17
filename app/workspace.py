"""应用数据存储：**工作区 = 本机上的一个文件夹**。

    <本机文件夹>/                   ← 工作区：知识库就来自这里的文档
    <数据目录>/                     ← 应用自己的数据，默认在系统数据目录里
        workspace.json              当前工作区路径 + 知识库排除名单
        conversations/<会话>.json    所有对话（都在默认工作区下面）
        kbsession/                  该文件夹的 FAISS 索引
        documents/                  默认工作区文件夹（用户没指定时用它）

数据目录由 `app/paths.py` 决定：Windows 在 `%APPDATA%\\agent-enterprise`，
macOS 在 `~/Library/Application Support/agent-enterprise`，Linux 在
`~/.local/share/agent-enterprise`。**刻意不放在项目目录里**——这样删掉项目
重新 clone 一份再跑，之前的对话和知识库都还在。旧版本放在 `public/appdata`
里的数据会在首次启动时自动搬过去。

设计取向：
  - **对话只有一个池子**，永远放在数据目录里。换文件夹不会把聊天记录弄丢，
    也不会往用户的文件夹里塞我们的文件。
  - **知识库 = 工作区文件夹里受支持的文档**（顶层扫描）。用户既能用界面上传
    （上传即写入该文件夹，是真实文件），也能自己在资源管理器里放进去。
  - 移除某个文件只是把它**排除出知识库**（记进排除名单），不会去删用户磁盘上的原文件——
    界面上的"移除"不该是一个删数据的动作。
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from .config import BASE_DIR
from .documents import SUPPORTED_SUFFIXES
from .paths import data_root

#: 一次最多索引多少个文件，避免误选一个巨大的目录把嵌入模型跑爆
MAX_FILES = 200

_LOCK = threading.RLock()


def appdata() -> Path:
    """应用数据目录（会顺带做一次旧数据搬迁）。"""
    return data_root(BASE_DIR)


def conversations_dir() -> Path:
    path = appdata() / "conversations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def index_dir() -> Path:
    path = appdata() / "kbsession"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_folder() -> Path:
    """默认工作区文件夹：也放在数据目录里，这样重新 clone 后还在。"""
    path = appdata() / "documents"
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_path() -> Path:
    return appdata() / "workspace.json"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_id(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_-]", "", str(value or ""))[:64]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # 原子替换，避免写一半断电留下半个 JSON


def _read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return dict(default or {})
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(default or {})


def ensure_layout() -> None:
    """确保数据目录与默认工作区文件夹都存在。"""
    appdata()
    for directory in (conversations_dir(), index_dir(), default_folder()):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 工作区文件夹

def settings() -> dict:
    ensure_layout()
    data = _read_json(settings_path())
    folder = str(data.get("folder") or "")
    if not folder or not Path(folder).is_dir():
        folder = str(default_folder())
        data["folder"] = folder
        _write_json(settings_path(), data)
    data.setdefault("excluded", [])
    return data


def workspace_folder() -> Path:
    return Path(settings()["folder"])


def set_workspace_folder(path: str) -> dict:
    """切换工作区文件夹。只接受真实存在的目录。"""
    target = Path(str(path or "")).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"文件夹不存在：{target}")
    if not target.is_dir():
        raise ValueError(f"不是文件夹：{target}")
    with _LOCK:
        data = settings()
        data["folder"] = str(target.resolve())
        # 换文件夹后旧的排除名单没有意义
        data["excluded"] = []
        _write_json(settings_path(), data)
    return workspace_info()


def workspace_info() -> dict:
    data = settings()
    folder = Path(data["folder"])
    return {
        "folder": str(folder),
        "name": folder.name or str(folder),
        "data_dir": str(appdata()),
        "is_default": str(folder) == str(default_folder()),
        "files": len(list_documents()),
        "excluded": list(data.get("excluded") or []),
    }


def list_directory(path: str | None = None) -> dict:
    """给界面的文件夹浏览器：列出某个目录下的子目录。"""
    target = Path(str(path or "")).expanduser() if path else Path.home()
    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(f"目录不存在：{target}")
    dirs, file_count = [], 0
    try:
        for entry in sorted(target.iterdir(), key=lambda item: item.name.lower()):
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    dirs.append({"name": entry.name, "path": str(entry)})
                elif entry.is_file():
                    file_count += 1
            except OSError:
                continue
    except PermissionError as exc:
        raise ValueError(f"没有权限读取该目录：{target}") from exc
    parent = target.parent if target.parent != target else None
    return {
        "path": str(target),
        "parent": str(parent) if parent else None,
        "dirs": dirs[:200],
        "files": file_count,
        "is_workspace": str(target) == str(workspace_folder()),
    }


# ---------------------------------------------------------------- 知识库文件

def list_documents() -> list[dict]:
    """扫描工作区文件夹里的受支持文档（顶层，跳过隐藏文件与排除名单）。"""
    folder = workspace_folder()
    if not folder.is_dir():
        return []
    excluded = set(settings().get("excluded") or [])
    items: list[dict] = []
    for entry in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
        try:
            if not entry.is_file() or entry.name.startswith("."):
                continue
        except OSError:
            continue
        if entry.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        stat = entry.stat()
        items.append(
            {
                "name": entry.name,
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
                "excluded": entry.name in excluded,
            }
        )
        if len(items) >= MAX_FILES:
            break
    return items


def active_documents() -> list[dict]:
    """参与索引的文档（排除名单里剔掉）。"""
    return [item for item in list_documents() if not item["excluded"]]


def document_path(name: str) -> Path:
    return workspace_folder() / Path(name).name


def add_document(name: str, data: bytes) -> Path:
    """上传即写入工作区文件夹——是真实文件，用户在资源管理器里也能看到。"""
    target = document_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    include_document(target.name)
    return target


def exclude_document(name: str) -> None:
    """移出知识库（不删磁盘上的文件）。"""
    with _LOCK:
        data = settings()
        excluded = set(data.get("excluded") or [])
        excluded.add(Path(name).name)
        data["excluded"] = sorted(excluded)
        _write_json(settings_path(), data)


def include_document(name: str) -> None:
    with _LOCK:
        data = settings()
        excluded = set(data.get("excluded") or [])
        excluded.discard(Path(name).name)
        data["excluded"] = sorted(excluded)
        _write_json(settings_path(), data)


def documents_fingerprint() -> str:
    """参与索引的文件集合指纹，用来判断要不要重建索引。"""
    return "|".join(f"{item['name']}:{item['size']}:{item['mtime']}" for item in active_documents())


# ---------------------------------------------------------------- 会话

def _conversation_path(conversation_id: str) -> Path:
    return conversations_dir() / f"{_safe_id(conversation_id)}.json"


def create_conversation(title: str = "新对话") -> dict:
    conversation_id = f"c-{uuid.uuid4().hex[:10]}"
    payload = {
        "id": conversation_id,
        "title": title,
        "created_at": _now(),
        "updated_at": _now(),
        "folder": str(workspace_folder()),
        "messages": [],
    }
    with _LOCK:
        _write_json(_conversation_path(conversation_id), payload)
    return payload


def list_conversations() -> list[dict]:
    items = []
    for path in conversations_dir().glob("*.json"):
        data = _read_json(path)
        if not data:
            continue
        items.append(
            {
                "id": data.get("id", path.stem),
                "title": data.get("title", "未命名"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "messages": len(data.get("messages") or []),
            }
        )
    items.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return items


def get_conversation(conversation_id: str) -> dict:
    data = _read_json(_conversation_path(conversation_id))
    if not data:
        raise FileNotFoundError(f"会话不存在：{conversation_id}")
    return data


def append_message(conversation_id: str, message: dict) -> dict:
    """追加一条消息；第一条用户消息顺便用来当会话标题。"""
    path = _conversation_path(conversation_id)
    with _LOCK:
        data = _read_json(path)
        if not data:
            raise FileNotFoundError(f"会话不存在：{conversation_id}")
        data.setdefault("messages", []).append(message)
        data["updated_at"] = _now()
        if message.get("role") == "user" and data.get("title") in (None, "", "新对话"):
            text = str(message.get("text") or "").strip().replace("\n", " ")
            data["title"] = (text[:28] + "…") if len(text) > 28 else text
        _write_json(path, data)
    return data


def delete_conversation(conversation_id: str) -> bool:
    path = _conversation_path(conversation_id)
    if path.exists():
        path.unlink()
        return True
    return False


def rename_conversation(conversation_id: str, title: str) -> dict:
    path = _conversation_path(conversation_id)
    with _LOCK:
        data = _read_json(path)
        if not data:
            raise FileNotFoundError(f"会话不存在：{conversation_id}")
        data["title"] = (title or "").strip()[:40] or data.get("title", "未命名")
        _write_json(path, data)
    return data


def clear_appdata() -> None:
    """清空应用数据（测试/重置用）。"""
    shutil.rmtree(conversations_dir(), ignore_errors=True)
    shutil.rmtree(index_dir(), ignore_errors=True)
    settings_path().unlink(missing_ok=True)


__all__ = [
    "active_documents",
    "add_document",
    "appdata",
    "append_message",
    "clear_appdata",
    "conversations_dir",
    "create_conversation",
    "default_folder",
    "delete_conversation",
    "document_path",
    "documents_fingerprint",
    "ensure_layout",
    "exclude_document",
    "get_conversation",
    "include_document",
    "index_dir",
    "list_conversations",
    "list_directory",
    "list_documents",
    "rename_conversation",
    "set_workspace_folder",
    "settings",
    "settings_path",
    "workspace_folder",
    "workspace_info",
]
