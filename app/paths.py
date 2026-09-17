"""数据目录：把「用户数据」和「代码」彻底分开。

代码可以随时重新 clone、升级、甚至删掉重来；数据必须留在原地。
所以对话历史、知识库索引、默认工作区文件夹都放在操作系统给应用的数据目录里：

    Windows   %APPDATA%\\agent-enterprise
    macOS     ~/Library/Application Support/agent-enterprise
    Linux     ~/.local/share/agent-enterprise      （遵循 XDG_DATA_HOME）

这样"删掉项目目录重新 clone 一份再跑"，之前的对话与知识库照样在。

想放到别处（比如移动硬盘、云盘同步目录）就设环境变量：

    AGENT_DATA_DIR=D:\\my-agent-data
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

#: 应用名，同时用作数据目录名
APP_NAME = "agent-enterprise"

_CACHE: Path | None = None

#: 迁移标记：写过这个文件就不再尝试从旧位置搬运数据
_MIGRATION_MARKER = ".migrated-from-repo"


def default_data_root() -> Path:
    """按操作系统惯例给出数据目录。"""
    override = os.getenv("AGENT_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    home = Path.home()
    if sys.platform == "win32":
        base = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
        root = Path(base) if base else home / "AppData" / "Roaming"
        return root / APP_NAME
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    xdg = os.getenv("XDG_DATA_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else home / ".local" / "share"
    return base / APP_NAME


def legacy_data_root(project_dir: Path) -> Path:
    """旧版本把数据放在项目里的 public/appdata，用于一次性搬迁。"""
    return project_dir / "public" / "appdata"


def data_root(project_dir: Path | None = None) -> Path:
    """数据目录（带缓存）。首次调用时顺带做一次旧数据搬迁。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    root = default_data_root()
    root.mkdir(parents=True, exist_ok=True)
    if project_dir is not None:
        migrate_legacy(project_dir, root)
    _CACHE = root
    return root


def migrate_legacy(project_dir: Path, root: Path) -> bool:
    """把旧版放在项目目录里的数据搬到新的数据目录。

    只在"新目录还没有数据、旧目录确实有数据"时搬一次，搬完写标记文件，
    避免反复执行；搬运用复制而不是移动，旧目录保持原样作为兜底。
    """
    legacy = legacy_data_root(project_dir)
    marker = root / _MIGRATION_MARKER
    if marker.exists() or not legacy.is_dir() or legacy.resolve() == root.resolve():
        return False
    # 新目录里已经有对话，就不覆盖用户现在的数据
    if (root / "conversations").is_dir() and any((root / "conversations").glob("*.json")):
        marker.write_text("skip: 新数据目录已有对话\n", encoding="utf-8")
        return False

    copied = 0
    for item in legacy.iterdir():
        target = root / item.name
        if item.is_dir():
            if target.exists():
                continue  # 已有同名目录就别动
            shutil.copytree(item, target)
            copied += 1
        elif not target.exists():
            shutil.copy2(item, target)
            copied += 1
    if copied:
        print(f"[数据迁移] 已把旧数据从 {legacy} 复制到 {root}（旧目录保留不动）")
    marker.write_text(f"copied {copied} items\n", encoding="utf-8")
    return bool(copied)


def data_root_label(project_dir: Path | None = None) -> str:
    """给界面/日志看的说明：数据在哪里、怎么改位置。"""
    root = data_root(project_dir)
    source = "AGENT_DATA_DIR" if os.getenv("AGENT_DATA_DIR", "").strip() else "系统默认位置"
    return f"{root}（{source}）"


def reset_cache() -> None:
    """清掉缓存（测试用）。"""
    global _CACHE
    _CACHE = None


__all__ = [
    "APP_NAME",
    "data_root",
    "data_root_label",
    "default_data_root",
    "legacy_data_root",
    "migrate_legacy",
    "reset_cache",
]
