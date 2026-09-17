"""本地知识库（RAG）接入层。

把 enterprise-research-agent 里那套 RAG 实现接到本项目的 Agent 上。这一层
只做三件事，边界刻意收得很窄：

1. **翻译配置**：把本项目的设置转成那套实现所需的环境变量，且必须在
   `import src.rag.*` 之前完成——那些库在导入时就把环境变量读成模块常量。
2. **保证入库只发生一次**：检索路径永远不写库。旧实现每次检索都
   `ingest(reset=True)`，等于每次问答都重切块、重向量化整份语料，
   单次要多花数秒并重写整个向量库。
3. **检索并格式化**：对外只暴露 `search_knowledge()`。

入库由 `build_kb.py` 显式完成；只有向量库为空且 `KB_AUTO_INGEST` 打开时，
才会在首次检索时兜底入库一次。
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
from pathlib import Path

from .config import get_settings

#: 可入库的语料后缀（与 src.rag.ingest 支持的范围保持一致）
_CORPUS_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".pdf"}

#: 入库是写操作，ToolNode 会并发调用工具，必须串行化
_INGEST_LOCK = threading.Lock()
_INGESTED = False


def _prepare_env() -> None:
    """在导入那套 RAG 实现之前把配置写进环境变量。"""
    settings = get_settings()
    os.environ.setdefault("CHROMA_DIR", settings.kb_vector_store)
    os.environ.setdefault("EMBEDDING_MODEL", settings.kb_embedding_model)

    root = str(Path(settings.enterprise_root))
    if root not in sys.path:
        sys.path.insert(0, root)


#: FAISS 后端的懒加载单例：嵌入模型加载较慢，只在首次检索时初始化
_FAISS_LOCK = threading.Lock()
_FAISS_STORE = None
_FAISS_EMBEDDER = None


def _faiss_index_dir() -> Path:
    return Path(get_settings().kb_faiss_index)


def faiss_available() -> bool:
    """语义索引是否已构建（build_rag_kb.py 的产物）。"""
    index_dir = _faiss_index_dir()
    return (index_dir / "index.faiss").exists() and (index_dir / "chunks.jsonl").exists()


def _faiss_chunk_count() -> int | None:
    """只读块数，不加载嵌入模型。"""
    path = _faiss_index_dir() / "chunks.jsonl"
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def load_faiss():
    """线程安全地载入 FAISS 索引与嵌入模型。"""
    global _FAISS_STORE, _FAISS_EMBEDDER
    if _FAISS_STORE is not None:
        return _FAISS_STORE, _FAISS_EMBEDDER
    with _FAISS_LOCK:
        if _FAISS_STORE is None:
            from rag_faiss import Embedder, FaissStore

            _FAISS_STORE = FaissStore.load(_faiss_index_dir())
            _FAISS_EMBEDDER = Embedder(get_settings().kb_embedding_model)
    return _FAISS_STORE, _FAISS_EMBEDDER


def reset_faiss_cache() -> None:
    """丢掉已缓存的索引与嵌入模型。

    用户导入/移除知识库后会切换索引目录，缓存不清就会继续用旧索引检索，
    表现成"明明导入了却查不到"。
    """
    global _FAISS_STORE
    with _FAISS_LOCK:
        _FAISS_STORE = None


def _search_faiss(query: str, top_k: int) -> str:
    """语义索引检索。

    输出格式刻意与旧后端保持一致（每条带「出处: 来源 § 标题」行），
    这样 Agent 工具的上下文、评测脚本的检索轨迹解析都不用改。
    """
    store, embedder = load_faiss()
    hits = store.search_one(query, top_k, embedder)
    if not hits:
        return "知识库中没有找到相关内容。"

    blocks = []
    for index, (score, chunk) in enumerate(hits, 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title") or meta.get("heading_path") or "未知来源"
        source = meta.get("source", "")
        body = (chunk.get("content") or "").strip()
        blocks.append(f"[文档{index}] {body}\n出处: {source} § {title}（相似度 {score:.3f}）")
    return "\n\n".join(blocks)


def _active_backend() -> str:
    """选择检索后端：KB_BACKEND 显式指定；auto 时有 FAISS 索引就优先用它。"""
    backend = (get_settings().kb_backend or "auto").strip().lower()
    if backend in {"faiss", "chroma"}:
        return backend
    return "faiss" if faiss_available() else "chroma"


def corpus_files() -> list[Path]:
    """列出语料目录里可入库的文件。"""
    corpus_dir = Path(get_settings().kb_corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        p for p in corpus_dir.rglob("*") if p.is_file() and p.suffix.lower() in _CORPUS_SUFFIXES
    )


def _vector_count() -> tuple[int | None, str]:
    """只读地查询向量库块数。返回 (块数, 错误说明)。"""
    _prepare_env()
    try:
        from src.rag.store import get_vectorstore

        ids = (get_vectorstore().get() or {}).get("ids") or []
    except Exception as exc:
        return None, f"向量库不可用：{type(exc).__name__}: {exc}"
    return len(ids), ""


def ingest_corpus(reset: bool = False, quiet: bool = False) -> tuple[bool, str]:
    """把语料目录入库。`reset=True` 先清空向量库。

    Returns:
        ``(是否成功, 说明)``
    """
    files = corpus_files()
    if not files:
        return False, (
            f"语料目录为空：{get_settings().kb_corpus_dir}\n"
            "请先运行：python build_kb.py"
        )

    _prepare_env()
    try:
        from src.rag.ingest import ingest as ingest_paths

        stream = io.StringIO() if quiet else None
        with contextlib.redirect_stdout(stream) if stream else contextlib.nullcontext():
            result = ingest_paths([str(p) for p in files], reset=reset)
    except Exception as exc:
        return False, f"入库失败：{type(exc).__name__}: {exc}"

    if result.get("errors"):
        return False, "部分语料入库失败：" + "; ".join(result["errors"])
    return True, f"已入库 {result.get('ingested', 0)} 块（来自 {len(files)} 个文件）"


def ensure_knowledge_base(force: bool = False) -> tuple[bool, str]:
    """确保向量库可用。已有内容时直接返回，不会重复入库。"""
    global _INGESTED

    if _INGESTED and not force:
        return True, "知识库已就绪"

    with _INGEST_LOCK:
        # 双重检查：并发进入时后来者直接复用前者的结果
        if _INGESTED and not force:
            return True, "知识库已就绪"

        count, error = _vector_count()
        if count is None:
            return False, error
        if count and not force:
            _INGESTED = True
            return True, f"知识库已就绪（{count} 块）"

        if not force and not get_settings().kb_auto_ingest:
            return False, (
                f"知识库为空（{count} 块），且已关闭自动入库（KB_AUTO_INGEST=0）。\n"
                "请先运行：python build_kb.py"
            )

        ok, note = ingest_corpus(reset=force, quiet=True)
        if ok:
            _INGESTED = True
        return ok, note


def search_knowledge(query: str, top_k: int = 3) -> str:
    """检索知识库并返回带出处的文本片段。

    后端由 ``KB_BACKEND`` 决定：``auto``（默认）在存在 FAISS 语义索引时优先用它，
    否则回退到旧的 Chroma 库。两种后端输出格式一致（都带「出处:」行），
    所以调用方——Agent 工具、MCP server、评测脚本——无需区分。
    """
    query = (query or "").strip()
    if not query:
        return "检索错误：查询词为空"

    if _active_backend() == "faiss":
        try:
            return _search_faiss(query, top_k)
        except Exception as exc:
            return f"[检索失败] FAISS: {type(exc).__name__}: {exc}"

    ready, note = ensure_knowledge_base()
    if not ready:
        return note

    try:
        from src.rag.retriever import search_text

        return search_text(query, k=top_k, enable_mqe=False, enable_hyde=False)
    except Exception as exc:
        return f"[检索失败] {type(exc).__name__}: {exc}"


def knowledge_status() -> str:
    """给人看的知识库状态摘要（不加载嵌入模型，避免只为打印状态就等几秒）。"""
    files = corpus_files()
    backend = _active_backend()
    if backend == "faiss":
        return (
            f"检索后端：FAISS（语义切分，400~800 字符）\n"
            f"索引目录：{_faiss_index_dir()}（{_faiss_chunk_count() or 0} 块）\n"
            f"嵌入模型：{get_settings().kb_embedding_model}"
        )
    count, error = _vector_count()
    if count is None:
        return error
    return (
        f"检索后端：Chroma（回退）\n"
        f"语料目录：{get_settings().kb_corpus_dir}（{len(files)} 个文件）\n"
        f"向量库：{get_settings().kb_vector_store}（{count} 块）"
    )
