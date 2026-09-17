"""RAG 检索链路：multi-query → HyDE → FAISS 多路检索 → RRF 融合 top-k。

这个模块**不关心**"谁来发起检索"：Agent 图把它包成 `rag_search` 工具（见
`app/tools.py`），消融实验（`rag_pipeline.py`）直接调用 `retrieve()`。
两种用法共享同一套检索实现，不会出现"图里检索一套、评测里检索另一套"。

    用户问题 / 工具入参
      → ① multi-query：改写成若干条更适合向量检索的语句
      → ② HyDE：为原问题与每条改写语句各生成一段"假设性答案"
      → ③ 用「原问题 + 改写语句 + 假设答案」多路检索 FAISS
      → ④ RRF 融合，取 top-k（默认 5）

模型只负责决定**要不要检索、检索什么**；检索质量由这里保证。
"""
from __future__ import annotations

import sys
from pathlib import Path

#: rag_faiss / rag_pipeline 是项目根下的顶层模块，保证任意工作目录都能导入
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_faiss import rrf_merge  # noqa: E402
from rag_pipeline import DEFAULT_MQ_COUNT, hyde, multi_query  # noqa: E402

from .config import get_settings  # noqa: E402
from .knowledge import load_faiss  # noqa: E402

#: 每条查询各自取回的候选数，融合前先各自粗排一遍
RETRIEVALS_PER_QUERY = 5

def get_llm():
    """改写 / HyDE 用的模型。

    跟随"当前请求选中的模型"（见 app/llm.py）：界面切了模型，检索侧也一起切，
    不会出现"图用 Qwen、检索还在用 DeepSeek"这种错配。
    """
    from .llm import get_active_llm

    return get_active_llm()


def chunk_title(chunk: dict) -> str:
    meta = chunk.get("metadata") or {}
    return meta.get("title") or meta.get("heading_path") or "未知来源"


def chunk_titles(chunk: dict) -> list[str]:
    """块覆盖的全部小节标题。

    一个块可能由多个小节合并而成（`metadata["titles"]`），只报主标题会让
    "命中的其实是合并块的次要小节"这种情况在评测里变成未召回，所以出处行里要全列出来。
    """
    meta = chunk.get("metadata") or {}
    titles = [str(title) for title in (meta.get("titles") or []) if title]
    primary = meta.get("title") or meta.get("heading_path")
    if primary and primary not in titles:
        titles.insert(0, str(primary))
    return titles or ["未知来源"]


def chunk_source(chunk: dict) -> str:
    return (chunk.get("metadata") or {}).get("source", "") or ""


def chunk_score(chunk: dict) -> float | None:
    """块与查询的最高原始余弦相似度（rrf_merge 回写的 `score`，不是融合分）。"""
    score = chunk.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def retrieve(
    question: str,
    llm=None,
    top_k: int | None = None,
    mq_count: int = DEFAULT_MQ_COUNT,
) -> dict:
    """跑一轮完整检索，返回 ``{"chunks", "queries", "hypotheticals", "search_queries"}``。"""
    settings = get_settings()
    llm = llm or get_llm()
    top_k = top_k or settings.kb_top_k
    store, embedder = load_faiss()

    queries, _ = multi_query(llm, question, mq_count) if mq_count else ([], {})
    inputs = [question] + list(queries)
    hypotheticals, _ = hyde(llm, inputs)
    search_queries = inputs + hypotheticals

    merged = rrf_merge(store.search_many(search_queries, RETRIEVALS_PER_QUERY, embedder), top_k)
    return {
        "chunks": [item for _, item in merged],
        "queries": queries,
        "hypotheticals": hypotheticals,
        "search_queries": search_queries,
        "top_k": top_k,
    }


def format_context(chunks: list[dict]) -> str:
    """渲染成带出处的文本。

    格式与旧知识库后端保持一致（每条带「出处: 来源 § 标题」行）：评测脚本靠这个
    格式解析检索轨迹，换成别的写法会让 Recall@k 直接算不出来。
    合并块覆盖多个小节时用「｜」并列，既不丢信息，也不需要额外的结构化字段。
    """
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        body = (chunk.get("content") or "").strip()
        score = chunk_score(chunk)
        score_text = f"（相似度 {score:.3f}）" if score is not None else ""
        titles = "｜".join(chunk_titles(chunk))
        blocks.append(f"[文档{index}] {body}\n出处: {chunk_source(chunk)} § {titles}{score_text}")
    return "\n\n".join(blocks)


def retrieve_context(query: str, top_k: int | None = None) -> str:
    """`rag_search` 工具的实现：跑完整链路，返回可直接进上下文的带出处文本。

    工具只传进来一个 query，改写与 HyDE 在这里内部完成——这就是"把 RAG 接进
    Agent 图"的关键：Agent 决定检索时机，链路决定检索质量。
    """
    query = (query or "").strip()
    if not query:
        return "检索错误：查询词为空"
    try:
        result = retrieve(query, top_k=top_k)
    except Exception as exc:
        return f"[检索失败] {type(exc).__name__}: {exc}"
    chunks = result["chunks"]
    if not chunks:
        return "知识库中没有找到相关内容。"
    return format_context(chunks)


__all__ = [
    "DEFAULT_MQ_COUNT",
    "RETRIEVALS_PER_QUERY",
    "chunk_score",
    "chunk_source",
    "chunk_title",
    "chunk_titles",
    "format_context",
    "get_llm",
    "retrieve",
    "retrieve_context",
]
