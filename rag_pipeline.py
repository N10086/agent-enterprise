"""新的检索问答管线：multi-query → HyDE → 向量检索 top-5 → 交给 LLM 作答。

按需求实现的流程：

    用户问题
      → ① multi-query：把模糊表述补全、把疑问句改写成陈述式的检索语句
      → ② HyDE：为这些改写后的问题各生成一段"假设性答案"
      → ③ 用这些假设答案去向量库检索，融合后取最相似的 5 个 chunk
      → ④ 把 5 个 chunk + 原始问题一起交给 LLM 作答

**为什么先改写再检索**：用户的问法和文档的写法天然对不上
（"谁更年长" vs "出生于 1965 年"），字面几乎不重合。
先转成陈述句、再转成"答案的样子"，能显著拉近与文档向量的距离。

每一个 LLM 调用都会记录 token 用量与耗时，用于事后衡量效率
（上下文长度、token 数、各阶段成本），而不只是看最终准确率。
"""
from __future__ import annotations

import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings

#: multi-query 改写出的问题个数（不含原问题）
DEFAULT_MQ_COUNT = 3
#: 最终送入作答的 chunk 数
DEFAULT_TOP_K = 5

_MQ_SYSTEM = (
    "你是检索查询改写助手。把用户的问题改写成更适合向量检索的查询。\n"
    "要求：\n"
    "1) 补全模糊指代（把「他/那位女性/这个地方」换成具体的实体或类别）；\n"
    "2) 疑问句改写成陈述句式或名词短语式的检索语句，例如"
    "「谁更年长？」→「A 与 B 的出生日期 年龄对比」；\n"
    "3) 每行一条，不要编号、不要引号、不要任何解释。\n"
    "**输出语言必须与用户问题完全一致：问题为英文时输出必须是英文，禁止翻译成中文。**"
)

_HYDE_SYSTEM = (
    "下面会给出若干个检索问题。请为每一个问题写一段 100 字以内的**假设性答案**"
    "（陈述句，尽量包含该领域的术语与具体表述）。\n"
    "这段文字用于向量检索，不要求事实正确，但风格要像真实资料。\n"
    "输出格式：每行一条，用「问题序号. 答案」开头，例如「1. xxx」。\n"
    "**关键：输出语言必须与对应问题完全一致。检索语料是英文维基百科段落，"
    "因此英文问题的假设答案必须是英文，禁止翻译成中文——语言不一致会让检索完全失效。**"
)

_ANSWER_SYSTEM = (
    "你是严谨的问答助手。只依据给定的资料回答用户问题。\n"
    "要求：\n"
    "1) 先给出直接答案（一句话），再给简要依据；\n"
    "2) 依据要标注资料编号，例如「依据资料 2」；\n"
    "3) 如果资料不足以回答，明确说明「资料不足」，不要凭记忆编造；\n"
    "4) 不要输出与问题无关的内容；\n"
    "5) **最后必须单独一行输出自评**：资料足够时输出 `判定: 资料充分`，"
    "不足以回答时输出 `判定: 资料不足`。这一行用于决定是否需要重新检索。"
)

#: 对外公开别名：图（app/graph.py）的作答节点复用同一段系统提示，避免两处提示词漂移
ANSWER_SYSTEM = _ANSWER_SYSTEM

_REFORMULATE_SYSTEM = (
    "上一轮检索没有找到足够资料来回答问题。请换一个角度重新拟一条检索查询。\n"
    "要求：\n"
    "1) 换用不同的关键词、同义表达或更具体的实体名，不要重复上一轮的查询；\n"
    "2) 只输出这一条查询，不要编号、不要解释；\n"
    "3) 语言必须与用户问题一致（英文问题必须输出英文）。"
)

#: 判定行的解析（模型未按格式输出时回退到拒答词表）
_VERDICT_RE = re.compile(r"判定\s*[:：]\s*(资料充分|资料不足|sufficient|insufficient)", re.I)
_INSUFFICIENT_RE = re.compile(
    r"资料不足|没有找到|未找到|无法确定|无法回答|不足以回答|insufficient|no relevant",
    re.IGNORECASE,
)


def make_llm(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model or settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url or None,
        temperature=temperature,
    )


def call_llm(llm: ChatOpenAI, messages: list) -> tuple[str, dict]:
    """调用一次 LLM，返回 (文本, 用量信息)。用量信息用于事后统计效率。"""
    start = time.time()
    response = llm.invoke(messages)
    usage = getattr(response, "usage_metadata", None) or {}
    return str(response.content), {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "seconds": round(time.time() - start, 3),
    }


def _lines(text: str) -> list[str]:
    out = []
    for line in (text or "").splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.、)])\s*", "", line).strip().strip('"“”')
        if cleaned:
            out.append(cleaned)
    return out


def multi_query(llm: ChatOpenAI, question: str, count: int = DEFAULT_MQ_COUNT) -> tuple[list[str], dict]:
    """把用户问题改写成若干更清晰的检索语句（不含原问题）。"""
    prompt = f"请把下面的问题改写成 {count} 条检索语句（保持与问题相同的语言）：\n{question}"
    text, usage = call_llm(llm, [SystemMessage(content=_MQ_SYSTEM), HumanMessage(content=prompt)])
    queries = [q for q in _lines(text)][:count]
    return queries, usage


def hyde(llm: ChatOpenAI, questions: list[str], max_chars: int = 200) -> tuple[list[str], dict]:
    """为每个问题生成一段假设性答案（一次调用生成全部，按序号对齐）。"""
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
    tail = "\n（再次强调：每条假设答案的语言必须与对应问题一致，英文问题必须用英文作答。）"
    text, usage = call_llm(
        llm, [SystemMessage(content=_HYDE_SYSTEM), HumanMessage(content=numbered + tail)]
    )
    answers = _lines(text)
    # 只保留「序号. 内容」里序号匹配的行，避免模型多输出
    cleaned: list[str] = []
    for line in answers:
        match = re.match(r"^(\d+)[.、)]?\s*(.+)$", line)
        if match:
            cleaned.append(match.group(2).strip()[:max_chars])
    return (cleaned or answers)[: len(questions)], usage


def format_chunks(chunks: list[dict]) -> str:
    blocks = []
    for index, item in enumerate(chunks, start=1):
        meta = item.get("metadata", {})
        title = meta.get("title") or meta.get("heading_path") or "未知来源"
        blocks.append(f"【资料 {index}】{title}\n{item.get('content', '').strip()}")
    return "\n\n".join(blocks)


def answer_with_chunks(llm: ChatOpenAI, question: str, chunks: list[dict]) -> tuple[str, dict]:
    """把检索到的 chunk 与原始问题一起交给 LLM 作答。"""
    context = format_chunks(chunks) if chunks else "（没有检索到相关资料）"
    prompt = f"【用户问题】{question}\n\n【检索到的资料】\n{context}\n\n请依据以上资料作答。"
    return call_llm(llm, [SystemMessage(content=_ANSWER_SYSTEM), HumanMessage(content=prompt)])


def split_verdict(answer: str) -> tuple[str, str]:
    """拆出模型的自评行。

    返回 ``(去掉自评行的答案, "sufficient" | "insufficient")``。
    自评行不能带进评测——否则裁判会看到内部的"资料不足"标记而先入为主。
    """
    text = answer or ""
    verdict = "sufficient"
    match = _VERDICT_RE.search(text)
    if match:
        value = match.group(1)
        verdict = (
            "insufficient" if ("不足" in value or value.lower() == "insufficient") else "sufficient"
        )
        text = (text[: match.start()] + text[match.end() :]).strip()
    elif _INSUFFICIENT_RE.search(text):
        verdict = "insufficient"  # 没按格式输出时用拒答词表兜底
    return text.strip(), verdict


def reformulate(
    llm: ChatOpenAI, question: str, previous_answer: str, previous_queries: list[str]
) -> tuple[str, dict]:
    """换角度重拟一条检索查询（纠错回环用）。"""
    prompt = (
        f"【原问题】{question}\n"
        f"【上一轮的回答（资料不足）】{(previous_answer or '')[:400]}\n"
        f"【上一轮已用过的查询】{'；'.join(previous_queries)}\n\n请给出新的检索查询："
    )
    text, usage = call_llm(llm, [SystemMessage(content=_REFORMULATE_SYSTEM), HumanMessage(content=prompt)])
    lines = _lines(text)
    return (lines[0] if lines else question), usage


def _chunk_key(chunk: dict) -> str:
    return chunk.get("metadata", {}).get("chunk_id") or chunk.get("content", "")[:80]


def merge_chunks(new_chunks: list[dict], old_chunks: list[dict], limit: int) -> list[dict]:
    """合并两轮证据：新一轮优先，按 chunk_id 去重，总量受 limit 约束。"""
    merged: list[dict] = []
    seen: set[str] = set()
    for chunk in list(new_chunks) + list(old_chunks):
        key = _chunk_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
        if len(merged) >= limit:
            break
    return merged


def run_agent_loop(
    llm: ChatOpenAI,
    store,
    embedder,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    max_rounds: int = 2,
    retrievals_per_query: int = 5,
) -> dict:
    """带纠错回环的检索问答（agent loop）。

    单轮流程与 ``hyde_only`` 相同；区别在于**作答后会自评资料是否充分**：
    不充分且预算还有剩余时，就换一条查询重新检索一轮，把新证据与旧证据
    合并后再作答。这样既保留了单轮方案的低成本，又拿回了旧 Agent 的纠错能力。

    ``max_rounds=2`` 即"最多给一次纠错机会"——每多一轮就多一次 HyDE 调用、
    一次作答调用和一份新上下文，成本随之近似线性增长，所以必须有硬预算。
    """
    from rag_faiss import rrf_merge

    started = time.time()
    rounds: list[dict] = []
    query = question
    used_queries: list[str] = []
    evidence: list[dict] = []
    clean_answer = ""
    verdict = "insufficient"

    for round_no in range(1, max_rounds + 1):
        round_start = time.time()
        hypotheticals, hyde_usage = hyde(llm, [query])
        hypotheticals = hypotheticals[:1] or [query]
        queries = [question, query] if round_no > 1 else [query]
        lists = store.search_many(queries + hypotheticals, retrievals_per_query, embedder)
        merged = rrf_merge(lists, top_k)
        fresh = [chunk for _, chunk in merged]
        evidence = merge_chunks(fresh, evidence, top_k)
        used_queries.append(query)

        raw_answer, answer_usage = answer_with_chunks(llm, question, evidence)
        clean_answer, verdict = split_verdict(raw_answer)

        rounds.append(
            {
                "round": round_no,
                "query": query,
                "hypotheticals": hypotheticals,
                "queries": queries + hypotheticals,
                "chunks": [
                    {
                        "title": c.get("metadata", {}).get("title", ""),
                        "titles": c.get("metadata", {}).get("titles", []),
                        "chunk_id": c.get("metadata", {}).get("chunk_id", ""),
                        "chars": len(c.get("content", "")),
                    }
                    for c in evidence
                ],
                "answer": clean_answer,
                "verdict": verdict,
                "tokens": hyde_usage["total_tokens"] + answer_usage["total_tokens"],
                "seconds": round(time.time() - round_start, 3),
            }
        )

        if verdict == "sufficient" or round_no == max_rounds:
            break
        new_query, reformulate_usage = reformulate(llm, question, clean_answer, used_queries)
        rounds[-1]["tokens"] += reformulate_usage["total_tokens"]
        rounds[-1]["next_query"] = new_query
        query = new_query

    # 供评测使用：第一轮与最终轮的证据集合
    retrieval = {
        "round1": {"chunks": rounds[0]["chunks"]},
        "final": {"chunks": rounds[-1]["chunks"]},
    }
    total_tokens = sum(item["tokens"] for item in rounds)
    return {
        "answer": clean_answer,
        "mode": "agent_loop",
        "rounds": rounds,
        "rounds_used": len(rounds),
        "corrected": len(rounds) > 1,
        "stages": {
            "rounds": rounds,
            "total_tokens": total_tokens,
            "total_seconds": round(time.time() - started, 3),
            "retrieval": {
                "mode": "agent_loop",
                "queries": used_queries,
                "context_chars": sum(
                    c.get("chars", 0) for c in rounds[-1]["chunks"]
                ),
            },
        },
        "retrieval": retrieval,
        "production_chunks": evidence,
    }


def run_pipeline(
    llm: ChatOpenAI,
    store,
    embedder,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    mq_count: int = DEFAULT_MQ_COUNT,
    retrievals_per_query: int = 5,
    mode: str = "hyde_only",
    max_rounds: int = 2,
) -> dict:
    """跑一条完整流程，返回答案、各阶段用量与各检索配置的对比数据。

    ``mode``:
        ``hyde_only``（默认）—— 问题 → HyDE 假设答案 → 检索 → 作答。
        ``agent_loop`` —— 在 hyde_only 之上加纠错回环：作答自评资料不足时
            换查询重检一轮再答（见 ``run_agent_loop``）。
        ``mq_hyde`` —— 问题 → multi-query 改写 → 对每个问题做 HyDE → 检索 → 作答。
    """
    from rag_faiss import rrf_merge

    if mode == "agent_loop":
        return run_agent_loop(
            llm,
            store,
            embedder,
            question,
            top_k=top_k,
            max_rounds=max_rounds,
            retrievals_per_query=retrievals_per_query,
        )

    stages: dict = {}
    start = time.time()

    queries: list[str] = []
    if mode == "mq_hyde":
        queries, mq_usage = multi_query(llm, question, mq_count)
    else:
        mq_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "seconds": 0.0}
    stages["multi_query"] = mq_usage

    # rag_only：完全不生成假设答案，直接用原问题检索（HyDE 的对照组）
    if mode == "rag_only":
        hypotheticals, hyde_usage = [], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "seconds": 0.0}
    else:
        hyde_inputs = [question] + queries
        hypotheticals, hyde_usage = hyde(llm, hyde_inputs)
    stages["hyde"] = hyde_usage

    if mode == "rag_only":
        configs = {"base": [question]}
    else:
        configs = {
            "base": [question],
            "hyde": hypotheticals[:1] or [question],
        }
        if mode == "mq_hyde":
            configs["mq"] = [question] + queries
            configs["mq_hyde"] = [question] + queries + hypotheticals
    production_config = mode if mode in configs else ("hyde" if "hyde" in configs else "base")

    ranked: dict = {}
    for name, query_list in configs.items():
        ranked[name] = store.search_many(query_list, retrievals_per_query, embedder)

    retrieval: dict = {}
    for name, lists in ranked.items():
        merged = rrf_merge(lists, top_k)
        retrieval[name] = {
            "chunks": [
                {
                    "title": item.get("metadata", {}).get("title", ""),
                    "titles": item.get("metadata", {}).get("titles", []),
                    "chunk_id": item.get("metadata", {}).get("chunk_id", ""),
                    "chars": len(item.get("content", "")),
                    "score": item.get("score", 0.0),
                    "rrf_score": item.get("rrf_score", 0.0),
                }
                for _, item in merged
            ],
            "top_chunks": [item for _, item in merged],
        }

    production = retrieval[production_config]["top_chunks"]
    answer, answer_usage = answer_with_chunks(llm, question, production)
    stages["answer"] = answer_usage

    context_chars = sum(len(c.get("content", "")) for c in production)
    stages["retrieval"] = {
        "mode": mode,
        "queries": configs[production_config],
        "mq_queries": queries,
        "hypotheticals": hypotheticals,
        "context_chars": context_chars,
        "seconds": 0.0,
    }
    stages["total_seconds"] = round(time.time() - start, 3)
    stages["total_tokens"] = sum(
        stages[key].get("total_tokens", 0)
        for key in ("multi_query", "hyde", "answer")
        if isinstance(stages.get(key), dict)
    )

    return {
        "answer": answer,
        "mode": mode,
        "stages": stages,
        "retrieval": {name: {"chunks": data["chunks"]} for name, data in retrieval.items()},
        "production_chunks": production,
    }
