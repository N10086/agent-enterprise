from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError

from .graph import app
from .state import snapshot

# 图步数上限：兜底保护，避免任何未预料到的回环把进程拖死
RECURSION_LIMIT = 40

#: 节点名 → 界面上显示的中文进度说明
NODE_LABELS = {
    "supervisor": "解析问题",
    "researcher": "判断是否需要工具",
    "tool_executor": "执行工具",
    "tool_result_reader": "整理工具结果",
    "grade_documents": "检查检索质量",
    "analyst": "组织答案",
    "reviewer": "审核答案",
}

INITIAL_STATE = {
    "steps": 0,
    "current_task": "",
    "task_type": "direct",
    "needs_tool": False,
    "tool_outputs": [],
    "research_results": [],
    "retrieval_best_score": 0.0,
    "document_grade": "",
    "final_answer": "",
    "status": "idle",
    "required_keywords": [],
    "missing_keywords": [],
    "verification": "pending",
    "review_notes": "",
    "failure_type": "",
    "revision_instruction": "",
    "failure_history": [],
    "retry_count": 0,
    "tool_rounds": 0,
}


def run_agent(question: str) -> str:
    """把问题交给图执行，返回最终答案。"""
    return run_agent_state(question)[0]


def run_agent_state(question: str, callbacks: list | None = None) -> tuple[str, dict]:
    """返回 ``(最终答案, 最终状态)``。

    评测需要状态里的工具调用轨迹（用了哪些工具、检索到哪些片段），
    只返回答案字符串就没法判断答案的来源，也无法算检索指标。
    ``callbacks`` 用于挂 token 统计等观测钩子，会透传给图的 LLM 调用。
    """
    initial_state = {
        **INITIAL_STATE,
        "messages": [
            SystemMessage(content="你是一个 V4 版本 Agent，负责接收用户问题并给出答案。"),
            HumanMessage(content=question),
        ],
    }

    print("\n=== START AGENT RUN ===")
    print("Question:", question)

    config: dict = {"recursion_limit": RECURSION_LIMIT}
    if callbacks:
        config["callbacks"] = callbacks

    final_state = initial_state
    try:
        for chunk in app.stream(initial_state, stream_mode="values", config=config):
            print("\n=== STATE SNAPSHOT ===")
            print(snapshot(chunk))
            final_state = chunk
    except GraphRecursionError:
        print(f"\n[WARN] 达到图步数上限 {RECURSION_LIMIT}，返回当前已生成的最佳答案。")

    return final_state.get("final_answer", ""), final_state


def build_initial_state(question: str) -> dict:
    return {
        **INITIAL_STATE,
        "messages": [
            SystemMessage(content="你是一个 V4 版本 Agent，负责接收用户问题并给出答案。"),
            HumanMessage(content=question),
        ],
    }


def _extract_sources(outputs: list[str]) -> list[dict]:
    """从打了标签的工具输出里抽出检索来源，供界面展示"依据哪几段"。"""
    import re

    pattern = re.compile(r"出处:\s*([^§\n]*)§\s*([^\n（(]+)")
    sources: list[dict] = []
    for output in outputs:
        for source, title in pattern.findall(output or ""):
            item = {"source": source.strip(), "title": title.strip()}
            if item["title"] and item not in sources:
                sources.append(item)
    return sources


def stream_agent_state(question: str, callbacks: list | None = None):
    """边跑边产出事件，供界面显示进度；最后一个事件是 ``done``。

    事件类型：
        node    —— 走到哪个节点（界面上显示"正在检索/正在组织答案"）
        tool    —— 模型决定调用某个工具及其入参
        sources —— 检索到的片段出处
        grade   —— 检索质量检查结论（relevant / irrelevant）
        answer  —— 最终答案
        done    —— 用量与耗时
    """
    import time

    initial_state = build_initial_state(question)
    config: dict = {"recursion_limit": RECURSION_LIMIT}
    if callbacks:
        config["callbacks"] = callbacks

    started = time.time()
    final_state = initial_state
    emitted_tools = 0
    grade_seen = False
    warning = ""

    stream = app.stream(initial_state, stream_mode=["updates", "values"], config=config)
    try:
        for item in stream:
            if isinstance(item, tuple) and len(item) == 2:
                mode, chunk = item
            else:  # 兼容只返回单模式的版本
                mode, chunk = "values", item
            if mode == "values":
                final_state = chunk
                continue
            for node, update in (chunk or {}).items():
                if not isinstance(update, dict):
                    continue
                yield {"type": "node", "node": node, "label": NODE_LABELS.get(node, node)}
                # 模型发出的工具调用（researcher 节点里带 tool_calls 的那条消息）
                for message in update.get("messages") or []:
                    for call in getattr(message, "tool_calls", None) or []:
                        emitted_tools += 1
                        yield {
                            "type": "tool",
                            "name": call.get("name"),
                            "args": call.get("args") or {},
                        }
                if update.get("document_grade"):
                    grade_seen = True
                    yield {
                        "type": "grade",
                        "grade": update["document_grade"],
                        "note": update.get("review_notes", ""),
                    }
                if node == "tool_result_reader":
                    sources = _extract_sources(update.get("tool_outputs") or [])
                    if sources:
                        yield {"type": "sources", "items": sources[-5:]}
    except GraphRecursionError:
        warning = f"达到图步数上限 {RECURSION_LIMIT}，返回当前已生成的最佳答案。"

    answer = final_state.get("final_answer", "")
    if not answer:
        # 兜底：模型把答案写在了最后一条 AI 消息里，而 final_answer 没落上
        for message in reversed(final_state.get("messages") or []):
            if getattr(message, "type", "") == "ai" and getattr(message, "content", ""):
                answer = str(message.content)
                break

    yield {
        "type": "answer",
        "text": answer,
        "status": final_state.get("status"),
        "verification": final_state.get("verification"),
        "tools_used": emitted_tools,
        "graded": grade_seen,
        "retry_count": final_state.get("retry_count", 0),
        "review_notes": final_state.get("review_notes", ""),
    }
    yield {
        "type": "done",
        "seconds": round(time.time() - started, 2),
        "warning": warning,
        "state": snapshot(final_state),
    }
