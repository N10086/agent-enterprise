"""图的共享状态。

`messages` 之外的字段都是"最后写入者覆盖"的普通通道；节点只返回自己要改的
字段，没返回的保持原值。整张图的路由几乎都由 `status` 这一个字符串驱动。
"""
from typing import Annotated, List, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    steps: int
    current_task: str
    task_type: str
    needs_tool: bool
    tool_outputs: List[str]
    research_results: List[str]
    # 本批检索片段的最高相似度（质检触发线的依据，也是观测指标）
    retrieval_best_score: float
    # 检索质量检查的结论：relevant / irrelevant（未做检查时为空串）
    document_grade: str
    final_answer: str
    status: str
    required_keywords: List[str]
    missing_keywords: List[str]
    verification: str
    review_notes: str
    failure_type: str
    revision_instruction: str
    failure_history: List[str]
    retry_count: int
    tool_rounds: int


def snapshot(state) -> dict:
    """状态的可读摘要，供节点日志与 runner 打印使用。

    只输出工具结果的**概要**（标签 + 前 120 字），不打印整段正文——
    否则一次日志会把 top-5 全文都刷出来。
    """
    outputs = state.get("tool_outputs", []) or []
    return {
        "status": state.get("status"),
        "steps": state.get("steps"),
        "current_task": state.get("current_task"),
        "task_type": state.get("task_type"),
        "needs_tool": state.get("needs_tool"),
        "tool_rounds": state.get("tool_rounds", 0),
        "tool_outputs": [f"{out[:120]}…" if len(out) > 120 else out for out in outputs],
        "evidence_items": len(state.get("research_results", []) or []),
        "retrieval_best_score": state.get("retrieval_best_score", 0.0),
        "document_grade": state.get("document_grade", ""),
        "required_keywords": state.get("required_keywords", []),
        "missing_keywords": state.get("missing_keywords", []),
        "verification": state.get("verification"),
        "final_answer": state.get("final_answer"),
        "review_notes": state.get("review_notes", ""),
        "retry_count": state.get("retry_count", 0),
        "failure_type": state.get("failure_type", ""),
        "revision_instruction": state.get("revision_instruction", ""),
        "failure_history": state.get("failure_history", []),
    }
