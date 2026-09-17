"""LangGraph 编排：原始 Agent 图 + RAG 接入（Agentic RAG）。

    supervisor → researcher ⇄ tool_executor → tool_result_reader → grade_documents
                     │                                                │
                     │                          ┌── 相关 ──→ analyst ──┘
                     │                          └── 不相关 → supervisor（改写后重检）
                     └────────────────────────────────────────────────┘
    analyst → reviewer → END / supervisor

节点职责：

- ``supervisor``：读用户问题、算出关键字要求、管理重试预算与修正指令。
  **不做任何"要不要用工具"的判断**。
- ``researcher``：把问题和工具（含 rag_search）一起交给 LLM，由模型用原生
  ``tool_calls`` 决定调哪个工具、或者不调直接作答。工具描述里写清了各自的适用时机。
- ``tool_executor``：执行工具（ToolNode）。
- ``tool_result_reader``：给工具结果打标签（``[Tool: rag_search] 检索到的文档片段：…``），
  避免裸值混进消息，让模型分不清来源与含义。
- ``grade_documents``：**检索质量检查**（对应开源实现里的 adaptive-RAG / CRAG）。
  检索片段与问题无关时不硬着头皮作答，而是回到 supervisor 换查询重检一轮；
  重试预算用尽则照常作答，由 analyst 说明资料不足。
- ``analyst``：汇总工具结果生成最终答案。
- ``reviewer``：格式与关键字检查，未通过则带着修正指令回到 supervisor。

这套结构就是 LangGraph 官方 Agentic RAG 教程的形状
（generate_query_or_respond → retrieve → grade_documents → generate/rewrite），
只是保留了本项目原有的 supervisor / tool_result_reader / reviewer 三处增强。
"""
import json
import os
import re
import sys
from functools import wraps
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from .config import get_settings
from .llm import ActiveLLM
from .session import active_tools, knowledge_policy
from .state import AgentState, snapshot
from .tools import all_tools, find_tool

#: Windows 控制台默认用 GBK，日志里一旦出现语料中的特殊字符（如 "Dâ"）
#: print 会抛 UnicodeEncodeError 直接中断整轮评测，所以这里把输出流放宽到
#: UTF-8 且不可编码字符用替代符，日志永远不该成为失败原因。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MAX_RETRIES = 2
MAX_TOOL_ROUNDS = 4
DEBUG_GRAPH_OUTPUT = os.getenv("AGENT_DEBUG_GRAPH", "0").lower() in {"1", "true", "yes", "on"}

#: rag_search 的输出里每条片段都带「（相似度 0.712）」，用它取整批片段的最高分
_SCORE_RE = re.compile(r"相似度\s*([0-9]*\.?[0-9]+)")

#: 决策侧绑定的工具由会话状态决定：未上传知识库时不给 rag_search（见 app/session.py）
#: 给工具结果加标签，避免裸值（如 "5"）混进消息让模型分不清来源
TOOL_LABELS = {
    "calculator": "计算结果",
    "get_current_time": "当前时间",
    "web_search": "搜索结果",
    "rag_search": "检索到的文档片段",
}


def tool_label(name: str) -> str:
    """工具结果的中文标签；MCP 这类外部工具不在内置表里，统一叫"工具结果"。"""
    return TOOL_LABELS.get(name, "工具结果")

# 由问题类型规则推导出的“证据型”要求：不要求字面出现，而是要求答案里存在对应证据，
# 否则像“计算结果”“当前时间”这种词几乎永远不会被原样写进答案，会导致无限空转重试。
EVIDENCE_REQUIREMENTS = {
    "计算结果": {
        "hint": "答案中必须给出明确的计算结果（含数字，形如 2 + 3 * 4 = 14）。",
        "patterns": [r"\d\s*=\s*-?\d", r"结果", r"等于"],
    },
    "当前时间": {
        "hint": "答案中必须给出具体的当前时刻（含日期或形如 14:23 的时间）。",
        "patterns": [r"\d{1,2}\s*[:：]\s*\d{2}", r"点", r"时间"],
    },
}

# 关键词列表里一旦出现在这些否定表述，说明该子句是限制条件而不是待包含的关键字。
NEGATION_MARKERS = ("不要出现", "不能出现", "不得出现", "别出现", "禁止出现", "不要包含", "不含", "避免")

# 审核失败类型：用于把“失败原因”转成下一轮可执行的修正指令，而不是只回传状态字符串。
FAILURE_EMPTY = "empty_answer"
FAILURE_TOO_SHORT = "too_short"
FAILURE_MISSING_KEYWORD = "missing_keyword"
FAILURE_IRRELEVANT_DOCS = "irrelevant_docs"

FAILURE_LABELS = {
    FAILURE_EMPTY: "答案为空",
    FAILURE_TOO_SHORT: "答案过短",
    FAILURE_MISSING_KEYWORD: "关键字未满足",
    FAILURE_IRRELEVANT_DOCS: "检索结果与问题无关",
}

settings = get_settings()
#: 不是普通 ChatOpenAI：每次调用都转发给"当前请求选中的模型"（见 app/llm.py），
#: 这样界面切模型不必改任何节点代码。
llm = ActiveLLM()


def _log(tag: str, value):
    print(f"\n[{tag}] {value}")


def _traced(node):
    """统一打印节点边界与出入状态，节点函数体内就不必重复写这些日志。"""
    name = node.__name__.removesuffix("_node")

    @wraps(node)
    def wrapper(state: AgentState):
        _log("Node", f"{name} start")
        _log("State", snapshot(state))
        result = node(state)
        if isinstance(result, dict):
            _log("State", snapshot({**state, **result}))
        _log("Node", f"{name} end")
        return result

    return wrapper


def detect_required_keywords(question: str):
    q = question.lower()
    required = []

    if any(word in q for word in ["计算", "算", "结果", "多少", "值", "等于"]):
        required.append("计算结果")
    if any(word in q for word in ["时间", "当前", "现在", "时刻"]):
        if re.search(r"(当前时间|现在几点|现在.*时间|时间.*现在|时刻|查询.*时间)", q):
            required.append("当前时间")

    match = re.search(r"(?:关键词|关键字)\s*[:：]?\s*(.+)", q)
    if match:
        tail = match.group(1)
        seen = set()
        for item in re.split(r"[，,、;；\n]+", tail):
            keyword = item.strip().strip("。！？：:；;、,. ")
            if not keyword or keyword in seen:
                continue
            if any(marker in keyword for marker in NEGATION_MARKERS):
                # 命中“但不要出现这几个词”这类否定从句，通常意味着列举到此结束
                break
            required.append(keyword)
            seen.add(keyword)

    return required


def keyword_satisfied(keyword: str, answer_text: str) -> bool:
    """判断答案是否满足某个关键字要求：字面命中，或满足对应的证据型规则。"""
    text = (answer_text or "").lower()
    if keyword.lower() in text:
        return True
    rule = EVIDENCE_REQUIREMENTS.get(keyword)
    if not rule:
        return False
    return any(re.search(pattern, text) for pattern in rule["patterns"])


def build_keyword_rule(required_keywords, missing_keywords):
    """把关键字要求渲染成给 LLM 的提示文本，字面要求与证据型要求分开表述。"""
    literal = [kw for kw in required_keywords if kw not in EVIDENCE_REQUIREMENTS]
    evidence = [kw for kw in required_keywords if kw in EVIDENCE_REQUIREMENTS]

    parts = []
    if literal:
        if missing_keywords:
            missed = [kw for kw in missing_keywords if kw in literal] or literal
            parts.append(
                "上一步答案缺失关键字：" + "、".join(missed) + "。请重写，必须保证这些关键字都出现。"
            )
        else:
            parts.append(
                "最终答案中必须严格出现这些关键字：" + "、".join(literal) + "。不要省略任何一个。"
            )
    parts.extend(EVIDENCE_REQUIREMENTS[kw]["hint"] for kw in evidence)

    return "\n" + "\n".join(parts) if parts else ""


def _record_failure(state, failure_type, detail=""):
    """把本轮失败追加进历史，供后续判断“同一问题是否反复出现”。"""
    tag = f"{failure_type}:{detail}" if detail else failure_type
    return list(state.get("failure_history", [])) + [tag]


def _repeat_count(state, failure_type):
    return sum(
        1 for tag in state.get("failure_history", []) if tag.split(":", 1)[0] == failure_type
    )


def build_revision_instruction(state, failure_type, answer_text="", missing_keywords=None):
    """把审核失败分类成下一轮可直接执行的修正指令，而不是只回传一个状态字符串。"""
    missing_keywords = missing_keywords or []

    if failure_type == FAILURE_EMPTY:
        instruction = (
            "上一轮没有产出任何答案内容。本轮必须直接输出答案正文，不得留空，"
            "也不得只输出标题或前缀。"
        )
    elif failure_type == FAILURE_TOO_SHORT:
        length = len((answer_text or "").strip())
        instruction = (
            f"上一轮答案只有 {length} 字，信息量不足。本轮必须先分点列出至少 3 条具体信息，"
            "再给出结论，不要用一句话敷衍。"
        )
    elif failure_type == FAILURE_IRRELEVANT_DOCS:
        instruction = (
            "上一轮检索到的文档片段与问题无关，说明检索用词选错了。"
            "本轮请换一批关键词/换一个角度重新检索（可以先用工具搜索确认实体名称），"
            "不要重复上一轮的检索词。"
        )
    elif failure_type == FAILURE_MISSING_KEYWORD:
        literal = [kw for kw in missing_keywords if kw not in EVIDENCE_REQUIREMENTS]
        evidence = [kw for kw in missing_keywords if kw in EVIDENCE_REQUIREMENTS]
        chunks = []
        if literal:
            chunks.append(
                "本轮必须把以下词原样写进答案正文（可放进小标题或要点）："
                + "、".join(literal)
                + "。"
            )
        if evidence:
            chunks.append(
                "同时必须满足：" + "；".join(EVIDENCE_REQUIREMENTS[kw]["hint"] for kw in evidence)
            )
        instruction = "".join(chunks) or "本轮答案仍未满足关键字要求，请补齐。"
    else:
        return ""

    repeats = _repeat_count(state, failure_type)
    if repeats:
        strategy = {
            FAILURE_EMPTY: "先写出答案的第一段正文，再补充细节。",
            FAILURE_TOO_SHORT: "把答案改写成“要点 + 说明 + 结论”的结构。",
            FAILURE_MISSING_KEYWORD: "先逐字写出待包含的关键字，再围绕每个关键字扩写说明。",
            FAILURE_IRRELEVANT_DOCS: "改用文档里可能出现的实体全名或英文原名再检索一次。",
        }.get(failure_type, "更换一种表述方式。")
        instruction = (
            f"同一问题已第 {repeats + 1} 次失败，本轮不得沿用上一轮写法，改为：{strategy}"
            + instruction
        )

    return instruction


def build_revision_rule(state):
    """把修正指令渲染成注入生成节点提示词的文本。"""
    instruction = state.get("revision_instruction", "")
    if not instruction:
        return ""
    label = FAILURE_LABELS.get(state.get("failure_type", ""), "审核未通过")
    return f"\n上一轮未通过（{label}），本轮必须修正：{instruction}"


@_traced
def supervisor_node(state: AgentState):
    """入口节点：只做记帐与路由，工具选择权交给 researcher。"""
    last_user_message = next(
        (msg for msg in reversed(state["messages"]) if getattr(msg, "type", "") == "human"),
        None,
    )
    question = last_user_message.content if last_user_message else state.get("current_task", "")
    required_keywords = detect_required_keywords(question)

    current_status = state.get("status", "idle")
    retry_count = state.get("retry_count", 0)

    if current_status == "needs_replan":
        # 由 reviewer（关键字没答到）或 grade_documents（检索不相关）触发：
        # 带上前一轮的失败分类与修正指令重新走一遍检索→作答
        result = {
            "current_task": question,
            "task_type": "tool",
            "needs_tool": False,
            "steps": state.get("steps", 0) + 1,
            "status": "research",
            "required_keywords": required_keywords,
            # 保留上一轮缺失的关键字，让 analyst 拿到“上次错在哪”的反馈
            "missing_keywords": state.get("missing_keywords", []),
            "verification": "pending",
            "review_notes": (
                f"第 {retry_count + 1} 次修正（"
                f"{FAILURE_LABELS.get(state.get('failure_type', ''), '原因未分类')}）。"
            ),
            "retry_count": retry_count + 1,
            # 把失败原因给出的修正指令原样带到下一轮，避免重试只是重放同一个提问
            "failure_type": state.get("failure_type", ""),
            "revision_instruction": state.get("revision_instruction", ""),
            "failure_history": state.get("failure_history", []),
        }
        route = "researcher"
    elif current_status == "needs_clarification":
        exhausted = retry_count >= MAX_RETRIES
        result = {
            "current_task": question,
            "task_type": "direct",
            "needs_tool": False,
            "steps": state.get("steps", 0) + 1,
            # 澄清同样要消耗重试预算，否则空答案/过短答案会无限往返
            "status": "force_finalize" if exhausted else "answer",
            "required_keywords": required_keywords,
            "missing_keywords": state.get("missing_keywords", []),
            "verification": "fail" if exhausted else "pending",
            "review_notes": (
                "格式检查连续失败且已达到最大重试次数，强制结束。"
                if exhausted
                else f"需要进一步澄清：{FAILURE_LABELS.get(state.get('failure_type', ''), '答案为空或过短')}。"
            ),
            "retry_count": retry_count if exhausted else retry_count + 1,
            "failure_type": state.get("failure_type", ""),
            "revision_instruction": state.get("revision_instruction", ""),
            "failure_history": state.get("failure_history", []),
        }
        route = "END" if exhausted else "analyst"
    elif current_status == "force_finalize":
        result = {
            "current_task": question,
            "task_type": "direct",
            "needs_tool": False,
            "steps": state.get("steps", 0) + 1,
            "status": "force_finalize",
            "required_keywords": required_keywords,
            "missing_keywords": state.get("missing_keywords", []),
            "verification": "fail",
            "review_notes": "已达到最大重试次数，强制结束。",
            "retry_count": retry_count,
            "failure_type": state.get("failure_type", ""),
            "revision_instruction": state.get("revision_instruction", ""),
            "failure_history": state.get("failure_history", []),
        }
        route = "END"
    else:
        result = {
            "current_task": question,
            "task_type": "direct",
            "needs_tool": False,
            "steps": state.get("steps", 0) + 1,
            "status": "research",
            "required_keywords": required_keywords,
            "missing_keywords": [],
            "verification": "pending",
            "tool_outputs": [],
            "research_results": [],
            "retrieval_best_score": 0.0,
            "document_grade": "",
            "review_notes": "",
            "retry_count": retry_count,
            "failure_type": "",
            "revision_instruction": "",
            "failure_history": [],
        }
        route = "researcher"

    _log("Route", f"status={result['status']} -> {route}")
    return result


@_traced
def researcher_node(state: AgentState):
    """把问题与工具一起交给 LLM，由模型自行决定调用哪个工具、或直接作答。"""
    required_keywords = state.get("required_keywords", [])
    keyword_rule = build_keyword_rule(required_keywords, [])
    revision_rule = build_revision_rule(state)
    policy = knowledge_policy()
    tool_rounds = state.get("tool_rounds", 0)

    prompt = [
        SystemMessage(
            content=(
                "你是 Researcher，负责在作答前把需要的事实取回来。"
                "工具由你自己判断要不要用、用哪一个："
                "精确算术用 calculator、时效与外部信息用 web_search、"
                "知识库里的资料用 rag_search；不需要工具就直接给出简洁且可验证的结论。"
                "要调用工具必须用原生 tool_calls，不要凭空编造工具结果。"
                + (policy + "\n" if policy else "")
                + keyword_rule
                + revision_rule
                + (f"\n（本轮是第 {state.get('retry_count', 0) + 1} 轮尝试，"
                   f"已用工具轮次 {tool_rounds}）" if tool_rounds else "")
            )
        ),
        *state["messages"],
    ]

    if tool_rounds < MAX_TOOL_ROUNDS:
        response = llm.bind_tools(active_tools()).invoke(prompt)
    else:
        # 工具轮次预算已用尽，强制直接给出结论
        response = llm.invoke(prompt)
    tool_calls = getattr(response, "tool_calls", []) or []

    result = {
        "messages": [response],
        "steps": state.get("steps", 0) + 1,
        "status": "tool" if tool_calls else "answer",
        "needs_tool": bool(tool_calls),
        "task_type": "tool" if tool_calls else "direct",
    }

    _log("Route", f"status={result['status']} -> {'tool_executor' if tool_calls else 'analyst'}")
    return result


@_traced
def tool_executor_node(state: AgentState):
    """执行上一轮模型发出的所有 tool_calls。

    为什么不直接用 `ToolNode(ALL_TOOLS)`：**MCP 工具是运行期才知道的**——外部
    server 连上之后才知道有哪些工具，而 ToolNode 在构图时就把工具清单拷走了。
    这里改成按名字现取（`tools.find_tool`），内置工具与 MCP 工具走同一条路径。
    """
    last_ai = next(
        (
            message
            for message in reversed(state.get("messages", []))
            if getattr(message, "tool_calls", None)
        ),
        None,
    )
    calls = getattr(last_ai, "tool_calls", []) or []
    outputs: list[ToolMessage] = []

    for call in calls:
        name = call.get("name") or "tool"
        tool = find_tool(name)
        if tool is None:
            content = f"未知工具：{name}"
        else:
            try:
                content = str(tool.invoke(call.get("args") or {}))
            except Exception as exc:
                content = f"工具执行失败：{type(exc).__name__}: {exc}"
        outputs.append(
            ToolMessage(content=content, tool_call_id=call.get("id") or name, name=name)
        )

    _log("Tool", "执行：" + "、".join(call.get("name", "?") for call in calls) or "无")
    return {
        "messages": outputs,
        "steps": state.get("steps", 0) + 1,
    }


@_traced
def tool_result_reader_node(state: AgentState):
    """给工具结果打标签并累积到证据池，同时决定要不要做检索质量检查。"""
    tool_messages = [
        message for message in state.get("messages", []) if getattr(message, "type", "") == "tool"
    ]
    research_results = list(state.get("research_results", []))
    tool_outputs = list(state.get("tool_outputs", []))

    labeled: list[ToolMessage] = []
    used_rag = False
    best_score: float | None = None
    for message in tool_messages:
        content = str(getattr(message, "content", ""))
        name = getattr(message, "name", "tool") or "tool"
        if not content:
            continue
        if name == "rag_search":
            used_rag = True
            # 片段文本里带着「（相似度 0.712）」，取最高分作为质检触发依据
            scores = [float(value) for value in _SCORE_RE.findall(content)]
            if scores:
                best_score = max(best_score or 0.0, max(scores))
        # 工具结果一律打标签再进消息，避免裸值（如 "5"）让模型分不清来源与含义
        entry = f"[Tool: {name}] {tool_label(name)}：{content}"
        # 同 id 回写 ⇒ add_messages 会就地更新该条消息，而不是再追加一份
        labeled.append(
            ToolMessage(content=entry, tool_call_id=message.tool_call_id, id=message.id, name=name)
        )
        if entry not in tool_outputs:
            tool_outputs.append(entry)
        if entry not in research_results:
            research_results.append(entry)

    # 质检的触发条件：确实查了知识库 + 开着质检 + 还有重试预算 + 这批片段"看着可疑"
    # （最高相似度低于触发线）。相似度只用来决定要不要额外花一次 LLM 调用，
    # 不用来决定丢弃哪些证据——证据照常进上下文，由 analyst 与质检共同判断。
    suspicious = best_score is None or best_score < settings.kb_grade_score
    grade = (
        used_rag
        and settings.kb_grade_retrieval
        and state.get("retry_count", 0) < MAX_RETRIES
        and suspicious
    )

    result = {
        "tool_outputs": tool_outputs,
        "research_results": research_results,
        "retrieval_best_score": best_score or 0.0,
        "steps": state.get("steps", 0) + 1,
        "tool_rounds": state.get("tool_rounds", 0) + 1,
        "status": "grade" if grade else "answer",
        "messages": labeled,
    }

    _log(
        "Route",
        f"最高相似度 {best_score if best_score is not None else '未知'}"
        f"（质检触发线 {settings.kb_grade_score}）"
        f" → {'grade_documents' if grade else 'analyst（跳过质检）'}",
    )
    return result


_GRADE_PROMPT = (
    "你是检索质量评估员。判断【检索片段】里是否包含回答【用户问题】所需的线索。\n"
    "只要片段中有关键词或语义上与问题相关的内容，就判 relevant；"
    "片段与问题完全无关（例如问的是甲、片段全在讲乙）判 irrelevant。\n"
    '只输出 JSON，不要多余文字：{"grade":"relevant|irrelevant","reason":"不超过15字"}'
)


@_traced
def grade_documents_node(state: AgentState):
    """检索质量检查（adaptive-RAG / CRAG 的 grading 步）。

    不相关不硬答，而是带着"换关键词重检"的指令回到 supervisor；
    重试预算由 supervisor 统一管理，不会无限循环。
    """
    question = state.get("current_task", "")
    context = "\n\n".join(state.get("research_results") or [])[:4000]
    grade, reason = "relevant", ""

    try:
        response = llm.invoke(
            [
                SystemMessage(content=_GRADE_PROMPT),
                HumanMessage(content=f"【用户问题】{question}\n\n【检索片段】\n{context}"),
            ]
        )
        text = str(getattr(response, "content", "") or "")
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            data = json.loads(match.group(0))
            grade = "irrelevant" if str(data.get("grade", "")).lower().startswith("irrel") else "relevant"
            reason = str(data.get("reason", ""))[:40]
    except Exception as exc:
        reason = f"评估失败：{type(exc).__name__}"

    if grade == "irrelevant":
        detail = reason or "grader"
        result = {
            "document_grade": grade,
            "steps": state.get("steps", 0) + 1,
            "status": "rewrite",
            "failure_type": FAILURE_IRRELEVANT_DOCS,
            "revision_instruction": build_revision_instruction(
                state, FAILURE_IRRELEVANT_DOCS
            ),
            "failure_history": _record_failure(state, FAILURE_IRRELEVANT_DOCS, detail),
            "review_notes": f"检索片段与问题无关（{detail}），换查询重检。",
        }
        _log("Route", f"status=rewrite -> supervisor（检索不相关：{detail}）")
        return result

    _log("Route", f"status=answer -> analyst（检索相关：{reason or 'ok'}）")
    return {
        "document_grade": grade,
        "steps": state.get("steps", 0) + 1,
        "status": "answer",
    }


@_traced
def analyst_node(state: AgentState):
    """把工具取回的证据汇总成最终答案。"""
    question = state.get("current_task", "")
    required_keywords = state.get("required_keywords", [])
    missing_keywords = state.get("missing_keywords", [])
    context = "\n\n".join(state.get("research_results", []))
    used_tool = bool(state.get("needs_tool"))

    keyword_rule = build_keyword_rule(required_keywords, missing_keywords)
    revision_rule = build_revision_rule(state)

    # 证据规则必须跟着"本轮到底有没有调用工具"走：
    # 用了工具却不给依据 ⇒ 只能说资料不足，不许编；
    # 没调工具（researcher 判断不需要外部资料）⇒ 让模型用自己的知识回答，
    # 否则会出现"检索没查、又不许用自身知识"的荒谬拒答。
    evidence_rule = (
        "本轮已经调用过工具，请**只依据**工具返回的资料作答；"
        "资料不足以回答时明确说明「资料不足」，不要凭记忆编造。"
        if used_tool
        else "本轮没有调用任何工具（因为不需要外部资料），请直接依据你自己的知识作答，"
        "并在答案里注明「依据模型自身知识」；只有确实没有把握时才说明无法确定。"
    )

    prompt = [
        SystemMessage(
            content=(
                "你是 Analyst。根据用户问题与已取得的资料输出最终答案。"
                "答案必须简洁、可验证，并符合关键字要求。"
                + evidence_rule
                + keyword_rule
                + revision_rule
            )
        ),
        HumanMessage(
            content=f"用户问题：{question}\n\n参考资料：\n{context or '（本轮没有调用工具，无参考资料）'}"
        ),
    ]

    response = llm.invoke(prompt)

    result = {
        "messages": [response],
        "final_answer": response.content,
        "steps": state.get("steps", 0) + 1,
        "status": "review",
    }
    _log("Route", "status=review -> reviewer")
    return result


def _failure_payload(state, failure_type, detail, answer_text="", missing_keywords=None):
    """构造审核失败时写回 state 的字段：失败类型 + 修正指令 + 失败历史。"""
    return {
        "failure_type": failure_type,
        "revision_instruction": build_revision_instruction(
            state, failure_type, answer_text=answer_text, missing_keywords=missing_keywords
        ),
        "failure_history": _record_failure(state, failure_type, detail),
    }


def format_check_node(state: AgentState):
    answer_text = state.get("final_answer") or ""
    stripped = answer_text.strip()

    if not stripped:
        failure_type, notes = FAILURE_EMPTY, "格式检查失败：答案为空。"
    elif len(stripped) < 10:
        failure_type, notes = FAILURE_TOO_SHORT, "格式检查失败：答案过短，可能缺少具体内容。"
    else:
        return {
            "missing_keywords": state.get("missing_keywords", []),
            "verification": "pending",
            "review_notes": "格式检查通过。",
            "status": "content_check",
        }

    return {
        "missing_keywords": state.get("missing_keywords", []),
        "verification": "fail",
        "review_notes": notes,
        "status": "needs_clarification",
        **_failure_payload(state, failure_type, "", answer_text=answer_text),
    }


def content_check_node(state: AgentState):
    answer_text = state.get("final_answer") or ""
    required_keywords = state.get("required_keywords", [])
    missing_keywords = [kw for kw in required_keywords if not keyword_satisfied(kw, answer_text)]
    retry_count = state.get("retry_count", 0)

    if missing_keywords:
        payload = _failure_payload(
            state,
            FAILURE_MISSING_KEYWORD,
            ",".join(missing_keywords),
            answer_text=answer_text,
            missing_keywords=missing_keywords,
        )
        if retry_count >= MAX_RETRIES:
            return {
                "missing_keywords": missing_keywords,
                "verification": "fail",
                "review_notes": f"关键字缺失：{','.join(missing_keywords)}。已达到最大重试次数。",
                "status": "force_finalize",
                **payload,
            }
        return {
            "missing_keywords": missing_keywords,
            "verification": "fail",
            "review_notes": f"关键字缺失：{','.join(missing_keywords)}，需要重检索与补充信息。",
            "status": "needs_replan",
            **payload,
        }

    return {
        "missing_keywords": [],
        "verification": "pass",
        "review_notes": "答案满足关键字要求，完成审核。",
        "status": "done",
        "failure_type": "",
        "revision_instruction": "",
    }


@_traced
def reviewer_node(state: AgentState):
    format_result = format_check_node(state)
    if format_result["status"] != "content_check":
        _log("Route", f"verification={format_result['verification']} -> {format_result['status']}")
        return format_result

    content_result = content_check_node({**state, **format_result})
    _log("Route", f"verification={content_result['verification']} -> {content_result['status']}")
    return content_result


workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("tool_result_reader", tool_result_reader_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("reviewer", reviewer_node)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state.get("status", "research"),
    {
        "research": "researcher",
        "answer": "analyst",
        "force_finalize": END,
    },
)
# 模型自己决定：调工具，还是直接作答
workflow.add_conditional_edges(
    "researcher",
    lambda state: state.get("status", "answer"),
    {
        "tool": "tool_executor",
        "answer": "analyst",
    },
)
workflow.add_edge("tool_executor", "tool_result_reader")
# 检索结果要先过质量检查，再决定作答还是换个查询重检
workflow.add_conditional_edges(
    "tool_result_reader",
    lambda state: state.get("status", "answer"),
    {
        "grade": "grade_documents",
        "answer": "analyst",
    },
)
workflow.add_conditional_edges(
    "grade_documents",
    lambda state: state.get("status", "answer"),
    {
        "answer": "analyst",
        "rewrite": "supervisor",
    },
)
workflow.add_edge("analyst", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    lambda state: state.get("status", "done"),
    {
        "done": END,
        "needs_replan": "supervisor",
        "force_finalize": END,
        "needs_clarification": "supervisor",
    },
)

app = workflow.compile()

if DEBUG_GRAPH_OUTPUT:
    # 打印 mermaid 图，并尝试导出一张 PNG（需要额外依赖，失败不影响运行）
    print(app.get_graph().draw_mermaid())
    try:
        graph_png = Path(__file__).resolve().parents[1] / "agent_graph.png"
        graph_png.write_bytes(app.get_graph().draw_png())
        print(f"图已导出：{graph_png}")
    except Exception as exc:
        print(f"[提示] 导出 PNG 失败（{type(exc).__name__}），已跳过")
