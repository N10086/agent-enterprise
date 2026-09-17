"""对 Agent 暴露的工具定义。

这里只放「工具签名 + 描述 + 各自的轻量实现」：
  - 搜索后端在 `web_search.py`
  - 知识库接入在 `knowledge.py`

工具描述统一写成"什么时候该调用 / 什么时候不该调用"，因为绑定给模型的是
描述本身，描述的质量直接决定模型会不会在正确的时机调用正确的工具。
"""
from __future__ import annotations

import ast
import operator
from datetime import datetime
from typing import Annotated

from langchain_core.tools import tool

from .config import get_settings
from .rag_chain import retrieve_context
from .web_search import run_web_search

# ---- calculator 的受限求值 ----------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_ABS_OPERAND = 10**12
_MAX_EXPONENT = 100


def _eval_node(node):
    """只放行数字与算术运算符，避免把模型输出直接交给 eval。"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        if abs(node.value) > _MAX_ABS_OPERAND:
            raise ValueError("数值过大")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("指数过大")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"不支持的表达式：{ast.dump(node)}")


# ---- 工具 ----------------------------------------------------------------


@tool
def calculator(
    expression: Annotated[
        str,
        "只含数字与 + - * / // % ** ( ) 的数学表达式，例如 \"(2 + 3) * 4\"。"
        "不要传入文字、单位、问号或等号（\"2+3=?\"、\"20 元 + 5 元\"都会解析失败）。",
    ],
) -> str:
    """计算数学表达式的精确数值结果（安全求值，不执行任意代码）。

    什么时候该调用：任何需要准确算术的场合——四则运算、百分比、乘方、取余，
    以及"帮我算一下 / 等于多少 / 一共多少 / 平均是多少 / 涨了几成"这类请求。
    什么时候不该调用：问题要的是外部实时信息（天气、新闻、当前时间）或纯文字任务时，
    不要用本工具硬凑数字。
    """
    try:
        return str(_eval_node(ast.parse(expression, mode="eval")))
    except Exception as exc:
        return f"Error: {exc}"


@tool
def get_current_time() -> str:
    """获取当前的系统日期与时间（格式 YYYY-MM-DD HH:MM:SS）。

    什么时候该调用：问题涉及任何"现在 / 今天 / 此刻 / 最近 / 还有几天"的时效性判断时，
    先调用本工具拿到当下日期时间，包括需要用当前日期做推算的场景
    （"今天星期几"、"距离春节还有几天"）。
    什么时候不该调用：与当前时刻无关的问题（纯知识问答、写作、翻译、数学计算）。

    能力限制：本工具只返回日期时间，不返回天气、新闻等外部数据；
    那类信息请改用 web_search。如果两个工具都给不出答案，应向用户说明限制，不要凭空编造。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def web_search(
    query: Annotated[
        str,
        "搜索关键词，像真人搜索那样写会得到更好结果，例如 \"北京 2026年9月 天气 预报\"，"
        "不要写成整句话或带问号的句子。",
    ],
    max_results: Annotated[int, "返回结果条数，1-8，默认 5"] = 5,
) -> str:
    """联网搜索最新信息，返回若干条"标题 / 链接 / 摘要"。

    什么时候该调用：问题需要训练数据之外或随时间变化的信息时——天气、新闻、赛事、
    股价、营业时间、票价、政策、产品价格、行程与交通的当前状况，以及任何带
    "最近 / 目前 / 今年 / 最新"的事实性问题；需要给出可核实来源时也应调用。
    什么时候不该调用：纯算术用 calculator；只需要当前日期时间用 get_current_time；
    知识库里已有的内部资料用 rag_search；纯写作、翻译、改写等无需外部信息的任务不要搜索。

    注意：摘要来自搜索引擎，可能过期或不准确，回答时应说明信息来源与时间；
    如果搜索失败或没有结果，如实告知用户无法获取实时信息，不要凭记忆编造。
    """
    try:
        max_results = max(1, min(int(max_results), 8))
    except (TypeError, ValueError):
        max_results = 5
    return run_web_search(query, max_results)


@tool
def rag_search(
    query: Annotated[
        str,
        "检索用的自然语言问句或关键词，例如 \"Scott Derrickson 是哪国人\"、"
        "\"Freakonomics 是哪一年的纪录片\"。",
    ],
    top_k: Annotated[int, "返回的文档片段数，1-8"] = get_settings().kb_top_k,
) -> str:
    """从本地知识库检索相关文档片段，返回片段正文与出处。

    检索走的是一条完整链路：先把问题改写成多条检索语句、再为它们生成假设性答案
    （HyDE），然后多路检索向量库并融合取最相似的 5 个片段——所以你只需要给一个
    自然的问句或关键词，改写由链路内部完成。

    什么时候该调用：问题涉及知识库收录的资料，且需要给出可溯源依据时——
    例如数据集/内部文档里的事实型问答、需要引用原文佐证的查询。
    检索结果会标明出自哪个文档的哪一节，便于在答案里标注来源。
    什么时候不该调用：需要最新或外部实时信息用 web_search；纯算术用 calculator；
    与知识库无关的通用常识、写作、翻译任务不必检索。

    注意：如果返回的片段与问题无关（或提示"没有找到相关内容"），
    说明知识库里没有依据，应如实说明，不要用常识硬凑成有出处的结论。
    """
    return retrieve_context(query, top_k)


# ---- 工具注册表 ----------------------------------------------------------
#
# 内置工具是写死的四个；MCP 工具是"运行时才知道有哪些"的，所以统一注册表只能在
# 调用时求值，不能用模块级常量。绑定侧（researcher）与执行侧（tool_executor）都走
# 这里，避免两份工具清单对不上。


def builtin_tools() -> list:
    return [calculator, get_current_time, web_search, rag_search]


def all_tools() -> list:
    """内置工具 + 外部 MCP 工具。

    MCP 工具与内置工具重名时以内置为准（由 `mcp_tools(reserved=...)` 过滤）；
    未配置 MCP 时就是原来的四个工具。
    """
    builtin = builtin_tools()
    from .mcp_client import mcp_tools

    return builtin + mcp_tools(reserved={tool.name for tool in builtin})


def find_tool(name: str):
    """按名字取工具；找不到返回 None（执行侧据此报错，而不是抛异常炸掉整张图）。"""
    return next((tool for tool in all_tools() if tool.name == name), None)


__all__ = [
    "all_tools",
    "builtin_tools",
    "calculator",
    "find_tool",
    "get_current_time",
    "rag_search",
    "web_search",
]
