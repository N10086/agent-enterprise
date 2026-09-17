"""把本项目的工具通过 MCP（Model Context Protocol）暴露出去。

**为什么要这一层**：项目里的 calculator / get_current_time / web_search / rag_search
原本只服务于本项目的 LangGraph 图，是"私有能力"。封装成 MCP server 之后，
Claude Desktop、Cursor、以及任何 MCP 客户端都能直接调用同一套工具——
工具与 Agent 解耦，不必为每个宿主再写一遍适配。

传输方式（`--transport`）：
  - `stdio`（默认）：本机进程间通信，供桌面端 / IDE 以子进程方式拉起；
  - `streamable-http`：给远程或多个客户端复用。

在 Claude Desktop / Cursor 里注册（`claude_desktop_config.json`）：

    {
      "mcpServers": {
        "agent-enterprise-tools": {
          "command": "<enterprise-research-agent>/.venv/Scripts/python.exe",
          "args": ["<test>/mcp_server.py"],
          "env": {"KB_VECTOR_STORE": "<test>/public/vector_store"}
        }
      }
    }

注意：`rag_search` 需要向量库存在（先跑 `python build_rag_kb.py`）；
未建库时该工具会返回明确的报错文本，而不是让整个 server 起不来。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402

# 复用现有实现，避免工具逻辑出现第二份拷贝：
#   - calculator / get_current_time 是 LangChain StructuredTool，用 .func 取回原函数
#   - web_search 直接调用底层实现（LLM 侧的工具包装不影响这里）
#   - rag_search 走与 Agent 图**完全同一条**检索链路（multi-query + HyDE + RRF 融合）
from app.rag_chain import retrieve_context  # noqa: E402
from app.tools import calculator as calculator_tool  # noqa: E402
from app.tools import get_current_time as time_tool  # noqa: E402
from app.web_search import run_web_search  # noqa: E402

server = MCPServer(
    name="agent-enterprise-tools",
    description="企业级 Agent 项目对外开放的工具集（计算/时间/联网搜索/本地知识库检索）",
)

#: MCP 工具描述 = 模型决定"何时调用"的唯一依据，因此写清适用边界
CALC_DESC = (
    "计算数学表达式的精确数值结果。适用于四则运算、百分比、乘方、取余；"
    "参数只接受数字与 + - * / // % ** ( ) 组成的表达式。"
)
TIME_DESC = (
    "获取当前系统日期与时间（YYYY-MM-DD HH:MM:SS）。"
    "适用于任何涉及「现在/今天/最近」的时效性判断；不返回天气、新闻等外部数据。"
)
SEARCH_DESC = (
    "联网搜索最新信息，返回若干条「标题/链接/摘要」。"
    "适用于天气、新闻、赛事、价格等训练数据之外或随时间变化的信息；"
    "搜索失败时返回错误说明，调用方应如实告知用户而不是编造。"
)
RAG_DESC = (
    "从本地知识库检索相关文档片段，返回片段正文与出处。"
    "内部会自动改写查询并生成假设性答案（HyDE）后多路检索、融合取最相似的片段，"
    "因此只需传入自然的问句或关键词。适用于知识库收录资料的事实型问答，便于在答案中标注来源。"
)


@server.tool(name="calculator", description=CALC_DESC)
def mcp_calculator(expression: str) -> str:
    return calculator_tool.func(expression)


@server.tool(name="get_current_time", description=TIME_DESC)
def mcp_get_current_time() -> str:
    return time_tool.func()


@server.tool(name="web_search", description=SEARCH_DESC)
def mcp_web_search(query: str, max_results: int = 5) -> str:
    try:
        count = max(1, min(int(max_results), 8))
    except (TypeError, ValueError):
        count = 5
    return run_web_search(query, count)


@server.tool(name="rag_search", description=RAG_DESC)
def mcp_rag_search(query: str, top_k: int = 5) -> str:
    """与 Agent 图内的 rag_search 共用 `retrieve_context`，保证两边检索质量一致。"""
    from app.config import get_settings

    return retrieve_context(query, top_k or get_settings().kb_top_k)


def main() -> int:
    parser = argparse.ArgumentParser(description="以 MCP 协议暴露本项目工具")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="stdio（默认，供桌面端/IDE 拉起）或 streamable-http（远程复用）",
    )
    args = parser.parse_args()
    server.run(args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
