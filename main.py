"""统一入口。

    python main.py ui                     启动本机网页界面（推荐）
    python main.py ask --question "..."   命令行直接问一句
    python main.py demo                   跑一个内置示例问题
    python mcp_server.py                  把工具以 MCP 协议暴露给别的客户端
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


def _run_agent_question(question: str) -> str:
    from app.runner import run_agent

    return str(run_agent(question))


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent Enterprise 统一入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    q_parser = subparsers.add_parser("ask", help="直接提问 Agent")
    q_parser.add_argument("--question", required=True, help="要问的问题")

    demo_parser = subparsers.add_parser("demo", help="运行内置示例问题")
    demo_parser.add_argument(
        "--question",
        default="请用三句话解释什么是检索增强生成（RAG）。",
    )

    ui_parser = subparsers.add_parser("ui", help="启动本机网页界面（推荐入口）")
    ui_parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认只允许本机访问")
    ui_parser.add_argument("--port", type=int, default=8760, help="端口，默认 8760")
    ui_parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")

    args = parser.parse_args()

    if args.command == "ui":
        from serve_ui import serve

        return serve(args.host, args.port, not args.no_browser)

    question = args.question
    answer = _run_agent_question(question)
    print("Q:", question)
    print("A:", answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
