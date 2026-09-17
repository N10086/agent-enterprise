"""MCP 客户端：把**外部** MCP server 的工具接进本项目的 Agent 图。

服务端那半边在 `mcp_server.py`（把本项目的工具暴露给别人）；这里补的是另一半：
本项目的 Agent 也能调用别人的 MCP 工具。加上它之后，工具集就不再是写死的四个，
而是「内置工具 + 任意 MCP server 提供的工具」。

**为什么需要一层桥**：MCP 的会话是异步的（anyio 事件循环 + 长连接），而 LangGraph
的图是同步跑的，工具必须是普通可调用对象。所以这里在**后台线程里跑一个事件循环**，
把会话一直挂住，工具调用通过 `run_coroutine_threadsafe` 同步等待结果——
连接只建一次，不是每次调用都重启子进程。

配置（环境变量 `MCP_SERVERS`，JSON）：

    {
      "agent-enterprise-tools": {
        "command": "path/to/python.exe",
        "args": ["path/to/mcp_server.py"],
        "env": {"KB_FAISS_INDEX": "..."}
      },
      "some-remote": {"url": "http://127.0.0.1:8000/mcp"}
    }

不配置时返回空列表，图的行为与没有 MCP 时完全一致（默认不改变任何评测结果）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from contextlib import AsyncExitStack
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from .config import get_settings

logger = logging.getLogger(__name__)

#: JSON Schema 类型 → Python 类型
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def parse_servers(raw: str) -> dict:
    """解析 `MCP_SERVERS`。格式错误时返回空配置并打日志，不让图起不来。"""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("MCP_SERVERS 不是合法 JSON（%s），已忽略", exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("MCP_SERVERS 必须是 {名字: 配置} 的字典，已忽略")
        return {}
    return {str(name): cfg for name, cfg in data.items() if isinstance(cfg, dict)}


def _args_model(tool_name: str, schema: dict | None):
    """把 MCP 的 inputSchema 转成 pydantic 参数模型，让模型能看到正确的参数签名。"""
    properties = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    fields: dict = {}
    for prop, spec in properties.items():
        spec = spec if isinstance(spec, dict) else {}
        py_type = _TYPE_MAP.get(str(spec.get("type", "string")).lower(), str)
        description = str(spec.get("description") or "")
        if prop in required:
            fields[prop] = (py_type, Field(..., description=description))
        else:
            fields[prop] = (Optional[py_type], Field(default=None, description=description))
    return create_model(f"{tool_name}_args", **fields) if fields else create_model(f"{tool_name}_args")


class MCPBridge:
    """后台事件循环 + 长连接会话，对外只暴露同步接口。"""

    def __init__(self, servers: dict):
        self.servers = servers
        self.sessions: dict = {}
        self.tools: list[dict] = []
        self.errors: list[str] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None

    # ---- 生命周期 ------------------------------------------------------
    def start(self, timeout: float = 60.0) -> bool:
        if not self.servers or self._thread is not None:
            return bool(self.sessions)
        self._thread = threading.Thread(target=self._run, name="mcp-bridge", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self.errors.append("启动超时")
            return False
        return bool(self.sessions)

    def _run(self) -> None:
        """后台线程入口：一个长驻协程把会话挂住，直到 close() 让它退出。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        try:
            # AsyncExitStack 必须在"同一个任务"里进出，否则 anyio 的 task group
            # 会报 "exit cancel scope in a different task" —— 所以连接、挂起、关闭
            # 都留在这个常驻协程里，工具调用只是往同一循环里投递请求。
            async with AsyncExitStack() as stack:
                for name, cfg in self.servers.items():
                    try:
                        session = await self._connect(stack, name, cfg)
                    except Exception as exc:
                        self.errors.append(f"{name}: {type(exc).__name__}: {exc}")
                        logger.warning("MCP server %s 连接失败：%s", name, exc)
                        continue
                    self.sessions[name] = session
                    await self._collect_tools(name, session)
                self._ready.set()
                await self._stop.wait()
        finally:
            self._ready.set()

    async def _connect(self, stack: AsyncExitStack, name: str, cfg: dict):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if cfg.get("url"):
            from mcp.client.streamable_http import streamable_http_client

            read, write, _ = await stack.enter_async_context(streamable_http_client(cfg["url"]))
        else:
            command = cfg.get("command") or "python"
            params = StdioServerParameters(
                command=command,
                args=[str(arg) for arg in (cfg.get("args") or [])],
                env={**os.environ, **(cfg.get("env") or {})} or None,
            )
            read, write = await stack.enter_async_context(stdio_client(params))

        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        logger.info("MCP server 已连接：%s", name)
        return session

    async def _collect_tools(self, name: str, session) -> None:
        listed = await session.list_tools()
        for tool in listed.tools:
            # mcp 2.x 的字段名是 input_schema（旧版/其他实现可能是 inputSchema），两个都认
            schema = (
                getattr(tool, "input_schema", None)
                or getattr(tool, "inputSchema", None)
                or {}
            )
            self.tools.append(
                {
                    "server": name,
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": schema,
                }
            )

    def close(self, timeout: float = 10.0) -> None:
        if self._loop is None or self._stop is None:
            return
        self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout)
        self.sessions.clear()

    # ---- 调用 ----------------------------------------------------------
    async def _acall(self, server: str, tool: str, arguments: dict):
        session = self.sessions[server]
        return await session.call_tool(tool, arguments)

    def call(self, server: str, tool: str, arguments: dict, timeout: float = 120.0) -> str:
        """同步调用远端工具，返回文本结果。异常一律转成可读文本，不炸整张图。"""
        if self._loop is None or server not in self.sessions:
            return f"[MCP] 未连接：{server}"
        future = asyncio.run_coroutine_threadsafe(self._acall(server, tool, arguments), self._loop)
        try:
            result = future.result(timeout)
        except Exception as exc:
            return f"[MCP] 调用失败：{type(exc).__name__}: {exc}"

        parts = []
        for item in getattr(result, "content", []) or []:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        body = "\n".join(parts)
        if getattr(result, "isError", False):
            return f"[MCP] 工具报错：{body}"
        return body

    # ---- 转成 LangChain 工具 -------------------------------------------
    def langchain_tools(self, reserved: set[str] | None = None) -> list:
        """把远端工具包成 LangChain 工具；与内置工具重名的会被跳过。"""
        reserved = reserved or set()
        built: list = []
        used: set[str] = set()
        for item in self.tools:
            name = item["name"]
            if name in reserved or name in used:
                logger.info("MCP 工具 %s 与已有工具重名，已跳过", name)
                continue
            used.add(name)
            built.append(self._as_tool(item))
        return built

    def _as_tool(self, item: dict) -> StructuredTool:
        server, name = item["server"], item["name"]
        bridge = self

        def _call(**kwargs) -> str:
            arguments = {key: value for key, value in kwargs.items() if value is not None}
            return bridge.call(server, name, arguments)

        description = item["description"] or f"由 MCP server「{server}」提供的工具 {name}。"
        return StructuredTool.from_function(
            func=_call,
            name=name,
            description=f"{description}（来源：MCP server「{server}」）",
            args_schema=_args_model(name, item["input_schema"]),
        )

    def status(self) -> str:
        if not self.servers:
            return "MCP：未配置外部 server（MCP_SERVERS 为空）"
        lines = [f"MCP：已配置 {len(self.servers)} 个 server，连接成功 {len(self.sessions)} 个"]
        for name, session in self.sessions.items():
            count = sum(1 for tool in self.tools if tool["server"] == name)
            lines.append(f"  - {name}：{count} 个工具")
        lines.extend(f"  ! {err}" for err in self.errors)
        return "\n".join(lines)


#: 全局单例：连接只建一次，图里每次调用都复用
_BRIDGE: MCPBridge | None = None
_LOCK = threading.Lock()


def get_bridge() -> MCPBridge | None:
    """按需建立 MCP 连接（未配置时返回 None，完全不介入）。"""
    global _BRIDGE
    if _BRIDGE is not None:
        return _BRIDGE
    with _LOCK:
        if _BRIDGE is None:
            servers = parse_servers(get_settings().mcp_servers)
            if not servers:
                return None
            bridge = MCPBridge(servers)
            bridge.start()
            _BRIDGE = bridge
    return _BRIDGE


def mcp_tools(reserved: set[str] | None = None) -> list:
    """外部 MCP 工具的 LangChain 包装（未配置/连不上时返回空列表）。"""
    bridge = get_bridge()
    if bridge is None:
        return []
    return bridge.langchain_tools(reserved=reserved)


def mcp_status() -> str:
    bridge = get_bridge()
    return "MCP：未配置外部 server（MCP_SERVERS 为空）" if bridge is None else bridge.status()


def reset() -> None:
    """关闭并清空单例（测试用）。"""
    global _BRIDGE
    with _LOCK:
        if _BRIDGE is not None:
            _BRIDGE.close()
        _BRIDGE = None


__all__ = ["MCPBridge", "get_bridge", "mcp_status", "mcp_tools", "parse_servers", "reset"]
