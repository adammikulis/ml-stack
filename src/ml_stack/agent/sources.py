"""Where an agent's tools come from: Python callables, or an MCP server over stdio or HTTP.

A source lists its tools as MCP-shaped dicts (``name``, ``description``, ``inputSchema``)
and calls one by name. `McpTools` needs the ``mcp`` extra.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = ["FunctionTools", "McpTools", "ToolOutput", "ToolSource"]


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """What a tool answered: its text, any structured payload, and whether it failed."""

    text: str
    structured: Any = None
    is_error: bool = False


class ToolSource(Protocol):
    """The two calls an agent makes of its tools."""

    async def list_tools(self) -> list[dict[str, Any]]: ...

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutput: ...


def _text_of(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False,
                                                           default=str)


class FunctionTools:
    """Tools backed by Python callables, run in a worker thread so a slow one does not
    stall the loop.

    Built from ``(schema, callable)`` pairs where schema is an OpenAI function schema, or
    from MCP-shaped dicts with ``fns`` giving ``{name: callable}``.
    """

    #: Raised by a callable and reported to the model as the tool's answer.
    FAILURES = (ValueError, TypeError, LookupError, OSError, RuntimeError, ArithmeticError)

    def __init__(self, tools: Iterable[Any], fns: Mapping[str, Callable[..., Any]] | None = None
                 ) -> None:
        self._tools: list[dict[str, Any]] = []
        self._fns: dict[str, Callable[..., Any]] = dict(fns or {})
        for item in tools:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                schema, fn = item
                spec = schema.get("function", schema)
                self._tools.append({"name": spec["name"],
                                    "description": spec.get("description", ""),
                                    "inputSchema": spec.get("parameters") or {
                                        "type": "object", "properties": {}}})
                self._fns[spec["name"]] = fn
            else:
                self._tools.append(dict(item))

    async def list_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutput:
        fn = self._fns.get(name)
        if fn is None:
            return ToolOutput(f"no such tool: {name}", is_error=True)
        try:
            value = await asyncio.to_thread(fn, **arguments)
        except self.FAILURES as exc:
            return ToolOutput(f"{type(exc).__name__}: {exc}", is_error=True)
        return ToolOutput(_text_of(value), structured=value)


class McpTools:
    """An MCP server's tools through the official ``mcp`` client, used as an async context
    manager: ``async with McpTools.stdio("python", ["-m", "server"]) as tools``."""

    def __init__(self, connect: Callable[[AsyncExitStack], Any]) -> None:
        self._connect = connect
        self._stack = AsyncExitStack()
        self._session: Any = None

    @classmethod
    def stdio(cls, command: str, args: Sequence[str] = (),
              env: Mapping[str, str] | None = None) -> McpTools:
        """A server spawned as ``command args`` and spoken to over its stdin and stdout."""
        async def connect(stack: AsyncExitStack) -> Any:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(command=command, args=list(args),
                                           env=dict(env) if env else None)
            return await stack.enter_async_context(stdio_client(params))

        return cls(connect)

    @classmethod
    def http(cls, url: str) -> McpTools:
        """A server reached at ``url`` over streamable HTTP."""
        async def connect(stack: AsyncExitStack) -> Any:
            from mcp.client.streamable_http import streamable_http_client

            return await stack.enter_async_context(streamable_http_client(url))

        return cls(connect)

    async def __aenter__(self) -> McpTools:
        from mcp import ClientSession

        opened = False
        try:
            streams = await self._connect(self._stack)
            self._session = await self._stack.enter_async_context(
                ClientSession(streams[0], streams[1]))
            await self._session.initialize()
            opened = True
        finally:
            if not opened:
                await self._stack.aclose()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    async def list_tools(self) -> list[dict[str, Any]]:
        listed = await self._session.list_tools()
        return [{"name": t.name, "description": t.description or "",
                 "inputSchema": t.model_dump(by_alias=True)["inputSchema"]}
                for t in listed.tools]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutput:
        result = (await self._session.call_tool(name, arguments)).model_dump(
            by_alias=True, exclude_none=True)
        text = "\n".join(c.get("text", "") for c in result.get("content") or []
                         if c.get("type") == "text")
        structured = result.get("structuredContent")
        return ToolOutput(text or (_text_of(structured) if structured is not None else ""),
                          structured=structured, is_error=bool(result.get("isError")))
