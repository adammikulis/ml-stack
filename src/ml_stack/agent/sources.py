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

__all__ = ["FunctionTools", "McpAuthError", "McpTools", "ToolOutput", "ToolSource"]


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


class McpAuthError(RuntimeError):
    """The MCP server refused the credentials (HTTP 401 or 403)."""


SENSITIVE = ("authorization", "proxy-authorization", "x-api-key", "cookie")
"""Header names whose values are never shown."""

CHECKED = (401, 403)


class McpTools:
    """An MCP server's tools through the official ``mcp`` client, used as an async context
    manager: ``async with McpTools.stdio("python", ["-m", "server"]) as tools``.

    Credentials given to `http` are sent on every request, are re-read from a callable
    ``bearer`` each time the connection is made, and are cut out of everything the server
    sends back and out of this object's repr.
    """

    def __init__(self, connect: Callable[[AsyncExitStack, McpTools], Any],
                 label: str = "mcp server") -> None:
        self._connect = connect
        self._label = label
        self._stack = AsyncExitStack()
        self._session: Any = None
        self._secrets: list[str] = []
        self._refused = 0

    def __repr__(self) -> str:
        return f"McpTools({self._label})"

    @classmethod
    def stdio(cls, command: str, args: Sequence[str] = (),
              env: Mapping[str, str] | None = None) -> McpTools:
        """A server spawned as ``command args`` and spoken to over its stdin and stdout."""
        async def connect(stack: AsyncExitStack, owner: McpTools) -> Any:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(command=command, args=list(args),
                                           env=dict(env) if env else None)
            return await stack.enter_async_context(stdio_client(params))

        return cls(connect, command)

    @classmethod
    def http(cls, url: str, *, headers: Mapping[str, str] | None = None,
             bearer: str | Callable[[], str] | None = None) -> McpTools:
        """A server reached at ``url`` over streamable HTTP. ``headers`` go on every request;
        ``bearer`` (a token, or a function returning the current one) is sent as
        ``Authorization: Bearer ...``."""
        async def connect(stack: AsyncExitStack, owner: McpTools) -> Any:
            sent = dict(headers or {})
            token = bearer() if callable(bearer) else bearer
            if token:
                sent["Authorization"] = f"Bearer {token}"
            owner._secrets = [v for k, v in sent.items() if k.lower() in SENSITIVE]
            owner._secrets += [token] if token else []
            return await stack.enter_async_context(owner._http_streams(url, sent))

        return cls(connect, url.split("?")[0])

    def _http_streams(self, url: str, headers: dict[str, str]) -> Any:
        try:
            from mcp.client.streamable_http import streamable_http_client
            from mcp.shared._httpx_utils import create_mcp_http_client
        except ImportError:
            from mcp.client.streamable_http import streamablehttp_client

            return streamablehttp_client(url, headers=headers)
        client = create_mcp_http_client(headers=headers)

        async def note(response: Any) -> None:
            if response.status_code in CHECKED:
                self._refused = response.status_code

        client.event_hooks["response"].append(note)
        return streamable_http_client(url, http_client=client)

    def _raise_if_refused(self) -> None:
        if self._refused:
            status, self._refused = self._refused, 0
            raise McpAuthError(
                f"{self._label} answered {status}: it did not accept the credentials sent "
                f"(an Authorization bearer token and any headers given). Check the token "
                f"is current and that the server expects this scheme.") from None

    def _clean(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[redacted]")
        return text

    async def __aenter__(self) -> McpTools:
        from mcp import ClientSession

        opened = False
        try:
            streams = await self._connect(self._stack, self)
            self._session = await self._stack.enter_async_context(
                ClientSession(streams[0], streams[1]))
            await self._session.initialize()
            opened = True
        finally:
            if not opened:
                self._raise_if_refused()
                await self._stack.aclose()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    async def list_tools(self) -> list[dict[str, Any]]:
        done = False
        try:
            listed = await self._session.list_tools()
            done = True
        finally:
            if not done:
                self._raise_if_refused()
        return [{"name": t.name, "description": self._clean(t.description or ""),
                 "inputSchema": t.model_dump(by_alias=True)["inputSchema"]}
                for t in listed.tools]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutput:
        done = False
        try:
            result = (await self._session.call_tool(name, arguments)).model_dump(
                by_alias=True, exclude_none=True)
            done = True
        finally:
            if not done:
                self._raise_if_refused()
        text = self._clean("\n".join(c.get("text", "") for c in result.get("content") or []
                                     if c.get("type") == "text"))
        structured = result.get("structuredContent")
        if structured is not None:
            structured = json.loads(self._clean(json.dumps(structured)))
        return ToolOutput(text or (_text_of(structured) if structured is not None else ""),
                          structured=structured, is_error=bool(result.get("isError")))
