"""Where an agent's tools come from: Python callables, or an MCP server over stdio or HTTP.

A source lists its tools as MCP-shaped dicts (``name``, ``description``, ``inputSchema``)
and calls one by name. `McpTools` needs the ``mcp`` extra.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ml_stack import sentinel
from ml_stack.agent.confined import Options, confine

__all__ = ["FunctionTools", "McpAuthError", "McpBlocked", "McpTools", "ToolOutput", "ToolSource"]


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

Elicit = Callable[[str, dict[str, Any]], Awaitable[Any]]
"""``elicit(message, details)`` -> the form values to send back (a dict), True to accept with
no values, or a false value to decline."""

Progress = Callable[[float, float | None, str | None], Awaitable[None]]


class McpBlocked(RuntimeError):
    """Sentinel holds this MCP server (quarantined, or not allowed), so it is not connected."""


class McpTools:
    """An MCP server's tools through the ``mcp`` 2.x client, used as an async context
    manager: ``async with McpTools.stdio("python", ["-m", "server"]) as tools``.

    The client negotiates the protocol version itself, so a server on the 2026-07-28 revision
    (stateless requests) and an older one are both reached; nothing here assumes a session.
    Credentials given to `http` are sent on every request, a callable ``bearer`` is read again
    each time the connection is made, and the token is cut out of everything the server sends
    back and out of this object's repr. A server that asks the user a question while a tool
    runs is answered by ``on_elicit``, or declined when there is none.
    """

    def __init__(self, transport: Callable[[McpTools], Any], label: str = "mcp server") -> None:
        self._transport = transport
        self._label = label
        self._client: Any = None
        self._secrets: list[str] = []
        self._cleanups: list[Callable[[], None]] = []
        self._refused = 0
        self.on_elicit: Elicit | None = None
        self.on_progress: Progress | None = None

    def __repr__(self) -> str:
        return f"McpTools({self._label})"

    @classmethod
    def stdio(cls, command: str, args: Sequence[str] = (),  # noqa: PLR0913
              env: Mapping[str, str] | None = None, *, policy: Any = None,
              project: str | None = None, reads: Sequence[str] = (),
              unsandboxed: Any = None) -> McpTools:
        """A server spawned as ``command args`` under the sandbox and spoken to over its stdin
        and stdout. ``policy`` (a `sandbox.Policy`) replaces the default ``mcp_server`` policy:
        the interpreter, ``project`` and ``reads`` readable, a scratch directory writable,
        loopback only, and ``env`` as the whole environment. With no sandbox available the server
        is not started unless ``unsandboxed`` is a `sandbox.AllowUnsandboxed`."""
        def transport(owner: McpTools) -> Any:
            from mcp import StdioServerParameters

            launch = confine(command, args, env, Options(policy, project, tuple(reads),
                                                         unsandboxed))
            owner._cleanups.append(launch.cleanup)
            return StdioServerParameters(command=launch.command, args=launch.args,
                                         env=launch.env)

        return cls(transport, command)

    @classmethod
    def http(cls, url: str, *, headers: Mapping[str, str] | None = None,
             bearer: str | Callable[[], str] | None = None) -> McpTools:
        """A server reached at ``url`` over streamable HTTP. ``headers`` go on every request;
        ``bearer`` (a token, or a function returning the current one) is sent as
        ``Authorization: Bearer ...``."""
        def transport(owner: McpTools) -> Any:
            from mcp.client.streamable_http import streamable_http_client
            from mcp.shared._httpx_utils import create_mcp_http_client

            sent = dict(headers or {})
            token = bearer() if callable(bearer) else bearer
            if token:
                sent["Authorization"] = f"Bearer {token}"
            owner._secrets = [v for k, v in sent.items() if k.lower() in SENSITIVE]
            owner._secrets += [token] if token else []
            client = create_mcp_http_client(headers=sent)

            async def note(response: Any) -> None:
                if response.status_code in CHECKED:
                    owner._refused = response.status_code

            client.event_hooks["response"].append(note)
            return streamable_http_client(url, http_client=client)

        return cls(transport, url.split("?")[0])

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

    async def _elicited(self, context: Any, params: Any) -> Any:
        from mcp_types import ElicitResult

        if self.on_elicit is None:
            return ElicitResult(action="decline")
        details = params.model_dump(by_alias=True, exclude_none=True)
        answer = await self.on_elicit(self._clean(params.message), details)
        if isinstance(answer, dict):
            return ElicitResult(action="accept", content=answer)
        return ElicitResult(action="accept", content={}) if answer else ElicitResult(
            action="decline")

    async def __aenter__(self) -> McpTools:
        from mcp import Client

        if not sentinel.default().mcp_allowed(self._label):
            raise McpBlocked(f"{self._label} is held by sentinel and is not connected; a person "
                             "releases it with `ml-stack-security release`")
        opened = False
        try:
            self._client = Client(self._transport(self), elicitation_callback=self._elicited)
            await self._client.__aenter__()
            opened = True
        finally:
            if not opened:
                while self._cleanups:
                    self._cleanups.pop()()
                self._raise_if_refused()
        return self

    async def __aexit__(self, *exc: object) -> None:
        try:
            await self._client.__aexit__(*exc)
        finally:
            while self._cleanups:
                self._cleanups.pop()()

    async def list_tools(self) -> list[dict[str, Any]]:
        done = False
        try:
            listed = await self._client.list_tools()
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
            result = (await self._client.call_tool(
                name, arguments, progress_callback=self.on_progress)).model_dump(
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
