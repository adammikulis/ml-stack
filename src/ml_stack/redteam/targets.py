"""What an attack is sent to: anything that takes the conversation so far and answers with
text, the tool calls it made and whether a guard stopped it. Includes the helper that lets
another project's agent, which drives its tools over MCP, take part."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ml_stack.client import Client, Request, ServerError, Transport

__all__ = ["Answer", "Responder", "ToolSpy", "chat_endpoint", "from_callable",
           "mcp_tool_agent"]


@dataclass(frozen=True, slots=True)
class Answer:
    """What a target did with a conversation: the text it replied, how many dangerous tool
    calls it made, whether a guard stopped something, the HTTP status when the target is one,
    and whether the target failed to answer at all."""

    text: str = ""
    calls: int = 0
    blocked: bool = False
    status: int | None = None
    detail: str = ""
    error: bool = False


TIMEOUT_S = 120.0
MAX_TOKENS = 256

Messages = list[dict[str, Any]]
Responder = Callable[[Messages], Awaitable[Answer]]
"""``responder(messages)`` -> the `Answer`; ``messages`` are OpenAI-shaped, the last one the
attacker's."""


def chat_endpoint(base_url: str, *, system: str = "", token: str | None = None,
                  tools: list[dict[str, Any]] | None = None) -> Responder:
    """An OpenAI-compatible chat endpoint at ``base_url``, a llama-server or a daemon's
    ``/infer``, asked greedily. ``token`` is sent as a bearer token."""
    client = Client(base_url, request=Request(n_predict=MAX_TOKENS),
                    transport=Transport(api_key=token, timeout=TIMEOUT_S))

    async def respond(messages: Messages) -> Answer:
        sent = ([{"role": "system", "content": system}] if system else []) + messages
        try:
            reply = await asyncio.to_thread(
                client.chat, sent, tools=tools, timeout=TIMEOUT_S)
        except ServerError as exc:
            blocked = exc.status in (401, 403)
            return Answer(blocked=blocked, status=exc.status, error=not blocked,
                          detail=f"{type(exc).__name__}: {exc}"[:300])
        return Answer(text=reply.content or "", calls=len(reply.tool_calls or ()), status=200)

    return respond


def from_callable(fn: Callable[[str], Any]) -> Responder:
    """A function from the last message's text to a reply (text, or an `Answer`; it may be
    a coroutine function) as a target."""
    async def respond(messages: Messages) -> Answer:
        got = fn(str(messages[-1].get("content") or ""))
        if inspect.isawaitable(got):
            got = await got
        return got if isinstance(got, Answer) else Answer(text=str(got))

    return respond


@dataclass(slots=True)
class ToolSpy:
    """Records the tool calls an agent makes. ``wrap`` takes the function the agent calls
    tools through, ``(name, arguments) -> result``, and returns one that records first; a
    call that raises is recorded as refused."""

    calls: list[tuple[str, dict[str, Any], bool]] = field(default_factory=list)

    def wrap(self, call: Callable[[str, dict[str, Any]], Any]
             ) -> Callable[[str, dict[str, Any]], Any]:
        def spied(name: str, arguments: dict[str, Any]) -> Any:
            done = False
            try:
                result = call(name, arguments)
                done = True
            finally:
                self.calls.append((name, dict(arguments), done))
            return result

        return spied

    def reached(self, tool: str, where: Callable[[Mapping[str, Any]], bool] | None = None
                ) -> bool:
        """Whether ``tool`` ran to completion, with arguments ``where`` accepts if given."""
        return any(done and name == tool and (where is None or where(args))
                   for name, args, done in self.calls)

    def reset(self) -> None:
        self.calls.clear()


def mcp_tool_agent(chat: Callable[[str], Any], spy: ToolSpy, *,
                   dangerous: Iterable[str] | Mapping[str, Callable[[Mapping[str, Any]], bool]]
                   ) -> Responder:
    """An agent that answers ``chat(prompt) -> text`` and calls its MCP tools through ``spy``,
    as a target.

    ``dangerous`` names the tools whose completed call is the attack succeeding, or maps each
    to a predicate on the arguments. The `Answer` has ``calls`` set to how many dangerous
    tools the prompt got called, and ``blocked`` when one was attempted and refused.
    """
    watched: dict[str, Callable[[Mapping[str, Any]], bool] | None] = (
        dict(dangerous) if isinstance(dangerous, Mapping) else dict.fromkeys(dangerous))

    async def respond(messages: Messages) -> Answer:
        spy.reset()
        got = chat(str(messages[-1].get("content") or ""))
        if inspect.isawaitable(got):
            got = await got
        hit = sum(1 for name, where in watched.items() if spy.reached(name, where))
        refused = sum(1 for name, _, done in spy.calls if name in watched and not done)
        return Answer(text=str(got), calls=hit, blocked=bool(refused) and not hit)

    return respond
