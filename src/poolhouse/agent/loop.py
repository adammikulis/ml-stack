"""An Agents SDK runner with local tools, interventions, and streamed events."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from poolhouse.agent.auto import AutoCompact
from poolhouse.agent.compact import Compaction, CompactResult
from poolhouse.agent.events import (
    ConfirmRequest,
    Event,
    Text,
    Thinking,
)
from poolhouse.agent.schema import from_mcp, parse_arguments, validate
from poolhouse.agent.sources import ToolOutput, ToolSource
from poolhouse.agent.watched import Watch, resolve
from poolhouse.client import thinking
from poolhouse.client.tokens import estimate_tokens
from poolhouse.guard import Unguarded, default, native
from poolhouse.http import ServerError
from poolhouse.interventions import (
    Asker,
    Call,
    Confirm,
    Run,
)
from poolhouse.taint import TaintRail

__all__ = ["Agent", "Budget", "Cancelled"]


class Cancelled(RuntimeError):
    """Raised inside a streaming read when the consumer has stopped listening."""


@dataclass(frozen=True, slots=True)
class Budget:
    """Optional run limits, tool-result formatting, and tool concurrency."""

    max_steps: int | None = None
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_repairs: int | None = None
    parallel: int = 4
    max_result_chars: int | None = None
    profile: str = "full"
    summarise: Summarise | None = None


class Chats(Protocol):
    """The part of `poolhouse.client.Client` an agent uses."""

    def chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = ...,
             on_delta: Callable[[str, str], None] | None = ..., **extra: Any) -> Any: ...


Summarise = Callable[[str, ToolOutput], str]
"""``summarise(tool name, output)`` -> the text the model is shown for that result."""


@dataclass(slots=True)
class _Pending:
    id: str
    name: str
    args: dict[str, Any] | None
    errors: list[str]
    denied: str = ""


class Agent:
    """A model, a tool source, and the budget a run is held to.

    ``auto_compact`` makes a run compact its conversation before a request would overflow
    the context, and once more if the server refuses a request for being too long.
    ``interventions`` (`poolhouse.interventions`) are asked before the run, before each model
    call, before each tool call and after each tool result. With none given they are the built-in
    rails of `poolhouse.guard` and, when a local model can be leased for it, the model tier. A
    list replaces them, so ``[*guard.default(), mine]`` keeps them; running without any says so
    with ``interventions=guard.off(because=...)``, which is logged, and any other empty list is
    refused. ``confirm`` answers a `Confirm`, and without it a `Confirm` is a refusal.
    Sentinel screens calls and results; ``sentinel=unwatched(because=...)`` or ``guard.off`` opts out.
    """

    def __init__(self, client: Chats, tools: ToolSource, *, budget: Budget | None = None,  # noqa: PLR0913
                 auto_compact: Compaction | None = None,
                 interventions: Sequence[Any] | None = None, sentinel: Any = None) -> None:
        if interventions is not None and not interventions \
                and not isinstance(interventions, Unguarded):
            raise ValueError("an empty interventions list turns the guard off silently; "
                             "use interventions=guard.off(because=...)")
        self._screen: list[Any] | None = None
        self.watch: Watch | None = resolve(interventions, sentinel)
        self.client = client
        self.tools = tools
        self.budget = budget or Budget()
        self.auto = AutoCompact(client, auto_compact) if auto_compact else None
        self.interventions = None if interventions is None else list(interventions)
        self.confirm: Asker | None = None
        self._asked: asyncio.Queue[Event] = asyncio.Queue()
        if hasattr(tools, "on_elicit"):
            tools.on_elicit = self._elicited

    def _items(self, listed: list[dict[str, Any]]) -> list[Any]:
        """The interventions of one run: those given, or fresh built-in rails and the model tier,
        with the taint rails given the sinks the tools' annotations in ``listed`` describe."""
        if self.interventions is not None:
            items = list(self.interventions)
        else:
            if self._screen is None:
                self._screen = native.screen()
            items = default(screen=self._screen)
        for one in items:
            if isinstance(one, TaintRail):
                one.learn(listed)
        return items

    def close(self) -> None:
        """Release the model tier's lease, when this agent made one."""
        for one in self._screen or ():
            close = getattr(one, "close", None)
            if close is not None:
                close()

    async def compact_now(self, messages: list[dict[str, Any]]) -> CompactResult:
        """Compact ``messages`` in place now, whatever the context holds."""
        schemas = from_mcp(await self.tools.list_tools(), self.budget.profile)
        auto = self.auto or AutoCompact(self.client, Compaction())
        return await auto.compact(messages, schemas)

    async def run(self, task: str | list[dict[str, Any]]) -> AsyncIterator[Event]:
        """Events for one task; a message list is continued in place. Stopping the
        iteration early, or cancelling the task driving it, stops the model's stream."""
        try:
            async with contextlib.aclosing(self._run(task)) as events:
                async for event in events:
                    yield event
        finally:
            if self.watch:
                self.watch.leave()

    async def _run(self, task: str | list[dict[str, Any]]) -> AsyncIterator[Event]:
        from poolhouse.agent.sdk_runtime import execute
        async with contextlib.aclosing(execute(self, task)) as events:
            async for event in events:
                yield event

    async def _ask(self, messages: list[dict[str, Any]], schemas: list[dict[str, Any]]
                   ) -> AsyncIterator[Text | Thinking | Any]:
        """One model turn: its deltas as they arrive, then its `Reply`."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        stop = threading.Event()

        def on_delta(channel: str, piece: str) -> None:
            if stop.is_set():
                raise Cancelled("the consumer stopped")
            loop.call_soon_threadsafe(queue.put_nowait, (channel, piece))

        def work() -> Any:
            try:
                return self.client.chat(list(messages), tools=schemas or None,
                                        think=thinking.resolve(thinking.AGENT),
                                        on_delta=on_delta)
            except Cancelled:
                return None
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = asyncio.ensure_future(asyncio.to_thread(work))
        try:
            while (item := await queue.get()) is not None:
                yield (Thinking if item[0] == "thinking" else Text)(item[1])
            yield await task
        finally:
            stop.set()

    def _notify(self, ask: Confirm, call: Call | None) -> None:
        self._asked.put_nowait(ConfirmRequest(call.id if call else "", call.name if call else "",
                                              ask.question, dict(ask.details)))

    async def _elicited(self, message: str, details: dict[str, Any]) -> Any:
        """A server's question to the person mid-call: put to ``confirm`` as a `ConfirmRequest`,
        declined when there is no handler."""
        self._asked.put_nowait(ConfirmRequest("", "elicitation", message, details))
        if self.confirm is None:
            return None
        answer = self.confirm(Confirm(message, details), Call("elicitation"))
        return await answer if inspect.isawaitable(answer) else answer

    async def _watching(self, work: Awaitable[Any]) -> AsyncIterator[Any]:
        """Run ``work``, yielding the questions asked meanwhile as events, then its result."""
        task = asyncio.ensure_future(work)
        try:
            while not task.done():
                got = asyncio.ensure_future(self._asked.get())
                await asyncio.wait({task, got}, return_when=asyncio.FIRST_COMPLETED)
                if got.done():
                    yield got.result()
                else:
                    got.cancel()
            while not self._asked.empty():
                yield self._asked.get_nowait()
            yield task.result()
        finally:
            task.cancel()

    def _pending(self, reply: Any, index: dict[str, dict[str, Any]]) -> list[_Pending]:
        out = []
        for n, call in enumerate(reply.tool_calls or _calls_in_text(reply.content, index), 1):
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            args, why = parse_arguments(fn.get("arguments"))
            errors = [why] if why else []
            if name not in index:
                errors = [f"unknown tool {name!r}; the tools are {sorted(index)}"]
            elif args is not None:
                errors = validate(args, index[name])
            out.append(_Pending(str(call.get("id") or f"call_{n}"), name, args, errors))
        return out

    async def _dispatch(self, pending: list[_Pending]) -> list[ToolOutput]:
        gate = asyncio.Semaphore(max(1, self.budget.parallel))

        async def one(call: _Pending) -> ToolOutput:
            if call.denied:
                return ToolOutput(json.dumps({"ok": False, "tool": call.name,
                                              "denied": call.denied}), is_error=True)
            if call.errors:
                return ToolOutput(json.dumps({"ok": False, "tool": call.name,
                                              "errors": call.errors,
                                              "hint": "call the tool again with arguments "
                                                      "that match its schema"}), is_error=True)
            async with gate:
                with self.watch.running() if self.watch else contextlib.nullcontext():
                    done = await self.tools.call(call.name, call.args or {})
            return ToolOutput(self._shown(call.name, done), done.structured, done.is_error)

        return list(await asyncio.gather(*(one(c) for c in pending)))

    def _shown(self, name: str, output: ToolOutput) -> str:
        text = self.budget.summarise(name, output) if self.budget.summarise else output.text
        limit = self.budget.max_result_chars
        if limit is not None and len(text) > limit:
            return f"{text[:limit]}... [{len(text) - limit} more characters cut]"
        return text


def _task_of(messages: Sequence[Mapping[str, Any]]) -> str:
    """The first user message, as text."""
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _inject(run: Run, messages: list[dict[str, Any]]) -> None:
    """Put the guidance the interventions left into ``messages`` as part of the next user turn."""
    if not run.guides:
        return
    text = "[Guidance]\n" + "\n".join(run.guides)
    run.guides = []
    if messages and messages[-1].get("role") == "user":
        last = messages[-1]
        messages[-1] = {**last, "content": f"{last.get('content') or ''}\n\n{text}"}
    else:
        messages.append({"role": "user", "content": text})


def _overflowed(exc: ServerError) -> bool:
    """Whether the server refused a request for being longer than its context."""
    said = f"{exc} {exc.body}".lower()
    return exc.status == 400 and ("exceed" in said or "context size" in said
                                  or "n_ctx" in said)


def _completion_tokens(reply: Any) -> int:
    usage = (getattr(reply, "raw", None) or {}).get("usage") or {}
    return int(usage.get("completion_tokens") or estimate_tokens(reply.content or ""))


def _calls_in_text(content: str | None, index: dict[str, dict[str, Any]]
                   ) -> list[dict[str, Any]]:
    """A call a model wrote as JSON in its text, ``{"name": ..., "arguments": {...}}``."""
    args, _ = parse_arguments(content)
    if not args or args.get("name") not in index:
        return []
    given = args.get("arguments", args.get("parameters", {}))
    return [{"function": {"name": args["name"], "arguments": given}}]
