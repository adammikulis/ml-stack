"""A bounded tool-calling loop over any `ToolSource`, streamed as events.

`Agent.run` asks the model, runs the tool calls it makes (in parallel), feeds the answers
back, and repeats until the model answers in text or a budget runs out. Each piece of the
turn arrives as an event from an async iterator: text deltas, tool calls, tool results,
rejected calls, and a final `Done`.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ml_stack.agent.auto import AutoCompact
from ml_stack.agent.compact import Compaction, CompactResult
from ml_stack.agent.events import (
    ConfirmRequest,
    Denied,
    Done,
    Event,
    Repair,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
)
from ml_stack.agent.schema import from_mcp, index_by_name, parse_arguments, validate
from ml_stack.agent.sources import ToolOutput, ToolSource
from ml_stack.agent.watched import Watch, resolve
from ml_stack.client import thinking
from ml_stack.client.tokens import estimate_tokens
from ml_stack.guard import Unguarded, default, native, start
from ml_stack.http import ServerError
from ml_stack.interventions import (
    Asker,
    Call,
    Confirm,
    Gate,
    Run,
    Screened,
)
from ml_stack.taint import TaintRail

__all__ = ["Agent", "Budget", "Cancelled"]


class Cancelled(RuntimeError):
    """Raised inside a streaming read when the consumer has stopped listening."""


@dataclass(frozen=True, slots=True)
class Budget:
    """What stops a run: steps (model turns), tool calls, completion tokens, and how many
    consecutive turns of nothing but rejected calls are tolerated. ``profile`` is how far the
    tool schemas are trimmed (`ml_stack.agent.schema.PROFILES`); ``summarise`` rewrites a
    tool result before the model sees it."""

    max_steps: int = 8
    max_tool_calls: int = 64
    max_tokens: int | None = None
    max_repairs: int = 2
    parallel: int = 4
    max_result_chars: int = 4000
    profile: str = "full"
    summarise: Summarise | None = None


class Chats(Protocol):
    """The part of `ml_stack.client.Client` an agent uses."""

    def chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = ...,
             on_delta: Callable[[str, str], None] | None = ..., **extra: Any) -> Any: ...


Summarise = Callable[[str, ToolOutput], str]
"""``summarise(tool name, output)`` -> the text the model is shown for that result."""


@dataclass(slots=True)
class _Refusal:
    reason: str = ""


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
    ``interventions`` (`ml_stack.interventions`) are asked before the run, before each model
    call, before each tool call and after each tool result. With none given they are the built-in
    rails of `ml_stack.guard` and, when a local model can be leased for it, the model tier. A
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
        messages = [{"role": "user", "content": task}] if isinstance(task, str) else task
        schemas = from_mcp(listed := await self.tools.list_tools(), self.budget.profile)
        index = index_by_name(schemas)
        spent = calls = rejected_turns = 0
        run = start(self._items(listed), offered=schemas, task=_task_of(messages),
                    confirm=self.confirm, notify=self._notify)
        run.context.messages = messages
        refused = _Refusal()
        async for event in self._decide(run, "before_invocation", refused):
            yield event
        for step in range(1, self.budget.max_steps + 1):
            reply = None
            run.context.step, run.context.tool_calls = step, calls
            if not refused.reason:
                async for event in self._decide(run, "before_model_call", refused):
                    yield event
            if refused.reason:
                yield Done("denied", refused.reason, step - 1, calls, spent, messages)
                return
            _inject(run, messages)
            if self.auto:
                for event in await self.auto.before(messages, schemas):
                    yield event
            for attempt in (0, 1):
                try:
                    async for piece in self._ask(messages, schemas):
                        if isinstance(piece, (Text, Thinking)):
                            yield piece
                        else:
                            reply = piece
                    break
                except ServerError as exc:
                    if attempt or not self.auto or not _overflowed(exc):
                        raise
                    for event in await self.auto.before(messages, schemas, force=True):
                        yield event
            spent += _completion_tokens(reply)
            pending = self._pending(reply, index)
            text = reply.content or ""
            text = self.watch.said(text) if self.watch else text
            if not pending:
                messages.append({"role": "assistant", "content": text})
                yield Done("answer", text, step, calls, spent, messages)
                return
            if calls + len(pending) > self.budget.max_tool_calls:
                yield Done("max_tool_calls", text, step, calls, spent, messages)
                return
            calls += len(pending)
            messages.append(_assistant(text, pending))
            run.context.step, run.context.tool_calls = step, calls
            async for event in self._vet(run, pending):
                yield event
            for one in pending:
                yield (Repair(one.id, one.name, one.errors) if one.errors
                       else Denied(one.id, one.name, one.denied) if one.denied
                       else ToolCall(one.id, one.name, one.args or {}))
            answers: list[ToolOutput] = []
            async for event in self._dispatching(self._dispatch(pending), answers):
                yield event
            async for event in self._results(run, pending, answers, messages):
                yield event
            rejected_turns = rejected_turns + 1 if all(p.errors for p in pending) else 0
            if rejected_turns > self.budget.max_repairs:
                yield Done("repairs_exhausted", text, step, calls, spent, messages)
                return
            if self.budget.max_tokens is not None and spent >= self.budget.max_tokens:
                yield Done("max_tokens", text, step, calls, spent, messages)
                return
        yield Done("max_steps", "", self.budget.max_steps, calls, spent, messages)

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

    async def _dispatching(self, work: Awaitable[list[ToolOutput]], out: list[ToolOutput]
                           ) -> AsyncIterator[Event]:
        """Run the calls, yielding the questions their servers ask meanwhile; the answers are
        put in ``out``."""
        async for item in self._watching(work):
            if isinstance(item, list):
                out.extend(item)
            else:
                yield item

    async def _decide(self, run: Run, hook: str, out: _Refusal) -> AsyncIterator[Event]:
        """Ask ``hook`` of the run's interventions, yielding the questions put to the person;
        a refusal is left in ``out``."""
        if self.watch:
            self.watch.enter()
        async for item in self._watching(run.decide(hook, run.context)):
            if isinstance(item, Gate):
                out.reason = "" if item.allowed else getattr(item.verdict, "reason", "")
                if not out.reason and self.watch:
                    out.reason = self.watch.frozen()
            else:
                yield item

    async def _results(self, run: Run, pending: list[_Pending], answers: list[ToolOutput],
                       messages: list[dict[str, Any]]) -> AsyncIterator[Event]:
        """Pass each answer through the interventions, then add it to ``messages``."""
        if self.watch:
            self.watch.enter()
        for one, answer in zip(pending, answers, strict=True):
            text = answer.text
            if not (one.errors or one.denied):
                async for item in self._watching(
                        run.after_tool(Call(one.name, one.args, one.id), text)):
                    if isinstance(item, Screened):
                        text = self.watch.shown(one.name, text, item) if self.watch else item.text
                    else:
                        yield item
            messages.append({"role": "tool", "tool_call_id": one.id, "name": one.name,
                             "content": text})
            yield ToolResult(one.id, one.name, text, answer.is_error)

    async def _vet(self, run: Run, pending: list[_Pending]) -> AsyncIterator[Event]:
        if self.watch:
            self.watch.enter()
        for one in pending:
            if one.errors:
                continue
            if self.watch and (why := self.watch.refuses(one.name, one.args)):
                one.denied = f"Denied: {why}"
                continue
            async for item in self._watching(run.before_tool(Call(one.name, one.args, one.id))):
                if isinstance(item, Gate):
                    one.denied = "" if item.allowed else getattr(item.verdict, "reason", "")
                    if one.denied and self.watch:
                        self.watch.denied(one.name, one.args, getattr(item.verdict, "by", ""),
                                          one.denied, logged=item.confirmed is None)
                else:
                    yield item

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
        if len(text) > limit:
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


def _assistant(text: str, pending: list[_Pending]) -> dict[str, Any]:
    return {"role": "assistant", "content": text, "tool_calls": [
        {"id": p.id, "type": "function", "function": {
            "name": p.name,
            "arguments": json.dumps(p.args if p.args is not None else {})}}
        for p in pending]}


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
