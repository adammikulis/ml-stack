"""A bounded tool-calling loop over any `ToolSource`, streamed as events.

`Agent.run` asks the model, runs the tool calls it makes (in parallel), feeds the answers
back, and repeats until the model answers in text or a budget runs out. Each piece of the
turn arrives as an event from an async iterator: text deltas, tool calls, tool results,
rejected calls, and a final `Done`.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ml_stack.agent.auto import AutoCompact
from ml_stack.agent.compact import Compaction, CompactResult
from ml_stack.agent.events import (
    Denied,
    Done,
    Event,
    Repair,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
)
from ml_stack.agent.interventions import InterventionContext
from ml_stack.agent.schema import from_mcp, index_by_name, parse_arguments, validate
from ml_stack.agent.sources import ToolOutput, ToolSource
from ml_stack.agent.vet import Confirmer, Verdict, Vetter
from ml_stack.client.tokens import estimate_tokens
from ml_stack.http import ServerError

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
    ``interventions`` are asked before the run, before each model call and before each tool
    call (`ml_stack.agent.interventions`); ``confirm`` answers their `Confirm`, and without
    it a `Confirm` is a refusal.
    """

    def __init__(self, client: Chats, tools: ToolSource, *, budget: Budget | None = None,
                 auto_compact: Compaction | None = None,
                 interventions: Sequence[Any] = ()) -> None:
        self.client = client
        self.tools = tools
        self.budget = budget or Budget()
        self.auto = AutoCompact(client, auto_compact) if auto_compact else None
        self.vet = Vetter(interventions)

    @property
    def confirm(self) -> Confirmer | None:
        return self.vet.confirm

    @confirm.setter
    def confirm(self, handler: Confirmer | None) -> None:
        self.vet.confirm = handler

    async def compact_now(self, messages: list[dict[str, Any]]) -> CompactResult:
        """Compact ``messages`` in place now, whatever the context holds."""
        schemas = from_mcp(await self.tools.list_tools(), self.budget.profile)
        auto = self.auto or AutoCompact(self.client, Compaction())
        return await auto.compact(messages, schemas)

    async def run(self, task: str | list[dict[str, Any]]) -> AsyncIterator[Event]:
        """Events for one task; a message list is continued in place. Stopping the
        iteration early, or cancelling the task driving it, stops the model's stream."""
        messages = [{"role": "user", "content": task}] if isinstance(task, str) else task
        schemas = from_mcp(await self.tools.list_tools(), self.budget.profile)
        index = index_by_name(schemas)
        spent = calls = rejected_turns = 0
        verdict = Verdict()
        async for event in self.vet.check("before_invocation",
                                          (_context(messages, 0, 0),), verdict):
            yield event
        for step in range(1, self.budget.max_steps + 1):
            reply = None
            async for event in self.vet.check("before_model_call",
                                              (_context(messages, step, calls),), verdict):
                yield event
            if verdict.denied:
                yield Done("denied", verdict.denied, step - 1, calls, spent, messages)
                return
            self.vet.inject(messages)
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
            if not pending:
                messages.append({"role": "assistant", "content": text})
                yield Done("answer", text, step, calls, spent, messages)
                return
            if calls + len(pending) > self.budget.max_tool_calls:
                yield Done("max_tool_calls", text, step, calls, spent, messages)
                return
            calls += len(pending)
            messages.append(_assistant(text, pending))
            async for event in self._vet(pending, messages, step, calls):
                yield event
            for one in pending:
                yield (Repair(one.id, one.name, one.errors) if one.errors
                       else Denied(one.id, one.name, one.denied) if one.denied
                       else ToolCall(one.id, one.name, one.args or {}))
            answers = await self._dispatch(pending)
            for one, answer in zip(pending, answers, strict=True):
                messages.append({"role": "tool", "tool_call_id": one.id, "name": one.name,
                                 "content": answer.text})
                yield ToolResult(one.id, one.name, answer.text, answer.is_error)
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

    async def _vet(self, pending: list[_Pending], messages: list[dict[str, Any]], step: int,
                   calls: int) -> AsyncIterator[Event]:
        for one in pending:
            if one.errors:
                continue
            verdict = Verdict()
            call = {"id": one.id, "name": one.name, "arguments": one.args}
            async for event in self.vet.check("before_tool_call",
                                              (call, _context(messages, step, calls)),
                                              verdict, ident=one.id, name=one.name):
                yield event
            one.denied = verdict.denied

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
                done = await self.tools.call(call.name, call.args or {})
            return ToolOutput(self._shown(call.name, done), done.structured, done.is_error)

        return list(await asyncio.gather(*(one(c) for c in pending)))

    def _shown(self, name: str, output: ToolOutput) -> str:
        text = self.budget.summarise(name, output) if self.budget.summarise else output.text
        limit = self.budget.max_result_chars
        if len(text) > limit:
            return f"{text[:limit]}... [{len(text) - limit} more characters cut]"
        return text


def _context(messages: Sequence[Mapping[str, Any]], step: int, calls: int
             ) -> InterventionContext:
    return InterventionContext(tuple(messages), step, calls)


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
