"""Compacting a conversation before a request would overflow the model's context."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Callable
from typing import Any

from ml_stack.agent.compact import Compaction, CompactResult, compact, has_open_calls
from ml_stack.agent.context import Counter, context_limit, context_usage
from ml_stack.agent.events import Compacted, Context
from ml_stack.agent.summarise import model_summarizer
from ml_stack.agent.transcript import Transcript

__all__ = ["AutoCompact", "Compacting"]


class AutoCompact:
    """A `Compaction` bound to the client whose context it watches."""

    def __init__(self, client: Any, config: Compaction) -> None:
        self.client = client
        spill = config.spill
        if spill.transcript is None:
            spill = dataclasses.replace(spill, transcript=Transcript())
        self.config = dataclasses.replace(
            config, spill=spill,
            summarizer=config.summarizer or (model_summarizer(client) if config.summarize
                                             else None))
        self.counter = Counter(client)
        self._limit = config.context_size

    @property
    def transcript(self) -> Transcript | None:
        return self.config.spill.transcript

    async def limit(self) -> int | None:
        """The context the model has: the configured size, else what the server reports."""
        if self._limit is None:
            self._limit = await asyncio.to_thread(context_limit, self.client)
        return self._limit

    async def before(self, messages: list[dict[str, Any]], schemas: list[dict[str, Any]],
                     *, force: bool = False) -> list[Any]:
        """Events for the context as it stands, after compacting in place when it is past
        the threshold (or ``force``) and no tool call is waiting for its result."""
        limit = await self.limit()
        if not limit:
            return []
        usage = await asyncio.to_thread(context_usage, messages, schemas, limit,
                                        count=self.counter)
        events: list[Any] = [Context(usage.used, limit, usage.fraction)]
        if not (force or usage.fraction >= self.config.threshold) or has_open_calls(messages):
            return events
        result = await self.compact(messages, schemas)
        if result.strategy_used != "none":
            tools = self.counter(json.dumps(schemas, sort_keys=True)) if schemas else 0
            events.append(Compacted(
                usage.fraction, round((result.tokens_after + tools) / limit, 4),
                result.dropped_count, result.strategy_used,
                result.tokens_before + tools, result.tokens_after + tools,
                result.notes))
        return events

    async def compact(self, messages: list[dict[str, Any]],
                      schemas: list[dict[str, Any]]) -> CompactResult:
        """Compact ``messages`` in place down to the target share of the context."""
        limit = await self.limit()
        if not limit:
            raise ValueError("the context size is unknown; set Compaction(context_size=...)")
        tools = self.counter(json.dumps(schemas, sort_keys=True)) if schemas else 0
        budget = int(limit * self.config.target) - tools
        result = await asyncio.to_thread(
            compact, messages, budget=max(1, budget), using=self.config, count=self.counter)
        if result.strategy_used != "none":
            messages[:] = result.messages
        return result


class Compacting:
    """A client that compacts the ``messages`` it is handed, in place, before each ``chat``
    that would fill the context; every other attribute is the wrapped client's."""

    def __init__(self, client: Any, config: Compaction,
                 on_event: Callable[[Any], None] | None = None) -> None:
        self.client = client
        self.auto = AutoCompact(client, config)
        self.on_event = on_event

    def chat(self, messages: list[dict[str, Any]], **extra: Any) -> Any:
        """The wrapped client's ``chat``, after compacting ``messages`` if they are past the
        threshold."""
        for event in asyncio.run(self.auto.before(messages, list(extra.get("tools") or []))):
            if self.on_event is not None:
                self.on_event(event)
        return self.client.chat(messages, **extra)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)
