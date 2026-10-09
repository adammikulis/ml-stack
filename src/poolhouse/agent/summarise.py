"""The summarizer that asks the model being served to write the summary message."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from poolhouse import sentinel
from poolhouse.agent.compact import Summarizer
from poolhouse.agent.watched import current_session
from poolhouse.sentinel.store import fingerprint

__all__ = ["PROMPT", "model_summarizer", "render"]

PROMPT = """You keep the memory of an assistant whose conversation no longer fits its context. \
Write the summary the assistant will read in place of the conversation, using exactly these \
sections:

Goals: what the user is trying to get done.
Decisions: what was decided or ruled out, and why.
Facts: names, numbers, identifiers, paths and values, copied character for character.
Open tasks: what is still to do, in order.
State: files, ids and tool results the next steps depend on.

Copy every identifier exactly. Write nothing that the conversation does not say. If an earlier \
summary is given, fold the new conversation into it and keep what still matters. Stay under \
{words} words."""

RESULT_CHARS = 1500
"""Longest tool result the summarizer is shown."""


def render(messages: Sequence[Mapping[str, Any]]) -> str:
    """The messages as lines of ``role: text`` for the summarizer to read."""
    lines = []
    for m in messages:
        role, content = m.get("role", ""), str(m.get("content") or "")
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments")
            lines.append(f"assistant called {fn.get('name')}("
                         f"{args if isinstance(args, str) else json.dumps(args)})")
        if role == "tool":
            cut = content if len(content) <= RESULT_CHARS else content[:RESULT_CHARS] + " ..."
            lines.append(f"tool {m.get('name', '')} returned: {cut}")
        elif content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def model_summarizer(client: Any, *, words: int = 350) -> Summarizer:
    """A summarizer that asks ``client`` (an `poolhouse.client.Client`) to write the summary."""
    system = PROMPT.format(words=words)

    def summarise(messages: Sequence[Mapping[str, Any]], prior: str) -> str:
        said = (f"Earlier summary:\n{prior}\n\n" if prior else "") \
            + f"Conversation to fold in:\n{render(messages)}"
        reply = client.chat([{"role": "system", "content": system},
                             {"role": "user", "content": said}])
        return _vetted(str(reply.content or ""))

    return summarise


def _vetted(text: str) -> str:
    """The summary as it may be stored and fed back: sentinel's memory screen holds one that
    repeats held content or carries a decoy value, and nothing is returned for it, so the
    compaction falls back to its other strategies instead of reusing it."""
    if not text.strip():
        return text
    session = current_session()
    got = sentinel.default().screen_memory(f"summary:{fingerprint(text)[:16]}", text,
                                           session=session)
    return "" if got.withheld else text
