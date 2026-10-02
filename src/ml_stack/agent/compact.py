"""Fitting a conversation into a token budget, cheapest step first.

`compact` works in stages: long tool results are cut to their head and tail, tool calls a
later identical call has replaced are dropped, the oldest messages are summarised into one
message, and what is still over is dropped outright. The system messages and the last
``keep_last`` messages are never touched, and an assistant message with tool calls always
travels with its results.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ml_stack.agent.context import Counter, message_text, message_tokens
from ml_stack.agent.transcript import Transcript
from ml_stack.http import ServerError

__all__ = [
    "ELIDED",
    "STAGES",
    "SUMMARY_PREFIX",
    "CompactResult",
    "Compaction",
    "Spill",
    "Summarizer",
    "compact",
    "has_open_calls",
]

SUMMARY_PREFIX = "[Summary of the earlier conversation]\n"
ELIDED = "characters elided"
STAGES = ("elide", "prune", "summarise", "truncate")
CHUNK_TOKENS = 3000
"""Most tokens of conversation handed to the summarizer in one call."""

Message = dict[str, Any]
Summarizer = Callable[[Sequence[Mapping[str, Any]], str], str]
"""``summarizer(messages, prior summary)`` -> the new summary text."""


@dataclass(frozen=True, slots=True)
class Spill:
    """Where cut tool results go, and how long a result may be before it is cut."""

    transcript: Transcript | None = None
    above: int = 400


@dataclass(frozen=True, slots=True)
class Compaction:
    """When an agent compacts, and to what: at ``threshold`` of the context, down to
    ``target`` of it, keeping the last ``keep_last`` messages. ``summarize`` False skips
    the model; ``context_size`` None asks the server. ``preserve`` names messages never cut,
    as substrings of their text or a predicate."""

    threshold: float = 0.80
    target: float = 0.50
    keep_last: int = 6
    summarize: bool = True
    context_size: int | None = None
    summarizer: Summarizer | None = None
    spill: Spill = field(default_factory=Spill)
    preserve: Sequence[str] | Callable[[Mapping[str, Any]], bool] = ()


@dataclass(frozen=True, slots=True)
class CompactResult:
    """The messages after compaction and what it took to get there."""

    messages: list[Message]
    summary_message: Message | None
    dropped_count: int
    tokens_before: int
    tokens_after: int
    strategy_used: str
    notes: tuple[str, ...] = ()


def has_open_calls(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the last assistant tool calls are still waiting for results."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            ids = {c.get("id") for c in messages[i].get("tool_calls") or []}
            answered = {m.get("tool_call_id") for m in messages[i + 1:] if m.get("role") == "tool"}
            return bool(ids - answered)
    return False


def _units(body: Sequence[Message]) -> list[list[Message]]:
    """``body`` as units that must stay whole: a tool-calling message with its results."""
    units: list[list[Message]] = []
    for m in body:
        if m.get("role") == "tool" and units and units[-1][0].get("tool_calls"):
            units[-1].append(m)
        else:
            units.append([m])
    return units


def _flat(units: Sequence[Sequence[Message]]) -> list[Message]:
    return [m for unit in units for m in unit]


def _tail_start(units: Sequence[Sequence[Message]], keep_last: int) -> int:
    """The index of the first unit kept verbatim: whole units covering ``keep_last`` messages."""
    kept = 0
    for i in range(len(units) - 1, -1, -1):
        if kept >= keep_last:
            return i + 1
        kept += len(units[i])
    return 0


def _keeps(unit: Sequence[Message], preserve: Callable[[Mapping[str, Any]], bool]) -> bool:
    return any(preserve(m) for m in unit)


def _is_summary(unit: Sequence[Message]) -> bool:
    return str(unit[0].get("content") or "").startswith(SUMMARY_PREFIX)


def _preserver(preserve: Sequence[str] | Callable[[Mapping[str, Any]], bool],
               messages: Sequence[Message]) -> Callable[[Mapping[str, Any]], bool]:
    """A test for messages never to cut: the last user message, and those ``preserve`` names
    (substrings of the text, or a predicate)."""
    last_user = next((id(m) for m in reversed(messages) if m.get("role") == "user"), None)
    named = [preserve] if isinstance(preserve, str) else preserve
    if callable(named):
        return lambda m: id(m) == last_user or bool(named(m))
    return lambda m: id(m) == last_user or any(s in message_text(m) for s in named)


def _signature(unit: Sequence[Message]) -> str | None:
    calls = unit[0].get("tool_calls")
    if not calls:
        return None
    return json.dumps([(c["function"]["name"], _args(c["function"].get("arguments")))
                       for c in calls], sort_keys=True, default=str)


def _args(raw: Any) -> Any:
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return raw


def _elide_one(message: Message, ref: str, above: int) -> Message:
    text = str(message.get("content") or "")
    room = above * 3
    cut = f"\n[... {len(text) - room} {ELIDED}; full text: {ref} ...]\n"
    return {**message, "content": text[:room * 2 // 3] + cut + text[len(text) - room // 3:]}


def _rolling(summarizer: Summarizer, messages: Sequence[Message], prior: str,
             count: Callable[[str], int]) -> str:
    """The summary of ``messages`` after the ``prior`` one, taken ``CHUNK_TOKENS`` at a time."""
    text, chunk, size = prior, [], 0
    for m in messages:
        cost = message_tokens(m, count)
        if chunk and size + cost > CHUNK_TOKENS:
            text, chunk, size = summarizer(chunk, text), [], 0
        chunk.append(m)
        size += cost
    return summarizer(chunk, text) if chunk else text


def _clip(text: str, count: Callable[[str], int], limit: int) -> str:
    while len(text) > 40 and count(text) > limit:
        text = text[:int(len(text) * 0.9)]
    return text


class _State:
    """The conversation as ``head`` plus ``units``, the last ``tail`` of which are kept."""

    def __init__(self, messages: Sequence[Message], keep_last: int,
                 preserve: Callable[[Mapping[str, Any]], bool]) -> None:
        lead = 0
        while lead < len(messages) and messages[lead].get("role") == "system":
            lead += 1
        self.head = list(messages[:lead])
        self.units = _units(list(messages[lead:]))
        self.tail = len(self.units) - _tail_start(self.units, keep_last)
        self.preserve = preserve
        self.dropped: list[Message] = []
        self.made: Message | None = None

    @property
    def old(self) -> range:
        return range(len(self.units) - self.tail)

    def messages(self) -> list[Message]:
        out = list(self.head)
        for i, unit in enumerate(self.units):
            out += unit
            if (_is_summary(unit) and i + 1 < len(self.units)
                    and self.units[i + 1][0].get("role") == "user"):
                out.append({"role": "assistant", "content": "Understood."})
        return out

    def tokens(self, count: Callable[[str], int]) -> int:
        return sum(message_tokens(m, count) for m in self.messages())

    def removable(self) -> list[int]:
        """Old units that are neither a summary nor preserved."""
        return [i for i in self.old if not _is_summary(self.units[i])
                and not _keeps(self.units[i], self.preserve)]

    def prior(self) -> str:
        first = self.units[0] if self.units and 0 in self.old else None
        return str(first[0]["content"])[len(SUMMARY_PREFIX):] if first and _is_summary(
            first) else ""

    def remove(self, gone: Sequence[int], kind: str, transcript: Transcript | None) -> None:
        removed = _flat([self.units[i] for i in gone])
        self.dropped += removed
        if transcript and removed:
            transcript.record(kind, removed)
        self.units = [u for i, u in enumerate(self.units) if i not in set(gone)]

    def summarise_into(self, text: str) -> None:
        """Make ``text`` the one summary unit, at the front of the conversation."""
        self.units = [u for i, u in enumerate(self.units) if not (_is_summary(u) and i == 0)]
        self.made = {"role": "user", "content": SUMMARY_PREFIX + text}
        self.units.insert(0, [self.made])


def _elide(state: _State, count: Callable[[str], int], spill: Spill, notes: list[str]) -> bool:
    done = 0
    for i in state.old:
        for j, m in enumerate(state.units[i]):
            text = m.get("content")
            if (m.get("role") != "tool" or not isinstance(text, str) or ELIDED in text
                    or state.preserve(m) or count(text) <= spill.above):
                continue
            ref = spill.transcript.spill(text) if spill.transcript else "not kept"
            state.units[i][j] = _elide_one(m, ref, spill.above)
            done += 1
    if done:
        notes.append(f"elided {done} long tool result(s)")
    return bool(done)


def _prune(state: _State, spill: Spill, notes: list[str]) -> bool:
    later: set[str] = set()
    gone: list[int] = []
    for i in range(len(state.units) - 1, -1, -1):
        sig = _signature(state.units[i])
        if sig is None:
            continue
        if sig in later and i in state.old and not _keeps(state.units[i], state.preserve):
            gone.append(i)
        later.add(sig)
    if gone:
        state.remove(gone, "superseded", spill.transcript)
        notes.append(f"dropped {len(gone)} superseded tool call(s)")
    return bool(gone)


def _summarise(state: _State, run: _Plan, notes: list[str]) -> bool:
    span = state.removable()
    if run.summarizer is None or not span:
        return False
    try:
        text = _rolling(run.summarizer, _flat([state.units[i] for i in span]), state.prior(),
                        run.count)
    except (ServerError, ValueError, RuntimeError, KeyError) as exc:
        notes.append(f"summary failed ({type(exc).__name__}: {exc})")
        return False
    if not text.strip():
        notes.append("summary was empty")
        return False
    state.remove(span, "summarised", run.spill.transcript)
    state.summarise_into(_clip(text.strip(), run.count, max(64, run.budget // 4)))
    notes.append(f"summarised {len(span)} unit(s)")
    return True


def _truncate(state: _State, run: _Plan, notes: list[str]) -> bool:
    total, gone = state.tokens(run.count), []
    for i in state.removable():
        if total <= run.budget:
            break
        total -= sum(message_tokens(m, run.count) for m in state.units[i])
        gone.append(i)
    if not gone:
        return False
    count = len(_flat([state.units[i] for i in gone]))
    prior = state.prior()
    state.remove(gone, "truncated", run.spill.transcript)
    state.summarise_into((prior + "\n" if prior else "")
                         + f"[{count} earlier messages were removed to fit the context.]")
    notes.append(f"removed {count} message(s) outright")
    return True


@dataclass(frozen=True, slots=True)
class _Plan:
    budget: int
    count: Callable[[str], int]
    summarizer: Summarizer | None
    spill: Spill


def compact(messages: Sequence[Message], *, budget: int, strategy: str = "auto",
            using: Compaction | None = None,
            count: Callable[[str], int] | None = None) -> CompactResult:
    """``messages`` fitted to ``budget`` tokens, as a `CompactResult`.

    ``strategy`` is ``auto`` (each stage in turn until the budget is met), or one of
    `STAGES` run alone. ``using`` sets what is kept (``keep_last``, ``preserve``), how
    results are cut (``spill``) and who writes the summary (``summarizer``); without a
    summarizer, ``auto`` goes from pruning straight to removal. The last user message is
    always kept.
    """
    if strategy != "auto" and strategy not in STAGES:
        raise ValueError(f"strategy {strategy!r} is not auto or one of {STAGES}")
    using = using or Compaction()
    run = _Plan(budget, count or Counter(), using.summarizer if using.summarize else None,
                using.spill)
    state = _State(messages, using.keep_last, _preserver(using.preserve, messages))
    before = state.tokens(run.count)
    if strategy == "auto" and before <= budget:
        return CompactResult(list(messages), None, 0, before, before, "none")
    steps: dict[str, Callable[[], bool]] = {}
    notes: list[str] = []
    steps["elide"] = lambda: _elide(state, run.count, run.spill, notes)
    steps["prune"] = lambda: _prune(state, run.spill, notes)
    steps["summarise"] = lambda: _summarise(state, run, notes)
    steps["truncate"] = lambda: _truncate(state, run, notes)
    used = []
    for stage in STAGES if strategy == "auto" else (strategy,):
        if strategy == "auto" and state.tokens(run.count) <= budget:
            break
        if steps[stage]():
            used.append(stage)
    after = state.tokens(run.count)
    if after > budget:
        notes.append(f"still {after - budget} tokens over: the kept messages alone exceed it")
    return CompactResult(state.messages(), state.made, len(state.dropped), before, after,
                         "+".join(used) or "none", tuple(notes))
