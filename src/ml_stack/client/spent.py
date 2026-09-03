"""What answering cost, read off the replies: which model, how many calls, how long, what it
read, wrote, kept and drafted. One object per answer; every reply is noted into it.

The bench's `Metered` counts the same things per question for a table. This is the shape
that rides along with an answer, so the page -- and a person testing one -- can see what
the served model is and what it spent, without a benchmark.

A `Spent` is the sum of its calls: `note` turns one reply into a `ml_stack.telemetry.Call`
-- the one record every part of the stack now writes -- and `add` folds it in. With
``keep_calls`` the calls are kept as well as summed, so an answer can be read call by call;
off by default, because the page wants a footer and not a transcript, and a hundred-turn
conversation with every call kept is megabytes nobody asked for. The bench turns it on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ml_stack.telemetry import Call


@dataclass
class Spent:
    model: str = ""                 # what the server says it is serving, from the reply
    calls: int = 0
    seconds: float = 0.0            # wall, on the asking side, across every call
    generating_ms: float = 0.0      # what the server spent reading and writing
    prompt_ms: float = 0.0          # of that, reading the prompt (prefill)
    predicted_ms: float = 0.0       # and writing the answer (decode)
    first_token: float | None = None  # seconds before the first call's answer began
    prompt_tokens: int = 0          # every token sent, every call (the conversation re-sends)
    completion_tokens: int = 0
    read_tokens: int = 0            # what the server had to read (timings.prompt_n)
    cached_tokens: int = 0          # what it kept from the call before (timings.cache_n)
    draft_tokens: int = 0           # guessed ahead by a draft head
    draft_taken: int = 0            # and accepted
    finish: str = ""                # the last reply's finish_reason
    truncated: bool = False         # any reply cut by the ceiling
    thinking_chars: int = 0
    answer_chars: int = 0
    tool_calls: int = 0
    # What the slot held: the largest prompt-plus-answer any one call put in the cache
    # (timings.cache_n + prompt_n + predicted_n), and the last. With a rolling window,
    # a summary and recall, the conversation's length says nothing about this; the peak
    # is what bounds how many users fit -- `ml-stack-serve fit --per-user <peak>`.
    context_peak: int = 0
    context_last: int = 0
    # Estimated tokens by what filled the prompt: system, tools, summary, recalled, window,
    # shortlist, question, tool_results, thinking, answer. Estimated by the token counter
    # in `ml_stack.client.tokens`; the server's own counts above are the exact ones.
    parts: dict[str, int] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list, repr=False)  # unused by the page; free
    # Every call, kept rather than only summed. Off by default: the page shows a footer and
    # the answer is written into a conversation store, where a transcript per turn is
    # megabytes. `ml-stack-bench` turns it on, and so does anything printing a trace.
    keep_calls: bool = False
    calls_detail: list[Call] = field(default_factory=list, repr=False)

    def note(self, reply: Any, took: float, **detail: Any) -> None:
        """Add one reply and the seconds it took to arrive.

        ``detail`` is what the reply cannot say and the caller can -- ``host``, ``port``,
        ``tool``, ``result_ids`` -- and goes straight to `Call.from_reply`.
        """
        call = Call.from_reply(reply, took, **detail)
        self.add(call)
        # The parts are estimated from the text, not counted off the reply, so they need
        # the reply itself and not the record of it.
        if getattr(reply, "thinking", None):
            self.part("thinking", reply.thinking)
        if getattr(reply, "content", None):
            self.part("answer", reply.content)

    def add(self, call: Call) -> None:
        """Fold one `Call` into the totals -- and keep it, with ``keep_calls``.

        Every number the page and the bench show is added up here and nowhere else, which
        is the point of there being one record: a timing that arrives on the reply is read
        by `Call.from_reply`, and everything downstream of it sums the same field.
        """
        self.calls += 1
        self.seconds += call.seconds
        if call.model and not self.model:
            self.model = call.model
        self.generating_ms += call.generating_ms
        self.prompt_ms += call.prompt_ms
        self.predicted_ms += call.predicted_ms
        if self.first_token is None:
            self.first_token = call.first_token
        self.prompt_tokens += call.prompt_tokens
        self.completion_tokens += call.completion_tokens
        self.cached_tokens += call.cache_n
        self.read_tokens += call.prompt_n
        self.context_last = call.held
        self.context_peak = max(self.context_peak, call.held)
        self.draft_tokens += call.draft_n
        self.draft_taken += call.draft_n_accepted
        if call.finish:
            self.finish = call.finish
            self.truncated = self.truncated or call.finish == "length"
        self.thinking_chars += call.thinking_chars
        self.answer_chars += call.result_chars
        self.tool_calls += len(call.asked)
        if self.keep_calls:
            self.calls_detail.append(call)

    def part(self, name: str, text: Any) -> None:
        """Count ``text`` (a string, or anything `str` makes one of) against ``name``."""
        from ml_stack.client.tokens import estimate_tokens

        self.parts[name] = self.parts.get(name, 0) + int(estimate_tokens(str(text or "")))

    @property
    def drafted(self) -> bool:
        return self.draft_tokens > 0

    @property
    def acceptance(self) -> float | None:
        """Share of drafted tokens the model accepted, or None without a draft head."""
        return self.draft_taken / self.draft_tokens if self.draft_tokens else None

    @property
    def tokens_per_second(self) -> float | None:
        """Completion tokens over the server's whole generating time, reading included --
        what a person waits on. `decode_tokens_per_second` is the model's own pace."""
        return (self.completion_tokens / (self.generating_ms / 1000)
                if self.generating_ms and self.completion_tokens else None)

    @property
    def decode_tokens_per_second(self) -> float | None:
        """Completion tokens over decode time alone: the speed the hardware writes at.
        The page showed 15 tok/s where the model decoded at 28 (2026-09-02) -- the
        difference was reading two thousand new tokens of tool results before each call."""
        return (self.completion_tokens / (self.predicted_ms / 1000)
                if self.predicted_ms and self.completion_tokens else None)

    @property
    def prompt_tokens_per_second(self) -> float | None:
        """Tokens read (not cached) over prefill time."""
        return (self.read_tokens / (self.prompt_ms / 1000)
                if self.prompt_ms and self.read_tokens else None)

    @staticmethod
    def totals(records: Any) -> dict[str, Any]:
        """Every `public()` record of a session added up: the counts and tokens summed, the
        derived rates recomputed over the sums, the models named once each, and ``answers``
        for how many records went in. What the page shows as "this session"."""
        summed = {k: 0 for k in ("calls", "seconds", "generating_ms", "prompt_ms", "predicted_ms",
                                 "prompt_tokens",
                                 "completion_tokens", "read_tokens", "cached_tokens",
                                 "draft_tokens", "draft_taken", "thinking_chars",
                                 "answer_chars", "tool_calls")}
        models: list[str] = []
        answers = 0
        truncated = 0
        firsts: list[float] = []
        peak = 0
        peaks: list[int] = []
        parts: dict[str, int] = {}
        for one in records or ():
            if not isinstance(one, Mapping) or not one.get("calls"):
                continue
            answers += 1
            for k in summed:
                summed[k] += one.get(k) or 0
            peak = max(peak, int(one.get("context_peak") or 0))
            peaks.append(int(one.get("context_peak") or 0))
            for name, n in (one.get("parts") or {}).items():
                parts[name] = parts.get(name, 0) + int(n or 0)
            name = str(one.get("model") or "")
            if name and name not in models:
                models.append(name)
            truncated += 1 if one.get("truncated") else 0
            if one.get("first_token") is not None:
                firsts.append(float(one["first_token"]))
        out: dict[str, Any] = {"answers": answers, **summed}
        out["seconds"] = round(float(summed["seconds"]), 3)
        out["generating_ms"] = round(float(summed["generating_ms"]), 1)
        out["models"] = models
        out["model"] = models[-1] if models else ""
        out["truncated"] = truncated
        out["drafted"] = summed["draft_tokens"] > 0
        out["acceptance"] = (round(summed["draft_taken"] / summed["draft_tokens"], 3)
                             if summed["draft_tokens"] else None)
        out["tokens_per_second"] = (round(summed["completion_tokens"]
                                          / (summed["generating_ms"] / 1000), 1)
                                    if summed["generating_ms"] and summed["completion_tokens"]
                                    else None)
        out["first_token_mean"] = round(sum(firsts) / len(firsts), 3) if firsts else None
        out["decode_tokens_per_second"] = (round(summed["completion_tokens"]
                                                 / (summed["predicted_ms"] / 1000), 1)
                                           if summed["predicted_ms"] and summed["completion_tokens"]
                                           else None)
        out["prompt_tokens_per_second"] = (round(summed["read_tokens"] / (summed["prompt_ms"] / 1000))
                                           if summed["prompt_ms"] and summed["read_tokens"] else None)
        out["context_peak"] = peak
        out["context_mean"] = round(sum(peaks) / len(peaks)) if peaks else 0
        out["parts"] = parts
        return out

    def public(self) -> dict[str, Any]:
        """Every field plus the derived ones, JSON-ready."""
        out = asdict(self)
        out.pop("steps", None)
        # Kept off the public record unless they were asked for: `keep_calls` is a knob and
        # not telemetry, and without it the record is byte-for-byte what it always was.
        out.pop("keep_calls", None)
        out.pop("calls_detail", None)
        if self.keep_calls:
            out["calls_detail"] = [one.public() for one in self.calls_detail]
        out["seconds"] = round(self.seconds, 3)
        out["generating_ms"] = round(self.generating_ms, 1)
        out["drafted"] = self.drafted
        out["acceptance"] = (round(self.acceptance, 3) if self.acceptance is not None else None)
        out["tokens_per_second"] = (round(self.tokens_per_second, 1)
                                    if self.tokens_per_second is not None else None)
        out["decode_tokens_per_second"] = (round(self.decode_tokens_per_second, 1)
                                           if self.decode_tokens_per_second is not None else None)
        out["prompt_tokens_per_second"] = (round(self.prompt_tokens_per_second, 0)
                                           if self.prompt_tokens_per_second is not None else None)
        out["prompt_ms"] = round(self.prompt_ms, 1)
        out["predicted_ms"] = round(self.predicted_ms, 1)
        return out
