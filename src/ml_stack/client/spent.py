"""What answering cost, read off the replies: which model, how many calls, how long, what it
read, wrote, kept and drafted. One object per answer; every reply is noted into it.

The bench's `Metered` counts the same things per question for a table. This is the shape
that rides along with an answer, so the page -- and a person testing one -- can see what
the served model is and what it spent, without a benchmark.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Spent:
    model: str = ""                 # what the server says it is serving, from the reply
    calls: int = 0
    seconds: float = 0.0            # wall, on the asking side, across every call
    generating_ms: float = 0.0      # what the server spent reading and writing
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

    def note(self, reply: Any, took: float) -> None:
        """Add one reply and the seconds it took to arrive."""
        self.calls += 1
        self.seconds += float(took)
        raw = getattr(reply, "raw", None) or {}
        model = raw.get("model")
        if isinstance(model, str) and model and not self.model:
            self.model = model
        usage = raw.get("usage") or {}
        timings = raw.get("timings") or {}
        prompt_ms = float(timings.get("prompt_ms") or 0)
        predicted_ms = float(timings.get("predicted_ms") or 0)
        self.generating_ms += prompt_ms + predicted_ms
        if self.first_token is None:
            self.first_token = round(max(0.0, float(took) - predicted_ms / 1000), 3)
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.cached_tokens += int(timings.get("cache_n")
                                  or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                                  or 0)
        self.read_tokens += int(timings.get("prompt_n") or 0)
        held = (int(timings.get("cache_n") or 0) + int(timings.get("prompt_n") or 0)
                + int(timings.get("predicted_n") or 0))
        if not held:
            held = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        self.context_last = held
        self.context_peak = max(self.context_peak, held)
        self.draft_tokens += int(timings.get("draft_n") or 0)
        self.draft_taken += int(timings.get("draft_n_accepted") or 0)
        finish = getattr(reply, "finish_reason", None)
        if finish:
            self.finish = str(finish)
            self.truncated = self.truncated or finish == "length"
        self.thinking_chars += len(getattr(reply, "thinking", None) or "")
        self.answer_chars += len(getattr(reply, "content", None) or "")
        if getattr(reply, "thinking", None):
            self.part("thinking", reply.thinking)
        if getattr(reply, "content", None):
            self.part("answer", reply.content)
        self.tool_calls += len(getattr(reply, "tool_calls", None) or ())

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
        """Completion tokens over the server's generating time."""
        return (self.completion_tokens / (self.generating_ms / 1000)
                if self.generating_ms and self.completion_tokens else None)

    @staticmethod
    def totals(records: Any) -> dict[str, Any]:
        """Every `public()` record of a session added up: the counts and tokens summed, the
        derived rates recomputed over the sums, the models named once each, and ``answers``
        for how many records went in. What the page shows as "this session"."""
        summed = {k: 0 for k in ("calls", "seconds", "generating_ms", "prompt_tokens",
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
        out["context_peak"] = peak
        out["context_mean"] = round(sum(peaks) / len(peaks)) if peaks else 0
        out["parts"] = parts
        return out

    def public(self) -> dict[str, Any]:
        """Every field plus the derived ones, JSON-ready."""
        out = asdict(self)
        out.pop("steps", None)
        out["seconds"] = round(self.seconds, 3)
        out["generating_ms"] = round(self.generating_ms, 1)
        out["drafted"] = self.drafted
        out["acceptance"] = (round(self.acceptance, 3) if self.acceptance is not None else None)
        out["tokens_per_second"] = (round(self.tokens_per_second, 1)
                                    if self.tokens_per_second is not None else None)
        return out
