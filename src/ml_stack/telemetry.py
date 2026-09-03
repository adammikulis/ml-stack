"""One model call, written down once.

Three records grew up separately and agreed only by construction: `client.spent.Spent`
(what one answer cost, added up off the replies), the bench's per-call trace
(`graph.bench.measure.Counting._reply`, the same numbers kept per call so a transcript and
a total are the same arithmetic twice), and `serve.fit`'s per-model fit (what a model costs
to hold, which is the other half of what a peak in `Spent` is measured against). The first
two count the *same reply*, field for field, and drifted apart every time a timing was
added to one of them.

`Call` is that reply, once: when it came back, which model on which host answered, what it
asked to call and with what, how much came back, why it stopped, and every number
llama.cpp's ``timings`` reports. `Spent` is now the sum of its `Call`s -- `Spent.note`
builds one and folds it in -- and with ``keep_calls`` it keeps them, so an answer can be
read call by call and not only as a total.

The bench's trace entries are the same record in a dict: `Call.from_trace` reads one back,
so ``ml-stack-bench show --trace`` and a page's ``/metrics`` can print the same rows. The
two-line change that would make the bench *write* `Call` rather than a hand-built dict is
in `from_trace`'s docstring; measure.py is not edited here.

Nothing in this module talks to a server or a store. It is a record and its arithmetic, so
a test builds one from a fake reply in a line.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["ARGS_CAP", "Call", "args_summary"]

ARGS_CAP = 200
"""How much of one string argument a call record keeps.

A tool argument is a graph id or a handful of words, except when a model has decided to
paste a paragraph into one, and that paragraph is the thing that makes a telemetry ring
megabytes. The shape of the argument is what a trace is read for; the tail of it is not.
"""


def args_summary(args: Any, *, cap: int = ARGS_CAP) -> dict[str, Any]:
    """One tool call's arguments, as a record keeps them: a dict, with long strings cut.

    Anything that is not a mapping is kept under ``_value`` rather than dropped -- a model
    that sent a list where the schema asked for an object is exactly what a trace is being
    read to find out.
    """
    if not isinstance(args, Mapping):
        return {"_value": str(args)[:cap]}
    out: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > cap:
            out[str(key)] = value[:cap] + "..."
        else:
            out[str(key)] = value
    return out


def _offered(tools: Any) -> list[str]:
    """The names of the tools a call was given to choose from, out of the schemas sent.

    A schema, or a name already pulled out of one: a caller that keeps the offered tools
    as strings should not have to wrap them to be recorded.
    """
    names = []
    for one in tools or ():
        if isinstance(one, Mapping):
            names.append(str((one.get("function") or {}).get("name") or one.get("name") or ""))
        else:
            names.append(str(one))
    return names


def _counted(timings: Mapping[str, Any], key: str) -> int | None:
    """``timings[key]`` as a count: 0 when the key is absent, ``None`` when it is null."""
    if key in timings and timings[key] is None:
        return None
    return int(timings.get(key) or 0)


def _asked(reply: Any, *, cap: int = ARGS_CAP) -> list[dict[str, Any]]:
    """The tool calls on one reply as ``[{"name", "args"}]``, arguments parsed.

    Arguments arrive as a JSON string the model wrote, so they are sometimes not JSON.
    Unparsable ones are kept verbatim under ``_unparsed``, which is the only way to see
    afterwards that the model's syntax, and not the schema, was what failed.
    """
    out = []
    for call in getattr(reply, "tool_calls", None) or ():
        fn = (call.get("function") or {}) if isinstance(call, Mapping) else {}
        try:
            args: Any = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {"_unparsed": str(fn.get("arguments") or "")[:cap]}
        out.append({"name": str(fn.get("name") or ""), "args": args_summary(args, cap=cap)})
    return out


@dataclass
class Call:
    """One call to a model server: what was asked of it, what came back, what it cost.

    Every number here is the server's own, off ``raw["timings"]`` and ``raw["usage"]`` --
    nothing is estimated. ``seconds`` is the only one measured on this side, and the gap
    between it and ``prompt_ms + predicted_ms`` is time spent waiting for a slot.
    """

    when: float = 0.0               # unix time the reply came back
    model: str = ""                 # what the server says it is serving
    host: str = ""                  # and where it is, when the caller knows
    port: int = 0
    tool: str = ""                  # the tool this call asked for; several joined by " + "
    args: dict[str, Any] = field(default_factory=dict)   # the first one's arguments, cut
    asked: list[dict[str, Any]] = field(default_factory=list)  # every call: name and args
    offered: list[str] = field(default_factory=list)     # the tools it was given to choose
    result_chars: int = 0           # how much came back: the reply's own text, uncut
    result_ids: int = 0             # how many graph ids the result named, when counted
    finish: str = ""                # finish_reason
    thinking_chars: int = 0
    prompt_ms: float = 0.0          # the server reading the prompt
    predicted_ms: float = 0.0       # and writing the answer
    load_ms: float = 0.0            # loading the weights first, when the server says
    prompt_n: int = 0               # tokens it had to read
    # None is "this server does not measure it" -- a backend with no cache count or no
    # draft head writes null into ``timings``; a key left out reads 0 as before.
    cache_n: int | None = 0         # and kept from the call before
    predicted_n: int = 0
    draft_n: int | None = 0         # guessed ahead by a draft head
    draft_n_accepted: int | None = 0  # and accepted
    # The usage totals as well as the timings: a conversation re-sends everything every
    # turn, so `prompt_tokens` counts the same words over and over while `prompt_n` counts
    # what was actually read. `Spent` sums both, and cannot be the sum of its calls unless
    # they are here.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0            # wall, on this side

    # ------------------------------------------------------------------ reading one

    @classmethod
    def from_reply(cls, reply: Any, took: float, *, tool: str = "", args: Any = None,
                   host: str = "", port: int = 0, model: str = "", when: float | None = None,
                   offered: Any = (), result_chars: int | None = None, result_ids: int = 0,
                   cap: int = ARGS_CAP) -> Call:
        """One `Reply` and the seconds it took, as a record.

        Everything the reply knows is read off it; everything it cannot know -- which
        server answered, how many graph ids the tool result named -- is the caller's to
        pass and defaults to empty rather than to a guess. ``tool`` and ``args`` override
        what the reply's own tool calls say, for a caller that already resolved them.
        """
        raw = getattr(reply, "raw", None) or {}
        usage = raw.get("usage") or {}
        timings = raw.get("timings") or {}
        made = _asked(reply, cap=cap)
        if "cache_n" in timings and timings["cache_n"] is None:
            cached = None
        else:
            cached = int(timings.get("cache_n")
                         or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                         or 0)
        return cls(
            when=float(time.time() if when is None else when),
            model=str(model or raw.get("model") or ""),
            host=str(host), port=int(port or 0),
            tool=str(tool) or " + ".join(one["name"] for one in made),
            args=args_summary(args, cap=cap) if args is not None
            else dict(made[0]["args"]) if made else {},
            asked=made,
            offered=_offered(offered),
            result_chars=int(len(getattr(reply, "content", None) or "")
                             if result_chars is None else result_chars),
            result_ids=int(result_ids or 0),
            finish=str(getattr(reply, "finish_reason", None) or ""),
            thinking_chars=len(getattr(reply, "thinking", None) or ""),
            prompt_ms=float(timings.get("prompt_ms") or 0),
            predicted_ms=float(timings.get("predicted_ms") or 0),
            load_ms=float(timings.get("load_ms") or 0),
            prompt_n=int(timings.get("prompt_n") or 0),
            cache_n=cached,
            predicted_n=int(timings.get("predicted_n") or 0),
            draft_n=_counted(timings, "draft_n"),
            draft_n_accepted=_counted(timings, "draft_n_accepted"),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            seconds=float(took),
        )

    @classmethod
    def from_trace(cls, entry: Mapping[str, Any]) -> Call:
        """One entry of the bench's transcript (`Counting.trace`) as a `Call`.

        The bench writes an assistant entry per reply -- ``model``, ``content``/``chars``,
        ``thinking_chars``, ``finish``, ``seconds``, ``offered``, ``tool_calls`` and the
        ``timings`` and ``tokens`` it read off the same reply -- and a ``tool`` entry per
        result, with the result's ``chars`` and the ``ids`` it named. Both map here: an
        assistant entry fills the timings, a tool entry fills ``tool``, ``result_chars``
        and ``result_ids`` and leaves them zero. Anything else (``role: "tools"``, the
        system and user messages) comes back as an empty record, so a caller can map a
        whole trace and drop the empties.

        The two-line change that would make `measure.Counting._reply` *write* this record
        rather than a dict shaped like it, once nobody reads the old key names::

            from ml_stack.telemetry import Call                       # 1: the import
            ...
            self.trace.append(Call.from_reply(reply, took, offered=tools,   # 2: the body
                                              cap=TRACE_CAP).public())

        It is not made here: `bench.transcript` and ``show --trace`` read the old keys, and
        the run store on disk is full of them. `from_trace` is the bridge until then, and
        is what lets ``show --trace`` print a page's calls and a bench's the same way.
        """
        entry = dict(entry or {})
        role = str(entry.get("role") or "")
        if role == "tool":
            return cls(tool=str(entry.get("name") or ""),
                       result_chars=int(entry.get("chars") or 0),
                       result_ids=int(entry.get("ids") or 0))
        if role != "assistant":
            return cls()
        timings = entry.get("timings") or {}
        tokens = entry.get("tokens") or {}
        made = [{"name": str(one.get("name") or ""),
                 "args": args_summary(one.get("args"))}
                for one in (entry.get("tool_calls") or ()) if isinstance(one, Mapping)]
        return cls(
            when=float(entry.get("when") or 0.0),
            model=str(entry.get("model") or ""),
            host=str(entry.get("host") or ""), port=int(entry.get("port") or 0),
            tool=str(entry.get("tool") or "") or " + ".join(one["name"] for one in made),
            args=dict(made[0]["args"]) if made else {},
            asked=made,
            offered=_offered(entry.get("offered")),
            result_chars=int(entry.get("chars") or 0),
            result_ids=int(entry.get("ids") or 0),
            finish=str(entry.get("finish") or ""),
            thinking_chars=int(entry.get("thinking_chars") or 0),
            prompt_ms=float(timings.get("prompt_ms") or 0),
            predicted_ms=float(timings.get("predicted_ms") or 0),
            prompt_n=int(timings.get("prompt_n") or 0),
            cache_n=int(timings.get("cache_n") or 0),
            predicted_n=int(timings.get("predicted_n") or 0),
            draft_n=int(timings.get("draft_n") or 0),
            draft_n_accepted=int(timings.get("draft_n_accepted") or 0),
            prompt_tokens=int(tokens.get("prompt") or 0),
            completion_tokens=int(tokens.get("completion") or 0),
            seconds=float(entry.get("seconds") or 0.0),
        )

    # ------------------------------------------------------------------ arithmetic

    @property
    def held(self) -> int | None:
        """What the slot held for this call: the prefix it kept, the prompt it read and the
        answer it wrote. Falls back to the usage totals on a server that reports no
        timings, because a peak of zero would read as a slot that held nothing. ``None``
        on a server that does not count the prefix it kept."""
        if self.cache_n is None:
            return None
        held = self.cache_n + self.prompt_n + self.predicted_n
        return held or (self.prompt_tokens + self.completion_tokens)

    @property
    def first_token(self) -> float:
        """Seconds before this call's answer began: the wall clock less the time the server
        spent writing. Nothing here streams, so it is not seen arriving -- what is known is
        that everything but decode came first."""
        return round(max(0.0, self.seconds - self.predicted_ms / 1000), 3)

    @property
    def generating_ms(self) -> float:
        return self.prompt_ms + self.predicted_ms

    @property
    def waited_ms(self) -> float:
        """Wall clock the server did not account for: queueing for a slot, and the wire."""
        return round(max(0.0, self.seconds * 1000 - self.generating_ms), 1)

    def public(self) -> dict[str, Any]:
        """Every field plus the derived ones, JSON-ready and rounded for reading."""
        out = asdict(self)
        out["when"] = round(self.when, 3)
        out["seconds"] = round(self.seconds, 3)
        out["prompt_ms"] = round(self.prompt_ms, 1)
        out["predicted_ms"] = round(self.predicted_ms, 1)
        out["generating_ms"] = round(self.generating_ms, 1)
        out["waited_ms"] = self.waited_ms
        out["first_token"] = self.first_token
        out["held"] = self.held
        return out
