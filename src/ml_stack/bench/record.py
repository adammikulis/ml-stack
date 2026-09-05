"""One measurement run as one record: what identifies it, what it describes, and its spread.

`Measured` is what `keep.save` writes and every reader reads: the model and what served
it, the serve shape, the `Asking`, the sampling, a digest of the system prompt and the
tool schemas (`prompt_digest`), the commit and the host, the timing, the per-question
rows, and -- for a run measured over several seeds -- a `Spread` per metric. Its accessors
are the one answer each question has: `build`, `head`, `made`, `identity`.

`ml_stack.serve.shape.Run` is the other kind of run: a model to serve, asked one way. That
one is a plan; this one is what a measurement left behind.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ml_stack.graph.asking import Asking

__all__ = ["Measured", "Spread", "prompt_digest"]


def prompt_digest(system: str, tools: Sequence[Any] = ()) -> str:
    """A digest of the system prompt and the tool schemas a question was asked under.

    Names, descriptions and parameters, the way `ml_stack.graph.cache.fingerprint` hashes
    them, so a run under an edited prompt is never read as the run before the edit.
    """
    named = sorted(
        (str(fn.get("name") or ""), str(fn.get("description") or ""),
         json.dumps(fn.get("parameters") or {}, sort_keys=True, default=str))
        for one in tools or ()
        if isinstance(one, Mapping) and isinstance(fn := (one.get("function") or {}), Mapping))
    blob = json.dumps([" ".join(str(system or "").split()), named], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Spread:
    """One metric over several seeds: the mean, the standard deviation, and how many seeds
    produced it."""

    mean: float = 0.0
    std: float = 0.0
    n: int = 0
    values: tuple[float, ...] = ()

    @classmethod
    def over(cls, values: Sequence[float]) -> Spread:
        """The mean and spread of ``values``; a single value has a spread of zero."""
        got = [float(v) for v in values]
        if not got:
            return cls()
        return cls(mean=statistics.fmean(got),
                   std=statistics.stdev(got) if len(got) > 1 else 0.0,
                   n=len(got), values=tuple(got))

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "n": self.n, "values": list(self.values)}

    @classmethod
    def from_dict(cls, one: Mapping[str, Any]) -> Spread:
        values = tuple(float(v) for v in (one.get("values") or ()))
        return cls(mean=float(one.get("mean") or 0.0), std=float(one.get("std") or 0.0),
                   n=int(one.get("n") or 0), values=values)


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _frozen(value: Any) -> Any:
    """``value`` as something hashable, so an identity can be a tuple."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _frozen(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_frozen(v) for v in value)
    return value


@dataclass(frozen=True)
class Measured:
    """One run, kept: everything that identifies it and everything it measured."""

    label: str = ""
    at: str = ""
    kind: str = ""
    key: str = ""
    model: str = ""
    binary: str = ""
    served_by: Mapping[str, Any] | None = None
    context: int = 0
    slots: int = 0
    cache_type: str = ""
    head: str = ""
    head_ahead: int | None = None
    reasoning_budget: int | None = None
    graph: str = ""
    finder: str = ""
    ways: Mapping[str, Any] | None = None
    sampling: Mapping[str, Any] = field(default_factory=dict)
    prompts: str = ""
    host: str = ""
    commit: str = ""
    seconds: float = 0.0
    rows: tuple[Mapping[str, Any], ...] = ()
    seeds: tuple[int, ...] = ()
    metrics: Mapping[str, Spread] = field(default_factory=dict)
    failures: tuple[str, ...] = ()
    server: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------- the store boundary

    @classmethod
    def from_dict(cls, one: Mapping[str, Any]) -> Measured:
        """A run as it is kept, read back."""
        server = dict(one.get("server") or {})
        asked = one.get("asking") if isinstance(one.get("asking"), Mapping) else None
        sampling = dict((asked or {}).get("sampling") or server.get("sampling") or {})
        known = {"at", "label", "kind", "key", "server", "asking", "rows", "prompts",
                 "seeds", "metrics", "failures", "seconds"}
        metrics = {str(k): Spread.from_dict(v) if isinstance(v, Mapping) else Spread()
                   for k, v in (one.get("metrics") or {}).items()}
        return cls(
            label=str(one.get("label") or ""),
            at=str(one.get("at") or ""),
            kind=str(one.get("kind") or ""),
            key=str(one.get("key") or ""),
            model=str(server.get("model") or ""),
            binary=str(server.get("binary") or ""),
            served_by=(dict(server["served_by"])
                       if isinstance(server.get("served_by"), Mapping) else None),
            context=int(server.get("context") or 0),
            slots=int(server.get("slots") or 0),
            cache_type=str(server.get("cache_type") or ""),
            head=str(server.get("draft_model") or server.get("draft") or ""),
            head_ahead=_int(server.get("spec_draft_max")),
            reasoning_budget=_int(server.get("reasoning_budget")),
            graph=str(server.get("graph") or ""),
            finder=str(server.get("finder") or ""),
            ways=dict(asked) if asked is not None else None,
            sampling=sampling,
            prompts=str(one.get("prompts") or ""),
            host=str(server.get("host") or ""),
            commit=str(server.get("commit") or ""),
            seconds=float(one.get("seconds") or 0.0),
            rows=tuple(one.get("rows") or ()),
            seeds=tuple(int(s) for s in (one.get("seeds") or ())),
            metrics=metrics,
            failures=tuple(str(f) for f in (one.get("failures") or ())),
            server=server,
            extra={k: v for k, v in one.items() if k not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        """The run as it is kept: no key that says nothing."""
        out: dict[str, Any] = {"at": self.at, "label": self.label}
        if self.kind:
            out["kind"] = self.kind
        out["server"] = dict(self.server)
        if self.ways:
            out["asking"] = dict(self.ways)
        if self.prompts:
            out["prompts"] = self.prompts
        out["rows"] = [dict(r) for r in self.rows]
        if self.seeds:
            out["seeds"] = list(self.seeds)
        if self.metrics:
            out["metrics"] = {k: v.to_dict() for k, v in self.metrics.items()}
        if self.failures:
            out["failures"] = list(self.failures)
        if self.seconds:
            out["seconds"] = self.seconds
        out.update(self.extra)
        return out

    # --------------------------------------------------------------------- what it names

    @property
    def build(self) -> str:
        """The named llama.cpp build a run was served on, the program for a run served by
        something else, "" for the managed current build."""
        from pathlib import Path

        from ml_stack.bench.backends import describe
        from ml_stack.serve.build import NAMED_DIR

        record = self.served_by
        if isinstance(record, Mapping) and record.get("program"):
            if str(record.get("program")).lower() not in ("llama.cpp", "llama-server"):
                return describe(record)
            if record.get("build"):
                return str(record["build"])
        if self.server.get("build"):
            return str(self.server["build"])
        binary = Path(self.binary or "")
        try:
            named = binary.resolve().relative_to(Path(NAMED_DIR).resolve())
        except (OSError, ValueError):
            try:
                named = binary.relative_to(Path(NAMED_DIR))
            except ValueError:
                return ""
        return named.parts[0] if named.parts else ""

    @property
    def head_said(self) -> str:
        """The draft head and how far it guessed -- ``mtp-a.gguf@n4`` -- or "-" for none."""
        if not self.head:
            return "-"
        return f"{self.head}@n{self.head_ahead}" if self.head_ahead is not None else self.head

    @property
    def dirty(self) -> bool:
        """Whether the tree that measured this had changes in it."""
        return "(dirty)" in self.commit

    @property
    def sha(self) -> str:
        """The commit that measured this, without the dirty mark."""
        return self.commit.split(" ")[0]

    @property
    def scored(self) -> list[Mapping[str, Any]]:
        """The rows that were scored: a question with something expected of it."""
        return [r for r in self.rows if r.get("expected")]

    @property
    def questions(self) -> int:
        return len(self.scored)

    @property
    def made_per_question(self) -> float | None:
        """Entries the answers named that no tool call produced, per scored question; None
        for a run kept before the count."""
        rows = self.scored
        if not rows or not any("unread_named" in r for r in rows):
            return None
        return sum(float(r.get("unread_named") or 0) for r in rows) / len(rows)

    @property
    def made(self) -> str:
        """`made_per_question` to one decimal; "-" for a run from before the count."""
        got = self.made_per_question
        return "-" if got is None else f"{got:.1f}"

    # ------------------------------------------------------------------ what identifies it

    @property
    def asked(self) -> Asking | None:
        """The `Asking` a run recorded, or None for one kept before the record existed."""
        if self.ways is None:
            return None
        fields = Asking.__dataclass_fields__
        return Asking(**{k: v for k, v in self.ways.items() if k in fields})

    @property
    def knows_asking(self) -> bool:
        """Whether the run recorded how it asked, rather than leaving it to its label."""
        return self.ways is not None

    @property
    def knows_prompts(self) -> bool:
        """Whether the run recorded a digest of what the model was shown."""
        return bool(self.prompts)

    @property
    def serving(self) -> tuple[Any, ...]:
        """What served this run: the model, the build, the head, and the serve shape."""
        return (self.model, self.build, self.head, self.head_ahead, self.context,
                self.slots, self.cache_type, self.reasoning_budget)

    @property
    def identity(self) -> tuple[Any, ...]:
        """Everything that makes this a different measurement from another one.

        Two runs with the same identity asked the same questions of the same model, served
        and prompted the same way; two runs whose system prompts differ by a character do
        not.
        """
        return (*self.serving, self.graph, self.finder, _frozen(self.ways),
                _frozen(self.sampling), self.prompts)

    @property
    def fingerprint(self) -> str:
        """`identity` as one string."""
        return hashlib.sha256(
            json.dumps(self.identity, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def undrafted(self) -> tuple[Any, ...]:
        """`identity` with the draft head taken out: what a drafted run's baseline shares."""
        me = list(self.identity)
        me[2], me[3] = "", None
        return tuple(me)


def of(one: Mapping[str, Any] | Measured) -> Measured:
    """``one`` as a `Measured`, whether it is one already or a run as it is kept."""
    return one if isinstance(one, Measured) else Measured.from_dict(one)
