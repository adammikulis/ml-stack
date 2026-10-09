"""Named, typed questions about one state: yes/no, choice and ordinal score, each answered with
probabilities by the router's backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from poolhouse.decide import router
from poolhouse.decide.base import Request, State, text_of
from poolhouse.decide.logprob import defang
from poolhouse.decide.types import DecideError, Decision, Option, clip, options_of

__all__ = ["Answer", "Question", "choice", "decide", "noul", "prepare", "score"]

NO, YES = "no", "yes"
FAILURES = (DecideError, ValueError, OSError, RuntimeError)


@dataclass(frozen=True, slots=True)
class Question:
    """One question: its kind, text, ordered options and, for a score, the value of each."""

    kind: str
    text: str
    options: tuple[Option, ...]
    values: tuple[int, ...] = ()


def noul(text: str) -> Question:
    """A yes/no question; its answer carries ``p_true``."""
    return Question("noul", _text(text), options_of([NO, YES]))


def choice(text: str, options: Mapping[str, str] | list[str]) -> Question:
    """A question with one answer among ``options``."""
    return Question("choice", _text(text), options_of(options))


def score(text: str, lo: int, hi: int) -> Question:
    """A question answered on the integer scale ``lo`` to ``hi``; its answer carries ``expected``."""
    if hi <= lo:
        raise ValueError(f"a score scale needs lo < hi, got {lo}..{hi}")
    values = tuple(range(lo, hi + 1))
    return Question("score", _text(text), options_of([str(v) for v in values]), values)


def _text(text: str) -> str:
    if not text.strip():
        raise ValueError("a question needs text")
    return text.strip()


@dataclass(frozen=True, slots=True)
class Answer:
    """The answer to one named question. A failed question has ``error`` set, ``abstained``
    True and no choice."""

    name: str
    kind: str
    choice: str | None = None
    scores: Mapping[str, float] = field(default_factory=dict)
    certainty: float = 0.0
    abstained: bool = True
    error: str = ""
    p_true: float | None = None
    expected: float | None = None
    backend: str = ""
    model: str = ""
    latency_ms: float = 0.0

    def public(self) -> dict[str, object]:
        """A JSON-ready dict."""
        return {"name": self.name, "kind": self.kind, "choice": self.choice,
                "scores": dict(self.scores), "certainty": self.certainty,
                "abstained": self.abstained, "error": self.error, "p_true": self.p_true,
                "expected": self.expected, "backend": self.backend, "model": self.model,
                "latency_ms": round(self.latency_ms, 3)}


def prepare(state: State) -> str:
    """The state as text a prompt can carry: clipped and defanged."""
    return defang(clip(text_of(state), "state"))


def _answer(name: str, q: Question, d: Decision) -> Answer:
    probs = d.scores
    return Answer(
        name, q.kind, d.choice, dict(probs), d.certainty, d.abstained, "",
        probs[YES] if q.kind == "noul" else None,
        sum(v * probs[str(v)] for v in q.values) if q.kind == "score" else None,
        d.backend, d.model, d.latency_ms)


def _failed(name: str, q: Question, why: object) -> Answer:
    return Answer(name, q.kind, error=f"{type(why).__name__}: {why}")


def _run(backend: str, config: router.Config, text: str, items: list[tuple[str, Question]],
         abstain_below: float | None) -> dict[str, Answer]:
    asks = [Request(q.text, text, q.options, abstain_below=abstain_below) for _, q in items]
    try:
        decider = router.build(backend, config)
        made = decider.decide_many(asks)
        return {n: _answer(n, q, d) for (n, q), d in zip(items, made, strict=True)}
    except FAILURES:
        pass
    out: dict[str, Answer] = {}
    for (name, q), ask in zip(items, asks, strict=True):
        try:
            got = router.build(backend, config).decide(
                ask.question, ask.state, ask.options, abstain_below=abstain_below)
            out[name] = _answer(name, q, got)
        except FAILURES as exc:
            out[name] = _failed(name, q, exc)
    return out


def decide(state: State, questions: Mapping[str, Question], *,
           config: router.Config | None = None, abstain_below: float | None = None
           ) -> dict[str, Answer]:
    """One `Answer` per named question about ``state``, in the order given.

    A question that cannot be answered gets an abstained `Answer` with ``error`` set; the others
    are unaffected.
    """
    cfg = config or router.DEFAULT
    try:
        text = prepare(state)
    except (TypeError, ValueError) as exc:
        return {n: _failed(n, q, exc) for n, q in questions.items()}
    groups: dict[str, list[tuple[str, Question]]] = {}
    out: dict[str, Answer] = {}
    for name, q in questions.items():
        try:
            backend = (router.choose(len(q.options), cfg) if cfg.backend == "auto"
                       else cfg.backend)
        except FAILURES as exc:
            out[name] = _failed(name, q, exc)
            continue
        groups.setdefault(backend, []).append((name, q))
    for backend, items in groups.items():
        out.update(_run(backend, cfg, text, items, abstain_below))
    return {n: out[n] for n in questions}


def parse(kind: str, spec: str) -> tuple[str, Question]:
    """``NAME=question`` (noul), ``NAME=question:a,b`` (choice) or ``NAME=question:lo..hi``
    (score) as a name and a `Question`."""
    name, eq, rest = spec.partition("=")
    name = name.strip()
    if not eq or not name:
        raise ValueError(f"expected NAME=question, got {spec!r}")
    if kind == "noul":
        return name, noul(rest)
    text, colon, tail = rest.rpartition(":")
    if not colon:
        raise ValueError(f"expected NAME=question:{'a,b' if kind == 'choice' else 'lo..hi'}, "
                         f"got {spec!r}")
    if kind == "choice":
        return name, choice(text, [o.strip() for o in tail.split(",")])
    lo, dots, hi = tail.partition("..")
    if not dots:
        raise ValueError(f"a score scale is lo..hi, got {tail!r}")
    return name, score(text, int(lo), int(hi))


def parse_all(noul_specs: list[str], choice_specs: list[str], score_specs: list[str]
              ) -> dict[str, Question]:
    """The questions the three CLI flag lists name; a repeated name is an error."""
    found: dict[str, Question] = {}
    for kind, specs in (("noul", noul_specs), ("choice", choice_specs), ("score", score_specs)):
        for spec in specs:
            name, q = parse(kind, spec)
            if name in found:
                raise ValueError(f"question name {name!r} is used twice")
            found[name] = q
    return found

