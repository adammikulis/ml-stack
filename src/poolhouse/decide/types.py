"""Options, decisions and the errors a decider raises."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

MAX_OPTIONS = 255
MAX_NAME = 200
MAX_TEXT = 200_000


class DecideError(RuntimeError):
    """A decider could not produce a decision."""


class BackendUnavailable(DecideError):
    """A backend's dependency, server or checkpoint is missing; the message says what."""


@dataclass(frozen=True, slots=True)
class Option:
    """One named choice and, optionally, a line saying what it means."""

    name: str
    description: str = ""


Options = Sequence[str] | Mapping[str, str] | Sequence[Option]


def options_of(options: Options, descriptions: Mapping[str, str] | None = None
               ) -> tuple[Option, ...]:
    """Ordered, validated options from names, a ``{name: description}`` map or `Option`s."""
    if isinstance(options, Mapping):
        found = [Option(str(k), str(v or "")) for k, v in options.items()]
    else:
        found = [o if isinstance(o, Option) else Option(str(o)) for o in options]
    if descriptions:
        unknown = set(descriptions) - {o.name for o in found}
        if unknown:
            raise ValueError(f"descriptions for options that are not offered: {sorted(unknown)}")
        found = [Option(o.name, descriptions.get(o.name, o.description)) for o in found]
    names = [o.name for o in found]
    if len(found) < 2:
        raise ValueError("a decision needs at least two options")
    if len(found) > MAX_OPTIONS:
        raise ValueError(f"at most {MAX_OPTIONS} options, got {len(found)}")
    if len(set(names)) != len(names):
        raise ValueError(f"option names must be unique: {names}")
    for o in found:
        if not o.name.strip() or len(o.name) > MAX_NAME:
            raise ValueError(f"an option name is empty or over {MAX_NAME} characters")
    return tuple(found)


def clip(text: str, what: str) -> str:
    """``text`` unchanged, or a ValueError when it is over the size every request is held to."""
    if len(text) > MAX_TEXT:
        raise ValueError(f"{what} is {len(text)} characters; the limit is {MAX_TEXT}")
    return text


@dataclass(frozen=True, slots=True)
class Decision:
    """The answer to one question: the option chosen and a probability for every option.

    ``choice`` is always the most probable option, even when ``abstained`` is set; an
    abstained decision is one the caller should not act on without more evidence.
    """

    choice: str
    scores: Mapping[str, float]
    margin: float
    abstained: bool = False
    latency_ms: float = 0.0
    backend: str = ""
    model: str = ""
    details: Mapping[str, object] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """The probability of the chosen option."""
        return self.scores[self.choice]

    @property
    def certainty(self) -> float:
        """0 for a uniform spread, 1 for all the mass on one option: ``(n*p - 1) / (n - 1)``."""
        n = len(self.scores)
        return (n * self.confidence - 1.0) / (n - 1) if n > 1 else 1.0

    @property
    def entropy(self) -> float:
        """Shannon entropy of the scores in bits."""
        return -sum(p * math.log2(p) for p in self.scores.values() if p > 0.0)

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """The ``k`` most probable options, best first; ties keep the order offered."""
        order = {name: i for i, name in enumerate(self.scores)}
        ranked = sorted(self.scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))
        return ranked[:k]

    def public(self) -> dict[str, object]:
        """A JSON-ready dict."""
        return {"choice": self.choice, "scores": dict(self.scores), "confidence": self.confidence,
                "margin": self.margin, "abstained": self.abstained,
                "latency_ms": round(self.latency_ms, 3), "backend": self.backend,
                "model": self.model, "details": dict(self.details)}


@dataclass(frozen=True, slots=True)
class Stamp:
    """Where a decision came from and what it cost: carried onto the `Decision`."""

    backend: str = ""
    model: str = ""
    latency_ms: float = 0.0
    details: Mapping[str, object] = field(default_factory=dict)


def decision_from(probs: Sequence[float], options: Sequence[Option], *,
                  abstain_below: float | None = None, stamp: Stamp | None = None) -> Decision:
    """A `Decision` from probabilities in option order, normalised and checked."""
    if len(probs) != len(options):
        raise DecideError(f"{len(probs)} probabilities for {len(options)} options")
    total = float(sum(probs))
    if not math.isfinite(total) or total <= 0.0 or any(p < 0 or math.isnan(p) for p in probs):
        raise DecideError(f"probabilities are not a distribution: {list(probs)}")
    scores = {o.name: float(p) / total for o, p in zip(options, probs, strict=True)}
    ranked = sorted(scores.values(), reverse=True)
    best = max(range(len(options)), key=lambda i: (scores[options[i].name], -i))
    choice = options[best].name
    held = abstain_below is not None and scores[choice] < abstain_below
    made = stamp or Stamp()
    return Decision(choice, scores, ranked[0] - ranked[1], held, made.latency_ms, made.backend,
                    made.model, made.details)
