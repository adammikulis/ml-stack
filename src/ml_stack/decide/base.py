"""The `Decider` protocol and the base class the backends share."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ml_stack.decide.calibrate import Calibration
from ml_stack.decide.types import (
    Decision,
    Option,
    Options,
    Stamp,
    clip,
    decision_from,
    options_of,
)

State = str | Mapping[str, Any] | Sequence[Any]


def text_of(state: State) -> str:
    """A state as text: a string stripped, anything else as indented JSON in a stable order."""
    if isinstance(state, str):
        return state.strip()
    return json.dumps(state, indent=2, ensure_ascii=False, default=str)


@dataclass(frozen=True, slots=True)
class Request:
    """One question for `Decider.decide_many`."""

    question: str
    state: State
    options: Options
    descriptions: Mapping[str, str] | None = None
    abstain_below: float | None = None


@dataclass(frozen=True, slots=True)
class Asked:
    """A validated question as a backend receives it: ``state`` is the text, ``raw`` what the
    caller passed."""

    question: str
    state: str
    raw: Any
    options: tuple[Option, ...]


class Decider(Protocol):
    """Chooses one of a named set of options and says how sure it is."""

    name: str

    def decide(self, question: str, state: State, options: Options, *,
               descriptions: Mapping[str, str] | None = None,
               abstain_below: float | None = None) -> Decision: ...

    def decide_many(self, requests: Sequence[Request]) -> list[Decision]: ...

    async def adecide(self, question: str, state: State, options: Options, *,
                      descriptions: Mapping[str, str] | None = None,
                      abstain_below: float | None = None) -> Decision: ...


class Many:
    """`decide_many` and `adecide` for anything with a `decide`."""

    def decide_many(self, requests: Sequence[Request]) -> list[Decision]:
        """One decision per request, in order."""
        return [self.decide(r.question, r.state, r.options, descriptions=r.descriptions,
                            abstain_below=r.abstain_below) for r in requests]

    async def adecide(self, question: str, state: State, options: Options, *,
                      descriptions: Mapping[str, str] | None = None,
                      abstain_below: float | None = None) -> Decision:
        """`decide` on a worker thread."""
        return await asyncio.to_thread(self.decide, question, state, options,
                                       descriptions=descriptions, abstain_below=abstain_below)


class BaseDecider(Many):
    """Validation, timing, calibration and abstention around one method, ``probabilities``.

    A subclass sets ``name`` and ``model`` and implements ``probabilities(asked)``, returning
    non-negative weights in option order and a dict of details. A `Calibration`, when given,
    reshapes them before the decision is made.
    """

    name = "base"
    model = ""
    calibration: Calibration | None = None

    def probabilities(self, asked: Asked) -> tuple[list[float], dict[str, Any]]:
        raise NotImplementedError

    def decide(self, question: str, state: State, options: Options, *,
               descriptions: Mapping[str, str] | None = None,
               abstain_below: float | None = None) -> Decision:
        """The decision for ``question`` about ``state`` among ``options``."""
        opts = options_of(options, descriptions)
        asked = clip(question.strip(), "question")
        text = clip(text_of(state), "state")
        began = time.perf_counter()
        probs, details = self.probabilities(Asked(asked, text, state, opts))
        if self.calibration is not None:
            probs = self.calibration.apply(probs)
        return decision_from(probs, opts, abstain_below=abstain_below, stamp=Stamp(
            self.name, self.model, (time.perf_counter() - began) * 1000.0, details))
