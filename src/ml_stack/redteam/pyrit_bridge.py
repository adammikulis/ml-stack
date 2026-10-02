"""PyRIT, as ml-stack uses it: a target wrapping any `Responder`, a scorer that reads
objective evidence instead of asking a model, and one call that sends a prompt through
converters and scores what came back.

PyRIT is imported inside the functions that need it, so this module loads without the
``redteam`` extra and `available` says whether the rest will work.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import metadata
from typing import Any

from ml_stack.redteam.targets import Answer, Responder

__all__ = ["Outcome", "Run", "available", "converter", "fire", "initialise", "version"]

#: converter name -> (class in pyrit.converter, keyword arguments); every one is deterministic
CONVERTERS: dict[str, tuple[str, dict[str, Any]]] = {
    "identity": ("", {}),
    "base64": ("Base64Converter", {}),
    "leetspeak": ("LeetspeakConverter", {"deterministic": True}),
    "confusables": ("UnicodeConfusableConverter", {"deterministic": True}),
    "rot13": ("ROT13Converter", {}),
    "zero_width": ("ZeroWidthConverter", {}),
}


def available() -> bool:
    """Whether PyRIT is installed."""
    try:
        metadata.version("pyrit")
    except metadata.PackageNotFoundError:
        return False
    return True


def version() -> str:
    """The installed PyRIT version, or an empty string."""
    return metadata.version("pyrit") if available() else ""


async def initialise() -> None:
    """PyRIT's memory in process memory, with no environment files, no default targets and no
    network."""
    from pyrit.setup import initialize_pyrit_async

    await initialize_pyrit_async("InMemory", load_defaults=False, env_files=[], silent=True)


def converter(name: str) -> Any:
    """The PyRIT converter called ``name`` in `CONVERTERS`, or None for identity."""
    cls, kwargs = CONVERTERS[name]
    if not cls:
        return None
    from pyrit import converter as converters

    return getattr(converters, cls)(**kwargs)


@dataclass(slots=True)
class Run:
    """One attempt's record, filled in as the attack runs: the prompt as sent, the target's
    answer and the seconds it took."""

    sent: str = ""
    answer: Answer | None = None
    seconds: float = 0.0


@cache
def _target_class() -> type:
    from pyrit.models import construct_response_from_request
    from pyrit.prompt_target import PromptTarget

    class ResponderTarget(PromptTarget):
        """A PyRIT target that hands the conversation to a `Responder`."""

        def __init__(self, *, responder: Responder, run: Run, name: str) -> None:
            super().__init__(endpoint=f"local://{name}", model_name=name)
            self._responder = responder
            self._run = run

        async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Any]
                                               ) -> list[Any]:
            messages = [{"role": str(piece.role), "content": piece.converted_value}
                        for message in normalized_conversation
                        for piece in message.message_pieces]
            self._run.sent = messages[-1]["content"]
            started = time.monotonic()
            answer = await self._responder(messages)
            self._run.seconds = time.monotonic() - started
            self._run.answer = answer
            request = normalized_conversation[-1].get_piece()
            return [construct_response_from_request(
                request=request, response_text_pieces=[answer.text or answer.detail or ""])]

        def _validate_request(self, *, normalized_conversation: list[Any]) -> None:
            return None

    return ResponderTarget


@cache
def _scorer_class() -> type:
    from pyrit.models import Score
    from pyrit.score import ScorerPromptValidator
    from pyrit.score.true_false.true_false_scorer import MessageTrueFalseScorer

    class EvidenceScorer(MessageTrueFalseScorer):
        """True when ``evidence()`` says the attacker's goal was reached."""

        _DEFAULT_VALIDATOR = ScorerPromptValidator(supported_data_types=["text"])

        def __init__(self, *, evidence: Callable[[], bool], name: str) -> None:
            self._evidence = evidence
            self._name = name
            super().__init__(validator=self._DEFAULT_VALIDATOR)

        def _build_identifier(self) -> Any:
            return self._create_identifier(params={"evidence": self._name})

        async def _score_piece_async(self, message_piece: Any, *, objective: str | None = None
                                     ) -> list[Any]:
            return [Score(score_value=str(bool(self._evidence())),
                          score_value_description=self._name, score_metadata=None,
                          score_type="true_false", score_category=[self._name],
                          score_rationale="objective evidence", message_piece_id=message_piece.id,
                          scorer_class_identifier=self.get_identifier(), objective=objective)]

    return EvidenceScorer


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one attack came to: whether the evidence says it succeeded, what was sent after
    conversion, and the target's answer."""

    succeeded: bool
    sent: str
    answer: Answer
    seconds: float


async def fire(responder: Responder, prompt: str, *, evidence: Callable[[Answer], bool],
               converters: Sequence[str] = ("identity",), name: str = "target") -> Outcome:
    """Send ``prompt`` to ``responder`` through PyRIT's `PromptSendingAttack`, converted by
    each of ``converters`` in turn, and score the result: ``evidence`` gets the target's
    `Answer` and says whether the attacker's goal was reached."""
    from pyrit.executor.attack import (
        AttackConverterConfig,
        AttackScoringConfig,
        PromptSendingAttack,
    )
    from pyrit.models import AttackOutcome
    from pyrit.prompt_normalizer import ConverterConfiguration

    run = Run()
    chain = [c for c in (converter(n) for n in converters) if c is not None]
    config = (AttackConverterConfig(
        request_converters=ConverterConfiguration.from_converters(converters=chain))
        if chain else AttackConverterConfig())
    attack = PromptSendingAttack(
        objective_target=_target_class()(responder=responder, run=run, name=name),
        attack_converter_config=config,
        attack_scoring_config=AttackScoringConfig(
            objective_scorer=_scorer_class()(
                evidence=lambda: run.answer is not None and evidence(run.answer), name=name)))
    result = await attack.execute_async(objective=prompt)
    return Outcome(result.outcome == AttackOutcome.SUCCESS, run.sent,
                   run.answer or Answer(detail="the target was never reached"), run.seconds)
