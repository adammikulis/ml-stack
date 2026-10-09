"""The rules backend: deterministic checks that pick an option when a pattern, a word or an
allowed value matches."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ml_stack.decide.base import Asked, BaseDecider, Decider, Many, State
from ml_stack.decide.types import Decision, Options, options_of

MAX_SCAN = 100_000


@dataclass(frozen=True, slots=True)
class Rule:
    """Votes for ``choice`` when the state matches.

    ``pattern`` is a regular expression searched in the text, ``contains`` words looked for
    in it, ``one_of`` values the whole text (or ``field`` of a mapping state) may equal. Any
    one match fires the rule. ``field`` limits the text to one key of a mapping state.
    """

    choice: str
    pattern: str = ""
    contains: tuple[str, ...] = ()
    one_of: tuple[str, ...] = ()
    field: str = ""
    weight: float = 1.0


class RulesDecider(BaseDecider):
    """Picks the option whose matching rules weigh most; one-hot on a match.

    With no match the ``default`` option wins when there is one, else the scores are uniform
    and ``details["matched"]`` is False. Case is ignored unless ``case_sensitive``.
    """

    name = "rules"

    def __init__(self, rules: Sequence[Rule], *, default: str | None = None,
                 case_sensitive: bool = False) -> None:
        self.rules = tuple(rules)
        self.default = default
        flags = 0 if case_sensitive else re.IGNORECASE
        self._compiled = [re.compile(r.pattern, flags) if r.pattern else None
                          for r in self.rules]
        self._fold = not case_sensitive
        self.model = f"{len(self.rules)} rules"

    def _text(self, rule: Rule, raw: Any) -> str:
        if rule.field and isinstance(raw, Mapping):
            raw = raw.get(rule.field, "")
        return str(raw)[:MAX_SCAN]

    def fires(self, index: int, raw: Any, text: str) -> bool:
        """Whether rule ``index`` matches the state."""
        rule = self.rules[index]
        body = self._text(rule, raw) if rule.field else text[:MAX_SCAN]
        pattern = self._compiled[index]
        if pattern is not None and pattern.search(body):
            return True
        folded = body.casefold() if self._fold else body
        if any((w.casefold() if self._fold else w) in folded for w in rule.contains):
            return True
        stripped = folded.strip()
        return any(stripped == (v.casefold() if self._fold else v) for v in rule.one_of)

    def probabilities(self, asked: Asked) -> tuple[list[float], dict[str, Any]]:
        names = [o.name for o in asked.options]
        votes = dict.fromkeys(names, 0.0)
        fired: list[int] = []
        for i, rule in enumerate(self.rules):
            if rule.choice in votes and self.fires(i, asked.raw, asked.state):
                votes[rule.choice] += rule.weight
                fired.append(i)
        if fired:
            best = max(names, key=lambda n: (votes[n], -names.index(n)))
            return [1.0 if n == best else 0.0 for n in names], {"matched": True, "rules": fired}
        if self.default in votes:
            return [1.0 if n == self.default else 0.0 for n in names], {"matched": False}
        return [1.0] * len(names), {"matched": False}


class Layered(Many):
    """Asks ``first`` (hard rules); when it matched, that is the answer, else ``then`` decides."""

    name = "layered"

    def __init__(self, first: RulesDecider, then: Decider) -> None:
        self.first, self.then = first, then

    def decide(self, question: str, state: State, options: Options, *,
               descriptions: Mapping[str, str] | None = None,
               abstain_below: float | None = None) -> Decision:
        """The rule's decision if one fired, else the other decider's."""
        opts = options_of(options, descriptions)
        ruled = self.first.decide(question, state, opts, abstain_below=abstain_below)
        if ruled.details.get("matched"):
            return ruled
        return self.then.decide(question, state, opts, abstain_below=abstain_below)
