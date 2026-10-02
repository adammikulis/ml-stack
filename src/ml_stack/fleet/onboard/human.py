"""What only a person at a terminal may do: export, rotate or revoke the signing key.

The same rule as sentinel's human grant (`ml_stack.sentinel.human`): a grant is minted only when
stdin and stdout are terminals, the environment carries no agent marker, and the person types
the subject back. It is kept as its own small module because sentinel is a separate branch;
when both are merged this one should delegate to sentinel's.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

__all__ = ["AGENT_MARKERS", "GRANT_TTL_S", "HumanGrant", "HumanRequired", "mint"]

AGENT_MARKERS = ("CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE")
GRANT_TTL_S = 120.0
_MINT = object()


class HumanRequired(PermissionError):
    """An action that only a person at a terminal may take was asked for by something else."""


@dataclass(frozen=True, slots=True)
class HumanGrant:
    """Permission for one ``action`` on one ``subject``, good until ``expires``."""

    action: str
    subject: str
    expires: float
    _token: object

    def check(self, action: str, subject: str, *, now: float | None = None) -> None:
        """Raise `HumanRequired` unless this grant covers ``action`` on ``subject`` now."""
        if self._token is not _MINT:
            raise HumanRequired("not a grant minted by a person")
        if self.action != action or self.subject != subject:
            raise HumanRequired(f"this grant is for {self.action} {self.subject}")
        if (time.time() if now is None else now) > self.expires:
            raise HumanRequired("the grant expired")


def mint(action: str, subject: str, *, typed: Callable[[str], str] = input,
         terminal: tuple[bool, bool] | None = None,
         env: Mapping[str, str] | None = None) -> HumanGrant:
    """A grant for ``action`` on ``subject`` when a person is at a terminal and types the
    subject back. Refused when stdin or stdout is not a terminal, when the environment carries
    an agent marker, or when the typed text differs."""
    env = os.environ if env is None else env
    tty_in, tty_out = terminal or (sys.stdin.isatty(), sys.stdout.isatty())
    marked = [name for name in AGENT_MARKERS if env.get(name)]
    if marked:
        raise HumanRequired(f"{action} is for a person; this process was started by an agent "
                            f"({marked[0]} is set)")
    if not (tty_in and tty_out):
        raise HumanRequired(f"{action} needs a terminal on stdin and stdout")
    if typed(f"type {subject} to {action} it: ").strip() != subject:
        raise HumanRequired(f"{action} not confirmed")
    return HumanGrant(action, subject, time.time() + GRANT_TTL_S, _MINT)
