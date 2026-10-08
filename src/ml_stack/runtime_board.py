"""Post runtime deploy outcomes and incidents to the workspace board under the acting agent."""

from __future__ import annotations

import contextlib
import io
import os
import re

from ml_stack import hook_diagnostics
from ml_stack.runtime_deploy import Outcome
from ml_stack.workspace import cli, tokens

COMMIT = re.compile(r"[0-9a-f]{40}")
LIMIT = 190
SHORT = 7


def _text(outcome: Outcome, previous: str, verb: str) -> tuple[str, str]:
    """The announcement kind and one line for an outcome."""
    now = outcome.commit[:SHORT]
    was = previous[:SHORT] if COMMIT.fullmatch(previous) else "none"
    if outcome.action == "switched":
        return "milestone", f"runtime {verb} {now} (was {was})"
    if outcome.action == "recovered":
        return "blocked", f"runtime recovered from a broken selection; {now} wanted, {was} was selected"
    return "blocked", f"runtime {verb} of {now} failed; {was} stays selected"


def incident(outcome: Outcome, stage: str) -> str:
    """Record a redacted hook-diagnostic for a failed or recovered deploy and return its id, or empty."""
    note = hook_diagnostics.record(RuntimeError(outcome.detail or outcome.action), "runtime-deploy", stage,
                                   metadata={"action": outcome.action, "commit": outcome.commit})
    found = re.search(r"diagnostic=([0-9a-f]{32})", note)
    return found[1] if found else ""


def announce(outcome: Outcome, previous: str, *, verb: str = "ensure", agent: str = "", label: str = "") -> bool:
    """Post one announcement for a switch, recovery or failure; True when the board took it."""
    if outcome.action in {"current", "busy", "held"}:
        return False
    who = agent or os.environ.get(tokens.AGENT_ENV, "")
    if not who or not COMMIT.fullmatch(outcome.commit):
        return False
    kind, text = _text(outcome, previous, verb)
    if kind == "blocked":
        found = incident(outcome, verb)
        text = f"{text} [diagnostic={found}]" if found else text
    argv = ["announce", kind, text[:LIMIT], "--agent", who, *(["--label", label] if label else [])]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            return cli.main(argv) == 0
        except SystemExit:
            return False
