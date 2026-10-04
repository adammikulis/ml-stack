"""What only a person at a terminal may do, and what an agent's tool call may never touch."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.person import AGENT_MARKERS, HumanRequired, require_person

__all__ = ["AGENT_MARKERS", "GRANT_TTL_S", "HumanGrant", "HumanRequired", "agent_may", "mint", "mint_clicked", "mint_pressed", "protect", "require_person"]

GRANT_TTL_S = 120.0
_MINT = object()

_FORBIDDEN = (
    "ml-stack security", "ml-stack-security", "ml_stack.sentinel", "ml_stack/sentinel",
    "ml_stack_sentinel", "sentinel.release", "sentinel.purge", "sentinel/state",
    "ml-stack-log", "ml_stack.activity", "ml_stack/activity", "ml-stack/activity", "ml_stack_activity", "activity.log",
)
_PROTECTED: set[str] = set()
_STATE_DIR = re.compile(r"ml-stack/+sentinel")
_VERBS = re.compile(r"\b(?:release|purge|unquarantine|disable|mode|baseline)\b")


@dataclass(frozen=True, slots=True)
class HumanGrant:
    """Permission for one ``action`` on one ``subject``, good until ``expires``. Built by
    `mint` after a person confirms at a terminal; `check` is what the store calls."""

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
    subject back (``terminal`` is stdin and stdout being terminals, read from the process
    when not given). Refused when stdin or stdout is not a terminal, when the environment
    carries an agent marker, or when the typed text differs."""
    require_person(action, terminal, env)
    if typed(f"type {subject} to {action} it: ").strip() != subject:
        raise HumanRequired(f"{action} not confirmed")
    return HumanGrant(action, subject, time.time() + GRANT_TTL_S, _MINT)


def mint_pressed(action: str, subject: str, *, pressed: Callable[[str], bool],
                 terminal: tuple[bool, bool] | None = None,
                 env: Mapping[str, str] | None = None) -> HumanGrant:
    """A grant after one confirming key instead of the typed id: ``pressed`` is shown what
    will happen and says whether the person pressed the key. The terminal and agent-marker
    checks are the same as `mint`'s, and they run before ``pressed`` is ever called. Used
    for a release, which restores what was held; a purge still goes through `mint`."""
    require_person(action, terminal, env)
    if not pressed(f"{action} {subject}"):
        raise HumanRequired(f"{action} not confirmed")
    return HumanGrant(action, subject, time.time() + GRANT_TTL_S, _MINT)


def mint_clicked(action: str, subject: str, *, answer: str, label: str,
                 env: Mapping[str, str] | None = None) -> HumanGrant:
    """A grant after a person pressed the dialog button ``label``: refused when the
    environment carries an agent marker or when ``answer`` is not exactly ``label``. Only the
    process that put the dialog up calls this, with what the dialog returned."""
    env = os.environ if env is None else env
    marked = [name for name in AGENT_MARKERS if env.get(name)]
    if marked:
        raise HumanRequired(f"{action} is for a person; this process was started by an agent "
                            f"({marked[0]} is set)")
    if not label or answer != label:
        raise HumanRequired(f"{action} not confirmed")
    return HumanGrant(action, subject, time.time() + GRANT_TTL_S, _MINT)


def protect(directory: Path | str) -> None:
    """Add a directory that `agent_may` refuses to let any tool call name."""
    _PROTECTED.add(str(Path(directory).resolve()).lower())
    _PROTECTED.add(str(Path(directory)).lower())


def _flatten(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        text = str(value)
    text = text.replace("\\", "").replace('"', " ").replace("'", " ").replace(",", " ")
    return re.sub(r"\s+", " ", text).lower()


def agent_may(tool: str, arguments: Mapping[str, Any] | None = None) -> str:
    """Why an agent's tool call must not run, or an empty string when sentinel has no
    objection. A call naming a sentinel verb, the sentinel code or its state directory is
    refused whatever the tool."""
    name = tool.lower()
    if "sentinel" in name or name.startswith("security_") or name == "security":
        return f"tool {tool} is a sentinel verb"
    flat = _flatten(arguments or {})
    guarded = {str(home.state("sentinel")).lower(), *_PROTECTED}
    if any(d in flat for d in guarded) or _STATE_DIR.search(flat):
        return "the call names sentinel's state directory"
    for needle in _FORBIDDEN:
        if needle in flat:
            return f"the call names {needle}"
    if "sentinel" in flat and _VERBS.search(flat):
        return "the call asks to change sentinel"
    return ""
