"""What the agent may call: the tool lists, the actions that wait for the person's yes,
and the actions only a person at a terminal can take, each with the command to run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ml_stack import home, mcp
from ml_stack.guard.untrusted import EXTERNAL
from ml_stack.interventions import Base, Call, Context, Deny, Proceed, Verdict
from ml_stack.net.cli import command as security_command
from ml_stack.sentinel.human import agent_may

__all__ = ["CONFIRM", "FENCED", "HUMAN_ONLY", "READ", "HumanOnlyRail",
           "refusal_for", "review_view"]

READ = frozenset({
    "serve_status", "models_find", "models_files", "bench_status", "bench_history", "bench_show",
    "fleet_peers", "setup_look", "doctor", "models_on_disk", "ollama_models", "jobs_status",
    "jobs_wait", "review_view"})
"""Tools that only look. They run without asking."""

CONFIRM: dict[str, str] = {
    "serve_up": "start a model server (takes GPU and memory)",
    "serve_down": "stop a running model server",
    "serve_escalate": "give a running server more slots (takes more memory)",
    "models_fetch": "start a download into the model cache",
    "bench_run": "start a benchmark (a long job on the GPU)",
    "bench_standard": "start the standard benchmark sets (a long job on the GPU)",
    "bench_speed": "start a speed measurement (a long job on the GPU)",
    "bench_compare": "write a comparison file",
    "bench_animate": "render a video and write a file",
}
"""Tools that change something or cost something: each call waits for the person's yes."""

FENCED = EXTERNAL | READ | frozenset(CONFIRM)
"""Tools whose results carry text nobody here wrote (names, logs, model cards, reasons)."""

HUMAN_ONLY: tuple[tuple[str, re.Pattern[str], str], ...] = tuple(
    (what, re.compile(pattern, re.I | re.S), command) for what, pattern, command in (
        ("release or purge something held in quarantine",
         r"\bquarantin\w*\b.{0,60}\b(?:releas\w*|purg\w*|clear\w*|remov\w*|free\w*|delet\w*|restor\w*)"
         r"|\b(?:releas\w*|purg\w*|unquarantin\w*|unblock\w*|free)\b.{0,60}\b(?:quarantin\w*|held)\b"
         r"|^(?:unquarantine|release|purge)\b",
         "ml-stack-security review"),
        ("approve a host for downloads",
         r"\b(?:approv\w*|allow\w*|trust\w*|whitelist\w*|add)\b.{0,40}\b(?:hosts?|domains?)\b"
         r"|\bapprove[-_ ]?host\b",
         "ml-stack-security approve-host HOST"),
        ("mint a human grant",
         r"\b(?:mint\w*|issu\w*|creat\w*|giv\w*|forg\w*|fak\w*)\b.{0,30}\bgrants?\b"
         r"|\bhuman[-_ ]?grants?\b",
         "ml-stack-security review"),
        ("change the sentinel or guard policy",
         r"\b(?:sentinel|guard|rails?|scan[-_ ]?policy|enforce|observe)\b.{0,40}"
         r"\b(?:off|disabl\w*|polic\w*|mode|weaken\w*|relax\w*|loosen\w*|turn\w*|switch\w*|set)\b"
         r"|\b(?:disabl\w*|turn\w*|switch\w*|weaken\w*|relax\w*|loosen\w*|set)\b.{0,30}"
         r"\b(?:sentinel|guard|rails?|scan[-_ ]?policy)\b",
         "ml-stack-security mode, or ml-stack-security scan-policy"),
        ("plant or remove baselines and decoys",
         r"\b(?:baseline|honey\w*|decoys?)\b.{0,30}\b(?:plant\w*|remov\w*|reset\w*|clear\w*)\b",
         "ml-stack-security baseline"),
        ("change the role this session runs under",
         r"\b(?:roles?|permissions?|privileges?)\b.{0,40}\b(?:runner|operator|raise\w*|elevat\w*"
         r"|upgrad\w*|admin|escalat\w*|grant\w*|chang\w*|switch\w*|set)\b"
         r"|\b(?:runner|operator|reader)\b.{0,20}\b(?:roles?|mode)\b",
         "/role NAME, typed at the prompt"),
    ))
"""Actions only a person at a terminal can take: what it is, how it is spelled, the command."""

_TOOL_NAME = re.compile(
    r"quarantin|releas|purg|approv|grant|mint|sentinel|security|baseline|honey|polic|guard|rail|"
    r"unblock|role|permission|privilege|rule|always", re.I)


def refusal_for(text: str) -> tuple[str, str] | None:
    """``(what, command)`` when ``text`` asks for an action only a person can take."""
    for what, pattern, command in HUMAN_ONLY:
        if pattern.search(text):
            return what, command
    return None


def _words(value: Any) -> str:
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _command_for(name: str) -> tuple[str, str]:
    low = name.lower()
    if re.search(r"quarantin|releas|purg|unblock", low):
        return HUMAN_ONLY[0][0], HUMAN_ONLY[0][2]
    if re.search(r"approv|host", low):
        return HUMAN_ONLY[1][0], HUMAN_ONLY[1][2]
    if re.search(r"grant|mint", low):
        return HUMAN_ONLY[2][0], HUMAN_ONLY[2][2]
    if re.search(r"baseline|honey", low):
        return HUMAN_ONLY[4][0], HUMAN_ONLY[4][2]
    if re.search(r"role|permission|privilege|rule|always", low):
        return HUMAN_ONLY[5][0], HUMAN_ONLY[5][2]
    return HUMAN_ONLY[3][0], HUMAN_ONLY[3][2]


def refused(what: str, command: str) -> str:
    """The sentence given back when a person-only action is asked for."""
    return (f"Only a person can {what}. This agent cannot, and no argument changes that. "
            f"Run this in your own terminal: {command}")


class HumanOnlyRail(Base):
    """Refuses a call to a tool named for a person-only action, and any call whose arguments
    name sentinel's code or state, with the command the person runs."""

    name = "human-only"

    def before_tool_call(self, call: Call, context: Context) -> Verdict:
        if _TOOL_NAME.search(call.name) and call.name not in READ | frozenset(CONFIRM):
            what, command = _command_for(call.name)
            return Deny(refused(what, command), self.name)
        if "agent-rules" in _words(call.arguments):
            return Deny(refused("change the always and never rules", "/rules, typed at the prompt"),
                        self.name)
        why = agent_may(call.name, call.arguments)
        if why:
            return Deny(refused("change what sentinel holds or decides", "ml-stack-security review")
                        + f" ({why})", self.name)
        return Proceed()


def _outside_state(value: Any) -> list[str]:
    """Strings in ``value`` that name a path outside ml-stack's state."""
    root = str(home.home().resolve())
    found: list[str] = []
    if isinstance(value, str):
        text = value.strip()
        if (text.startswith(("/", "~")) or ".." in Path(text).parts) \
                and not str(Path(text).expanduser().resolve()).startswith(root):
            found.append(text)
    elif isinstance(value, Mapping):
        for item in value.values():
            found += _outside_state(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found += _outside_state(item)
    return found


def review_view(what: str = "status", limit: int = 20) -> dict[str, Any]:
    """Read-only security views: ``what`` is ``status`` (mode, counts, what is held),
    ``held`` (everything in quarantine), ``events``, ``hosts``, ``downloads`` or
    ``scanners``. Acting on what is held is for the person, in their own terminal."""
    views = {"status": ["status", "--json"], "held": ["review", "--list", "--json"],
             "events": ["events", "--json"], "hosts": ["hosts", "--json"],
             "downloads": ["downloads", "--json", "--limit", str(max(1, min(int(limit), 100)))],
             "scanners": ["scanners", "--json"]}
    if what not in views:
        return {"error": f"what is one of {', '.join(views)}"}
    return mcp._captured(lambda: security_command(views[what]))
