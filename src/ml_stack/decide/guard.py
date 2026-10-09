"""`ToolCallGuard`: an intervention that asks a decider about each tool call before it runs."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from ml_stack.decide.base import Decider
from ml_stack.decide.guards.scoperules import violations
from ml_stack.decide.guards.states import (
    QUESTIONS,
    destructive_state,
    grounded_state,
    requested_state,
)
from ml_stack.decide.types import DecideError, Decision
from ml_stack.interventions import Base, Call, Confirm, Context, Deny, Verdict, merge

MAX_OUTPUT = 2000
MAX_STATE = 20_000
"""The longest question text a call is judged on; a longer call is confirmed, never cut (a cut
could hide the part that does the harm)."""

DEFAULT_ACTIONS: Mapping[str, Mapping[str, str]] = {
    "destructive": {"safe": "ok", "reversible": "ok", "destructive": "confirm"},
    "grounded": {"grounded": "ok", "injected": "deny"},
    "requested": {"requested": "ok", "unrequested": "confirm"},
}
"""For each check, what each answer leads to: ``ok``, ``confirm`` or ``deny``."""


@dataclass(frozen=True, slots=True)
class Policy:
    """How a guard turns answers into verdicts.

    ``abstain_below`` is the confidence under which an answer is not trusted and the call is
    confirmed. ``actions`` maps each check's options to ``ok``, ``confirm`` or ``deny``.
    ``scope`` is what a path or host outside the project leads to. ``errors`` is what a
    decider that fails leads to. ``trusted_tools`` skip the learned checks.
    """

    abstain_below: float = 0.8
    actions: Mapping[str, Mapping[str, str]] = field(default_factory=lambda: DEFAULT_ACTIONS)
    scope: str = "confirm"
    errors: str = "confirm"
    trusted_tools: Collection[str] = ()


def last_tool_output(context: Context) -> str:
    """The most recent tool result in the context, clipped; empty if there is none."""
    for message in reversed(context.messages):
        if message.get("role") == "tool":
            return str(message.get("content", ""))[:MAX_OUTPUT]
    return ""


class ToolCallGuard(Base):
    """Checks a tool call for destruction, injection and leaving the project.

    The ``destructive`` and ``grounded`` checks ask ``decider`` and act on its answer through
    ``policy``; an answer below ``policy.abstain_below`` becomes a Confirm. The scope check
    is deterministic and runs when ``project_dir`` is given. The worst verdict of the checks
    is returned, with each check's reason in it. ``last`` holds the decisions of the latest
    call for logging; the call's arguments are never written anywhere by the guard.
    """

    def __init__(self, decider: Decider, *, project_dir: str = "",
                 allowed_hosts: Collection[str] = (), policy: Policy | None = None,
                 checks: Collection[str] = ("destructive", "grounded")) -> None:
        unknown = set(checks) - {"destructive", "grounded", "requested"}
        if unknown:
            raise ValueError(f"unknown checks {sorted(unknown)}")
        self.decider = decider
        self.project_dir = project_dir
        self.allowed_hosts = tuple(allowed_hosts)
        self.policy = policy or Policy()
        self.checks = tuple(checks)
        self.last: dict[str, Decision] = {}

    def _state(self, check: str, call: Call, context: Context) -> str:
        if check == "destructive":
            return destructive_state(context.task, call.name, call.arguments or {})
        make = requested_state if check == "requested" else grounded_state
        return make(context.task, last_tool_output(context), call.name, call.arguments or {})

    def _verdict(self, check: str, answer: Decision, call: Call) -> Verdict | None:
        name = f"{call.name}: {QUESTIONS[check][0]}"
        if answer.abstained:
            return Confirm(f"The guard is unsure ({answer.confidence:.2f} for "
                           f"{answer.choice!r}). {name}", {"check": check, **answer.public()},
                           "tool-call-guard")
        action = self.policy.actions.get(check, {}).get(answer.choice, "ok")
        if action == "deny":
            return Deny(f"{check} check: {answer.choice} ({answer.confidence:.2f}). {name}")
        if action == "confirm":
            return Confirm(f"{check} check: {answer.choice} ({answer.confidence:.2f}). {name}",
                           {"check": check, **answer.public()})
        return None

    def before_tool_call(self, call: Call, context: Context) -> Verdict:
        """The verdict for ``call``."""
        found: list[Verdict] = []
        self.last = {}
        if self.project_dir:
            bad = violations(call.arguments or {}, self.project_dir, self.allowed_hosts)
            if bad:
                reason = f"{call.name} leaves the project: {'; '.join(bad)}"
                found.append(Deny(reason) if self.policy.scope == "deny"
                             else Confirm(reason, {"check": "scope", "violations": bad}))
        if call.name in self.policy.trusted_tools:
            return merge(found)
        for check in self.checks:
            question, options = QUESTIONS[check]
            state = self._state(check, call, context)
            if len(state) > MAX_STATE:
                found.append(Confirm(f"{check} check not run: the call is {len(state)} characters, "
                                     f"too long for the guard to read", {"check": check}))
                continue
            try:
                answer = self.decider.decide(question, state, options,
                                             abstain_below=self.policy.abstain_below)
            except (DecideError, ValueError, OSError) as exc:
                reason = f"{check} check could not run ({exc})"
                found.append(Deny(reason) if self.policy.errors == "deny"
                             else Confirm(reason, {"check": check}))
                continue
            self.last[check] = answer
            verdict = self._verdict(check, answer, call)
            if verdict is not None:
                found.append(verdict)
        return merge(found)
