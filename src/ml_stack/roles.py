"""The roles an agent session runs under: which tools each may call and when the person is asked.

``ROLES`` is the table. ``RoleRail`` enforces the current role on every call, ``PlanLedger``
holds the steps of a plan the person said go to, and ``validate`` checks a table.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ml_stack.chatpolicy import _TOOL_NAME, CONFIRM, READ, _outside_state
from ml_stack.interventions import Base, Call, Confirm, Context, Deny, Proceed, Verdict
from ml_stack.rules import Rules, blocked_reason, describe
from ml_stack.taint.ledger import ledger_of

__all__ = ["OWN", "ROLES", "Extension", "PlanLedger", "Role", "RoleRail", "get", "validate"]

OWN = ("ask_user", "plan", "done")
"""The loop's own tools: they reach the person and change nothing."""
GPU_CLASS = frozenset(CONFIRM) | {"jobs_wait"}
"""Tools whose wall time counts against a role's ``max_gpu_seconds``."""
ASKS = ("never", "each", "outside-plan")


@dataclass(frozen=True)
class Role:
    """What a session may call (``tools``, besides the loop's own) and how the person is
    asked: ``each`` acting call asks, ``outside-plan`` asks unless the approved plan names
    the call, ``never`` for a role with nothing that acts."""

    name: str
    summary: str
    tools: frozenset[str]
    asks: str
    max_calls: int
    max_gpu_seconds: float | None = None

    @property
    def acts(self) -> bool:
        return bool(self.tools & frozenset(CONFIRM))


@dataclass(frozen=True)
class Extension:
    """What a feature adds to a session: ``tools`` (``(schema, callable)`` pairs), text for the
    system message from ``context``, the names in ``reads`` that only look (every role may
    call them) and the names in ``asks`` that change something, with what each does (roles
    that act ask first). A name on the person-only floor is refused."""

    tools: Callable[[], Sequence[tuple[dict[str, Any], Callable[..., Any]]]] = lambda: ()
    context: Callable[[], str] = lambda: ""
    reads: frozenset[str] = frozenset()
    asks: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bad = sorted(n for n in (*self.reads, *self.asks) if _TOOL_NAME.search(n) or n in OWN)
        if bad:
            raise ValueError(f"extension tools {bad} are on the person-only floor or the loop's own")

    def allowed(self, role: Role) -> frozenset[str]:
        """The extension's tool names ``role`` may call."""
        return self.reads | (frozenset(self.asks) if role.acts else frozenset())


def _table(*roles: Role) -> Mapping[str, Role]:
    return MappingProxyType({r.name: r for r in roles})


ROLES: Mapping[str, Role] = _table(
    Role("reader", "reads only: nothing is offered that starts, stops, downloads or writes",
         frozenset(READ), "never", max_calls=60, max_gpu_seconds=0.0),
    Role("operator", "reads run; each call that acts asks the person first",
         frozenset(READ) | frozenset(CONFIRM), "each", max_calls=200),
    Role("runner", "runs the calls an approved plan names without asking again; anything else asks",
         frozenset(READ) | frozenset(CONFIRM), "outside-plan", max_calls=200,
         max_gpu_seconds=6 * 3600.0),
)
DEFAULT = "operator"
TASK_DEFAULT = "runner"


def validate(table: Mapping[str, Role] = ROLES) -> None:
    """Raise ``ValueError`` unless every role in ``table`` is sound: its tools are known
    ones, none is on the person-only floor, and its asking policy fits its tools."""
    known = READ | frozenset(CONFIRM)
    for key, role in table.items():
        problems = []
        if key != role.name:
            problems.append(f"filed under {key!r}")
        if role.asks not in ASKS:
            problems.append(f"asks {role.asks!r}, not one of {ASKS}")
        if role.tools - known:
            problems.append(f"unknown tools {sorted(role.tools - known)}")
        if any(_TOOL_NAME.search(name) for name in role.tools):
            problems.append("a tool on the person-only floor")
        if role.acts == (role.asks == "never"):
            problems.append("asking policy does not fit what the tools do")
        if role.max_calls < 1 or (role.max_gpu_seconds or 0.0) < 0:
            problems.append("limits must be positive")
        if problems:
            raise ValueError(f"role {key}: " + "; ".join(problems))


def get(name: str) -> Role:
    """The role called ``name``; ``ValueError`` listing the roles when there is none."""
    if name not in ROLES:
        raise ValueError(f"no role {name!r}: the roles are {', '.join(ROLES)}")
    return ROLES[name]


_STEP = re.compile(r"^\W*([A-Za-z_][A-Za-z0-9_]*)")


def _leaves(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield str(value)


def _names(step: str, leaf: str) -> bool:
    return re.search(rf"(?<![\w.-]){re.escape(leaf)}(?!\w)", step) is not None


class PlanLedger:
    """The steps of plans the person said go to. A step names a tool first; a call is inside
    it when the tool matches and every value the call carries appears in the step. A step
    covers one call."""

    def __init__(self) -> None:
        self.steps: list[str] = []

    def approve(self, steps: Iterable[Any]) -> None:
        self.steps += [str(s) for s in steps]

    def clear(self) -> None:
        self.steps = []

    def take(self, call: Call) -> bool:
        """Whether an approved step covers ``call``; the step is used up when it does."""
        for at, step in enumerate(self.steps):
            named = _STEP.match(step)
            if named and named.group(1) == call.name and call.arguments is not None \
                    and all(_names(step, leaf) for leaf in _leaves(call.arguments)):
                del self.steps[at]
                return True
        return False


class RoleRail(Base):
    """Holds each call to the current ``role``: a tool outside it is denied, one that acts
    asks the person (or runs, for a ``runner`` call inside the approved plan), and a run that
    has read outside text asks whatever the role. The role changes only through
    ``set``, which the model has no tool to reach."""

    name = "role"

    def __init__(self, role: Role, offered: Callable[[], set[str]],
                 plan: PlanLedger | None = None, *, extension: Extension | None = None,
                 rules: Rules | None = None) -> None:
        self.role, self.offered = role, offered
        self.plan = plan or PlanLedger()
        self.extension = extension or Extension()
        self.rules = rules
        self.gpu_seconds = 0.0

    def set(self, role: Role) -> None:
        self.role, self.gpu_seconds = role, 0.0
        self.plan.clear()

    def spent(self, name: str, seconds: float) -> None:
        """Count ``seconds`` of ``name`` against the GPU-time limit."""
        if name in GPU_CLASS:
            self.gpu_seconds += seconds

    def before_tool_call(self, call: Call, context: Context) -> Verdict:
        role = self.role
        if call.name in OWN:
            return Proceed() if call.name in self.offered() else Deny(
                f"{call.name} is not a tool the {role.name} role offers", self.name)
        ext = self.extension
        if call.name not in role.tools | ext.allowed(role) or call.name not in self.offered():
            return Deny(f"{call.name} is not a tool the {role.name} role offers", self.name)
        if call.name in READ or call.name in ext.reads:
            return Proceed()
        does = CONFIRM.get(call.name) or ext.asks[call.name]
        if role.max_gpu_seconds is not None and self.gpu_seconds >= role.max_gpu_seconds \
                and call.name in GPU_CLASS:
            return Deny(f"the {role.name} role has used its {role.max_gpu_seconds:.0f} s "
                        f"of GPU time", self.name)
        args = call.arguments or {}
        outside = _outside_state(args)
        tainted = self._read_outside_text(context)
        rule = self.rules.covers(call.name, call.arguments, role.name, tainted) \
            if self.rules else None
        if rule is not None and rule.verdict == "never":
            self.rules.fire(rule)
            return Deny(f"a rule you set says: {describe(rule)}", self.name)
        if rule is not None and not outside:
            self.rules.fire(rule)
            return Proceed()
        reasons = []
        if outside:
            reasons.append(f"It names paths outside ml-stack's state: {', '.join(outside[:3])}.")
        if role.asks == "outside-plan" and not reasons:
            if tainted:
                reasons.append("The run has read text from outside, so it asks even for a "
                               "call the plan names.")
            elif self.plan.take(call):
                return Proceed()
            else:
                reasons.append("It is not in the plan you approved.")
        elif role.asks == "outside-plan":
            self.plan.take(call)
        no_always = ("memory and extension writes always ask" if call.name in ext.asks else
                     "it names a path outside ml-stack's state" if outside else
                     "the run has read text from outside" if tainted else
                     blocked_reason(call.name, args))
        return Confirm(f"{call.name} will {does}. {' '.join(reasons)}".strip(),
                       {"tool": call.name, "arguments": args, "role": role.name,
                        "always_ok": not no_always, "always_blocked": no_always}, self.name)

    @staticmethod
    def _read_outside_text(context: Context) -> bool:
        return context.tainted or ledger_of(context.notes).contaminated
