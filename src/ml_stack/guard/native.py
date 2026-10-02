"""The model-based guard tier: a small local model, leased through the broker, screens tool
results before the main model reads them and high-impact tool calls before they run.

`screen` returns the interventions to add after the built-in rails, or none when no model is
installed that can be used (the built-in rails then stand alone). Nothing is downloaded: a
candidate counts only when its file is already on this machine and fits in the free memory.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ml_stack import hub
from ml_stack.decide.base import Many, State
from ml_stack.decide.guard import Policy, ToolCallGuard
from ml_stack.decide.logprob import Chat, LogprobDecider
from ml_stack.decide.types import DecideError, Decision, Options
from ml_stack.fleet.sizing import estimate
from ml_stack.guard.judge import SYSTEM, Judge, TextScreen
from ml_stack.guard.policy import SENSITIVE
from ml_stack.interventions import Base, Call, Context, Proceed, Verdict
from ml_stack.serve import broker_wire

__all__ = ["CANDIDATES", "ENV", "CallScreen", "Leased", "pick_model", "screen"]

logger = logging.getLogger("ml_stack.guard")

ENV = "MLSTACK_GUARD_JUDGE"
"""``off`` turns the model-based tier off; a URL names a server to use; any other value names the
model file to lease."""
CANDIDATES = (
    "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
    "Qwen3-VL-4B-Instruct-Q4_K_M.gguf",
    "Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
)
"""Instruction models whose answer to a one-letter question read from log-probabilities has been
measured (`docs/guardrails.md`, `docs/decision-models.md`), smallest first."""
CHECKS = ("destructive", "grounded", "requested")
PURPOSE = "guard"
FAILED = (RuntimeError, OSError, ValueError)
CONTEXT = 4096
COOLDOWN_S = 60.0


def pick_model(candidates: Sequence[str] = CANDIDATES, *,
               room: Callable[[], int | None] = hub.free_memory,
               size: Callable[[str], int] | None = None) -> str:
    """The first of ``candidates`` whose file is on this machine and fits in the free memory
    (unknown room fits), as a path; empty when none does."""
    sizer = size or (lambda path: estimate(path, context=CONTEXT, draft=""))
    free = room()
    for name in candidates:
        found = hub.located(name)
        if found is None:
            continue
        need = sizer(str(found))
        if free is not None and need and need > free:
            continue
        return str(found)
    return ""


class Leased(Many):
    """A `Decider` that leases a server through the broker the first time it is asked.

    After a lease fails it answers with a `DecideError` for ``cooldown`` seconds without trying
    again, so a machine with no room does not make every screened result wait for a lease.
    """

    name = "leased"

    def __init__(self, *, model: str = "", url: str = "", lease_timeout: float = 20.0,
                 request_timeout: float = 20.0, cooldown: float = COOLDOWN_S) -> None:
        self.model, self.url = model, url.rstrip("/")
        self.lease_timeout, self.request_timeout = lease_timeout, request_timeout
        self.cooldown = cooldown
        self.clock: Callable[[], float] = time.monotonic
        self.inner: dict[str, LogprobDecider] = {}
        self.grant: Any = None
        self.down_until = 0.0
        self._lock = threading.Lock()

    def _ready(self, system: str = "") -> LogprobDecider:
        with self._lock:
            if system in self.inner:
                return self.inner[system]
            if self.clock() < self.down_until:
                raise DecideError("the guard's model is not available; retrying later")
            try:
                url = self.url or (str(self.grant.base_url) if self.grant else self._lease())
            except FAILED as exc:
                self.down_until = self.clock() + self.cooldown
                raise DecideError(f"no server for the guard's model: {exc}") from exc
            chat = Chat(url=url, timeout=self.request_timeout, **({"system": system} if system else {}))
            self.inner[system] = LogprobDecider(chat)
            return self.inner[system]

    def asking(self, system: str) -> _View:
        """A view of this lease that asks with ``system`` as the system prompt."""
        return _View(self, system)

    def _lease(self) -> str:
        model = self.model or pick_model()
        if not model:
            raise DecideError("no installed model can serve as the guard's judge")
        self.grant = broker_wire.lease(PURPOSE, [model], spec={"context": CONTEXT},
                                       timeout=self.lease_timeout)
        return str(self.grant.base_url)

    def decide(self, question: str, state: State, options: Options, *,
               descriptions: Mapping[str, str] | None = None,
               abstain_below: float | None = None) -> Decision:
        """`LogprobDecider.decide` on the leased server."""
        return self._ready().decide(question, state, options, descriptions=descriptions,
                                    abstain_below=abstain_below)

    def close(self) -> None:
        """Release the lease, if one is held."""
        with self._lock:
            grant, self.grant, self.inner = self.grant, None, {}
        if grant is not None:
            try:
                broker_wire.release(grant.lease)
            except FAILED as exc:
                logger.debug("guard lease not released: %s", exc)


class _View(Many):
    """`Leased.asking`: the same lease, a different system prompt."""

    name = "leased"

    def __init__(self, parent: Leased, system: str) -> None:
        self.parent, self.system = parent, system

    def decide(self, question: str, state: State, options: Options, *,
               descriptions: Mapping[str, str] | None = None,
               abstain_below: float | None = None) -> Decision:
        """`LogprobDecider.decide` on the parent's server."""
        return self.parent._ready(self.system).decide(
            question, state, options, descriptions=descriptions, abstain_below=abstain_below)

    def close(self) -> None:
        """Release the parent's lease."""
        self.parent.close()


@dataclass
class CallScreen(Base):
    """Asks `ToolCallGuard` about each call to a high-impact tool (``impactful``) before it runs.

    A decider that cannot answer, or answers with low confidence, is a `Confirm`: the person is
    asked rather than the call let through.
    """

    guard: ToolCallGuard
    impactful: Collection[str] = SENSITIVE
    name = "call-judge"

    def before_tool_call(self, call: Call, context: Context) -> Verdict:
        if call.name not in self.impactful:
            return Proceed()
        return self.guard.before_tool_call(call, context)

    def close(self) -> None:
        close = getattr(self.guard.decider, "close", None)
        if close is not None:
            close()


def screen(*, decider: Any = None, judge: Judge | None = None,
           env: Mapping[str, str] | None = None, impactful: Collection[str] = SENSITIVE,
           policy: Policy | None = None) -> list[Any]:
    """The interventions of the model-based tier: a `TextScreen` over tool results and a
    `CallScreen` over high-impact calls. Empty when the environment variable `ENV` says ``off``
    or no installed model fits; a given ``decider`` (or ``judge``) is used as it is."""
    setting = (os.environ if env is None else env).get(ENV, "").strip()
    if setting.lower() in ("off", "0", "no", "false"):
        return []
    if decider is None and judge is None:
        if setting.startswith(("http://", "https://")):
            decider = Leased(url=setting)
        else:
            found = hub.located(setting) if setting else None
            path = str(found) if found else pick_model()
            if not path:
                logger.info("no installed model can serve as the guard's judge; the built-in "
                            "rails stand alone")
                return []
            decider = Leased(model=path)
    if judge is None:
        judge = Judge(decider.asking(SYSTEM) if isinstance(decider, Leased) else decider)
    call_guard = ToolCallGuard(decider if decider is not None else judge.decider,
                               policy=policy or Policy(), checks=CHECKS)
    return [TextScreen(judge), CallScreen(call_guard, impactful)]
