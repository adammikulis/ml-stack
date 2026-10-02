"""The intervention that keeps untrusted content out of the arguments of privileged tools."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ml_stack import log
from ml_stack.interventions import Base, Call, Confirm, Context, Deny, Proceed, Verdict
from ml_stack.taint.events import TaintEvent, emit
from ml_stack.taint.judge import INTENT, Finding, Registries, judge, schema_for
from ml_stack.taint.labels import Label, Level
from ml_stack.taint.ledger import Ledger, ledger_of
from ml_stack.taint.sinks import HARD, Capability, Sinks, ml_stack_tools, sinks_from_mcp

__all__ = ["TaintOff", "TaintRail", "off"]

logger = logging.getLogger("ml_stack.guard")
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class TaintOff:
    """Put in an agent's interventions, leaves taint tracking out of that agent."""

    because: str


def off(because: str) -> TaintOff:
    """A `TaintOff` for ``because``, which is logged and printed; a blank reason raises."""
    if not because.strip():
        raise ValueError("turning taint tracking off needs a because=")
    message = f"guard: taint turned off: {because.strip()}"
    logger.warning(message)
    log.warn(message)
    return TaintOff(because.strip())


class TaintRail(Base):
    """Once untrusted text has entered the context, a call to a tool that can change something
    runs only if each of its arguments is vouched for (see `ml_stack.taint.judge`).

    A value repeating untrusted text sent to an ``exec``, ``egress`` or ``credential`` tool is a
    `Deny`; every other unvouched call is a `Confirm`. For those three capabilities a call with
    no argument the person typed is a `Confirm` as well. ``validated_tools`` maps a tool to the
    name of the validated extraction its result comes from: such a result does not contaminate
    and its values are vouched under that name. ``emit_events`` sends a `TaintEvent` for each
    refusal or question."""

    name = "taint"

    def __init__(self, sinks: Sinks | None = None, *, registries: Registries | None = None,
                 validated_tools: Mapping[str, str] | None = None,
                 emit_events: bool = True) -> None:
        self.sinks = sinks or ml_stack_tools()
        self.registries: dict[str, Callable[[], Iterable[str]]] = dict(registries or {})
        self.validated_tools = dict(validated_tools or {})
        self.emit_events = emit_events
        self.approved: list[str] = []

    def learn(self, tools: Iterable[Mapping[str, Any]]) -> None:
        """Take sinks from the annotations of MCP ``tools`` for the names no sink is set for."""
        known = {n: s for n, s in sinks_from_mcp(tools).items() if not self.sinks.known(n)}
        self.sinks = self.sinks.with_(**known)

    def approve(self, text: str) -> None:
        """Take ``text``, which the person has read and agreed to, as text the person typed."""
        self.approved.append(text)

    def _ledger(self, context: Context) -> Ledger:
        ledger = ledger_of(context.notes)
        ledger.sync(context.messages, task=context.task,
                    trusted_tools=self.sinks.named("user"),
                    local_tools=self.sinks.named("local") | set(self.validated_tools))
        while self.approved:
            ledger.admit(self.approved.pop(), Label(Level.USER, "approved"))
        if context.tainted and not ledger.contaminated:
            ledger.contaminate("flagged")
        return ledger

    def before_model_call(self, context: Context) -> Verdict:
        self._ledger(context)
        return Proceed()

    def after_tool_call(self, call: Call, result: str, context: Context) -> Verdict:
        ledger = self._ledger(context)
        sink = self.sinks.get(call.name)
        if call.name in self.validated_tools:
            ledger.vouch(self.validated_tools[call.name], _strings(result))
        elif sink.result == "user":
            ledger.admit(result, Label(Level.USER, f"tool:{call.name}"))
        elif sink.result == "untrusted" or context.tainted:
            ledger.admit(result, Label(Level.UNTRUSTED, ledger.origin(f"tool:{call.name}")))
        return Proceed()

    def before_tool_call(self, call: Call, context: Context) -> Verdict:
        ledger = self._ledger(context)
        sink = self.sinks.get(call.name)
        if not ledger.contaminated or sink.capability == Capability.READ \
                or call.arguments is None:
            return Proceed()
        found = judge(call.arguments, sink, schema_for(context.tools, call.name), ledger,
                      self.registries)
        unsafe = [f for f in found if f.status != "safe"]
        hard = sink.capability in HARD
        if hard and any(f.status == "proven" for f in unsafe):
            return self._refuse(call, sink.capability, unsafe, ledger, deny=True)
        if unsafe or (hard and not any(f.why in INTENT for f in found)):
            return self._refuse(call, sink.capability, unsafe, ledger, deny=False)
        return Proceed()

    def _refuse(self, call: Call, capability: Capability, unsafe: list[Finding],
                ledger: Ledger, *, deny: bool) -> Verdict:
        origins = list(dict.fromkeys(o for f in unsafe for o in f.origins)) or ledger.flagged[:3]
        names = ", ".join(dict.fromkeys(f.arg for f in unsafe)) or "no argument the person typed"
        rows = [{"argument": f.arg, "status": f.status, "origins": list(f.origins),
                 "preview": f.preview} for f in unsafe]
        if deny:
            verdict: Verdict = Deny(
                f"{call.name} ({capability.value}) was given {names} repeating text from "
                f"{', '.join(origins)}, which the person did not write", self.name)
        else:
            verdict = Confirm(
                f"{call.name} ({capability.value}) would use {names} after the run read "
                f"untrusted text from {', '.join(origins)}", {
                    "tool": call.name, "capability": capability.value, "arguments": rows},
                self.name)
        if self.emit_events:
            emit(TaintEvent(
                "taint.denied" if deny else "taint.confirm", "warning" if deny else "notice",
                f"tool_call:{call.name}", {
                    "capability": capability.value, "origins": origins,
                    "arguments": [{"argument": f.arg, "status": f.status, "digest": f.digest}
                                  for f in unsafe],
                    "flagged": len(ledger.flagged)}))
        return verdict


def _strings(text: str) -> list[str]:
    """Every string in ``text`` read as JSON, or ``text`` itself when it is not."""
    try:
        data = json.loads(text)
    except ValueError:
        return [text]
    out: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif value is not None:
            out.append(str(value))

    walk(data)
    return out
