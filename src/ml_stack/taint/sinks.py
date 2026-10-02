"""Privileged tools classified by what they can do, and what makes an argument safe."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = ["HARD", "Arg", "Capability", "Sink", "Sinks", "claude_code", "ml_stack_tools", "sinks_from_mcp"]


class Capability(StrEnum):
    """What a tool can do to the world."""

    READ = "read"
    EXEC = "exec"
    EGRESS = "egress"
    CREDENTIAL = "credential"
    FLEET = "fleet"
    WRITE = "write"
    STATE = "state"


HARD = frozenset({Capability.EXEC, Capability.EGRESS, Capability.CREDENTIAL})
"""Capabilities where a value traced to untrusted text is refused, not asked about."""


@dataclass(frozen=True, slots=True)
class Arg:
    """How one argument of a sink may be vouched for besides being typed by the person.

    ``registry`` names a list the system owns that the value must be in; ``pattern`` is a regular
    expression the whole value must match; ``low`` and ``high`` bound a number; ``validated``
    names a validated extraction the value must have come from; ``inert`` says the argument
    cannot cause an effect."""

    registry: str = ""
    pattern: str = ""
    low: float | None = None
    high: float | None = None
    validated: str = ""
    inert: bool = False


@dataclass(frozen=True, slots=True)
class Sink:
    """A tool: what it can do (``capability``), how its arguments may be vouched for (``args``,
    with ``other`` for any argument not named), and how far its result is trusted (``result``:
    ``untrusted``, ``local`` for a deterministic read of this machine, or ``user`` for the
    person's own words)."""

    capability: Capability
    args: Mapping[str, Arg] = field(default_factory=dict)
    other: Arg = Arg()
    result: str = "untrusted"


@dataclass(slots=True)
class Sinks:
    """The sink for each tool name; a tool not listed is ``unknown``."""

    by_name: dict[str, Sink] = field(default_factory=dict)
    unknown: Sink = Sink(Capability.STATE)

    def get(self, name: str) -> Sink:
        return self.by_name.get(name, self.unknown)

    def known(self, name: str) -> bool:
        return name in self.by_name

    def with_(self, **tools: Sink) -> Sinks:
        """These sinks added to (or replacing) the ones here."""
        return Sinks({**self.by_name, **tools}, self.unknown)

    def with_reads(self, names: Iterable[str], result: str = "untrusted") -> Sinks:
        """The tools ``names`` as read-only."""
        return self.with_(**{n: Sink(Capability.READ, result=result) for n in names})

    def named(self, result: str) -> frozenset[str]:
        """The tools whose result is trusted at the level ``result``."""
        return frozenset(n for n, s in self.by_name.items() if s.result == result)


def sinks_from_mcp(tools: Iterable[Mapping[str, object]]) -> dict[str, Sink]:
    """A sink for each MCP tool from its annotations: ``readOnlyHint`` makes it a read (its
    result is local unless ``openWorldHint`` is true or absent), ``openWorldHint`` on a tool that
    changes things makes it egress, anything else is a state change."""
    out: dict[str, Sink] = {}
    for tool in tools:
        notes = tool.get("annotations")
        notes = notes if isinstance(notes, Mapping) else {}
        if notes.get("readOnlyHint") is True:
            result = "local" if notes.get("openWorldHint") is False else "untrusted"
            out[str(tool.get("name"))] = Sink(Capability.READ, result=result)
        elif notes.get("openWorldHint") is True:
            out[str(tool.get("name"))] = Sink(Capability.EGRESS)
        else:
            out[str(tool.get("name"))] = Sink(Capability.STATE)
    return out


def _port() -> Arg:
    return Arg(low=1024, high=65535)


def ml_stack_tools() -> Sinks:
    """The sinks for the `ml_stack.mcp` tools and the `ml_stack.do` loop's own."""
    reads = ("models_find", "models_files", "ollama_models", "speech_transcribe", "fleet_peers")
    local = ("serve_status", "bench_status", "bench_history", "bench_show", "setup_look",
             "speech_providers", "doctor", "decide", "models_on_disk",
             "jobs_status", "jobs_wait", "plan", "done")
    sinks = Sinks().with_reads(reads).with_reads(local, result="local")
    return sinks.with_(
        ask_user=Sink(Capability.READ, result="user"),
        serve_up=Sink(Capability.FLEET, {
            "model": Arg(registry="models"), "port": _port(),
            "context": Arg(low=0, high=1_048_576), "parallel": Arg(low=1, high=64),
            "escalate": Arg()}, result="local"),
        serve_down=Sink(Capability.FLEET, {"port": _port()}, result="local"),
        serve_escalate=Sink(Capability.FLEET, {"port": _port(), "add": Arg(low=1, high=16)},
                            result="local"),
        models_fetch=Sink(Capability.EGRESS, result="local"),
        fleet_join=Sink(Capability.CREDENTIAL, result="local"),
        speech_say=Sink(Capability.WRITE, result="local"),
        world_make=Sink(Capability.WRITE, {"size": Arg(pattern=r"small|medium|large"),
                                           "seed": Arg(low=0, high=2**31)}, result="local"),
        bench_run=Sink(Capability.EXEC, other=Arg(registry="models"), result="local"),
        **{f"bench_{sub}": Sink(Capability.EXEC, result="local")
           for sub in ("standard", "speed", "compare", "animate", "sweep", "gate")},
        conversation_compact=Sink(Capability.WRITE, result="local"),
    )


def claude_code() -> Sinks:
    """The sinks for the Claude Code tools the SDK harness runs."""
    reads = ("Read", "Grep", "Glob", "WebSearch", "BashOutput", "TodoWrite", "Task")
    return Sinks().with_reads(reads).with_(
        Bash=Sink(Capability.EXEC), Write=Sink(Capability.WRITE), Edit=Sink(Capability.WRITE),
        MultiEdit=Sink(Capability.WRITE), NotebookEdit=Sink(Capability.WRITE),
        WebFetch=Sink(Capability.EGRESS), KillShell=Sink(Capability.STATE))
