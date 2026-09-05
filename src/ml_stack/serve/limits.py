"""How much of this machine ml-stack may take, written down where every part reads it.

A machine says what it *can* do -- `ml_stack.hub.machine_room` reads the wiring limit, the
cores are countable, the ports are free or not. What it *may* do is a person's decision,
and until now there was nowhere to put it: a laptop shared with somebody else's work, a
workstation that must keep memory for a desktop, a machine that will serve two models and
no more.

That decision lives here, in one file (`FILE`), and everything that could exceed it reads
the same record:

* `hub.room` caps the memory a model may use, so every preflight, fit and lease already
  honours it without being told;
* `ServerManager.lease` refuses a server that would be one too many, or a lease asking for
  more seats than allowed, before anything is started;
* `ml_stack.serve.reclaim` stops an idle server once ``idle_s`` has passed.

Nothing here is a default with an opinion: every field starts at 0, which means *no limit
of ours* -- the machine's own answer stands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.records import Document
from ml_stack.units import human_bytes

__all__ = ["Limits", "changed", "clear", "read", "where", "write"]


def where(path: Path | str | None = None) -> Path:
    """The file this machine's limits are kept in. ``MLSTACK_LIMITS_FILE`` moves it."""
    return _DOC.path(path)


@dataclass(frozen=True)
class Limits:
    """What ml-stack may take on this machine. 0 anywhere means no limit of ours."""

    memory_bytes: int = 0
    """The most a model and its caches may use, whatever the machine would allow."""
    servers: int = 0
    """The most model servers to run at once."""
    seats: int = 0
    """The most conversations one server may hold."""
    idle_s: float = 0.0
    """Stop a server that has not been seen busy for this long. 0 leaves it running."""

    def room(self, machine: int) -> int:
        """The memory a model may use here: the machine's answer, capped by ours."""
        if self.memory_bytes <= 0:
            return machine
        return min(self.memory_bytes, machine) if machine > 0 else self.memory_bytes

    def refusal(self, *, running: int = 0, seats: int = 1) -> str:
        """Why a lease may not go ahead, or "" when it may.

        ``running`` is how many servers are already up that this one would join.
        """
        if self.servers and running >= self.servers:
            return (f"this machine is set to run {self.servers} server(s) at once and "
                    f"{running} are up; stop one, or raise the limit "
                    f"(ml-stack-serve limits --servers N)")
        if self.seats and seats > self.seats:
            return (f"this lease asks for {seats} seat(s) and this machine is set to allow "
                    f"{self.seats}; ask for fewer, or raise the limit "
                    f"(ml-stack-serve limits --seats N)")
        return ""

    def said(self) -> list[str]:
        """One line per limit, for a person; empty when nothing is limited."""

        out = []
        if self.memory_bytes:
            out.append(f"memory   a model may use {human_bytes(self.memory_bytes)}")
        if self.servers:
            out.append(f"servers  {self.servers} at once")
        if self.seats:
            out.append(f"seats    {self.seats} on one server")
        if self.idle_s:
            out.append(f"idle     stop a server unused for {self.idle_s:.0f}s")
        return out


_DOC: Document[Limits] = Document(
    default=lambda: home.cache("limits.json"), env="MLSTACK_LIMITS_FILE",
    build=lambda held: Limits(**{f: held[f] for f in Limits.__dataclass_fields__
                                 if f in held}),
    unbuild=asdict, empty=Limits)


def read(path: Path | str | None = None) -> Limits:
    """The limits on disk, or none at all. A file that will not parse is no limits."""
    return _DOC.read(path)


def write(limits: Limits, path: Path | str | None = None) -> Path:
    """Write ``limits`` down and return where they went."""
    return _DOC.write(limits, path)


def clear(path: Path | str | None = None) -> Path:
    """Take every limit off this machine."""
    return write(Limits(), path)


def changed(path: Path | str | None = None, **fields: Any) -> Limits:
    """Write the limits with ``fields`` laid over them, and return what they now are."""
    unknown = set(fields) - set(Limits.__dataclass_fields__)
    if unknown:
        raise ValueError(f"no such limit: {', '.join(sorted(unknown))}")
    now = replace(read(path), **fields)
    write(now, path)
    return now
