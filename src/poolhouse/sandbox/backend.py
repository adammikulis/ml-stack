"""The seam between a policy and an operating system's sandbox: what a backend answers, and how
one is chosen."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from poolhouse.sandbox.policy import Policy, PolicyError

__all__ = ["Availability", "Backend", "SandboxUnavailable", "Wrapped", "program_of"]


class SandboxUnavailable(RuntimeError):
    """No sandbox can hold this policy here, and running without one was not chosen."""


@dataclass(frozen=True, slots=True)
class Availability:
    """Whether a backend can run here, and the sentence saying why not."""

    ok: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Wrapped:
    """The argv that runs the command inside the sandbox, and what a run reads back."""

    argv: list[str]
    tag: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


class Backend(Protocol):
    """One way of confining a process: ``wrap`` turns a command and a policy into the command
    that runs it confined, ``denials`` reads back what the sandbox refused."""

    name: str

    def available(self) -> Availability: ...

    def wrap(self, argv: Sequence[str], policy: Policy) -> Wrapped: ...

    def denials(self, tag: str, since: float) -> list[dict[str, str]]: ...


def program_of(argv: Sequence[str], env_path: str) -> str:
    """The absolute path ``argv[0]`` runs as, searched along ``env_path`` when it has no slash."""
    if not argv or not argv[0]:
        raise PolicyError("no command to run")
    first = argv[0]
    if "/" in first:
        if not first.startswith("/"):
            raise PolicyError(f"command {first!r} is not an absolute path")
        found = first
    else:
        found = shutil.which(first, path=env_path) or ""
        if not found:
            raise PolicyError(f"command {first!r} is not on PATH={env_path!r}")
    return found
