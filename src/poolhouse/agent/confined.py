"""An MCP server started under the sandbox: the launch command and the scratch directory it
writes to."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from poolhouse import sandbox, sentinel
from poolhouse.sandbox import policies
from poolhouse.sentinel.adapters import sandbox_listener

__all__ = ["Launch", "Options", "confine"]


@dataclass(slots=True)
class Launch:
    """What to execute for the server, its whole environment (None: inherit), and the call that
    removes its scratch directory."""

    command: str
    args: list[str]
    env: dict[str, str] | None
    cleanup: Callable[[], None]


@dataclass(frozen=True, slots=True)
class Options:
    """How to confine a server: a ``policy`` (default: ``mcp_server`` for ``project``), extra
    ``reads``, and the named reason to run without a sandbox when none is available."""

    policy: sandbox.Policy | None = None
    project: str | None = None
    reads: Sequence[str] = ()
    unsandboxed: sandbox.AllowUnsandboxed | None = None


def confine(command: str, args: Sequence[str], env: Mapping[str, str] | None,
            options: Options | None = None) -> Launch:
    """The launch of ``command args`` under ``options``. Raises `SandboxUnavailable`, after reporting it, when no sandbox can hold
    the policy and ``unsandboxed`` does not name a reason to go without."""
    program = shutil.which(command) or command
    scratch = os.path.realpath(tempfile.mkdtemp(prefix="poolhouse-mcp-"))

    def cleanup() -> None:
        shutil.rmtree(scratch, ignore_errors=True)

    options = options or Options()
    unsandboxed = options.unsandboxed
    held = options.policy or policies.mcp_server(program, options.project or Path.cwd(), scratch,
                                                 env=env, reads=options.reads)
    report = sandbox_listener(sentinel.default())
    try:
        argv, _tag, chosen = sandbox.wrapped([program, *args], held, unsandboxed=unsandboxed)
    except sandbox.SandboxUnavailable as exc:
        cleanup()
        report("sandbox.unavailable", {"severity": "warning", "command": command,
                                       "policy": held.name, "reason": str(exc)})
        raise
    except sandbox.PolicyError:
        cleanup()
        raise
    if chosen is None:
        report("sandbox.unsandboxed", {"severity": "warning", "command": command,
                                       "policy": held.name,
                                       "reason": unsandboxed.reason if unsandboxed else ""})
        return Launch(program, list(args),
                      sentinel.default().scrub_env(env) if env else None, cleanup)
    return Launch(argv[0], argv[1:], sentinel.default().scrub_env(held.env), cleanup)
