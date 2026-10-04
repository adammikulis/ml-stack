"""A shell tool for an agent loop: every command runs under the ``bash`` policy."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from typing import Any

from ml_stack.sandbox import policies
from ml_stack.sandbox.policy import AllowUnsandboxed
from ml_stack.sandbox.run import Events, SandboxViolation, run

__all__ = ["SCHEMA", "SandboxedBash"]

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command. It can read the project, write only a scratch "
                       "directory, and has no network.",
        "parameters": {"type": "object", "required": ["command"],
                       "properties": {"command": {"type": "string"}}},
    },
}


class SandboxedBash:
    """The ``bash`` tool. ``(SCHEMA, tool)`` goes to `FunctionTools`; ``close()`` removes the
    scratch directory. A refused operation raises `SandboxViolation`, so the model reads it as a
    tool error."""

    def __init__(self, project: str | os.PathLike[str], *, on_event: Events | None = None,
                 unsandboxed: AllowUnsandboxed | None = None,
                 tainted: Callable[[], bool] = lambda: False) -> None:
        self.project = os.path.realpath(project)
        self.scratch = os.path.realpath(tempfile.mkdtemp(prefix="ml-stack-bash-"))
        self.on_event, self.unsandboxed, self.tainted = on_event, unsandboxed, tainted

    def __call__(self, command: str) -> str:
        policy = policies.bash(self.project, self.scratch)
        if self.tainted():
            policy = policy.without_network()
        result = run([os.path.realpath("/bin/bash"), "-c", command], policy, unsandboxed=self.unsandboxed,
                     on_event=self.on_event, cwd=self.project)
        if result.denials:
            try:
                result.check()
            except SandboxViolation as exc:
                raise SandboxViolation(f"{exc}: {result.stderr.strip()[:300]}") from None
        tail = "" if result.ok else f"\n[exit {result.returncode}{' timed out' if result.timed_out else ''}]"
        return (result.stdout + result.stderr).rstrip() + tail

    @property
    def tool(self) -> tuple[dict[str, Any], Callable[[str], str]]:
        return SCHEMA, self

    def close(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)
