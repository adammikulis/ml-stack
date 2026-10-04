"""``python -m ml_stack.workspace.localcoding NAME``: runs the Codex harness for a coding agent
(detached by `agent start --profile coding`) and keeps the agent's status file."""

from __future__ import annotations

import os
import sys

from ml_stack.workspace import localagent as la, localharness as lh
from ml_stack.workspace.service import Workspace

__all__ = ["run_detached"]


def run_detached(argv: list[str] | None = None) -> int:
    """Launch the harness on the agent's model and report its end in the status file."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: python -m ml_stack.workspace.localcoding NAME\n")
        return 2
    os.environ["ML_STACK_AGENT"] = "1"
    os.environ["ML_STACK_NONINTERACTIVE"] = "1"
    ws, name = Workspace(), la.check_name(args[0])
    agent = la.load(ws, name)
    launch = lh.launcher()
    status = la.Status(ws, name)
    if agent is None or launch is None:
        status.update(state="failed", detail="the Codex harness is not available")
        return 1
    status.update(state="working", detail=f"{lh.CODEX} on {agent.model_name}")
    code = launch(agent.model, agent.role, agent.project, harness=lh.CODEX, name=name,
                  orders_from=list(agent.orders_from))
    status.update(state="stopped" if not code else "failed", detail=f"the harness exited with {code}")
    return int(code or 0)


if __name__ == "__main__":  # pragma: no cover - the detached entry point
    raise SystemExit(run_detached())
