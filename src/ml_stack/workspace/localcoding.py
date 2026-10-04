"""The maintained coding harness consumes a local agent's authenticated inbox tasks."""
from __future__ import annotations

import os
import sys
import threading
import time

from ml_stack import activity
from ml_stack.fleet.conversations import Conversations
from ml_stack.workspace import localagent as la, localloop, tokens, work_reputation
from ml_stack.workspace.coding_turns import Manager, Turn
from ml_stack.workspace.harness_seat import Seat
from ml_stack.workspace.service import Workspace

__all__ = ["run_detached"]


class BoundManager(Manager):
    def __init__(self, store, ws, identity):
        super().__init__(store)
        self.workspace, self.identity = ws, identity

    def _seat(self, name, folder, parent, say):
        return Seat(self.identity, base=self.workspace.base, managed_inbox=True)


def perform(ws, agent, row, why, stopped):
    """Run one authorized inbox task and return its native harness result."""
    actor = agent.identity or agent.name
    reputation = work_reputation.brief(ws, tokens.load(ws.base, actor))
    la.Status(ws, agent.name).update(reputation=reputation)
    store = Conversations(la.folder(ws) / f"{agent.name}-chats")
    conversation = store.start(model=agent.model, title=f"Workspace task {row['seq']}", settings={
        "mode": "coding", "harness": agent.harness, "role": agent.role,
        "project": agent.project, "context": agent.ctx, "draft": "auto"})
    manager = BoundManager(store, ws, agent.identity or agent.name)
    turn = Turn(conversation.id)
    finished = threading.Event()
    deadline = time.monotonic() + localloop.caps_of(agent).seconds

    def supervise():
        while not finished.wait(0.1):
            if stopped() or time.monotonic() >= deadline:
                turn.cancel()
                break

    watcher = threading.Thread(target=supervise, daemon=True)
    watcher.start()
    refs = {"agent": actor, "workspace_message": str(row["seq"]), "conversation": conversation.id, "project": agent.project}
    activity.record("agent.task", actor=actor, subject=f"Workspace task {row['seq']}",
                    outcome="started", refs=refs, meta={"source": "workspace", "model": agent.model_name})
    try:
        manager._run(turn, conversation, localloop._frame(row, why, reputation))
    finally:
        finished.set()
        watcher.join(timeout=1)
    if not turn.error and not turn.cancelled.is_set() and not turn.text.strip():
        turn.error = "The coding harness ended without an answer"
    activity.record("agent.task", actor=actor, subject=f"Workspace task {row['seq']}",
                    outcome="cancelled" if turn.cancelled.is_set() else "error" if turn.error else "completed",
                    refs=refs, meta={"source": "workspace", "session": turn.session, "model": agent.model_name})
    if turn.error or turn.cancelled.is_set():
        return "status", turn.error or "The coding task was cancelled", 1
    return "answer", turn.text, 1


def run_detached(argv: list[str] | None = None) -> int:
    """Process authorized coding tasks until stopped, retaining the registered agent identity."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: python -m ml_stack.workspace.localcoding NAME\n")
        return 2
    os.environ["ML_STACK_AGENT"] = "1"
    os.environ["ML_STACK_NONINTERACTIVE"] = "1"
    ws, name = Workspace(), la.check_name(args[0])
    return localloop.run(ws, name, localloop.Settings(signals=True,
        serve=lambda _: localloop.Held(None, {}),
        execute=lambda agent, row, why, stopped: perform(ws, agent, row, why, stopped)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_detached())
