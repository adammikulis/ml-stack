"""The test runner's board side: the acting session, and the thread and inbox notices of runs and jobs."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.log import warn
from ml_stack.workspace import cli, tokens
from ml_stack.workspace.identity import TOKEN_ENV, Denied
from ml_stack.workspace.rates import RateLimited
from ml_stack.workspace.screen import Refused

__all__ = ["Acting", "BoardEvents", "acting", "follow_thread", "thread_for_key"]

LABEL_ENV = "ML_STACK_WORKSPACE_LABEL"
BOARD_ERRORS = (Denied, Refused, RateLimited, ValueError, OSError, KeyError)


@dataclass
class Acting:
    """An authenticated workspace session: the workspace, its token and the identity it stands for."""

    ws: Any
    token: str
    identity: dict[str, str]


def acting(agent: str = "", label: str = "") -> Acting | None:
    """The session for ``--agent``/``--label`` or the environment, or None when none is configured."""
    named = agent or os.environ.get(tokens.AGENT_ENV, "")
    if not named and not os.environ.get(TOKEN_ENV):
        return None
    args = argparse.Namespace(agent=named, label=label or os.environ.get(LABEL_ENV, ""), token_file="")
    try:
        ws, token = cli._context(args)
        who = ws.auth(token)
    except (Denied, OSError, ValueError, SystemExit):
        return None
    return Acting(ws, token, {"id": who.id, "label": args.label, "parent": who.parent or "",
                              "source": "workspace-session"})


def thread_for_key(folder: Path, prefix: str) -> int:
    """The board thread of the newest run whose reuse key starts with ``prefix``, or 0."""
    found = sorted((folder / "threads").glob(f"{prefix}*"), key=lambda p: p.stat().st_mtime) \
        if (folder / "threads").is_dir() and prefix.isalnum() else []
    try:
        return int(found[-1].read_text(encoding="utf-8")) if found else 0
    except (OSError, ValueError):
        return 0


def follow_thread(seat: Acting, thread: int) -> dict[str, Any]:
    """Subscribe the session's agent to a run's board thread (one inbox message per update)."""
    return seat.ws.board.subscribe(seat.token, "thread", str(thread), "inbox", True)


class BoardEvents:
    """Posts a run's milestones to the project board and the inboxes that follow them.

    ``watchers`` lists the identities that follow the task a job is attached to.
    """

    def __init__(self, seat: Acting, watchers: Callable[[], list[str]] | None = None) -> None:
        self.seat, self.watchers = seat, watchers
        self.agent = seat.identity
        self.threads: dict[str, int] = {}
        self.board = ""
        self.sender = ""

    def _sender_token(self) -> str:
        """The token notices go out under: the agent's ``test-runner`` delegate, which can send and
        whose messages reach the agent's own inbox; the agent's own token for a delegate or a remote workspace."""
        if not self.sender:
            self.sender = self.seat.token
            who = self.seat.identity
            if not who.get("parent") and hasattr(self.seat.ws, "delegate"):
                child = f"{who['id']}/test-runner"
                try:
                    token = tokens.load(self.seat.ws.base, child)
                    self.seat.ws.auth(token)
                except (Denied, OSError, ValueError):
                    made = self.seat.ws.delegate(self.seat.token, "test-runner", can=("send",))
                    token = tokens.read_file(Path(made["token_file"]))
                self.sender = token
        return self.sender

    @staticmethod
    def _try(what: str, call: Callable[[], Any]) -> Any:
        try:
            return call()
        except BOARD_ERRORS as error:
            warn(f"test: board notice ({what}) not sent: {error}")
            return None

    def _project_board(self) -> str:
        if not self.board:
            found = self.seat.ws.board.list(self.seat.token)
            self.board = next((b["name"] for b in found if b["project"] and b["member"]), "-")
        return self.board

    def _post(self, subject: str, body: str, reply_to: int = 0) -> int:
        board = self._project_board()
        if board == "-":
            return 0
        sent = self.seat.ws.send(self._sender_token(), board, "status", body, subject=subject, reply_to=reply_to)
        return int(sent["seq"])

    def claimed(self, file: str, key: str) -> int:
        """Open the key's thread on the project board."""
        seq = self._try("claim", lambda: self._post(
            f"test {Path(file).name} {key[:8]}",
            f"test-run key={key[:12]} file={file} state=running agent={self.seat.identity['id']}")) or 0
        self.threads[key] = seq
        return seq

    def finished(self, file: str, key: str, outcome: str, entry: str) -> None:
        """Reply in the key's thread with how the run ended."""
        if self.threads.get(key):
            tail = "" if outcome == "pass" else " waiters: run it yourself"
            self._try("finish", lambda: self._post(
                f"test {Path(file).name} {outcome}",
                f"test-result key={key[:12]} file={file} outcome={outcome} entry={entry}{tail}", self.threads[key]))

    def waiting(self, file: str, owner: dict) -> None:
        """Follow the thread of the run this request waits on."""
        if owner.get("thread"):
            self._try("subscribe", lambda: follow_thread(self.seat, int(owner["thread"])))

    def wait_failed(self, file: str, owner: dict) -> None:
        """Nothing beyond the thread reply, which the owner posts."""

    def canary_mismatch(self, file: str, detail: str) -> None:
        """Announce a cached pass that failed when re-executed."""
        line = f"test reuse canary mismatch: {file}: {detail}"[:190]
        self._try("canary", lambda: self.seat.ws.announce(self._sender_token(), "blocked", line))

    def job_started(self, job: str, spec: dict) -> int:
        """Open the job's thread on the project board."""
        who = self.seat.identity["id"]
        return self._try("job start", lambda: self._post(
            f"test job {job}", f"test-job job={job} tier={(spec.get('argv') or ['?'])[0]} owner={who} state=running")) or 0

    def job_done(self, job: str, spec: dict, status: dict, thread: int = 0) -> None:
        """Reply in the job's thread and send one message to the submitter and task watchers."""
        state = status.get("state")
        outcome = "pass" if status.get("exit") == 0 else state if state in ("cancelled", "failed") else "fail"
        summary = status.get("summary", {})
        body = (f"test-job job={job} tier={(spec.get('argv') or ['?'])[0]} outcome={outcome} "
                f"exit={status.get('exit')} ran={summary.get('ran', '?')} reused={summary.get('reused', '?')} "
                f"result=scripts/test result {job}")
        listeners: list[str] = []
        if thread:
            sent = self._try("job done", lambda: self._post(f"test job {job} {outcome}", body, thread))
            row = self.seat.ws.bus.get(sent) if sent else None
            listeners = self.seat.ws.board.listeners(row) if row else []
        followers = (self._try("task watchers", self.watchers) if self.watchers else None) or []
        for name in dict.fromkeys([self.seat.identity["id"], *followers]):
            if name not in listeners:
                self._try("direct", lambda name=name: self.seat.ws.send(
                    self._sender_token(), name, "status", body, subject=f"test job {job} {outcome}"))
