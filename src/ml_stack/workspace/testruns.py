"""The test runner's view of the workspace: who is running, which project's store, what to tell the board."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.activity import reuse
from ml_stack.workspace import project, tokens
from ml_stack.workspace.identity import TOKEN_ENV, Denied
from ml_stack.workspace.rates import RateLimited
from ml_stack.workspace.screen import Refused

__all__ = ["Acting", "BoardEvents", "acting", "scope", "verified"]

LABEL_ENV = "ML_STACK_WORKSPACE_LABEL"
BOARD_ERRORS = (Denied, Refused, RateLimited, ValueError, OSError, KeyError)


@dataclass
class Acting:
    """An authenticated workspace session: the workspace, its token and the identity it stands for."""

    ws: Any
    token: str
    identity: dict[str, str]


def scope(root: Path) -> str:
    """A short stable name for the project that ``root`` belongs to; ``local`` outside any project."""
    found = project.describe(start=root)
    return hashlib.sha256(found["key"].encode()).hexdigest()[:16] if found.get("key") else "local"


def acting(agent: str = "", label: str = "") -> Acting | None:
    """The session for ``--agent``/``--label`` or the environment, or None when none is configured."""
    from ml_stack.workspace import cli

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


def verified(entry_id: str, where: str, base: Path | None = None) -> dict[str, Any]:
    """The facts of runner entry ``entry_id`` in project scope ``where``; ValueError unless it verifies."""
    found = reuse.find(entry_id, where, base)
    if found is None:
        raise ValueError(f"no runner entry {entry_id[:24]!r} verifies in this project's store")
    return {"id": entry_id, "file": found["file"], "key": found["lookup"], "tree": found["tree"],
            "outcome": found["outcome"], "junit_sha256": found["junit_sha256"], "counts": found["counts"],
            "agent": found["agent"].get("id", ""), "created": found["created"]}


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


def follow_task(seat: Acting, task: str) -> dict[str, Any]:
    """Subscribe the session's agent to a task; runs attached to it with ``--task`` notify its watchers."""
    from ml_stack.workspace.taskboard import TaskBoard

    return TaskBoard(seat.ws).subscribe(seat.token, task)


def evidence(entry_ids: object, base: Path | None = None) -> list[dict[str, Any]]:
    """The verified facts of the runner entries a worker cites, for the current project's store."""
    if not isinstance(entry_ids, list) or len(entry_ids) > 16 or not all(isinstance(i, str) for i in entry_ids):
        raise ValueError("test entries are a list of at most 16 runner entry ids")
    where = scope(Path.cwd())
    return [verified(entry_id, where, base) for entry_id in entry_ids]


class BoardEvents:
    """Posts a run's milestones to the project board and the inboxes that follow them."""

    def __init__(self, seat: Acting, task: str = "") -> None:
        self.seat, self.task = seat, task
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

    def _try(self, what: str, call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except BOARD_ERRORS as error:
            print(f"test: board notice ({what}) not sent: {error}", file=sys.stderr)
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
        seq = self._try("claim", self._post, f"test {Path(file).name} {key[:8]}",
                        f"test-run key={key[:12]} file={file} state=running agent={self.seat.identity['id']}") or 0
        self.threads[key] = seq
        return seq

    def finished(self, file: str, key: str, outcome: str, entry: str) -> None:
        """Reply in the key's thread with how the run ended."""
        if self.threads.get(key):
            tail = "" if outcome == "pass" else " waiters: run it yourself"
            self._try("finish", self._post, f"test {Path(file).name} {outcome}",
                      f"test-result key={key[:12]} file={file} outcome={outcome} entry={entry}{tail}",
                      self.threads[key])

    def waiting(self, file: str, owner: dict) -> None:
        """Follow the thread of the run this request waits on."""
        if owner.get("thread"):
            self._try("subscribe", self.seat.ws.board.subscribe, self.seat.token, "thread",
                      str(owner["thread"]), "inbox", True)

    def wait_failed(self, file: str, owner: dict) -> None:
        """Nothing beyond the thread reply, which the owner posts."""

    def canary_mismatch(self, file: str, detail: str) -> None:
        """Announce a cached pass that failed when re-executed."""
        line = f"test reuse canary mismatch: {file}: {detail}"[:190]
        self._try("canary", lambda: self.seat.ws.announce(self._sender_token(), "blocked", line))

    def job_started(self, job: str, spec: dict) -> int:
        """Open the job's thread on the project board."""
        who = self.seat.identity["id"]
        return self._try("job start", self._post, f"test job {job}",
                         f"test-job job={job} tier={(spec.get('argv') or ['?'])[0]} owner={who} state=running") or 0

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
            sent = self._try("job done", self._post, f"test job {job} {outcome}", body, thread)
            if sent:
                row = self.seat.ws.bus.get(sent)
                listeners = self.seat.ws.board.listeners(row) if row else []
        watchers = self._try("task watchers", self._watchers) or []
        for name in dict.fromkeys([self.seat.identity["id"], *watchers]):
            if name not in listeners:
                self._try("direct", lambda name=name: self.seat.ws.send(
                    self._sender_token(), name, "status", body, subject=f"test job {job} {outcome}"))

    def _watchers(self) -> list[str]:
        from ml_stack.workspace.taskboard import TaskBoard

        return TaskBoard(self.seat.ws).watchers(self.task) if self.task else []
