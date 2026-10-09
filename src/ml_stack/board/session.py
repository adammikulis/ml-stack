"""A session on a board: the node's methods for one token, answered as small dataclasses."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.board import credentials, place
from ml_stack.board.client import Client, Conflict, Denied

__all__ = ["AGENT_ENV", "TOKEN_ENV", "Agent", "Claim", "Entry", "Native", "Note", "Page", "Registration", "Session",
           "connect", "find", "register"]

AGENT_ENV = "ML_STACK_WORKSPACE_AGENT"
TOKEN_ENV = "ML_STACK_WORKSPACE_TOKEN"  # noqa: S105 - the name of a variable, not a credential


@dataclass(frozen=True, slots=True)
class Entry:
    """One board entry as the node folded it. ``ref`` is the short id of a message of this device."""

    id: str
    ref: str
    kind: str
    sender: str
    foreign: bool
    channel: str
    fields: dict[str, Any]
    at_ms: int

    @property
    def text(self) -> str:
        return str(self.fields.get("body", ""))


@dataclass(frozen=True, slots=True)
class Page:
    entries: list[Entry]
    cursor: dict[str, int]


@dataclass(frozen=True, slots=True)
class Agent:
    name: str
    parent: str
    family: str
    model: str
    model_state: str
    harness: str
    retired: bool
    seen_ms: int


@dataclass(frozen=True, slots=True)
class Claim:
    kind: str
    key: str
    owner: str
    expires_in_s: int
    expiring_soon: bool
    lease: str
    pid: int
    changed: bool = True


@dataclass(frozen=True, slots=True)
class Note:
    id: str
    ref: str
    kind: str
    title: str
    body: str
    author: str
    source: str
    tags: list[str]
    trust: str
    stale: bool
    ttl_days: int
    supersedes: list[str]
    superseded_by: str
    verify_cmd: str
    last_verified: dict[str, Any] | None
    binding: bool = False


@dataclass(frozen=True, slots=True)
class Native:
    """A session as its harness knows it: the model it says it runs, the harness and its own session id."""

    model: str
    harness: str
    id: str


@dataclass(frozen=True, slots=True)
class Registration:
    name: str
    token: str
    parent: str
    created: bool
    board: str


def _entry(raw: dict[str, Any]) -> Entry:
    seq = str(raw["seq"])
    return Entry(raw["id"], seq if not raw["foreign"] else raw["id"], raw["kind"], raw["sender"], raw["foreign"],
                 raw["channel"], raw["fields"], int(raw["hlc"][0]))


def _agent(raw: dict[str, Any]) -> Agent:
    return Agent(raw["name"], raw["parent"], raw["family"], raw["model"], raw["model_state"], raw["harness"],
                 raw["retired"], raw["seen_ms"])


def _claim(raw: dict[str, Any], changed: bool = True) -> Claim:
    return Claim(raw["kind"], raw["key"], raw["owner"], raw["expires_in_s"], raw["expiring_soon"], raw["lease"],
                 raw["pid"], changed)


def _note(raw: dict[str, Any]) -> Note:
    names = ("id", "ref", "kind", "title", "body", "author", "source", "tags", "trust", "stale", "ttl_days",
             "supersedes", "superseded_by", "verify_cmd", "last_verified")
    return Note(*(raw[n] for n in names))


@dataclass(slots=True)
class Session:
    """The caller's token on one board of one node."""

    client: Client
    board: str
    token: str
    named: str = ""
    clock: Callable[[], float] = time.time

    def _call(self, method: str, **params: Any) -> Any:
        return self.client.call(method, self.board, self.token, **params)

    @property
    def name(self) -> str:
        """The name the node gives this token."""
        if not self.named:
            self.named = str(self._call("whoami")["name"])
        return self.named

    def whoami(self, model: str = "", harness: str = "") -> Agent:
        """Who the token is; ``model`` and ``harness`` record what the session says it runs on."""
        params = {k: v for k, v in (("model", model), ("harness", harness)) if v}
        got = self._call("whoami", **params)
        self.named = got["name"]
        return _agent(got["identity"])

    def agents(self, retired: bool = False) -> list[Agent]:
        return [_agent(a) for a in self._call("agents", retired=retired)["agents"]]

    def retire(self, target: str = "") -> dict[str, Any]:
        """End ``target`` (default this session): its tokens, leases and name."""
        return dict(self._call("retire", target=target))

    def register(self, native: Native) -> Registration:
        """Register a native session as a subagent of this token's owner."""
        return register(self.client, self.board, native, self.token)

    def post(self, to: str, kind: str, body: str, subject: str = "", reply_id: str = "") -> dict[str, Any]:
        """Write a message to ``#general``, ``#announcements`` or one session."""
        fields = {"type": kind, "to": to, "body": body, "subject": subject}
        if reply_id:
            fields["reply_id"] = reply_id
        return dict(self._call("post", kind="message", fields=fields))

    def read(self, since: dict[str, int] | None = None, **filters: Any) -> Page:
        """Entries after ``since``; the page's cursor is the ``since`` of the next read of this question.

        ``filters`` are ``kind``, ``channel``, ``by`` (a sender), ``limit``, ``inbox`` (messages to this
        session) and ``conversation`` (messages between this session and one other)."""
        params = {("with" if k == "conversation" else k): v for k, v in filters.items() if v}
        if since:
            params["since"] = since
        got = self._call("read", **params)
        return Page([_entry(e) for e in got["entries"]], got["cursor"])

    def note_add(self, kind: str, title: str, body: str, **extra: Any) -> dict[str, Any]:
        """Write a note; ``extra`` may hold ``source``, ``tags``, ``supersedes`` (note references),
        ``verify_cmd`` and ``ttl_days``."""
        return dict(self._call("post", kind="note", fields={"nkind": kind, "title": title, "body": body, **extra}))

    def notes(self, ref: str = "", query: str = "", kind: str = "", everything: bool = False, limit: int = 10) -> list[Note]:
        params = {k: v for k, v in (("ref", ref), ("query", query), ("kind", kind), ("all", everything)) if v}
        return [_note(n) for n in self._call("notes", limit=limit, **params)["notes"]]

    def note_verify(self, ref: str, exit_code: int, out_sha: str) -> Note:
        return _note(self._call("note_verify", note=ref, exit=exit_code, out_sha=out_sha)["notes"][0])

    def claim(self, kind: str, key: str, ttl_s: int = 0, pid: int = 0) -> Claim:
        """Take a claim; `Conflict` when another session holds it."""
        params = {k: v for k, v in (("ttl_s", ttl_s), ("pid", pid)) if v}
        try:
            got = self._call("claim", kind=kind, key=key, **params)
        except Denied as err:
            if " is held by " in str(err):
                raise Conflict(err.code, str(err)) from None
            raise
        return _claim(got["claim"], got["changed"])

    def release(self, kind: str, key: str) -> bool:
        return bool(self._call("release", kind=kind, key=key)["released"])

    def claims(self, kind: str = "") -> list[Claim]:
        return [_claim(c) for c in self._call("claims", **({"kind": kind} if kind else {}))["claims"]]

    def renew_claims(self, ttl_s: int) -> list[Claim]:
        """Extend every claim this session holds by ``ttl_s``."""
        mine = [c for c in self.claims() if c.owner == self.name]
        for c in mine:
            self._call("lease_renew", id=c.lease, ttl_s=ttl_s)
        return mine

    def cursor(self, question: str) -> dict[str, int]:
        return credentials.cursors(self.client.state, self.board, self.name).get(question, {})

    def keep(self, question: str, cursor: dict[str, int]) -> None:
        credentials.remember(self.client.state, self.board, self.name, question, cursor)


def register(client: Client, board: str, native: Native, parent_token: str = "") -> Registration:
    """Register a native session on ``board`` (as a subagent of ``parent_token``'s owner when given);
    its token is kept in a private file for later commands."""
    got = client.call("register", board, parent_token, model=native.model, harness=native.harness, session=native.id)
    credentials.store(client.state, board, got["name"], got["token"])
    return Registration(got["name"], got["token"], got["parent"], got["created"], board)


def find(harness: str, native_id: str, *, cwd: Path | None = None, client: Client | None = None) -> str:
    """The name the node gave the native session ``native_id`` of ``harness`` on this directory's board,
    or an empty string when it has none."""
    node = client or Client()
    try:
        return str(node.call("session_lookup", place.resolve(node, cwd), harness=harness, session=native_id)["name"])
    except Denied:
        return ""


def connect(*, token_file: str = "", agent: str = "", board: str = "", cwd: Path | None = None,
            client: Client | None = None) -> Session:
    """The session of the caller: its token from a file, the agent named (or $ML_STACK_WORKSPACE_AGENT),
    or $ML_STACK_WORKSPACE_TOKEN, on the board of the directory it runs in."""
    node = client or Client()
    where = place.resolve(node, cwd, board)
    named = agent or os.environ.get(AGENT_ENV, "")
    if token_file:
        token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
    elif named:
        token = credentials.load(node.state, where, named)
    elif os.environ.get(TOKEN_ENV, "").strip():
        token = os.environ[TOKEN_ENV].strip()
    else:
        raise Denied("denied", "no token: run as a registered session ($ML_STACK_WORKSPACE_AGENT) or give --token-file")
    return Session(node, where, token, named)
