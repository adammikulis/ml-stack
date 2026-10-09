"""The note commands of `poolhouse-workspace`: decisions, rules, facts and questions with a trust level each."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from poolhouse.board.session import Note, Session
from poolhouse.command import flag
from poolhouse.workspace import limits, notes, render
from poolhouse.workspace.board_cli import screened
from poolhouse.workspace.identity import Denied
from poolhouse.workspace.screen import fence

__all__ = ["KINDS", "TABLE"]

KINDS = notes.KINDS
OUTPUT_BYTES = notes.OUTPUT_BYTES


def shown(n: Note) -> dict[str, Any]:
    """A note as the commands print it: fenced text, its trust and whether anything replaced it."""
    status = f"superseded by {n.superseded_by.rsplit(':', 1)[-1]}" if n.superseded_by else "current"
    proof = n.last_verified
    if proof:
        status += f"; last run exit {proof['exit']} by {proof['by']}"
    text = fence(f"{n.title}\n{n.body}" if n.title else n.body, f"workspace:note#{n.ref}", f"{n.kind} by {n.author}").text
    return {"id": n.ref, "kind": n.kind, "title": n.title, "author": n.author, "trust": n.trust, "stale": n.stale,
            "status": status, "source": n.source, "tags": n.tags, "ttl_days": n.ttl_days, "verify_cmd": n.verify_cmd,
            "superseded_by": n.superseded_by, "binding": False, "text": text}


def add(args: argparse.Namespace, s: Session) -> Any:
    """Add a note; the node sets its trust level."""
    body = render.body(args.body)
    screened("the note", args.title, body, args.source, args.verify_cmd)
    sent = s.note_add(args.kind, args.title, body, source=args.source, tags=[t for t in args.tags.split(",") if t],
                      supersedes=[r.strip() for r in args.supersedes.split(",") if r.strip()],
                      verify_cmd=args.verify_cmd, ttl_days=int(-(-args.ttl_days // 1)))
    return shown(s.notes(ref=sent["id"])[0])


def search(args: argparse.Namespace, s: Session) -> Any:
    """Notes matching the words."""
    return [shown(n) for n in s.notes(query=args.query, kind=args.kind, everything=args.all, limit=args.limit)]


def get(args: argparse.Namespace, s: Session) -> Any:
    """One note."""
    return shown(s.notes(ref=args.id)[0])


def verify(args: argparse.Namespace, s: Session) -> Any:
    """Run the note's re-derive command if the owner allow-listed it, and record the result."""
    note = s.notes(ref=args.id)[0]
    argv = shlex.split(note.verify_cmd) if note.verify_cmd else []
    if not argv:
        raise ValueError(f"note {args.id} has no re-derive command")
    lim = limits.load(limits.root())
    if not notes.allowed_command(argv, lim.verify_allow):
        raise Denied("that command is not on the owner's verify_allow list in limits.json")
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TMPDIR") if k in os.environ}
    try:
        done = subprocess.run(argv, cwd=Path(args.cwd), env=env, capture_output=True, timeout=lim.verify_timeout_s, check=False)
        code, out = done.returncode, done.stdout + done.stderr
    except subprocess.TimeoutExpired as late:
        code, out = 124, (late.stdout or b"") + b"[timed out]"
    except OSError as err:
        code, out = 127, str(err).encode()
    return shown(s.note_verify(args.id, code, hashlib.sha256(out[:OUTPUT_BYTES]).hexdigest()))


TABLE = (
    ("notes-add", "add a note; the node sets its trust level", [
        flag("kind", choices=KINDS), flag("title"), flag("body", help="- reads stdin"),
        flag("--source", default=""), flag("--tags", default="", help="comma separated"),
        flag("--supersedes", default="", help="comma separated note numbers"),
        flag("--verify-cmd", default="", help="a command that re-derives the fact"),
        flag("--ttl-days", type=float, default=0.0, help="days until it is stale")], add),
    ("notes-search", "notes matching the words", [
        flag("query"), flag("--kind", choices=KINDS, default=""),
        flag("--all", action="store_true", help="include superseded notes"), flag("--limit", type=int, default=10)],
     search),
    ("notes-get", "one note", [flag("id")], get),
    ("notes-verify", "run the note's allow-listed command and record the result",
     [flag("id"), flag("--cwd", default=".")], verify),
)
