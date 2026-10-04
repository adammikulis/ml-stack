"""The `attach` and `file` commands of `ml-stack-workspace`."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import Any

from ml_stack.command import flag
from ml_stack.workspace import limits
from ml_stack.workspace.files import Attachment, Where, valid_handle
from ml_stack.workspace.screen import Refused
from ml_stack.workspace.service import Workspace

__all__ = ["ATTACH", "FILE", "attach", "file"]

ATTACH = [
    flag("path", help="a file in your worktree, or - to read stdin"),
    flag("--to", required=True, help="a board such as #general, an agent id, or thread:SEQ"),
    flag("--name", default="", help="the name shown (default: the file's own name)"),
    flag("--note", default="", help="one sentence; put detail in the file"),
    flag("--reply-to", type=int, default=0), flag("--derived-from", default="",
                                                   help="a file handle or message number this came from")]
FILE = [
    flag("target", nargs="*", help="a file handle, or: list, search QUERY, delete HANDLE"),
    flag("--meta", action="store_true", help="metadata only (the default)"),
    flag("--text", action="store_true", help="the text, fenced as data and cut short"),
    flag("--out", default="", help="write the bytes to a new file inside your worktree"),
    flag("--limit", type=int, default=0), flag("--all", action="store_true"),
    flag("--board", default=""), flag("--project", default=""), flag("--by", default=""),
    flag("--derived-from", default="")]


def _read(path: str, ws: Workspace) -> tuple[bytes, str]:
    cap = ws.limits.file_bytes + 1
    if path == "-":
        return sys.stdin.buffer.read(cap), ""
    target = Path(path).expanduser()
    base = limits.root().resolve()
    if target.resolve().is_relative_to(base):
        raise Refused("a file inside the workspace's own state is never attached")
    info = target.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise Refused("attach a regular file, not a link or a folder")
    fd = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as handle:
        return handle.read(cap), target.name


def attach(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    """Post PATH (or stdin) as a file message."""
    data, own = _read(args.path, ws)
    return ws.files.attach(token, args.to, data, Attachment(
        name=args.name or own or "stdin.txt", note=args.note, reply_to=args.reply_to,
        derived_from=args.derived_from))


def file(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    """Describe, read, list, search or delete files."""
    first, rest = (args.target[0] if args.target else "list"), args.target[1:]
    where = Where(args.board, args.project, args.by, args.derived_from)
    if first == "list":
        return ws.files.list(token, where, args.limit, args.all)
    if first == "search":
        return ws.files.search(token, " ".join(rest), where, args.limit)
    handle = rest[0] if first == "delete" and rest else first.removeprefix("file:")
    if first == "delete":
        return ws.files.delete(token, handle)
    if not valid_handle(handle) or rest:
        raise ValueError("name one file handle (12 hex characters), or list, search QUERY, delete HANDLE")
    if args.out:
        return ws.files.save(token, handle, args.out, [Path.cwd()])
    if args.text:
        return ws.files.read_text(token, handle, args.limit, args.all)
    return ws.files.meta(token, handle)
