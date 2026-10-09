"""Why a model lease exists and who took it: what a lease records and how it reads back."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import psutil

from poolhouse.sentinel.redaction import MASK, redact, redact_value

__all__ = [
    "ENV_AGENT",
    "ENV_FOR",
    "NO_REASON",
    "asked",
    "describe",
    "lines",
    "observe",
    "program",
    "record_from",
    "told",
]

ENV_FOR = "POOLHOUSE_LEASE_FOR"
ENV_AGENT = "POOLHOUSE_WORKSPACE_AGENT"
NO_REASON = "(no reason given)"
CHAIN_DEPTH = 4
MAX_TEXT = 200
COMMAND_TEXT = 400


def _line(text: Any) -> str:
    """One line of at most ``MAX_TEXT`` characters, secrets masked."""
    return redact(" ".join(str(text or "").split()))[:MAX_TEXT]


def program() -> str:
    """The entry point this process is running: its module, else the script's name."""
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None and spec.name:
        return spec.name.removesuffix(".__main__")
    return Path(sys.argv[0]).name if sys.argv and sys.argv[0] else ""


def told(reason: str) -> None:
    """Make ``reason`` the default for every lease this process and its children take."""
    if reason:
        os.environ[ENV_FOR] = reason


def _word(word: str, before: str) -> str:
    """One argument with secrets masked: a value after a key-like flag or `Bearer`, a token or
    assigned secret in it. A path is masked a segment at a time."""
    if before.lower() == "bearer":
        return MASK
    flag = before.lstrip("-") if before.startswith("-") and "=" not in before else ""
    if flag:
        return str(redact_value(word, name=flag))
    if word.startswith(("/", "~", ".")) and "=" not in word:
        return "/".join(redact(part) for part in word.split("/"))
    return redact(word)


def _argv(argv: list[str]) -> str:
    """A command line, one line, secrets masked."""
    words = [_word(word, argv[at - 1] if at else "") for at, word in enumerate(argv)]
    return " ".join(" ".join(words).split())[:COMMAND_TEXT]


def _git(cwd: str) -> tuple[str, str]:
    """``(worktree root, branch)`` of the repository holding ``cwd``, or two empty strings.
    Read from ``.git`` and its ``HEAD``; a detached head is its short hash."""
    for root in (Path(cwd), *Path(cwd).parents):
        marker = root / ".git"
        try:
            if marker.is_file():
                gitdir = marker.read_text(encoding="utf-8").removeprefix("gitdir:").strip()
                marker = Path(gitdir) if Path(gitdir).is_absolute() else root / gitdir
            elif not marker.is_dir():
                continue
            head = (marker / "HEAD").read_text(encoding="utf-8").strip()
        except OSError:
            return "", ""
        return str(root), head.removeprefix("ref: refs/heads/") if head.startswith("ref:") else head[:10]
    return "", ""


def describe(pid: int) -> dict[str, Any]:
    """What can be read off process ``pid`` now: its start time, working directory, git
    worktree and branch, and the command lines of it and its parents. Empty when it has gone."""
    try:
        proc = psutil.Process(pid)
        chain = [proc, *proc.parents()][:CHAIN_DEPTH]
        out: dict[str, Any] = {"started": float(proc.create_time()),
                               "chain": [f"{p.pid}: {_argv(_cmdline(p))}" for p in chain]}
        cwd = proc.cwd()
    except psutil.Error:
        return {}
    root, branch = _git(cwd)
    return {**out, "cwd": cwd, "worktree": root, "branch": branch}


def _cmdline(proc: psutil.Process) -> list[str]:
    try:
        return proc.cmdline()
    except psutil.Error:
        return []


def asked(reason: str = "", requester: str = "") -> dict[str, Any]:
    """What this process says about a lease it is about to take, with what it can read of
    itself as a fallback for a broker that cannot read it. ``$POOLHOUSE_LEASE_FOR`` leads the
    reason; ``requester`` falls back to the workspace agent label, then the program."""
    name, said = program(), os.environ.get(ENV_FOR, "")
    why = f"{said} ({reason})" if said and reason and said != reason else reason or said
    return {"reason": _line(why),
            "requester": _line(requester or os.environ.get(ENV_AGENT) or name),
            "program": _line(name), "origin": describe(os.getpid())}


def observe(pid: int, claim: Mapping[str, Any] | None = None, *, behalf: int = 0) -> dict[str, Any]:
    """The record of a lease taken by ``pid``: the broker's own reading of the process
    (``behalf`` when another process asked through ``pid``), under what the asker said."""
    claim = claim or {}
    seen = describe(behalf or pid) or (claim.get("origin") if isinstance(claim.get("origin"), Mapping) else {})
    joined = describe(pid)
    return {"reason": _line(claim.get("reason")) or NO_REASON,
            "requester": _line(claim.get("requester")) or _line(claim.get("program")) or "(unknown)",
            "program": _line(claim.get("program")), "pid": pid,
            "pid_started": joined.get("started"), "taken": time.time(),
            **{k: seen.get(k, "" if k != "chain" else []) for k in ("cwd", "worktree", "branch", "chain")},
            "started": seen.get("started")}


def record_from(entry: Mapping[str, Any]) -> dict[str, Any]:
    """A record read back from disk or the wire, with every field present."""
    return {"reason": str(entry.get("reason") or NO_REASON),
            "requester": str(entry.get("requester") or "(unknown)"),
            "program": str(entry.get("program") or ""), "pid": int(entry.get("pid") or 0),
            "pid_started": entry.get("pid_started"), "taken": float(entry.get("taken") or 0.0),
            "cwd": str(entry.get("cwd") or ""), "worktree": str(entry.get("worktree") or ""),
            "branch": str(entry.get("branch") or ""), "chain": list(entry.get("chain") or []),
            "started": entry.get("started")}


def _age(taken: float, now: float) -> str:
    seconds = max(0, int(now - taken))
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def lines(entry: Mapping[str, Any], *, now: float | None = None, indent: str = "  ") -> list[str]:
    """How a person reads one lease record: why, who, from where, how long ago."""
    one = record_from(entry)
    by = one["requester"] + (f" ({one['program']})" if one["program"] not in ("", one["requester"]) else "")
    where = " on ".join(x for x in (one["worktree"] or one["cwd"], one["branch"]) if x)
    ago = f", {_age(one['taken'], now or time.time())} ago" if one["taken"] else ""
    out = [f"{indent}why      {one['reason']}",
           f"{indent}by       {by}, pid {one['pid']}{ago}"]
    if where:
        out.append(f"{indent}where    {where}")
    for at, parent in enumerate(one["chain"]):
        out.append(f"{indent}{'process ' if at == 0 else 'parent  '} {parent}")
    return out
