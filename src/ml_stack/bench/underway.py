"""Whether a measurement is underway, what it is, and starting one in the background.

`MEASURING` names the subcommands that put load on the GPU; `measuring_file` and
`measuring_lock_file` are where the run holding the lock writes itself; `measuring` reads
that back, `remember` and `ended` write it, `asking_said` says how a command line will
ask, and `detach` re-runs a command owned by no terminal.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.home_dir()`,
# `bench._parser()` -- so anything patchable is looked up there at call time.
from ml_stack import bench, jobs
from ml_stack.bench.askings import sampling_from
from ml_stack.bench.keep import _commit
from ml_stack.client.chat import Client
from ml_stack.serve.process import pid_exists
from ml_stack.serve.serving import DEFAULT_CACHE, SAMPLERS

__all__ = ["MEASURING", "_last_line", "_locked_by", "_named_in", "asking_said", "detach",
           "ended", "measuring", "measuring_file", "measuring_lock_file", "remember"]


# Which subcommands put load on the GPU, and so must never overlap with each other.
MEASURING = ("run", "sweep", "drafts", "concurrent", "extract", "speed")


def measuring_file() -> Path:
    """Where the run holding the measuring lock writes its pid, argv, log, start time and
    how it is asking."""
    return bench.home_dir() / "measuring.json"


def measuring_lock_file() -> Path:
    """Where the run holding the measuring lock is named, whatever else it wrote."""
    return bench.home_dir() / "measuring.lock"


def _locked_by() -> int | None:
    """The pid written into the measuring lock, or None."""
    try:
        said = measuring_lock_file().read_text(encoding="utf-8").split()
    except OSError:
        return None
    return int(said[-1]) if said and said[-1].isdigit() else None


def measuring() -> dict[str, Any] | None:
    """The measurement still running, or None. Read from `measuring_file`; a record marked
    ended, or one whose pid has gone, is a measurement that finished.

    A live lock with no record of its own is still a measurement, reported with the little
    the lock knows: a machine whose GPU is busy must never read as idle.
    """
    try:
        record = json.loads(measuring_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    if isinstance(record, dict) and not record.get("ended") and pid_exists(record.get("pid")):
        return record
    pid = _locked_by()
    if pid is None or not pid_exists(pid) \
            or (isinstance(record, dict) and record.get("pid") == pid):
        return None
    return {"pid": pid, "argv": [], "log": "", "how": {},
            "started": "", "lock_only": True}


def asking_said(argv: Sequence[str]) -> dict[str, Any]:
    """How a run started with ``argv`` will ask: the sampling, the draft head and its
    depth, the cache type, the thinking budget, the context and the slots."""
    try:
        args = bench._parser().parse_args([a for a in argv if a not in ("--detach", "--no-queue")])
    except SystemExit:
        return {}
    asked = sampling_from(args)
    sampling = dict(Client(**{k: v for k, v in asked.items() if k in SAMPLERS}).sampling)
    if asked.get("n_predict") is not None:
        sampling["n_predict"] = asked["n_predict"]
    named = [*(getattr(args, "serve_draft", []) or []), *(getattr(args, "draft", []) or [])]
    heads = ([] if getattr(args, "no_draft", False)
             else [str(h).rsplit("/", 1)[-1] or "none" for h in named])
    ahead = getattr(args, "n_max", None)
    return {"sampling": sampling, "card": bool(getattr(args, "card", False)),
            "head": ", ".join(heads),
            "head_ahead": ", ".join(str(n) for n in ahead) if isinstance(ahead, list) else ahead,
            "cache_type": str(getattr(args, "serve_kv", "") or "") or DEFAULT_CACHE,
            "reasoning_budget": getattr(args, "reasoning_budget", None),
            "context": int(getattr(args, "context", 0) or 0),
            "slots": max(1, int(getattr(args, "parallel", 1) or 1))}


def remember(argv: Sequence[str], *, pid: int, log: str = "", started: str = "",
             commit: str | None = None) -> dict[str, Any]:
    """Write `measuring_file` for the run named by ``argv``, and return what was written.

    A log already recorded under this pid is kept, so a detached child taking the lock
    does not lose the log its parent opened for it.
    """
    was: dict[str, Any] = {}
    try:
        found = json.loads(measuring_file().read_text(encoding="utf-8"))
        was = found if isinstance(found, dict) and found.get("pid") == pid else {}
    except (OSError, ValueError):
        pass
    record = {"pid": int(pid), "argv": list(argv),
            "log": str(log or was.get("log") or ""),
            "started": started or str(was.get("started") or time.strftime("%FT%T")),
            "commit": _commit() if commit is None else commit,
            "how": asking_said(argv)}
    measuring_file().parent.mkdir(parents=True, exist_ok=True)
    measuring_file().write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def ended() -> None:
    """Mark this process's record finished, so nothing reads it as a live measurement."""
    try:
        record = json.loads(measuring_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(record, dict) or record.get("pid") != os.getpid():
        return
    record["ended"] = time.strftime("%FT%T")
    measuring_file().write_text(json.dumps(record, indent=1), encoding="utf-8")


def _named_in(argv: Sequence[str]) -> str:
    """What to call a detached run's log: its label, or the first model it measures."""
    try:
        args = bench._parser().parse_args(list(argv))
    except SystemExit:
        return "bench"
    named = (getattr(args, "label", "") or next(iter(getattr(args, "serve", []) or []), "")
             or next(iter(getattr(args, "on", []) or []), "").partition("=")[0]
             or getattr(args, "model", "") or "bench")
    return re.sub(r"[^\w.-]+", "-", str(named).rsplit("/", 1)[-1].removesuffix(".gguf"))[:40]


def detach(argv: Sequence[str]) -> Path:
    """Run ``ml-stack-bench argv`` owned by no terminal, and return the log it writes."""
    rest = [a for a in argv if a != "--detach"]
    cmd = next((a for a in rest if a in MEASURING), "bench")
    log = (bench.home_dir() / "logs"
           / f"{cmd}-{_named_in(rest)}-{time.strftime('%Y%m%dT%H%M%S')}.log")
    commit = _commit()
    # `history` reads the log's header back once `measuring.json` has moved on; a second
    # `--detach` queues behind the measuring lock rather than being refused here
    ran = jobs.detach("ml_stack.bench", rest, log=log,
                      lines=[f"commit: {commit}"] if commit else [])
    remember(rest, pid=ran.pid, log=str(ran.log), started=ran.started, commit=commit)
    jobs.record("bench", pid=ran.pid, argv=rest, log=str(ran.log), started=ran.started,
                home=bench.home_dir() / "jobs", refuse_if_alive=False)
    return ran.log


def _last_line(log: Path) -> str:
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return next((ln for ln in reversed(lines) if ln.strip()), "")
