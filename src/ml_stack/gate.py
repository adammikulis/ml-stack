"""One generation at a time per accelerator pool, across every process on this machine.

Servers that ml-stack started may all be up together, but the requests sent to them wait in
line: a request to any server in a pool runs only when every request ahead of it in that
pool has finished. The line is a directory of ticket files under ``<state>/gate/<pool>``.
A ticket is held with an exclusive file lock for as long as its request runs, and the
kernel drops the lock when the process dies, so a crashed holder frees the line without
anybody cleaning up. Tickets are ordered by the time they were taken; the oldest live one
runs.

`turn` is called from `ml_stack.http` for every generation or embedding request, so no
caller asks for it. A URL that is not on a server in the lease registry, or is not on this
machine, is not queued. ``ML_STACK_PARALLEL_REQUESTS=1`` or `parallel` lets requests run
at the same time and logs that it did so.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import sys
import threading
import time
import urllib.parse
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.lock import release, take

__all__ = ["DEFAULT_WAIT_S", "ENV_PARALLEL", "ENV_WAIT", "QueueTimeout", "is_generation",
           "parallel", "pool_of", "snapshot", "turn"]

logger = logging.getLogger(__name__)

ENV_WAIT = "ML_STACK_REQUEST_WAIT_S"
"""Seconds a request waits for its turn before it gives up."""

ENV_PARALLEL = "ML_STACK_PARALLEL_REQUESTS"
"""Set to ``1`` to let requests to one pool run at the same time."""

DEFAULT_WAIT_S = 600.0
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
GENERATING = ("/chat/completions", "/completions", "/completion", "/embeddings",
              "/embedding", "/infill", "/v1/messages")
"""Endpoints that run the model; the last path segments a request is queued on."""

_POLL_S = (0.005, 0.02, 0.05, 0.1)
_held = threading.local()
_parallel: contextvars.ContextVar[str] = contextvars.ContextVar("ml_stack_parallel", default="")
_told: set[str] = set()


class QueueTimeout(TimeoutError):
    """A request waited its whole allowance and was never at the front of the line."""


def is_generation(url: str) -> bool:
    """Whether ``url`` is an endpoint that runs the model."""
    path = urllib.parse.urlsplit(url).path.rstrip("/")
    return path.endswith(GENERATING)


def _registry() -> dict[str, Any]:
    try:
        parsed = json.loads(home.moved("servers.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def pool_of(url: str) -> str:
    """The pool a request to ``url`` queues in, or "" when it is not queued: the host is
    not this machine, or no server in the lease registry listens on that port."""
    parts = urllib.parse.urlsplit(url)
    if (parts.hostname or "").lower() not in LOOPBACK or not parts.port:
        return ""
    for key, entry in _registry().items():
        if not isinstance(entry, dict):
            continue
        try:
            port = int(entry.get("port", key))
        except (TypeError, ValueError):
            continue
        if port == parts.port:
            return str(entry.get("pool") or "gpu")
    return ""


def _wait_s(wait_s: float | None) -> float:
    if wait_s is not None:
        return wait_s
    try:
        return float(os.environ.get(ENV_WAIT) or DEFAULT_WAIT_S)
    except ValueError:
        return DEFAULT_WAIT_S


def _allowed_parallel() -> str:
    """Who asked for parallel requests, or "" when nobody did."""
    named = _parallel.get()
    if named:
        return named
    return f"{ENV_PARALLEL}=1" if os.environ.get(ENV_PARALLEL, "").strip() in ("1", "true", "yes") else ""


@contextmanager
def parallel(reason: str) -> Iterator[None]:
    """Let requests sent inside the block run beside other requests to the same pool.

    ``reason`` names the caller and is logged the first time it is used.
    """
    token = _parallel.set(reason or "parallel")
    try:
        yield
    finally:
        _parallel.reset(token)


def _dir(pool: str) -> Path:
    path = home.state("gate", pool)
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _directory_lock(path: Path) -> Iterator[None]:
    """The short lock that makes taking a ticket and reading the line one step."""
    fd = os.open(path / ".line", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while not take(fd):
            time.sleep(0.002)
        try:
            yield
        finally:
            release(fd)
    finally:
        os.close(fd)


def _tickets(path: Path) -> list[str]:
    return sorted(p.name for p in path.iterdir() if not p.name.startswith("."))


def _ahead(path: Path, mine: str) -> list[str]:
    """The live tickets older than ``mine``; a ticket whose holder has gone is removed."""
    live: list[str] = []
    for name in _tickets(path):
        if name >= mine:
            break
        try:
            fd = os.open(path / name, os.O_RDWR)
        except OSError:
            continue
        try:
            if take(fd):
                release(fd)
                with contextlib.suppress(OSError):
                    (path / name).unlink()
            else:
                live.append(name)
        finally:
            os.close(fd)
    return live


def _describe(path: Path, name: str) -> str:
    try:
        info = json.loads((path / name).read_text(encoding="utf-8"))
        since = time.time() - float(info.get("since") or time.time())
        return (f"pid {info.get('pid')} ({info.get('label') or 'unnamed'}) for "
                f"{since:.0f}s on {info.get('url')}")
    except (OSError, ValueError, TypeError):
        return "another process"


def _take_ticket(path: Path, url: str) -> tuple[str, int]:
    name = f"{time.time_ns():020d}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    with _directory_lock(path):
        fd = os.open(path / name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        take(fd)
        label = " ".join(sys.argv[:2])
        note = json.dumps({"pid": os.getpid(), "url": url, "since": time.time(),
                           "label": label[:80]})
        os.write(fd, note.encode("utf-8"))
    return name, fd


def _drop(path: Path, name: str, fd: int) -> None:
    with contextlib.suppress(OSError):
        (path / name).unlink()
    with contextlib.suppress(OSError):
        release(fd)
    with contextlib.suppress(OSError):
        os.close(fd)


@contextmanager
def turn(url: str, *, wait_s: float | None = None) -> Iterator[None]:
    """Hold the front of the line for the pool ``url`` is in while the block runs.

    A URL that is not queued, a thread that already holds the pool, and a caller inside
    `parallel` run at once. Raises `QueueTimeout` after ``wait_s`` (else ``ML_STACK_REQUEST_WAIT_S``,
    else ten minutes) naming the request that held the front.
    """
    pool = pool_of(url)
    if not pool:
        yield
        return
    reason = _allowed_parallel()
    if reason:
        if reason not in _told:
            _told.add(reason)
            logger.warning("requests to the %s pool run in parallel (%s); they are not queued",
                           pool, reason)
        yield
        return
    mine: set[str] = _held.__dict__.setdefault("pools", set())
    if pool in mine:
        yield
        return

    path = _dir(pool)
    name, fd = _take_ticket(path, url)
    allowance = _wait_s(wait_s)
    began = time.monotonic()
    step = 0
    try:
        while True:
            with _directory_lock(path):
                ahead = _ahead(path, name)
            if not ahead:
                break
            waited = time.monotonic() - began
            if waited >= allowance:
                raise QueueTimeout(
                    f"waited {waited:.0f}s for its turn on the {pool} pool; {len(ahead)} "
                    f"request(s) ahead, the front is {_describe(path, ahead[0])}. "
                    f"{ENV_WAIT} sets how long to wait; {ENV_PARALLEL}=1 sends in parallel")
            time.sleep(_POLL_S[min(step, len(_POLL_S) - 1)])
            step += 1
        mine.add(pool)
        try:
            yield
        finally:
            mine.discard(pool)
    finally:
        _drop(path, name, fd)


def snapshot() -> dict[str, list[dict[str, Any]]]:
    """Each pool's line, oldest first; the first entry with ``running`` true holds the turn."""
    root = home.state("gate")
    out: dict[str, list[dict[str, Any]]] = {}
    if not root.is_dir():
        return out
    for pool in sorted(p for p in root.iterdir() if p.is_dir()):
        rows: list[dict[str, Any]] = []
        for name in _tickets(pool):
            try:
                fd = os.open(pool / name, os.O_RDWR)
            except OSError:
                continue
            try:
                if take(fd):
                    release(fd)
                    continue
                info = json.loads((pool / name).read_text(encoding="utf-8") or "{}")
            except (OSError, ValueError):
                info = {}
            finally:
                os.close(fd)
            rows.append({**info, "ticket": name})
        if rows:
            rows[0]["running"] = True
            out[pool.name] = rows
    return out
