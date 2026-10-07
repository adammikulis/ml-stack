"""Execute claimed canonical tasks with renewable authentication and lease heartbeats."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ml_stack.workspace import tokens
from ml_stack.workspace.identity import Denied


@dataclass(frozen=True)
class Settings:
    stopped: Callable[[], bool] = lambda: False
    heartbeat_s: float = 20.0
    wall_s: float | None = None


def execute(board, identity: str, task_id: str, allocation_id: str, run: Callable,
            **options):
    """Claim one queued task and persist its proposal or actionable blockage."""
    settings = Settings(**options)

    def token():
        return tokens.load(board.ws.base, identity)
    task = board.get(token(), task_id)
    if task['state'] != 'queued':
        raise ValueError('only a queued canonical task may start')
    limits = [value for value in (task['limits'].get('max_wall_s'), settings.wall_s) if value is not None]
    seconds = min(limits) if limits else None
    board.claim(token(), task_id, allocation_id)
    task = board.get(token(), task_id)
    done, cancel = threading.Event(), threading.Event()
    deadline = time.monotonic() + seconds if seconds is not None else None
    failure = []

    def supervise():
        next_heartbeat = time.monotonic()
        while not done.wait(0.05):
            if settings.stopped() or (deadline is not None and time.monotonic() >= deadline):
                failure.append('Worker stopped' if settings.stopped() else 'Task wall time limit reached')
                cancel.set()
                return
            if time.monotonic() >= next_heartbeat:
                try:
                    board.heartbeat(token(), task_id)
                except (Denied, OSError, RuntimeError, ValueError) as error:
                    failure.append(f'Task lease heartbeat failed: {error}')
                    cancel.set()
                    return
                next_heartbeat = time.monotonic() + settings.heartbeat_s

    def checkpoint(value):
        if cancel.is_set():
            raise RuntimeError(failure[0])
        return board.checkpoint(token(), task_id, value)

    watcher = threading.Thread(target=supervise, daemon=True)
    watcher.start()
    try:
        project = Path(task['lease']['resource']['project']).resolve(strict=True)
        proposal = run(task, project, cancel.is_set, checkpoint)
        if failure:
            raise RuntimeError(failure[0])
        return board.submit(token(), task_id, proposal)
    except (Denied, OSError, RuntimeError, ValueError) as error:
        reason = failure[0] if failure else str(error) or type(error).__name__
        return board.block(token(), task_id, reason[:2000])
    finally:
        done.set()
        watcher.join(timeout=1)
