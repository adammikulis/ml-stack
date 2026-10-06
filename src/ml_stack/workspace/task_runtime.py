"""Execute claimed canonical tasks with renewable authentication and lease heartbeats."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ml_stack.workspace import localloop, resource_allocations, tokens
from ml_stack.workspace.identity import Denied


@dataclass(frozen=True)
class Settings:
    stopped: Callable[[], bool] = lambda: False
    heartbeat_s: float = 20.0


def execute(board, identity: str, task_id: str, allocation_id: str, run: Callable,
            **options):
    """Claim one queued task and persist its proposal or actionable blockage."""
    settings = Settings(**options)

    def token():
        return tokens.load(board.ws.base, identity)
    task = board.get(token(), task_id)
    if task['state'] != 'queued':
        raise ValueError('only a queued canonical task may start')
    runner = resource_allocations._worker(board.ws, identity)
    seconds = min(task['limits'].get('max_wall_s', 3600), localloop.caps_of(runner).seconds)
    board.claim(token(), task_id, allocation_id)
    task = board.get(token(), task_id)
    done, cancel = threading.Event(), threading.Event()
    deadline = time.monotonic() + seconds
    failure = []

    def supervise():
        next_heartbeat = time.monotonic()
        while not done.wait(0.05):
            if settings.stopped() or time.monotonic() >= deadline:
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
