"""Put the running host (the daemon Poolside starts) onto the runtime `ml-stack runtime ensure` selected.

The host only ever stops through its own launcher control, the job-preserving replacement `ml-stack --restart` asks for: it
drains requests, checkpoints jobs and closes its stores before it exits. Nothing here signals a process, so a store writer is
never killed; a host that cannot be replaced that way is reported and left running.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

from ml_stack import jobs, runtime
from ml_stack.fleet import launch
from ml_stack.fleet.measuring import same_commit
from ml_stack.runtime_deploy import Outcome

RESTART_WAIT_S = 120.0
POLL_S = 0.5
HEALTH_TRIES = 3
"""A loaded machine can miss one answer; nothing is declared not running on a single miss."""
OLD_HOST = ("this host predates graceful replacement and cannot be restarted from here: "
            "quit and reopen Poolside once; every later deploy restarts it by itself")


def _detach(module: str, argv: Sequence[str], *, log: Path) -> None:
    jobs.detach(module, argv, log=log)


def _health(port: int) -> dict | None:
    for attempt in range(HEALTH_TRIES):
        if attempt:
            time.sleep(POLL_S)
        if (said := launch.already_running(port)) is not None:
            return said
    return None


def _waited(port: int, wanted: str, wait_s: float, health: Callable[[int], dict | None]) -> bool:
    deadline = time.monotonic() + wait_s
    while True:
        said = health(port)
        if said is not None and same_commit(str(said.get("commit") or ""), wanted):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_S)


class Seams(NamedTuple):
    """What restart reaches outside itself for: the detached process it runs and the health probe it asks."""

    spawn: Callable[..., None] = _detach
    health: Callable[[int], dict | None] = _health
    held: Callable[[int], bool] = launch.port_held


def restart(port: int, root: Path, *, wait_s: float = RESTART_WAIT_S, force: bool = False,
            seams: Seams | None = None) -> Outcome:
    """Replace the host on `port` with one running the selected runtime, keeping its port and root.

    The replacement is a detached `ml-stack --restart` run by the selected runtime: it asks the old host to stop at the
    job-preserving boundary, waits for its port to free and starts the new host in the same place. With `wait_s` above 0
    this returns once the host reports the selected commit.
    """
    chosen = runtime.available()
    if chosen is None:
        return Outcome("failed", detail="no verified runtime is selected; run `ml-stack runtime ensure` first")
    spawn, health, held = seams or Seams()
    running = health(port)
    if running is None and held(port):
        return Outcome("failed", chosen.commit, f"something holds port {port} and did not answer /health; retry shortly")
    if running is None:
        return Outcome("not-running", chosen.commit, f"nothing answers on port {port}; the next start runs the selected runtime")
    if not force and same_commit(str(running.get("commit") or ""), chosen.commit):
        return Outcome("current", chosen.commit, f"the host on port {port} already runs it")
    if not running.get("launcher_control"):
        return Outcome("failed", chosen.commit, OLD_HOST)
    spawn("ml_stack.fleet.launch", ["--restart", "--no-browser", "--port", str(port), "--root", str(root)],
          log=runtime.directory() / "restart.log")
    if wait_s <= 0:
        return Outcome("restarting", chosen.commit, f"replacing the host on port {port}; log: {runtime.directory() / 'restart.log'}")
    if _waited(port, chosen.commit, wait_s, health):
        return Outcome("restarted", chosen.commit, f"the host on port {port} runs it")
    return Outcome("failed", chosen.commit, f"the host on port {port} still runs another commit after {wait_s:.0f}s; see restart.log")
