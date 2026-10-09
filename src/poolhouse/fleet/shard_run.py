"""Running one accepted shard: a scratch checkout, the test runner on the named tier, a result."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from poolhouse.credentials import child_environment
from poolhouse.platform import start_grouped, terminate_process_group

from . import shard_result, shard_tree

GRACE_S = 30.0
CUT_S = 2.0
"""How long a cancelled run's output is waited for: a child it left behind may hold the pipe open, and the node ends it."""
POLL_S = 1.0
MOST_LOG = 64 << 20
"""The output a runner may write before it is cancelled: a runaway test must not fill the disk."""
RESULT = "result.json"
COMMIT = ["git", "-c", "user.name=shard", "-c", "user.email=shard@localhost", "-c", "core.autocrlf=false",
          "-c", "core.safecrlf=false", "commit", "-q", "-m", "shard"]
DROPPED = ("DEV_TEST_JOB", "DEV_TEST_AGENT", "DEV_TEST_PYTEST_ENDPOINT", "DEV_TEST_PYTEST_TOKEN",
           "DEV_TEST_REMOTE_BROKER", "CLAUDECODE", "AI_AGENT", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ATTENDED",
           "CODEX_THREAD_ID", "CODEX_SESSION_ID", "POOLHOUSE_AGENT", "POOLHOUSE_BOARD", "POOLHOUSE_HOME",
           "POOLHOUSE_NONINTERACTIVE", "POOLHOUSE_SESSION_HARNESS", "POOLHOUSE_SESSION_ID", "POOLHOUSE_WORKSPACE_AGENT",
           "POOLHOUSE_WORKSPACE_DENYLIST", "POOLHOUSE_WORKSPACE_TOKEN", "PYTEST_ADDOPTS", "PYTEST_PLUGINS",
           "PYTEST_CURRENT_TEST", "PYTEST_XDIST_WORKER")
"""What a run never inherits: the sender's say, an agent's markers (the runner behaves differently for an agent)."""


def shard_command(tree: Path, tier: str, files: list[str], junit: Path) -> list[str]:
    """The only command a shard runs: this device's test runner, on this device's Python, for the tier."""
    return [sys.executable, str(tree / "scripts" / "test"), tier, "--no-reuse", f"--junitxml={junit}", *files]


def environment(base: Path) -> dict[str, str]:
    """The child environment: this device's, less secrets and the sender's say, with a private state root."""
    env = child_environment()
    for name in DROPPED:
        env.pop(name, None)
    env.update({"POOLHOUSE_HOME": str(base / "home"), "POOLHOUSE_NO_REAL_KEYSTORE": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def checkout(data: bytes, digest: str, tree: Path) -> None:
    """Unpack the verified tree and give it a git history of one commit, as the runner expects."""
    shard_tree.unpack(data, digest, tree)
    quiet = ["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false"]
    for step in ([*quiet, "init", "-q"], [*quiet, "add", "--all"], COMMIT):
        subprocess.run(step, cwd=tree, check=True, capture_output=True, env=child_environment())


def cpu_seconds() -> float:
    """CPU seconds spent so far by this process's finished children."""
    times = os.times()
    return times.children_user + times.children_system


def stop(proc: subprocess.Popen) -> None:
    """Cancel a runner that overran: an interrupt first, so it cancels its job, then all of it.

    The runner shares this process's group, which the node leads, so the node's own kill of that
    group ends whatever this leaves behind."""
    if sys.platform != "win32":
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(GRACE_S)
            return
        except subprocess.TimeoutExpired:
            pass
    try:
        terminate_process_group(proc, force=True)
    except OSError:
        proc.kill()
    proc.wait()


def keep_output(pipe, out, flooded: threading.Event) -> None:
    """Copy a runner's output into ``out`` up to MOST_LOG bytes, then only drain it, so the disk stays bounded."""
    kept = 0
    for chunk in iter(lambda: pipe.read(65536), b""):
        if kept <= MOST_LOG:
            out.write(chunk[:MOST_LOG + 1 - kept])
            kept += len(chunk)
        if kept > MOST_LOG:
            flooded.set()


@dataclass
class Limit:
    """How long a run may take, and the event that cancels it from outside."""

    seconds: int
    cancel: threading.Event = field(default_factory=threading.Event)


def execute(argv: list[str], tree: Path, env: dict[str, str], log: Path, limit: Limit) -> int:
    """Run ``argv``; its exit status, or 124 when the time limit or the output limit cancelled it, 130 when ``limit.cancel`` was set."""
    flooded = threading.Event()
    with log.open("wb") as out:
        proc = start_grouped(argv, cwd=tree, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
        reader = threading.Thread(target=keep_output, args=(proc.stdout, out, flooded), daemon=True)
        reader.start()
        deadline = time.monotonic() + limit.seconds
        code = None
        while code is None:
            try:
                code = proc.wait(max(0.0, min(POLL_S, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                if limit.cancel.is_set():
                    stop(proc)
                    code = 130
                elif time.monotonic() >= deadline or flooded.is_set():
                    stop(proc)
                    code = 124
        reader.join(CUT_S if code in (124, 130) else GRACE_S)
        return 124 if flooded.is_set() else code


def run_shard(shard: dict, data: bytes, folder: Path,
              command: Callable[[Path, str, list[str], Path], list[str]] = shard_command,
              cancel: threading.Event | None = None) -> dict:
    """Run ``shard`` in a scratch checkout under ``folder`` and write ``result.json`` there."""
    scratch = Path(tempfile.mkdtemp(prefix="shard-", dir=folder))
    began, cpu = time.monotonic(), cpu_seconds()
    tree, junit, log = scratch / "tree", scratch / "junit.xml", scratch / "output.log"
    try:
        checkout(data, shard["tree_sha256"], tree)
        code = execute(command(tree, shard["tier"], shard["files"], junit), tree, environment(scratch), log,
                       Limit(shard["timeout_s"], cancel or threading.Event()))
        tail = log.read_text(errors="replace")[-shard_result.MOST_TAIL:] if log.is_file() else ""
        result = shard_result.build(shard, junit, code, (time.monotonic() - began, cpu_seconds() - cpu), tail)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result = {**shard_result.build(shard, junit, 70, (time.monotonic() - began, 0.0), str(exc)),
                  "state": "failed"}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    (folder / RESULT).write_text(json.dumps(result))
    return result
