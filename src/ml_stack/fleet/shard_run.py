"""Running one accepted shard: a scratch checkout, `scripts/test all` on the named files, a result."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from ml_stack.credentials import child_environment
from ml_stack.platform import start_process

from . import shard_result, shard_tree

GRACE_S = 30.0
COMMIT = ["git", "-c", "user.name=shard", "-c", "user.email=shard@localhost", "commit", "-q", "-m", "shard"]
DROPPED = ("DEV_TEST_JOB", "DEV_TEST_AGENT", "DEV_TEST_PYTEST_ENDPOINT", "DEV_TEST_PYTEST_TOKEN",
           "DEV_TEST_REMOTE_BROKER", "CLAUDECODE", "ML_STACK_HOME", "ML_STACK_NONINTERACTIVE",
           "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST", "PYTEST_XDIST_WORKER")


def shard_command(tree: Path, files: list[str], junit: Path) -> list[str]:
    """The only command a shard runs: this device's test runner on the named files."""
    return [sys.executable, str(tree / "scripts" / "test"), "all", "--no-reuse",
            f"--junitxml={junit}", *files]


def environment(base: Path) -> dict[str, str]:
    """The child environment: this device's, less secrets and the sender's say, with a private state root."""
    env = child_environment()
    for name in DROPPED:
        env.pop(name, None)
    env.update({"ML_STACK_HOME": str(base / "home"), "ML_STACK_NO_REAL_KEYSTORE": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def checkout(data: bytes, digest: str, tree: Path) -> None:
    """Unpack the verified tree and give it a git history of one commit, as the runner expects."""
    shard_tree.unpack(data, digest, tree)
    for step in (["git", "init", "-q"], ["git", "add", "--all"], COMMIT):
        subprocess.run(step, cwd=tree, check=True, capture_output=True, env=child_environment())


def cpu_seconds() -> float:
    """CPU seconds spent so far by this process's finished children."""
    times = os.times()
    return times.children_user + times.children_system


def stop(proc: subprocess.Popen) -> None:
    """Cancel a runner that overran: an interrupt first, so it cancels its job, then the group."""
    if sys.platform != "win32":
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(GRACE_S)
            return
        except subprocess.TimeoutExpired:
            pass
    proc.kill()
    proc.wait()


def execute(argv: list[str], tree: Path, env: dict[str, str], seconds: int, log: Path) -> int:
    """Run ``argv``; its exit status, or 124 when the time limit cancelled it."""
    with log.open("wb") as out:
        proc = start_process(argv, cwd=tree, env=env, stdout=out, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
        try:
            return proc.wait(seconds)
        except subprocess.TimeoutExpired:
            stop(proc)
            return 124


def run_shard(shard: dict, data: bytes, folder: Path,
              command: Callable[[Path, list[str], Path], list[str]] = shard_command) -> dict:
    """Run ``shard`` in a scratch checkout under ``folder`` and write ``result.json`` there."""
    scratch = Path(tempfile.mkdtemp(prefix="shard-", dir=folder))
    began, cpu = time.monotonic(), cpu_seconds()
    tree, junit, log = scratch / "tree", scratch / "junit.xml", scratch / "output.log"
    try:
        checkout(data, shard["tree_sha256"], tree)
        code = execute(command(tree, shard["files"], junit), tree, environment(scratch),
                       shard["timeout_s"], log)
        tail = log.read_text(errors="replace")[-shard_result.MOST_TAIL:] if log.is_file() else ""
        result = shard_result.build(shard, junit, code, (time.monotonic() - began, cpu_seconds() - cpu), tail)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result = {**shard_result.build(shard, junit, 70, (time.monotonic() - began, 0.0), str(exc)),
                  "state": "failed"}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    (folder / f"{shard['id']}.json").write_text(json.dumps(result))
    return result
