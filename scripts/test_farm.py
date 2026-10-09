"""`scripts/test farm`: run test files on a paired device, or split them between this machine and it.

    scripts/test farm --to DEVICE FILE...            the whole list as one shard on DEVICE
    scripts/test farm --to DEVICE --split FILE...    divide the list between this machine and DEVICE
    scripts/test farm --to DEVICE --check            say whether DEVICE takes test shards

The device runs `scripts/test all` on exactly this tree (docs/test-farm.md). Exit status is the
worst of the local and remote runs; 70 means the shard never ran.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ml_stack.activity import reuse
from ml_stack.fleet import shard_client, shard_result, shard_split, shard_tree
from ml_stack.fleet.remote import Peer, PeerError
from ml_stack.workspace import testruns

NOT_RUN = 70
DEFAULT_TIMEOUT = 1800
SETUP_S = 3.0


def parser() -> argparse.ArgumentParser:
    """The farm command line."""
    p = argparse.ArgumentParser(prog="scripts/test farm", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="test files, tests/NAME.py")
    p.add_argument("--to", required=True, help="name of a paired device")
    p.add_argument("--split", action="store_true", help="run part of the list here and part there")
    p.add_argument("--check", action="store_true", help="only ask whether the device takes test shards")
    p.add_argument("--host", default="", help="address to use instead of the peer book's")
    p.add_argument("--port", type=int, default=shard_client.DAEMON_PORT)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="seconds a shard may take")
    return p


def history_path(root: Path) -> Path:
    """Where this project's per-file duration history lives."""
    return reuse.reuse_base() / testruns.scope(root) / "farm-durations.json"


def summary(result: dict) -> str:
    """A pytest-style last line for a remote result."""
    totals = {key: sum(c[key] for c in result["files"].values()) for key in ("passed", "failed", "error", "skipped")}
    shown = ", ".join(f"{n} {key}" for key, n in totals.items() if n) or "no tests ran"
    return f"{shown} in {result['wall_s']:.1f}s"


def show(result: dict, name: str) -> list[str]:
    """The lines a remote run prints, in the shape of a local one."""
    where = result["platform"]
    lines = [f"+ ran on {name}: {where['system']}, python {where['python']}, {where['cpus']} cpus"]
    lines += [f"FAILED {f['nodeid']} - {f['message'].splitlines()[0][:200] if f['message'] else f['state']}"
              for f in result["failures"]]
    if result["exit"] and not result["failures"]:
        lines += result["output_tail"].splitlines()[-20:]
    lines.append(f"{summary(result)} (exit {result['exit']}, {result['cpu_s']:.1f}s cpu, on {name})")
    return lines


def remote(peer: Peer, root: Path, files: list[str], timeout: int, out: dict) -> None:
    """Send ``files`` to ``peer`` and put its result, or the reason it never ran, in ``out``."""
    try:
        began = time.monotonic()
        shard_id = shard_client.send(peer, root, files, timeout)
        out["sent_s"] = time.monotonic() - began
        out["result"] = shard_client.wait(peer, shard_id, timeout_s=timeout + 120)
    except (PeerError, shard_tree.TreeError, OSError) as exc:
        out["error"] = str(exc)


def local(root: Path, files: list[str], out: dict) -> None:
    """Run ``files`` with this machine's runner and note its junit file and exit status."""
    junit = Path(tempfile.mkstemp(suffix=".xml")[1])
    began = time.monotonic()
    code = subprocess.run([sys.executable, str(root / "scripts" / "test"), "all", "--no-reuse", f"--junitxml={junit}", *files],
                          cwd=root, check=False).returncode
    out.update({"exit": code, "wall_s": time.monotonic() - began, "junit": junit})


def plan_for(args: argparse.Namespace, root: Path, here_workers: int, there_workers: int) -> dict[str, list[str]]:
    """Which files go where: everything to the device, or a duration-balanced split."""
    if not args.split:
        return {args.to: list(args.files)}
    stay = {name for name in args.files if shard_split.local_only(root, name)}
    targets = [shard_split.Target("local", here_workers), shard_split.Target(args.to, there_workers, SETUP_S)]
    return shard_split.split(list(args.files), targets, shard_split.load(history_path(root)), stay)


def learn(root: Path, parts: dict[str, dict], files: list[str]) -> None:
    """Fold what each side measured into the duration history."""
    for part in parts.values():
        if "junit" in part:
            part["files"] = shard_result.read_junit(part["junit"], files)["files"]
            part["junit"].unlink(missing_ok=True)
        if "files" in part:
            shard_split.record(history_path(root), part)


def check(peer: Peer, name: str) -> int:
    """Print whether ``name`` takes shards; 0 when it does."""
    try:
        got = shard_client.capability(peer)
    except PeerError as exc:
        print(f"farm: {name} did not answer: {exc}", file=sys.stderr)
        return NOT_RUN
    where = got["platform"]
    print(f"farm: {name} takes test shards: {got['accepts']} ({where['system']}, python {where['python']}, "
          f"{where['cpus']} cpus, {got['active']} running)")
    return 0 if got["accepts"] else NOT_RUN


def main(argv: list[str], root: Path, connect: Callable[..., Peer] = shard_client.device_peer) -> int:
    """Run the farm command; the exit status."""
    args = parser().parse_args(argv)
    try:
        peer = connect(args.to, port=args.port, host=args.host)
    except shard_client.NoDevice as exc:
        print(f"farm: {exc}", file=sys.stderr)
        return NOT_RUN
    if args.check:
        return check(peer, args.to)
    if not args.files:
        print("farm: name the test files to run", file=sys.stderr)
        return NOT_RUN
    cpus = (os.cpu_count() or 2) - 1
    try:
        there = shard_client.capability(peer)
    except PeerError as exc:
        print(f"farm: {args.to} did not answer: {exc}", file=sys.stderr)
        return NOT_RUN
    if not there["accepts"]:
        print(f"farm: {args.to} does not take test shards; its person turns them on", file=sys.stderr)
        return NOT_RUN
    plan = plan_for(args, root, cpus, there["platform"]["cpus"])
    for name, files in plan.items():
        print(f"farm: {len(files)} files on {name}")
    return execute(args, peer, root, plan)


def execute(args: argparse.Namespace, peer: Peer, root: Path, plan: dict[str, list[str]]) -> int:
    """Start the remote part, run the local part here, and report both."""
    remote_out: dict = {}
    local_out: dict = {}
    thread = threading.Thread(target=remote, args=(peer, root, plan.get(args.to, []), args.timeout, remote_out))
    if plan.get(args.to):
        thread.start()
    if plan.get("local"):
        local(root, plan["local"], local_out)
    if thread.is_alive() or plan.get(args.to):
        thread.join()
    status = int(local_out.get("exit", 0))
    if remote_out.get("error") or (plan.get(args.to) and "result" not in remote_out):
        print(f"farm: the shard did not run on {args.to}: {remote_out.get('error', 'no result')}", file=sys.stderr)
        return NOT_RUN
    if "result" in remote_out:
        print("\n".join(show(remote_out["result"], args.to)))
        status = max(status, int(remote_out["result"]["exit"]))
    learn(root, {"local": local_out, args.to: remote_out.get("result", {})}, list(args.files))
    return status
