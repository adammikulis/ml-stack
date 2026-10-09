"""`scripts/test TIER [FILE...] --on DEVICE|all`: the same tiers, run on other devices of the pool.

    scripts/test all tests/test_x.py --on "Windows PC"     those files on one device
    scripts/test quick --on all                            the files a change reaches, on every other device
    scripts/test gate --on wsl-box                         the structural checks there
    scripts/test full --on all --split                     divide the whole list between this machine and the devices

The receiving node runs `scripts/test` on exactly this tree, on its own Python 3.13, and sends the result
back; the printed lines are a local run's, with the device and its platform named. A file that passed on a
device with this content, platform and tier is not sent again (`--no-reuse` sends it): a pass on one device
never satisfies another. Each result is posted to the board, keyed by device. Exit status is the worst of
the runs; 70 means a device never ran its part (not enrolled, shards off, unreachable, refused).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import affected
import testreuse_key as keys

from ml_stack import features
from ml_stack.activity import reuse
from ml_stack.activity.gate import tree_hash
from ml_stack.fleet import shard_split
from ml_stack.fleet.shard_spec import TEST_FILE
from ml_stack.testfarm import report
from ml_stack.testfarm.client import ShardError, Shards, choose, pool_devices
from ml_stack.testfarm.consent import FEATURE
from ml_stack.testfarm.ledger import Ledger, key as ledger_key
from ml_stack.workspace import testruns

NOT_RUN = 70
TIERS = ("quick", "fast", "full", "slow", "all", "gate")
SETUP_S = 3.0
OPTION = re.compile(r"^-")


@dataclass
class Run:
    """What every part of one remote run shares: who asks, the record, the tier, the file keys and the tree."""

    shards: Shards
    ledger: Ledger
    tier: str
    content: dict[str, str]
    tree: str = ""


class Part:
    """One device's share of a run: what to send, what was reused, and how it ended."""

    def __init__(self, device: dict, caps: dict, files: list[str]) -> None:
        self.device, self.caps, self.files = device, caps, files
        self.reused: list[str] = []
        self.result: dict = {}
        self.error = ""

    @property
    def name(self) -> str:
        return self.device["name"]

    @property
    def fingerprint(self) -> str:
        return self.device["fingerprint"]


def refuse(why: str) -> int:
    """Say why a remote run cannot start; the exit status of an argument error."""
    print(f"test: {why}", file=sys.stderr)
    return 4


def reached(root: Path, base: str) -> tuple[str, list[str]] | None:
    """What ``quick`` runs elsewhere: the test files the change reaches (tier ``all``), the full tier when a
    changed path maps to nothing, None when nothing differs."""
    try:
        changed, deleted = affected.changed_since(root, base)
    except (RuntimeError, IndexError) as exc:
        print(f"quick: cannot diff against {base} ({exc}); running the full tier there")
        return "full", []
    if not changed:
        return None
    picked = affected.select(root, changed, deleted, base=base)
    if picked.unmapped:
        print(f"quick: {picked.unmapped[0]} maps to no test; running the full tier there")
        return "full", []
    return "all", sorted(picked.files)


def lookups(root: Path, tier: str, files: list[str], command_for: Callable[[str, str], list[str]]) -> dict[str, str]:
    """The content key of each file that may be reused; a file that must always run is left out."""
    found = {}
    for file in files:
        look = keys.lookup(root, file, command_for(tier, file))
        if not look.barred:
            found[file] = look.key
    return found


def reuse_passes(part: Part, run: Run) -> None:
    """Move the files that already passed on this device, with this content, out of what is sent."""
    for file in list(part.files):
        if file in run.content and run.ledger.passed(part.fingerprint, ledger_key(part.device, part.caps["platform"], run.tier, run.content[file])):
            part.files.remove(file)
            part.reused.append(file)


def send(run: Run, part: Part, root: Path, timeout: int) -> None:
    """Send the part's files to its device and keep the result, or the reason it never ran."""
    try:
        began = time.monotonic()
        shard = run.shards.send(part.fingerprint, root, run.tier, part.files, timeout)
        print(f"test: {len(part.files) or 'every'} test files sent to {part.name} in {time.monotonic() - began:.1f}s", flush=True)
        part.result = run.shards.wait(part.fingerprint, shard, timeout_s=timeout + 180)
    except ShardError as exc:
        part.error = str(exc)


def local(root: Path, tier: str, files: list[str]) -> int:
    """This machine's share of a split run, through the same runner."""
    return subprocess.run([sys.executable, str(root / "scripts" / "test"), tier, *files], cwd=root, check=False).returncode


def history(root: Path) -> Path:
    """Where this project's per-file duration history lives."""
    return reuse.reuse_base() / testruns.scope(root) / "farm-durations.json"


def divide(root: Path, parts: list[Part], files: list[str]) -> list[str]:
    """Give each device (and this machine) a duration-balanced share of ``files``; this machine's share."""
    stay = {name for name in files if shard_split.local_only(root, name)}
    targets = [shard_split.Target("local", 1), *(shard_split.Target(p.name, p.caps["platform"]["cpus"], SETUP_S) for p in parts)]
    plan = shard_split.split(files, targets, shard_split.load(history(root)), stay)
    for part in parts:
        part.files = plan.get(part.name, [])
    return plan.get("local", [])


def record(run: Run, part: Part) -> None:
    """Keep the run as the device's last result and its passes, and post it to the board (a board that is down is no failure)."""
    result = part.result
    n = report.counts(result)
    summary = {"tier": run.tier, "files": len(part.files), "exit": result["exit"], "passed": n["passed"],
               "failed": n["failed"] + n["error"], "tree": run.tree}
    passes = run.ledger.passes_of(part.device, part.caps["platform"], run.tier, result, lambda f: run.content.get(f, ""))
    run.ledger.record(part.device, summary, passes)
    line = report.board_line(part.device, run.tier, len(part.files), result, run.tree)
    try:
        run.shards.session.post("#general", "status", line, subject=f"test {run.tier} on {part.name}: {'PASS' if result['exit'] == 0 else 'FAIL'}")
    except Exception as exc:  # noqa: BLE001 - the board is a notice; the run's result is already printed and recorded
        print(f"test: board notice not sent: {exc}", file=sys.stderr)


def ready(shards: Shards, devices: list[dict]) -> list[Part]:
    """A part for each device that takes tests; the ones that do not are named with the reason."""
    parts = []
    for device in devices:
        try:
            caps = shards.capability(device["fingerprint"])
        except ShardError as exc:
            caps = {"accepts": False, "reason": f"did not answer: {exc}"}
        if caps["accepts"]:
            parts.append(Part(device, caps, []))
        else:
            print(f"test: {device['name']} does not take tests: {caps['reason']}", file=sys.stderr)
    return parts


def finish(run: Run, part: Part, sent: bool) -> int:
    """Print one device's outcome and record it; its exit status."""
    if part.error:
        print(f"test: nothing ran on {part.name}: {part.error}", file=sys.stderr)
        return NOT_RUN
    if not sent:
        print(f"+ {part.name}: every test file passed there with this content ({len(part.reused)} reused)")
        return 0
    print("\n".join(report.show(part.result, part.name, part.reused)))
    record(run, part)
    return int(part.result["exit"])


def main(args: argparse.Namespace, rest: list[str], root: Path, command_for: Callable[[str, str], list[str]],
         reuse_on: bool) -> int:
    """Run the tier on the devices ``--on`` names; the exit status."""
    tier, files = args.tier, list(rest)
    if tier not in TIERS:
        return refuse(f"{tier} does not run on another device; use one of {', '.join(TIERS)}")
    if any(OPTION.match(f) or not TEST_FILE.match(f) for f in files):
        return refuse("a run on another device takes test files tests/NAME.py only: no options, node ids or paths")
    if tier == "gate" and files:
        return refuse("the gate takes no files")
    if not features.enabled(FEATURE):
        return refuse(f"running tests on another device is an experimental feature and off here; a person turns it on with `ml-stack features enable {FEATURE}`")
    try:
        chosen = choose(args.on, pool_devices())
        shards = Shards()
    except Exception as exc:  # noqa: BLE001 - no node, no session, no such device: one line and the status of a run that never started
        print(f"test: {exc}", file=sys.stderr)
        return NOT_RUN
    if tier == "quick":
        picked = reached(root, args.base)
        if picked is None:
            print(f"quick: nothing differs from {args.base}; running nothing")
            return 0
        tier, files = picked
    parts = ready(shards, chosen)
    status = NOT_RUN if len(parts) < len(chosen) else 0
    if not parts:
        return status
    content = lookups(root, tier, files, command_for) if files and tier != "gate" else {}
    run = Run(shards, Ledger(reuse.reuse_base() / testruns.scope(root)), tier, content)
    mine = divide(root, parts, files) if args.split and files else []
    for part in parts:
        if not (args.split and files):
            part.files = list(files)
            if reuse_on:
                reuse_passes(part, run)
    sending = [p for p in parts if p.files or not (files or p.reused)]
    threads = [threading.Thread(target=send, args=(run, p, root, int(min(args.timeout, 3600)))) for p in sending]
    for thread in threads:
        thread.start()
    if mine:
        status = max(status, local(root, tier, mine))
    for thread in threads:
        thread.join()
    run.tree = tree_hash(root)
    for part in parts:
        status = max(status, finish(run, part, part in sending))
        if part.result:
            shard_split.record(history(root), part.result)
    return status
