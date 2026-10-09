"""``ml-stack-test-devices``: the pool's other devices, whether each takes tests, and its last result here."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ml_stack.activity import reuse
from ml_stack.board.client import NodeError
from ml_stack.log import say
from ml_stack.workspace import testruns

from . import report
from .client import ShardError, Shards, choose, pool_devices
from .ledger import Ledger


def ago(then: float) -> str:
    """``3m ago`` style age of a timestamp."""
    seconds = max(0, int(time.time() - then))
    for size, word in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return f"{seconds // size}{word} ago"
    return f"{seconds}s ago"


def last_line(last: dict) -> str:
    """One device's last recorded run, or a dash."""
    if not last:
        return "-"
    return f"{'PASS' if last['exit'] == 0 else 'FAIL'} {last['tier']} {last['passed']} passed {last['failed']} failed {ago(last['at'])}"


def describe(shards: Shards, device: dict, ledger: Ledger) -> dict:
    """One row: the device, what it says about taking shards, and the last run recorded for it."""
    row = {**device, "last": ledger.last(device["fingerprint"])}
    try:
        row["caps"] = shards.capability(device["fingerprint"])
    except ShardError as exc:
        row["caps"] = {"accepts": False, "reason": f"did not answer: {exc}"}
    return row


def table(rows: list[dict]) -> list[str]:
    """The lines `ml-stack-test-devices` prints."""
    lines = [f"{'DEVICE':<20} {'PLATFORM':<34} {'SLOTS':<6} {'TAKES TESTS':<12} LAST RESULT"]
    for row in rows:
        caps = row["caps"]
        platform = report.where(caps["platform"]) if "platform" in caps else "-"
        slots = f"{caps['free']}/{caps['most_active']}" if "free" in caps else "-"
        takes = "yes" if caps["accepts"] else "no"
        lines.append(f"{row['name']:<20} {platform:<34} {slots:<6} {takes:<12} {last_line(row['last'])}")
        if not caps["accepts"]:
            lines.append(f"{'':<20} {caps.get('reason', '')}")
    return lines


def command(argv: list[str] | None = None) -> int:
    """List the devices (``[DEVICE] [--json]``); 1 when no node answers or no registered session is available."""
    words = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in words
    named = [w for w in words if w != "--json"]
    if len(named) > 1 or any(w.startswith("-") for w in named):
        say("usage: ml-stack-test-devices [DEVICE] [--json]")
        return 2
    try:
        chosen = choose(named[0] if named else "all", pool_devices())
        ledger = Ledger(reuse.reuse_base() / testruns.scope(Path.cwd()))
        shards = Shards()
        rows = [describe(shards, device, ledger) for device in chosen]
    except (ShardError, NodeError) as exc:
        say(f"test-devices: {exc}")
        return 1
    say(json.dumps(rows, indent=2) if as_json else "\n".join(table(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(command())
