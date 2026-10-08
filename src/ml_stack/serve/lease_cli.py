"""``ml-stack-serve leases|history``: who holds each model server and why, and the leases that ended."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ml_stack.command import flag
from ml_stack.log import say
from ml_stack.serve import holding, lease_history, ops, provenance
from ml_stack.serve.broker import BrokerError
from ml_stack.serve.leases import lease_file
from ml_stack.units import parse_duration

__all__ = ["OPTIONS_HISTORY", "OPTIONS_LEASES", "cmd_history", "cmd_leases", "other_holders"]

OPTIONS_LEASES = [flag("--json", action="store_true", help="print the holders as JSON")]

OPTIONS_HISTORY = [
    flag("--model", default="", metavar="NAME", help="only leases of a model whose name contains this"),
    flag("--since", default="", metavar="WHEN",
         help="only leases that ended since: a date (2026-10-01), or how long ago (3d, 2h, 90m)"),
    flag("--json", action="store_true", help="print the rows as JSON"),
]


def history_file() -> Path:
    """Where the ended leases are kept."""
    return lease_file().with_name("lease-history.ladybug")


def since_epoch(text: str, *, now: float | None = None) -> float:
    """Epoch seconds for a ``--since`` value; raises ``ValueError`` when it is neither a date
    nor a span."""
    now = time.time() if now is None else now
    if not text:
        return 0.0
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return time.mktime(time.strptime(text, "%Y-%m-%d"))
    days = re.fullmatch(r"(\d+(?:\.\d+)?)d", text)
    span = float(days.group(1)) * 86400.0 if days else parse_duration(text)
    if span is None:
        raise ValueError(f"not a date or a span: {text!r}")
    return now - span


def snapshot() -> dict[str, Any]:
    """The servers, holders and queue of the broker this machine uses; raises `BrokerError`
    or ``OSError`` when none is running."""
    return dict(ops.manager_for().broker.snapshot())


def other_holders(servers: list[Mapping[str, Any]], port: int, *, besides: str = "") -> list[Mapping[str, Any]]:
    """The holders of the server on ``port`` other than lease ``besides``."""
    return [h for s in servers if s["port"] == port for h in s["holders"] if h["lease"] != besides]


def cmd_leases(args: argparse.Namespace) -> int:
    """``ml-stack-serve leases`` -- every server the broker holds, and for each holder why, who and from where."""
    try:
        held_now = snapshot()
    except (BrokerError, OSError) as exc:
        say(str(exc))
        return 1
    if args.json:
        say(json.dumps({"servers": held_now["servers"], "queue": held_now["queue"]}, indent=2))
        return 0
    for held in held_now["servers"]:
        joined = [(hold.id, row) for hold in holding.holds() if hold.port == held["port"]
                  for row in holding.joined(hold.id)]
        say(f":{held['port']}  {Path(held['model']).name}  {held.get('device') or '-'}  "
            f"{len(held['holders']) + len(joined)} holder(s)")
        for holder in held["holders"]:
            say(f"  lease {holder['lease'][:8]}")
            for line in provenance.lines(holder, indent="    "):
                say(line)
        for hold_id, row in joined:
            say(f"  joined the lease {hold_id} with `up`")
            for line in provenance.lines(row, indent="    "):
                say(line)
    for waiting in held_now["queue"]:
        say(f"waiting: {waiting['model']} for pid {waiting['pid']} -- {waiting.get('reason') or provenance.NO_REASON}"
            f" ({waiting.get('requester') or waiting['label']})")
    if not (held_now["servers"] or held_now["queue"]):
        say("the broker holds nothing")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """``ml-stack-serve history`` -- the leases that have ended: model, why, who, branch, started, ended."""
    try:
        floor = since_epoch(args.since)
    except ValueError as exc:
        say(str(exc))
        return 2
    found = lease_history.rows(history_file(), model=args.model, since=floor)
    if args.json:
        say(json.dumps(found, indent=2))
        return 0
    if not found:
        say("no ended lease on record")
        return 0
    stamp = "%F %T"
    for row in found:
        began = time.strftime(stamp, time.localtime(float(row.get("taken") or 0)))
        end = time.strftime(stamp, time.localtime(float(row["ended"])))
        say(f"{began} -> {end}  {float(row['duration_s']):.0f}s  {row['model']}  "
            f"{row.get('reason') or provenance.NO_REASON}  [{row.get('requester', '')}"
            + (f" on {row['branch']}" if row.get("branch") else "") + "]")
    return 0
