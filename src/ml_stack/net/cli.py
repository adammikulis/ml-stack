"""``ml-stack security`` commands for what comes in from the internet: recent downloads with
their provenance and scan status, the hosts that may be reached, the scanners on this
machine, and the policy for files nobody could scan. Changing the host list or the policy
needs a person at a terminal."""

from __future__ import annotations

import argparse
import getpass
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import files
from ml_stack.command import flag, option
from ml_stack.log import say, warn
from ml_stack.net import policy, provenance
from ml_stack.net.scan import ACTIONS, CATEGORIES, ScanPolicy
from ml_stack.net.scanners import ClamAV, MacNotice, WindowsDefender, default_scanners
from ml_stack.sentinel import human
from ml_stack.sentinel.cli import COMMANDS

__all__ = ["command"]

JSON = option("json")


def _out(args: argparse.Namespace, data: Any, lines: Callable[[], list[str]]) -> int:
    say(json.dumps(data, indent=2, default=str) if args.json else "\n".join(lines()))
    return 0


def _state(row: dict[str, Any]) -> str:
    path = Path(str(row.get("path") or ""))
    if row.get("outcome") == "held":
        return "held"
    if row.get("kind") == "git":
        return "present" if path.is_dir() else "gone"
    if not path.is_file():
        return "gone"
    if row.get("size") and path.stat().st_size != row["size"]:
        return "changed"
    return "present"


@COMMANDS.command("downloads", help="recent downloads with where they came from and how they scanned",
                  options=(JSON, flag("--limit", type=int, default=20),
                           flag("--held", action="store_true", help="only files that were held"),
                           flag("--verify", action="store_true",
                                help="re-hash each file that is still there")))
def downloads(args: argparse.Namespace) -> int:
    """List recent downloads from the index."""
    rows = [r for r in provenance.recent(max(args.limit * 5, 200))
            if not args.held or r.get("outcome") == "held"][:args.limit]
    for row in rows:
        row["state"] = _state(row)
        if args.verify and row["state"] == "present" and row.get("kind") != "git":
            row["state"] = ("present" if files.sha256_file(row["path"]) == row.get("sha256")
                            else "changed")

    def lines() -> list[str]:
        out = []
        for r in rows:
            out.append(f"{r.get('fetched_at', '')}  {r.get('outcome', ''):<4}  {r.get('state', ''):<7}"
                       f" {r.get('kind', ''):<11} {r.get('size', 0):>12}  {r.get('url', '')}")
            out.append(f"    sha256 {str(r.get('sha256', ''))[:16]}  {r.get('scan', '')}"
                       + (f"  held: {r.get('reason')}" if r.get("outcome") == "held" else ""))
        return out or ["no downloads recorded"]

    return _out(args, rows, lines)


@COMMANDS.command("hosts", help="the hosts that may be reached and the ones a person approved",
                  options=(JSON,))
def hosts(args: argparse.Namespace) -> int:
    """List the allow-list and the approvals."""
    current = policy.default()
    data = {"allowed": list(current.allowed),
            "approved": [{"host": a.host, "at": a.at, "by": a.by, "note": a.note,
                          "until": a.until} for a in current.approvals()]}

    def lines() -> list[str]:
        approved = [f"  {a['host']}  by {a['by']}  {a['note']}" for a in data["approved"]]
        return ["allowed without asking:", *[f"  {h}" for h in data["allowed"]],
                "approved by a person:", *(approved or ["  none"])]

    return _out(args, data, lines)


@COMMANDS.command("approve-host", help="let downloads and fetches reach one more host",
                  options=(flag("host"), flag("--hours", type=float, default=0.0,
                                              help="lapse after this many hours (default: never)"),
                           flag("--note", default="")))
def approve_host(args: argparse.Namespace) -> int:
    """Record an approval for one exact host name (a person at a terminal)."""
    try:
        human.mint("approve-host", args.host)
    except human.HumanRequired as exc:
        warn(f"ml-stack security: {exc}")
        return 2
    row = policy.default().approve(args.host, by=getpass.getuser(), note=args.note,
                                   for_s=args.hours * 3600)
    say(f"{row.host} approved by {row.by}"
        + (f" for {args.hours:g} hours" if args.hours else ""))
    return 0


@COMMANDS.command("scanners", help="which virus scanners this machine has and how fresh they are",
                  options=(JSON,))
def scanners(args: argparse.Namespace) -> int:
    """Report each backend honestly: present or not, and what it can and cannot do."""
    rows = []
    for scanner in default_scanners():
        row = {"name": scanner.name, "available": scanner.available(), "note": ""}
        if isinstance(scanner, ClamAV) and row["available"]:
            row["note"] = scanner.freshness() or "signature database is current"
        elif isinstance(scanner, MacNotice) and row["available"]:
            row["note"] = MacNotice.ADVICE
        elif isinstance(scanner, WindowsDefender) and not row["available"]:
            row["note"] = "only on Windows"
        rows.append(row)
    return _out(args, rows, lambda: [
        f"{r['name']:<18} {'available' if r['available'] else 'not available':<14} {r['note']}"
        for r in rows])


@COMMANDS.command("scan-policy", help="show, or set, whether files nobody could scan are kept",
                  options=(JSON, flag("category", nargs="?", default="",
                                      choices=["", *CATEGORIES, "scan_models"]),
                           flag("action", nargs="?", default="",
                                choices=["", *ACTIONS, "on", "off"])))
def scan_policy(args: argparse.Namespace) -> int:
    """Print the policy, or set one category (a person at a terminal)."""
    current = ScanPolicy.load()
    if args.category:
        if not args.action:
            warn("ml-stack security: say what to do: refuse, warn or allow (on or off for scan_models)")
            return 2
        try:
            human.mint("scan-policy", args.category)
        except human.HumanRequired as exc:
            warn(f"ml-stack security: {exc}")
            return 2
        if args.category == "scan_models":
            current = ScanPolicy(**{**vars_of(current), "scan_models": args.action == "on"})
        else:
            current = ScanPolicy(**{**vars_of(current), args.category: args.action})
        current.save()
    data = vars_of(current)
    return _out(args, data, lambda: [f"{k}: {v}" for k, v in data.items()])


def vars_of(value: ScanPolicy) -> dict[str, Any]:
    """A policy as plain values."""
    return {name: getattr(value, name) for name in (*CATEGORIES, "scan_models")}


command = COMMANDS.run
