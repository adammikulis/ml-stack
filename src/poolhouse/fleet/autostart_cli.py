"""The `prepare`, `install`, `rollback`, `status` and `verify` subcommands of `python -m poolhouse.fleet.autostart`."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from poolhouse.log import say, warn
from poolhouse.person import HumanRequired

from . import autostart_apply, autostart_check
from .autostart_manifest import ROLES, ManifestError
from .autostart_prepare import PrepareError, Spec, prepare

__all__ = ["COMMANDS", "add_commands", "run"]

COMMANDS = ("prepare", "install", "rollback", "status", "verify")


def add_commands(sub: argparse._SubParsersAction) -> None:
    """Add the autostart subcommands to the installer's subparsers."""
    prep = sub.add_parser("prepare", help="stage unit files and a manifest; nothing is installed")
    prep.add_argument("--role", action="append", choices=ROLES, required=True)
    prep.add_argument("--system", action="store_true", help="stage for the system-wide location")
    prep.add_argument("--launchers", required=True, help="the absolute directory holding the stable launchers")
    prep.add_argument("--slots", type=int, default=1)
    prep.add_argument("--label", action="append", default=[])
    prep.add_argument("--report", default="")
    prep.add_argument("--json", action="store_true")
    inst = sub.add_parser("install", help="install a prepared manifest (a person at a terminal only)")
    inst.add_argument("--manifest", required=True)
    inst.add_argument("--system", action="store_true")
    undo = sub.add_parser("rollback", help="restore what the last install replaced (a person at a terminal only)")
    undo.add_argument("--system", action="store_true")
    for name in ("status", "verify"):
        read = sub.add_parser(name, help="compare what is installed with what was prepared; changes nothing")
        read.add_argument("--json", action="store_true")


def _prepare(a: argparse.Namespace) -> int:
    spec = Spec(tuple(a.role), "system" if a.system else "user", Path(a.launchers) if a.launchers else None,
                a.slots, tuple(a.label), a.report)
    try:
        done = prepare(spec)
    except (PrepareError, ManifestError, OSError, RuntimeError) as exc:
        warn(f"not prepared: {exc}")
        return 2
    if a.json:
        say(json.dumps({**done.manifest.to_dict(), "path": str(done.path), "command": done.command}))
        return 0
    expires = time.strftime("%Y-%m-%d %H:%M", time.localtime(done.manifest.expires))
    say(f"prepared {', '.join(r.role for r in done.manifest.roles)} in {done.path.parent} (valid until {expires})")
    say("nothing is installed. To install, run this yourself at your own terminal:")
    say(done.command)
    return 0


def _outcome(done: autostart_apply.Outcome) -> int:
    for line in done.lines:
        (say if done.ok else warn)(line)
    if done.rollback:
        say(f"to undo: {done.rollback}")
    return 0 if done.ok else 2


def _report(a: argparse.Namespace) -> int:
    report = autostart_check.check()
    if a.json:
        say(json.dumps({"state": report.state, "reasons": list(report.reasons), "roles": dict(report.roles),
                        "manifest": report.manifest}))
    else:
        say(f"autostart: {report.state}")
        for why in report.reasons:
            say(f"  {why}")
    return 0 if a.cmd == "status" or report.state == "current" else 1


def run(a: argparse.Namespace) -> int:
    """Carry out the autostart subcommand in ``a``."""
    if a.cmd == "prepare":
        return _prepare(a)
    if a.cmd in ("status", "verify"):
        return _report(a)
    try:
        if a.cmd == "install":
            return _outcome(autostart_apply.install(a.manifest, system=a.system))
        return _outcome(autostart_apply.rollback(system=a.system))
    except HumanRequired as exc:
        warn(str(exc))
        return 2
