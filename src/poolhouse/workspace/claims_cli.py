"""The claim commands of `poolhouse-workspace`: own a branch, worktree, port, area, install or server."""

from __future__ import annotations

import argparse
from typing import Any

from poolhouse.board.session import Claim, Session
from poolhouse.command import flag
from poolhouse.workspace import claims, limits
from poolhouse.workspace.cli_options import FOR_AGENT

__all__ = ["KINDS", "TABLE", "shown"]

KINDS = ("branch", "worktree", "port", "server", "install", "area")
CLAIM = [flag("kind", choices=KINDS), flag("key")]


def shown(c: Claim) -> dict[str, Any]:
    """A claim as the commands print it."""
    return {"kind": c.kind, "key": c.key, "owner": c.owner, "expires_in_s": c.expires_in_s,
            "expiring_soon": c.expiring_soon, "pid": c.pid, "changed": c.changed}


def claim(args: argparse.Namespace, s: Session) -> Any:
    """Own something until the time runs out or the process named by --pid ends."""
    ttl = int(args.ttl or limits.load(limits.root()).claim_ttl_s)
    return shown(s.claim(args.kind, claims.normal(args.kind, args.key), ttl, args.pid))


def release(args: argparse.Namespace, s: Session) -> Any:
    """Give a claim up."""
    key = claims.normal(args.kind, args.key)
    return {"kind": args.kind, "key": key, "released": s.release(args.kind, key)}


def listing(args: argparse.Namespace, s: Session) -> Any:
    """Every live claim."""
    return [shown(c) for c in s.claims(args.kind)]


def heartbeat(args: argparse.Namespace, s: Session) -> Any:
    """Renew every claim the caller holds."""
    ttl = int(args.ttl or limits.load(limits.root()).claim_ttl_s)
    return {"renewed": len(s.renew_claims(ttl)), "capped": []}


def who(args: argparse.Namespace, s: Session) -> Any:
    """Who owns this?"""
    key = claims.normal(args.kind, args.key)
    return next((shown(c) for c in s.claims(args.kind) if c.key == key), {"owner": None})


TABLE = (
    ("claim", "own a branch, worktree, port, area, install environment or server", [
        *CLAIM, flag("--ttl", type=float, default=0.0, help="seconds; renew with heartbeat"),
        flag("--pid", type=int, default=0, help="release when this process is gone")], claim),
    ("release", "give a claim up", CLAIM, release),
    ("claims", "every live claim", [FOR_AGENT, flag("--kind", choices=KINDS, default="")], listing),
    ("heartbeat", "renew every claim you hold", [flag("--ttl", type=float, default=0.0)], heartbeat),
    ("who", "who owns this?", CLAIM, who),
)
