"""Commands that put a ready branch on the landing queue and steer it."""

from __future__ import annotations

import argparse
from typing import Any

from poolhouse.board.session import Session
from poolhouse.command import flag
from poolhouse.workspace import landing, limits
from poolhouse.workspace.screen import fence

__all__ = ["TABLE"]


def queue_view(args: argparse.Namespace, s: Session) -> Any:
    """The queue and the current gate, fenced as data."""
    return {"authority": "none", "text": fence("\n".join(landing.status_lines(s, limits.root())), "workspace:landing",
                                                "branch names and details written by agents").text}


TABLE = [
    ("land-request",
     "hand a ready branch to the landing queue: BRANCH, its exact full SHA, the affected test selectors "
     "you ran and what it replaces; needs an independent land-review before the runner lands it",
     [flag("branch"), flag("sha", help="the full 40-character commit the branch tip is at now"),
      flag("--test", dest="selectors", action="append", default=[], metavar="SELECTOR",
           help="an affected test file or selector you ran (repeat for each)"),
      flag("--replaces", default="", help="one line: what this change replaces or supersedes"),
      flag("--target", default="", help="development branch (default: the branch the primary checkout is on; never main)")],
     lambda a, s: landing.request(s, {"branch": a.branch, "sha": a.sha, "selectors": a.selectors,
                                      "replaces": a.replaces,
                                      "target": a.target})),
    ("land-review",
     "independently review a land-request at its exact SHA (you cannot review your own or your subagent's)",
     [flag("id", metavar="REQUEST"), flag("sha", help="the full SHA you reviewed; must equal the request's"),
      flag("--verdict", choices=("accept", "reject"), default="accept")],
     lambda a, s: landing.review(s, a.id, a.sha, a.verdict)),
    ("land-cancel", "cancel a landing request (the requester, a person, a lead or the runner)",
     [flag("id", metavar="REQUEST")], lambda a, s: landing.cancel(s, a.id)),
    ("land-pause", "stop the landing runner starting new batches (a person or a lead)",
     [flag("--reason", default="")], lambda a, s: landing.brake(s, True, a.reason)),
    ("land-resume", "let the landing runner start batches again (a person or a lead)", [],
     lambda a, s: landing.brake(s, False)),
    ("land-queue", "the landing queue, who holds the runner and what the current gate last did", [],
     queue_view),
]
