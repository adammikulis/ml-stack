"""Commands that put a ready branch on the landing queue and steer it."""

from __future__ import annotations

import argparse
from typing import Any

from ml_stack.command import flag
from ml_stack.workspace import landing
from ml_stack.workspace.screen import fence
from ml_stack.workspace.service import Workspace


def queue_view(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    """The queue and the current gate, fenced as data."""
    ws.auth(token)
    return {"authority": "none", "text": fence("\n".join(landing.status_lines(ws)), "workspace:landing",
                                                "branch names and details written by agents").text}


TABLE = [
    ("land-request",
     "hand a ready branch to the landing queue: BRANCH, its exact full SHA, the affected test selectors "
     "you ran and what it replaces; needs an independent land-review before the runner lands it",
     [flag("branch"), flag("sha", help="the full 40-character commit the branch tip is at now"),
      flag("--test", dest="selectors", action="append", default=[], metavar="SELECTOR",
           help="an affected test file or selector you ran (repeat for each)"),
      flag("--replaces", default="", help="one line: what this change replaces or supersedes"),
      flag("--target", default="", help="development branch (default 0.2dev; never main)")],
     lambda a, w, t: landing.request(w, t, {"branch": a.branch, "sha": a.sha, "selectors": a.selectors,
                                            "replaces": a.replaces, "target": a.target})),
    ("land-review",
     "independently review a land-request at its exact SHA (you cannot review your own or a delegate's)",
     [flag("id", metavar="REQUEST"), flag("sha", help="the full SHA you reviewed; must equal the request's"),
      flag("--verdict", choices=("accept", "reject"), default="accept")],
     lambda a, w, t: landing.review(w, t, a.id, a.sha, a.verdict)),
    ("land-cancel", "cancel a landing request (the requester, a person, a lead or the runner)",
     [flag("id", metavar="REQUEST")], lambda a, w, t: landing.cancel(w, t, a.id)),
    ("land-pause", "stop the landing runner starting new batches (a person or a lead)",
     [flag("--reason", default="")], lambda a, w, t: landing.brake(w, t, True, a.reason)),
    ("land-resume", "let the landing runner start batches again (a person or a lead)", [],
     lambda a, w, t: landing.brake(w, t, False)),
    ("land-queue", "the landing queue, who holds the runner and what the current gate last did", [],
     queue_view),
]
