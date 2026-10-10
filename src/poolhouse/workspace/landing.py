"""The landing queue seen from a session: requests, independent reviews, brakes and the runner's states.

The queue is a fold of the board's `landing` entries written on this device (and on a device it names with
`trust`), made by the node, which stamps every entry with the session behind the token and applies the rules
that need the board (independence, the runner's claim). Entries of any other device stay on the board and are
listed as foreign and ignored. What the node cannot know is a model's tier, so this module holds the bar for who
may ask for the development branch and checks it, from the standing the node stamped on each entry, when the runner
is about to land.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poolhouse.board.client import Denied
from poolhouse.board.session import Agent, Session
from poolhouse.devbranch import development_branch
from poolhouse.lock import pid_alive
from poolhouse.workspace.model_tiers import tier_of
from poolhouse.workspace.modelid import VERIFIED

__all__ = ["Queue", "beat", "brake", "cancel", "eligible", "fold", "refusal", "request", "review", "runner_claim",
           "standing", "status_lines", "supervisor_state", "transition", "trust"]

OPEN = ("queued", "needs-review", "running")
CAUSES = ("revoked", "forged")
SUPERVISOR_STATUS = "land-runner.json"
RUNNER_TTL_S = 1800


@dataclass
class Queue:
    """The node's view: requests in order, the pause state and the runner's last mark."""

    requests: dict[str, dict[str, Any]] = field(default_factory=dict)
    paused: bool = False
    paused_by: str = ""
    beat: dict[str, Any] | None = None
    foreign: dict[str, Any] = field(default_factory=dict)
    trusted_devices: list[str] = field(default_factory=list)
    foreign_claims: list[dict[str, Any]] = field(default_factory=list)

    def waiting(self) -> list[dict[str, Any]]:
        """Requests the runner may act on, oldest first."""
        return [r for r in self.requests.values() if r["status"] in ("queued", "needs-review")]


def runner_claim(target: str = "") -> tuple[str, str]:
    """The claim whose holder is the landing runner: the development branch itself.

    Task integration claims the same branch, so a runner and an integration never land at once.
    """
    return "branch", target or development_branch()


def tier_reason(model: str) -> str:
    """Why a session on ``model`` may not land work, or an empty string: the model must be listed and above the
    lowest tier.

    A model id counts as the session claimed it: the node has no model verification yet, so the table is
    read as if every listed id were verified (a stub, listed in docs/node.md with the grants).
    """
    return tier_of(model, VERIFIED).reason


def eligible(who: Agent) -> str:
    """Why ``who`` may not land work, or an empty string."""
    return tier_reason(who.model)


def fold(s: Session) -> Queue:
    """The queue as the board stands."""
    got = s.call("land_queue")
    requests = {r["id"]: {**r, "ts": r["ts_ms"] / 1000, "reviews": {n: {**v, "ts": v["ts_ms"] / 1000}
                                                                    for n, v in r["reviews"].items()}}
                for r in got["requests"]}
    return Queue(requests, got["paused"], got["paused_by"], got["beat"], got["foreign"], got["trusted_devices"],
                 got["foreign_claims"])


def _caller(s: Session) -> Agent:
    me = s.whoami()
    why = eligible(me)
    if why:
        raise Denied("denied", why)
    return me


def request(s: Session, fields: dict[str, Any]) -> dict[str, Any]:
    """Queue a branch at an exact commit; the requester is whoever the token names."""
    _caller(s)
    return dict(s.call("land_request", branch=fields["branch"], sha=fields["sha"],
                       target=fields.get("target") or development_branch(),
                       selectors=[str(t) for t in fields.get("selectors", [])], replaces=fields.get("replaces", "")))


def review(s: Session, rid: str, sha: str, verdict: str) -> dict[str, Any]:
    """Record an independent review of exactly ``sha``; the author and its delegates cannot review."""
    _caller(s)
    return dict(s.call("land_review", req=rid, sha=sha, verdict=verdict))


def cancel(s: Session, rid: str) -> dict[str, Any]:
    """Cancel an open request: its requester, the runner or a session that steers landing."""
    return dict(s.call("land_cancel", req=rid))


def brake(s: Session, pause: bool, reason: str = "") -> dict[str, Any]:
    """Pause or resume the runner."""
    return dict(s.call("land_brake", pause=pause, reason=reason))


def trust(s: Session, device: str, trusted: bool = True) -> dict[str, Any]:
    """Count (or stop counting) the landing entries of another device of the pool, by its certificate fingerprint.

    Landing authority is this device's own until this says otherwise; the node needs a session with no parent that
    holds the grant `land_trust` and records the change on the board.
    """
    return dict(s.call("land_trust", device=device, trusted=trusted))


def transition(s: Session, rid: str, status: str, detail: str = "", **evidence: Any) -> None:
    """The runner records how a request moved; only the session holding the runner claim may."""
    params = {"evidence": evidence} if evidence else {}
    s.call("land_state", req=rid, status=status, detail=detail[:400], **params)


def beat(s: Session, what: str) -> None:
    """The runner's progress mark, read by the watchdog and the status page."""
    s.call("land_beat", what=what[:300])


def _related(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two stamped standings are one session or one is an ancestor of the other."""
    return a["name"] == b["name"] or a["name"] in b["ancestors"] or b["name"] in a["ancestors"]


def _stamped(entry: dict[str, Any], who: str) -> dict[str, Any]:
    """The standing the node stamped on a request or review by ``who``, or an empty dict when it carries none."""
    found = entry.get("standing")
    ok = isinstance(found, dict) and found.get("name") == who and isinstance(found.get("ancestors"), list)
    return found if ok else {}


def _ended_for_cause(s: Session) -> dict[str, str]:
    """The sessions of this board retired as revoked or forged, by name, with why."""
    return {a.name: a.retired_reason for a in s.agents(retired=True) if a.retired and a.retired_reason in CAUSES}


def refusal(s: Session, req: dict[str, Any]) -> str:
    """Why ``req`` can never land, or an empty string: its requester was ended for cause after it asked."""
    why = _ended_for_cause(s).get(req["by"], "")
    return f"the requester {req['by']} was {why} after it asked" if why else ""


def standing(s: Session, req: dict[str, Any]) -> str:
    """Why the runner must not land ``req`` now, or an empty string.

    The node stamped the requester's and each reviewer's name, ancestors and model on the entry when it was
    written, so a worker that asked and then ended (its session retired) stays landable. What can still change is
    an end for cause: a session retired as ``revoked`` or ``forged`` after it asked or reviewed no longer counts.
    A reviewer counts when its stamped model was at the landing level and its stamped lineage was independent of
    the requester's; any standing reject blocks.
    """
    cause = _ended_for_cause(s)
    asked = _stamped(req, req["by"])
    if not asked:
        return "the node stamped no standing on the request"
    if tier_reason(asked["model"]):
        return "the requester was not at the landing level when it asked"
    live = {}
    for name, found in req["reviews"].items():
        seen = _stamped(found, name)
        if seen and not tier_reason(seen["model"]) and name not in cause and not _related(seen, asked):
            live[name] = found["verdict"]
    if "reject" in live.values():
        return "an independent reviewer rejected it"
    return "" if "accept" in live.values() else "needs review: no independent accept that still stands"


def supervisor_state(base: Path) -> dict[str, Any]:
    """What the runner's supervisor last recorded in the workspace at ``base``, with ``alive`` from its pid.

    Empty when no supervisor ever started here.
    """
    try:
        row = json.loads((base / SUPERVISOR_STATUS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    pid = int(row.get("supervisor") or 0)
    return {**row, "alive": pid > 0 and pid_alive(pid)}


def supervisor_line(base: Path) -> str:
    """One line for ``digest --status``: whether the runner is kept alive, and how to start it."""
    now = supervisor_state(base)
    if not now.get("alive"):
        return "Runner supervisor: NOT RUNNING; start it with scripts/land up"
    return f"Runner supervisor: up (pid {now['supervisor']}), {now.get('restarts', 0)} restarts, " \
           f"{now.get('state', '')}; log {now.get('log', '')}"


def status_lines(s: Session, base: Path) -> list[str]:
    """The queue and the current gate as plain lines for ``digest --status``."""
    queue = fold(s)
    held = next((c for c in s.claims("branch") if c.key == runner_claim()[1]), None)
    head = f"Landing queue: {'PAUSED by ' + queue.paused_by if queue.paused else 'running'}; runner " \
           f"{held.owner if held else 'none'}"
    if queue.beat:
        head += f"; last step {queue.beat['age_s']}s ago: {queue.beat['what']}"
    rows = [f"{r['id']} {r['branch']}@{r['sha'][:8]} by {r['by']}: {r['status']}"
            f"{' - ' + r['detail'] if r['detail'] else ''}"
            for r in queue.requests.values() if r["status"] in OPEN or r["status"] == "needs-human"]
    ignored = [f"{e['ev']} by {e['by']} on {e['device']}: foreign, ignored" for e in queue.foreign.get("latest", [])[-5:]]
    ignored += [f"claim on branch {c['branch']} held by {c['holder']} on {c['device']}: foreign, ignored"
                for c in queue.foreign_claims]
    if queue.foreign.get("total"):
        ignored.insert(0, f"{queue.foreign['total']} entries of other devices, foreign, ignored (trusted devices: "
                          f"{', '.join(d[:12] for d in queue.trusted_devices) or 'none'}); latest:")
    return [head, supervisor_line(base), *(rows or ["(queue empty)"]), *ignored]
