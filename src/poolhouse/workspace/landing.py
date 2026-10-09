"""The landing queue on the board: requests, independent reviews, brakes and the runner's states.

Every row is appended to one hash-chained log stamped with the authenticated identity, never a
name the caller supplies. The queue is a fold of that log, so it needs no other state. A request
names one branch and one exact commit; a new request for the same branch supersedes the old one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from poolhouse.devbranch import development_branch
from poolhouse.workspace.chain import ChainLog, held
from poolhouse.workspace.claims import alive
from poolhouse.workspace.identity import AGENT, HUMAN, LEAD, Denied, Identity
from poolhouse.workspace.model_tiers import tier_of

__all__ = ["Queue", "beat", "brake", "cancel", "eligible", "fold", "identity_of", "log",
           "request", "review", "runner_claim", "standing", "status_lines", "supervisor_state", "transition"]

SHA = re.compile(r"[0-9a-f]{40}")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
OPEN = ("queued", "needs-review", "running")
TERMINAL = ("landed", "landed-unpushed", "failed", "needs-human", "refused", "cancelled", "superseded")
MAX_SELECTORS = 64
MAX_TEXT = 400
SUPERVISOR_STATUS = "land-runner.json"


def log(ws) -> ChainLog:
    """The landing log of this workspace."""
    return ChainLog(ws.base / "landing.jsonl", ws.clock)


def runner_claim(ws, target: str = "") -> tuple[str, str]:
    """The claim whose holder is the landing runner: the development branch itself.

    Task integration claims the same branch, so a runner and an integration never land at once.
    """
    return "branch", target or development_branch()


def eligible(ws, who: Identity) -> str:
    """Why ``who`` may not land work, or an empty string.

    A person or a lead is at the landing level. An agent is when it is a top-level session (not a
    delegated helper) with a live token, the right to send, and a verified model above the lowest
    tier: the same bar as the coordinator, because a request asks for the development branch.
    """
    if who.role in (HUMAN, LEAD):
        return ""
    if who.role != AGENT or who.parent:
        return "a delegated helper cannot land; its parent requests landing"
    if "send" not in who.can or not ws.registry.role_of(who.id):
        return "the identity lacks the right to send or has expired"
    return tier_of(*ws.registry.model_of(who.id)).reason


def lineage(a: str, b: str) -> bool:
    """Whether one identity is the other or a delegate of it."""
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def controller(ws, who: Identity) -> bool:
    """Whether ``who`` is a person, a lead or the identity that holds the runner claim."""
    owned = ws.who_owns(*runner_claim(ws))
    return who.role in (HUMAN, LEAD) or bool(owned and owned.get("owner") == who.id)


class Queue:
    """The folded view: requests in order, the pause state and the last runner beat."""

    def __init__(self) -> None:
        self.requests: dict[str, dict[str, Any]] = {}
        self.paused = False
        self.paused_by = ""
        self.beat: dict[str, Any] = {}

    def waiting(self) -> list[dict[str, Any]]:
        """Requests the runner may act on, oldest first."""
        return [r for r in self.requests.values() if r["status"] in ("queued", "needs-review")]

    def accepted(self, req: dict[str, Any]) -> bool:
        """Whether a still-valid independent accept stands for this request."""
        verdicts = [v["verdict"] for v in req["reviews"].values()]
        return "accept" in verdicts and "reject" not in verdicts


def _apply(queue: Queue, row: dict[str, Any]) -> None:
    kind, by = row.get("ev"), row.get("by", "")
    if kind == "request":
        row = {**row, "id": f"land-{row['seq']}"}
        for old in queue.requests.values():
            if old["branch"] == row["branch"] and old["status"] in OPEN:
                old.update(status="superseded", detail=f"replaced by {row['id']}")
        queue.requests[row["id"]] = {**{k: row[k] for k in ("id", "branch", "sha", "target", "selectors",
                                                            "replaces", "by", "ts")},
                                     "status": "needs-review", "detail": "no independent review yet",
                                     "reviews": {}}
        return
    if kind == "pause":
        queue.paused, queue.paused_by = True, by
    elif kind == "resume":
        queue.paused, queue.paused_by = False, ""
    elif kind == "beat":
        queue.beat = {"by": by, "ts": row["ts"], "what": row.get("what", "")}
    req = queue.requests.get(row.get("req", ""))
    pushed_late = req is not None and req["status"] == "landed-unpushed" and kind == "state" \
        and row.get("status") == "landed"
    if req is None or (req["status"] in TERMINAL and kind != "review" and not pushed_late):
        return
    if kind == "review":
        req["reviews"][by] = {"verdict": row["verdict"], "ts": row["ts"]}
        _refresh(req)
    elif kind == "cancel":
        req.update(status="cancelled", detail=f"cancelled by {by}")
    elif kind == "state":
        req.update(status=row["status"], detail=row.get("detail", ""), evidence=row.get("evidence", {}))


def _refresh(req: dict[str, Any]) -> None:
    if req["status"] in ("queued", "needs-review"):
        verdicts = [v["verdict"] for v in req["reviews"].values()]
        stands = "accept" in verdicts and "reject" not in verdicts
        req.update(status="queued" if stands else "needs-review",
                   detail="" if stands else "no independent review yet")


def fold(ws) -> Queue:
    """The queue as the log stands."""
    queue = Queue()
    for row in log(ws).rows():
        _apply(queue, row)
    return queue


def _clean(text: str, what: str) -> str:
    if len(text) > MAX_TEXT or "\n" in text:
        raise ValueError(f"{what} is one line of at most {MAX_TEXT} characters")
    return text.strip()


def _write(ws, body: dict[str, Any]) -> dict[str, Any]:
    with held(ws.base / "landing.lock"):
        return log(ws).append(body)


def request(ws, token: str, fields: dict[str, Any]) -> dict[str, Any]:
    """Queue a branch at an exact commit; the requester is the authenticated identity."""
    who = ws.auth(token)
    ws._may(who, "send")
    why = eligible(ws, who)
    if why:
        ws.audit("land.refused", who.id, reason=why)
        raise Denied(why)
    branch, sha, target = fields["branch"], fields["sha"], fields.get("target") or development_branch()
    if not NAME.fullmatch(branch) or branch in ("main", "master") or not NAME.fullmatch(target) \
            or target in ("main", "master"):
        raise ValueError("branch and target must be plain branch names, never main")
    if not SHA.fullmatch(sha):
        raise ValueError("land-request needs the full 40-character commit SHA of the branch tip")
    selectors = [str(s) for s in fields.get("selectors", [])]
    if not selectors or len(selectors) > MAX_SELECTORS or any(len(s) > 300 for s in selectors):
        raise ValueError("name the affected test selectors (at least one, at most 64)")
    row = _write(ws, {"ev": "request", "by": who.id, "branch": branch, "sha": sha, "target": target,
                      "selectors": selectors, "replaces": _clean(fields.get("replaces", ""), "replaces")})
    rid = f"land-{row['seq']}"
    ws.audit("land.request", who.id, id=rid, branch=branch, sha=sha)
    return {"id": rid, "branch": branch, "sha": sha, "status": "needs-review"}


def review(ws, token: str, rid: str, sha: str, verdict: str) -> dict[str, Any]:
    """Record an independent review of exactly ``sha``; the author and its delegates cannot review."""
    who = ws.auth(token)
    ws._may(who, "send")
    queue = fold(ws)
    req = queue.requests.get(rid)
    if req is None or req["status"] in TERMINAL:
        raise ValueError(f"{rid} is not an open request")
    why = eligible(ws, who)
    if why or lineage(who.id, req["by"]):
        raise Denied(why or "an independent reviewer is not the requester or one of its delegates")
    if sha != req["sha"]:
        raise ValueError(f"{rid} is for {req['sha']}, not {sha}; review the exact commit")
    if verdict not in ("accept", "reject"):
        raise ValueError("verdict is accept or reject")
    _write(ws, {"ev": "review", "by": who.id, "req": rid, "verdict": verdict, "sha": sha})
    ws.audit("land.review", who.id, id=rid, verdict=verdict)
    return {"id": rid, "verdict": verdict}


def cancel(ws, token: str, rid: str) -> dict[str, Any]:
    """Cancel an open request: its requester, a person, a lead or the runner."""
    who = ws.auth(token)
    req = fold(ws).requests.get(rid)
    if req is None or req["status"] in TERMINAL:
        raise ValueError(f"{rid} is not an open request")
    if who.id != req["by"] and not controller(ws, who):
        raise Denied("only the requester, a person, a lead or the runner cancels a request")
    _write(ws, {"ev": "cancel", "by": who.id, "req": rid})
    ws.audit("land.cancel", who.id, id=rid)
    return {"id": rid, "status": "cancelled"}


def brake(ws, token: str, pause: bool, reason: str = "") -> dict[str, Any]:
    """Pause or resume the runner; a person, a lead or the runner."""
    who = ws.auth(token)
    if not controller(ws, who):
        raise Denied("only a person, a lead or the runner pauses or resumes landing")
    _write(ws, {"ev": "pause" if pause else "resume", "by": who.id, "reason": _clean(reason, "reason")})
    ws.audit("land.pause" if pause else "land.resume", who.id, reason=reason)
    return {"paused": pause}


def transition(ws, token: str, rid: str, status: str, detail: str = "", **evidence: Any) -> None:
    """The runner records how a request moved; only the identity holding the runner claim may."""
    who = ws.auth(token)
    if not controller(ws, who):
        raise Denied("only the landing runner records request states")
    _write(ws, {"ev": "state", "by": who.id, "req": rid, "status": status,
                "detail": detail[:MAX_TEXT], "evidence": evidence})
    ws.audit("land.state", who.id, id=rid, status=status)


def beat(ws, token: str, what: str) -> None:
    """The runner's progress mark, read by the watchdog and the status page."""
    who = ws.auth(token)
    _write(ws, {"ev": "beat", "by": who.id, "what": what[:120]})


def identity_of(ws, name: str) -> Identity | None:
    """The live identity registered as ``name``, or None when it expired or was revoked."""
    if not ws.registry.role_of(name):
        return None
    info = ws.registry.info(name)
    return Identity(name, info["role"], info["parent"], tuple(info["can"]))


def standing(ws, req: dict[str, Any]) -> str:
    """Why the runner must not land ``req`` now, or an empty string.

    Re-derived from the registry at landing time, not trusted from when the rows were written: the
    requester must still be at the landing level, and an accepting reviewer must still be live,
    at the landing level and independent of the requester, with no rejection standing.
    """
    asker = identity_of(ws, req["by"])
    if asker is None or eligible(ws, asker):
        return "the requester is no longer at the landing level"
    live = {}
    for name, found in req["reviews"].items():
        who = identity_of(ws, name)
        if who is not None and not eligible(ws, who) and not lineage(name, req["by"]):
            live[name] = found["verdict"]
    if "reject" in live.values():
        return "an independent reviewer rejected it"
    return "" if "accept" in live.values() else "needs review: no live independent accept"


def supervisor_state(base: Path) -> dict[str, Any]:
    """What the runner's supervisor last recorded in the workspace at ``base``, with ``alive`` from its pid.

    Empty when no supervisor ever started here.
    """
    try:
        row = json.loads((base / SUPERVISOR_STATUS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    pid = int(row.get("supervisor") or 0)
    return {**row, "alive": pid > 0 and alive(pid)}


def supervisor_line(base: Path) -> str:
    """One line for ``digest --status``: whether the runner is kept alive, and how to start it."""
    now = supervisor_state(base)
    if not now.get("alive"):
        return "Runner supervisor: NOT RUNNING; start it with scripts/land up"
    return f"Runner supervisor: up (pid {now['supervisor']}), {now.get('restarts', 0)} restarts, " \
           f"{now.get('state', '')}; log {now.get('log', '')}"


def status_lines(ws) -> list[str]:
    """The queue and the current gate as plain lines for ``digest --status``."""
    queue = fold(ws)
    owned = ws.who_owns(*runner_claim(ws))
    head = f"Landing queue: {'PAUSED by ' + queue.paused_by if queue.paused else 'running'}; runner " \
           f"{owned['owner'] if owned else 'none'}"
    if queue.beat:
        head += f"; last step {int(ws.clock() - queue.beat['ts'])}s ago: {queue.beat['what']}"
    rows = [f"{r['id']} {r['branch']}@{r['sha'][:8]} by {r['by']}: {r['status']}"
            f"{' - ' + r['detail'] if r['detail'] else ''}"
            for r in queue.requests.values() if r["status"] in OPEN or r["status"] == "needs-human"]
    return [head, supervisor_line(ws.base), *(rows or ["(queue empty)"])]
