"""Invites made by a joined agent: bounded by limits only the person changes, asked about when
the policy says so, announced on `#announcements`, and revoked with the issuer's tree."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ml_stack import requests
from ml_stack.log import warn
from ml_stack.workspace import tokens
from ml_stack.workspace.boards import ANNOUNCE
from ml_stack.workspace.identity import AGENT, LEAD, Denied, Identity

if TYPE_CHECKING:
    from ml_stack.workspace.service import Workspace

__all__ = ["MAX_DEPTH", "ROLE_ENV", "TAINT_ENV", "adopt", "announce_milestone", "issue", "policy"]

READ_ONLY, APPROVE_FIRST, PLAN_AND_GO = "read-only", "approve-first", "plan-and-go"
RANK = {READ_ONLY: 0, APPROVE_FIRST: 1, PLAN_AND_GO: 2}
ROLE_ENV = "ML_STACK_ROLE"
TAINT_ENV = "ML_STACK_TAINTED"
MAX_DEPTH = 2
HOUR_S = 3_600.0
GREETER = Identity("workspace", AGENT)
Ask = Callable[["Workspace", Identity, str, float], None]


def policy(ws: Workspace, env: Mapping[str, str] | None = None) -> str:
    """How an agent's invite is decided: the person's ``agent_invite_ask`` limit, tightened (never
    loosened) by the caller's ``$ML_STACK_ROLE``, and ``approve-first`` whenever ``$ML_STACK_TAINTED`` is set."""
    env = os.environ if env is None else env
    named = ws.limits.agent_invite_ask
    chosen = named if named in RANK else APPROVE_FIRST
    said = env.get(ROLE_ENV, "")
    if said in RANK and RANK[said] < RANK[chosen]:
        chosen = said
    if env.get(TAINT_ENV) and chosen == PLAN_AND_GO:
        chosen = APPROVE_FIRST
    return chosen


def _over(what: str, have: int | float, cap: int | float, key: str) -> Denied:
    return Denied(f"{what}: {have}, the limit is {cap}; only the person can change it "
                  f"(`{key}` in the workspace's limits.json)")


def announce_milestone(ws: Workspace, text: str) -> None:
    """Post ``text`` to `#announcements` as a workspace milestone."""
    ws.post(GREETER, ANNOUNCE, "milestone", text, subject="milestone", announce=True)


def _request(ws: Workspace, who: Identity, summary: str, wait_s: float) -> None:
    info = ws.registry.info(who.id)
    model, state = ws.registry.model_of(who.id)
    handle = requests.raise_request(requests.Ask(
        "tool_call", summary,
        "A joined agent asked to bring a new agent into the workspace. Nothing it read decides this; you do.",
        ("allow-once", "deny"),
        requests.Origin(who.id, str(info["project"].get("name", "")), "", model, state), ttl=wait_s))
    warn(f"Waiting for person approval: request {handle.id or '(not stored)'}. "
        "Open the app approvals/requests view to allow or deny it; "
        f"timeout {wait_s:.0f} seconds. A person can use `connect --code-only` instead.",
        flush=True)
    if not handle.wait(wait_s).approved:
        raise Denied(f"the person has not approved request {handle.id or '(not stored)'}; "
                     f"no invite was made")


def _checks(ws: Workspace, who: Identity, ttl_s: float, uses: int) -> int:
    """`Denied` unless ``who`` may make an invite now; the depth its joiners would have."""
    lim, reg = ws.limits, ws.registry
    info = reg.info(who.id)
    if who.role not in (AGENT, LEAD) or not reg.role_of(who.id):
        raise Denied("only a joined agent invites an agent; a person uses `connect`")
    if who.parent and not info["invited_by"]:
        raise Denied("a delegated identity cannot invite")
    if info["strikes"] >= lim.agent_invite_strikes:
        raise _over(f"{who.id}'s children had {info['strikes']} messages held or refused",
                    info["strikes"], lim.agent_invite_strikes, "agent_invite_strikes")
    depth = int(info["depth"]) + 1
    cap = min(lim.agent_invite_depth, MAX_DEPTH)
    if depth > cap:
        raise _over(f"a new agent here would be {depth} levels below the person",
                    depth, cap, "agent_invite_depth")
    if not 0 < ttl_s <= lim.agent_invite_ttl_s:
        raise _over(f"an invite lasts {ttl_s:.0f} s", f"{ttl_s:.0f} s", f"{lim.agent_invite_ttl_s:.0f} s",
                    "agent_invite_ttl_s")
    if not 1 <= uses <= lim.agent_invite_uses:
        raise _over(f"an invite for {uses} agents", uses, lim.agent_invite_uses, "agent_invite_uses")
    now = ws.clock()
    made = ws.invites.made_by(who.id)
    open_now = [m for m in made if m["open"]]
    if len(open_now) >= lim.agent_invites_open:
        raise _over(f"{who.id} has {len(open_now)} invites outstanding", len(open_now),
                    lim.agent_invites_open, "agent_invites_open")
    lately = [m for m in made if m["created"] > now - HOUR_S]
    if len(lately) >= lim.agent_invites_per_hour:
        raise _over(f"{who.id} made {len(lately)} invites in the last hour", len(lately),
                    lim.agent_invites_per_hour, "agent_invites_per_hour")
    held = len(reg.children(who.id)) + sum(m["left"] for m in open_now)
    if held + uses > lim.max_children:
        raise _over(f"{who.id} has {held} live children and open places", held + uses,
                    lim.max_children, "max_children")
    tree = len(reg.descendants(reg.root_of(who.id), live=True))
    if tree + uses > lim.agent_tree_live:
        raise _over(f"{reg.root_of(who.id)} has {tree} live descendants", tree + uses,
                    lim.agent_tree_live, "agent_tree_live")
    live = sum(1 for n in reg.ids() if reg.role_of(n))
    if live + uses > lim.agents_live:
        raise _over("the workspace holds live identities", live + uses, lim.agents_live, "agents_live")
    return depth


def issue(ws: Workspace, who: Identity, want: tuple[str, float, int],
          env: Mapping[str, str] | None = None, ask: Ask | None = None) -> dict[str, Any]:
    """A one-time code for a new agent that would join as ``who``'s child; ``want`` is the suggested
    id, the lifetime in seconds and the number of uses. The refusal is audited."""
    hint, ttl_s, uses = want
    ttl_s = ttl_s or ws.limits.invite_ttl_s
    try:
        how = policy(ws, env)
        if how == READ_ONLY:
            raise Denied("this session may only read; it cannot invite an agent")
        depth = _checks(ws, who, ttl_s, uses)
        if how == APPROVE_FIRST:
            summary = (f"invite a new agent as a child of {who.id} "
                       f"({uses} use{'s' if uses > 1 else ''}, {ttl_s / 60:.0f} min)")
            (ask or (lambda w, i, s, t: _request(w, i, s, t)))(ws, who, summary, ws.limits.agent_invite_wait_s)
    except Denied as err:
        ws.audit("agent_invite.refused", who.id, why=str(err)[:120])
        raise
    info = ws.registry.info(who.id)
    code = ws.invites.create(hint, ttl_s, dict(info["project"]), uses, (who.id, tuple(who.can)))
    ws.audit("agent_invite.create", who.id, uses=uses, ttl_s=ttl_s, depth=depth, policy=how)
    announce_milestone(ws, f"{who.id} invited a new agent ({uses} use{'s' if uses > 1 else ''}, "
                  f"{ttl_s / 60:.0f} min)")
    return {"code": code, "uses": uses, "ttl_s": ttl_s, "project": str(info["project"].get("name", ""))}


def adopt(ws: Workspace, issuer: str, name: str, can: tuple[str, ...]) -> None:
    """Register ``name`` as ``issuer``'s child, write its token file, audit and announce it."""
    lim = ws.limits
    made = ws.registry.adopt(issuer, name, lim.child_ttl_s, can,
                             (lim.max_children, lim.agent_tree_live, lim.agents_live))
    tokens.store(ws.base, name, made)
    info = ws.registry.info(name)
    ws.audit("agent_invite.join", name, issuer=issuer, depth=info["depth"], can=info["can"])
    announce_milestone(ws, f"{name} joined as {issuer}'s child")
