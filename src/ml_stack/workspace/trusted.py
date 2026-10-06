"""A trusted machine: a person marks this OS user's machine once, and an agent here joins the
workspace by name, with the standard agent role, without an invite code."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml_stack.sentinel import human
from ml_stack.workspace import onboard, tokens
from ml_stack.workspace.agent_invites import TAINT_ENV, announce_milestone
from ml_stack.workspace.identity import AGENT, LEAD, Denied, Identity
from ml_stack.workspace.rates import RateLimited
from ml_stack.workspace.screen import Refused

if TYPE_CHECKING:
    from ml_stack.workspace.service import Workspace

__all__ = ["TRUST_FILE", "join", "path", "trust", "untrust"]

TRUST_FILE = "trusted-machine"
TRUSTED = Identity("trusted-machine", LEAD)
SECRET = re.compile(r"[0-9a-f]{64}")
UNTRUSTED = ("this machine is not trusted for a join without a code: ask the person for a paste "
             "block from `ml-stack-workspace connect`, or to run `ml-stack-workspace trust-machine` "
             "at their own terminal")


def path(base: Path) -> Path:
    """The trust file under the workspace directory ``base``, registered with sentinel."""
    found = base / TRUST_FILE
    human.protect(found)
    return found


def _fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()[:12]


def _untainted(what: str, env: Mapping[str, str]) -> None:
    if env.get(TAINT_ENV):
        raise Denied(f"{what} is refused while ${TAINT_ENV} is set: this session read untrusted "
                     f"content; ask the person for a code from `ml-stack-workspace connect`")


def trust(ws: Workspace, project: dict[str, str], *, terminal: tuple[bool, bool] | None = None,
          env: Mapping[str, str] | None = None) -> Path:
    """Trust this OS user's machine for joins without a code, for ``project`` only when it is not
    empty; initialises the workspace when needed. A person at a terminal only. The trust file."""
    human.require_person("workspace trust-machine", terminal, env)
    _untainted("trust-machine", os.environ if env is None else env)
    tokens.prepare(ws.base)
    if not ws.registry.ids():
        tokens.store(ws.base, tokens.OWNER_FILE, ws.init("owner"))
    secret = secrets.token_hex(32)
    target = tokens.write_private(path(ws.base), json.dumps(
        {"secret": secret, "project": dict(project), "since": ws.clock()}) + "\n")
    ws.audit("trusted_machine.on", onboard.SETUP.id, project=project.get("name", ""),
             fingerprint=_fingerprint(secret))
    return target


def untrust(ws: Workspace, *, terminal: tuple[bool, bool] | None = None,
            env: Mapping[str, str] | None = None) -> bool:
    """Stop joins without a code; agents that joined keep their tokens. A person at a terminal
    only. Whether the machine was trusted."""
    human.require_person("workspace untrust-machine", terminal, env)
    target = path(ws.base)
    was = tokens.problem(target) != "missing"
    target.unlink(missing_ok=True)
    if was:
        ws.audit("trusted_machine.off", onboard.SETUP.id)
    return was


def _proof(base: Path) -> dict[str, Any]:
    target = path(base)
    why = tokens.problem(target)
    if why == "missing":
        raise Denied(UNTRUSTED)
    why = why or tokens.problem(base)
    if why:
        raise Denied(f"the trust file {target}: {why}; the person runs "
                     f"`ml-stack-workspace trust-machine` again")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict) or not SECRET.fullmatch(str(data.get("secret", ""))):
        raise Denied(f"the trust file {target} is damaged; the person runs "
                     f"`ml-stack-workspace trust-machine` again")
    return data


def _reusable(ws: Workspace, name: str) -> bool:
    if not ws.registry.role_of(name):
        return False
    try:
        who = ws.auth(tokens.load(ws.base, name))
    except Denied:
        return False
    return who.id == name and who.role == AGENT and not who.parent


def join(ws: Workspace, wanted: str, claim: tuple[str, str] = ("", ""), *,
         here: dict[str, str] | None = None, env: Mapping[str, str] | None = None) -> tuple[str, bool]:
    """Join as ``wanted`` without a code on a trusted machine, working in the project ``here``:
    ``(name, reused)``, where ``reused`` means a working token for ``wanted`` was kept. The role
    is always the standard agent role; a taken id is suffixed and a revoked one refused."""
    _untainted("a join without a code", os.environ if env is None else env)
    proof = _proof(ws.base)
    onboard.check_claim(claim)
    name, here = onboard.usable(wanted), dict(here or {})
    if name == "lead":
        raise ValueError("'lead' is given only by the person; pick another short id such as claude-code")
    scope = dict(proof.get("project") or {})
    if scope and scope.get("key") != here.get("key"):
        raise Denied(f"this machine is trusted for project {scope.get('name', '?')} only; "
                     f"join from inside it, or ask the person for a code from `connect`")
    if _reusable(ws, name):
        onboard.record_claim(ws, name, claim)
        return name, True
    if name in ws.registry.ids() and ws.registry.info(name)["revoked"]:
        raise Denied(f"{name} was revoked; only the person brings it back "
                     f"(`ml-stack-workspace setup --rotate {name}`)")
    name = onboard.pick_name(ws, name)
    ws.registry.within(TRUSTED.id, ws.limits.mints_per_identity, ws.limits.agents_live)
    tokens.store(ws.base, name, ws.registry.mint(TRUSTED, name, AGENT, onboard.TOKEN_S))
    if here:
        ws.registry.set_project(onboard.SETUP, name, here)
    ws.board.place(name, here)
    ws.audit("trusted_machine.join", name, project=here.get("name", ""),
             fingerprint=_fingerprint(proof["secret"]))
    onboard.record_claim(ws, name, claim)
    try:
        announce_milestone(ws, f"{name} joined from this trusted machine")
    except (RateLimited, Refused):
        ws.audit("trusted_machine.announce_dropped", name)
    return name, False
