"""Who is writing: per-agent capability tokens, minted by a person or the lead."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.files import read_json, write_json
from ml_stack.workspace.chain import held
from ml_stack.workspace.modelid import (
    CLAIMED,
    HISTORY_MAX,
    INHERITED,
    VERIFIED,
    clean_harness,
    clean_model,
)

__all__ = [
    "AGENT",
    "AGENT_MARKERS",
    "CAPS",
    "HUMAN",
    "LEAD",
    "TOKEN_ENV",
    "Denied",
    "Identity",
    "Registry",
    "valid_name",
]

TOKEN_ENV = "ML_STACK_WORKSPACE_TOKEN"  # noqa: S105
PREFIX = "mlws1."
HUMAN, LEAD, AGENT = "human", "lead", "agent"
ROLES = (HUMAN, LEAD, AGENT)
MINTS = {HUMAN: frozenset(ROLES), LEAD: frozenset({AGENT}), AGENT: frozenset()}
AGENT_MARKERS = ("CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE")
RESERVED = frozenset({"*", "all", "everyone", "workspace", "system", "human", "owner-token"})
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,47}$")
VERSION = 2
CAPS = ("send", "read", "claim")
"""What a delegated identity may be allowed to do; a top-level identity holds all of them."""


class Denied(PermissionError):
    """The token is missing, wrong, expired or revoked, or its role may not do this."""


@dataclass(frozen=True, slots=True)
class Identity:
    """An authenticated sender: its id and role."""

    id: str
    role: str
    parent: str = ""
    can: tuple[str, ...] = CAPS

    @property
    def trust(self) -> str:
        """The trust level of what this sender writes."""
        return "human" if self.role == HUMAN else "agent-claimed"


def valid_name(name: str) -> bool:
    """Whether ``name`` can be an agent id."""
    return bool(NAME.match(name)) and name not in RESERVED and ".." not in name


def valid_id(name: str) -> bool:
    """Whether ``name`` is an agent id, or a delegated ``parent/child`` id."""
    head, slash, tail = name.partition("/")
    return valid_name(head) and (not slash or valid_name(tail))


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


class Registry:
    """The agents that may write, held as ``agents.json`` with a hash of each secret."""

    def __init__(self, base: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = base / "agents.json"
        self.clock = clock

    def _load(self) -> dict[str, dict[str, Any]]:
        data = read_json(self.path, {})
        agents = data.get("agents") if isinstance(data, dict) else None
        if not isinstance(agents, dict):
            return {}
        if int(data.get("version", 1)) < 2:
            agents = {n: {"model": "", "harness": "", "model_state": "", "models": [], **e}
                      for n, e in agents.items()}
        return dict(agents)

    def _save(self, agents: Mapping[str, Any]) -> None:
        write_json(self.path, {"version": VERSION, "agents": dict(agents)})
        self.path.chmod(0o600)

    def ids(self) -> list[str]:
        """Every registered id."""
        return sorted(self._load())

    def info(self, name: str) -> dict[str, Any]:
        """``name``'s role, expiry and whether it was revoked; no secret, no hash."""
        entry = self._load().get(name, {})
        return {"role": str(entry.get("role", "")), "expires": float(entry.get("expires", 0.0)),
                "revoked": bool(entry.get("revoked", not entry)), "parent": entry.get("parent", ""),
                "can": list(entry.get("can", CAPS)), "project": dict(entry.get("project", {})),
                "model": str(entry.get("model", "")), "harness": str(entry.get("harness", "")),
                "model_state": str(entry.get("model_state", "")),
                "models": list(entry.get("models", [])),
                "label_models": dict(entry.get("label_models", {}))}

    def model_of(self, name: str, label: str = "") -> tuple[str, str]:
        """``(model, state)`` shown for ``name`` (and its helper ``label``): the label's own
        model, else the identity's own, else its parent's marked inherited; ``("", "")`` when none."""
        agents = self._load()
        entry = agents.get(name, {})
        if label and entry.get("label_models", {}).get(label):
            return str(entry["label_models"][label]), CLAIMED
        if entry.get("model"):
            return str(entry["model"]), INHERITED if label else str(entry.get("model_state") or CLAIMED)
        parent = agents.get(str(entry.get("parent") or (name.partition("/")[0] if "/" in name else "")), {})
        if label and parent.get("label_models", {}).get(label):
            return str(parent["label_models"][label]), CLAIMED
        if parent.get("model") and (label or "/" in name):
            return str(parent["model"]), INHERITED
        return "", ""

    def record_model(self, name: str, model: str, harness: str, state: str, *,
                     label: str = "") -> tuple[str, str]:
        """Write ``name``'s model (or its helper ``label``'s) and return the ``(model, state)``
        it held before; appends to the history when the model or its state changed. No
        permission check here: callers decide who may."""
        if state not in (VERIFIED, CLAIMED):
            raise ValueError(f"a recorded model is {VERIFIED} or {CLAIMED}")
        model, harness = clean_model(model) if model else "", clean_harness(harness)
        with held(self.path.with_name("agents.lock")):
            agents = self._load()
            entry = agents.get(name)
            if entry is None or entry.get("revoked"):
                raise ValueError(f"no agent called {name}")
            if label:
                if len(entry.setdefault("label_models", {})) >= 20 and label not in entry["label_models"]:
                    raise ValueError("an agent keeps at most 20 helper models")
                entry["label_models"][label] = model
                self._save(agents)
                return "", ""
            before = (str(entry.get("model", "")), str(entry.get("model_state", "")))
            if not model:
                entry["harness"] = harness
                self._save(agents)
                return before
            entry["model"], entry["model_state"] = model, state
            if harness:
                entry["harness"] = harness
            if before != (model, state):
                history = [*entry.get("models", []), {"model": model, "verified": state == VERIFIED,
                                                      "since": self.clock()}]
                entry["models"] = history[-HISTORY_MAX:]
            self._save(agents)
            return before

    def _live(self, agents: Mapping[str, Any], entry: Mapping[str, Any] | None) -> bool:
        if not entry or entry.get("revoked"):
            return False
        if entry.get("expires") and self.clock() > float(entry["expires"]):
            return False
        return not entry.get("parent") or self._live(agents, agents.get(entry["parent"]))

    def role_of(self, name: str) -> str:
        """The role registered under ``name``, or an empty string."""
        agents = self._load()
        entry = agents.get(name)
        return str(entry["role"]) if entry and self._live(agents, entry) else ""

    def set_project(self, by: Identity, name: str, project: dict[str, str]) -> None:
        """Record the project ``name`` is connected for; only a human identity may."""
        if by.role != HUMAN:
            raise Denied("only a person's setup sets an agent's project")
        with held(self.path.with_name("agents.lock")):
            agents = self._load()
            if name not in agents:
                raise ValueError(f"no agent called {name}")
            agents[name]["project"] = dict(project)
            self._save(agents)

    def readers(self) -> list[str]:
        """The live identities that read every board: a person or a lead."""
        agents = self._load()
        return sorted(n for n, e in agents.items() if e.get("role") != AGENT and self._live(agents, e))

    def within(self, minter: str, minted: int, live: int) -> None:
        """`Denied` when ``minter`` has minted ``minted`` live identities or the workspace
        holds ``live`` live top-level ones."""
        agents = self._load()
        mine = sum(1 for e in agents.values() if e.get("minted_by") == minter
                   and not e.get("parent") and self._live(agents, e))
        every = sum(1 for e in agents.values() if not e.get("parent") and self._live(agents, e))
        if mine >= minted or every >= live:
            raise Denied(f"{minter} holds {mine} live minted identities ({every} in all); "
                         f"the limits are {minted} and {live}")

    def children(self, parent: str) -> list[str]:
        """The live delegated identities of ``parent``."""
        agents = self._load()
        return sorted(n for n, e in agents.items()
                      if e.get("parent") == parent and self._live(agents, e))

    def init(self, name: str, env: Mapping[str, str] | None = None, ttl_s: float = 0.0) -> str:
        """Register the first, human identity and return its token. Refused when an agent
        started this process or when an identity exists already."""
        environment = os.environ if env is None else env
        found = [m for m in AGENT_MARKERS if environment.get(m)]
        if found:
            raise Denied(f"init needs a person at a terminal; {found[0]} says an agent started this")
        with held(self.path.with_name("agents.lock")):
            if self._load():
                raise Denied("the workspace is initialised already")
            return self._add(HUMAN, name, HUMAN, ttl_s)

    def _add(self, minter: str, name: str, role: str, ttl_s: float) -> str:
        if not valid_name(name):
            raise ValueError(f"{name!r} is not a usable agent id (a-z, 0-9, . _ -; up to 48)")
        agents = self._load()
        if name in agents and self._live(agents, agents[name]):
            raise ValueError(f"{name} is registered already")
        secret = secrets.token_urlsafe(32)
        now = self.clock()
        agents[name] = {"role": role, "hash": _hash(secret), "created": now, "minted_by": minter,
                        "expires": now + ttl_s if ttl_s else 0.0, "revoked": False}
        self._save(agents)
        return f"{PREFIX}{name}.{secret}"

    def mint(self, by: Identity, name: str, role: str, ttl_s: float) -> str:
        """A new token for ``name``, if ``by`` may mint that role."""
        if role not in ROLES:
            raise ValueError(f"role must be one of {', '.join(ROLES)}")
        if role not in MINTS[by.role]:
            raise Denied(f"a {by.role} token cannot mint a {role} token")
        with held(self.path.with_name("agents.lock")):
            return self._add(by.id, name, role, ttl_s)

    def revoke(self, by: Identity, name: str) -> None:
        """Stop ``name``'s token working, if ``by`` may mint that role."""
        with held(self.path.with_name("agents.lock")):
            agents = self._load()
            entry = agents.get(name)
            if entry is None:
                raise ValueError(f"no agent called {name}")
            mine = entry.get("parent") == by.id
            if str(entry["role"]) not in MINTS[by.role] and name != by.id and not mine:
                raise Denied(f"a {by.role} token cannot revoke a {entry['role']} token")
            entry["revoked"] = True
            self._save(agents)

    def delegate(self, by: Identity, name: str, ttl_s: float, can: tuple[str, ...],
                 most: int) -> str:
        """A token for ``by``'s child ``by.id/name``: it holds at most ``by``'s rights, lasts no
        longer than ``by``'s token, and cannot delegate."""
        if by.parent or by.role not in (AGENT, LEAD):
            raise Denied("only a top-level agent or lead token delegates")
        if not valid_name(name):
            raise ValueError(f"{name!r} is not a usable agent id (a-z, 0-9, . _ -; up to 48)")
        wanted = tuple(dict.fromkeys(can)) or by.can
        if not set(wanted) <= set(by.can) or not set(wanted) <= set(CAPS):
            raise Denied(f"a child can be given {', '.join(by.can)} at most")
        child = f"{by.id}/{name}"
        with held(self.path.with_name("agents.lock")):
            agents = self._load()
            if self._live(agents, agents.get(child)):
                raise ValueError(f"{child} is registered already")
            if len(self.children(by.id)) >= most:
                raise Denied(f"{by.id} has {most} live delegates already")
            now = self.clock()
            stop = float(agents[by.id].get("expires", 0.0))
            secret = secrets.token_urlsafe(32)
            agents[child] = {"role": AGENT, "hash": _hash(secret), "created": now,
                             "minted_by": by.id, "parent": by.id, "can": list(wanted),
                             "expires": min(now + ttl_s, stop) if stop else now + ttl_s,
                             "revoked": False}
            self._save(agents)
        return f"{PREFIX}{child}.{secret}"

    def authenticate(self, token: str) -> Identity:
        """The identity ``token`` stands for; raises `Denied` for anything else."""
        head, _, secret = token.removeprefix(PREFIX).rpartition(".") if token.startswith(PREFIX) \
            else ("", "", "")
        agents = self._load()
        entry = agents.get(head) if secret else None
        good = bool(entry) and hmac.compare_digest(_hash(secret), str(entry["hash"]))
        if not good or entry is None:
            raise Denied("the token is not recognised")
        if entry.get("revoked"):
            raise Denied("the token was revoked")
        if entry.get("expires") and self.clock() > float(entry["expires"]):
            raise Denied("the token expired")
        parent = str(entry.get("parent", ""))
        if parent and not self._live(agents, agents.get(parent)):
            raise Denied("the token's parent was revoked or expired")
        return Identity(head, str(entry["role"]), parent, tuple(entry.get("can", CAPS)))
