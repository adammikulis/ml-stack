"""Who is writing: per-agent capability tokens, minted by a person."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poolhouse.files import read_json, writing
from poolhouse.windows_private import problem as windows_problem, restrict
from poolhouse.workspace import device_metadata
from poolhouse.workspace.chain import held
from poolhouse.workspace.modelid import (
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
    "BoardUnavailable",
    "Denied",
    "Identity",
    "Registry",
    "valid_name",
]

TOKEN_ENV = "POOLHOUSE_WORKSPACE_TOKEN"  # noqa: S105
PREFIX = "mlws1."
HUMAN, LEAD, AGENT = "human", "lead", "agent"
ROLES = (HUMAN, LEAD, AGENT)
MINTS = {HUMAN: frozenset(ROLES), LEAD: frozenset(), AGENT: frozenset()}
AGENT_MARKERS = ("CLAUDECODE", "POOLHOUSE_AGENT", "POOLHOUSE_NONINTERACTIVE")
RESERVED = frozenset({"*", "all", "everyone", "workspace", "system", "human", "owner-token"})
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,47}$")
VERSION = 2
CAPS = ("send", "read", "claim")
"""What a delegated identity may be allowed to do; a top-level identity holds all of them."""


class Denied(PermissionError):
    """The token is missing, wrong, expired or revoked, or its role may not do this."""


class BoardUnavailable(Denied):
    """The project board could not be reached or authenticated: local work carries on with a
    warning, and the claim records attribute once it is back."""


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


def _outranks(stored: dict, incoming: dict) -> bool:
    """Whether a stored record rests on observation or an authenticated pairing that an agent's
    own report must not replace. A report that is itself paired by the daemon may refresh it."""
    if incoming.get('verification') != 'agent-reported' or incoming.get('peer_verification') == 'paired':
        return False
    return stored.get('verification') in ('paired', 'local-observed') or stored.get('peer_verification') == 'paired'


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
        base = self.path.parent
        self._storage()
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == 'nt':
            restrict(base)
            if self.path.exists():
                restrict(self.path)
        else:
            if base.stat().st_uid != os.getuid() or (self.path.exists() and self.path.stat().st_uid != os.getuid()):
                raise Denied('agent registry storage belongs to another user')
            base.chmod(0o700)
        with writing(self.path) as tmp:
            restrict(tmp) if os.name == 'nt' else tmp.chmod(0o600)
            tmp.write_text(json.dumps({"version": VERSION, "agents": dict(agents)},
                                      indent=2, ensure_ascii=False), encoding='utf-8')

    def _storage(self):
        owned = False
        for path in (self.path, *self.path.parents):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode) or (os.name == 'nt'
                    and info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                raise Denied('agent registry storage cannot be a redirected path')
            if path == self.path and not stat.S_ISREG(info.st_mode):
                raise Denied('agent registry storage requires a plain file')
            if path == self.path.parent and not stat.S_ISDIR(info.st_mode):
                raise Denied('agent registry storage requires a plain directory')
            if path in (self.path, self.path.parent) or not owned:
                if os.name == 'nt':
                    reason = windows_problem(path)
                    if reason not in ('', 'has unrestricted Windows access',
                                      'Windows permissions grant access to another account'):
                        raise Denied('agent registry storage ownership could not be verified')
                elif info.st_uid != os.getuid():
                    raise Denied('agent registry storage belongs to another user')
                owned = True

    def ids(self) -> list[str]:
        """Every registered id."""
        return sorted(self._load())

    def info(self, name: str) -> dict[str, Any]:
        """``name``'s role, expiry and whether it was revoked; no secret, no hash."""
        agents = self._load()
        entry = agents.get(name, {})
        parent_name = entry.get('parent', '')
        parent = agents.get(parent_name, {}) if parent_name != name else {}
        device = dict(entry.get('device', {}))
        parent_device = dict(parent.get('device', {}))
        if parent and (not device or device.get('inherited_from') == parent_name):
            device = device_metadata.inherited(parent_device, parent_name)
        else:
            device = device_metadata.normalize(device)
        harness = str(entry.get('harness', ''))
        harness_state = str(entry.get('harness_state', entry.get('model_state', '') if harness else ''))
        if not harness and parent.get('harness'):
            harness, harness_state = str(parent['harness']), INHERITED
        return {"role": str(entry.get("role", "")), "expires": float(entry.get("expires", 0.0)),
                "revoked": bool(entry.get("revoked", not entry)), "parent": entry.get("parent", ""),
                "can": list(entry.get("can", CAPS)), "project": dict(entry.get("project", {})),
                "depth": int(entry.get("depth", 0)), "invited_by": str(entry.get("invited_by", "")),
                "strikes": int(entry.get("strikes", 0)),
                "model": str(entry.get("model", "")), "harness": harness,
                "model_state": str(entry.get("model_state", "")),
                "harness_state": harness_state,
                "device": device,
                "models": list(entry.get("models", [])),
                "created": float(entry.get("created", 0)), "presentation": dict(entry.get("presentation", {}))}

    def model_of(self, name: str) -> tuple[str, str]:
        """``(model, state)`` shown for ``name``: its own, else its parent's marked inherited;
        ``("", "")`` when none."""
        agents = self._load()
        entry = agents.get(name, {})
        if entry.get("model"):
            return str(entry["model"]), str(entry.get("model_state") or CLAIMED)
        parent = agents.get(str(entry.get("parent") or (name.partition("/")[0] if "/" in name else "")), {})
        if parent.get("model") and (entry.get("parent") or "/" in name):
            return str(parent["model"]), INHERITED
        return "", ""

    def ensure_presentation(self, name: str) -> None:
        """Assign a stable readable ordinal, including expired and revoked records."""
        with held(self.path.with_name('agents.lock')):
            agents = self._load()
            entry = agents.get(name)
            if not entry or entry.get('role') != AGENT or entry.get('parent') or entry.get('presentation'):
                return
            device = entry.get('device', {}).get('device_id')
            peers = [row.get('presentation', {}) for row in agents.values()]
            ordinal = max((row.get('ordinal', 0) for row in peers), default=0) + 1
            entry['presentation'] = {'device': device, 'ordinal': ordinal, 'kind': 'unknown'}
            self._save(agents)

    def register_session(self, token: str, harness: str = "") -> None:
        """An actor's main-session presentation; never adds rights or removes parentage."""
        clean_harness(harness)
        self.ensure_presentation(self.authenticate(token).id)
        with held(self.path.with_name('agents.lock')):
            who = self.authenticate(token)
            if who.role != AGENT or who.parent:
                raise Denied('only a top-level standard agent registers a main session')
            agents = self._load()
            entry = agents[who.id]
            if harness:
                confidence = entry.get('harness_state', entry.get('model_state', '') if entry.get('harness') else '')
                trusted = confidence == VERIFIED
                if trusted and entry.get('harness') and entry['harness'] != harness:
                    raise Denied('the harness was recorded by its launcher; a report cannot replace it')
                entry['harness'] = harness
                entry['harness_state'] = VERIFIED if trusted else CLAIMED
            presentation = entry.setdefault('presentation', {})
            presentation['kind'] = 'main'
            self._save(agents)

    def record_device_claim(self, token: str, metadata: dict) -> None:
        """Record an authenticated actor's device report without granting device authority."""
        who = self.authenticate(token)
        self._record_device(who.id, device_metadata.reported(metadata, observed_at=self.clock()))

    def record_profile(self, token: str, device: dict | None = None, harness: str = '') -> dict:
        """Record the authenticated agent's own reported device and harness facts in one step."""
        clean_harness(harness)
        reported = None if device is None else device_metadata.reported(device, observed_at=self.clock())
        with held(self.path.with_name('agents.lock')):
            who = self.authenticate(token)
            if who.role != AGENT:
                raise Denied('profile reporting requires an agent capability')
            agents = self._load()
            entry = agents[who.id]
            state = entry.get('harness_state', entry.get('model_state', '') if entry.get('harness') else '')
            if harness and state == VERIFIED and entry.get('harness') != harness:
                raise Denied('the harness was recorded by its launcher; a report cannot replace it')
            if reported is not None and not _outranks(entry.get('device', {}), reported):
                entry['device'] = reported
            if harness:
                entry['harness'] = harness
                entry['harness_state'] = VERIFIED if state == VERIFIED else CLAIMED
            self._save(agents)
        return {'id': who.id, **self.info(who.id)}

    def _record_device(self, name: str, metadata: dict) -> None:
        """Persist normalized device provenance supplied by the trusted registration adapter."""
        device = device_metadata.normalize(metadata)
        with held(self.path.with_name('agents.lock')):
            agents = self._load()
            if not self._live(agents, agents.get(name)):
                raise Denied('device provenance requires a live registered actor')
            if _outranks(agents[name].get('device', {}), device):
                return
            if agents[name].get('device') != device:
                agents[name]['device'] = device
                self._save(agents)

    def record_model(self, name: str, model: str, harness: str, state: str) -> tuple[str, str]:
        """Write ``name``'s model and return the ``(model, state)`` it held before; appends to the history when the model or its state changed. No
        permission check here: callers decide who may."""
        if state not in (VERIFIED, CLAIMED):
            raise ValueError(f"a recorded model is {VERIFIED} or {CLAIMED}")
        model, harness = clean_model(model) if model else "", clean_harness(harness)
        with held(self.path.with_name("agents.lock")):
            agents = self._load()
            entry = agents.get(name)
            if entry is None or entry.get("revoked"):
                raise ValueError(f"no agent called {name}")
            before = (str(entry.get("model", "")), str(entry.get("model_state", "")))
            if not model:
                entry["harness"] = harness
                entry["harness_state"] = state if harness else ""
                self._save(agents)
                return before
            entry["model"], entry["model_state"] = model, state
            if harness:
                entry["harness"] = harness
                entry["harness_state"] = state
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

    @staticmethod
    def _below(agents: Mapping[str, Any], name: str) -> list[str]:
        found: list[str] = []
        todo = [name]
        while todo:
            head = todo.pop()
            for n, e in sorted(agents.items()):
                if e.get("parent") == head and n not in found:
                    found.append(n)
                    todo.append(n)
        return found

    @staticmethod
    def _root(agents: Mapping[str, Any], name: str) -> str:
        seen = {name}
        while (up := str(agents.get(name, {}).get("parent", ""))) and up not in seen:
            seen.add(up)
            name = up
        return name

    def descendants(self, name: str, live: bool = False) -> list[str]:
        """Every identity below ``name`` (children, their children, ...), ``live`` ones only if asked."""
        agents = self._load()
        return [n for n in self._below(agents, name) if not live or self._live(agents, agents[n])]

    def root_of(self, name: str) -> str:
        """The top-level identity ``name`` descends from (``name`` itself when it has no parent)."""
        return self._root(self._load(), name)

    def adopt(self, parent: str, name: str, ttl_s: float, can: tuple[str, ...], limits: tuple[int, int, int]) -> str:
        """A token for ``name``, a standard agent that is ``parent``'s child: it holds at most
        ``parent``'s rights and lasts no longer than ``parent``'s token. ``limits`` is the most
        live children of ``parent``, live descendants of its root and live identities in all;
        `Denied` names the one that is full."""
        most_children, most_tree, most_live = limits
        if not valid_name(name):
            raise ValueError(f"{name!r} is not a usable agent id (a-z, 0-9, . _ -; up to 48)")
        with held(self.path.with_name("agents.lock")):
            agents = self._load()
            top = agents.get(parent)
            if top is None or not self._live(agents, top):
                raise Denied(f"{parent} is revoked or expired")
            if self._live(agents, agents.get(name)):
                raise ValueError(f"{name} is registered already")
            rights = tuple(c for c in CAPS if c in top.get("can", CAPS) and c in can)
            kids = sum(1 for e in agents.values() if e.get("parent") == parent and self._live(agents, e))
            if kids >= most_children:
                raise Denied(f"{parent} has {kids} live children; the limit is {most_children}")
            root = self._root(agents, parent)
            tree = sum(1 for n in self._below(agents, root) if self._live(agents, agents[n]))
            if tree >= most_tree:
                raise Denied(f"{root} has {tree} live descendants; the limit is {most_tree}")
            every = sum(1 for e in agents.values() if self._live(agents, e))
            if every >= most_live:
                raise Denied(f"the workspace holds {every} live identities; the limit is {most_live}")
            now = self.clock()
            stop = float(top.get("expires", 0.0))
            secret = secrets.token_urlsafe(32)
            agents[name] = {"role": AGENT, "hash": _hash(secret), "created": now, "minted_by": parent,
                            "parent": parent, "invited_by": parent, "can": list(rights),
                            "depth": int(top.get("depth", 0)) + 1,
                            "expires": min(now + ttl_s, stop) if stop else now + ttl_s, "revoked": False}
            self._save(agents)
        return f"{PREFIX}{name}.{secret}"

    def strike(self, name: str) -> int:
        """Count one misbehaviour of ``name`` against its parent; the parent's strikes so far."""
        with held(self.path.with_name("agents.lock")):
            agents = self._load()
            top = agents.get(str(agents.get(name, {}).get("parent", "")))
            if top is None:
                return 0
            top["strikes"] = int(top.get("strikes", 0)) + 1
            self._save(agents)
            return int(top["strikes"])

    def revoke_tree(self, by: Identity, name: str) -> list[str]:
        """Revoke ``name`` and everything below it, if ``by`` may revoke ``name``; the names revoked."""
        self.revoke(by, name)
        with held(self.path.with_name("agents.lock")):
            agents = self._load()
            below = self._below(agents, name)
            for n in below:
                agents[n]["revoked"] = True
            self._save(agents)
        return [name, *below]

    def children(self, parent: str) -> list[str]:
        """The live delegated identities of ``parent``."""
        agents = self._load()
        return sorted(n for n, e in agents.items()
                      if e.get("parent") == parent and self._live(agents, e))

    def init(self, name: str, env: Mapping[str, str] | None = None, ttl_s: float = 0.0) -> str:
        """Register the first human identity; refuse agent processes and existing humans."""
        environment = os.environ if env is None else env
        found = [m for m in AGENT_MARKERS if environment.get(m)]
        if found:
            raise Denied(f"init needs a person at a terminal; {found[0]} says an agent started this")
        with held(self.path.with_name("agents.lock")):
            if any(entry.get("role") == HUMAN for entry in self._load().values()):
                raise Denied("the workspace is initialised already")
            return self._add(HUMAN, name, HUMAN, ttl_s)

    def bootstrap_agent(self, name: str, project: dict[str, str], limits: tuple[int, int]) -> str:
        """Create or recover a local standard agent for one project and return its token."""
        if not project.get("key"):
            raise Denied("local agent initialization needs a project")
        with held(self.path.with_name("agents.lock")):
            agents = self._load()
            entry = agents.get(name)
            if entry and entry.get("session_device"):
                raise Denied("device sessions recover only through their enrolled device authority")
            if not self._live(agents, entry):
                self.within(str((entry or {}).get("minted_by", "local-account")), *limits)
            if entry:
                if entry.get("role") != AGENT or entry.get("parent") or entry.get("revoked"):
                    raise Denied("local recovery needs an active top-level standard agent")
                if entry.get("project", {}).get("key") != project["key"]:
                    raise Denied("this identity belongs to another project")
                secret = secrets.token_urlsafe(32)
                entry.update(hash=_hash(secret), expires=0.0)
                self._save(agents)
                return f"{PREFIX}{name}.{secret}"
            token = self._add("local-account", name, AGENT, 0.0)
            agents = self._load()
            agents[name]["project"] = dict(project)
            self._save(agents)
            return token

    def _add(self, minter: str, name: str, role: str, ttl_s: float) -> str:
        if not valid_name(name):
            raise ValueError(f"{name!r} is not a usable agent id (a-z, 0-9, . _ -; up to 48)")
        agents = self._load()
        if name in agents and self._live(agents, agents[name]):
            raise ValueError(f"{name} is registered already")
        secret = secrets.token_urlsafe(32)
        now = self.clock()
        agents[name] = {"role": role, "hash": _hash(secret), "created": now, "minted_by": minter,
                        "expires": now + ttl_s if ttl_s else 0.0, "revoked": False,
                        "device": device_metadata.current() if minter == HUMAN else dict(agents.get(minter, {}).get("device", {}))}
        self._save(agents)
        return f"{PREFIX}{name}.{secret}"

    def enroll_project(self, name: str, project: dict[str, str], ttl_s: float) -> tuple[str, str]:
        """Atomically issue a new bounded agent capability for a project."""
        if not valid_name(name):
            raise ValueError("invalid project agent name")
        if (not isinstance(project, dict) or not re.fullmatch("[a-f0-9]{32}", project.get("key", ""))
                or not isinstance(project.get("name"), str) or len(project["name"]) > 256
                or set(project) != {"key", "name", "cluster", "cluster_id"}
                or not isinstance(project.get("cluster"), str) or not 1 <= len(project["cluster"]) <= 128
                or not isinstance(project.get("cluster_id"), str)
                or not re.fullmatch("[a-f0-9]{64}", project["cluster_id"])):
            raise ValueError("invalid project capability scope")
        if not math.isfinite(ttl_s) or not 0 < ttl_s <= 30 * 86_400:
            raise ValueError("project agent lifetime must be between zero and 30 days")
        with held(self.path.with_name("agents.lock")):
            agents = self._load()
            if sum(self._live(agents, entry) for entry in agents.values()) >= 64:
                raise Denied("the workspace holds 64 live identities")
            wanted = name
            while name in agents:
                name = f"{wanted[:40]}-{secrets.token_hex(3)}"
            secret = secrets.token_urlsafe(32)
            now = self.clock()
            agents[name] = {"role": AGENT, "hash": _hash(secret), "created": now,
                            "minted_by": "dev-cluster", "expires": now + ttl_s,
                            "revoked": False, "project": dict(project), "can": list(CAPS)}
            self._save(agents)
        return name, f"{PREFIX}{name}.{secret}"

    def renew_dev_project(self, token: str, project_id: str, cluster: str, cluster_id: str) -> Identity:
        """Bind a live automatic project capability to its authenticated Dev cluster."""
        if (not re.fullmatch("[a-f0-9]{32}", project_id)
                or not isinstance(cluster, str) or not 1 <= len(cluster) <= 128
                or not re.fullmatch("[a-f0-9]{64}", cluster_id)):
            raise ValueError("invalid project cluster scope")
        with held(self.path.with_name("agents.lock")):
            who = self.authenticate(token)
            agents = self._load()
            entry = agents[who.id]
            scope = entry.get("project", {})
            if (who.role != AGENT or who.parent or entry.get("session_device")
                    or entry.get("minted_by") != "dev-cluster"
                    or scope.get("key") != project_id or not scope.get("cluster_id")):
                raise Denied("cluster renewal requires this project's live automatic Dev agent")
            entry["project"] = {**scope, "cluster": cluster, "cluster_id": cluster_id}
            self._save(agents)
            return who

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

    def revoke_self(self, token: str, cleanup: Callable[[Identity], None]) -> Identity:
        """Revoke the current agent capability and clean its resources under the identity lock."""
        with held(self.path.with_name("agents.lock")):
            who = self.authenticate(token)
            if who.role != AGENT:
                raise Denied("self revocation requires an agent capability")
            agents = self._load()
            agents[who.id]["revoked"] = True
            self._save(agents)
            cleanup(who)
            return who

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
            if sum(self._live(agents, entry) for entry in agents.values()) >= 64:
                raise Denied("the workspace holds 64 live identities")
            if len(self.children(by.id)) >= most:
                raise Denied(f"{by.id} has {most} live delegates already")
            now = self.clock()
            stop = float(agents[by.id].get("expires", 0.0))
            secret = secrets.token_urlsafe(32)
            agents[child] = {"role": AGENT, "hash": _hash(secret), "created": now,
                             "minted_by": by.id, "parent": by.id, "can": list(wanted),
                             "expires": min(now + ttl_s, stop) if stop else now + ttl_s,
                             "revoked": False,
                             "device": device_metadata.inherited(agents[by.id].get("device", {}), by.id)}
            self._save(agents)
        return f"{PREFIX}{child}.{secret}"

    def authenticate(self, token: str) -> Identity:
        """The identity ``token`` stands for; raises `Denied` for anything else."""
        if not isinstance(token, str):
            raise Denied("the token is not recognised")
        head, _, secret = token.removeprefix(PREFIX).rpartition(".") if token.startswith(PREFIX) \
            else ("", "", "")
        agents = self._load()
        entry = agents.get(head) if secret else None
        good = bool(entry) and isinstance(entry.get("hash"), str) and hmac.compare_digest(
            _hash(secret), entry["hash"])
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
