"""A local model that joins the workspace as an agent: its record, its status file and who it obeys.

The folder ``local-agents/`` under the workspace holds, per agent NAME, ``NAME.json`` (what the
person asked for), ``NAME.status.json`` (what the loop is doing), ``NAME.stop`` (the kill
switch the loop checks every step) and ``NAME.log``. No file holds a token: that stays in
``tokens/NAME``.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ml_stack import roles
from ml_stack.chatpolicy import APPROVE_FIRST, PLAN_AND_GO, READ_ONLY
from ml_stack.files import read_json, write_json
from ml_stack.serve.process import pid_exists, started_at
from ml_stack.workspace.identity import HUMAN, LEAD, valid_id, valid_name
from ml_stack.workspace.service import Workspace

__all__ = ["APPROVE_FIRST", "DEFAULT_ORDERS_FROM", "ORDER_KINDS", "PLAN_AND_GO", "READ_ONLY",
           "Agent", "Status", "alive", "check_name", "check_orders", "check_project", "folder",
           "load", "names", "obeys", "role_choices", "save", "status_of"]

DEFAULT_ORDERS_FROM = ("claude",)
ORDER_KINDS = ("task", "question")
MOST_ORDERERS = 8
RESERVED = frozenset({"admin", "system", "human", "workspace", "owner", "root", "setup", "agent"})
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f‪-‮⁦-⁩]")


def folder(ws: Workspace) -> Path:
    """The directory of local-agent files, owner only."""
    path = ws.base / "local-agents"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def role_choices() -> list[dict[str, str]]:
    """Every role as ``{"name", "summary"}``, in the order of `roles.ROLES`."""
    return [{"name": r.name, "summary": r.summary} for r in roles.ROLES.values()]


def check_name(name: str) -> str:
    """``name`` when it can be a local agent's id; ValueError otherwise."""
    if not valid_name(name) or name in RESERVED or name.startswith(("ml-stack", "doctor-")) \
            or len(name) > 40:
        raise ValueError(f"{name!r} cannot be an agent name (a-z, 0-9, . _ -; up to 40; "
                         f"not a reserved word)")
    return name


def check_project(path: str, base: Path) -> str:
    """The resolved folder ``path`` names, or an empty string for none; ValueError for text that
    is not a plain existing folder or one that holds the workspace's own state."""
    if not path:
        return ""
    if _CONTROL.search(path) or len(path) > 1024:
        raise ValueError("the project path holds control characters or is too long")
    found = Path(path).expanduser().resolve()
    if not found.is_dir():
        raise ValueError(f"the project {found.name} is not a folder")
    here = base.resolve()
    if found == here or here in found.parents:
        raise ValueError("the project cannot be the workspace's own state folder")
    return str(found)


def check_orders(names: list[str]) -> tuple[str, ...]:
    """The agents the person named as givers of orders; ValueError for a name that is no id."""
    unique = tuple(dict.fromkeys(n.strip() for n in names if n.strip()))
    if len(unique) > MOST_ORDERERS or not all(valid_id(n) for n in unique):
        raise ValueError(f"name at most {MOST_ORDERERS} agents, each a plain id")
    return unique


@dataclass(frozen=True, slots=True)
class Agent:
    """What the person asked for when the agent was started."""

    name: str
    model: str
    identity: str = ""
    model_name: str = ""
    size_bytes: int = 0
    role: str = roles.DEFAULT
    profile: str = "chat"
    harness: str = "ml-stack-agent"
    ctx: int = 32768
    effort: str = "off"
    max_effort: str = "medium"
    project: str = ""
    orders_from: tuple[str, ...] = DEFAULT_ORDERS_FROM
    started: float = 0.0
    pid: int = 0
    process_started: float = 0.0
    log: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    max_output_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["orders_from"] = list(self.orders_from)
        return out


def _paths(ws: Workspace, name: str) -> dict[str, Path]:
    base = folder(ws)
    return {"record": base / f"{name}.json", "status": base / f"{name}.status.json",
            "stop": base / f"{name}.stop", "log": base / f"{name}.log"}


def pause_file(ws: Workspace, name: str) -> Path:
    """The file that pauses the worker before its next queued task."""
    return folder(ws) / f"{check_name(name)}.pause"


def stop_file(ws: Workspace, name: str) -> Path:
    """The file whose presence tells ``name``'s loop to stop at its next step."""
    return _paths(ws, check_name(name))["stop"]


def log_file(ws: Workspace, name: str) -> Path:
    return _paths(ws, check_name(name))["log"]


def save(ws: Workspace, agent: Agent) -> None:
    """Write ``agent``'s record."""
    path = _paths(ws, check_name(agent.name))["record"]
    write_json(path, {"version": 1, **agent.as_dict()})
    path.chmod(0o600)


def load(ws: Workspace, name: str) -> Agent | None:
    """The record of ``name``, or None when there is none or it is not one of ours."""
    row = read_json(_paths(ws, check_name(name))["record"], None)
    if not isinstance(row, dict) or row.get("version") != 1 or row.get("name") != name:
        return None
    fields = {k: row[k] for k in Agent.__dataclass_fields__ if k in row}
    fields["orders_from"] = tuple(str(n) for n in fields.get("orders_from", ()))
    if fields.get("max_output_tokens") == 8192 and not fields.get("extra", {}).get("explicit_output_limit"):
        fields["max_output_tokens"] = None
    return Agent(**fields)


def names(ws: Workspace) -> list[str]:
    """Every agent with a record."""
    return sorted(p.name.removesuffix(".json") for p in folder(ws).glob("*.json")
                  if not p.name.endswith(".status.json") and valid_name(p.name.removesuffix(".json")))


def alive(agent: Agent) -> bool:
    """Whether the process recorded for ``agent`` is still the one that was started."""
    if not pid_exists(agent.pid):
        return False
    began = started_at(agent.pid)
    return not (began and agent.process_started and abs(began - agent.process_started) > 5.0)


class Status:
    """The loop's own status file: its state, what it last did and when it last looked."""

    def __init__(self, ws: Workspace, name: str) -> None:
        self.path = _paths(ws, name)["status"]
        self.data: dict[str, Any] = read_json(self.path, {}) or {}

    def update(self, **fields: Any) -> None:
        self.data = {**self.data, **fields, "beat": time.time(), "pid": os.getpid()}
        write_json(self.path, self.data)
        self.path.chmod(0o600)


def status_of(ws: Workspace, name: str) -> dict[str, Any]:
    """What the loop of ``name`` last wrote: ``state``, ``detail``, ``steps``, ``tasks`` and so on."""
    found = read_json(_paths(ws, check_name(name))["status"], {})
    return found if isinstance(found, dict) else {}


def obeys(ws: Workspace, agent: Agent, row: dict[str, Any]) -> str:
    """Why the agent acts on message ``row``, or an empty string when it is information only.

    Only a clear ``task`` or ``question`` from a live person or lead identity, an identity the
    person named, or a delegate of one of those is acted on; a message to ``*`` and one held in
    quarantine never is, and a message on a board must name the agent."""
    sender = str(row.get("from", ""))
    if row.get("state") != "clear" or row.get("type") not in ORDER_KINDS or sender == agent.name \
            or row.get("to") == "*":
        return ""
    if str(row.get("to", "")).startswith("#") and f"@{agent.name}" not in str(row.get("raw") or ""):
        return ""
    role = ws.registry.role_of(sender)
    if not role:
        return ""
    if role in (HUMAN, LEAD):
        return f"the {role}"
    top = sender.partition("/")[0]
    if sender in agent.orders_from or top in agent.orders_from:
        return f"{top}, whom the person named"
    parent = str(ws.registry.info(sender).get("parent") or "")
    if parent and (ws.registry.role_of(parent) in (HUMAN, LEAD) or parent in agent.orders_from):
        return f"a delegate of {parent}"
    return ""
