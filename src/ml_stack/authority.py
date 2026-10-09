"""The delegable authority registry: which gates a lead agent may pass and which stay a person's."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Mapping
from typing import Any

from ml_stack import home
from ml_stack.files import read_json, write_json
from ml_stack.person import HumanRequired, require_person

__all__ = ["DELEGATED", "FLOOR_ENV", "GATES", "PERSON", "PRESETS", "audit_rows", "record", "require",
           "resolve", "set_state", "show", "state_of"]

DELEGATED = "delegated"
PERSON = "person"
FLOOR_ENV = "ML_STACK_AUTHORITY_FLOOR"
AGENT_ENV = "ML_STACK_WORKSPACE_AGENT"
LABEL_ENV = "ML_STACK_WORKSPACE_LABEL"
DELEGATING = ("CLAUDECODE", "ML_STACK_AGENT")

GATES: dict[str, str] = {
    "workspace.setup": "workspace",
    "workspace.model": "workspace",
    "workspace.agents": "workspace",
    "workspace.coordinator": "workspace",
    "sentinel.quarantine": "sentinel",
    "sentinel.policy": "sentinel",
    "requests.answer": "requests",
    "memory.admin": "memory",
    "reputation.admin": "reputation",
    "activity.export": "activity",
    "fleet.recovery": "fleet",
    "serve.wired-limit": "serve",
    "chat.policy": "chat",
    "runtime.deploy": "runtime",
}
"""Each delegable gate and its group."""

PROD_DELEGATED = frozenset({"workspace.setup", "workspace.model", "workspace.agents"})
PRESETS = ("dev", "prod")


def _file():
    return home.state("authority.json")


def _read() -> dict[str, Any]:
    data = read_json(_file(), {})
    return data if isinstance(data, dict) else {}


def state_of(gate: str, project: str = "") -> str:
    """``person`` or ``delegated`` for ``gate``: the project's setting, else the machine's, else delegated."""
    if gate not in GATES:
        raise KeyError(f"no authority gate called {gate}")
    if os.environ.get(FLOOR_ENV) == PERSON:
        return PERSON
    data = _read()
    scoped = (data.get("projects") or {}).get(project, {}) if project else {}
    return scoped.get(gate) or (data.get("gates") or {}).get(gate) or DELEGATED


def resolve(names: Iterable[str]) -> list[str]:
    """The gates ``names`` select: ``ALL``, a group, or a gate; unknown names raise `KeyError`."""
    out: list[str] = []
    for name in names:
        if name == "ALL":
            found = list(GATES)
        elif name in GATES:
            found = [name]
        else:
            found = [g for g, group in GATES.items() if group == name]
        if not found:
            raise KeyError(f"no authority gate or group called {name}")
        out += [g for g in found if g not in out]
    return out


def show(project: str = "") -> dict[str, Any]:
    """Every gate with its group and state, and the last preset applied."""
    data = _read()
    return {"preset": data.get("preset", "dev"), "project": project,
            "gates": [{"gate": g, "group": group, "state": state_of(g, project)} for g, group in GATES.items()]}


def _log(entry: Mapping[str, Any]) -> None:
    path = home.state("authority-audit.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": time.time(), **entry}) + "\n")


def record(event: str, **fields: Any) -> None:
    """Append an ``event`` a person took to the audit log, with ``fields``."""
    _log({"event": event, "by": PERSON, **fields})


def audit_rows() -> list[dict[str, Any]]:
    """Every recorded flip and delegated use, oldest first."""
    path = home.state("authority-audit.jsonl")
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line] if path.exists() else []


def set_state(names: Iterable[str], new: str, *, by: str, project: str = "",
              preset: str = "") -> list[dict[str, str]]:
    """Set the selected gates to ``new`` for the machine (or ``project``) and audit who changed what.
    Returns each gate's change; a helper identity (``parent/name``) is refused."""
    if "/" in by:
        raise HumanRequired("authority is flipped by a lead agent or a person, not a helper")
    if new not in (PERSON, DELEGATED):
        raise ValueError(f"authority state must be {PERSON} or {DELEGATED}")
    gates = resolve(names)
    before = {g: state_of(g, project) for g in gates}
    data = _read()
    target = data.setdefault("projects", {}).setdefault(project, {}) if project else data.setdefault("gates", {})
    target.update(dict.fromkeys(gates, new))
    if preset:
        data["preset"] = preset
    write_json(_file(), data)
    changes = [{"gate": g, "from": before[g], "to": new} for g in gates]
    _log({"event": "authority.set", "by": by, "project": project, "gates": gates,
          "from": sorted(set(before.values())), "to": new, "preset": preset})
    return changes


def require(gate: str, action: str, terminal: tuple[bool, bool] | None = None,
            env: Mapping[str, str] | None = None, *, project: str = "") -> str:
    """Pass a person at a terminal, or a lead agent when ``gate`` is delegated; returns who passed.
    Raises `HumanRequired` otherwise; each delegated pass is audited."""
    if state_of(gate, project) == PERSON:
        require_person(action, terminal, env)
        return PERSON
    found = os.environ if env is None else env
    marker = next((m for m in DELEGATING if found.get(m)), "")
    if not marker:
        require_person(action, terminal, env)
        return PERSON
    who, tag = found.get(AGENT_ENV, ""), found.get(LABEL_ENV, "")
    if tag or "/" in who:
        raise HumanRequired(f"{action} is delegated to the lead agent, not a helper")
    _log({"event": "authority.use", "gate": gate, "action": action, "agent": who or marker,
          "project": project})
    return DELEGATED

