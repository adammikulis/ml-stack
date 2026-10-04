"""Start, stop and list local agents. Callers have established that a person asked: the terminal
command checks for a person at a terminal, the browser route for the person's session."""

from __future__ import annotations

import contextlib
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ml_stack import jobs, roles
from ml_stack.serve.process import pid_exists, started_at
from ml_stack.workspace import localagent as la, localmodel, onboard, tokens
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import AGENT
from ml_stack.workspace.service import Workspace

__all__ = ["Ask", "Started", "Stopped", "Unavailable", "listing", "start", "stop"]

LOOP = "ml_stack.workspace.localloop"
STOP_WAIT_S = 20.0


class Unavailable(ValueError):
    """No model can be started: ``hint`` is the one command that would fix it."""

    def __init__(self, problem: str, hint: str = "") -> None:
        super().__init__(problem + (f"; run: {hint}" if hint else ""))
        self.problem, self.hint = problem, hint


@dataclass(frozen=True, slots=True)
class Ask:
    """What the person chose in the start form or on the command line."""

    model: str = localmodel.AUTO
    name: str = ""
    role: str = roles.DEFAULT
    think: bool = False
    project: str = ""
    orders_from: tuple[str, ...] = la.DEFAULT_ORDERS_FROM


@dataclass(frozen=True, slots=True)
class Started:
    """The agent that is running after `start`; ``already`` is true when it was running before."""

    name: str
    pid: int
    model: str
    role: str
    already: bool = False


@dataclass(frozen=True, slots=True)
class Stopped:
    """What `stop` did: the process ended (``forced`` when it had to be killed) and the lease it held."""

    name: str
    was_running: bool
    forced: bool = False
    lease_released: bool = False
    notes: list[str] = field(default_factory=list)


def _mint(ws: Workspace, name: str, project: dict[str, str]) -> None:
    """The agent's token file and place on its project's board; the standard agent role only."""
    token = ws.registry.mint(onboard.SETUP, name, AGENT, onboard.TOKEN_S)
    ws.audit("mint", onboard.SETUP.id, agent=name, role=AGENT, via="local-agent")
    tokens.store(ws.base, name, token)
    if project:
        ws.registry.set_project(onboard.SETUP, name, project)
    ws.board.place(name, project)


def start(ws: Workspace, ask: Ask, *, pick: localmodel.Pick | None = None,
          spawn: Callable[..., Any] | None = None) -> Started:
    """Join a local model to the workspace and run its loop detached; the same name again reports
    the agent already running. Raises `Unavailable` when no suitable model is downloaded or it
    would not fit, ValueError for a bad name, role or project."""
    from ml_stack.workspace import project as projects

    role = roles.get(ask.role).name
    folder_ = la.check_project(ask.project, ws.base)
    orders = la.check_orders(list(ask.orders_from))
    chosen = pick or localmodel.choose(ask.model)
    if not chosen.ok:
        raise Unavailable(chosen.problem, chosen.hint)
    name = la.check_name(ask.name or localmodel.agent_name(chosen.name))
    with held(la.folder(ws) / "start.lock"):
        if not ws.registry.ids():
            tokens.store(ws.base, tokens.OWNER_FILE, ws.init("owner"))
        have = la.load(ws, name)
        if have is not None and la.alive(have):
            return Started(name, have.pid, have.model_name, have.role, already=True)
        if ws.registry.role_of(name) and have is None:
            raise ValueError(f"{name} is another agent's name; pass a different --name")
        if ws.registry.role_of(name):
            ws.registry.revoke(onboard.SETUP, name)
        _mint(ws, name, projects.describe(folder_) if folder_ else {})
        la.stop_file(ws, name).unlink(missing_ok=True)
        agent = la.Agent(name=name, model=chosen.ref, model_name=chosen.name,
                         size_bytes=chosen.size_bytes, role=role, think=ask.think,
                         project=folder_, orders_from=orders, started=time.time())
        la.save(ws, agent)
        job = (spawn or jobs.detach)(LOOP, [name], log=la.log_file(ws, name), kind=name,
                                     home=la.folder(ws) / "jobs")
        la.save(ws, replace(agent, pid=job.pid, process_started=started_at(job.pid) or 0.0,
                            log=str(job.log)))
    ws.audit("local-agent.start", onboard.SETUP.id, agent=name, role=role)
    return Started(name, job.pid, chosen.name, role)


def _ended(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while pid_exists(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def stop(ws: Workspace, name: str, *, release: Callable[[str], Any] | None = None,
         wait_s: float = STOP_WAIT_S) -> Stopped:
    """End ``name``: the stop file first, then SIGTERM (SIGKILL after ``wait_s``), then the model
    lease, its token and its files. ValueError when there is no such local agent."""
    agent = la.load(ws, la.check_name(name))
    if agent is None:
        raise ValueError(f"no local agent called {name}; `agent list` shows them")
    la.stop_file(ws, name).write_text("stop\n", encoding="utf-8")
    running, forced, notes = la.alive(agent), False, []
    if running:
        with contextlib.suppress(ProcessLookupError):
            os.kill(agent.pid, signal.SIGTERM)
        if not _ended(agent.pid, wait_s):
            forced = True
            with contextlib.suppress(ProcessLookupError):
                os.kill(agent.pid, signal.SIGKILL)
            _ended(agent.pid, 5.0)
    lease = str((la.status_of(ws, name).get("lease") or {}).get("id") or "")
    freed = False
    if lease:
        try:
            freed = bool((release or _release)(lease))
        except (OSError, RuntimeError) as err:
            notes.append(f"the lease was not released now ({err}); it ends with the process")
    if ws.registry.role_of(name):
        ws.registry.revoke(onboard.SETUP, name)
    tokens_file = tokens.directory(ws.base) / name
    tokens_file.unlink(missing_ok=True)
    for path in (la.stop_file(ws, name), la.folder(ws) / f"{name}.json",
                 la.folder(ws) / f"{name}.status.json", la.folder(ws) / "jobs" / f"{name}.json"):
        path.unlink(missing_ok=True)
    ws.audit("local-agent.stop", onboard.SETUP.id, agent=name, forced=forced, lease=freed)
    return Stopped(name, running, forced, freed, notes)


def _release(lease: str) -> bool:
    from ml_stack.serve import broker_wire

    return bool(broker_wire.release(lease))


def listing(ws: Workspace) -> list[dict[str, Any]]:
    """Every local agent: its model, role, state, steps, last message, memory held and lease."""
    out = []
    for name in la.names(ws):
        agent = la.load(ws, name)
        if agent is None:
            continue
        status, live = la.status_of(ws, name), la.alive(agent)
        state = str(status.get("state") or "starting") if live else (
            "failed" if status.get("state") == "failed" else "stopped")
        out.append({
            "name": name, "model": agent.model_name, "role": agent.role, "think": agent.think,
            "project": Path(agent.project).name if agent.project else "", "running": live,
            "state": state, "detail": str(status.get("detail") or ""),
            "steps": int(status.get("steps") or 0), "tasks": int(status.get("tasks") or 0),
            "ignored": int(status.get("ignored") or 0),
            "last_message": status.get("last_message") or {},
            "memory_bytes": agent.size_bytes if live and state != "failed" else 0,
            "lease": bool((status.get("lease") or {}).get("id")) and live,
            "orders_from": list(agent.orders_from), "started": agent.started,
            "beat": float(status.get("beat") or 0.0)})
    return out
