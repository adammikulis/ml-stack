"""Start, stop and list local agents. Callers have established that a person asked: the terminal
command checks for a person at a terminal, the browser route for the person's session."""

from __future__ import annotations

import contextlib
import hashlib
import os
import signal
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ml_stack import jobs, roles
from ml_stack.serve import broker_wire
from ml_stack.serve.process import pid_exists, started_at
from ml_stack.workspace import (
    backlog,
    device_agent,
    guide,
    issuepump,
    localagent as la,
    localeffort as le,
    localharness as lh,
    localloop,
    localmodel,
    localprofile as lp,
    onboard,
    project as projects,
    task_authority,
    tokens,
)
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import AGENT, LEAD, Denied
from ml_stack.workspace.modelid import clean_model
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
    effort: str = "off"
    max_effort: str = "medium"
    profile: str = "chat"
    ctx: int = 0
    project: str = ""
    orders_from: tuple[str, ...] = la.DEFAULT_ORDERS_FROM
    harness: str = lh.PI
    max_output_tokens: int = 8192
    authority: tuple[str, str] | None = None
    repo: str = ""


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


def start(ws: Workspace, ask: Ask, *, pick=None, spawn=None, person_token="", parent_token="") -> Started:
    """Start a worker and bind person-authorized launches to the persistent device account."""
    if parent_token and ws.auth(parent_token).role not in (AGENT, LEAD):
        raise Denied('coding queue launch requires an authenticated agent parent')
    if person_token:
        device_agent.enroll(ws, person_token)
    result = _start(ws, ask, pick=pick, spawn=spawn, parent_token=parent_token)
    if person_token:
        device_agent.bind_worker(ws, person_token, result.name)
    elif parent_token and result.already:
        device_agent.bind_owned_worker(ws, parent_token, result.name)
    return result


def launch_parent(ws: Workspace, project: str) -> str:
    """Return the authenticated local launcher agent's token for this project."""
    found = projects.describe(project)
    if not found:
        raise ValueError('local worker launch requires a project')
    name = 'local-app-launcher-' + hashlib.sha256(found['key'].encode('utf-8')).hexdigest()[:16]
    connected = guide.agent_connect(ws, name, found)
    return tokens.load(ws.base, connected['id'])


def _worker_identity(ws, have, name, project, parent_token=""):
    if have and ws.registry.role_of(have.identity or name):
        identity = have.identity or name
        child = ws.auth(tokens.load(ws.base, identity))
        if parent_token and child.parent != ws.auth(parent_token).id:
            raise Denied('coding queue launch requires this worker registered parent')
        return identity
    if parent_token:
        return ws.delegate(parent_token, name)['id']
    _mint(ws, name, project)
    return name


def _running(have, chosen):
    if have.model != chosen.ref:
        raise ValueError(f"{have.name} already runs {have.model_name}; stop and restart that same name to change models")
    return Started(have.name, have.pid, have.model_name, have.role, already=True)


def _start(ws: Workspace, ask: Ask, *, pick: localmodel.Pick | None = None,
          spawn: Callable[..., Any] | None = None, parent_token="") -> Started:
    """Join a local model to the workspace and run its loop detached; the same name again reports
    the agent already running. Raises `Unavailable` when no suitable model is downloaded or it
    would not fit, ValueError for a bad name, role or project."""
    if isinstance(ask.max_output_tokens, bool) or not isinstance(ask.max_output_tokens, int) or ask.max_output_tokens < 1:
        raise ValueError("maximum output tokens must be a positive integer")
    role = roles.get(ask.role).name
    ceiling = le.valid(ask.max_effort)
    effort = le.clamp(le.valid(ask.effort, allow_auto=True), ceiling) if ask.effort != le.AUTO else le.AUTO
    folder_ = la.check_project(ask.project, ws.base)
    orders = la.check_orders(list(ask.orders_from))
    prof = lp.profile(ask.profile)
    ctx = ask.ctx or prof.ctx
    if parent_token and prof.name == 'coding' and (not folder_ or
            ws.registry.info(ws.auth(parent_token).id).get('project') != projects.describe(folder_)):
        raise Denied('coding queue launch requires the parent registered project')
    chosen = pick or localmodel.choose(
        ask.model, selection=localmodel.Selection(coding=prof.name == "coding", context=ctx))
    if not chosen.ok:
        raise Unavailable(chosen.problem, chosen.hint)
    if not ctx:
        try:
            ctx = localmodel.context_for(chosen, coding=prof.name == "coding")
        except (OSError, ValueError) as error:
            raise Unavailable(str(error)) from error
    if prof.name == "coding":
        return _coding(ws, ask, chosen, ctx, folder_, parent_token)
    problem, hint = lp.admit(chosen.ref or chosen.name, ctx)
    if problem:
        raise Unavailable(problem, hint)
    name = la.check_name(ask.name or "local-agent")
    with held(la.folder(ws) / "start.lock"):
        have = la.load(ws, name)
        if have is not None and la.alive(have):
            return _running(have, chosen)
        if ws.registry.role_of(name) and have is None:
            raise ValueError(f"{name} is another agent's name; pass a different --name")
        identity = _worker_identity(ws, have, name, projects.describe(folder_) if folder_ else {}, parent_token)
        _record_model(ws, identity, chosen)
        la.stop_file(ws, name).unlink(missing_ok=True)
        agent = la.Agent(name=name, identity=identity, model=chosen.ref, model_name=chosen.name,
                         size_bytes=chosen.size_bytes, role=role, profile=prof.name, ctx=ctx, effort=effort, max_effort=ceiling, max_output_tokens=ask.max_output_tokens,
                         project=folder_, orders_from=orders, started=time.time(), extra=dict(have.extra) if have else {})
        la.save(ws, agent)
        if parent_token:
            device_agent.bind_owned_worker(ws, parent_token, name)
        job = (spawn or jobs.detach)(LOOP, [name], log=la.log_file(ws, name), kind=name,
                                     home=la.folder(ws) / "jobs")
        la.save(ws, replace(agent, pid=job.pid, process_started=started_at(job.pid) or 0.0,
                            log=str(job.log)))
    ws.audit("local-agent.start", onboard.SETUP.id, agent=name, role=role)
    return Started(name, job.pid, chosen.name, role)


def _coding(ws: Workspace, ask: Ask, chosen: localmodel.Pick | None, ctx: int, project: str,
            parent_token="") -> Started:
    """Start a coding worker through its configured native harness and maintained broker."""
    chosen = chosen or localmodel.choose(ask.model, selection=localmodel.Selection(coding=True, context=ctx))
    if not chosen.ok:
        raise Unavailable(chosen.problem, chosen.hint)
    problem, hint = lp.admit(chosen.ref or chosen.name, ctx)
    if problem:
        raise Unavailable(problem, hint)
    role = roles.get(ask.role).name
    if ask.harness not in ("pi", "codex", "claude"):
        raise ValueError("coding harness is pi, codex or claude")
    name = la.check_name(ask.name or "local-coding")
    with held(la.folder(ws) / "start.lock"):
        have = la.load(ws, name)
        if ask.authority:
            authorized, _ = task_authority.authorize(ws, ask.authority[0], name, ask.authority[1])
            if chosen.ref != authorized.model:
                raise ValueError('task launch must preserve the saved worker model')
            if any(getattr(ask, key) != getattr(authorized, key) for key in
                   ('model', 'name', 'role', 'effort', 'max_effort', 'ctx', 'project', 'orders_from')):
                raise ValueError('saved worker configuration changed before task launch')
            have = replace(have, extra={**have.extra, 'task_caps': asdict(localloop.caps_of(have))})
        if have is not None and la.alive(have):
            return _running(have, chosen)
        agent = la.Agent(name=name, model=chosen.ref, model_name=chosen.name,
                         size_bytes=chosen.size_bytes, role=role, profile="coding", harness=ask.harness,
                         ctx=ctx, project=project, effort=(le.AUTO if ask.effort == le.AUTO else le.clamp(le.valid(ask.effort), le.valid(ask.max_effort))), max_effort=ask.max_effort, max_output_tokens=ask.max_output_tokens,
                         extra=dict(have.extra) if have else {}, orders_from=la.check_orders(list(ask.orders_from)),
                         started=time.time())
        identity = (have.identity or have.name) if ask.authority else _worker_identity(
            ws, have, name, projects.describe(project) if project else {}, parent_token)
        agent = replace(agent, identity=identity)
        if not ask.authority:
            _record_model(ws, identity, chosen, ask.harness)
        la.save(ws, agent)
        if parent_token:
            device_agent.bind_owned_worker(ws, parent_token, name)
        job = jobs.detach(lh.RUNNER, [name], log=la.log_file(ws, name), kind=name,
                          home=la.folder(ws) / "jobs")
        la.save(ws, replace(agent, pid=job.pid, process_started=started_at(job.pid) or 0.0,
                            log=str(job.log)))
    repo = ask.repo or (backlog.repository(project) if project else "")
    if repo and parent_token:
        issuepump.configure_and_start(ws, parent_token, name, repo, project)
    ws.audit("local-agent.start", onboard.SETUP.id, agent=name, role=role, harness=ask.harness)
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
    """Stop the owned process and holder while preserving its identity and saved preferences."""
    agent = la.load(ws, la.check_name(name))
    if agent is None:
        raise ValueError(f"no local agent called {name}; `agent list` shows them")
    la.stop_file(ws, name).write_text("stop\n", encoding="utf-8")
    lease = str((la.status_of(ws, name).get("lease") or {}).get("id") or "")
    owned = bool(release) or _owns_lease(agent, lease)
    running, forced, notes = la.alive(agent), False, []
    if running:
        with contextlib.suppress(ProcessLookupError):
            os.kill(agent.pid, signal.SIGTERM)
        if not _ended(agent.pid, wait_s):
            forced = True
            with contextlib.suppress(ProcessLookupError):
                os.kill(agent.pid, signal.SIGKILL)
            _ended(agent.pid, 5.0)
    freed = False
    if lease and owned:
        try:
            freed = bool((release or _release)(lease))
        except (OSError, RuntimeError) as err:
            notes.append(f"the lease was not released now ({err}); it ends with the process")
    la.save(ws, replace(agent, pid=0, process_started=0.0))
    la.Status(ws, name).update(state='stopped', detail='Identity and preferences retained', lease={})
    for path in (la.stop_file(ws, name), la.pause_file(ws, name), la.folder(ws) / "jobs" / f"{name}.json"):
        path.unlink(missing_ok=True)
    ws.audit("local-agent.stop", onboard.SETUP.id, agent=name, forced=forced, lease=freed)
    return Stopped(name, running, forced, freed, notes)


def _record_model(ws: Workspace, name: str, chosen: localmodel.Pick, harness: str = lh.OWN) -> None:
    """Record the authenticated worker's configured model and harness claim."""
    try:
        model = clean_model(chosen.name)
    except ValueError:
        model = localmodel.short_name(chosen.name)
    ws.claim_model(tokens.load(ws.base, name), model, harness)


def _owns_lease(agent: la.Agent, lease: str) -> bool:
    if not lease or not agent.pid:
        return False
    try:
        status = broker_wire.status(start=False)
    except (OSError, RuntimeError):
        return False
    return any(holder.get("lease") == lease and holder.get("pid") == agent.pid
               and holder.get("pid_started") == agent.process_started
               for server in status.get("servers", []) for holder in server.get("holders", []))


def _release(lease: str) -> bool:
    return bool(broker_wire.release(lease))



def pause(ws: Workspace, name: str, paused: bool) -> dict[str, Any]:
    """Pause or resume queued tasks; an active task finishes before pausing."""
    name = la.check_name(name)
    if la.load(ws, name) is None:
        raise ValueError(f"no local agent called {name}")
    flag = la.pause_file(ws, name)
    if paused:
        flag.write_text("paused\n", encoding="utf-8")
    else:
        flag.unlink(missing_ok=True)
    return {"name": name, "paused": paused}

def listing(ws: Workspace) -> list[dict[str, Any]]:
    """Every local agent: its model, role, state, steps, last message, memory held and lease."""
    out = []
    for name in la.names(ws):
        agent = la.load(ws, name)
        if agent is None:
            continue
        status, live = la.status_of(ws, name), la.alive(agent)
        if live and float(status.get("beat") or 0) < agent.started:
            status = {}
        state = str(status.get("state") or "starting") if live else (
            "failed" if status.get("state") == "failed" else "stopped")
        out.append({
            "name": name, "identity": agent.identity or name, "model": agent.model_name, "role": agent.role, "effort": status.get("effort") or agent.effort, "max_effort": agent.max_effort, "profile": agent.profile, "harness": agent.harness, "ctx": agent.ctx, "max_output_tokens": agent.max_output_tokens,
            "project": Path(agent.project).name if agent.project else "", "running": live,
            "state": state, "detail": str(status.get("detail") or ""),
            "steps": int(status.get("steps") or 0), "tasks": int(status.get("tasks") or 0),
            "ignored": int(status.get("ignored") or 0),
            "last_message": status.get("last_message") or {},
            "backlog": issuepump.status(ws, name),
            "memory_bytes": agent.size_bytes if live and state != "failed" else 0,
            "lease": bool((status.get("lease") or {}).get("id")) and live,
            "paused": la.pause_file(ws, name).exists(), "orders_from": list(agent.orders_from), "started": agent.started,
            "beat": float(status.get("beat") or 0.0)})
    return out
