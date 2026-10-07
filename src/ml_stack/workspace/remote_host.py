"""Agent capabilities scoped to one shared project workspace."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

from ml_stack.files import read_json
from ml_stack.workspace import (
    device_metadata,
    device_sessions,
    fleet_routes,
    onboard,
    remote_tasks,
    remote_workers,
    tokens,
)
from ml_stack.workspace.boards import ANNOUNCE
from ml_stack.workspace.chain import held
from ml_stack.workspace.claims import Conflict, normal
from ml_stack.workspace.identity import AGENT, Denied
from ml_stack.workspace.integration_git import git
from ml_stack.workspace.modelid import CLAIMED, clean_harness, clean_model
from ml_stack.workspace.project_history import adopt
from ml_stack.workspace.rates import RateLimited, Rates
from ml_stack.workspace.remote_protocol import METHODS
from ml_stack.workspace.screen import Refused
from ml_stack.workspace.service import Workspace
from ml_stack.workspace.work_reputation import standings

MAX_REPLY = 512 * 1024


def _reputation(ws, token, args, kwargs):
    if args or set(kwargs) - {"agent", "offset"}:
        raise ValueError("reputation takes an agent and evidence offset")
    return standings(ws, token, **kwargs)


def _bound_arguments(bound, ws):
    if "limit" in bound.arguments:
        requested = int(bound.arguments["limit"])
        default = ws.limits.read_items if requested == 0 else requested
        bound.arguments["limit"] = (100 if bound.arguments.get("widen")
                                    else min(max(default, 1), 100))
    if "widen" in bound.arguments:
        bound.arguments["widen"] = False
    if "timeout_s" in bound.arguments:
        bound.arguments["timeout_s"] = min(max(float(bound.arguments["timeout_s"]), 0), 20)


class WorkspaceHost:
    """Serve isolated registered project boards to authenticated agents."""

    def start_worker(self, project, request, *, admission, cluster_key):
        """Start a worker within the registered project and admitted cluster."""
        return remote_workers.start(self.projects, project, request, admission=admission, cluster_key=cluster_key)

    def __init__(self, projects: Any) -> None:
        self.projects = projects

    def workspace(self, project_id: str) -> Workspace:
        return Workspace(self.projects.workspace_base(project_id))

    def person_board(self, request, project_id: str) -> bool:
        """Dispatch the authenticated person's selected project Board request."""
        return fleet_routes.route(request, workspace=self.workspace(project_id),
                                  prefix=f"/ui/projects/{project_id}/board/")

    def prepare(self, project_id: str) -> dict:
        if hasattr(self.projects, "claim_authority"):
            self.projects.claim_authority(project_id)
        ws = self.workspace(project_id)
        if not ws.registry.ids():
            tokens.store(ws.base, tokens.OWNER_FILE, ws.init())
        return self.status(project_id)

    def status(self, project_id: str) -> dict:
        project = self.projects.get(project_id)
        authority = getattr(project, "authority_machine", None)
        if authority == "":
            return {"project_id": project_id, "name": project.name, "state": "unconfigured",
                    "authority_machine": "", "board_host": "", "agents": [], "boards": [],
                    "messages": [], "history": []}
        if authority and authority != getattr(self.projects, "machine", authority):
            return {"project_id": project_id, "name": project.name, "state": "connection_required",
                    "authority_machine": authority, "board_host": project.board_host,
                    "agents": [], "boards": [], "messages": [], "history": []}
        ws = self.workspace(project_id)
        agents = [row for row in ws.registered() if row["role"] == AGENT]
        seen = {row["who"]: row["ts"] for row in ws.audit_log.rows()
                if row.get("event") == "remote.seen"}
        for agent in agents:
            agent["last_seen"] = seen.get(agent["id"], 0)
            agent["online"] = bool(agent["last_seen"] and ws.clock() - agent["last_seen"] < 90
                                   and ws.registry.role_of(agent["id"]))
        boards = ws.board.store.state()[0]
        chosen = {name for name, info in boards.items() if info.get("project") == project_id or name == ANNOUNCE}
        messages = [ws.deliver(row) for row in ws.board._rows()
                    if row.get("to") in chosen][-20:]
        return {"project_id": project_id, "name": project.name,
                "authority_machine": getattr(project, "authority_machine", ""),
                "board_host": getattr(project, "board_host", ""),
                "agents": agents,
                "boards": sorted(chosen), "messages": messages,
                "history": read_json(ws.base / "adopted-history.json", {}).get("messages", []),
                "state": "connected" if any(agent["online"] for agent in agents)
                else "offline" if agents else "awaiting_agents" if ws.registry.ids()
                else "unconfigured"}

    def invite(self, project_id: str, hint: str = "", uses: int = 1) -> dict:
        self.prepare(project_id)
        ws = self.workspace(project_id)
        project = self.projects.get(project_id)
        code = ws.invites.create(hint, 600, {"key": project_id, "name": project.name},
                                 uses=min(max(int(uses), 1), 3))
        return {"project_id": project_id, "code": code, "ttl_s": 600,
                "uses": min(max(int(uses), 1), 3)}

    def adopt(self, project_id: str, history: dict) -> dict:
        self.prepare(project_id)
        return adopt(self.workspace(project_id).base, history)

    def enroll(self, project_id: str, body: dict, *, cluster: str = "", cluster_id: str = "") -> tuple[int, dict]:
        """Issue a project agent capability after authenticated Dev cluster admission."""
        try:
            if len(json.dumps(body).encode()) > 32 * 1024:
                return 413, {"error": "workspace operation exceeds the size limit"}
            project = self.projects.get(project_id)
            if (not cluster or not cluster_id or body.get("cluster") != cluster
                    or body.get("cluster_id") != cluster_id):
                raise Denied("agent enrollment requires its authenticated Dev cluster")
            if body.get("authority_machine") != self.projects.machine:
                raise Denied("agent enrollment requires the selected local project authority")
            if not project.authority_machine:
                project = self.projects.claim_authority(project_id, expected_machine=self.projects.machine)
            if project.authority_machine != self.projects.machine or not project.board_host:
                raise Denied("agent enrollment requires the selected local project authority")
            ws = self.workspace(project_id)
            Rates(ws.base / "enrollment", 10, 600, ws.clock).admit("dev-cluster")
            wanted = str(body.get("name") or "")
            onboard.pick_name(ws, wanted)
            model = str(body.get("model") or "")
            if model:
                model = clean_model(model)
            harness = clean_harness(str(body.get("harness") or ""))
            device = body.get("device")
            if device is not None:
                device = device_metadata.normalize({**device, "verification": "agent-reported",
                                                    "source": "agent-report", "peer_id": None})
            scope = {"key": project_id, "name": project.name, "cluster": cluster, "cluster_id": cluster_id}
            name, token = ws.registry.enroll_project(wanted.strip().lower(), scope, onboard.TOKEN_S)
            if device is not None:
                ws.registry.record_device_claim(token, device)
            if model or harness:
                ws.registry.record_model(name, model, harness, CLAIMED)
            ws.board.place(name, scope)
            ws.audit("remote.enroll", name, project_id=project_id,
                     admission="dev-cluster", model=model, harness=harness)
            ws.audit("remote.seen", name, project_id=project_id)
            return 201, {"id": name, "token": token, "project_id": project_id}
        except Denied as exc:
            return 403, {"error": str(exc)}
        except RateLimited as exc:
            return 429, {"error": str(exc)}
        except (ValueError, TypeError) as exc:
            return 400, {"error": str(exc)}

    def renew(self, project_id: str, body: dict, *, cluster: str = "", cluster_id: str = "") -> tuple[int, dict]:
        """Renew an existing automatic capability under authenticated Dev admission."""
        try:
            if len(json.dumps(body).encode()) > 32 * 1024:
                return 413, {"error": "workspace operation exceeds the size limit"}
            project = self.projects.get(project_id)
            if (not cluster or not cluster_id or body.get("cluster") != cluster
                    or body.get("cluster_id") != cluster_id):
                raise Denied("agent renewal requires its authenticated Dev cluster")
            if (body.get("authority_machine") != self.projects.machine
                    or project.authority_machine != self.projects.machine or not project.board_host):
                raise Denied("agent renewal requires the selected local project authority")
            ws = self.workspace(project_id)
            who = ws.registry.renew_dev_project(str(body.get("agent_token") or ""),
                                                 project_id, cluster, cluster_id)
            ws.audit("remote.renew", who.id, project_id=project_id, admission="dev-cluster")
            return 200, {"id": who.id, "project_id": project_id, "cluster_id": cluster_id}
        except Denied as exc:
            return 403, {"error": str(exc)}
        except (ValueError, TypeError) as exc:
            return 400, {"error": str(exc)}

    def answer(self, project_id: str, action: str, body: dict, *, device=None, admission: tuple[str, str, bool] = ("", "", False)) -> tuple[int, dict]:
        try:
            if len(json.dumps(body).encode()) > 32 * 1024:
                return 413, {"error": "workspace operation exceeds the size limit"}
            ws = self.workspace(project_id)
            if action == 'ensure':
                if set(body) != {'name', 'model', 'harness', 'project', 'agent_token'}:
                    raise ValueError('agent registration carries identity labels and project')
                document = {**body, 'project': {'key': project_id}}
                name, token = device_sessions.ensure(ws, device, self.projects, document,
                                                    str(body['agent_token']))
                self._identity(ws, project_id, token)
                return 200, {'id': name, 'token': token, 'project_id': project_id}
            if action == "join":
                name = onboard.join(ws, str(body.get("code") or ""),
                                    str(body.get("name") or ""),
                                    claim=(str(body.get("model") or ""),
                                           str(body.get("harness") or "")))
                token = tokens.load(ws.base, name)
                self._identity(ws, project_id, token, admission=admission)
                ws.audit("remote.seen", name, project_id=project_id)
                return 201, {"id": name, "token": token, "project_id": project_id}
            if action != "board":
                return 404, {"error": "no such workspace operation"}
            token = str(body.get("agent_token") or "")
            device_sessions.check(ws, token, device, self.projects)
            who = self._identity(ws, project_id, token, admission=admission)
            ws.audit("remote.seen", who.id, project_id=project_id)
            operation = str(body.get("operation") or "")
            args, kwargs = body.get("args", []), body.get("kwargs", {})
            if (not isinstance(args, list) or len(args) > 12 or not isinstance(kwargs, dict)
                    or "token" in kwargs or len(kwargs) > 12):
                raise ValueError("invalid operation arguments")
            result = self._result(ws, who, project_id, (token, operation, args, kwargs))
            reply = {"result": result}
            if len(json.dumps(reply).encode()) > MAX_REPLY:
                return 413, {"error": "answer too large; request fewer messages"}
            return 200, reply
        except (Denied, Refused) as exc:
            return 403, {"error": str(exc)}
        except RateLimited as exc:
            return 429, {"error": str(exc)}
        except Conflict as exc:
            return 409, {"error": str(exc)}
        except (ValueError, TypeError) as exc:
            return 400, {"error": str(exc)}

    def _result(self, ws, who, project_id, request):
        token, operation, args, kwargs = request
        if operation in {"whoami", "agents", "claims", "who", "history"} and "read" not in who.can:
            raise Denied("agent capability has no read permission")
        if operation == "work.reputation":
            result = _reputation(ws, token, args, kwargs)
        elif operation == "task.command":
            if len(args) != 1 or kwargs:
                raise ValueError("canonical task operation takes one typed command")
            result = remote_tasks.command(ws, token, self.projects.get(project_id), args[0])
        elif operation in {"native.reserve", "native.release", "native.heartbeat"}:
            ws._may(who, "claim")
            with held(ws.registry.path.with_name("agents.lock")):
                current = ws.registry.authenticate(token)
                result = self._native_claims(ws, current, project_id, (token, operation, args, kwargs))
        elif operation == "delegate":
            result = self._delegate(ws, who, project_id, args, kwargs)
        elif operation == "revoke_self":
            if args or kwargs:
                raise ValueError("self revocation takes no target")
            ws.registry.revoke_self(token, lambda actor: self._release_owned_claims(ws, actor))
            ws.audit("remote.revoke_self", who.id, project_id=project_id)
            result = {"id": who.id, "revoked": True}
        elif operation == "whoami":
            result = {"id": who.id, **ws.registry.info(who.id)}
        elif operation == "history":
            result = read_json(ws.base / "adopted-history.json", {}).get("messages", [])
        elif operation == "agents":
            result = [row for row in ws.registered() if row["role"] == AGENT]
        elif operation == "claims":
            result = [public for row in ws.claims.listing()
                      if (public := self._public_claim(project_id, row)) is not None]
        elif operation == "who":
            if len(args) != 2 or kwargs:
                raise ValueError("ownership lookup takes a project resource kind and relative key")
            kind, key = args
            mapped = self._native_resource(project_id, kind, key)
            row = ws.who_owns(kind, mapped)
            result = self._public_claim(project_id, row) if row else None
        elif operation in METHODS:
            owner, name = (ws.board, operation[6:]) if operation.startswith("board.") else (ws, operation)
            method = getattr(owner, name)
            bound = inspect.signature(method).bind(token, *args, **kwargs)
            _bound_arguments(bound, ws)
            result = method(*bound.args, **bound.kwargs)
        else:
            raise Denied("this operation is unavailable to remote agents")
        return result

    def _delegate(self, ws, who, project_id, args, kwargs):
        if len(args) != 1 or kwargs:
            raise ValueError("delegation takes one child name")
        name = args[0]
        if not isinstance(name, str):
            raise ValueError("delegation takes a child name")
        made = ws.registry.delegate(who, name, min(ws.limits.child_ttl_s, 28_800),
                                    who.can, min(ws.limits.max_children, 8))
        child = f"{who.id}/{name}"
        scope = ws.registry.info(ws.registry.root_of(who.id))["project"]
        ws.board.place(child, scope)
        ws.audit("remote.delegate", who.id, child=child, project_id=project_id)
        return {"id": child, "token": made, "project_id": project_id,
                  "expires": ws.registry.info(child)["expires"]}

    def _release_owned_claims(self, ws, who):
        with held(ws.claims.lock):
            claims = ws.claims._load()
            ws.claims._save({key: row for key, row in claims.items() if row["owner"] != who.id})

    def _public_claim(self, project_id, row):
        row = {key: row[key] for key in ("kind", "key", "owner", "pid", "since", "expires", "note") if key in row}
        if row["kind"] == "branch":
            return row
        if row["kind"] != "area":
            return None
        root = Path(normal("area", self.projects.get(project_id).root))
        key = Path(row["key"])
        if not key.is_relative_to(root) or key == root:
            return None
        return {**row, "key": key.relative_to(root).as_posix()}

    def _native_claims(self, ws, who, project_id, request):
        token, operation, args, kwargs = request
        if operation == "native.heartbeat":
            if args or set(kwargs) - {"ttl_s"}:
                raise ValueError("native heartbeat accepts only a bounded lifetime")
            ttl = kwargs.get("ttl_s", 0.0)
            if type(ttl) not in (int, float) or not math.isfinite(ttl) or not 0 <= ttl <= 3600:
                raise ValueError("native heartbeat lifetime is between zero and 3600 seconds")
            made = ws.claims.renew(who, ttl_s=ttl,
                                   predicate=lambda row: self._public_claim(project_id, row) is not None)
            return [public for row in made if (public := self._public_claim(project_id, row)) is not None]
        if operation == "native.release":
            if len(args) != 2 or kwargs:
                raise ValueError("native release takes a resource kind and relative key")
            kind, key = args
            mapped = self._native_resource(project_id, kind, key)
            try:
                gone = ws.release(token, kind, mapped)
            except Denied as exc:
                raise Denied("project resource belongs to another worker") from exc
            except ValueError as exc:
                raise ValueError("project resource is not claimed") from exc
            return {**gone, "key": key}
        if len(args) != 1 or not isinstance(args[0], list) or not 1 <= len(args[0]) <= 128:
            raise ValueError("native reservation takes 1-128 resource pairs")
        if set(kwargs) - {"label"}:
            raise ValueError("native reservation accepts only a label")
        label = kwargs.get("label", "")
        if not isinstance(label, str) or len(label) > 200 or any(ord(char) < 32 for char in label):
            raise ValueError("native reservation label is bounded text")
        resources, originals = [], {}
        for resource in args[0]:
            if not isinstance(resource, (list, tuple)) or len(resource) != 2:
                raise ValueError("native reservation takes resource pairs")
            kind, key = resource
            mapped = self._native_resource(project_id, kind, key)
            resources.append((kind, mapped))
            originals[(kind, normal(kind, mapped))] = key
        try:
            made = ws.claims.reserve(who, resources, {"note": label})
        except Conflict as exc:
            raise Conflict("a requested project resource belongs to another worker", {}) from exc
        ws.audit("remote.native.reserve", who.id, project_id=project_id, resources=len(made))
        return [{**row, "key": originals[(row["kind"], row["key"])]} for row in made]

    def _native_resource(self, project_id, kind, key):
        if kind not in {"area", "branch"} or not isinstance(key, str):
            raise ValueError("native reservations use project areas and branches")
        if kind == "branch":
            key = normal(kind, key)
            try:
                git(self.workspace(project_id).base, "check-ref-format", "--branch", key)
            except RuntimeError as exc:
                raise ValueError("native reservation requires a valid Git branch") from exc
            return key
        if (not key or len(key) > 4096 or "\\" in key or key.startswith("/")
                or any(part in {"", ".", ".."} for part in key.split("/"))
                or any(ord(char) < 32 for char in key)):
            raise ValueError("native areas are relative project paths")
        root = Path(self.projects.get(project_id).root).resolve()
        target = (root / key).resolve()
        if not target.is_relative_to(root):
            raise Denied("native area escapes its project")
        return str(target)

    def _identity(self, ws: Workspace, project_id: str, token: str, *, admission: tuple[str, str, bool] = ("", "", False)) -> Any:
        who = ws.auth(token)
        if who.role != AGENT:
            raise Denied("remote workspace access requires a project agent capability")
        name = ws.registry.root_of(who.id)
        scope = ws.registry.info(name)["project"]
        cluster, cluster_id, dev_admission = admission
        if scope.get("cluster_id") and dev_admission is not True:
            raise Denied("automatic agent access requires active authenticated Dev admission")
        if scope.get("cluster") and (scope["cluster"] != cluster or scope.get("cluster_id") != cluster_id):
            raise Denied("agent capability belongs to another cluster")
        if scope.get("key") != project_id:
            raise Denied("agent capability belongs to another project")
        return who
