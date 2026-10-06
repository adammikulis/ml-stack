"""Agent capabilities scoped to one shared project workspace."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from ml_stack.files import read_json
from ml_stack.workspace import onboard, tokens
from ml_stack.workspace.boards import ANNOUNCE
from ml_stack.workspace.claims import Conflict, normal
from ml_stack.workspace.identity import AGENT, Denied
from ml_stack.workspace.project_history import adopt
from ml_stack.workspace.rates import RateLimited, Rates
from ml_stack.workspace.screen import Refused
from ml_stack.workspace.modelid import CLAIMED, clean_model, clean_harness
from ml_stack.workspace.service import Workspace

MAX_REPLY = 512 * 1024
METHODS = frozenset({"send", "inbox", "outbox", "ack", "thread", "announce", "claim_model", "nudge", "wait",
                     "claim", "release", "heartbeat", "renew", "board.list", "board.read",
                     "board.threads", "board.join", "board.leave", "board.dm",
                     "board.subscribe", "board.unsubscribe", "board.subs", "board.digest",
                     "board.rollup", "board.summary", "board.mentions"})


class WorkspaceHost:
    """Serve isolated registered project boards to authenticated agents."""

    def __init__(self, projects: Any) -> None:
        self.projects = projects

    def workspace(self, project_id: str) -> Workspace:
        return Workspace(self.projects.workspace_base(project_id))

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
            model = clean_model(str(body.get("model") or ""))
            harness = clean_harness(str(body.get("harness") or ""))
            if not model or not harness:
                raise ValueError("agent enrollment requires model and harness claims")
            scope = {"key": project_id, "name": project.name, "cluster": cluster, "cluster_id": cluster_id}
            name, token = ws.registry.enroll_project(wanted.strip().lower(), scope, onboard.TOKEN_S)
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

    def answer(self, project_id: str, action: str, body: dict, *, cluster: str = "", cluster_id: str = "") -> tuple[int, dict]:
        try:
            if len(json.dumps(body).encode()) > 32 * 1024:
                return 413, {"error": "workspace operation exceeds the size limit"}
            ws = self.workspace(project_id)
            if action == "join":
                name = onboard.join(ws, str(body.get("code") or ""),
                                    str(body.get("name") or ""),
                                    claim=(str(body.get("model") or ""),
                                           str(body.get("harness") or "")))
                token = tokens.load(ws.base, name)
                self._identity(ws, project_id, token, cluster=cluster, cluster_id=cluster_id)
                ws.audit("remote.seen", name, project_id=project_id)
                return 201, {"id": name, "token": token, "project_id": project_id}
            if action != "board":
                return 404, {"error": "no such workspace operation"}
            token = str(body.get("agent_token") or "")
            who = self._identity(ws, project_id, token, cluster=cluster, cluster_id=cluster_id)
            ws.audit("remote.seen", who.id, project_id=project_id)
            operation = str(body.get("operation") or "")
            args, kwargs = body.get("args", []), body.get("kwargs", {})
            if (not isinstance(args, list) or len(args) > 12 or not isinstance(kwargs, dict)
                    or "token" in kwargs or len(kwargs) > 12):
                raise ValueError("invalid operation arguments")
            if operation in {"whoami", "agents", "claims", "who", "history"} and "read" not in who.can:
                raise Denied("agent capability has no read permission")
            if operation in {"native.reserve", "native.release"}:
                ws._may(who, "claim")
                result = self._native_claims(ws, who, token, project_id, operation, args, kwargs)
            elif operation == "delegate":
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
                result = {"id": child, "token": made, "project_id": project_id,
                          "expires": ws.registry.info(child)["expires"]}
            elif operation == "revoke_self":
                if args or kwargs:
                    raise ValueError("self revocation takes no target")
                ws.registry.revoke(who, who.id)
                ws.audit("remote.revoke_self", who.id, project_id=project_id)
                result = {"id": who.id, "revoked": True}
            elif operation == "whoami":
                result = {"id": who.id, **ws.registry.info(who.id)}
            elif operation == "history":
                result = read_json(ws.base / "adopted-history.json", {}).get("messages", [])
            elif operation == "agents":
                result = [row for row in ws.registered() if row["role"] == AGENT]
            elif operation == "claims":
                result = ws.claims.listing()
            elif operation == "who":
                result = ws.who_owns(*args, **kwargs)
            elif operation in METHODS:
                owner, name = (ws.board, operation[6:]) if operation.startswith("board.") else (ws, operation)
                method = getattr(owner, name)
                if operation == "claim":
                    kwargs["pid"] = 0
                bound = inspect.signature(method).bind(token, *args, **kwargs)
                if "limit" in bound.arguments:
                    bound.arguments["limit"] = min(max(int(bound.arguments["limit"]), 1), 100)
                if "widen" in bound.arguments:
                    bound.arguments["widen"] = False
                if "timeout_s" in bound.arguments:
                    bound.arguments["timeout_s"] = min(max(float(bound.arguments["timeout_s"]), 0), 20)
                result = method(*bound.args, **bound.kwargs)
            else:
                raise Denied("this operation is unavailable to remote agents")
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

    def _native_claims(self, ws, who, token, project_id, operation, args, kwargs):
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
            return normal(kind, key)
        if (not key or len(key) > 4096 or "\\" in key or key.startswith("/")
                or any(part in {"", ".", ".."} for part in key.split("/"))
                or any(ord(char) < 32 for char in key)):
            raise ValueError("native areas are relative project paths")
        root = Path(self.projects.get(project_id).root).resolve()
        target = (root / key).resolve()
        if not target.is_relative_to(root):
            raise Denied("native area escapes its project")
        return str(target)

    def _identity(self, ws: Workspace, project_id: str, token: str, *, cluster: str = "", cluster_id: str = "") -> Any:
        who = ws.auth(token)
        if who.role != AGENT:
            raise Denied("remote workspace access requires a project agent capability")
        name = ws.registry.root_of(who.id)
        scope = ws.registry.info(name)["project"]
        if scope.get("cluster") and (scope["cluster"] != cluster or scope.get("cluster_id") != cluster_id):
            raise Denied("agent capability belongs to another cluster")
        if scope.get("key") != project_id:
            raise Denied("agent capability belongs to another project")
        return who
