"""Agent capabilities scoped to one shared project workspace."""

from __future__ import annotations

import inspect
import json
from typing import Any

from ml_stack.files import read_json
from ml_stack.workspace import onboard, tokens
from ml_stack.workspace.claims import Conflict
from ml_stack.workspace.identity import AGENT, Denied
from ml_stack.workspace.project_history import adopt
from ml_stack.workspace.rates import RateLimited
from ml_stack.workspace.screen import Refused
from ml_stack.workspace.service import Workspace

MAX_REPLY = 512 * 1024
METHODS = frozenset({"send", "inbox", "outbox", "ack", "thread", "announce", "claim_model",
                     "claim", "release", "heartbeat", "board.list", "board.read",
                     "board.threads", "board.join", "board.leave", "board.dm",
                     "board.subscribe", "board.unsubscribe", "board.subs", "board.digest",
                     "board.rollup", "board.summary"})


class WorkspaceHost:
    """Serve isolated registered project boards to authenticated agents."""

    def __init__(self, projects: Any) -> None:
        self.projects = projects

    def workspace(self, project_id: str) -> Workspace:
        return Workspace(self.projects.workspace_base(project_id))

    def prepare(self, project_id: str) -> dict:
        ws = self.workspace(project_id)
        if not ws.registry.ids():
            tokens.store(ws.base, tokens.OWNER_FILE, ws.init())
        return self.status(project_id)

    def status(self, project_id: str) -> dict:
        ws = self.workspace(project_id)
        project = self.projects.get(project_id)
        agents = [row for row in ws.registered() if row["role"] == AGENT]
        seen = {row["who"]: row["ts"] for row in ws.audit_log.rows()
                if row.get("event") == "remote.seen"}
        for agent in agents:
            agent["last_seen"] = seen.get(agent["id"], 0)
            agent["online"] = bool(agent["last_seen"] and ws.clock() - agent["last_seen"] < 90)
        boards = ws.board.store.state()[0]
        chosen = {name for name, info in boards.items() if info.get("project") == project_id}
        messages = [ws.deliver(row) for row in ws.board._rows()
                    if row.get("to") in chosen][-20:]
        return {"project_id": project_id, "name": project.name,
                "authority_machine": getattr(project, "authority_machine", ""),
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

    def answer(self, project_id: str, action: str, body: dict) -> tuple[int, dict]:
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
                self._identity(ws, project_id, token)
                ws.audit("remote.seen", name, project_id=project_id)
                return 201, {"id": name, "token": token, "project_id": project_id}
            if action != "board":
                return 404, {"error": "no such workspace operation"}
            token = str(body.get("agent_token") or "")
            who = self._identity(ws, project_id, token)
            ws.audit("remote.seen", who.id, project_id=project_id)
            operation = str(body.get("operation") or "")
            args, kwargs = body.get("args", []), body.get("kwargs", {})
            if (not isinstance(args, list) or len(args) > 12 or not isinstance(kwargs, dict)
                    or "token" in kwargs or len(kwargs) > 12):
                raise ValueError("invalid operation arguments")
            if operation in {"whoami", "agents", "claims", "who", "history"} and "read" not in who.can:
                raise Denied("agent capability has no read permission")
            if operation == "whoami":
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

    def _identity(self, ws: Workspace, project_id: str, token: str) -> Any:
        who = ws.auth(token)
        if who.role != AGENT:
            raise Denied("remote workspace access requires a project agent capability")
        name = ws.registry.root_of(who.id)
        if ws.registry.info(name)["project"].get("key") != project_id:
            raise Denied("agent capability belongs to another project")
        return who
