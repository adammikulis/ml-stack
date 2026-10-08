"""Project-scoped typed task commands over sealed workspace transport."""

import json
from collections import Counter
from types import SimpleNamespace

from ml_stack.workspace import coordinator_calls, task_outcomes
from ml_stack.workspace.coordination import workspace_id
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.taskboard import TaskBoard

FIELDS = {"tasks": {"limit"}, "task": {"id"}, "task-create": {"spec"},
          "task-claim": {"id", "allocation_id"}, "task-heartbeat": {"id"},
          "task-checkpoint": {"id", "value"}, "task-submit": {"id", "value"},
          "task-review": {"id", "value"}, "task-credit": {"id"}}
PRIVATE = frozenset({"resource", "execution_config", "interpreter", "environment", "source_project",
                     "worktree", "candidate", "primary", "cwd", "token", "secret", "allocation_proof"})


class Parser:
    def parse_args(self, argv):
        if len(argv) != 2 or argv[0] not in FIELDS:
            raise ValueError("unsupported canonical task command")
        payload = json.loads(argv[1])
        if type(payload) is not dict or set(payload) - FIELDS[argv[0]]:
            raise ValueError("unsupported canonical task command fields")
        return SimpleNamespace(agent="", token_file="", payload=payload, id=payload.get("id"), json=True)


def scope(ws, token, project_id):
    who = ws.auth(token)
    grant = ws.registry.info(ws.registry.root_of(who.id))["project"]
    if not grant or grant.get("key") != project_id:
        raise Denied("canonical tasks require this exact existing project grant")
    return grant


def checked(board, token, ident, grant):
    task = board.get(token, ident)
    if task.get("project") != grant:
        raise Denied("canonical task belongs to another project grant")
    return task


def dispatch(ws, token, grant, action, payload):
    board = TaskBoard(ws)
    if action == "tasks":
        limit = payload.get("limit", 50)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("canonical task listing limit is 1-100")
        rows = [task for task in board.list(token)["tasks"] if task.get("project") == grant]
        return {"tasks": rows[:limit], "metrics": {"states": dict(Counter(row["state"] for row in rows)),
                                                   "total": len(rows)}}
    if action == "task-create":
        spec = payload.get("spec")
        if type(spec) is not dict:
            raise ValueError("task creation requires a bounded specification")
        if spec.get("project", grant) != grant:
            raise Denied("task creation cannot change the existing project grant")
        for ident in spec.get("deps", []):
            checked(board, token, ident, grant)
        return board.create(token, {**spec, "project": grant})
    task = checked(board, token, payload.get("id"), grant)
    if action == "task":
        return task
    if action == "task-claim":
        return board.claim(token, task["id"], payload.get("allocation_id"))
    if action == "task-heartbeat":
        return board.heartbeat(token, task["id"])
    if action == "task-credit":
        return task_outcomes.credit(ws, token, task["id"])
    if action == "task-review":
        return task_outcomes.review(ws, token, task["id"], payload.get("value"))
    method = board.checkpoint if action == "task-checkpoint" else board.submit
    return method(token, task["id"], payload.get("value"))


def public(value, roots):
    if isinstance(value, dict):
        return {public(key, roots): public(item, roots) for key, item in value.items() if key not in PRIVATE}
    if isinstance(value, list):
        return [public(item, roots) for item in value]
    if isinstance(value, str):
        for root in roots:
            if root:
                value = value.replace(root, "[canonical private path]")
    return value


def command(ws, token, project, document):
    if type(document) is not dict or set(document) != {"action", "payload", "request_id"}:
        raise ValueError("canonical task command requires action, payload and stable request_id")
    action, payload = document["action"], document["payload"]
    if type(action) is not str or action not in FIELDS or type(payload) is not dict:
        raise ValueError("unsupported typed canonical task command")
    grant = scope(ws, token, project.id)
    if set(payload) - FIELDS[action]:
        raise ValueError("unsupported canonical task command fields")
    if action == "task-create":
        spec = payload.get("spec")
        if type(spec) is not dict or spec.get("project", grant) != grant:
            raise Denied("task creation requires the existing project grant")
        payload = {"spec": {**spec, "project": grant}}
    elif action != "tasks":
        checked(TaskBoard(ws), token, payload.get("id"), grant)
    handlers = {name: (lambda args, workspace, capability, name=name:
                      dispatch(workspace, capability, scope(workspace, capability, project.id), name, args.payload))
                for name in FIELDS}
    request = {"workspace": workspace_id(ws), "request_id": document["request_id"],
               "argv": [action, json.dumps(payload, sort_keys=True)]}
    roots = (str(ws.base), str(project.root))
    try:
        result = coordinator_calls.execute(ws, token, request, Parser(), handlers)
    except (Denied, ValueError) as exc:
        raise type(exc)(public(str(exc), roots)) from exc
    return public(result, roots)
