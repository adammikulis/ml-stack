"""Fixed local browser agent runtime installation and progress."""

from ml_stack import agent_dependency

from . import invite_routes, runtime_repair
from .setup_jobs import jobs, provenance


def _public_job(row):
    if row is None:
        return None
    return {"id": row["id"], "state": row["state"],
            "note": runtime_repair.safe_error(row["note"]), "error": runtime_repair.safe_error(row["error"])}


class RuntimeRepairRoutes:
    def _fixed_repair_request(self):
        try:
            size = int(self.header("Content-Length", "0") or 0)
        except ValueError:
            return False
        return 0 <= size <= 1024 and self.body() == {}

    def public_route(self):
        if self.path != "/ui/agent-runtime/status":
            return super().public_route()
        if self.method != "POST" or not self._local_browser():
            self.send(403, {"error": "Agent runtime status requires this computer's browser."})
            return True
        if not self._fixed_repair_request():
            self.send(400, {"error": "Agent runtime status takes no options."})
            return True
        queue = jobs(self.ui) if self.ui.root is not None else None
        if queue is not None:
            runtime_repair.resume(self.ui, queue)
        rows = [row for row in queue.all() if row["kind"] == runtime_repair.KIND] if queue else []
        error = agent_dependency.problem()
        self.send(200, {"job": _public_job(rows[-1] if rows else None), "runtime_ready": not error})
        return True

    def route(self):
        if self.path not in {"/ui/agent-runtime/install", "/ui/agent-runtime/repair"}:
            return super().route()
        if (self.method != "POST" or not self._local_browser()
                or not invite_routes.person_session(self)):
            self.send(403, {"error": "Install the agent runtime from this computer's signed-in browser."})
            return True
        if not self._fixed_repair_request():
            self.send(400, {"error": "Agent runtime installation takes no options."})
            return True
        if self.ui.root is None:
            self.send(409, {"error": "This interface cannot install an app runtime."})
            return True
        try:
            row = runtime_repair.request(self.ui, provenance(self), repair=self.path.endswith("/repair"))
        except (ValueError, OSError) as error:
            self.send(409, {"error": runtime_repair.safe_error(error)})
            return True
        self.send(202, {"ok": True, "job": _public_job(row)})
        return True
