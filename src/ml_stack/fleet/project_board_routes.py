"""Local human controls for the canonical project board."""

import re

from ml_stack.workspace.identity import Denied
from ml_stack.workspace.remote_host import WorkspaceHost


class ProjectBoardRoutes:
    def route(self) -> bool:
        match = re.fullmatch(r"/ui/projects/([a-f0-9]{32})/(board|invite|adopt)", self.path)
        if match is None:
            return super().route()
        projects = getattr(self.ui, "projects", None)
        if projects is None:
            self.send(501, {"error": "project registry unavailable"})
            return True
        if (self.client_ip not in {"127.0.0.1", "::1"} or not self.ui.authed(self.cookie)
                or not self.ui.host_ok(self.host_header)):
            self.send(403, {"error": "project board controls require a signed-in local person"})
            return True
        host = WorkspaceHost(projects)
        try:
            if self.method == "POST" and (
                    self.header("Origin") not in {f"http://{self.host_header}", f"https://{self.host_header}"}
                    or self.header("Sec-Fetch-Site", "same-origin") != "same-origin"):
                self.send(403, {"error": "project board changes come from this page"})
                return True
            if match[2] == "board" and self.method == "GET":
                self.send(200, host.status(match[1]))
            elif match[2] == "invite" and self.method == "POST":
                req = self.body()
                made = host.invite(match[1], str(req.get("hint") or ""), int(req.get("uses") or 1))
                made["host"] = f"http://{self.host_header}"
                made["command"] = (f"ml-stack-workspace remote join {made['code']} --name NAME "
                                   f"--model MODEL --harness HARNESS --host HOST_URL "
                                   f"--project-id {match[1]}")
                self.send(201, made)
            elif match[2] == "adopt" and self.method == "POST":
                self.send(200, host.adopt(match[1], self.body()))
            else:
                self.send(405, {"error": "read board status or create an agent invitation"})
        except (Denied, ValueError, TypeError, KeyError) as exc:
            self.send(409, {"error": str(exc)})
        return True
