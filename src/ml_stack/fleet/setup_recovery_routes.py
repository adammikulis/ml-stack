"""Local setup joins a live cluster from a person-selected recovery file."""

from __future__ import annotations

import contextlib

from . import recovery
from .discovery import DiscoveryError, discover, in_cluster, require_name


class SetupRecoveryRoutes:
    def public_route(self) -> bool:
        if self.path != "/ui/setup/recovery" or self.method != "POST":
            return super().public_route()
        ui = self.ui
        if in_cluster(ui.cluster_key_path) and not ui.authed(self.cookie):
            self.send(401, {"error": "sign in before joining another cluster"})
            return True
        why = self._may_setup() if not in_cluster(ui.cluster_key_path) else ""
        if why:
            self.send(403, {"error": why})
            return True
        try:
            length = int(self.header("Content-Length", "0"))
            if not 0 < length <= 12000:
                raise DiscoveryError("recovery upload must be at most 12000 bytes")
            req = self.body()
            if not isinstance(req, dict) or set(req) != {"group", "recovery"}:
                raise DiscoveryError("select a cluster and its recovery file")
            member = recovery.parse_recovery(req["recovery"])
            if member.group != require_name(req["group"]):
                raise DiscoveryError("recovery file belongs to a different cluster")
            if not ui.throttle.acquire():
                raise DiscoveryError("another join is in progress; try again")
            try:
                if not discover(member.key, timeout_s=1.5, port=ui.discovery_port):
                    raise DiscoveryError("no live cluster authenticated this recovery file")
                recovery.adopt_recovery(member, ui.cluster_key_path)
            finally:
                ui.throttle.release()
        except (DiscoveryError, ValueError, OSError) as exc:
            self.send(400, {"error": str(exc)})
            return True
        ui.rejoined()
        if ui.on_join is not None:
            with contextlib.suppress(Exception):
                ui.on_join()
        session = ui.sessions.open("setup")
        self.send(200, ui.state(), {"Set-Cookie": ui.sessions.cookie_header(session)})
        return True
