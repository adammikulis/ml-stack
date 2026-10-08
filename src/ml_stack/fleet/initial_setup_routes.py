"""First-run device name and mode selection from the browser this daemon launched."""
from __future__ import annotations

import urllib.parse

from . import automatic_clusters, cluster_modes, invite_routes
from .discovery import DiscoveryError, memberships, require_name
from .session import parse_cookie


class InitialSetupRoutes:
    def public_route(self) -> bool:
        if self.path != "/ui/setup/initial":
            return super().public_route()
        if self.method != "POST":
            self.send(405, {"error": "use POST"})
            return True
        if not self._local_browser():
            self.ui.record("session.refused", reason="setup-headers", source=self.client_ip)
            self.send(403, {"error": "open setup in this computer's browser"})
            return True
        presented = self.ui.sessions.get(parse_cookie(self.cookie))
        if presented is None or not presented.credentialed:
            self.ui.record("session.refused", reason="setup-without-credential", source=self.client_ip)
            self.send(403, {"error": "open ml-stack from its own window, or run: ml-stack peers open"})
            return True
        try:
            self._initial_preferences()
            session = self.ui.sessions.open("setup", presented.origin)
            self.ui.record("session.open", who="setup", origin=presented.origin, source=self.client_ip)
            self.send(200, {"ok": True, **self.ui.state()},
                      {"Set-Cookie": self.ui.sessions.cookie_header(session), "Cache-Control": "no-store"})
        except (DiscoveryError, ValueError, OSError) as error:
            self.send(400, {"error": str(error)})
        return True

    def _local_browser(self) -> bool:
        try:
            host = urllib.parse.urlsplit("//" + self.host_header).hostname
        except ValueError:
            return False
        session = self.ui.sessions.get(parse_cookie(self.cookie))
        origin = self.header("Origin")
        return (invite_routes.local(self.client_ip) and host in {"localhost", "127.0.0.1", "::1"}
                and origin in {"http://" + self.host_header, "https://" + self.host_header}
                and self.header("Sec-Fetch-Site") == "same-origin"
                and not any(self.header(name) for name in
                            ("Authorization", "X-ML-Stack-Token", "X-ML-Stack-Agent"))
                and (session is None or session.who != "token"))

    def _initial_preferences(self) -> None:
        length = int(self.header("Content-Length", "0"))
        if not 0 < length <= 4096:
            raise ValueError("setup request must be between 1 and 4096 bytes")
        body = self.body()
        if not isinstance(body, dict) or set(body) - {"name", "cluster_mode", "automatic"}:
            raise ValueError("choose a device name and mode")
        if "automatic" in body and not isinstance(body["automatic"], bool):
            raise ValueError("automatic must be a boolean")
        name = require_name(body.get("name", ""))
        mode = cluster_modes.validate(body.get("cluster_mode"))
        if self.ui.settings is None or self.ui.settings_path is None:
            raise DiscoveryError("device settings are unavailable")
        selected = memberships(self.ui.cluster_key_path)
        if (selected and selected[0].mode == "prod" and self.ui.settings.setup_done
                and not invite_routes.person_session(self)):
            raise DiscoveryError("sign in before changing Production setup")
        fresh_auto = (len(selected) == 1 and selected[0].mode == "dev"
                      and selected[0].selection == "automatic" and not self.ui.settings.setup_done
                      and not self.ui.settings.cluster_mode)
        if selected and selected[0].mode != mode and not (fresh_auto and mode == "prod"):
            raise DiscoveryError("leave the current cluster before changing modes")
        with self.ui.join_guard():
            if selected and selected[0].mode != mode:
                self.ui.leave(selected[0].group)
            if mode == "dev":
                if body.get("automatic") is True:
                    automatic_clusters.select_automatic(self.ui.cluster_key_path, port=self.ui.discovery_port)
                else:
                    automatic_clusters.ensure(self.ui.cluster_key_path, mode="dev", port=self.ui.discovery_port)
            self.ui.set_name(name)
            self.ui.settings.cluster_mode = mode
            self.ui.settings.save(self.ui.settings_path)
            self.ui.rejoined()
