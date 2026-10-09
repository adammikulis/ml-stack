"""Installed UI extensions run after Fleet's normal API authorization."""

from importlib.metadata import entry_points

GROUP = "poolhouse.ui_routes"


class ExtensionRoutes:
    def route(self) -> bool:
        for entry in entry_points(group=GROUP):
            base = f"/ui/{entry.name}"
            if self.path == base or self.path.startswith(base + "/"):
                if not self.ui.credentialed(self.cookie):
                    self.ui.record("person.refused", reason="extension-without-credential",
                                   source=self.client_ip, route=entry.name)
                    self.send(403, {"error": "open poolhouse from its own window, or run: poolhouse peers open"})
                    return True
                return bool(entry.load()(self))
        return super().route()
