"""Installed UI extensions run after Fleet's normal API authorization."""

from importlib.metadata import entry_points

GROUP = "ml_stack.ui_routes"


class ExtensionRoutes:
    def route(self) -> bool:
        for entry in entry_points(group=GROUP):
            base = f"/ui/{entry.name}"
            if self.path == base or self.path.startswith(base + "/"):
                return bool(entry.load()(self))
        return super().route()
