"""Installed UI extensions run after Fleet's normal API authorization."""

from importlib.metadata import entry_points

GROUP = "ml_stack.ui_routes"


class ExtensionRoutes:
    def route(self) -> bool:
        for entry in entry_points(group=GROUP):
            if self.path.startswith(f"/ui/{entry.name}/"):
                return bool(entry.load()(self))
        return super().route()
