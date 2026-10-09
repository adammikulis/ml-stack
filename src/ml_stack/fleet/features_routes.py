"""``/ui/features``: the experimental features with their stage, risk and state, read-only.

Turning one on or off is ``ml-stack features enable|disable NAME``, which audits who did it.
"""

from __future__ import annotations

from ml_stack import features

__all__ = ["FeatureRoutes"]


class FeatureRoutes:
    """The list the settings screen shows."""

    def route(self) -> bool:
        if self.path == "/ui/features" and self.method == "GET":
            path = self.ui.settings_path
            self.send(200, {"features": features.listing(str(path.parent) if path else "")})
            return True
        return super().route()
