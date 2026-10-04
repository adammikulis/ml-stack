"""``/ui/room``: sizing the wiring limit for a model, and the person-only change to it."""

from __future__ import annotations

import urllib.parse
from typing import Any

from ml_stack.sentinel.human import HumanRequired

from .daemon import LOOPBACK

__all__ = ["RoomRoutes"]

PATHS = ("/ui/room", "/ui/room/apply", "/ui/room/reset")


def _origin_ok(origin: str, host: str) -> bool:
    """An Origin header, when there is one, must name the host the request was sent to."""
    if not origin:
        return True
    return urllib.parse.urlparse(origin).netloc.lower() == host.lower()


class RoomRoutes:
    """Everything here is read-only except apply and reset, which answer a person's click
    in a browser session on this machine and nothing else."""

    def route(self) -> bool:
        if self.path not in PATHS:
            return super().route()
        if self.path == "/ui/room" and self.method == "GET":
            return self._room()
        if self.path != "/ui/room" and self.method == "POST":
            return self._change()
        return super().route()

    def _room(self) -> bool:
        from ml_stack import hub
        from ml_stack.serve import wired

        name = self.asked("model")
        models = sorted({m.id for m in hub.discover(formats=("gguf",))})
        if not name:
            self.send(200, {"models": models, "kv_types": list(wired.KV_TYPES),
                            "contexts": list(wired.CONTEXTS), "original_mb": wired.original_mb()})
            return True
        try:
            context = int(self.asked("ctx", "131072"))
            plan = wired.plan(name, context=context, kv=self.asked("kv", "q8_0"),
                              mtp=self.asked("mtp", "1") not in ("0", "false", "off"))
        except (FileNotFoundError, ValueError) as no:
            self.send(400, {"error": str(no)[:300]})
            return True
        self.send(200, {"plan": plan.as_dict(), "original_mb": wired.original_mb()})
        return True

    def _person(self) -> str:
        """Empty when this request is a person's browser session on this machine, else why not."""
        ui = self.ui
        if self.client_ip != LOOPBACK or not ui.host_ok(self.host_header):
            return "the wiring limit is changed from the machine itself, in its own browser"
        if not _origin_ok(self.header("Origin"), self.host_header):
            return "this request came from another web page"
        if self.header("Authorization") or self.header("X-ML-Stack-Token"):
            return "an access token cannot change the wiring limit; a person at the page can"
        session = ui.sessions.get(self._cookie_value())
        if session is not None and session.who == "token":
            return "a session opened with an access token cannot change the wiring limit"
        from .discovery import in_cluster

        if in_cluster(ui.cluster_key_path) and session is None:
            return "sign in first"
        return ""

    def _cookie_value(self) -> str:
        from .session import parse_cookie

        return parse_cookie(self.cookie)

    def _change(self) -> bool:
        from ml_stack.serve import wired

        why = self._person()
        if why:
            self.send(403, {"error": why})
            return True
        runner: Any = getattr(self.ui, "room_runner", None)
        try:
            if self.path == "/ui/room/reset":
                done = wired.reset(via="osascript", runner=runner)
            else:
                done = wired.apply(self.body().get("mb"), via="osascript", runner=runner)
        except HumanRequired as no:
            self.send(403, {"error": str(no)})
            return True
        except ValueError as no:
            self.send(400, {"error": str(no)[:300]})
            return True
        self.send(200 if done.ok else 502,
                  {"ok": done.ok, "message": done.message, "before_mb": done.before_mb,
                   "after_mb": done.after_mb, "resets_on_reboot": True,
                   "original_mb": wired.original_mb()})
        return True
