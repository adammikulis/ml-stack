"""``/ui/room``: sizing the wiring limit for a model, and the person-only change to it."""

from __future__ import annotations

import platform
import urllib.parse

from ml_stack import hub
from ml_stack.sentinel.human import HumanRequired
from ml_stack.serve import wired, wired_apply

from .discovery import in_cluster
from .session import parse_cookie

__all__ = ["RoomRoutes"]

LOOPBACK = "127.0.0.1"
PATHS = ("/ui/room", "/ui/room/apply", "/ui/room/reset", "/ui/room/keep")


def _origin_ok(origin: str, host: str) -> bool:
    """An Origin header, when there is one, must name the host the request was sent to."""
    return not origin or urllib.parse.urlparse(origin).netloc.lower() == host.lower()


def _machine(hooks: wired.Hooks) -> dict[str, object]:
    total = hooks.total_bytes()
    kept = wired_apply.state(hooks)
    return {"supported": (hooks.system or platform.system()) == "Darwin", "total": total, "default_mb": int(total * 0.75) // wired.MIB,
            "max_mb": wired.max_mb(total), "min_mb": wired.MIN_MB,
            "warn_below_bytes": wired.warn_below_bytes(total),
            "reserve_bytes": wired.reserve_bytes(total), "live_mb": kept.live_mb,
            "kept": kept.kept, "kept_mb": kept.kept_mb, "drift": kept.drift,
            "original_mb": kept.original_mb}


class RoomRoutes:
    """Everything here is read-only except apply, keep and reset, which answer a person's click
    in a browser session on this machine and nothing else."""

    def _hooks(self) -> wired.Hooks:
        return getattr(self.ui, "room_hooks", None) or wired.Hooks()

    def route(self) -> bool:
        if self.path not in PATHS:
            return super().route()
        if self.path == "/ui/room" and self.method == "GET":
            return self._room()
        if self.path != "/ui/room" and self.method == "POST":
            return self._change()
        return super().route()

    def _room(self) -> bool:
        hooks = self._hooks()
        out: dict[str, object] = {"machine": _machine(hooks),
                                  "kv_types": list(wired.KV_TYPES),
                                  "contexts": list(wired.CONTEXTS),
                                  "models": sorted({m.id for m in hub.discover(formats=("gguf",))})}
        name = self.asked("model")
        if name:
            try:
                ask = wired.Ask(int(self.asked("ctx", "131072")), self.asked("kv", "q8_0"),
                                self.asked("mtp", "1") not in ("0", "false", "off"))
                out["plan"] = wired.plan(name, ask, hooks).as_dict()
            except (FileNotFoundError, ValueError) as no:
                self.send(400, {"error": str(no)[:300]})
                return True
        self.send(200, out)
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
        session = ui.sessions.get(parse_cookie(self.cookie))
        if session is not None and session.who == "token":
            return "a session opened with an access token cannot change the wiring limit"
        if in_cluster(ui.cluster_key_path) and session is None:
            return "sign in first"
        return ""

    def _apply(self, hooks: wired.Hooks) -> wired_apply.Applied:
        if self.path == "/ui/room/reset":
            return wired_apply.reset(via="osascript", hooks=hooks)
        body = self.body()
        keep = body.get("keep")
        if keep is not None and not isinstance(keep, bool):
            raise ValueError("keep must be true or false")
        if self.path == "/ui/room/keep":
            return wired_apply.set_limit(None, keep=False, via="osascript", hooks=hooks)
        return wired_apply.set_limit(body.get("mb"), keep=keep, via="osascript", hooks=hooks)

    def _change(self) -> bool:
        why = self._person()
        if why:
            self.send(403, {"error": why})
            return True
        hooks = self._hooks()
        try:
            done = self._apply(hooks)
        except HumanRequired as no:
            self.send(403, {"error": str(no)})
            return True
        except ValueError as no:
            self.send(400, {"error": str(no)[:300]})
            return True
        self.send(200 if done.ok else 502,
                  {"ok": done.ok, "message": done.message, "before_mb": done.before_mb,
                   "after_mb": done.after_mb, "machine": _machine(hooks)})
        return True
