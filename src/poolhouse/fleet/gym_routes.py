"""Authenticated live Gym sessions."""

from __future__ import annotations

import json
import os
import time

from poolhouse.gym import catalogue, manager
from poolhouse.gym.models import model_choices
from poolhouse.gym.transport import interpreter

from .files import safe_relpath
from .gym_interpreters import configure_interpreters
from .jobs import DaemonError
from .request_fields import field, object_body


class GymRoutes:
    """Environment catalogue, snapshots and session controls."""

    def route(self) -> bool:
        if not self.path.startswith("/ui/gym"):
            return super().route()
        try:
            configure_interpreters(self.ui)
            if self.path == "/ui/gym/catalogue" and self.method == "GET":
                self.send(200, {"environments": catalogue(), "models": model_choices()})
                return True
            if self.path == "/ui/gym/sessions" and self.method == "GET":
                self.send(200, {"sessions": manager.list()})
                return True
            if self.path == "/ui/gym/sessions" and self.method == "POST":
                req = object_body(self)
                environment_name = field(req, "environment", str, "")
                environment = getattr(self.ui, "environment", None)
                if environment is not None:
                    environment.require_current_runtime(python=interpreter(environment_name))
                config = field(req, "config", dict, {})
                world = field(config, "world", dict, {})
                runner = self.ui.runner
                if runner is not None:
                    os.environ["POOLHOUSE_GYM_FILES_ROOT"] = str(runner.files_root)
                if world.get("mode") == "manual":
                    if runner is None:
                        raise ValueError("Manual worlds require the daemon files root")
                    for name in ("map_file", "net_file", "route_file"):
                        if name in world:
                            safe_relpath(runner.files_root, field(world, name, str))
                self.send(201, manager.create(environment_name,
                          config=config, controller=field(req, "controller", str, "manual"),
                          seed=field(req, "seed", int, 0)))
                return True
            prefix = "/ui/gym/sessions/"
            if self.path.startswith(prefix):
                sid = self.path[len(prefix):]
                if sid.endswith("/events") and self.method == "GET":
                    return self._gym_events(manager, sid[:-7])
                if self.method == "GET":
                    self.send(200, manager.get(sid))
                elif self.method == "DELETE":
                    self.send(200, manager.close(sid))
                elif self.method == "POST":
                    req = object_body(self)
                    self.send(200, manager.control(sid, field(req, "command", str, ""),
                                                  payload=field(req, "payload", dict, {})))
                else:
                    self.send(405, {"error": "Use GET, POST or DELETE."})
                return True
        except ImportError as exc:
            self.send(501, {"error": f"Install poolhouse Gym dependencies: {exc}"})
            return True
        except (DaemonError, KeyError, ValueError, RuntimeError, OSError) as exc:
            self.send(400, {"error": str(exc)})
            return True
        self.send(405, {"error": "Unsupported Gym route or method."})
        return True

    def _gym_events(self, manager, sid: str) -> bool:
        snapshot = manager.get(sid)
        handler = self.handler
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Connection", "close")
        handler.end_headers()
        deadline = time.monotonic() + 20
        previous = None
        try:
            while time.monotonic() < deadline:
                encoded = json.dumps(snapshot)
                if encoded != previous:
                    handler.wfile.write(("data: " + encoded + "\n\n").encode())
                    handler.wfile.flush()
                    previous = encoded
                time.sleep(0.2)
                snapshot = manager.get(sid)
        except (BrokenPipeError, ConnectionResetError, KeyError):
            pass
        return True
