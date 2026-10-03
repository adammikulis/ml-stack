"""Authenticated live Gym sessions."""

from __future__ import annotations

import json
import time


class GymRoutes:
    """Environment catalogue, snapshots and session controls."""

    def route(self) -> bool:
        if not self.path.startswith("/ui/gym"):
            return super().route()
        try:
            from ml_stack.gym import catalogue, manager
            if self.path == "/ui/gym/catalogue" and self.method == "GET":
                self.send(200, {"environments": catalogue()})
                return True
            if self.path == "/ui/gym/sessions" and self.method == "POST":
                req = self.body()
                self.send(201, manager.create(str(req.get("environment") or ""),
                          config=req.get("config"), controller=str(req.get("controller") or "manual"),
                          seed=int(req.get("seed", 0))))
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
                    req = self.body()
                    self.send(200, manager.control(sid, str(req.get("command") or ""),
                                                  payload=req.get("payload")))
                else:
                    self.send(405, {"error": "Use GET, POST or DELETE."})
                return True
        except ImportError as exc:
            self.send(501, {"error": f"Install ml-stack Gym dependencies: {exc}"})
            return True
        except (KeyError, ValueError, RuntimeError, OSError) as exc:
            self.send(400, {"error": str(exc)})
            return True
        return super().route()

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
