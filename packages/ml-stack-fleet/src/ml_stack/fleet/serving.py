"""Model servers on this machine, and reaching the ones on other machines."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Endpoint", "Served", "Serving", "discover_serving"]

# Each health path waits this long, and the beacon rebuilds the list every 10s.
PROBE_TIMEOUT = 1.0
CONNECT_TIMEOUT = 0.25
LIVE_CACHE_S = 5.0
HEALTH_PATHS = ("/health", "/v1/models", "/props")


@dataclass
class Served:
    """One model server running on this machine."""

    port: int
    models: list[str] = field(default_factory=list)
    slots: int = 1
    started_at: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        return asdict(self)


class Serving:
    """What this machine has loaded, and whether it is answering."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self._live: tuple[float, list[Served]] = (0.0, [])

    def register(self, port: int, models: list[str] | None = None,
                 slots: int = 1) -> Served:
        served = Served(port=port, models=[_name(m) for m in (models or [])],
                        slots=slots)
        current = [s for s in self.all() if s.port != port]
        self._write([*current, served])
        return served

    def unregister(self, port: int) -> None:
        self._write([s for s in self.all() if s.port != port])

    def all(self) -> list[Served]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []
        out = []
        for row in raw if isinstance(raw, list) else []:
            try:
                out.append(Served(port=int(row["port"]),
                                  models=list(row.get("models") or []),
                                  slots=int(row.get("slots") or 1),
                                  started_at=float(row.get("started_at") or 0)))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def live(self, *, force: bool = False) -> list[Served]:
        """Only the ones answering, rechecked at most every ``LIVE_CACHE_S``.

        Registration is a claim; a server that died leaves its entry behind, and a
        beacon advertising a model nobody can reach sends work to a dead port.
        """
        age, cached = self._live
        if not force and time.time() - age < LIVE_CACHE_S:
            return cached
        found = [s for s in self.all() if answers(s.port)]
        self._live = (time.time(), found)
        return found

    def port_for(self, model: str = "") -> int | None:
        for served in self.live():
            if not model or any(model.lower() in m.lower() for m in served.models):
                return served.port
        return None

    def public(self) -> list[dict[str, Any]]:
        return [s.public() for s in self.live()]

    def _write(self, rows: list[Served]) -> None:
        self._live = (0.0, [])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump([s.public() for s in rows], fh, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def answers(port: int, *, timeout: float = PROBE_TIMEOUT) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port),
                                      timeout=min(timeout, CONNECT_TIMEOUT)):
            pass
    except OSError:
        return False
    for path in HEALTH_PATHS:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
                if r.status < 400:
                    return True
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            continue
    return False


def _name(model: str) -> str:
    return Path(str(model)).name


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A model server on some machine, reachable through that machine's daemon."""

    peer: str
    base_url: str
    token: str
    models: tuple[str, ...] = ()
    slots: int = 1
    free: int = 1

    def client_kwargs(self) -> dict[str, str]:
        """Straight into ``ml_stack.client.Client(**endpoint.client_kwargs())``."""
        return {"base_url": f"{self.base_url}/infer", "api_key": self.token}


def discover_serving(key: bytes, *, model: str = "", timeout_s: float = 2.0,
                     group: str | None = None, port: int | None = None
                     ) -> list[Endpoint]:
    """Every machine on the network with a model loaded."""
    from .discovery import derive_token, discover

    token = derive_token(key)
    out = []
    for beacon in discover(key, timeout_s=timeout_s, group=group, port=port):
        for served in (beacon.device.get("serving") or []):
            models = tuple(served.get("models") or ())
            if model and not any(model.lower() in m.lower() for m in models):
                continue
            out.append(Endpoint(peer=beacon.name, base_url=beacon.base_url,
                                token=token, models=models,
                                slots=int(served.get("slots") or 1),
                                free=beacon.free))
    return out
