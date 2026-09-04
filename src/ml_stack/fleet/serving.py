"""Model servers on this machine, and reaching the ones on other machines."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any

from ml_stack.files import write_json

__all__ = ["Endpoint", "Hosting", "NoRoom", "Served", "Serving", "Started",
           "discover_serving", "start_model", "stop_model"]

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
        write_json(self.path, [s.public() for s in rows])


@dataclass(frozen=True, slots=True)
class Started:
    """A model server this process leased, and its entry in the registry."""

    port: int
    lease: Any
    manager: Any
    served: Served | None = None


def start_model(root: Path | str, model_path: Path | str, *, name: str | None = None,
                context: int = 8192, parallel: int = 1, manager: Any = None,
                serving: Serving | None = None, port: int | None = None) -> Started:
    """Run ``model_path`` on this machine with ``parallel`` seats of ``context`` tokens.
    Registers the port when given a ``Serving``."""
    from ml_stack.serve import (
        LlamaServerBackend, ServerManager, ServerSpec, free_port)

    if manager is None:
        from .llama import ensure_server
        manager = ServerManager(
            backend=LlamaServerBackend(binary=ensure_server(root)))

    from .models import draft_beside

    if port is None:
        port = free_port()
    parallel = max(1, int(parallel))
    extra: tuple[str, ...] = ()
    draft = draft_beside(Path(model_path))
    if draft is not None:
        # -md is what this build calls --spec-draft-model.
        extra = ("-md", str(draft), "-ngld", "99")
    lease = manager.lease(ServerSpec(model=model_path, port=port, context=int(context),
                                     parallel=parallel, extra_args=extra))
    served = None
    if serving is not None:
        served = serving.register(port, [name or Path(model_path).name], slots=parallel)
    return Started(port=port, lease=lease, manager=manager, served=served)


class NoRoom(RuntimeError):
    """The model and its seats do not fit in this machine's room."""


class Hosting:
    """The model servers this process started, by port."""

    def __init__(self, root: Path | str, serving: Serving, *, manager: Any = None,
                 fits: Callable[[], Sequence[Any]] | None = None) -> None:
        self.root = Path(root).expanduser()
        self.serving = serving
        self.manager = manager
        self.fits = fits
        self.leases: dict[int, Any] = {}

    def already(self, name: str) -> Served | None:
        """The live server holding ``name``, or None."""
        wanted = _name(name).lower()
        for served in self.serving.live(force=True):
            if any(_name(m).lower() == wanted for m in served.models):
                return served
        return None

    def fits_in(self, name: str, *, context: int, parallel: int, room: int) -> str:
        """"" when the model with ``parallel`` seats fits in ``room`` bytes by its memory
        record, or a line saying what it needs. A model with no record passes."""
        from ml_stack.hub import _human
        from ml_stack.serve.fit import records
        from .plan import fit_for

        if room <= 0:
            return ""
        fit = fit_for(name, list(self.fits() if self.fits else records()))
        if fit is None:
            return ""
        loaded, each = fit.at_room(room).line(context)
        need = loaded + max(1, int(parallel)) * each
        if need <= room:
            return ""
        return (f"{_human(need)} for {parallel} seat(s) at {context} tokens; "
                f"this machine has {_human(room)}")

    def start(self, model_path: Path | str, *, name: str = "", context: int = 8192,
              parallel: int = 1, room: int = 0) -> Served:
        """Serve ``model_path`` here, or raise `NoRoom`."""
        name = name or Path(model_path).name
        why = self.fits_in(name, context=int(context), parallel=int(parallel), room=int(room))
        if why:
            raise NoRoom(f"{name} does not fit: {why}")
        started = start_model(self.root, model_path, name=name, context=int(context),
                              parallel=int(parallel), manager=self.manager,
                              serving=self.serving)
        self.manager = started.manager
        self.leases[started.port] = started.lease
        return started.served or Served(port=started.port, models=[name],
                                        slots=max(1, int(parallel)))

    def stop(self, port: int) -> None:
        """Release the server on ``port``; a port this process did not start is only
        taken out of the registry."""
        held = self.leases.pop(port, None)
        if held is None or self.manager is None:
            self.serving.unregister(port)
            return
        stop_model(Started(port=port, lease=held, manager=self.manager),
                   serving=self.serving)


def stop_model(started: Started, serving: Serving | None = None) -> None:
    """Release the lease. Takes the port out of the registry when given one."""
    if started.manager is not None:
        started.manager.release(started.lease)
    if serving is not None:
        serving.unregister(started.port)


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
