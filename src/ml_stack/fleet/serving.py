"""Model servers on this machine, and reaching the ones on other machines."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ml_stack import lock
from ml_stack.client.counters import read_speculative
from ml_stack.client.health import is_healthy, reported_models, serving_params
from ml_stack.client.settings import Transport
from ml_stack.files import write_json
from ml_stack.serve.process import every_server
from ml_stack.units import human_bytes

__all__ = ["Endpoint", "Hosting", "NoRoom", "Served", "Serving", "Started",
           "discover_serving", "start_model", "stop_model"]

# Each health path waits this long, and the beacon rebuilds the list every 10s.
PROBE_TIMEOUT = 1.0
CONNECT_TIMEOUT = 0.25
LIVE_CACHE_S = 5.0


@dataclass
class Served:
    """One model server running on this machine."""

    port: int
    models: list[str] = field(default_factory=list)
    slots: int = 1
    started_at: float = field(default_factory=time.time)
    context: int | None = None
    slot_context: int | None = None
    draft: str = ""
    spec_type: str = ""
    draft_status: str = "unknown"
    aliases: list[str] = field(default_factory=list)

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
        with lock.only_one(self.path.with_suffix(".lock"), timeout=5, announce=lambda text: None):
            current = [s for s in self.all() if s.port != port]
            self._write([*current, served])
        return served

    def unregister(self, port: int) -> None:
        with lock.only_one(self.path.with_suffix(".lock"), timeout=5, announce=lambda text: None):
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
                port = row["port"]
                if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                    continue
                out.append(Served(port=port,
                                  models=list(row.get("models") or []),
                                  slots=int(row.get("slots") or 1),
                                  started_at=float(row.get("started_at") or 0),
                                  context=row.get("context"), slot_context=row.get("slot_context"),
                                  draft=str(row.get("draft") or ""), spec_type=str(row.get("spec_type") or ""),
                                  draft_status=str(row.get("draft_status") or "unknown"),
                                  aliases=list(row.get("aliases") or [])))
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
        snapshot = self.all()
        running = {row["port"]: row for row in every_server()}
        verified = {served.port: _identity(served, running.get(served.port, {})) for served in snapshot}
        found = [served for served, answering in verified.values() if answering]
        if snapshot:
            with lock.only_one(self.path.with_suffix(".lock"), timeout=5, announce=lambda text: None):
                old = {served.port: served for served in snapshot}
                current = self.all()
                reconciled = [verified[served.port][0] if served == old.get(served.port) else served for served in current]
                kept = [served for served in reconciled if served is not None]
                if kept != current:
                    self._write(kept)
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
                serving: Serving | None = None, port: int | None = None,
                escalate: bool = False) -> Started:
    """Run ``model_path`` on this machine with ``parallel`` slots of ``context`` tokens.
    Registers the port when given a ``Serving``.

    ``escalate=True`` grows a server already up on ``port`` with fewer slots than
    ``parallel`` asks for, rather than refusing -- see
    :meth:`~ml_stack.serve.ServerManager.escalate`.
    """
    from ml_stack.serve import LlamaServerBackend, ServerManager, ServerSpec, free_port

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
                                     parallel=parallel, extra_args=extra),
                          escalate=escalate)
    served = None
    if serving is not None:
        served = serving.register(port, [name or Path(model_path).name], slots=parallel)
    return Started(port=port, lease=lease, manager=manager, served=served)


class NoRoom(RuntimeError):
    """The model and its slots do not fit in this machine's room."""


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
        """"" when the model with ``parallel`` slots fits in ``room`` bytes by its memory
        record, or a line saying what it needs. A model with no record passes."""
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
        return (f"{human_bytes(need)} for {parallel} slot(s) at {context} tokens; "
                f"this machine has {human_bytes(room)}")

    def start(self, model_path: Path | str, *, name: str = "", context: int = 8192,
              parallel: int = 1, room: int = 0, escalate: bool = True) -> Served:
        """Serve ``model_path`` here, or raise `NoRoom`.

        ``escalate`` (on by default: a pool is exactly the place more than one
        conversation is expected) grows a server already up with fewer slots than
        ``parallel`` asks for rather than refusing.
        """
        name = name or Path(model_path).name
        why = self.fits_in(name, context=int(context), parallel=int(parallel), room=int(room))
        if why:
            raise NoRoom(f"{name} does not fit: {why}")
        started = start_model(self.root, model_path, name=name, context=int(context),
                              parallel=int(parallel), manager=self.manager,
                              serving=self.serving, escalate=escalate)
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


def _identity(served: Served, process: dict) -> tuple[Served | None, bool]:
    try:
        with socket.create_connection(("127.0.0.1", served.port), timeout=CONNECT_TIMEOUT):
            pass
    except OSError:
        return None, False
    def local_only(url: str) -> str:
        target = urlsplit(url)
        if target.scheme != "http" or target.hostname != "127.0.0.1" or target.port != served.port:
            raise ValueError("model metadata must stay on its registered local endpoint")
        return url

    try:
        names = reported_models(f"http://127.0.0.1:{served.port}", timeout=PROBE_TIMEOUT, guard=local_only)
    except (OSError, ValueError, HTTPException):
        return served, False
    if not names:
        return served, False
    base = f"http://127.0.0.1:{served.port}"
    try:
        params = serving_params(base, timeout=PROBE_TIMEOUT, guard=local_only)
        counts = read_speculative(base, timeout=PROBE_TIMEOUT, guard=local_only)
    except (OSError, ValueError, HTTPException):
        params, counts = None, None
    slots = params.total_slots if params and type(params.total_slots) is int and params.total_slots > 0 else served.slots
    slot_context = params.n_ctx if params and type(params.n_ctx) is int and params.n_ctx > 0 else None
    context = slot_context * slots if slot_context else None
    spec_type = str(process.get("spec_type") or "")
    draft = str(process.get("draft") or "")
    state = "active" if counts and counts.drafted > 0 else "configured" if spec_type else "unknown"
    return Served(served.port, [_name(name) for name in names], slots, served.started_at,
                  context, slot_context, Path(draft).name if draft else "", spec_type, state, names), True


def answers(port: int, *, timeout: float = PROBE_TIMEOUT) -> bool:
    """Whether a model server on ``port`` of this machine is up."""
    try:
        with socket.create_connection(("127.0.0.1", port),
                                      timeout=min(timeout, CONNECT_TIMEOUT)):
            pass
    except OSError:
        return False
    return is_healthy(f"http://127.0.0.1:{port}", timeout=timeout)


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

    def client_kwargs(self) -> dict[str, Any]:
        """Straight into ``ml_stack.client.Client(**endpoint.client_kwargs())``."""
        return {"base_url": f"{self.base_url}/infer",
                "transport": Transport(api_key=self.token)}


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
