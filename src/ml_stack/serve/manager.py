"""Leasing a server: start one, or adopt the one already running."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

from ml_stack.client import is_healthy, reported_models
from ml_stack.client.health import ServingParams, serving_params
from ml_stack.serve.backend import (
    LlamaServerBackend,
    ServerBackend,
    ServerFailed,
    ServerInfo,
    ServerSpec,
)
from ml_stack.serve.binary import CACHE_ROOT
from ml_stack.serve.process import kill_process_tree, pid_exists
from ml_stack.serve.ports import DEFAULT_HOST, reclaim_port

logger = logging.getLogger(__name__)

STATE_FILE = CACHE_ROOT / "servers.json"
UNAVAILABLE_COOLDOWN_S = 3.0


def merge_state(on_disk: dict, mine: dict, owner_pid: int) -> dict:
    """This process's servers, merged over records other *live* processes left."""
    merged = {
        key: entry
        for key, entry in on_disk.items()
        if isinstance(entry, dict)
        and entry.get("owner_pid") != owner_pid
        and pid_exists(entry.get("owner_pid"))
    }
    merged.update(mine)
    return merged


def model_matches(reported: str, wanted: str | Path) -> bool:
    """Whether a server reporting ``reported`` is serving ``wanted``."""
    wanted_name = Path(str(wanted).removeprefix("hf:")).name.lower()
    reported_name = Path(reported).name.lower()
    if not wanted_name or not reported_name:
        return False
    return wanted_name in reported_name or reported_name in wanted_name


def shape_mismatch(
    spec: ServerSpec,
    models: list[str],
    params: ServingParams | None,
) -> list[str]:
    """Each field in which a running server differs from ``spec``. Empty when it fits."""
    out: list[str] = []

    if models and not any(model_matches(m, spec.model) for m in models):
        serving = ", ".join(repr(Path(name).name) for name in models)
        asked = Path(str(spec.model).removeprefix("hf:")).name
        out.append(f"model: asked for {asked!r}, serving {serving}")

    if params is None:
        return out

    slots = max(int(spec.parallel or 1), 1)
    if params.total_slots is not None and params.total_slots < slots:
        out.append(f"slots: asked for {slots}, serving {params.total_slots}")

    # llama-server reports the context of one slot: --ctx-size divided by -np.
    per_slot = int(spec.context) // slots
    if params.n_ctx is not None and params.n_ctx < per_slot:
        out.append(f"context: asked for {per_slot} per slot, serving {params.n_ctx}")

    return out


def recorded_servers(state_file: Path | None = None) -> dict[int, dict]:
    """Every server in the lease file, keyed by port."""
    try:
        parsed = json.loads((state_file or STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}

    out: dict[int, dict] = {}
    for key, entry in parsed.items():
        if not isinstance(entry, dict):
            continue
        try:
            out[int(entry.get("port", key))] = entry
        except (TypeError, ValueError):
            continue
    return out


class ServerManager:
    """Leases model servers, one per (model, port), shared across this machine."""

    def __init__(
        self,
        backend: ServerBackend | None = None,
        *,
        state_file: Path | None = None,
    ) -> None:
        self.backend = backend or LlamaServerBackend()
        self.state_file = state_file or STATE_FILE
        self._mine: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._port_locks: dict[int, threading.Lock] = {}
        self._unavailable_until: dict[int, float] = {}

    # ------------------------------------------------------------------ leasing

    def lease(self, spec: ServerSpec, *, timeout: float = 300.0) -> ServerInfo:
        """A healthy server for ``spec``. Starts one only if there is not one already."""
        now = time.monotonic()
        until = self._unavailable_until.get(spec.port, 0.0)
        if now < until:
            raise ServerFailed(
                f"port {spec.port} was marked unavailable {until - now:.1f}s ago; "
                "not retrying yet (negative cache)"
            )

        with self._port_lock(spec.port):
            adopted = self.adopt(spec)
            if adopted is not None:
                return adopted

            try:
                info = self.backend.start(spec, timeout=timeout)
            except ServerFailed:
                self._unavailable_until[spec.port] = time.monotonic() + UNAVAILABLE_COOLDOWN_S
                raise

            self._unavailable_until.pop(spec.port, None)
            self._record(spec, info)
            return info

    def adopt(self, spec: ServerSpec) -> ServerInfo | None:
        """The already-running server for ``spec``, if there is one. Else ``None``."""
        base_url = f"http://{DEFAULT_HOST}:{spec.port}"
        if not is_healthy(base_url, timeout=1.0):
            return None

        mismatch = shape_mismatch(spec, reported_models(base_url), serving_params(base_url))
        if mismatch:
            raise ServerFailed(
                f"port {spec.port} is already serving a different shape -- "
                + "; ".join(mismatch)
                + ". Stop it, or lease on a different port."
            )

        logger.info("adopting the server already healthy on %s", base_url)
        return ServerInfo(
            base_url=base_url,
            port=spec.port,
            pid=self._recorded_pid(spec.port),
            backend=self.backend.name,
            adopted=True,
        )

    def release(self, info: ServerInfo, *, grace_s: float = 5.0) -> None:
        """Stop a server this process started. Adopted servers are left running."""
        if info.adopted:
            logger.debug("not stopping %s: we adopted it", info.base_url)
            return
        if info.pid:
            kill_process_tree(info.pid, grace_s=grace_s)
        self._forget(info.port)

    def detach(self, info: ServerInfo) -> None:
        """Record the server under its own pid and stop tracking it in this process."""
        entry = self._mine.pop(str(info.port), None)
        if entry is None or not info.pid:
            self._save()
            return
        with self._lock:
            state = merge_state(self._load(), self._mine, os.getpid())
            state[str(info.port)] = {**entry, "owner_pid": info.pid}
            self._write(state)

    def stop_all(self, *, grace_s: float = 5.0) -> list[int]:
        """Stop every server this process started."""
        stopped: list[int] = []
        for entry in list(self._mine.values()):
            pid = entry.get("pid")
            if isinstance(pid, int) and pid_exists(pid):
                stopped += kill_process_tree(pid, grace_s=grace_s)
        self._mine.clear()
        self._save()
        return stopped

    # ------------------------------------------------------------------ state file

    def _record(self, spec: ServerSpec, info: ServerInfo) -> None:
        self._mine[str(spec.port)] = {
            "port": info.port,
            "pid": info.pid,
            "backend": info.backend,
            "model": str(spec.model),
            "owner_pid": os.getpid(),
            "base_url": info.base_url,
        }
        self._save()

    def _forget(self, port: int) -> None:
        self._mine.pop(str(port), None)
        self._save()

    def _load(self) -> dict:
        try:
            parsed = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _write(self, state: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_file)

    def _save(self) -> None:
        with self._lock:
            self._write(merge_state(self._load(), self._mine, os.getpid()))

    def _recorded_pid(self, port: int) -> int | None:
        entry = self._load().get(str(port))
        pid = entry.get("pid") if isinstance(entry, dict) else None
        return pid if isinstance(pid, int) else None

    def _port_lock(self, port: int) -> threading.Lock:
        with self._lock:
            return self._port_locks.setdefault(port, threading.Lock())

    def reclaim(self, port: int) -> bool:
        """Free ``port`` if one of our servers is holding it."""
        recorded = self._recorded_pid(port)
        return reclaim_port(port, recorded_pids=[recorded] if recorded else None)


_DEFAULT = ServerManager()


@contextmanager
def serve(
    model: str | Path,
    *,
    port: int | None = None,
    context: int = 4096,
    timeout: float = 300.0,
    manager: ServerManager | None = None,
    **spec_kwargs: object,
) -> Iterator[ServerInfo]:
    """Run a server for the duration of the block, yielding its ``ServerInfo``."""
    from ml_stack.serve.ports import free_port

    manager = manager or _DEFAULT
    spec = ServerSpec(
        model=model,
        port=port if port is not None else free_port(),
        context=context,
        **spec_kwargs,  # type: ignore[arg-type]
    )
    info = manager.lease(spec, timeout=timeout)
    try:
        yield info
    finally:
        manager.release(info)


def stop_all_servers() -> list[int]:
    """Stop every model server recorded on this machine by any live owner."""
    stopped: list[int] = []
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return stopped

    for entry in state.values() if isinstance(state, dict) else []:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("pid")
        if isinstance(pid, int) and pid_exists(pid):
            stopped += kill_process_tree(pid)

    STATE_FILE.unlink(missing_ok=True)
    return stopped
