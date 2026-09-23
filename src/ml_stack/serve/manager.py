"""Leasing a server: start one, or adopt the one already running."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from ml_stack.client import is_healthy, reported_models
from ml_stack.client.health import serving_params
from ml_stack.files import write_json
from ml_stack.hub import free_memory
from ml_stack.hub import room as machine_room
from ml_stack.serve.backend import (
    Lease,
    LlamaServerBackend,
    ServerBackend,
    ServerFailed,
    ServerInfo,
    ServerSpec,
    default_slot_save_path,
)
from ml_stack.serve.escalation import (
    Escalating,
    plan_for,
    restore,
    save_live,
    slots_on,
    summarise,
)
from ml_stack.serve.events import Event, emit
from ml_stack.serve.leases import (
    lease_file,
    merge_state,
    orphaned,
    reap_one,
    recorded_servers,
)
from ml_stack.serve.matching import model_matches, serving_mismatch
from ml_stack.serve.mlx_tree import MlxTreeBackend, is_mlx
from ml_stack.serve.ports import DEFAULT_HOST, free_port, port_is_free, reclaim_port
from ml_stack.serve.process import (
    kill_process_tree,
    measuring,
    pid_exists,
    self_or_ancestor,
)
from ml_stack.serve.python_engines import ENGINES
from ml_stack.serve.weights import scaled_timeout, weight_of

logger = logging.getLogger(__name__)


class Measuring(ServerFailed):
    """A measurement holds the card, so no second model is loaded onto it."""


def measurement_on_the_card() -> dict[str, Any] | None:
    """The measurement holding the bench's lock, or None when nothing is or this process
    is the holder."""
    held = measuring()
    if not held or self_or_ancestor(held.get("pid")):
        return None
    return held


def measurement_said(held: dict[str, Any]) -> str:
    """The measurement named for a message: its command, its pid and when it started."""
    argv = " ".join(str(a) for a in (held.get("argv") or []))
    what = f"ml-stack-bench {argv}" if argv else "a measurement that wrote no record of itself"
    since = f", started {held['started']}" if held.get("started") else ""
    return f"{what} (pid {held.get('pid')}){since}"


UNAVAILABLE_COOLDOWN_S = 3.0
STATE_LOCK_TIMEOUT_S = 30.0

# How much of what is free a second model may take before it is judged not to fit. Below 1.0
# because a model needs its weights *and* room to work in, and a machine that fills itself
# exactly swaps instead of serving.
BESIDE_HEADROOM = 0.8


class ServerManager:
    """Leases model servers, one per (model, port), shared across this machine."""

    def __init__(
        self,
        backend: ServerBackend | None = None,
        *,
        state_file: Path | None = None,
    ) -> None:
        self.backend = backend or LlamaServerBackend()
        self.tree: ServerBackend = MlxTreeBackend()
        self.state_file = state_file or lease_file()
        self.say: Callable[[str], None] | None = None
        self._mine: dict[str, dict] = {}
        self._processes: dict[int, Any] = {}
        self._lock = threading.Lock()
        self._port_locks: dict[int, threading.Lock] = {}
        self._unavailable_until: dict[int, float] = {}

    def backend_for(self, spec: ServerSpec) -> ServerBackend:
        """The backend that serves ``spec``: the engine it names, tree decoding for MLX
        weights, anything else this manager's own."""
        if spec.engine:
            if spec.engine not in ENGINES:
                raise ServerFailed(f"no serving engine {spec.engine!r}; there are "
                                   f"{', '.join(sorted(ENGINES))}")
            return ENGINES[spec.engine]
        return self.tree if is_mlx(spec.model) else self.backend

    # ------------------------------------------------------------------ leasing

    def lease(self, spec: ServerSpec, *, timeout: float | None = None,
              roam: bool = True, check_flags: bool = True, preflight: bool = True,
              warmup_request: bool = True, escalate: bool = False, anyway: bool = False,
              on_event: Event | None = None,
              say: Callable[[str], None] | None = None) -> ServerInfo:
        """A healthy server for ``spec``. Starts one only if there is not one already.

        A server on the port whose record names a leasing process that has gone is an
        orphan: one serving what was asked for is adopted and its record made this
        process's; one serving something else is stopped before a server is started.
        Either way ``say`` (else ``self.say``, else the log) is told.

        When the port is busy with something else and this machine has the memory to hold
        both, it is served beside it on a free port rather than refused: a small model does
        not need the large one evicted, and making a person pick another port by hand is
        work a machine can do. ``roam=False`` for a caller that truly needs *that* port —
        the one that expects every consumer to meet on it.

        ``escalate=True`` is for a caller that may genuinely need more than one concurrent
        cache: when the only reason a running server does not match ``spec`` is that it
        holds fewer slots than asked, it is grown (or, if that will not fit, split, or
        summarised and split -- see :meth:`escalate`) rather than refused. A spec with no
        ``slot_save_path`` is given this manager's own default so a later escalation has
        somewhere to save a live conversation before the relaunch.

        ``timeout=None`` (the default) scales with the weights on disk -- see
        ``scaled_timeout`` -- so a caller that never thought about it still gets a timeout
        sized for what it is actually waiting on. A caller that passes a number means it,
        and gets exactly that instead.

        Starting a server is refused with `Measuring` while another process holds the
        bench's measuring lock; adopting one already up is not. ``anyway=True`` starts it
        regardless.
        """
        if escalate:
            # llama.cpp's slot-save file carries the cache's stream count, and a restore
            # raises "n_stream mismatch" the moment that count differs from the file's --
            # which a change in slot count always does unless every stream is one shared
            # buffer throughout, slot count or no. A lease that may later escalate is
            # kv_unified from its first launch, not only from the relaunch.
            if not spec.slot_save_path:
                spec = replace(spec, slot_save_path=str(default_slot_save_path()))
            if not spec.kv_unified:
                spec = replace(spec, kv_unified=True)

        now = time.monotonic()
        until = self._unavailable_until.get(spec.port, 0.0)
        if now < until:
            raise ServerFailed(
                f"port {spec.port} was marked unavailable {until - now:.1f}s ago; "
                "not retrying yet (negative cache)"
            )

        resolved_timeout = (
            timeout if timeout is not None else scaled_timeout(weight_of(spec.model)))
        starting = {"check_flags": check_flags, "preflight": preflight,
                    "warmup_request": warmup_request}
        refused = self._over_limit(spec)
        if refused:
            raise ServerFailed(refused)

        with self._port_lock(spec.port):
            told = say or self.say or logger.info
            entry = self._load().get(str(spec.port))
            stray = entry if isinstance(entry, dict) and orphaned(entry) else None
            try:
                adopted = self.adopt(spec)
            except ServerFailed as why:
                if escalate:
                    running = self._slots_shortfall(spec)
                    if running is not None:
                        return self.escalate(
                            running, add_slots=max(1, int(spec.parallel or 1))
                            - max(1, int(running.parallel or 1)),
                            timeout=resolved_timeout, anyway=anyway, on_event=on_event,
                            say=told)
                if stray is None:
                    if not roam or not port_is_free(spec.port):
                        elsewhere = (self._beside(spec, timeout=resolved_timeout,
                                                  on_event=on_event, anyway=anyway,
                                                  **starting)
                                    if roam else None)
                        if elsewhere is not None:
                            return elsewhere
                    raise
                self._stop_orphan(spec.port, stray, say=told, why=str(why))
            else:
                if adopted is not None:
                    if stray is not None:
                        self._take_over(spec.port, stray, say=told)
                    emit(on_event, "ready", port=spec.port, adopted=True)
                    return adopted

            try:
                info = self._launch(spec, timeout=resolved_timeout, on_event=on_event,
                                    anyway=anyway, **starting)
            except Measuring:
                raise
            except ServerFailed:
                self._forget(spec.port)
                self._unavailable_until[spec.port] = time.monotonic() + UNAVAILABLE_COOLDOWN_S
                raise

            self._unavailable_until.pop(spec.port, None)
            self._record(spec, info)
            return info

    def _launch(self, spec: ServerSpec, *, timeout: float, on_event: Event | None = None,
                anyway: bool = False, **starting: Any) -> ServerInfo:
        """Start ``spec``, telling ``on_event`` when the load begins and ends.

        The one place a fresh process is asked for, so ``up``, an adopt that falls
        through to a real start, and :meth:`escalate`'s relaunch all say the same thing
        the same way. Refused with `Measuring` while somebody else measures this card,
        unless ``anyway``.
        """
        held = None if anyway else measurement_on_the_card()
        if held is not None:
            raise Measuring(
                f"port {spec.port}: the card is being measured by "
                f"{measurement_said(held)}. Loading a second model onto it would spoil "
                f"that measurement and this one. Wait for it to finish, stop it with "
                f"'ml-stack-bench stop', or pass --anyway to load beside it.")
        emit(on_event, "loading", port=spec.port, model=Path(str(spec.model)).name,
              slots=max(1, int(spec.parallel or 1)))
        info = self.backend_for(spec).start(spec, lease=self._pending(spec), timeout=timeout,
                                  **starting)
        emit(on_event, "ready", port=spec.port, load_s=info.load_s, warmup_s=info.warmup_s)
        return info

    def _slots_shortfall(self, spec: ServerSpec) -> ServerSpec | None:
        """The settings actually running on ``spec.port``, if the only way they disagree with
        ``spec`` is holding fewer slots than asked. ``None`` for any other disagreement,
        or for nothing answering at all -- an escalation is a repair for one specific
        mismatch, not a second way to adopt."""
        base_url = f"http://{DEFAULT_HOST}:{spec.port}"
        if not is_healthy(base_url, timeout=1.0):
            return None
        models = reported_models(base_url)
        params = serving_params(base_url)
        loaded_file = params.model if params else None
        if models and not any(model_matches(m, spec.model, loaded_file=loaded_file) for m in models):
            return None
        if params is None or params.total_slots is None or params.n_ctx is None:
            return None
        wanted_slots = max(int(spec.parallel or 1), 1)
        if params.total_slots >= wanted_slots:
            return None
        per_slot = int(spec.context) // wanted_slots
        if params.n_ctx < per_slot:
            return None
        return replace(spec, parallel=params.total_slots,
                       context=params.n_ctx * params.total_slots)

    def _stop_orphan(self, port: int, entry: dict, *, say: Callable[[str], None],
                     why: str) -> None:
        """Stop the orphaned server recorded on ``port`` and drop its record."""
        say(f"port {port}: stopping the orphaned server (pid {entry['pid']}) the process "
            f"that leased it (pid {entry['owner_pid']}) left behind -- {why}")
        kill_process_tree(int(entry["pid"]))
        self._save()

    def _take_over(self, port: int, entry: dict, *, say: Callable[[str], None]) -> None:
        """Record the orphaned server on ``port`` as held by this process."""
        say(f"port {port}: adopted the orphaned server (pid {entry['pid']}) the process "
            f"that leased it (pid {entry['owner_pid']}) left behind; this process holds it "
            "now")
        self._mine[str(port)] = {**entry, "owner_pid": os.getpid()}
        self._save()

    def _beside(self, spec: ServerSpec, *, timeout: float, on_event: Event | None = None,
                anyway: bool = False, **starting: Any) -> ServerInfo | None:
        """Serve it next to whatever holds the port, when there is room. None when there is not.

        Room is judged against the weights on disk: a model is roughly its file size in
        memory, and a machine that cannot hold one more should say so rather than start a
        load that will be killed halfway or swap the other server to a crawl.
        """
        room = free_memory()
        wanted = weight_of(spec.model)
        if room is not None and wanted and wanted > room * BESIDE_HEADROOM:
            return None
        moved = replace(spec, port=free_port())
        with self._port_lock(moved.port):
            try:
                info = self._launch(moved, timeout=timeout, on_event=on_event,
                                    anyway=anyway, **starting)
            except Measuring:
                raise
            except ServerFailed:
                self._forget(moved.port)
                return None
            self._record(moved, info)
            return info

    def adopt(self, spec: ServerSpec) -> ServerInfo | None:
        """The already-running server for ``spec``, if there is one. Else ``None``."""
        base_url = f"http://{DEFAULT_HOST}:{spec.port}"
        if not is_healthy(base_url, timeout=1.0):
            return None

        mismatch = serving_mismatch(spec, reported_models(base_url), serving_params(base_url))
        if mismatch:
            raise ServerFailed(
                f"port {spec.port} is already serving different settings -- "
                + "; ".join(mismatch)
                + ". Stop it, or lease on a different port."
            )

        logger.info("adopting the server already healthy on %s", base_url)
        return ServerInfo(
            base_url=base_url,
            port=spec.port,
            pid=self._recorded_pid(spec.port),
            backend=self.backend_for(spec).name,
            adopted=True,
        )

    def escalate(self, spec: ServerSpec, *, add_slots: int = 1, room: int | None = None,
                timeout: float | None = None, anyway: bool = False,
                on_event: Event | None = None,
                say: Callable[[str], None] | None = None) -> ServerInfo:
        """Grow the server on ``spec.port`` by ``add_slots`` more concurrent conversations,
        keeping every one already live.

        ``spec`` is the settings actually running -- its ``context`` and ``parallel`` are
        what the port is serving now, not what a caller wishes it were (:meth:`lease`'s
        ``escalate=True`` works this out with :meth:`_slots_shortfall` before calling
        here). Every slot with a live conversation is saved through
        ``/slots/{id}?action=save`` before anything stops. The whole cache grows when
        ``fit`` says the extra room is there; otherwise the existing total is split
        across the larger slot count, and any conversation too long for what that leaves
        it is summarised on the model itself and re-seeded in place of its cache -- which
        is kept regardless, named in every message about that slot. Raises
        :class:`~ml_stack.serve.escalation.EscalationRefused` only when a live
        conversation would be dropped and summarising it did not rescue that.
        """
        told = say or self.say or logger.info
        current_slots = max(1, int(spec.parallel or 1))
        new_slots = current_slots + max(1, int(add_slots))
        per_slot = max(1, int(spec.context) // current_slots)
        run = Escalating(f"http://{DEFAULT_HOST}:{spec.port}", spec.port,
                         timeout=timeout, on_event=on_event)

        if not is_healthy(run.base_url, timeout=2.0):
            raise ServerFailed(f"nothing is answering on port {spec.port} to escalate")
        if not spec.slot_save_path:
            raise ServerFailed(
                f"port {spec.port} was not started with --slot-save-path; its live "
                "conversations cannot be saved before a relaunch"
            )

        live, prompts = slots_on(run)
        room_bytes = int(room) if room is not None else machine_room()
        plan = plan_for(spec, new_slots=new_slots, per_slot=per_slot, live=live,
                        room_bytes=room_bytes)

        emit(on_event, "escalating", port=spec.port, mode=plan.mode, reason=plan.reason,
             from_slots=current_slots, to_slots=new_slots)
        told(f"port {spec.port}: escalating from {current_slots} to {new_slots} slot(s) "
            f"by {plan.mode} -- {plan.reason}")

        saved = save_live(run, live)
        summaries = summarise(run, plan.too_long, prompts=prompts, saved=saved)

        pid = self._recorded_pid(spec.port)
        emit(on_event, "stopping", port=spec.port, pid=pid)
        if pid:
            kill_process_tree(pid)
        self._forget(spec.port)

        # kv_unified keeps the cache's stream count at 1 across the relaunch; any other
        # value makes a save from the old slot count unrestorable into the new one,
        # "n_stream mismatch" thrown by llama.cpp's own state reader whichever slot it is.
        new_spec = replace(spec, parallel=new_slots, context=plan.new_context,
                           kv_unified=True)
        resolved_timeout = (
            timeout if timeout is not None else scaled_timeout(weight_of(spec.model)))
        info = self._launch(new_spec, timeout=resolved_timeout, on_event=on_event,
                            anyway=anyway)
        # recorded now, not after every restore: a restore failure below must not leave a
        # live, healthy process this manager has forgotten it started
        self._record(new_spec, info)

        restore(replace(run, base_url=info.base_url), live, summaries=summaries,
                saved=saved)

        emit(on_event, "done", port=spec.port, slots=new_slots, mode=plan.mode)
        told(f"port {spec.port}: now serving {new_slots} slot(s)")
        return info

    def release(self, info: ServerInfo, *, grace_s: float = 5.0) -> None:
        """Stop a server this process started. Adopted servers are left running."""
        if info.adopted:
            logger.debug("not stopping %s: we adopted it", info.base_url)
            return
        held = self._processes.pop(info.port, None)
        if held is None:
            held = info.process
        if info.pid:
            kill_process_tree(info.pid, grace_s=grace_s)
        reap_one(held, grace_s=grace_s)
        self._forget(info.port)

    def detach(self, info: ServerInfo) -> None:
        """Record the server under its own pid and stop tracking it in this process."""
        self._processes.pop(info.port, None)
        entry = self._mine.pop(str(info.port), None)
        if entry is None or not info.pid:
            self._save()
            return
        with self._exclusive():
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
        for held in list(self._processes.values()):
            reap_one(held, grace_s=grace_s)
        self._processes.clear()
        self._mine.clear()
        self._save()
        return stopped

    # ------------------------------------------------------------------ state file

    def _pending(self, spec: ServerSpec) -> Lease:
        """Write the server down before it exists, and hand the backend the proof.

        The record carries the port, the model and this process as owner with no pid yet;
        `_record` fills the pid in once the server answers, `_forget` takes the entry out
        if it never does. Between the two, a port that looks unrecorded is one this
        manager is starting on -- never one it may kill.
        """
        self._mine[str(spec.port)] = {
            "port": spec.port, "pid": None, "backend": self.backend_for(spec).name,
            "model": str(spec.model), "owner_pid": os.getpid(), "pending": True,
        }
        self._save()
        return Lease(port=spec.port, owner_pid=os.getpid(), state_file=str(self.state_file))

    def _record(self, spec: ServerSpec, info: ServerInfo) -> None:
        self._mine[str(spec.port)] = {
            "port": info.port,
            "pid": info.pid,
            "backend": info.backend,
            "model": str(spec.model),
            "owner_pid": os.getpid(),
            "base_url": info.base_url,
            "load_s": info.load_s,
            "warmup_s": info.warmup_s,
        }
        if info.process is not None:
            self._processes[info.port] = info.process
        self._save()

    def _forget(self, port: int, *, grace_s: float = 5.0) -> None:
        reap_one(self._processes.pop(port, None), grace_s=grace_s)
        self._mine.pop(str(port), None)
        self._save()

    def _load(self) -> dict:
        try:
            parsed = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _write(self, state: dict) -> None:
        write_json(self.state_file, state)

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        """Hold the state file against every other thread and process for the block."""
        from ml_stack.lock import only_one

        with self._lock, only_one(self.state_file.with_suffix(".lock"), wait=True,
                                  timeout=STATE_LOCK_TIMEOUT_S, announce=logger.debug):
            yield

    def _reap(self) -> None:
        """Drop every child of ours that has already exited, so none stays defunct."""
        for port, held in list(self._processes.items()):
            try:
                if held.poll() is not None:
                    self._processes.pop(port, None)
            except OSError:
                self._processes.pop(port, None)

    def _save(self) -> None:
        self._reap()
        with self._exclusive():
            self._write(merge_state(self._load(), self._mine, os.getpid()))

    def _recorded_pid(self, port: int) -> int | None:
        entry = self._load().get(str(port))
        pid = entry.get("pid") if isinstance(entry, dict) else None
        return pid if isinstance(pid, int) else None

    def _port_lock(self, port: int) -> threading.Lock:
        with self._lock:
            return self._port_locks.setdefault(port, threading.Lock())

    def _over_limit(self, spec: ServerSpec) -> str:
        """Why this machine's limits refuse this lease, or "". A server already up on this
        port is adopted rather than added, so it is not counted against the server limit."""
        from ml_stack.limits import read

        limits = read()
        if not (limits.servers or limits.slots):
            return ""
        running = sum(1 for port, entry in recorded_servers(self.state_file).items()
                      if port != spec.port and pid_exists(int(entry.get("pid") or 0)))
        return limits.refusal(running=running, slots=max(1, int(spec.parallel or 1)))

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
    timeout: float | None = None,
    manager: ServerManager | None = None,
    roam: bool = True,
    escalate: bool = False,
    anyway: bool = False,
    on_event: Event | None = None,
    say: Callable[[str], None] | None = None,
    **spec_kwargs: object,
) -> Iterator[ServerInfo]:
    """Run a server for the duration of the block, yielding its ``ServerInfo``.

    ``roam``, ``escalate``, ``anyway``, ``on_event`` and ``say`` go to
    :meth:`ServerManager.lease`.
    """
    manager = manager or _DEFAULT
    spec = ServerSpec(
        model=model,
        port=port if port is not None else free_port(),
        context=context,
        **spec_kwargs,  # type: ignore[arg-type]
    )
    info = manager.lease(spec, timeout=timeout, roam=roam, escalate=escalate,
                         anyway=anyway, on_event=on_event, say=say)
    try:
        yield info
    finally:
        manager.release(info)


def stop_all_servers() -> list[int]:
    """Stop every model server recorded on this machine by any live owner."""
    stopped: list[int] = []
    held = lease_file()
    try:
        state = json.loads(held.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return stopped

    for entry in state.values() if isinstance(state, dict) else []:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("pid")
        if isinstance(pid, int) and pid_exists(pid):
            stopped += kill_process_tree(pid)

    held.unlink(missing_ok=True)
    return stopped
