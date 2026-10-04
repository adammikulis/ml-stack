"""Leasing a server: start one, or adopt the one already running."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ml_stack.client import is_healthy, reported_models
from ml_stack.client.chat import forget_server
from ml_stack.client.health import serving_params
from ml_stack.files import write_json
from ml_stack.hub import free_memory, installed_for, room as machine_room
from ml_stack.limits import read as limits_read
from ml_stack.lock import only_one
from ml_stack.serve import admission, exit_guard, guarded, mtp, provenance, quant_guard, unmanaged
from ml_stack.serve.admitting import STATE_LOCK_TIMEOUT_S, Admitting
from ml_stack.serve.backend import (
    Lease,
    LlamaServerBackend,
    ServerBackend,
    ServerFailed,
    ServerInfo,
    ServerSpec,
    UnknownFlag,
    default_slot_save_path,
)
from ml_stack.serve.binary import BinaryNotFound
from ml_stack.serve.escalation import (
    Escalating,
    plan_for,
    restore,
    save_live,
    slots_on,
    summarise,
)
from ml_stack.serve.events import Caller, Event, Growth, emit
from ml_stack.serve.leases import (
    lease_file,
    merge_state,
    orphaned,
    reap_one,
    recorded_servers,
    same_process,
)
from ml_stack.serve.matching import model_matches, serving_mismatch
from ml_stack.serve.mlx_tree import MlxTreeBackend, is_mlx
from ml_stack.serve.ports import DEFAULT_HOST, free_port, port_is_free, reclaim_port
from ml_stack.serve.process import (
    cmdline_digest,
    kill_process_tree,
    measuring,
    pid_exists,
    self_or_ancestor,
    started_at,
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

# How much of what is free a second model may take before it is judged not to fit. Below 1.0
# because a model needs its weights *and* room to work in, and a machine that fills itself
# exactly swaps instead of serving.
BESIDE_HEADROOM = 0.8


@dataclass(frozen=True)
class Starting:
    """How a lease may be satisfied: ``roam`` lets it be served on another port, ``escalate``
    lets a server with too few slots be grown, ``anyway`` starts it during a measurement,
    ``iq`` is the IQ-quantisation mode (``off``, ``warn``, ``block``), ``who`` names the asker,
    and the rest are the checks the start runs."""

    roam: bool = True
    check_flags: bool = True
    preflight: bool = True
    warmup_request: bool = True
    escalate: bool = False
    anyway: bool = False
    iq: str = ""
    who: str = ""

    def checks(self) -> dict[str, bool]:
        """The checks a start runs, as the keyword arguments a backend takes."""
        return {"check_flags": self.check_flags, "preflight": self.preflight,
                "warmup_request": self.warmup_request}


def reusing_installed(spec: ServerSpec) -> ServerSpec:
    """``spec`` with each ``hf:owner/repo/file`` model, projector and draft that is already
    installed -- from the Hub cache, llama.cpp's cache, LM Studio or Ollama -- replaced by
    the file, so it is not downloaded again."""
    changes: dict[str, str] = {}
    for field in ("model", "mmproj", "draft"):
        value = getattr(spec, field)
        found = installed_for(value) if isinstance(value, str) and value.startswith("hf:") else None
        if found:
            changes[field] = str(found.path)
    return replace(spec, **changes) if changes else spec


class ServerManager(Admitting):
    """Leases model servers, one per (model, port), shared across this machine."""

    def __init__(
        self,
        backend: ServerBackend | None = None,
        *,
        state_file: Path | None = None,
        broker: Any = None,
        stop_on_exit: bool = True,
    ) -> None:
        self.stop_on_exit = stop_on_exit
        self.iq = ""
        self.backend = backend or LlamaServerBackend()
        self.tree: ServerBackend = MlxTreeBackend()
        self.state_file = state_file or lease_file()
        self.say: Callable[[str], None] | None = None
        self._broker = broker
        self.confirm: Callable[[str], bool] | None = None
        self._leases: dict[str, ServerInfo] = {}
        self._mine: dict[str, dict] = {}
        self._processes: dict[int, Any] = {}
        self._lock = threading.Lock()
        self._port_locks: dict[int, threading.Lock] = {}
        self._unavailable_until: dict[int, float] = {}
        self._swept = False

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

    def with_backend(self, backend: ServerBackend) -> ServerManager:
        """A manager over ``backend`` that shares this one's records and processes."""
        other = ServerManager(backend, state_file=self.state_file, broker=self._broker,
                              stop_on_exit=self.stop_on_exit)
        other._mine, other._processes = self._mine, self._processes
        other._lock, other._port_locks = self._lock, self._port_locks
        other.tree = self.tree
        return other

    @property
    def broker(self) -> Any:
        """The broker every start, release and escalation goes through."""
        if self._broker is None:
            from ml_stack.serve.broker_wire import broker_for  # broker_wire imports this module

            self._broker = broker_for(self)
        return self._broker

    def lease(self, spec: ServerSpec, *, timeout: float | None = None,
              roam: bool = True, check_flags: bool = True, preflight: bool = True,
              warmup_request: bool = True, escalate: bool = False, anyway: bool = False,
              iq: str = "", on_event: Event | None = None,
              say: Callable[[str], None] | None = None, reason: str = "") -> ServerInfo:
        """A healthy server for ``spec``, from the broker: one already up that fits, else a
        new one when the machine has the memory for it. The broker waits for memory and
        refuses with `AdmissionRefused` when none comes free. ``reason`` is why, in one line.

        ``on_event`` and ``say`` are called with a broker running in this process and not
        with the machine's broker. The rest of the arguments are those of
        :meth:`_start_server`.
        """
        spec = reusing_installed(spec)
        wanted = quant_guard.mode(iq or self.iq)
        if wanted == "block":
            quant_guard.enforce(spec.model, gpu_layers=spec.n_gpu_layers, asked=wanted)
        how = Starting(roam, check_flags, preflight, warmup_request, escalate, anyway, wanted)
        info = self.broker.start(spec, Caller(on_event=on_event, say=say or self.say,
                                              claim=provenance.asked(reason)),
                                 timeout=timeout, options=asdict(how))
        if info.lease:
            self._leases[info.lease] = info
        return info

    def _start_server(self, spec: ServerSpec, *, timeout: float | None = None,
                      how: Starting | None = None, on_event: Event | None = None,
                      say: Callable[[str], None] | None = None) -> ServerInfo:
        """A healthy server for ``spec``: the one on its port, else one already up elsewhere that
        serves it, else a new one. Called by the broker and by nothing else.

        A server on the port whose leasing process has gone is an orphan: one serving what
        was asked for is adopted, one serving something else is stopped. With ``how.roam``
        a busy port is served beside, on a free port, when the memory allows. With
        ``how.escalate`` a server with fewer slots than asked is grown rather than
        refused. ``timeout=None`` scales with the weights on disk. `Measuring` refuses a
        start while another process holds the bench's measuring lock, unless ``how.anyway``.
        ``say`` (else ``self.say``, else the log) is told of each decision.
        """
        how = how or Starting()
        roam, escalate, anyway = how.roam, how.escalate, how.anyway
        told = say or self.say or logger.info
        spec = self._permitted(spec, escalate)
        iq = self._quant_guard(spec, how)
        spec, drafting = self._with_mtp(spec, escalate)
        if drafting.note:
            (told if drafting.worth_saying else logger.info)(f"port {spec.port}: {drafting.note}")
        resolved_timeout = (
            timeout if timeout is not None else scaled_timeout(weight_of(spec.model)))

        self._sweep_orphans(spec.port, told)
        with self._port_lock(spec.port):
            entry = self._load().get(str(spec.port))
            stray = entry if isinstance(entry, dict) and orphaned(entry) else None
            try:
                adopted = self.adopt(spec)
            except ServerFailed as why:
                if escalate:
                    running = self._slots_shortfall(spec)
                    if running is not None:
                        more = (max(1, int(spec.parallel or 1))
                                - max(1, int(running.parallel or 1)))
                        return self._escalate(
                            running, Growth(add_slots=more, timeout=resolved_timeout,
                                            anyway=anyway),
                            Caller(on_event=on_event, say=told))
                if stray is None:
                    if roam and (reused := self._reusable(spec, on_event=on_event)):
                        return reused
                    if not roam or not port_is_free(spec.port):
                        elsewhere = (self._beside(spec, timeout=resolved_timeout,
                                                  on_event=on_event, anyway=anyway,
                                                  **how.checks())
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
                if (stray is None and not port_is_free(spec.port)
                        and is_healthy(f"http://{DEFAULT_HOST}:{spec.port}", timeout=1.0)):
                    taken = self._unmanaged_on(spec, told)
                    if taken is not None:
                        emit(on_event, "ready", port=spec.port, adopted=True)
                        return taken
                    if roam and (reused := self._reusable(spec, on_event=on_event)):
                        return reused
                    elsewhere = (self._beside(spec, timeout=resolved_timeout,
                                              on_event=on_event, anyway=anyway, **how.checks())
                                 if roam else None)
                    if elsewhere is not None:
                        return elsewhere
                    raise ServerFailed(
                        f"port {spec.port} is served by a server ml-stack did not start. "
                        f"It is left alone; lease on a different port, or set "
                        f"{unmanaged.ENV}=auto to adopt servers that pass its checks.")

            if roam and (reused := self._reusable(spec, on_event=on_event)):
                return reused
            try:
                info = self._launch(spec, timeout=resolved_timeout, on_event=on_event,
                                    anyway=anyway, reuse=roam, **how.checks())
            except (Measuring, admission.AdmissionRefused, guarded.SentinelRefused):
                self._forget(spec.port)
                raise
            except ServerFailed:
                self._forget(spec.port)
                self._unavailable_until[spec.port] = time.monotonic() + UNAVAILABLE_COOLDOWN_S
                raise

            self._unavailable_until.pop(spec.port, None)
            if not info.mtp_note:
                info = replace(info, mtp_note=drafting.note)
            if not info.adopted:
                self._record(spec, info, iq=iq)
            return info

    def _quant_guard(self, spec: ServerSpec, how: Starting) -> quant_guard.IqQuant | None:
        """The IQ quantisation ``spec`` is when it was warned about; `BlockedQuant` in strict
        mode. llama.cpp specs only. The mode is the lease's own, else this process's; a
        lease from the broker wire carries none."""
        if self.backend_for(spec) is not self.backend:
            return None
        return quant_guard.enforce(spec.model, gpu_layers=spec.n_gpu_layers,
                                   asked=how.iq,
                                   who=how.who)

    def _permitted(self, spec: ServerSpec, escalate: bool) -> ServerSpec:
        """``spec`` as it will be started, or `ServerFailed` when the port was just given up
        on or a limit on this machine refuses the lease."""
        spec = reusing_installed(spec)
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

        if why := guarded.blocked(spec.model):
            raise guarded.SentinelRefused(why)
        now = time.monotonic()
        until = self._unavailable_until.get(spec.port, 0.0)
        if now < until:
            raise ServerFailed(
                f"port {spec.port} was marked unavailable {until - now:.1f}s ago; "
                "not retrying yet (negative cache)"
            )

        refused = self._over_limit(spec)
        if refused:
            raise ServerFailed(refused)
        return spec

    def _with_mtp(self, spec: ServerSpec, escalate: bool) -> tuple[ServerSpec, mtp.Plan]:
        """``spec`` with the MTP head it is served with by default, and the plan that chose it.

        A head the default picked is verified against sentinel's pin like the weights; one
        that fails is left out and the lease goes on without it.
        """
        try:
            binary = self.backend_for(spec).binary  # type: ignore[attr-defined]
        except (BinaryNotFound, OSError, AttributeError):
            binary = None
        chosen = mtp.plan(spec, binary=binary, escalate=escalate)
        if chosen.draft:
            try:
                guarded.verify(chosen.draft, state_file=self.state_file, stop=self.reclaim)
            except guarded.SentinelRefused as refused:
                chosen = mtp.Plan(note=f"MTP off: {refused}", loud=True)
        return mtp.applied(spec, chosen), chosen

    def _launch(self, spec: ServerSpec, *, timeout: float, on_event: Event | None = None,
                anyway: bool = False, reuse: bool = False, **starting: Any) -> ServerInfo:
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
        guarded.verify(spec.model, state_file=self.state_file, stop=self.reclaim)
        if spec.draft and spec.mtp is not True:
            guarded.verify(spec.draft, state_file=self.state_file, stop=self.reclaim)
        emit(on_event, "loading", port=spec.port, model=Path(str(spec.model)).name,
              slots=max(1, int(spec.parallel or 1)))
        admitted = self._admitted(spec, on_event=on_event, reuse=reuse, load_s=timeout)
        if isinstance(admitted, ServerInfo):
            return admitted
        try:
            info = self.backend_for(spec).start(spec, lease=admitted, timeout=timeout, **starting)
        except (ServerFailed, UnknownFlag) as why:
            if spec.mtp is not True:
                raise
            mtp.failed(spec.model, spec.draft, getattr(self.backend_for(spec), "binary", None))
            spec = replace(spec, draft=None, spec_type="", mtp=False)
            note = f"MTP off: the server would not start with it ({str(why).splitlines()[0]})"
            note += "; docs/serving.md, 'Multi-token prediction', has the one-command check"
            (self.say or logger.warning)(f"port {spec.port}: {note}; starting without")
            info = replace(self.backend_for(spec).start(spec, lease=admitted, timeout=timeout,
                                                        **starting), mtp_note=note)
        else:
            if spec.mtp is True:
                info = replace(info, mtp=Path(str(spec.draft)).name if spec.draft else "embedded",
                               mtp_note=f"MTP on: {spec.spec_type}")
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

    def _sweep_orphans(self, keep: int, say: Callable[[str], None]) -> None:
        """Once per manager, stop every server on this machine whose leasing process has gone,
        except the one on ``keep``, which the lease about to run adopts or replaces. Only a
        record that proves its pid still belongs to the server (start time and command
        line) is acted on."""
        with self._lock:
            if self._swept:
                return
            self._swept = True
        for port, entry in recorded_servers(self.state_file).items():
            if port != keep and orphaned(entry, strict=True):
                with self._port_lock(port):
                    self._stop_orphan(port, entry, say=say, why="swept before a new start")

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
                                    anyway=anyway, reuse=True, **starting)
            except (Measuring, admission.AdmissionRefused, guarded.SentinelRefused):
                self._forget(moved.port)
                raise
            except ServerFailed:
                self._forget(moved.port)
                return None
            if not info.adopted:
                self._record(moved, info)
            return info

    def adopt(self, spec: ServerSpec) -> ServerInfo | None:
        """The running server for ``spec`` on its port, if the lease record holds one there.
        ``None`` for a port nothing answers on, and for one only an unmanaged server
        answers on."""
        base_url = f"http://{DEFAULT_HOST}:{spec.port}"
        if str(spec.port) not in self._load() or not is_healthy(base_url, timeout=1.0):
            return None

        mismatch = serving_mismatch(spec, reported_models(base_url), serving_params(base_url))
        entry = self._load().get(str(spec.port), {})
        if not admission.compatible(spec, entry, mismatch):
            mismatch = mismatch or ["recorded cache, draft, or template settings differ"]
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
        """:meth:`_escalate`, asked of the broker."""
        return self.broker.escalate(spec, Growth(add_slots, room, timeout, anyway),
                                    Caller(on_event=on_event, say=say or self.say))

    def _escalate(self, spec: ServerSpec, growth: Growth, caller: Caller) -> ServerInfo:
        """Grow the server on ``spec.port`` by ``growth.add_slots`` conversations, keeping
        every one already live.

        ``spec`` is the settings the port is serving now. Each live slot is saved through
        ``/slots/{id}?action=save`` first. The cache grows when ``fit`` says the room is
        there, else the existing total is split across more slots, and a conversation too
        long for its share is summarised and re-seeded. Raises `EscalationRefused` when a
        live conversation would be dropped and summarising it did not rescue that.
        """
        add_slots, room, timeout, anyway = (growth.add_slots, growth.room, growth.timeout,
                                            growth.anyway)
        on_event, say = caller.on_event, caller.say
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
            exit_guard.release(pid)
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
        """Let go of a lease; the broker stops the server when nobody else holds it. A
        record with no lease is stopped directly."""
        if info.lease:
            self._leases.pop(info.lease, None)
            self.broker.drop(info, grace_s=grace_s)
            return
        self._stop_server(info, grace_s=grace_s)

    def _stop_server(self, info: ServerInfo, *, grace_s: float = 5.0) -> None:
        """Stop a server this process started. Adopted servers are left running."""
        if info.adopted:
            logger.debug("not stopping %s: we adopted it", info.base_url)
            return
        held = self._processes.pop(info.port, None)
        if held is None:
            held = info.process
        if info.pid:
            kill_process_tree(info.pid, grace_s=grace_s)
        exit_guard.release(info.pid)
        reap_one(held, grace_s=grace_s)
        self._forget(info.port)

    def detach(self, info: ServerInfo) -> None:
        """Record the server under its own pid, held by nobody, until it is taken down."""
        self.broker.detach(info)

    def _detach(self, info: ServerInfo) -> None:
        """Record the server under its own pid and stop tracking it in this process."""
        self._processes.pop(info.port, None)
        exit_guard.release(info.pid)
        entry = self._mine.pop(str(info.port), None)
        if entry is None or not info.pid:
            self._save()
            return
        with self._exclusive():
            state = merge_state(self._load(), self._mine, os.getpid())
            state[str(info.port)] = {**entry, "owner_pid": info.pid}
            self._write(state)

    def stop_all(self, *, grace_s: float = 5.0) -> list[int]:
        """Release every lease this process holds and stop every server it started."""
        stopped: list[int] = []
        for info in list(self._leases.values()):
            self.release(info, grace_s=grace_s)
            if info.pid:
                stopped.append(info.pid)
        for entry in list(self._mine.values()):
            pid = entry.get("pid")
            if isinstance(pid, int) and pid_exists(pid):
                stopped += kill_process_tree(pid, grace_s=grace_s)
            exit_guard.release(pid)
        for held in list(self._processes.values()):
            reap_one(held, grace_s=grace_s)
        self._processes.clear()
        self._mine.clear()
        self._save()
        return stopped

    def close(self) -> None:
        """Stop every server this manager started. Servers it adopted are left running."""
        self.stop_all()

    def __enter__(self) -> ServerManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ state file

    def _pending(self, spec: ServerSpec, *, est_bytes: int = 0) -> Lease:
        """Write the server down before it exists, and hand the backend the proof.

        The record carries the port, the model and this process as owner with no pid yet;
        `_record` fills the pid in once the server answers, `_forget` takes the entry out
        if it never does. Between the two, a port that looks unrecorded is one this
        manager is starting on -- never one it may kill.
        """
        self._mine[str(spec.port)] = {
            "port": spec.port, "pid": None, "backend": self.backend_for(spec).name,
            "model": str(spec.model), "owner_pid": os.getpid(), "pending": True,
            "pool": admission.pool_of(spec), "est_bytes": est_bytes,
            "embedding": bool(spec.embedding), "mmproj": bool(spec.mmproj),
            "context": int(spec.context), "parallel": int(spec.parallel or 1),
        }
        self._save()
        return Lease(port=spec.port, owner_pid=os.getpid(), state_file=str(self.state_file),
                     stop_on_exit=self.stop_on_exit)

    def _record(self, spec: ServerSpec, info: ServerInfo,
                iq: quant_guard.IqQuant | None = None) -> None:
        self._mine[str(spec.port)] = {
            "port": info.port,
            "pid": info.pid,
            "backend": info.backend,
            "model": str(spec.model),
            "owner_pid": os.getpid(),
            "pool": admission.pool_of(spec),
            "est_bytes": (self._mine.get(str(spec.port)) or {}).get("est_bytes", 0),
            "embedding": bool(spec.embedding),
            "mmproj": bool(spec.mmproj),
            "mtp": info.mtp,
            "cache_type_k": spec.cache_type_k,
            "cache_type_v": spec.cache_type_v,
            "draft": str(spec.draft or ""),
            "spec_type": spec.spec_type,
            "chat_template_file": str(spec.chat_template_file or ""),
            "mtp_note": info.mtp_note,
            "context": int(spec.context),
            "parallel": int(spec.parallel or 1),
            "base_url": info.base_url,
            "load_s": info.load_s,
            "warmup_s": info.warmup_s,
            "started": started_at(info.pid),
            "cmdline": cmdline_digest(info.pid),
            **({"log": str(info.log_path)} if info.log_path else {}),
            **({"iq_warning": iq.name} if iq is not None else {}),
        }
        forget_server(info.base_url)
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
        limits = limits_read()
        if not (limits.servers or limits.slots):
            return ""
        running = sum(1 for port, entry in recorded_servers(self.state_file).items()
                      if port != spec.port and pid_exists(int(entry.get("pid") or 0)))
        return limits.refusal(running=running, slots=max(1, int(spec.parallel or 1)))

    def reclaim(self, port: int) -> bool:
        """Free ``port`` if one of our servers is holding it."""
        recorded = self._recorded_pid(port)
        return reclaim_port(port, recorded_pids=[recorded] if recorded else None)


_DEFAULT: ServerManager | None = None
_DEFAULT_LOCK = threading.Lock()


def default_manager() -> ServerManager:
    """The manager `serve` uses when it is not given one, built on first use."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = ServerManager()
        return _DEFAULT


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
    iq: str = "",
    on_event: Event | None = None,
    say: Callable[[str], None] | None = None,
    reason: str = "",
    **spec_kwargs: object,
) -> Iterator[ServerInfo]:
    """Run a server for the duration of the block, yielding its ``ServerInfo``.

    ``roam``, ``escalate``, ``anyway``, ``iq``, ``on_event``, ``say`` and ``reason`` go to
    :meth:`ServerManager.lease`.
    """
    manager = manager or default_manager()
    spec = ServerSpec(
        model=model,
        port=port if port is not None else free_port(),
        context=context,
        **spec_kwargs,  # type: ignore[arg-type]
    )
    info = manager.lease(spec, timeout=timeout, roam=roam, escalate=escalate,
                         anyway=anyway, iq=iq, on_event=on_event, say=say, reason=reason)
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

    kept = {}
    for key, entry in state.items() if isinstance(state, dict) else []:
        if not isinstance(entry, dict):
            continue
        if entry.get("unmanaged"):
            kept[key] = entry
            continue
        pid = entry.get("pid")
        if isinstance(pid, int) and same_process(entry):
            stopped += kill_process_tree(pid)

    if kept:
        write_json(held, kept)
    else:
        held.unlink(missing_ok=True)
    return stopped
