"""Leasing a server: start one, or adopt the one already running."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from ml_stack.client import is_healthy, reported_models
from ml_stack.client.health import ServingParams, serving_params
from ml_stack.client.http import ServerError, request_json
from ml_stack.serve.backend import (
    DEFAULT_SLOT_SAVE_PATH,
    Lease,
    LlamaServerBackend,
    ServerBackend,
    ServerFailed,
    ServerInfo,
    ServerSpec,
)
from ml_stack.serve.binary import CACHE_ROOT
from ml_stack.serve.ports import free_port, port_is_free
from ml_stack.serve.process import kill_process_tree, pid_exists
from ml_stack.serve.ports import DEFAULT_HOST, reclaim_port

logger = logging.getLogger(__name__)

Event = Callable[[dict[str, Any]], None]
"""``on_event({"event": name, ...fields})`` -- a step a caller waiting on a lease or an
escalation can show as it happens, not after. A handler's own errors are swallowed: a
broken display must not fail the lease it is only reporting on."""

# What a slot's cache is asked to become when a live conversation cannot be summarised.
SUMMARY_PROMPT = (
    "Summarise this conversation so far in under 200 words. Keep every fact, decision "
    "and open question a continuation of it would need.")

# Appended to a slot's own cached prompt text for a raw /completion continuation, so the
# shared prefix is a cache hit and only this tail and the generation are new work.
SUMMARY_SUFFIX = (
    "\n\nSummarise the conversation above in under 200 words. Keep every fact, decision "
    "and open question a continuation of it would need.\n\nSummary:")


def _emit(on_event: Event | None, event: str, **fields: Any) -> None:
    if on_event is None:
        return
    try:
        on_event({"event": event, **fields})
    except Exception:  # noqa: BLE001 - a caller's display is not this lease's problem
        pass


class EscalationRefused(ServerFailed):
    """Growing or splitting a server's seats would drop a live conversation's cache, and
    summarising it did not rescue that. The saved cache named in the message is kept."""

STATE_FILE = CACHE_ROOT / "servers.json"
UNAVAILABLE_COOLDOWN_S = 3.0

# How much of what is free a second model may take before it is judged not to fit. Below 1.0
# because a model needs its weights *and* room to work in, and a machine that fills itself
# exactly swaps instead of serving.
BESIDE_HEADROOM = 0.8

# The flat timeout this used to be, kept as the floor: a small model that always loaded in
# ten seconds must not suddenly wait less than 300 just because it is small.
DEFAULT_TIMEOUT_S = 300.0
_GB = 1024 ** 3


def scaled_timeout(weights_bytes: int, *, base: float = DEFAULT_TIMEOUT_S) -> float:
    """A load timeout that grows with the weights, so an 87G model is not raced against a
    timeout sized for a 4G one. 60s plus 1.5s per GB of weights, or ``base`` -- whichever is
    larger. ``weights_bytes`` is 0 for an `hf:` reference not yet on disk, and 0 leaves the
    floor untouched: an unknown size is not the same as an enormous one."""
    return max(base, 60.0 + 1.5 * (weights_bytes / _GB))


def free_memory() -> int | None:
    """Bytes this machine could still give a model, or None when it will not say."""
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().available)


def weight_of(model: str | Path) -> int:
    """Roughly what a model will take, from the weights on disk. 0 when they are not here.

    A `hf:` reference has not been downloaded yet the first time, so its size is unknown and
    unknown is not the same as enormous — it is left to the load to find out.
    """
    if isinstance(model, str) and model.startswith("hf:"):
        return 0
    where = Path(model)
    if not where.exists():
        return 0
    # a sharded model names its first file; the others sit beside it
    shards = sorted(where.parent.glob(where.name.replace("00001", "*"))) or [where]
    return sum(s.stat().st_size for s in shards if s.is_file())


def merge_state(on_disk: dict, mine: dict, owner_pid: int) -> dict:
    """This process's servers, merged over every other record whose owner or server is alive."""
    merged = {
        key: entry
        for key, entry in on_disk.items()
        if isinstance(entry, dict)
        and entry.get("owner_pid") != owner_pid
        and (pid_exists(entry.get("owner_pid")) or pid_exists(entry.get("pid")))
    }
    merged.update(mine)
    return merged


def orphaned(entry: dict) -> bool:
    """Whether a record's server is running on after the process that leased it has gone."""
    owner, pid = entry.get("owner_pid"), entry.get("pid")
    return (isinstance(owner, int) and isinstance(pid, int) and owner != pid
            and not pid_exists(owner) and pid_exists(pid))


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


def already_up(model: str, port: int, *, state_file: Path | None = None) -> dict | None:
    """The recorded server on ``port`` if it serves ``model`` (by file name) and its
    process is alive -- whatever its shape. A conversation that would lease one seat on a
    port already holding the same weights in another shape uses what is up rather than
    reloading them: the reload is the cost, the shape is not (Adam, 2026-09-03)."""
    entry = recorded_servers(state_file).get(int(port))
    if not entry:
        return None
    if Path(str(entry.get("model") or "")).name != Path(str(model)).name:
        return None
    return entry if pid_exists(int(entry.get("pid") or 0)) else None


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
        self.say: Callable[[str], None] | None = None
        self._mine: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._port_locks: dict[int, threading.Lock] = {}
        self._unavailable_until: dict[int, float] = {}

    # ------------------------------------------------------------------ leasing

    def lease(self, spec: ServerSpec, *, timeout: float | None = None,
              roam: bool = True, check_flags: bool = True, preflight: bool = True,
              warmup_request: bool = True, escalate: bool = False,
              on_event: Event | None = None,
              say: Callable[[str], None] | None = None) -> ServerInfo:
        """A healthy server for ``spec``. Starts one only if there is not one already.

        A server on the port whose record names a leasing process that has gone is an
        orphan: one serving the shape asked for is adopted and its record made this
        process's; one serving another shape is stopped before a server is started.
        Either way ``say`` (else ``self.say``, else the log) is told.

        When the port is busy with something else and this machine has the memory to hold
        both, it is served beside it on a free port rather than refused: a small model does
        not need the large one evicted, and making a person pick another port by hand is
        work a machine can do. ``roam=False`` for a caller that truly needs *that* port —
        the one that expects every consumer to meet on it.

        ``escalate=True`` is for a caller that may genuinely need more than one concurrent
        cache: when the only reason a running server does not match ``spec`` is that it
        holds fewer seats than asked, it is grown (or, if that will not fit, split, or
        summarised and split -- see :meth:`escalate`) rather than refused. A spec with no
        ``slot_save_path`` is given this manager's own default so a later escalation has
        somewhere to save a live conversation before the relaunch.

        ``timeout=None`` (the default) scales with the weights on disk -- see
        ``scaled_timeout`` -- so a caller that never thought about it still gets a timeout
        sized for what it is actually waiting on. A caller that passes a number means it,
        and gets exactly that instead.
        """
        if escalate:
            # llama.cpp's slot-save file carries the cache's stream count, and a restore
            # raises "n_stream mismatch" the moment that count differs from the file's --
            # which a change in seat count always does unless every stream is one shared
            # buffer throughout, seat count or no. A lease that may later escalate is
            # kv_unified from its first launch, not only from the relaunch.
            if not spec.slot_save_path:
                spec = replace(spec, slot_save_path=str(DEFAULT_SLOT_SAVE_PATH))
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
                            running, add_seats=max(1, int(spec.parallel or 1))
                            - max(1, int(running.parallel or 1)),
                            timeout=resolved_timeout, on_event=on_event, say=told)
                if stray is None:
                    if not roam or not port_is_free(spec.port):
                        elsewhere = (self._beside(spec, timeout=resolved_timeout,
                                                  on_event=on_event, **starting)
                                    if roam else None)
                        if elsewhere is not None:
                            return elsewhere
                    raise
                self._stop_orphan(spec.port, stray, say=told, why=str(why))
            else:
                if adopted is not None:
                    if stray is not None:
                        self._take_over(spec.port, stray, say=told)
                    _emit(on_event, "ready", port=spec.port, adopted=True)
                    return adopted

            try:
                info = self._launch(spec, timeout=resolved_timeout, on_event=on_event,
                                    **starting)
            except ServerFailed:
                self._forget(spec.port)
                self._unavailable_until[spec.port] = time.monotonic() + UNAVAILABLE_COOLDOWN_S
                raise

            self._unavailable_until.pop(spec.port, None)
            self._record(spec, info)
            return info

    def _launch(self, spec: ServerSpec, *, timeout: float, on_event: Event | None = None,
                **starting: Any) -> ServerInfo:
        """Start ``spec``, telling ``on_event`` when the load begins and ends.

        The one place a fresh process is asked for, so ``up``, an adopt that falls
        through to a real start, and :meth:`escalate`'s relaunch all say the same thing
        the same way.
        """
        _emit(on_event, "loading", port=spec.port, model=Path(str(spec.model)).name,
              seats=max(1, int(spec.parallel or 1)))
        info = self.backend.start(spec, lease=self._pending(spec), timeout=timeout,
                                  **starting)
        _emit(on_event, "ready", port=spec.port, load_s=info.load_s, warmup_s=info.warmup_s)
        return info

    def _slots_shortfall(self, spec: ServerSpec) -> ServerSpec | None:
        """The shape actually running on ``spec.port``, if the only way it disagrees with
        ``spec`` is holding fewer seats than asked. ``None`` for any other disagreement,
        or for nothing answering at all -- an escalation is a repair for one specific
        mismatch, not a second way to adopt."""
        base_url = f"http://{DEFAULT_HOST}:{spec.port}"
        if not is_healthy(base_url, timeout=1.0):
            return None
        models = reported_models(base_url)
        if models and not any(model_matches(m, spec.model) for m in models):
            return None
        params = serving_params(base_url)
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
                **starting: Any) -> ServerInfo | None:
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
                info = self._launch(moved, timeout=timeout, on_event=on_event, **starting)
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

    def escalate(self, spec: ServerSpec, *, add_seats: int = 1, room: int | None = None,
                timeout: float | None = None, on_event: Event | None = None,
                say: Callable[[str], None] | None = None) -> ServerInfo:
        """Grow the server on ``spec.port`` by ``add_seats`` more concurrent conversations,
        keeping every one already live.

        ``spec`` is the shape actually running -- its ``context`` and ``parallel`` are
        what the port is serving now, not what a caller wishes it were (:meth:`lease`'s
        ``escalate=True`` works this out with :meth:`_slots_shortfall` before calling
        here). Every slot with a live conversation is saved through
        ``/slots/{id}?action=save`` before anything stops.

        Grows the whole cache -- every seat keeping its size -- when ``fit`` says the
        extra room is there. Otherwise splits the existing total across the larger seat
        count, when every live conversation is short enough for the smaller per-seat
        context that leaves each with. A conversation that is not short enough is
        summarised on the model itself, on the slot that already holds it, and the
        summary is re-seeded in its place after the relaunch rather than the saved cache
        -- which is kept regardless, named in every message about that slot. Raises
        :class:`EscalationRefused` only when a live conversation would be dropped and
        summarising it did not rescue that.
        """
        from ml_stack.client.chat import Client
        from ml_stack.hub import _human
        from ml_stack.hub import room as machine_room

        told = say or self.say or logger.info
        current_slots = max(1, int(spec.parallel or 1))
        new_slots = current_slots + max(1, int(add_seats))
        per_seat = max(1, int(spec.context) // current_slots)
        base_url = f"http://{DEFAULT_HOST}:{spec.port}"

        if not is_healthy(base_url, timeout=2.0):
            raise ServerFailed(f"nothing is answering on port {spec.port} to escalate")
        if not spec.slot_save_path:
            raise ServerFailed(
                f"port {spec.port} was not started with --slot-save-path; its live "
                "conversations cannot be saved before a relaunch"
            )

        try:
            slots = request_json(f"{base_url}/slots", method="GET", timeout=timeout or 30.0)
        except ServerError as exc:
            raise ServerFailed(
                f"port {spec.port} has no /slots endpoint to escalate from: {exc}") from exc
        if not isinstance(slots, list):
            raise ServerFailed(
                f"port {spec.port}: /slots answered with {type(slots).__name__}, not a list")

        live = sorted((int(s["id"]), int(s.get("n_prompt_tokens") or 0)) for s in slots
                     if isinstance(s, dict) and int(s.get("n_prompt_tokens") or 0) > 0)
        # Only present with LLAMA_SERVER_SLOTS_DEBUG=1 (backend.py sets it whenever
        # slot_save_path is), which is what a summary is asked to read rather than the
        # bare instruction a slot's cache cannot answer on its own.
        prompts = {int(s["id"]): str(s.get("prompt") or "")
                  for s in slots if isinstance(s, dict) and s.get("id") is not None}

        room_bytes = int(room) if room is not None else machine_room()
        fit = self._fit_for(spec.model, room=room_bytes)

        mode = ""
        new_context = per_seat * new_slots
        reason = ""
        if fit is not None:
            loaded, each = fit.line(per_seat)
            need = loaded + new_slots * each
            if need <= room_bytes:
                mode = "grow"
                reason = (f"{new_slots} seats of {per_seat:,} tokens need {_human(need)}, "
                          f"which fits in {_human(room_bytes)} of room")

        too_long: list[tuple[int, int]] = []
        if mode != "grow":
            new_context = int(spec.context)
            split_per_seat = max(1, new_context // new_slots)
            too_long = [(sid, tok) for sid, tok in live if tok > split_per_seat]
            if too_long:
                mode = "summarize"
                reason = (f"{len(too_long)} live conversation(s) do not fit the "
                          f"{split_per_seat:,}-token seat a split leaves them and will be "
                          "summarised")
            else:
                mode = "split"
                reason = (f"the existing {new_context:,}-token cache split across "
                          f"{new_slots} seats is {split_per_seat:,} each")

        _emit(on_event, "escalating", port=spec.port, mode=mode, reason=reason,
              from_seats=current_slots, to_seats=new_slots)
        told(f"port {spec.port}: escalating from {current_slots} to {new_slots} seat(s) "
            f"by {mode} -- {reason}")

        stamp = time.strftime("%Y%m%dT%H%M%S")
        saved: dict[int, str] = {}
        for sid, tok in live:
            filename = f"escalate-{spec.port}-{sid}-{stamp}.bin"
            _emit(on_event, "saving", port=spec.port, slot=sid, tokens=tok,
                  filename=filename)
            try:
                request_json(f"{base_url}/slots/{sid}?action=save",
                            payload={"filename": filename}, timeout=timeout or 120.0)
            except ServerError as exc:
                raise ServerFailed(
                    f"could not save slot {sid} on port {spec.port}: {exc}") from exc
            saved[sid] = filename

        summaries: dict[int, str] = {}
        for sid, tok in too_long:
            _emit(on_event, "summarizing", port=spec.port, slot=sid, tokens=tok)
            prior = prompts.get(sid, "")
            try:
                if prior:
                    # A raw continuation of the slot's own cached prompt: the shared
                    # prefix is a cache hit, so this costs the generation and nothing
                    # about the reprocessing the coordinator's cheap-summary case rests on.
                    summary = Client(base_url, slot=sid, timeout=timeout or 120.0).complete(
                        prior + SUMMARY_SUFFIX, n_predict=512)
                else:
                    # No prompt text to read (LLAMA_SERVER_SLOTS_DEBUG was not on for
                    # this server) -- the model is asked cold and told nothing.
                    reply = Client(base_url, slot=sid, timeout=timeout or 120.0).chat(
                        [{"role": "user", "content": SUMMARY_PROMPT}], n_predict=512)
                    summary = (getattr(reply, "content", "") or "").strip()
            except Exception as exc:  # noqa: BLE001 - summarising failed; refuse, do not lose it
                raise EscalationRefused(
                    f"slot {sid} on port {spec.port} holds {tok:,} tokens, too long for "
                    f"the seat a split leaves it, and summarising it failed: {exc}. Its "
                    f"cache is kept at {saved[sid]}."
                ) from exc
            if not summary:
                raise EscalationRefused(
                    f"slot {sid} on port {spec.port} holds {tok:,} tokens, too long for "
                    f"the seat a split leaves it, and summarising it returned nothing. "
                    f"Its cache is kept at {saved[sid]}."
                )
            summaries[sid] = summary
            # The text itself, not just its length: llama-server's chat API is stateless
            # per request, so a slot's cache being re-seeded with this summary carries no
            # memory a later /v1/chat/completions call will see on its own -- the caller
            # holding the conversation has to fold the summary into its own transcript to
            # actually continue from it, and can only do that if this event carries it.
            _emit(on_event, "summarized", port=spec.port, slot=sid, summary=summary,
                  tokens=len(summary.split()))

        pid = self._recorded_pid(spec.port)
        _emit(on_event, "stopping", port=spec.port, pid=pid)
        if pid:
            kill_process_tree(pid)
        self._forget(spec.port)

        # kv_unified keeps the cache's stream count at 1 across the relaunch; any other
        # value makes a save from the old seat count unrestorable into the new one, "n_stream
        # mismatch" thrown by llama.cpp's own state reader regardless of which slot.
        new_spec = replace(spec, parallel=new_slots, context=new_context, kv_unified=True)
        resolved_timeout = (
            timeout if timeout is not None else scaled_timeout(weight_of(spec.model)))
        info = self._launch(new_spec, timeout=resolved_timeout, on_event=on_event)
        new_base = info.base_url
        # recorded now, not after every restore: a restore failure below must not leave a
        # live, healthy process this manager has forgotten it started
        self._record(new_spec, info)

        for sid, tok in live:
            if sid in summaries:
                _emit(on_event, "restoring", port=spec.port, slot=sid, mode="summary")
                try:
                    Client(new_base, slot=sid, timeout=timeout or 120.0).complete(
                        summaries[sid], n_predict=1)
                except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
                    raise ServerFailed(
                        f"could not re-seed the summary for slot {sid} on port "
                        f"{spec.port}: {exc}. Its full cache is kept at {saved[sid]}."
                    ) from exc
            else:
                _emit(on_event, "restoring", port=spec.port, slot=sid, mode="cache")
                try:
                    request_json(f"{new_base}/slots/{sid}?action=restore",
                                payload={"filename": saved[sid]}, timeout=timeout or 120.0)
                except ServerError as exc:
                    raise ServerFailed(
                        f"could not restore slot {sid} on port {spec.port} from "
                        f"{saved[sid]}: {exc}"
                    ) from exc

        _emit(on_event, "done", port=spec.port, seats=new_slots, mode=mode)
        told(f"port {spec.port}: now serving {new_slots} seat(s)")
        return info

    @staticmethod
    def _fit_for(model: str | Path, *, room: int) -> Any:
        """The measured :class:`~ml_stack.serve.fit.Fit` for ``model`` at ``room``, or
        ``None`` when nothing has measured it -- an escalation with no fit record cannot
        claim growing fits, and falls to splitting instead."""
        from ml_stack.serve.fit import records

        for one in records(room=room):
            if model_matches(one.model, model):
                return one
        return None

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

    def _pending(self, spec: ServerSpec) -> Lease:
        """Write the server down before it exists, and hand the backend the proof.

        The record carries the port, the model and this process as owner with no pid yet;
        `_record` fills the pid in once the server answers, `_forget` takes the entry out
        if it never does. Between the two, a port that looks unrecorded is one this
        manager is starting on -- never one it may kill.
        """
        self._mine[str(spec.port)] = {
            "port": spec.port, "pid": None, "backend": self.backend.name,
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
    timeout: float | None = None,
    manager: ServerManager | None = None,
    roam: bool = True,
    escalate: bool = False,
    on_event: Event | None = None,
    say: Callable[[str], None] | None = None,
    **spec_kwargs: object,
) -> Iterator[ServerInfo]:
    """Run a server for the duration of the block, yielding its ``ServerInfo``.

    ``roam``, ``escalate``, ``on_event`` and ``say`` go to :meth:`ServerManager.lease`.
    """
    from ml_stack.serve.ports import free_port

    manager = manager or _DEFAULT
    spec = ServerSpec(
        model=model,
        port=port if port is not None else free_port(),
        context=context,
        **spec_kwargs,  # type: ignore[arg-type]
    )
    info = manager.lease(spec, timeout=timeout, roam=roam, escalate=escalate,
                         on_event=on_event, say=say)
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
