"""What ``ml-stack-serve`` does, as functions that take values and return them.

One per subcommand -- `status`, `processes`, `up`, `preflight`, `shapes`, `fit_markdown`,
`tensors`, `measure`, `memory`, `limits`, `reclaim`, `down`, `orphans`, `escalate` --
with the helpers they share. `ml_stack.serve.cli` parses and prints; nothing here does
either. `Refused` carries the lines a command says when it will not act.
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.client import is_healthy, reported_models
from ml_stack.client.health import serving_params
from ml_stack.fleet.serving import Serving
from ml_stack.log import warn
from ml_stack.serve import fit as fit_mod
from ml_stack.serve import limits as limits_mod
from ml_stack.serve import preflight as preflight_mod
from ml_stack.serve import profile as profile_mod
from ml_stack.serve import reclaim as reclaim_mod
from ml_stack.serve.backend import (
    LlamaServerBackend,
    ServerFailed,
    ServerInfo,
    ServerSpec,
    default_slot_save_path,
)
from ml_stack.serve.binary import BinaryNotFound
from ml_stack.serve.manager import (
    ServerManager,
    lease_file,
    orphaned,
    recorded_servers,
)
from ml_stack.serve.ports import DEFAULT_HOST, server_pids_on_port
from ml_stack.serve.process import every_server, machine_memory, pid_exists
from ml_stack.units import human_bytes

__all__ = ["DEFAULT_ROOT", "FIT_HEAD", "PLIST", "PROBE_TIMEOUT", "Limits", "Machine",
           "Processes", "Recorded", "Refused", "Resolved", "Snapshot", "Started",
           "Status", "Stopped", "alongside", "announce", "base_url_for", "beacon",
           "cache_of", "down", "drafted", "escalate", "fit_markdown", "fit_page", "judge",
           "limits", "look", "manager_for", "measure", "memory", "orphans", "preflight",
           "processes", "reclaim", "reclaim_once", "resolve_model", "resolve_spec", "shapes",
           "status", "tensors", "up", "withdraw", "write_plist"]

PROBE_TIMEOUT = 2.0
DEFAULT_ROOT = "~/.ml-stack/traind"
"""Where the daemon keeps the list of what this machine is serving; peers read it."""


class Refused(Exception):
    """A command that will not act, and the lines saying why; the first is the error."""

    def __init__(self, *lines: str) -> None:
        super().__init__(lines[0] if lines else "")
        self.lines = tuple(lines)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One server that is answering, and what a lease for it would do."""

    port: int
    base_url: str
    model: str | None = None
    quant: str | None = None
    context: int | None = None
    slots: int | None = None
    pid: int | None = None
    owner_pid: int | None = None
    holder_running: bool = False
    recorded: bool = False
    load_s: float | None = None
    warmup_s: float | None = None
    verdict: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Status:
    """What answered on the ports asked about, and what a lease for each would do."""

    ports: tuple[int, ...]
    servers: tuple[Snapshot, ...]
    foreign: tuple[dict[str, int], ...]


@dataclass(frozen=True, slots=True)
class Processes:
    """Every llama-server on this machine, and which of them a lease records."""

    found: tuple[dict[str, Any], ...]
    leased: frozenset[int]
    strays: tuple[dict[str, Any], ...]
    foreign: tuple[dict[str, int], ...]


@dataclass(frozen=True, slots=True)
class Resolved:
    """A spec with 'auto' answered, and a line per thing that was worked out."""

    spec: ServerSpec
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Started:
    """A server leased, the shape it is serving, and what the fleet was told."""

    info: ServerInfo
    spec: ServerSpec
    announced: str


@dataclass(frozen=True, slots=True)
class Stopped:
    """A server released: its port, the pid, and whether that pid was still running."""

    port: int
    base_url: str
    pid: int | None
    was_running: bool
    owner_pid: int | None = None


@dataclass(frozen=True, slots=True)
class Recorded:
    """One model measured at load, what it said, and the file it was written to."""

    model: str
    said: str
    where: Path


@dataclass(frozen=True, slots=True)
class Machine:
    """What a model may use here, what is installed, and what is held right now."""

    room: int
    total: int
    held: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class Limits:
    """What this machine allows, where that is written, and what a model may use."""

    lines: tuple[str, ...]
    where: Path
    machine: int
    room: int


def base_url_for(port: int) -> str:
    return f"http://{DEFAULT_HOST}:{port}"


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _float_or_none(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def look(port: int, records: dict[int, dict]) -> Snapshot | None:
    """What is serving on ``port``, or ``None`` when nothing answers there."""
    url = base_url_for(port)
    if not is_healthy(url, timeout=PROBE_TIMEOUT):
        return None

    params = serving_params(url)
    models = reported_models(url)
    entry = records.get(port) or {}

    reported = (params.model if params else None) or (models[0] if models else None)
    reported = reported or entry.get("model")
    owner = _int_or_none(entry.get("owner_pid"))

    return Snapshot(
        port=port,
        base_url=url,
        model=Path(str(reported)).name if reported else None,
        quant=params.quant if params else None,
        context=params.n_ctx if params else None,
        slots=params.total_slots if params else None,
        pid=_int_or_none(entry.get("pid")),
        owner_pid=owner,
        holder_running=pid_exists(owner),
        recorded=bool(entry),
        load_s=_float_or_none(entry.get("load_s")),
        warmup_s=_float_or_none(entry.get("warmup_s")),
    )


def judge(manager: ServerManager, snapshot: Snapshot, spec: ServerSpec) -> Snapshot:
    """``snapshot`` with the verdict a lease for ``spec`` would reach."""
    try:
        info = manager.adopt(spec)
    except ServerFailed as exc:
        return replace(snapshot, verdict="refuse", reason=str(exc))
    return replace(snapshot, verdict="adopt" if info is not None else "start")


def cache_of(model: str) -> tuple[Path, int] | None:
    """The model root ``model`` lies under, with its weight bytes; the file's own
    directory when it is under none; None when neither is there."""
    from ml_stack.fleet.models import holding
    from ml_stack.hub import default_roots

    path = home.expand(str(model or ""))
    if not str(model or "").strip():
        return None
    for root in default_roots(home.home()):
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if root.is_dir():
            return root, holding(root)[1]
    if path.parent.is_dir() and str(path.parent) not in ("", "."):
        return path.parent, holding(path.parent)[1]
    return None


def status(*, port: int, model: str = "", context: int = 0, parallel: int = 1) -> Status:
    """What answers on the recorded ports and ``port``, each judged for a lease of
    ``model`` at ``context`` and ``parallel`` -- or of whatever it already serves."""
    records = recorded_servers(lease_file())
    ports = sorted({*records, int(port)})
    manager = ServerManager(state_file=lease_file())

    found: list[Snapshot] = []
    foreign: list[dict[str, int]] = []
    for one in ports:
        snapshot = look(one, records)
        if snapshot is None:
            continue
        if not snapshot.recorded:
            # a health-answering server this machine never leased -- the backend never kills
            # one, so it is reported and left alone rather than judged
            pids = server_pids_on_port(one)
            if pids:
                foreign.extend({"port": one, "pid": pid} for pid in pids)
                continue
        wanted = model or snapshot.model
        if wanted:
            snapshot = judge(
                manager, snapshot,
                ServerSpec(model=wanted, port=one, context=context, parallel=parallel))
        found.append(snapshot)
    return Status(tuple(ports), tuple(found), tuple(foreign))


def processes() -> Processes:
    """Every llama-server on this machine, with the ones no lease records picked out."""
    records = recorded_servers(lease_file())
    found = every_server()
    strays = [o for o in found if o["port"] not in records and not o.get("defunct")]
    return Processes(tuple(found), frozenset(records), tuple(strays),
                     tuple({"port": o["port"], "pid": o["pid"]} for o in strays))


def beacon(root: str) -> Serving | None:
    """This machine's beacon, or None when no fleet was ever set up here.

    A machine with no daemon has no beacon to write to, and creating one would advertise a
    model to nobody through a file nothing reads.
    """
    where = home.expand(root)
    return Serving(where / "serving.json") if where.is_dir() else None


def announce(root: str, spec: ServerSpec) -> str:
    """Tell the fleet this machine is serving it. Returns a line to print, or ''."""
    try:
        known = beacon(root)
        if known is None:
            return ""
        known.register(spec.port, models=[str(spec.model)], slots=spec.parallel)
        return f"announced to the fleet on port {spec.port}"
    except Exception as exc:  # noqa: BLE001 - a server that works unannounced still works
        return f"could not announce it to the fleet: {exc}"


def withdraw(root: str, port: int) -> str:
    """Take ``port`` out of the fleet's beacon; returns a line to print, or ''."""
    # A registration outlives the server it describes, and every peer then pays a timeout
    # to find that out.
    try:
        known = beacon(root)
        if known is not None:
            known.unregister(port)
    except Exception as exc:  # noqa: BLE001 - a stale entry costs a peer one timeout
        return f"  could not withdraw it from the fleet: {exc}"
    return ""


def alongside(model: str, asked: str, prefix: str, *, best: bool = False) -> str:
    """A file shipped with ``model`` whose name starts with ``prefix``, resolving 'auto'.

    An `hf:` reference is asked of the Hub; a local path is answered by looking in the
    model's own directory. ``best`` picks the most precise of several, which is what a
    vision projector wants and what alphabetical order gets wrong: `mmproj-BF16` beats
    `mmproj-F32` on the letter B.
    """
    if asked.lower() != "auto":
        return asked
    from ml_stack.hub import _precision, beside

    reference = str(model)
    if reference.startswith("hf:"):
        return beside("/".join(reference[3:].split("/")[:2]), prefix, best=best)
    # A Hub cache keeps one folder per revision, and a sharded download puts the weights in
    # a per-quantisation subfolder with the projector at the snapshot root: beside the file,
    # then the directory above, then every revision of the same repository.
    where = home.expand(reference).parent
    places: list[tuple[Path, str]] = [(where, f"{prefix}*.gguf"),
                                      (where.parent, f"{prefix}*.gguf")]
    for parent in where.parents:
        if parent.name == "snapshots":
            places.append((parent, f"*/**/{prefix}*.gguf"))
            break
    for here, pattern in places:
        found = sorted(here.glob(pattern))
        if found:
            return str(min(found, key=lambda f: _precision(f.name)) if best else found[0])
    return ""


def drafted(model: str, asked: str, *, borrows: bool | None = None,
            binary: str | Path | None = None) -> str:
    """The draft head to serve with ``model``, resolving 'auto'.

    'auto' is `hub.choose_head`'s decision for ``binary`` (``None``: the one `find_binary`
    would pick), and its reason is written to stderr. ``borrows`` overrides what the binary
    says. A head is named by the method it implements -- `mtp-`, `eagle3-` -- and
    `hub.spec_for` reads which `--spec-type` it needs.
    """
    if asked.lower() != "auto":
        return asked
    from ml_stack.hub import choose_head

    chosen = choose_head(model, binary=binary, borrows=borrows)
    warn(f"draft head: {chosen.path or 'none'} -- {chosen.why}")
    return chosen.path


def resolve_model(named: str) -> str:
    """A bare model name found in the Hub cache; a path or an ``hf:`` reference as given.

    A name copied out of `ml-stack-models files` used to be read as a relative path and
    fail preflight with "shards missing" for a model that was on the machine.
    """
    if not named or named.startswith("hf:") or "/" in named:
        return named
    from ml_stack.hub import located

    found = located(named)
    return str(found) if found is not None else named


def manager_for(binary: str = "", build: str = "") -> ServerManager:
    """A manager over the named binary or build, or over whichever one is current."""
    from ml_stack.serve import backend as backend_module

    made = (backend_module.LlamaServerBackend(binary=binary or None, build=build or None)
            if (binary or build) else None)
    return ServerManager(made, state_file=lease_file())


def _binary_of(manager: ServerManager) -> Path | None:
    try:
        return manager.backend.binary
    except (BinaryNotFound, OSError):
        return None


def _head_for(spec: ServerSpec, manager: ServerManager) -> tuple[str, list[str]]:
    """The draft head 'auto' picks for ``spec``, and the lines saying why."""
    from ml_stack.hub import choose_head

    # The chooser is told which binary will serve: a head that borrows its target's
    # embeddings loads only under a named fork build.
    chosen = choose_head(str(spec.model), binary=_binary_of(manager))
    build_said = "a fork build" if chosen.borrows else "mainline"
    notes = [f"draft head: {chosen.path} -- {chosen.why} (serving with {build_said})"
             if chosen.path else f"no draft head served -- {chosen.why}"]
    if chosen.note:
        hint = "" if chosen.borrows else " Serve with --build NAME to use one."
        notes.append(f"  {chosen.note}{hint}")
    return chosen.path, notes


def resolve_spec(spec: ServerSpec, *, manager: ServerManager) -> Resolved:
    """``spec`` with 'auto' answered for the draft head and the vision projector, the
    speculation kind read off the head, and the context YaRN would need."""
    notes: list[str] = []
    draft = str(spec.draft or "")
    if draft.lower() == "auto":
        draft, said = _head_for(spec, manager)
        notes += said
    asked_mmproj = str(spec.mmproj or "")
    seeing = alongside(str(spec.model), asked_mmproj, "mmproj-", best=True)
    # A head implements one method and says which in its name.
    kind = str(spec.spec_type or "")
    if draft and not kind:
        from ml_stack.hub import spec_for

        kind = spec_for(draft)
    if asked_mmproj.lower() == "auto" and not seeing:
        notes.append("no vision projector is shipped beside that model; it will not read "
                     "pictures")
    spec = replace(spec, draft=draft or None, mmproj=seeing or None, spec_type=kind)
    spec, yarn_said = LlamaServerBackend.resolved_context(spec)
    if yarn_said:
        notes.append(yarn_said)
    return Resolved(spec, tuple(notes))


def preflight(spec: ServerSpec, *, manager: ServerManager) -> Any:
    """Every check a load would run, without starting or adopting anything."""
    from ml_stack.hub import room

    binary = manager.backend.binary
    # a draft named by hf: file is fetched and served by path, exactly as start() does
    return preflight_mod.Preflight(LlamaServerBackend.resolved_draft(spec), binary=binary,
                                   limit_bytes=room())


def up(spec: ServerSpec, *, manager: ServerManager, timeout: float | None = None,
       escalate: bool = False, root: str = DEFAULT_ROOT,
       say: Callable[[str], None] | None = None,
       on_event: Callable[[dict], None] | None = None) -> Started:
    """Lease a server for ``spec`` -- adopting one already up, or starting one -- record it
    under its own pid, and tell the fleet."""
    if say is not None:
        manager.say = say
    info = manager.lease(spec, timeout=timeout, escalate=escalate, on_event=on_event)
    held = recorded_servers(lease_file()).get(info.port) or {}
    if not info.adopted or held.get("owner_pid") == os.getpid():
        # a server this process started, or an orphan it took over: on the record under the
        # server's own pid once this command has exited
        manager.detach(info)
    return Started(info, spec, announce(root, spec))


def shapes(model: str = "") -> list[Any]:
    """The measured shapes: every record, or the one for ``model``.

    Raises `Refused` when a model is named and nothing has measured it.
    """
    every = profile_mod.profiles()
    if not model:
        return every
    found = profile_mod.profile_for(model, records=every)
    if found is None:
        raise Refused(f"nothing measured for {model.rsplit('/', 1)[-1]}. "
                      "`ml-stack-bench sweep` measures it and `ml-stack-bench report "
                      "--profile` writes the record.")
    return [found]


def tensors(models: Iterable[str]) -> list[str]:
    """What each model file is made of, from its GGUF header alone."""
    import struct

    out = []
    for one in models:
        try:
            out.append(fit_mod.render_tensors(resolve_model(one)))
        except (OSError, ValueError, struct.error) as exc:
            raise Refused(f"cannot read {one}: {exc}") from exc
    return out


def measure(names: Sequence[str], *, spec: ServerSpec, backend: LlamaServerBackend,
            timeout: float | None = None, room: int = 0, draft: str = "",
            resident: tuple[int, int] = (0, 0)) -> list[Recorded]:
    """Serve each named model once at ``-lv 4`` and record what it allocated."""
    from ml_stack.serve.manager import weight_of

    out: list[Recorded] = []
    for named in names:
        model = resolve_model(str(named))
        # told which binary serves, as `up` is: a head that borrows its target's embeddings
        # is offered to a fork build and withheld from mainline
        try:
            serving_binary: Path | None = backend.binary
        except (BinaryNotFound, OSError):
            serving_binary = None
        head = drafted(model, draft, binary=serving_binary)
        kind = ""
        if head:
            from ml_stack.hub import spec_for

            kind = spec_for(head)
        one = replace(spec, model=model, draft=head or None, spec_type=kind)
        try:
            measured = fit_mod.measure(one, backend=backend, timeout=timeout)
        except Exception as exc:
            raise Refused(f"could not measure {Path(model).name}: {exc}") from exc
        if not measured.measured:
            raise Refused(f"{Path(model).name} loaded but its log said nothing about a "
                          "cache. That is what a build too old for `-lv 4` looks like; "
                          "nothing was recorded.")
        record = fit_mod.Fit.of(
            # named as the person named it: a bare file name, the file in an hf: reference,
            # or a path's last part -- never the Hub cache's blob hash
            measured, model=Path(str(named).rsplit("/", 1)[-1]).name,
            weights=preflight_mod._shards_of(one)[0] or weight_of(model),
            draft=preflight_mod._ref_bytes(head or None), room=room,
            cache_type=one.cache_type_k or measured.cache_type, spec=kind,
            context=one.context, parallel=one.parallel,
            # from the header, not the log: how much of this file is a gathered table
            table_bytes=fit_mod.table_bytes(model),
            resident_peak=int(resident[0]), resident_after=int(resident[1]))
        out.append(Recorded(record.model, measured.said(), fit_mod.add(record)))
    return out


FIT_HEAD = (
    "# What fits\n\nMeasured at load, not estimated -- see "
    "`src/ml_stack/data/fit.json`.\n"
    "\nA model is smaller in memory than it is on disk, and the gap is tens of "
    "gigabytes on the models worth serving. llama.cpp mmaps the GGUF and copies "
    "into a device buffer only the tensors the backend takes; the rest stay mapped "
    "in the file and are paged, so they never appear in 'Real Mem'. Where a block "
    "below says *on disk / in GPU memory / mapped on the CPU*, those three numbers "
    "come from the load log's own `load_tensors: ... model buffer size` lines, and "
    "the one that has to fit beside the KV cache is the middle one.\n"
    "\nThe usual culprits are a lookup table that is gathered rather than "
    "multiplied (`Qwen3.8-Flash-Next`'s `per_layer_token_embd.weight` is a single "
    "26.8G n-gram table, paged a row at a time as distinct n-grams turn up), a "
    "tensor type the backend has no kernel for, an output layer that was not "
    "offloaded, and anything past `--n-gpu-layers`. "
    "`ml-stack-serve fit MODEL --tensors` totals a file's tensors by type and by "
    "what they are for, and needs nothing running.\n")


def fit_markdown(contexts: Sequence[int], rooms: Sequence[int], *, here: int,
                 drawn: str = "") -> str:
    """Every measured record as Markdown: this machine first, then a section per room."""
    every = fit_mod.records()
    head = FIT_HEAD
    if drawn:
        # The chart sits beside the file it is named in, so the Markdown refers to it by
        # name alone and the pair can be moved together.
        head += f"\n![How many fit, and what it costs]({Path(drawn).name})\n"
    parts = [head + f"\n## This machine ({human_bytes(here)})\n\n"
             + fit_mod.render(every, list(contexts), here, True)]
    for asked in rooms:
        if asked != here:
            parts.append(f"## A machine with {human_bytes(asked)}\n\n"
                         + fit_mod.render(every, list(contexts), asked, True))
    return "\n\n".join(parts) + "\n"


def fit_page(name: str = "") -> Any:
    """The fit page on loopback: the same routes the app mounts, on a free port."""
    from ml_stack.fleet.ui import serve_page

    return serve_page(name=name or platform.node() or "this machine")


PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>stack.ml.wired-limit</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/sbin/sysctl</string>
    <string>-w</string>
    <string>iogpu.wired_limit_mb={mb}</string>
  </array>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
"""


def memory() -> Machine:
    """What a model may use here, what is installed, and what is held right now."""
    from ml_stack.hub import room, total_memory

    now = room()
    return Machine(now, total_memory(), machine_memory() if now else None)


def write_plist(where: Path, mb: int) -> Path:
    """Write a boot-time setting for the wiring limit and return where it went."""
    where.write_text(PLIST.format(mb=mb), encoding="utf-8")
    return where


def limits(*, memory_size: str = "", servers: int | None = None, seats: int | None = None,
           idle: str = "", clear: bool = False) -> Limits:
    """Set whatever is named, then read back what this machine allows.

    Every limit is off until somebody sets one. Raises `Refused` on a size or a length of
    time that cannot be read.
    """
    from ml_stack.bench.history import parse_duration
    from ml_stack.hub import machine_room

    if clear:
        return Limits((), limits_mod.clear(), 0, 0)

    asked: dict[str, Any] = {}
    if memory_size:
        try:
            asked["memory_bytes"] = fit_mod.parse_room(memory_size)
        except ValueError as why:
            raise Refused(str(why)) from why
    if idle:
        seconds = parse_duration(idle)
        if seconds is None:
            raise Refused(f"cannot read {idle!r} as a length of time; try 10m")
        asked["idle_s"] = seconds
    for name, value in (("servers", servers), ("seats", seats)):
        if value is not None:
            asked[name] = int(value)
    if asked:
        limits_mod.changed(**asked)

    read_back = limits_mod.read()
    machine = machine_room()
    return Limits(tuple(read_back.said()), limits_mod.where(), machine,
                  read_back.room(machine))


def reclaim(*, idle: str = "") -> tuple[float, Any]:
    """The idle seconds to reclaim after, and a fresh watcher to look with.

    Raises `Refused` when no idle time is given and none is set.
    """
    from ml_stack.bench.history import parse_duration

    older = parse_duration(idle) if idle else limits_mod.read().idle_s
    if not older:
        raise Refused("no idle time given and none is set; nothing to reclaim "
                      "(ml-stack-serve limits --idle 10m)")
    return older, reclaim_mod.Idleness()


def reclaim_once(older: float, watcher: Any, *, settle: float = 0.0,
                 say: Callable[[str], None] = print) -> Any:
    """One pass: look at what is recorded, wait ``settle`` seconds, stop what is idle."""
    watcher.look(dict(recorded_servers(lease_file())))
    if settle:
        time.sleep(settle)
    return reclaim_mod.reclaim_idle(older_than=older, idleness=watcher, say=say)


def down(port: int, *, root: str = DEFAULT_ROOT) -> tuple[Stopped, str]:
    """Stop the server this machine recorded on ``port`` and withdraw it from the fleet."""
    records = recorded_servers(lease_file())
    entry = records.get(port)
    url = base_url_for(port)

    if entry is None:
        if not is_healthy(url, timeout=PROBE_TIMEOUT):
            raise Refused(f"nothing is serving on port {port}.")
        held = server_pids_on_port(port)
        where = f" (pid {held[0]})" if held else ""
        raise Refused(f"something is serving on {url}{where}, and this machine has no "
                      "record of starting it.", "  stop it the way it was started.")

    owner = _int_or_none(entry.get("owner_pid"))
    pid = _int_or_none(entry.get("pid"))
    if owner is not None and owner != pid and pid_exists(owner):
        raise Refused(f"{url} is held by process {owner}, which is still running.",
                      "  that process started it and will stop it.")

    running = pid_exists(pid)
    ServerManager(state_file=lease_file()).release(
        ServerInfo(base_url=str(entry.get("base_url") or url), port=port, pid=pid,
                   backend=str(entry.get("backend") or "")))
    return Stopped(port, url, pid, running, owner), withdraw(root, port)


def orphans(*, root: str = DEFAULT_ROOT) -> list[tuple[Stopped, str]]:
    """Stop every recorded server whose leasing process has gone; leave every other."""
    records = recorded_servers(lease_file())
    found = [(port, entry) for port, entry in sorted(records.items()) if orphaned(entry)]
    manager = ServerManager(state_file=lease_file())
    out = []
    for port, entry in found:
        url = base_url_for(port)
        pid = int(entry["pid"])
        manager.release(ServerInfo(base_url=str(entry.get("base_url") or url), port=port,
                                   pid=pid, backend=str(entry.get("backend") or "")))
        out.append((Stopped(port, url, pid, True, _int_or_none(entry.get("owner_pid"))),
                    withdraw(root, port)))
    return out


def escalate(port: int, *, add: int = 1, room: str = "", timeout: float | None = None,
             slot_save_path: str = "",
             on_event: Callable[[dict], None] | None = None) -> ServerInfo:
    """Grow the seats the server on ``port`` holds, keeping every live conversation."""
    manager = ServerManager(state_file=lease_file())
    base_url = base_url_for(port)
    if not is_healthy(base_url, timeout=PROBE_TIMEOUT):
        raise Refused(f"nothing is answering on port {port} to escalate")
    params = serving_params(base_url)
    if params is None or not params.model or params.n_ctx is None or params.total_slots is None:
        raise Refused(f"port {port} does not say enough about its own shape to escalate "
                      "-- is /props answering, with --slots enabled?")
    current = ServerSpec(model=params.model, port=port,
                         context=int(params.n_ctx) * int(params.total_slots),
                         parallel=int(params.total_slots),
                         slot_save_path=slot_save_path or str(default_slot_save_path()))
    return manager.escalate(current, add_seats=max(1, int(add)),
                            room=fit_mod.parse_room(room) if room else None,
                            timeout=timeout,
                            on_event=on_event)
