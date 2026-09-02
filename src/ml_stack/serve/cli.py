"""``ml-stack-serve`` -- see what is serving, put a model up, take one down."""

from __future__ import annotations

import argparse
import contextlib
import platform
import subprocess
import pathlib
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ml_stack.client import is_healthy, reported_models
from ml_stack.client.health import serving_params
from ml_stack.fleet.serving import Serving
from ml_stack.serve import build
from ml_stack.serve.backend import ServerFailed, ServerInfo, ServerSpec
from ml_stack.serve.binary import BinaryNotFound
from ml_stack.serve.manager import DEFAULT_TIMEOUT_S, STATE_FILE, ServerManager, recorded_servers
from ml_stack.serve.ports import DEFAULT_HOST, server_pids_on_port
from ml_stack.serve.process import pid_exists

_SPEC = ServerSpec(model="")
DEFAULT_PORT = _SPEC.port
DEFAULT_CONTEXT = _SPEC.context
DEFAULT_PARALLEL = _SPEC.parallel
DEFAULT_TIMEOUT = DEFAULT_TIMEOUT_S
PROBE_TIMEOUT = 2.0
# The per-user contexts `fit` tabulates unless told otherwise; named here so --help can
# say them without importing the module that measures.
FIT_PER_USER = (4096, 8192, 16384, 32768, 65536, 131072)


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


def _lease_line(snapshot: Snapshot) -> str:
    if not snapshot.recorded:
        return "none on record -- 'ml-stack-serve down' will not stop this server"
    pid = f"server pid {snapshot.pid}" if snapshot.pid else "server pid not recorded"
    if snapshot.owner_pid is None:
        return pid
    if snapshot.owner_pid == snapshot.pid:
        return f"{pid}, started by 'ml-stack-serve up'"
    if snapshot.holder_running:
        return f"{pid}, held by process {snapshot.owner_pid}"
    return f"{pid}, the process that started it (pid {snapshot.owner_pid}) has gone"


def _verdict_line(snapshot: Snapshot, model: str, parallel: int) -> str:
    ask = f"'ml-stack-serve up {model} --parallel {parallel}'"
    if snapshot.verdict == "adopt":
        return f"{ask} would adopt this server"
    if snapshot.verdict == "refuse":
        return f"{ask} would be refused -- {snapshot.reason}"
    return f"{ask} would start its own server"


def every_server() -> list[dict]:
    """Every llama-server process on this machine, leased or not: pid, port, model, memory.
    A server nobody recorded -- a Homebrew one from before the managed build, a hand start
    -- holds memory `status` cannot otherwise see, and `pgrep` by hand is what the guard
    refuses."""
    try:
        import psutil
    except ImportError:
        return []
    out = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            argv = list(proc.info.get("cmdline") or [])
            name = str(proc.info.get("name") or "")
        except (psutil.Error, OSError):
            continue
        head = Path(argv[0]).name if argv else name
        if "llama-server" not in head and "llama-server" not in name:
            continue

        def after(flag: str, short: str = "") -> str:
            for i, a in enumerate(argv[:-1]):
                if a == flag or (short and a == short):
                    return argv[i + 1]
                if a.startswith(flag + "="):
                    return a.split("=", 1)[1]
            return ""

        mem = proc.info.get("memory_info")
        try:
            state = str(proc.status())
        except (psutil.Error, OSError):
            state = ""
        out.append({"pid": int(proc.info["pid"]), "port": int(after("--port") or 8080),
                    "defunct": state == psutil.STATUS_ZOMBIE,
                    "model": after("--model", "-m") or after("-hf") or "",
                    "binary": argv[0] if argv else name,
                    "rss": int(getattr(mem, "rss", 0) or 0)})
    return sorted(out, key=lambda r: r["port"])


def cmd_status(args: argparse.Namespace) -> int:
    records = recorded_servers(STATE_FILE)
    if getattr(args, "every", False):
        from ml_stack.hub import pretty_name

        found = every_server()
        if not found:
            print("no llama-server is running on this machine.")
            return 1
        for one in found:
            if one.get("defunct"):
                # a zombie holds no memory and answers no port; it is waiting to be reaped
                print(f"  pid {one['pid']}  defunct -- exited, not yet reaped; holds nothing")
                continue
            leased = "leased" if one["port"] in records else "NOT leased -- nobody records it"
            rss = f"{one['rss'] / 2**30:.1f}G" if one["rss"] else "?"
            print(f"  :{one['port']}  pid {one['pid']}  {pretty_name(one['model']) or '?'}  "
                  f"{rss} resident  {leased}  ({one['binary']})")
        strays = [o for o in found if o["port"] not in records and not o.get("defunct")]
        if strays:
            print(f"  {len(strays)} not leased: 'ml-stack-serve down --port N' stops one")
        return 0
    ports = sorted({*records, args.port})
    manager = ServerManager(state_file=STATE_FILE)

    found: list[Snapshot] = []
    for port in ports:
        snapshot = look(port, records)
        if snapshot is None:
            continue
        model = args.model or snapshot.model
        if model:
            snapshot = judge(
                manager,
                snapshot,
                ServerSpec(model=model, port=port, context=args.context,
                           parallel=args.parallel),
            )
        found.append(snapshot)

    if args.json:
        print(json.dumps(
            {"serving": bool(found), "ports_checked": ports,
             "servers": [asdict(s) for s in found]},
            indent=2))
        return 0 if found else 1

    if not found:
        print("nothing is serving on port " + ", ".join(str(p) for p in ports) + ".")
        print(f"  'ml-stack-serve up <model>' would start one on port {args.port}.")
        return 1

    for snapshot in found:
        quant = f"  ({snapshot.quant})" if snapshot.quant else ""
        print(snapshot.base_url)
        print(f"  model    {snapshot.model or 'not reported'}{quant}")
        print(f"  context  {snapshot.context if snapshot.context is not None else 'not reported'}"
              " per slot")
        print(f"  slots    {snapshot.slots if snapshot.slots is not None else 'not reported'}")
        print(f"  lease    {_lease_line(snapshot)}")
        if snapshot.load_s is not None:
            warm = f", warm-up {snapshot.warmup_s:.1f}s" if snapshot.warmup_s is not None else ""
            print(f"  loaded   in {snapshot.load_s:.1f}s{warm}")
        if snapshot.verdict:
            print("  " + _verdict_line(snapshot, args.model or snapshot.model or "<model>",
                                       args.parallel))
    return 0


# Where the daemon keeps the list of what this machine is serving. Peers read it to find a
# machine that already has a model loaded, so a server nobody announced is a server nobody
# else can use.
DEFAULT_ROOT = "~/.ml-stack/traind"


def beacon(root: str) -> Serving | None:
    """This machine's beacon, or None when no fleet was ever set up here.

    A machine with no daemon has no beacon to write to, and creating one would advertise a
    model to nobody through a file nothing reads. Announcing is for machines in a fleet.
    """
    where = Path(root).expanduser()
    return Serving(where / "serving.json") if where.is_dir() else None


def announce(args: argparse.Namespace, spec: ServerSpec) -> str:
    """Tell the fleet this machine is serving it. Returns a line to print, or ''."""
    try:
        known = beacon(args.root)
        if known is None:
            return ""
        known.register(spec.port, models=[str(spec.model)], slots=spec.parallel)
        return f"announced to the fleet on port {spec.port}"
    except Exception as exc:  # noqa: BLE001 - a server that works unannounced still works
        return f"could not announce it to the fleet: {exc}"


def alongside(model: str, asked: str, prefix: str, *, best: bool = False) -> str:
    """A file shipped with ``model`` whose name starts with ``prefix``, resolving 'auto'.

    An `hf:` reference is asked of the Hub; a local path is answered by looking in the
    model's own directory, because a cached repository puts what travels with the weights
    beside them. Anything else is taken as written.

    ``best`` picks the most precise of several, which is what a vision projector wants and
    what plain alphabetical order gets wrong: sorted by name, `mmproj-BF16` beats
    `mmproj-F32` on the letter B.
    """
    if asked.lower() != "auto":
        return asked
    from ml_stack.hub import _precision, beside

    reference = str(model)
    if reference.startswith("hf:"):
        return beside("/".join(reference[3:].split("/")[:2]), prefix, best=best)
    # "Shipped with this model" means in this repository, not in this directory, and not
    # even in this revision of it. A Hub cache keeps one folder per revision, so weights
    # fetched in August and a draft head fetched today land in different ones -- which is
    # exactly what happened, and `--draft auto` reported no head for a model that ships
    # three. A sharded download also puts the weights in a per-quantisation subfolder and
    # leaves the projector at the snapshot root.
    #
    # So: beside the file, then the directory above, then every revision of the same
    # repository. Widening, and stopping at the first place that has one.
    where = pathlib.Path(reference).expanduser().parent
    places: list[tuple[pathlib.Path, str]] = [(where, f"{prefix}*.gguf"),
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

    'auto' is `hub.choose_head`'s decision -- the one resolver `up`, the bench and the app
    share -- made for ``binary`` (``None``: the one `find_binary` would pick), and its
    reason is printed to stderr so a head withheld from mainline is withheld out loud.
    ``borrows`` overrides what the binary says, for a caller that knows better.

    Anything other than 'auto' is taken as written. A head is named by the method it
    implements -- `mtp-` for multi-token prediction, `eagle3-` for EAGLE3 -- and
    `hub.spec_for` reads which `--spec-type` the one chosen needs.
    """
    if asked.lower() != "auto":
        return asked
    from ml_stack.hub import choose_head

    chosen = choose_head(model, binary=binary, borrows=borrows)
    print(f"draft head: {chosen.path or 'none'} -- {chosen.why}", file=sys.stderr)
    return chosen.path


def resolve_model(named: str) -> str:
    """A bare model name, found in the Hub cache -- a path or an ``hf:`` reference is used
    exactly as given.

    A name copied straight out of `ml-stack-models files` -- no directory, no `hf:` prefix
    -- used to be read as a relative path and fail preflight with "shards missing" for a
    model that was on the machine the whole time: `up gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
    --preflight-only` did exactly that. `graph.bench.find_model` already solved the same
    problem for the bench by asking `fleet.models` where a bare name lives; `hub.located`
    is the same idea, narrowed to the Hub cache and an exact filename -- what `up` is
    actually handed.
    """
    if not named or named.startswith("hf:") or "/" in named:
        return named
    from ml_stack.hub import located

    found = located(named)
    return str(found) if found is not None else named


def cmd_up(args: argparse.Namespace) -> int:
    from ml_stack.serve.backend import LlamaServerBackend, UnknownFlag

    model = resolve_model(str(args.model))
    if model != str(args.model):
        print(f"resolved {args.model} -> {model}", file=sys.stderr)

    chosen = str(getattr(args, "binary", "") or "")
    build_name = str(getattr(args, "build", "") or "")
    manager = ServerManager(
        LlamaServerBackend(binary=chosen or None, build=build_name or None)
        if (chosen or build_name) else None,
        state_file=STATE_FILE)
    asked = str(getattr(args, "draft", "") or "")
    draft = asked
    if asked.lower() == "auto":
        # The chooser is told which binary will serve: a named fork build is the only case
        # a head that borrows its target's embeddings can load, and offering one to
        # 'current' is offering something that fails at the far end of a multi-gigabyte
        # load. Which binary that is, it reads off the path -- not off the flags.
        from ml_stack.hub import choose_head

        try:
            binary_path: Path | None = manager.backend.binary
        except (BinaryNotFound, OSError):
            binary_path = None
        chosen = choose_head(model, binary=binary_path)
        draft = chosen.path
        build_said = "a fork build" if chosen.borrows else "mainline"
        if draft:
            print(f"draft head: {draft} -- {chosen.why} (serving with {build_said})",
                  file=sys.stderr)
        else:
            print(f"no draft head served -- {chosen.why}", file=sys.stderr)
        if chosen.note:
            hint = "" if chosen.borrows else " Serve with --build NAME to use one."
            print(f"  {chosen.note}{hint}", file=sys.stderr)
    seeing = alongside(model, str(getattr(args, "mmproj", "") or ""), "mmproj-",
                       best=True)
    # A head implements one method and says which in its name. Serving an EAGLE3 head
    # without --spec-type draft-eagle3 is asking it to do something it does not do.
    kind = str(getattr(args, "spec", "") or "")
    if draft and not kind:
        from ml_stack.hub import spec_for

        kind = spec_for(draft)
    if str(getattr(args, "mmproj", "")).lower() == "auto" and not seeing:
        print("no vision projector is shipped beside that model; it will not read pictures",
              file=sys.stderr)
    spec = ServerSpec(model=model, port=args.port, context=args.context,
                      parallel=args.parallel, draft=draft or None, mmproj=seeing or None,
                      spec_type=kind,
                      spec_draft_max=getattr(args, "spec_n_max", None),
                      spec_draft_ngl=getattr(args, "draft_ngl", None),
                      lookup_dynamic=str(getattr(args, "lookup_cache", "") or "") or None,
                      override_tensor=tuple(getattr(args, "on_cpu", []) or ()),
                      cpu_moe=bool(getattr(args, "cpu_moe", False)))

    if getattr(args, "preflight_only", False):
        from ml_stack.hub import room
        from ml_stack.serve.preflight import Preflight

        try:
            binary_path = manager.backend.binary
        except (BinaryNotFound, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        # a draft named by hf: file is fetched and served by path, exactly as start() does;
        # a preflight of the unresolved reference refused it instead (measured 2026-09-01)
        from ml_stack.serve.backend import LlamaServerBackend

        spec = LlamaServerBackend.resolved_draft(spec)
        report = Preflight(spec, binary=binary_path, limit_bytes=room())
        print(report.said())
        return 0 if report.ok else 1

    try:
        info = manager.lease(spec, timeout=args.timeout)
    except UnknownFlag as exc:
        # Refused before the load, not at the end of it: the build was asked what it
        # accepts and the answer is printed one flag per line, with the nearest it has.
        print(exc, file=sys.stderr)
        return 2
    except (ServerFailed, BinaryNotFound, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not info.adopted:
        manager.detach(info)

    told = announce(args, spec)

    if args.json:
        print(json.dumps({"base_url": info.base_url, "port": info.port, "pid": info.pid,
                          "adopted": info.adopted, "model": str(spec.model),
                          "context": spec.context, "parallel": spec.parallel,
                          "draft": str(spec.draft or ""),
                          "announced": told.startswith("announced")}, indent=2))
        return 0

    where = f" (pid {info.pid})" if info.pid else ""
    print(f"{'adopted' if info.adopted else 'started'} {info.base_url}{where}")
    if chosen:
        print(f"  with {chosen}")
    if spec.draft:
        print(f"  guessing ahead with {str(spec.draft).rsplit('/', 1)[-1]}")
    if spec.mmproj:
        print(f"  reading pictures with {str(spec.mmproj).rsplit('/', 1)[-1]}")
    if spec.spec_type:
        print(f"  guessing ahead by {spec.spec_type}")
    if told:
        print(f"  {told}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    """``ml-stack-serve fit`` -- how many people fit on this machine, at what context.

    Reads the measured records rather than a formula: `preflight`'s estimate counts every
    layer as full attention, and gemma4, gpt-oss and qwen4exp each disagree with that in a
    different way. `--measure` serves a model once at `-lv 4`, reads what llama.cpp says it
    allocated, and writes that into the source of truth.
    """
    from ml_stack.hub import room as machine_room
    from ml_stack.serve import fit as fit_mod

    if getattr(args, "ui", False):
        return _fit_ui()

    asked_rooms: list[int] = []
    for said in (getattr(args, "room", None) or []):
        try:
            asked_rooms.append(fit_mod.parse_room(said))
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    # The listing answers for one machine -- the first room named, or this one. The chart
    # draws every room asked for, which is the whole point of naming more than one.
    room = asked_rooms[0] if asked_rooms else machine_room()

    per_user = [int(n) for n in (getattr(args, "per_user", None) or [])]
    wanted = [Path(str(m)).name.lower() for m in (getattr(args, "model", None) or [])]

    if getattr(args, "measure", False):
        if not wanted:
            print("error: --measure needs a model to measure", file=sys.stderr)
            return 2
        code = _measure_each(args, room=room)
        if code:
            return code

    rows = fit_mod.records(room=room)
    if wanted:
        rows = [r for r in rows if r.model.lower() in wanted
                or any(w in r.model.lower() for w in wanted)]
        if not rows:
            print("nothing measured for " + ", ".join(wanted)
                  + " -- `ml-stack-serve fit MODEL --measure` serves it once and records "
                    "what it allocated.", file=sys.stderr)
            return 1

    contexts = per_user or list(fit_mod.DEFAULT_PER_USER)
    print(fit_mod.render(rows, contexts, room, bool(getattr(args, "md", False))))

    parallel = int(getattr(args, "parallel", 1) or 1)
    if parallel > 1:
        print()
        for row in rows:
            print(f"{row.model}: {parallel} users fit at "
                  f"{row.longest(parallel):,} tokens each")

    drawn = ""
    picture = str(getattr(args, "plot", "") or "")
    if picture:
        try:
            drawn = fit_mod.plot(rows, picture,
                                 # this machine's room first, solid; each --room after it
                                 rooms=[machine_room(), *(r for r in asked_rooms
                                                          if r != machine_room())],
                                 at=int(getattr(args, "at", 32768) or 32768),
                                 machine=platform.node() or "this machine")
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"\ndrew {drawn}", file=sys.stderr)
        if getattr(args, "open", False):
            from ml_stack.platform import open_path

            print(f"opened with {open_path(drawn)}", file=sys.stderr)

    where = str(getattr(args, "write", "") or "")
    if where:
        every = fit_mod.records()
        head = ("# What fits\n\nMeasured at load, not estimated -- see "
                "`src/ml_stack/data/fit.json`.\n")
        if drawn:
            # The chart sits beside the file it is named in, so the Markdown refers to it
            # by name alone and the pair can be moved together.
            head += f"\n![How many fit, and what it costs]({Path(drawn).name})\n"
        parts = [head + f"\n## This machine ({fit_mod._human(machine_room())})\n\n"
                 + fit_mod.render(every, contexts, machine_room(), True)]
        for asked in asked_rooms:
            if asked != machine_room():
                parts.append(f"## A machine with {fit_mod._human(asked)}\n\n"
                             + fit_mod.render(every, contexts, asked, True))
        Path(where).expanduser().write_text("\n\n".join(parts) + "\n", encoding="utf-8")
        print(f"\nwrote {where}", file=sys.stderr)
    return 0


def _fit_ui() -> int:
    """``fit --ui``: the interactive page, on loopback, until Ctrl-C.

    Not a second implementation of anything -- `fleet.ui.serve_page` mounts the same route
    table the app mounts, so `/ui/fit` and `/ui/fit.json` are the app's, and a machine with
    no daemon running still gets the page.
    """
    from ml_stack.fleet.ui import serve_page
    from ml_stack.platform import open_path

    server = serve_page(name=platform.node() or "this machine")
    where = f"http://127.0.0.1:{server.server_port}/ui/fit"
    print(f"the fit page is at {where}\n(loopback only; Ctrl-C to stop)", file=sys.stderr)
    print(f"opened with {open_path(where)}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    finally:
        server.server_close()
    return 0


def _measure_each(args: argparse.Namespace, *, room: int) -> int:
    """Serve each named model once and record what it allocated. Returns an exit code."""
    from ml_stack.serve import fit as fit_mod
    from ml_stack.serve.backend import LlamaServerBackend
    from ml_stack.serve.manager import weight_of
    from ml_stack.serve.preflight import _ref_bytes, _shards_of

    binary = str(getattr(args, "binary", "") or "")
    build_name = str(getattr(args, "build", "") or "")
    backend = (LlamaServerBackend(binary=binary or None, build=build_name or None)
               if (binary or build_name) else LlamaServerBackend())

    # Two slots, not one: the sliding-window and recurrent caches are sized for however many
    # sequences were asked for, and a measurement taken over one cannot tell a per-sequence
    # cost from a constant. `--parallel` is about who fits, not about how it was measured.
    slots = max(2, int(getattr(args, "parallel", 1) or 1))
    kv = str(getattr(args, "kv", "") or "")

    for named in args.model:
        model = resolve_model(str(named))
        # told which binary serves, as `up` does: a head that borrows its target's embeddings
        # is offered to a fork build and withheld from mainline -- asked without it, the
        # Flash-Next head was withheld from the fork and the drafted record never existed
        try:
            serving_binary: Path | None = backend.binary
        except Exception:  # noqa: BLE001 - no binary is choose_head's problem, said out loud
            serving_binary = None
        draft = drafted(model, str(getattr(args, "draft", "") or ""), binary=serving_binary)
        kind = ""
        if draft:
            from ml_stack.hub import spec_for

            kind = spec_for(draft)
        spec = ServerSpec(model=model, port=args.port, context=args.context,
                          parallel=slots, draft=draft or None, spec_type=kind,
                          cache_type_k=kv, cache_type_v=kv, warmup=False)
        try:
            measured = fit_mod.measure(spec, backend=backend, timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001 - whatever the load said, say it here
            print(f"error: could not measure {Path(model).name}: {exc}", file=sys.stderr)
            return 2
        if not measured.measured:
            print(f"error: {Path(model).name} loaded but its log said nothing about a "
                  "cache. That is what a build too old for `-lv 4` looks like; nothing "
                  "was recorded.", file=sys.stderr)
            return 2
        record = fit_mod.Fit.of(
            # named as the person named it: a bare file name, the file in an hf: reference,
            # or a path's last part -- never the Hub cache's blob hash `located` resolves to
            measured, model=Path(str(named).rsplit("/", 1)[-1]).name,
            weights=_shards_of(spec)[0] or weight_of(model),
            draft=_ref_bytes(draft or None), room=room,
            cache_type=kv or measured.cache_type, spec=kind, context=args.context,
            parallel=slots)
        where = fit_mod.add(record)
        print(f"measured {record.model}: {measured.said()}", file=sys.stderr)
        print(f"  recorded in {where}", file=sys.stderr)
    return 0


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


def machine_memory() -> dict | None:
    """What the machine holds: total, used, wired, free, the llama-servers' resident total,
    everything else's, and the five largest non-server processes -- from psutil, or None
    without it."""
    from ml_stack.hub import _human

    try:
        import psutil
    except ImportError:
        return None
    try:
        vm = psutil.virtual_memory()
    except Exception:  # noqa: BLE001
        return None
    servers = 0
    rest: list[tuple[int, str]] = []
    for proc in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            mem = proc.info.get("memory_info")
            rss = int(getattr(mem, "rss", 0) or 0)
            argv = list(proc.info.get("cmdline") or [])
            head = Path(argv[0]).name if argv else str(proc.info.get("name") or "")
        except (psutil.Error, OSError):
            continue
        if "llama-server" in head:
            servers += rss
        elif rss:
            rest.append((rss, head))
    rest.sort(reverse=True)
    return {"total": int(vm.total), "used": int(vm.total - vm.available),
            "wired": int(getattr(vm, "wired", 0) or 0), "free": int(vm.available),
            "servers": servers, "others": sum(r for r, _ in rest),
            "largest": [f"{name} {_human(r)}" for r, name in rest[:5]]}


def cmd_memory(args: argparse.Namespace) -> int:
    """``ml-stack-serve memory`` -- what this machine will let a model use, and for how long.

    On unified memory the ceiling that matters is not how much RAM there is, it is how much
    of it Metal will wire: a model and its KV cache have to fit under `iogpu.wired_limit_mb`.
    That setting is a runtime one and **goes back to the default on every reboot**, so a
    model that loaded yesterday can fail today with an error that never mentions memory.
    """
    from ml_stack.hub import _human, room

    total = 0
    with contextlib.suppress(Exception):
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                   text=True, timeout=5).stdout.strip())
    now = room()
    if not now:
        print("this machine does not report a wiring limit; nothing to do here")
        return 0

    print(f"a model may use about {_human(now)}"
          + (f" of {_human(total)} installed" if total else ""))
    if total:
        default = int(total * 0.75)
        if now > default * 1.02:
            print(f"  raised from the ~{_human(default)} default -- and **not** kept: this "
                  f"resets on reboot")
        else:
            print(f"  this is the default share; {_human(total)} is installed")

    # What the rest of the machine holds right now, so a higher limit is chosen against
    # what it would take from the desktop rather than guessed (Adam, 2026-09-02: "take a
    # look at what os and apps are using, can we increase vram from 110 to something
    # higher or is that our ceiling?")
    held = machine_memory()
    if held:
        print(f"\nright now: {_human(held['used'])} used of {_human(held['total'])} "
              f"({_human(held['wired'])} wired, {_human(held['free'])} free)")
        servers = held["servers"]
        others = held["others"]
        print(f"  llama-server(s): {_human(servers)}; everything else: {_human(others)}"
              + (f" -- {', '.join(held['largest'])}" if held["largest"] else ""))
        headroom = int(total) - int(now) if total else 0
        if total:
            print(f"  the limit leaves {_human(headroom)} for everything else; the rest of "
                  f"the machine holds {_human(others)} now"
                  + (" -- room to raise it" if others < headroom * 0.6
                     else " -- close to it; raising it means swapping when a model fills it"))
        want_mb = int(getattr(args, "limit", 0) or 0)
        if want_mb and total:
            left = int(total) - want_mb * 1024 * 1024
            print(f"  at {want_mb} MB the rest of the machine would have {_human(left)}"
                  + (" -- less than it holds now" if left < others else ""))
    want = args.persist
    if want is None:
        print("\n  ml-stack-serve memory --persist [MB]   to write a boot-time setting")
        return 0

    mb = int(want) if want else now // (1024 * 1024)
    where = Path(args.write or "./stack.ml.wired-limit.plist")
    where.write_text(PLIST.format(mb=mb), encoding="utf-8")
    print(f"\nwrote {where} -- it sets iogpu.wired_limit_mb={mb} at every boot.")
    print("Installing it needs root, so it is left to you:")
    print(f"  sudo cp {where} /Library/LaunchDaemons/stack.ml.wired-limit.plist")
    print("  sudo chown root:wheel /Library/LaunchDaemons/stack.ml.wired-limit.plist")
    print("  sudo launchctl load -w /Library/LaunchDaemons/stack.ml.wired-limit.plist")
    print(f"\nOr for this boot only:  sudo sysctl -w iogpu.wired_limit_mb={mb}")
    print("\nLeave headroom: everything else on the machine shares this memory, and a "
          "machine that wires all of it stops being usable before it stops serving.")
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    records = recorded_servers(STATE_FILE)
    entry = records.get(args.port)
    url = base_url_for(args.port)

    if entry is None:
        if not is_healthy(url, timeout=PROBE_TIMEOUT):
            print(f"nothing is serving on port {args.port}.")
            return 1
        held = server_pids_on_port(args.port)
        where = f" (pid {held[0]})" if held else ""
        print(f"error: something is serving on {url}{where}, and this machine has no "
              "record of starting it.", file=sys.stderr)
        print("  stop it the way it was started.", file=sys.stderr)
        return 2

    owner = _int_or_none(entry.get("owner_pid"))
    pid = _int_or_none(entry.get("pid"))
    if owner is not None and owner != pid and pid_exists(owner):
        print(f"error: {url} is held by process {owner}, which is still running.",
              file=sys.stderr)
        print("  that process started it and will stop it.", file=sys.stderr)
        return 2

    running = pid_exists(pid)
    ServerManager(state_file=STATE_FILE).release(
        ServerInfo(base_url=str(entry.get("base_url") or url), port=args.port, pid=pid,
                   backend=str(entry.get("backend") or "")))
    # A registration outlives the server it describes, and the beacon then sends work to a
    # port nothing answers on. `live()` probes before advertising, so a stale entry is not
    # fatal — but leaving one behind means every peer pays a timeout to find that out.
    try:
        known = beacon(args.root)
        if known is not None:
            known.unregister(args.port)
    except Exception as exc:  # noqa: BLE001
        print(f"  could not withdraw it from the fleet: {exc}", file=sys.stderr)

    if running:
        print(f"stopped {url} (pid {pid})")
    else:
        print(f"nothing was running on port {args.port}; removed the record")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ml-stack-serve",
        description="See which model is being served on this machine, put one up, take it down.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    status = sub.add_parser("status", help="what is serving, and what a lease would do")
    status.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port to check besides the recorded ones (default: {DEFAULT_PORT})")
    status.add_argument("--model", default="",
                        help="ask what leasing this model would do (default: whatever is "
                             "already serving)")
    status.add_argument("--context", type=int, default=DEFAULT_CONTEXT,
                        help=f"the context that lease would ask for (default: "
                             f"{DEFAULT_CONTEXT})")
    status.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL,
                        help=f"the slots that lease would ask for (default: "
                             f"{DEFAULT_PARALLEL})")
    status.add_argument("--every", action="store_true",
                        help="every llama-server process on this machine, leased or not -- "
                             "a stray one holds memory a lease cannot see")
    status.add_argument("--json", action="store_true",
                        help="print one JSON object instead of the human listing")

    up = sub.add_parser("up", help="serve a model, or adopt the one already serving it")
    up.add_argument("model", help="path to a .gguf file, or hf:owner/repo/file.gguf")
    up.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"port to serve on (default: {DEFAULT_PORT})")
    up.add_argument("--context", type=int, default=DEFAULT_CONTEXT,
                    help=f"tokens across all slots (default: {DEFAULT_CONTEXT})")
    up.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL,
                    help=f"slots to serve at once (default: {DEFAULT_PARALLEL})")
    up.add_argument("--timeout", type=float, default=None,
                    help="seconds to wait for it to load (default: scales with the "
                         f"weights on disk -- 60s + 1.5s/GB, floor {DEFAULT_TIMEOUT:.0f}s)")
    up.add_argument("--json", action="store_true",
                    help="print one JSON object instead of the human line")
    up.add_argument("--preflight-only", action="store_true",
                    help="run every check a load would run -- shards present, "
                         "architecture this build reads, an estimate against what this "
                         "machine may use, every flag the build accepts -- and print the "
                         "report without starting or adopting anything. Exits 0 or 1")
    up.add_argument("--root", default=DEFAULT_ROOT,
                    help=f"the fleet root whose beacon to announce in, when there is one "
                         f"(default: {DEFAULT_ROOT})")
    up.add_argument("--binary", default="", metavar="PATH",
                    help="the llama-server to run, when the one on PATH cannot read this "
                         "model. A release lags master by an architecture or two: gemma-4 "
                         "and qwen3moe are in the current release, qwen4exp is not, so "
                         "Qwen3.8-Flash-Next needs a build from master and says "
                         "'unknown model architecture' without one")
    up.add_argument("--build", default="", metavar="NAME",
                    help="serve with a named build 'ml-stack-serve build --name NAME' made "
                         "-- a fork kept beside 'current' rather than replacing it, e.g. a "
                         "fork whose fixes have not reached mainline yet. Ignored if "
                         "--binary is also given")
    up.add_argument("--mmproj", default="", metavar="PATH_OR_AUTO",
                    help="the vision projector, so the model can read a picture -- a path, "
                         "an hf: reference, or 'auto' to take the most precise one shipped "
                         "with the weights. An hf: model already pulls a projector by "
                         "itself, so 'auto' is for choosing a better one than it would: a "
                         "projector is a fraction of the weights and carries all of the "
                         "seeing, so quantising it is a false economy")
    up.add_argument("--spec", default="", metavar="TYPE",
                    help="how to guess ahead: an ngram-* kind needs no second model at all, "
                         "proposing tokens it has already seen in the prompt, which suits "
                         "work that copies from its context and costs no memory. "
                         "ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache, "
                         "or a draft-* kind with --draft. Left unset, the server decides")
    up.add_argument("--spec-n-max", type=int, default=None, metavar="N",
                    help="tokens guessed ahead each step (server default 3)")
    up.add_argument("--on-cpu", action="append", default=[], metavar="PATTERN=BUFFER",
                    help="keep tensors matching a pattern off the GPU, e.g. "
                         "'ngram.*=CPU'. This is what Qwen3.8-Flash-Next's N-gram Embedding "
                         "wants -- a 51B lookup table whose addresses are known in advance, "
                         "meant to sit in host memory and be prefetched rather than hold "
                         "GPU. Repeatable. Read the tensor names from the model rather than "
                         "guessing at the pattern")
    up.add_argument("--cpu-moe", action="store_true",
                    help="keep every Mixture-of-Experts weight on the CPU, which is how a "
                         "35B with 3B active fits a machine that could not hold it all")
    up.add_argument("--lookup-cache", default="", metavar="FILE",
                    help="an n-gram cache kept on disk and updated as it generates, so what "
                         "was learnt answering one question speculates the next. Only the "
                         "ngram-cache kind uses it; the other ngram kinds look up the "
                         "prompt itself and keep nothing")
    up.add_argument("--draft-ngl", type=int, default=None, metavar="N",
                    help="layers of the draft model to put on the GPU. Without it the draft "
                         "runs where the server puts it by default, which can be the CPU -- "
                         "and a draft slower than the model it is guessing for is a loss")
    up.add_argument("--draft", default="", metavar="MODEL_OR_AUTO",
                    help="a small model to guess ahead, which the large one checks in one "
                         "pass -- a path, an hf: reference, or 'auto' to use the draft head "
                         "shipped beside the weights (the mtp- file in a QAT repository)")

    fit_p = sub.add_parser(
        "fit", help="how many people fit at a given context, from measured KV numbers")
    fit_p.add_argument("model", nargs="*",
                       help="which measured models to report (a bare name, a path, or an "
                            "hf: reference). Default: every model that has been measured")
    fit_p.add_argument("--measure", action="store_true",
                       help="serve each named model once at -lv 4, read what llama.cpp says "
                            "it allocated -- the base cache per token, the sliding-window "
                            "and recurrent caches per sequence, the compute buffers -- and "
                            "record it. This is the only way the numbers get in: a formula "
                            "over the GGUF header counts every layer as full attention, and "
                            "gemma4 (18 layers share a cache, the rest slide), gpt-oss "
                            "(every other layer slides) and qwen4exp (three layers in four "
                            "are recurrent) each disagree with that differently")
    fit_p.add_argument("--draft", default="", metavar="MODEL_OR_AUTO",
                       help="measure it with a draft head as well -- a path, an hf: "
                            "reference, or 'auto'. A draft *model* keeps its own cache at "
                            "the same context, which is the real cost of drafting with one")
    fit_p.add_argument("--kv", default="", metavar="TYPE",
                       help="measure with the main model's KV cache stored as this: f16 "
                            "(the server's own default), q8_0, q4_0. A record is kept per "
                            "cache type, because that is what changes the per-token cost")
    fit_p.add_argument("--room", action="append", default=[], metavar="SIZE",
                       help="ask about a machine with this much memory instead of this one "
                            "-- 24G, 24576M, or a plain number of bytes. Default: what "
                            "`ml-stack-serve memory` says a model may use here. Repeatable: "
                            "the listing answers for the first, and --plot draws every one "
                            "of them, solid then dashed, so a laptop and a card can be "
                            "compared in the same picture")
    fit_p.add_argument("--per-user", type=int, action="append", dest="per_user",
                       default=[], metavar="N",
                       help="a per-user context to put in the table. Repeatable; default "
                            f"{', '.join(str(n) for n in FIT_PER_USER)}")
    fit_p.add_argument("--parallel", type=int, default=1, metavar="N",
                       help="also say the longest context N users could each be given "
                            "(default: 1, which is the line every block prints anyway). "
                            "Measuring always serves two slots, so a per-sequence cost can "
                            "be told apart from a constant")
    fit_p.add_argument("--plot", default="", metavar="FILE.png",
                       help="draw it: two panels, one figure -- how many users fit against "
                            "the context each gets, and what the memory costs as they "
                            "arrive. The second is the one worth having: a large model with "
                            "a small cache starts higher and climbs more slowly than a small "
                            "model with a fat one, and the picture is where they cross. "
                            ".png, .svg or .pdf; needs matplotlib")
    fit_p.add_argument("--open", action="store_true",
                       help="with --plot: open the picture when it is drawn")
    fit_p.add_argument("--ui", action="store_true",
                       help="put the same two panels up as a page you can move: a room "
                            "slider, a per-user context slider, a users slider and a model "
                            "per checkbox, redrawn as you drag. Serves on loopback, opens a "
                            "browser at it, and stays up until Ctrl-C. The fleet app shows "
                            "the same page under Fit")
    fit_p.add_argument("--at", type=int, default=32768, metavar="N",
                       help="the per-user context the second panel charges at "
                            "(default: 32768)")
    fit_p.add_argument("--md", action="store_true",
                       help="print Markdown rather than the plain listing")
    fit_p.add_argument("--write", default="", metavar="FILE",
                       help="write the Markdown for every record to a file -- at this "
                            "machine's room, and at --room's as a second section")
    fit_p.add_argument("--context", type=int, default=32768, metavar="N",
                       help="the context to measure at (default: 32768). The per-token cost "
                            "does not depend on it; a long one just measures it precisely")
    fit_p.add_argument("--port", type=int, default=DEFAULT_PORT,
                       help=f"the port to measure on (default: {DEFAULT_PORT})")
    fit_p.add_argument("--timeout", type=float, default=None,
                       help="seconds to wait for the measured load (default: scales with "
                            "the weights on disk)")
    fit_p.add_argument("--binary", default="", metavar="PATH",
                       help="the llama-server to measure with, when the one on PATH cannot "
                            "read this model")
    fit_p.add_argument("--build", default="", metavar="NAME",
                       help="measure with a named build, the way `up --build NAME` serves "
                            "with one")

    memory = sub.add_parser("memory", help="how much a model may use here, and whether that "
                                           "survives a reboot")
    memory.add_argument("--persist", nargs="?", const="", default=None, metavar="MB",
                        help="write a boot-time setting for the wiring limit; the megabytes "
                             "default to whatever is set now")
    memory.add_argument("--write", default="", metavar="FILE",
                        help="where to write it (default: ./stack.ml.wired-limit.plist)")
    memory.add_argument("--limit", type=int, default=0, metavar="MB",
                        help="preview: what the rest of the machine would have under this "
                             "wiring limit, against what it holds now")

    down = sub.add_parser("down", help="stop a server started on this machine")
    down.add_argument("--port", type=int, default=DEFAULT_PORT,
                      help=f"port of the server to stop (default: {DEFAULT_PORT})")
    down.add_argument("--root", default=DEFAULT_ROOT,
                      help=f"the fleet root to withdraw it from (default: {DEFAULT_ROOT})")

    build_p = sub.add_parser(
        "build", help="build llama-server from llama.cpp's own master (or download the "
                      "newest release), and switch to it once it is verified")
    build_p.add_argument("--from", dest="source_kind", default="", choices=["source", "release"],
                         help="'source' compiles master with cmake, 'release' downloads the "
                              "newest GitHub release with an asset for this machine. "
                              "Default: source when a compiler is on PATH, release otherwise")
    build_p.add_argument("--commit", default="", metavar="SHA",
                         help="build this commit instead of master's tip (--from source only)")
    build_p.add_argument("--jobs", type=int, default=0, metavar="N",
                         help="parallel compile jobs (default: every core)")
    build_p.add_argument("--source", default="", metavar="DIR",
                         help="reuse a checkout here instead of cloning/updating the "
                              "managed one")
    build_p.add_argument("--force", action="store_true",
                         help="rebuild or redownload even if this commit/release is "
                              "already installed")
    build_p.add_argument("--check", action="store_true",
                         help="report the installed build's commit, age and "
                              "architectures -- builds nothing")
    build_p.add_argument("--rollback", action="store_true",
                         help="point 'current' back at the previous verified build")
    build_p.add_argument("--persist", action="store_true",
                         help="install a weekly refresh (a LaunchAgent on macOS, a "
                              "Scheduled Task on Windows) that reruns this on its own")
    build_p.add_argument("--adopt", default="", metavar="DIR",
                         help="register a flat build directory that already exists -- a "
                              "hand-built binary, or a release zip unpacked by hand -- as "
                              "a managed build, verify it, and switch to it now, without "
                              "compiling or downloading anything")
    build_p.add_argument("--repo", default="", metavar="OWNER/REPO",
                         help="build a fork instead of ggml-org/llama.cpp's own master -- "
                              "combine with --name to keep it beside 'current' instead of "
                              "replacing it, e.g. --repo unslothai/llama.cpp --name unsloth")
    build_p.add_argument("--ref", default="", metavar="TAG_OR_BRANCH_OR_SHA",
                         help="the fork's ref to build (--from source, with --repo; "
                              "default: its default branch's tip)")
    build_p.add_argument("--tag", default="", metavar="TAG",
                         help="the fork's release tag to download (--from release, with "
                              "--repo; default: the newest release with a matching asset)")
    build_p.add_argument("--name", default="", metavar="NAME",
                         help="keep this build at ~/.ml-stack/llama.cpp/named/NAME instead "
                              "of replacing 'current' -- requires --repo. Select it with "
                              "'ml-stack-serve up --build NAME' or $MLSTACK_LLAMA_BUILD=NAME")
    build_p.add_argument("--list", action="store_true",
                         help="show 'current' and every named build, with commit, age and "
                              "repo -- builds nothing")

    args = ap.parse_args(argv)
    return {"status": cmd_status, "up": cmd_up, "down": cmd_down, "memory": cmd_memory,
            "fit": cmd_fit, "build": build.cmd_build}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
