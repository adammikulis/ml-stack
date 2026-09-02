"""``ml-stack-serve`` -- see what is serving, put a model up, take one down."""

from __future__ import annotations

import argparse
import contextlib
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


def cmd_status(args: argparse.Namespace) -> int:
    records = recorded_servers(STATE_FILE)
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

    memory = sub.add_parser("memory", help="how much a model may use here, and whether that "
                                           "survives a reboot")
    memory.add_argument("--persist", nargs="?", const="", default=None, metavar="MB",
                        help="write a boot-time setting for the wiring limit; the megabytes "
                             "default to whatever is set now")
    memory.add_argument("--write", default="", metavar="FILE",
                        help="where to write it (default: ./stack.ml.wired-limit.plist)")

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
            "build": build.cmd_build}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
