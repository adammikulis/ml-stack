"""``ml-stack-serve`` -- see what is serving, put a model up, take one down.

Every subcommand parses its arguments and prints; the work is in `ml_stack.serve.ops`.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

from ml_stack import home, hub
from ml_stack.command import Group, flag, option
from ml_stack.log import say, warn
from ml_stack.serve import build, ops
from ml_stack.serve.backend import ServerFailed, ServerSpec, parse_context
from ml_stack.serve.binary import BinaryNotFound
from ml_stack.serve.manager import DEFAULT_TIMEOUT_S
from ml_stack.serve.ops import DEFAULT_ROOT, Refused, base_url_for
from ml_stack.serve.profile import ASK, WORKLOADS
from ml_stack.serve.shape import said_cache, split_cache_type
from ml_stack.units import human_bytes

__all__ = ["COMMANDS", "DEFAULT_ROOT", "main"]

_SPEC = ServerSpec(model="")
DEFAULT_PORT = _SPEC.port
DEFAULT_CONTEXT = _SPEC.context
DEFAULT_PARALLEL = _SPEC.parallel
DEFAULT_TIMEOUT = DEFAULT_TIMEOUT_S
# The per-user contexts `fit` tabulates unless told otherwise; named here so --help can
# say them without importing the module that measures.
FIT_PER_USER = (4096, 8192, 16384, 32768, 65536, 131072)

COMMANDS = Group(
    "ml-stack-serve",
    "See which model is being served on this machine, put one up, take it down.")
main = COMMANDS.run


def _lease_line(snapshot: ops.Snapshot) -> str:
    if not snapshot.recorded:
        return "none on record -- 'ml-stack-serve down' will not stop this server"
    pid = f"server pid {snapshot.pid}" if snapshot.pid else "server pid not recorded"
    if snapshot.owner_pid is None:
        return pid
    if snapshot.owner_pid == snapshot.pid:
        return f"{pid}, started by 'ml-stack-serve up'"
    if snapshot.holder_running:
        return f"{pid}, held by process {snapshot.owner_pid}"
    return (f"{pid}, orphaned -- the process that started it (pid {snapshot.owner_pid}) "
            "has gone; 'ml-stack-serve down --orphans' stops it")


def _verdict_line(snapshot: ops.Snapshot, model: str, parallel: int) -> str:
    seats = f" --parallel {parallel}" if int(parallel or 1) != 1 else ""
    ask = f"'ml-stack-serve up {model}{seats}'"
    if snapshot.verdict == "adopt":
        return f"{ask} would adopt this server"
    if snapshot.verdict == "refuse":
        return f"{ask} would be refused -- {snapshot.reason}"
    return f"{ask} would start its own server"


def _say_processes(as_json: bool) -> int:
    """``status --every``: every llama-server on this machine, leased or not."""
    from ml_stack.fleet.models import sized
    from ml_stack.hub import pretty_name

    got = ops.processes()
    if as_json:
        say(json.dumps({"serving": bool(got.found), "servers": list(got.found),
                        "foreign": list(got.foreign)}, indent=2))
        return 0 if got.found else 1
    if not got.found:
        say("no llama-server is running on this machine.")
        return 1
    for one in got.found:
        if one.get("defunct"):
            # a zombie holds no memory and answers no port; it is waiting to be reaped
            say(f"  pid {one['pid']}  defunct -- exited, not yet reaped; holds nothing")
            continue
        leased = "leased" if one["port"] in got.leased else "NOT leased -- nobody records it"
        rss = f"{one['rss'] / 2**30:.1f}G" if one["rss"] else "?"
        say(f"  :{one['port']}  pid {one['pid']}  {pretty_name(one['model']) or '?'}  "
            f"{rss} resident  {leased}  ({one['binary']})")
        cache = ops.cache_of(one["model"])
        if cache is not None:
            say(f"      cache  {cache[0]}  ({sized(cache[1])})")
        if one["port"] not in got.leased:
            say(f"    foreign -- pid {one['pid']}, not started by ml-stack; left alone")
    if got.strays:
        say(f"  {len(got.strays)} not leased: 'ml-stack-serve down --port N' stops one")
    return 0


@COMMANDS.command(
    "status", help="what is serving, what its draft head is keeping, and what a lease "
                   "would do",
    options=[
        option("port", default=DEFAULT_PORT,
               help=f"port to check besides the recorded ones (default: {DEFAULT_PORT})"),
        option("model", help="ask what leasing this model would do (default: whatever is "
                             "already serving)"),
        option("context", type=parse_context, default=DEFAULT_CONTEXT,
               help=f"the context that lease would ask for -- 32768, 256k, 1m "
                    f"(default: {DEFAULT_CONTEXT})"),
        option("parallel", default=DEFAULT_PARALLEL,
               help=f"the slots that lease would ask for (default: {DEFAULT_PARALLEL})"),
        flag("--every", action="store_true",
             help="every llama-server process on this machine, leased or not -- a stray "
                  "one holds memory a lease cannot see"),
        option("json", help="print one JSON object instead of the human listing"),
    ])
def cmd_status(args: argparse.Namespace) -> int:
    if getattr(args, "every", False):
        return _say_processes(bool(args.json))

    found = ops.status(port=args.port, model=args.model, context=args.context,
                       parallel=args.parallel)
    if args.json:
        say(json.dumps(
            {"serving": bool(found.servers) or bool(found.foreign),
             "ports_checked": list(found.ports),
             "servers": [asdict(s) for s in found.servers],
             "foreign": list(found.foreign)},
            indent=2))
        return 0 if (found.servers or found.foreign) else 1

    if not found.servers and not found.foreign:
        say("nothing is serving on port " + ", ".join(str(p) for p in found.ports) + ".")
        say(f"  'ml-stack-serve up <model>' would start one on port {args.port}.")
        return 1

    for snapshot in found.servers:
        quant = f"  ({snapshot.quant})" if snapshot.quant else ""
        say(snapshot.base_url)
        say(f"  model    {snapshot.model or 'not reported'}{quant}")
        say(f"  context  {snapshot.context if snapshot.context is not None else 'not reported'}"
            " per slot")
        say(f"  slots    {snapshot.slots if snapshot.slots is not None else 'not reported'}")
        say(f"  lease    {_lease_line(snapshot)}")
        if snapshot.load_s is not None:
            warm = f", warm-up {snapshot.warmup_s:.1f}s" if snapshot.warmup_s is not None else ""
            say(f"  loaded   in {snapshot.load_s:.1f}s{warm}")
        for line in _drafting_lines(snapshot.drafting):
            say(line)
        if snapshot.verdict:
            say("  " + _verdict_line(snapshot, args.model or snapshot.model or "<model>",
                                     args.parallel))
    for held in found.foreign:
        say(base_url_for(held["port"]))
        say(f"  foreign -- pid {held['pid']}, not started by ml-stack; left alone")
        say("  drafting " + (f"{held['draft']}, from its command line"
                             if held.get("draft") else
                             "no draft head -- every token is written by the model itself"))
    return 0


def _drafting_lines(drafting: ops.Drafting | None) -> list[str]:
    """What `status` says about a server's draft head: what it is, and how much is kept."""
    if drafting is None:
        return []
    if not drafting.loaded:
        return ["  drafting no draft head -- every token is written by the model itself"]
    named = [drafting.head or "a head the command line does not name"]
    if drafting.spec_type:
        named.append(drafting.spec_type)
    if drafting.ahead is not None:
        named.append(f"{drafting.ahead} token(s) ahead per pass")
    return [f"  drafting {', '.join(named)}", "           " + _kept_line(drafting)]


def _kept_line(drafting: ops.Drafting) -> str:
    """How much of the draft the model has kept since this server came up."""
    if not drafting.metrics:
        return ("acceptance unknown: this server was started without --metrics, so it "
                "reports no counters")
    counted = drafting.counted
    if counted is None:
        return "acceptance unknown: this server answered no counters"
    if not counted.drafts:
        return "nothing drafted yet since the server came up"
    return (f"{(counted.acceptance or 0) * 100:.1f}% of drafted tokens kept since the server "
            f"came up (higher is better), "
            f"{counted.tokens_per_draft or 0:.1f} tokens per verification pass")


def from_profile(args: argparse.Namespace, model: str,
                 workload: str = "") -> tuple[object | None, list[str]]:
    """Fill every flag ``up`` was not given from this model's profile for ``workload``.

    Only the startup half: the draft depth goes in as the server's default, and a caller
    that can override it per request sends its own. "Not given" is "still the parser's own
    default", which is the only thing argparse can be asked afterwards. Returns the profile
    (or None) and one line per field it filled.
    """
    from ml_stack.serve.profile import profile_for, resolved

    found = profile_for(model, workload=workload or ASK)
    if found is None:
        return None, []
    # One seat holds the whole measured cache unless --parallel was given, in which case
    # each of those seats gets what one measured seat got.
    seats = int(getattr(args, "parallel", DEFAULT_PARALLEL) or DEFAULT_PARALLEL)
    shape = found.shape(seats=seats, resolve=False)
    # The head is recorded by file name and llama-server needs a path. 'auto' is left
    # alone: `resolve_spec` answers it, and it has to know which binary will serve.
    head = found.draft
    if head and head.lower() != "auto":
        head = resolved(model, head, "", build=found.build)[0]
    # dest -> the value the profile would have, for the flags whose default `up` defines
    wanted = {
        "context": shape.context,
        "parallel": max(1, seats),
        "build": found.build,
        "draft": head,
        "spec": found.spec_type,
        "spec_n_max": found.spec_draft_max,
        "kv": found.cache_type,
        "draft_kv": found.draft_cache_type,
        "mmproj": found.mmproj,
        "reasoning_budget": found.reasoning_budget,
    }
    defaults = {"context": DEFAULT_CONTEXT, "parallel": DEFAULT_PARALLEL, "build": "",
                "draft": "", "spec": "", "spec_n_max": None, "kv": "", "draft_kv": "",
                "mmproj": "", "reasoning_budget": None}
    took: list[str] = []
    for dest, value in wanted.items():
        if value in (None, "", ()) or getattr(args, dest, None) != defaults[dest]:
            continue
        setattr(args, dest, value)
        took.append(f"{dest} {value}")
    if found.extra_args:
        took.append(" ".join(found.extra_args))
    if shape.note:
        took.append(shape.note)
    return found, took


_EVENT_LINES = {
    "loading": lambda e: f"loading {e.get('model', '')} ({e.get('seats', 1)} seat(s))",
    "ready": lambda e: (
        f"ready in {e['load_s']:.1f}s" if e.get("load_s") is not None else "ready"),
    "escalating": lambda e: (
        f"escalating {e.get('from_seats')} -> {e.get('to_seats')} seat(s) by "
        f"{e.get('mode')} -- {e.get('reason', '')}"),
    "saving": lambda e: f"saving slot {e.get('slot')} ({e.get('tokens', 0):,} tokens)",
    "summarizing": lambda e: f"summarising slot {e.get('slot')} ({e.get('tokens', 0):,} tokens)",
    "summarized": lambda e: (
        f"summarised slot {e.get('slot')} to {e.get('tokens', 0):,} words: "
        f"{e.get('summary', '')}"),
    "stopping": lambda e: "stopping the old server",
    "restoring": lambda e: f"restoring slot {e.get('slot')} ({e.get('mode', 'cache')})",
    "done": lambda e: f"now serving {e.get('seats')} seat(s)",
}


def _print_event(event: dict) -> None:
    """One line per step, as it happens -- ``up``'s and ``escalate``'s own progress."""
    said = _EVENT_LINES.get(str(event.get("event")))
    warn(said(event) if said else str(event.get("event")))


def _asked_spec(args: argparse.Namespace, model: str, extra: tuple[str, ...]) -> ServerSpec:
    """The spec ``up`` was asked for, before 'auto' is answered."""
    kv = str(getattr(args, "kv", "") or "")
    draft_k, draft_v = split_cache_type(str(getattr(args, "draft_kv", "") or ""))
    return ServerSpec(
        model=model, port=args.port, context=args.context, parallel=args.parallel,
        draft=str(getattr(args, "draft", "") or "") or None,
        mmproj=str(getattr(args, "mmproj", "") or "") or None,
        spec_type=str(getattr(args, "spec", "") or ""),
        cache_type_k=kv, cache_type_v=kv,
        spec_draft_type_k=draft_k, spec_draft_type_v=draft_v,
        kv_unified=getattr(args, "kv_unified", None),
        embedding=bool(getattr(args, "embedding", False)),
        spec_draft_max=getattr(args, "spec_n_max", None),
        spec_draft_ngl=getattr(args, "draft_ngl", None),
        lookup_dynamic=str(getattr(args, "lookup_cache", "") or "") or None,
        override_tensor=tuple(getattr(args, "on_cpu", []) or ()),
        reasoning_budget=getattr(args, "reasoning_budget", None),
        # llama-server's own flags, which only a profile carries: `up` has no flag of its
        # own for `-ub 2048`
        extra_args=extra,
        cpu_moe=bool(getattr(args, "cpu_moe", False)))


@COMMANDS.command(
    "up", help="serve a model, or adopt the one already serving it",
    options=[
        flag("model", help="path to a .gguf file, or hf:owner/repo/file.gguf"),
        option("port", default=DEFAULT_PORT,
               help=f"port to serve on (default: {DEFAULT_PORT})"),
        option("context", type=parse_context, default=DEFAULT_CONTEXT,
               help=f"tokens across all slots -- 32768, 256k, 1m (default: "
                    f"{DEFAULT_CONTEXT}). Beyond what the model trained at, YaRN is "
                    "turned on by itself; it scales every position, so shorter "
                    "conversations on this server pay for it too"),
        option("parallel", default=DEFAULT_PARALLEL,
               help=f"slots to serve at once (default: {DEFAULT_PARALLEL})"),
        flag("--escalate", action="store_true",
             help="when a server is already up with fewer slots than --parallel asks for, "
                  "grow (or split, or summarise and split) it rather than refusing -- "
                  "keeps every live conversation"),
        option("timeout",
               help="seconds to wait for it to load (default: scales with the weights on "
                    f"disk -- 60s + 1.5s/GB, floor {DEFAULT_TIMEOUT:.0f}s)"),
        option("json", help="print one JSON object instead of the human line"),
        flag("--preflight-only", action="store_true",
             help="run every check a load would run -- shards present, architecture this "
                  "build reads, an estimate against what this machine may use, every flag "
                  "the build accepts -- and print the report without starting or adopting "
                  "anything. Exits 0 or 1"),
        flag("--root", default=DEFAULT_ROOT,
             help=f"the fleet root whose beacon to announce in, when there is one "
                  f"(default: {DEFAULT_ROOT})"),
        flag("--binary", default="", metavar="PATH",
             help="the llama-server to run, when the one on PATH cannot read this model. A "
                  "release lags master by an architecture or two: gemma-4 and qwen3moe are "
                  "in the current release, qwen4exp is not, so Qwen3.8-Flash-Next needs a "
                  "build from master and says 'unknown model architecture' without one"),
        flag("--build", default="", metavar="NAME",
             help="serve with a named build 'ml-stack-serve build --name NAME' made -- a "
                  "fork kept beside 'current' rather than replacing it, e.g. a fork whose "
                  "fixes have not reached mainline yet. Ignored if --binary is also given"),
        flag("--mmproj", default="", metavar="PATH_OR_AUTO",
             help="the vision projector, so the model can read a picture -- a path, an hf: "
                  "reference, or 'auto' to take the most precise one shipped with the "
                  "weights. An hf: model already pulls a projector by itself, so 'auto' is "
                  "for choosing a better one than it would: a projector is a fraction of "
                  "the weights and carries all of the seeing, so quantising it is a false "
                  "economy"),
        flag("--spec", default="", metavar="TYPE",
             help="how to guess ahead: an ngram-* kind needs no second model at all, "
                  "proposing tokens it has already seen in the prompt, which suits work "
                  "that copies from its context and costs no memory. ngram-simple, "
                  "ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache, or a draft-* kind "
                  "with --draft. Left unset, the server decides"),
        flag("--spec-n-max", type=int, default=None, metavar="N",
             help="tokens guessed ahead each step (server default 3)"),
        flag("--on-cpu", action="append", default=[], metavar="PATTERN=BUFFER",
             help="keep tensors matching a pattern off the GPU, e.g. "
                  "'per_layer_token_embd=CPU' for Qwen3.8-Flash-Next's 27G n-gram table on "
                  "a discrete GPU whose VRAM it would not fit beside the weights. Never on "
                  "unified memory (a Mac): there the CPU and GPU halves are the same RAM, "
                  "the build already places the table where a gather is cheapest, and "
                  "forcing it buys nothing. ml-stack sets none of these by itself. "
                  "Repeatable. Read the tensor names from the model rather than guessing "
                  "at the pattern"),
        flag("--cpu-moe", action="store_true",
             help="keep every Mixture-of-Experts weight on the CPU, which is how a 35B with "
                  "3B active fits a machine that could not hold it all"),
        flag("--lookup-cache", default="", metavar="FILE",
             help="an n-gram cache kept on disk and updated as it generates, so what was "
                  "learnt answering one question speculates the next. Only the ngram-cache "
                  "kind uses it; the other ngram kinds look up the prompt itself and keep "
                  "nothing"),
        flag("--draft-ngl", type=int, default=None, metavar="N",
             help="layers of the draft model to put on the GPU. Without it the draft runs "
                  "where the server puts it by default, which can be the CPU -- and a "
                  "draft slower than the model it is guessing for is a loss"),
        flag("--embedding", action="store_true",
             help="serve an embedding model (llama-server --embedding), the way the graph's "
                  "vectors and the thread's recall want one"),
        flag("--kv", default="q8_0", metavar="TYPE",
             help="what the KV cache is stored as (default q8_0: measured 2026-09-02 on "
                  "Flash-Next, F1 unchanged, faster, half the cache; the recurrent state is "
                  "not the KV and stays); f16 is the full-size cache"),
        flag("--kv-unified", action=argparse.BooleanOptionalAction, default=None,
             help="one cache pool for every slot, masked per sequence, rather than a cache "
                  "per slot; --no-kv-unified asks for the latter outright. Left unset, the "
                  "build decides"),
        flag("--draft-kv", default="", metavar="TYPE",
             help="what the draft head's own KV cache is stored as. It is a second cache, "
                  "not the one --kv sets, and llama.cpp stores it at full size whatever "
                  "--kv says. The head only proposes and the model checks every token, so "
                  "a smaller cache here costs acceptance and never correctness. The types "
                  "this build takes are in its --help; f16 is its default. One value sets "
                  "both halves, K/V sets them apart"),
        flag("--draft", default="", metavar="MODEL_OR_AUTO",
             help="a small model to guess ahead, which the large one checks in one pass -- "
                  "a path, an hf: reference, or 'auto' to use the draft head shipped beside "
                  "the weights (the mtp- file in a QAT repository)"),
        flag("--reasoning-budget", type=int, default=None, metavar="TOKENS",
             dest="reasoning_budget",
             help="how many tokens a turn may think for before it is made to answer; 0 "
                  "turns the thinking off. A ceiling on n_predict cuts the answer instead, "
                  "which is the wrong end"),
        flag("--profile", action="store_true",
             help="fill every flag not given from this model's measured profile -- the "
                  "build, head, cache, thinking and llama-server flags that answered best "
                  "(`ml-stack-serve profile MODEL` prints it). A flag given wins over the "
                  "record"),
        flag("--for", dest="workload", default=ASK, choices=sorted(WORKLOADS),
             metavar="WORKLOAD",
             help=f"which workload the profile is for: "
                  f"{'; '.join(f'{k}, {v}' for k, v in WORKLOADS.items())} "
                  f"(default: {ASK})"),
        flag("--anyway", action="store_true",
             help="start the server even while a measurement holds this card; both its "
                  "timings and anything measured through this server are then two models "
                  "sharing a GPU"),
    ])
def cmd_up(args: argparse.Namespace) -> int:
    from ml_stack.serve.backend import UnknownFlag

    model = str(hub.located(args.model) or args.model)
    if model != str(args.model):
        warn(f"resolved {args.model} -> {model}")

    profile = None
    if getattr(args, "profile", False):
        wanted = str(getattr(args, "workload", "") or ASK)
        profile, took = from_profile(args, model, wanted)
        if profile is None:
            warn(f"no measured profile for {model.rsplit('/', 1)[-1]} doing {wanted}; "
                 "serving as asked -- `ml-stack-bench report --profile` writes one from "
                 "the store")
        else:
            warn(f"profile {profile.model} for {profile.workload}: "
                 + (", ".join(took) or "nothing left to fill"))
            if profile.note:
                warn(f"  {profile.note}")

    chosen = str(getattr(args, "binary", "") or "")
    manager = ops.manager_for(chosen, str(getattr(args, "build", "") or ""))
    extra = tuple(profile.extra_args) if profile is not None else ()
    resolved = ops.resolve_spec(_asked_spec(args, model, extra), manager=manager)
    for note in resolved.notes:
        warn(note)
    spec = resolved.spec

    if getattr(args, "preflight_only", False):
        try:
            report = ops.preflight(spec, manager=manager)
        except (BinaryNotFound, OSError) as exc:
            warn(f"error: {exc}")
            return 2
        say(report.said())
        return 0 if report.ok else 1

    try:
        started = ops.up(spec, manager=manager, timeout=args.timeout,
                         escalate=bool(getattr(args, "escalate", False)),
                         anyway=bool(getattr(args, "anyway", False)),
                         root=args.root, say=warn, on_event=_print_event)
    except UnknownFlag as exc:
        # Refused before the load, not at the end of it: the build was asked what it
        # accepts and the answer is printed one flag per line, with the nearest it has.
        warn(exc)
        return 2
    except (ServerFailed, BinaryNotFound, OSError) as exc:
        warn(f"error: {exc}")
        return 2

    info, told = started.info, started.announced
    if args.json:
        say(json.dumps({"base_url": info.base_url, "port": info.port, "pid": info.pid,
                        "adopted": info.adopted, "model": str(spec.model),
                        "context": spec.context, "parallel": spec.parallel,
                        "draft": str(spec.draft or ""),
                        "draft_cache_type": said_cache(spec.spec_draft_type_k,
                                                       spec.spec_draft_type_v),
                        "announced": told.startswith("announced")}, indent=2))
        return 0

    where = f" (pid {info.pid})" if info.pid else ""
    say(f"{'adopted' if info.adopted else 'started'} {info.base_url}{where}")
    if chosen:
        say(f"  with {chosen}")
    if spec.draft:
        say(f"  guessing ahead with {str(spec.draft).rsplit('/', 1)[-1]}")
    if spec.spec_draft_type_k or spec.spec_draft_type_v:
        say(f"  the draft's own cache stored as {said_cache(spec.spec_draft_type_k, spec.spec_draft_type_v)}")
    if spec.mmproj:
        say(f"  reading pictures with {str(spec.mmproj).rsplit('/', 1)[-1]}")
    if spec.spec_type:
        say(f"  guessing ahead by {spec.spec_type}")
    if told:
        say(f"  {told}")
    return 0


@COMMANDS.command(
    "profile", help="the shape a model measured best in for each workload: what to serve "
                    "it with, and how to ask it",
    options=[
        flag("model", nargs="?", default="",
             help="a model file, path or hf: reference; every record with none"),
        flag("--for", dest="workload", default="", choices=sorted(WORKLOADS),
             metavar="WORKLOAD",
             help=f"one workload rather than all of them: "
                  f"{'; '.join(f'{k}, {v}' for k, v in WORKLOADS.items())}"),
        option("json", help="the records as JSON, exactly as they are kept"),
    ])
def cmd_profile(args: argparse.Namespace) -> int:
    """``ml-stack-serve profile [MODEL] [--for WORKLOAD]`` -- the shape a model measured
    best in, one block per workload.

    Exit 1 when a model was named and nothing has measured it for any workload.
    """
    from ml_stack.serve.profile import said

    model = str(getattr(args, "model", "") or "")
    wanted = str(getattr(args, "workload", "") or "")
    try:
        chosen = ops.shapes(model, workload=wanted)
    except Refused as no:
        warn(no.lines[0])
        return 1
    if getattr(args, "json", False):
        say(json.dumps([one.as_dict() for one in chosen], indent=2))
        return 0
    if not chosen:
        say("no model has a measured shape yet. `ml-stack-bench sweep` measures one and "
            "`ml-stack-bench report --profile` writes the record.")
        return 0
    say("\n\n".join(said(one) for one in chosen))
    if model and not wanted:
        held = {one.workload for one in chosen}
        for named in (w for w in WORKLOADS if w not in held):
            say(f"\nnothing has measured {model.rsplit('/', 1)[-1]} for {named} "
                f"({WORKLOADS[named]}); `--for {named}` serves the {ASK} record and "
                f"says so")
    return 0


def _fit_ui() -> int:
    """``fit --ui``: the interactive page, on loopback, until Ctrl-C."""
    from ml_stack.platform import open_path

    server = ops.fit_page()
    where = f"http://127.0.0.1:{server.server_port}/ui/fit"
    warn(f"the fit page is at {where}\n(loopback only; Ctrl-C to stop)")
    warn(f"opened with {open_path(where)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        warn("\nstopped")
    finally:
        server.server_close()
    return 0


def _measure_each(args: argparse.Namespace, *, room: int) -> int:
    """Serve each named model once and record what it allocated. Returns an exit code."""
    from ml_stack.serve.backend import LlamaServerBackend

    binary = str(getattr(args, "binary", "") or "")
    build_name = str(getattr(args, "build", "") or "")
    backend = (LlamaServerBackend(binary=binary or None, build=build_name or None)
               if (binary or build_name) else LlamaServerBackend())
    kv = str(getattr(args, "kv", "") or "")
    draft_k, draft_v = split_cache_type(str(getattr(args, "draft_kv", "") or ""))
    spec = ServerSpec(model="", port=args.port, context=args.context,
                      parallel=max(1, int(getattr(args, "parallel", 1) or 1)),
                      cache_type_k=kv, cache_type_v=kv,
                      spec_draft_type_k=draft_k, spec_draft_type_v=draft_v, warmup=False)
    try:
        recorded = ops.measure(
            list(args.model), spec=spec, backend=backend, timeout=args.timeout, room=room,
            draft=str(getattr(args, "draft", "") or ""),
            resident=(int(getattr(args, "resident_peak", 0) or 0),
                      int(getattr(args, "resident_after", 0) or 0)))
    except Refused as no:
        warn(f"error: {no.lines[0]}")
        return 2
    for one in recorded:
        warn(f"measured {one.model}: {one.said}")
        warn(f"  recorded in {one.where}")
    return 0


@COMMANDS.command(
    "fit", help="how many people fit at a given context, from measured KV numbers",
    options=[
        flag("model", nargs="*",
             help="which measured models to report (a bare name, a path, or an hf: "
                  "reference). Default: every model that has been measured"),
        flag("--measure", action="store_true",
             help="serve each named model once at -lv 4, read what llama.cpp says it "
                  "allocated -- the base cache per token, the sliding-window and recurrent "
                  "caches per sequence, the compute buffers -- and record it. This is the "
                  "only way the numbers get in: a formula over the GGUF header counts every "
                  "layer as full attention, and gemma4 (18 layers share a cache, the rest "
                  "slide), gpt-oss (every other layer slides) and qwen4exp (three layers in "
                  "four are recurrent) each disagree with that differently"),
        flag("--tensors", action="store_true",
             help="what the model file is made of, from its GGUF header alone -- no server, "
                  "no GPU. The largest tensors with their type and shape, the totals per "
                  "tensor type, and the totals per role (gathered table, experts, "
                  "attention, embedding). This is the answer to 'the file is 103.7G and the "
                  "process holds 90G, what is the rest': a gathered lookup table -- "
                  "Flash-Next's `per_layer_token_embd.weight` is one 26.8G n-gram table -- "
                  "is paged a row at a time and most of it never becomes resident"),
        flag("--resident-peak", dest="resident_peak", type=int, default=0, metavar="BYTES",
             help="with --measure: the peak RSS the served process actually reached during "
                  "a real run (the bench reports one). Recorded as a weights figure, with "
                  "the compute buffers and that run's own caches taken back off, and used "
                  "as the intercept in preference to anything read off the load log -- a "
                  "paged lookup table only becomes resident as it is walked, so this is the "
                  "one number the arithmetic cannot derive"),
        flag("--resident-after", dest="resident_after", type=int, default=0, metavar="N",
             help="with --resident-peak: how many questions had been answered when that "
                  "peak was taken. A resident figure with no run length beside it cannot be "
                  "argued with -- a table paged a row at a time reads low after two "
                  "questions and high after two hundred"),
        flag("--draft", default="", metavar="MODEL_OR_AUTO",
             help="measure it with a draft head as well -- a path, an hf: reference, or "
                  "'auto'. A draft *model* keeps its own cache at the same context, which "
                  "is the real cost of drafting with one"),
        flag("--kv", default="q8_0", metavar="TYPE",
             help="measure with the main model's KV cache stored as this: q8_0 (the "
                  "default), f16, q4_0. A record is kept per cache type, because that is "
                  "what changes the per-token cost"),
        flag("--draft-kv", default="", metavar="TYPE",
             help="with --draft: measure with the head's own KV cache stored as this. It "
                  "is a second cache the head keeps at the same context, and llama.cpp "
                  "stores it at full size whatever --kv says, so this is where a drafted "
                  "model's cache cost is decided"),
        flag("--room", action="append", default=[], metavar="SIZE",
             help="ask about a machine with this much memory instead of this one -- 24G, "
                  "24576M, or a plain number of bytes. Default: what `ml-stack-serve "
                  "memory` says a model may use here. Repeatable: the listing answers for "
                  "the first, and --plot draws every one of them, solid then dashed, so a "
                  "laptop and a card can be compared in the same picture"),
        flag("--per-user", type=int, action="append", dest="per_user", default=[],
             metavar="N",
             help="a per-user context to put in the table. Repeatable; default "
                  f"{', '.join(str(n) for n in FIT_PER_USER)}"),
        option("parallel", default=1, metavar="N",
               help="also say the longest context N users could each be given (default: 1, "
                    "which is the line every block prints anyway). With --measure, the "
                    "slots the model is served on"),
        flag("--plot", default="", metavar="FILE.png",
             help="draw it: two panels, one figure -- how many users fit against the "
                  "context each gets, and what the memory costs as they arrive. The second "
                  "is the one worth having: a large model with a small cache starts higher "
                  "and climbs more slowly than a small model with a fat one, and the "
                  "picture is where they cross. .png, .svg or .pdf; needs matplotlib"),
        flag("--open", action="store_true",
             help="with --plot: open the picture when it is drawn"),
        flag("--ui", action="store_true",
             help="put the same two panels up as a page you can move: a room slider, a "
                  "per-user context slider, a users slider and a model per checkbox, "
                  "redrawn as you drag. Serves on loopback, opens a browser at it, and "
                  "stays up until Ctrl-C. The fleet app shows the same page under Fit"),
        flag("--at", type=int, default=32768, metavar="N",
             help="the per-user context the second panel charges at (default: 32768)"),
        flag("--md", action="store_true",
             help="print Markdown rather than the plain listing"),
        flag("--write", default="", metavar="FILE",
             help="write the Markdown for every record to a file -- at this machine's room, "
                  "and at --room's as a second section"),
        option("context", type=parse_context, default=32768, metavar="N",
               help="the context to measure at -- 32768, 256k, 1m (default: 32768). The "
                    "per-token cost does not depend on it; a long one just measures it "
                    "precisely"),
        option("port", default=DEFAULT_PORT,
               help=f"the port to measure on (default: {DEFAULT_PORT})"),
        option("timeout",
               help="seconds to wait for the measured load (default: scales with the "
                    "weights on disk)"),
        flag("--binary", default="", metavar="PATH",
             help="the llama-server to measure with, when the one on PATH cannot read this "
                  "model"),
        flag("--build", default="", metavar="NAME",
             help="measure with a named build, the way `up --build NAME` serves with one"),
    ])
def cmd_fit(args: argparse.Namespace) -> int:
    """``ml-stack-serve fit`` -- how many people fit on this machine, at what context.

    Reads the measured records rather than a formula. `--measure` serves a model once at
    `-lv 4` and writes what llama.cpp says it allocated; `--tensors` reads the GGUF header
    and needs nothing running.
    """
    from ml_stack.hub import room as machine_room
    from ml_stack.serve import fit as fit_mod

    if getattr(args, "ui", False):
        return _fit_ui()

    named = [str(m) for m in (getattr(args, "model", None) or [])]
    if getattr(args, "tensors", False):
        # A header read, not a load: no server, no GPU, no lease.
        if not named:
            warn("error: --tensors needs a model to look inside")
            return 2
        try:
            for said in ops.tensors(named):
                say(said)
        except Refused as no:
            warn(f"error: {no.lines[0]}")
            return 2
        return 0

    asked_rooms: list[int] = []
    for said in (getattr(args, "room", None) or []):
        try:
            asked_rooms.append(fit_mod.parse_room(said))
        except ValueError as exc:
            warn(f"error: {exc}")
            return 2
    # The listing answers for one machine -- the first room named, or this one. The chart
    # draws every room asked for.
    room = asked_rooms[0] if asked_rooms else machine_room()

    per_user = [int(n) for n in (getattr(args, "per_user", None) or [])]
    wanted = [Path(m).name.lower() for m in named]

    if getattr(args, "measure", False):
        if not wanted:
            warn("error: --measure needs a model to measure")
            return 2
        code = _measure_each(args, room=room)
        if code:
            return code

    rows = fit_mod.records(room=room)
    if wanted:
        rows = [r for r in rows if r.model.lower() in wanted
                or any(w in r.model.lower() for w in wanted)]
        if not rows:
            warn("nothing measured for " + ", ".join(wanted)
                 + " -- `ml-stack-serve fit MODEL --measure` serves it once and records "
                   "what it allocated.")
            return 1

    contexts = per_user or list(fit_mod.DEFAULT_PER_USER)
    say(fit_mod.render(rows, contexts, room, bool(getattr(args, "md", False))))

    parallel = int(getattr(args, "parallel", 1) or 1)
    if parallel > 1:
        say()
        for row in rows:
            say(f"{row.model}: {parallel} users fit at "
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
            warn(f"error: {exc}")
            return 2
        warn(f"\ndrew {drawn}")
        if getattr(args, "open", False):
            from ml_stack.platform import open_path

            warn(f"opened with {open_path(drawn)}")

    where = str(getattr(args, "write", "") or "")
    if where:
        home.expand(where).write_text(
            ops.fit_markdown(contexts, asked_rooms, here=machine_room(), drawn=drawn),
            encoding="utf-8")
        warn(f"\nwrote {where}")
    return 0


@COMMANDS.command(
    "memory", help="how much a model may use here, and whether that survives a reboot",
    options=[
        flag("--persist", nargs="?", const="", default=None, metavar="MB",
             help="write a boot-time setting for the wiring limit; the megabytes default "
                  "to whatever is set now"),
        flag("--write", default="", metavar="FILE",
             help="where to write it (default: ./stack.ml.wired-limit.plist)"),
        flag("--limit", type=int, default=0, metavar="MB",
             help="preview: what the rest of the machine would have under this wiring "
                  "limit, against what it holds now"),
    ])
def cmd_memory(args: argparse.Namespace) -> int:
    """``ml-stack-serve memory`` -- what this machine will let a model use, and for how long.

    On unified memory the ceiling that matters is `iogpu.wired_limit_mb`: a runtime setting
    that goes back to the default on every reboot.
    """
    machine = ops.memory()
    total, now = machine.total, machine.room
    if not now:
        say("this machine does not report a wiring limit; nothing to do here")
        return 0

    say(f"a model may use about {human_bytes(now)}"
        + (f" of {human_bytes(total)} installed" if total else ""))
    if total:
        default = int(total * 0.75)
        if now > default * 1.02:
            say(f"  raised from the ~{human_bytes(default)} default -- and **not** kept: this "
                f"resets on reboot")
        else:
            say(f"  this is the default share; {human_bytes(total)} is installed")

    held = machine.held
    if held:
        say(f"\nright now: {human_bytes(held['used'])} used of {human_bytes(held['total'])} "
            f"({human_bytes(held['wired'])} wired, {human_bytes(held['free'])} free)")
        servers = held["servers"]
        others = held["others"]
        say(f"  llama-server(s): {human_bytes(servers)}; everything else: {human_bytes(others)}"
            + (f" -- {', '.join(held['largest'])}" if held["largest"] else ""))
        headroom = int(total) - int(now) if total else 0
        if total:
            say(f"  the limit leaves {human_bytes(headroom)} for everything else; the rest of "
                f"the machine holds {human_bytes(others)} now"
                + (" -- room to raise it" if others < headroom * 0.6
                   else " -- close to it; raising it means swapping when a model fills it"))
        want_mb = int(getattr(args, "limit", 0) or 0)
        if want_mb and total:
            left = int(total) - want_mb * 1024 * 1024
            say(f"  at {want_mb} MB the rest of the machine would have {human_bytes(left)}"
                + (" -- less than it holds now" if left < others else ""))
    want = args.persist
    if want is None:
        say("\n  ml-stack-serve memory --persist [MB]   to write a boot-time setting")
        return 0

    mb = int(want) if want else now // (1024 * 1024)
    where = ops.write_plist(Path(args.write or "./stack.ml.wired-limit.plist"), mb)
    say(f"\nwrote {where} -- it sets iogpu.wired_limit_mb={mb} at every boot.")
    say("Installing it needs root, so it is left to you:")
    say(f"  sudo cp {where} /Library/LaunchDaemons/stack.ml.wired-limit.plist")
    say("  sudo chown root:wheel /Library/LaunchDaemons/stack.ml.wired-limit.plist")
    say("  sudo launchctl load -w /Library/LaunchDaemons/stack.ml.wired-limit.plist")
    say(f"\nOr for this boot only:  sudo sysctl -w iogpu.wired_limit_mb={mb}")
    say("\nLeave headroom: everything else on the machine shares this memory, and a "
        "machine that wires all of it stops being usable before it stops serving.")
    return 0


@COMMANDS.command(
    "limits", help="how much of this machine ml-stack may take",
    options=[
        flag("--memory", default="", metavar="SIZE",
             help="the most a model and its caches may use here -- 90G, 24576M, a plain "
                  "number of bytes. Every preflight, fit and lease reads it"),
        flag("--servers", type=int, default=None, metavar="N",
             help="the most model servers to run at once"),
        flag("--seats", type=int, default=None, metavar="N",
             help="the most conversations one server may hold"),
        flag("--idle", default="", metavar="TIME",
             help="stop a server unused for this long -- 10m, 1h, 600. "
                  "`ml-stack-serve reclaim` and the fleet daemon act on it"),
        flag("--clear", action="store_true", help="take every limit off this machine"),
    ])
def cmd_limits(args: argparse.Namespace) -> int:
    """``ml-stack-serve limits`` -- how much of this machine ml-stack may take.

    Set nothing and it prints what is set; every limit is off until somebody sets one.
    """
    try:
        limits = ops.limits(memory_size=args.memory, servers=args.servers,
                            seats=args.seats, idle=args.idle, clear=bool(args.clear))
    except Refused as no:
        warn(f"error: {no.lines[0]}")
        return 2

    if args.clear:
        say(f"every limit is off; {limits.where} says so")
        return 0

    if limits.lines:
        say(f"what ml-stack may take here ({limits.where}):")
        for line in limits.lines:
            say(f"  {line}")
    else:
        say("nothing is limited here; ml-stack may use whatever this machine allows")
    if limits.machine:
        say(f"\nthis machine allows {human_bytes(limits.machine)}; a model may use "
            f"{human_bytes(limits.room)}")
    return 0


@COMMANDS.command(
    "reclaim", help="stop the servers nobody is using",
    options=[
        flag("--idle", default="", metavar="TIME",
             help="how long unused is idle (default: what `limits --idle` set)"),
        flag("--settle", default="30", metavar="TIME",
             help="how long to watch before deciding, on top of what earlier looks found "
                  "(default: %(default)ss)"),
        flag("--watch", action="store_true",
             help="keep looking rather than making one pass"),
        flag("--every", default="60", metavar="TIME",
             help="with --watch: how often to look (default: %(default)ss)"),
    ])
def cmd_reclaim(args: argparse.Namespace) -> int:
    """``ml-stack-serve reclaim`` -- stop the servers nobody is using.

    Idleness is asked of each server and what the looks found is kept, so a pass adds
    `--settle` seconds of its own watching to whatever the daemon has already seen.
    """
    from ml_stack.bench.history import parse_duration
    from ml_stack.serve.reclaim import watching

    try:
        older, watcher = ops.reclaim(idle=args.idle)
    except Refused as no:
        warn(no.lines[0])
        return 2
    if args.watch:
        every = parse_duration(args.every) or 60.0
        say(f"watching every {every:.0f}s, reclaiming after {older:.0f}s idle; "
            "Ctrl-C to stop", flush=True)
        try:
            with watching(older_than=older, every=every, idleness=watcher, say=print):
                while True:
                    time.sleep(3600)
        except KeyboardInterrupt:
            return 0
    stopped = ops.reclaim_once(older, watcher, settle=parse_duration(args.settle) or 0.0)
    if not stopped:
        say("nothing has been idle that long")
    return 0


@COMMANDS.command(
    "down", help="stop a server started on this machine",
    options=[
        option("port", default=DEFAULT_PORT,
               help=f"port of the server to stop (default: {DEFAULT_PORT})"),
        flag("--orphans", action="store_true",
             help="instead of a port: stop every recorded server whose leasing process has "
                  "gone, and nothing else"),
        flag("--root", default=DEFAULT_ROOT,
             help=f"the fleet root to withdraw it from (default: {DEFAULT_ROOT})"),
    ])
def cmd_down(args: argparse.Namespace) -> int:
    if getattr(args, "orphans", False):
        found = ops.orphans(root=args.root)
        if not found:
            say("no orphaned server on record.")
            return 0
        for stopped, said in found:
            if said:
                warn(said)
            say(f"stopped {stopped.base_url} (pid {stopped.pid}), orphaned by pid "
                f"{stopped.owner_pid}")
        return 0

    try:
        stopped, said = ops.down(args.port, root=args.root)
    except Refused as no:
        if len(no.lines) == 1:
            say(no.lines[0])
            return 1
        warn(f"error: {no.lines[0]}")
        for line in no.lines[1:]:
            warn(line)
        return 2
    if said:
        warn(said)
    if stopped.was_running:
        say(f"stopped {stopped.base_url} (pid {stopped.pid})")
    else:
        say(f"nothing was running on port {stopped.port}; removed the record")
    return 0


@COMMANDS.command(
    "escalate",
    help="grow the seats a running server holds, keeping every live conversation",
    options=[
        option("port", default=DEFAULT_PORT,
               help=f"port of the server to grow (default: {DEFAULT_PORT})"),
        flag("--add", type=int, default=1, metavar="N",
             help="how many more seats to ask for (default: 1)"),
        flag("--slot-save-path", default="", metavar="PATH",
             help="where the running server saves slots, if it was not started through 'up "
                  "--escalate' (default: this manager's own)"),
        option("timeout",
               help="seconds to wait for the relaunch to load (default: scales with the "
                    "weights on disk)"),
        flag("--room", default="", metavar="SIZE",
             help="judge the grow-vs-split decision against this much room (e.g. 24G) "
                  "instead of what this machine actually has"),
        option("json", help="print one JSON object instead of the human lines"),
    ])
def cmd_escalate(args: argparse.Namespace) -> int:
    """``ml-stack-serve escalate`` -- grow a running server's seats in place."""
    try:
        info = ops.escalate(args.port, add=args.add,
                            room=str(getattr(args, "room", "") or ""),
                            timeout=args.timeout,
                            slot_save_path=str(getattr(args, "slot_save_path", "") or ""),
                            on_event=_print_event)
    except (Refused, ServerFailed, ValueError) as exc:
        warn(f"error: {exc}")
        return 2

    if args.json:
        say(json.dumps({"base_url": info.base_url, "port": info.port, "pid": info.pid}))
        return 0
    say(f"port {args.port}: now at {info.base_url}" + (f" (pid {info.pid})" if info.pid else ""))
    return 0


@COMMANDS.command(
    "build",
    help="build llama-server from llama.cpp's own master (or download the newest release), "
         "and switch to it once it is verified",
    options=[
        flag("--from", dest="source_kind", default="", choices=["source", "release"],
             help="'source' compiles master with cmake, 'release' downloads the newest "
                  "GitHub release with an asset for this machine. Default: source when a "
                  "compiler is on PATH, release otherwise"),
        flag("--commit", default="", metavar="SHA",
             help="build this commit instead of master's tip (--from source only)"),
        flag("--jobs", type=int, default=0, metavar="N",
             help="parallel compile jobs (default: every core)"),
        flag("--source", default="", metavar="DIR",
             help="reuse a checkout here instead of cloning/updating the managed one"),
        flag("--force", action="store_true",
             help="rebuild or redownload even if this commit/release is already installed"),
        flag("--check", action="store_true",
             help="report the installed build's commit, age and architectures -- builds "
                  "nothing"),
        flag("--rollback", action="store_true",
             help="point 'current' back at the previous verified build"),
        flag("--persist", action="store_true",
             help="install a weekly refresh (a LaunchAgent on macOS, a Scheduled Task on "
                  "Windows) that reruns this on its own"),
        flag("--adopt", default="", metavar="DIR",
             help="register a flat build directory that already exists -- a hand-built "
                  "binary, or a release zip unpacked by hand -- as a managed build, verify "
                  "it, and switch to it now, without compiling or downloading anything"),
        flag("--repo", default="", metavar="OWNER/REPO",
             help="build a fork instead of ggml-org/llama.cpp's own master -- combine with "
                  "--name to keep it beside 'current' instead of replacing it, e.g. --repo "
                  "unslothai/llama.cpp --name unsloth"),
        flag("--ref", default="", metavar="TAG_OR_BRANCH_OR_SHA",
             help="the fork's ref to build (--from source, with --repo; default: its "
                  "default branch's tip)"),
        flag("--tag", default="", metavar="TAG",
             help="the fork's release tag to download (--from release, with --repo; "
                  "default: the newest release with a matching asset)"),
        flag("--name", default="", metavar="NAME",
             help="keep this build at ~/.ml-stack/llama.cpp/named/NAME instead of replacing "
                  "'current' -- requires --repo. Select it with 'ml-stack-serve up --build "
                  "NAME' or $MLSTACK_LLAMA_BUILD=NAME"),
        flag("--list", action="store_true",
             help="show 'current' and every named build, with commit, age and repo -- "
                  "builds nothing"),
    ])
def cmd_build(args: argparse.Namespace) -> int:
    return build.cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
