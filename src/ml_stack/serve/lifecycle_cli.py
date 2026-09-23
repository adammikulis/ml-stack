"""``ml-stack-serve up|down|escalate|build``: starting a server, adopting one, growing its
slots in place, stopping it, and switching the binary it is served by."""

from __future__ import annotations

import argparse
import json

from ml_stack import hub
from ml_stack.command import flag, option
from ml_stack.log import say, warn
from ml_stack.serve import build, ops
from ml_stack.serve.backend import ServerFailed, ServerSpec, UnknownFlag, parse_context
from ml_stack.serve.binary import BinaryNotFound
from ml_stack.serve.ops import Refused
from ml_stack.serve.profile import ASK, WORKLOADS, profile_for, resolved
from ml_stack.serve.serving import said_cache, split_cache_type
from ml_stack.serve.weights import DEFAULT_TIMEOUT_S

__all__ = ["OPTIONS_BUILD", "OPTIONS_DOWN", "OPTIONS_ESCALATE", "OPTIONS_UP",
           "cmd_build", "cmd_down", "cmd_escalate", "cmd_up", "from_profile"]

_SPEC = ServerSpec(model="")
DEFAULT_PORT = _SPEC.port
DEFAULT_CONTEXT = _SPEC.context
DEFAULT_PARALLEL = _SPEC.parallel
DEFAULT_TIMEOUT = DEFAULT_TIMEOUT_S


def from_profile(args: argparse.Namespace, model: str,
                 workload: str = "") -> tuple[object | None, list[str]]:
    """Fill every flag ``up`` was not given from this model's profile for ``workload``.

    Only the startup half: the draft depth goes in as the server's default, and a caller
    that can override it per request sends its own. "Not given" is "still the parser's own
    default", which is the only thing argparse can be asked afterwards. Returns the profile
    (or None) and one line per field it filled.
    """
    found = profile_for(model, workload=workload or ASK)
    if found is None:
        return None, []
    # One slot holds the whole measured cache unless --parallel was given, in which case
    # each of those slots gets what one measured slot got.
    slots = int(getattr(args, "parallel", DEFAULT_PARALLEL) or DEFAULT_PARALLEL)
    serving = found.serving(slots=slots, resolve=False)
    # The head is recorded by file name and llama-server needs a path. 'auto' is left
    # alone: `resolve_spec` answers it, and it has to know which binary will serve.
    head = found.draft
    if head and head.lower() != "auto":
        head = resolved(model, head, "", build=found.build)[0]
    # dest -> the value the profile would have, for the flags whose default `up` defines
    wanted = {
        "context": serving.context,
        "parallel": max(1, slots),
        "build": found.build,
        "draft": head,
        "spec": found.spec_type,
        "spec_n_max": found.spec_draft_max,
        "spec_p_min": found.spec_p_min,
        "kv": found.cache_type,
        "draft_kv": found.draft_cache_type,
        "mmproj": found.mmproj,
        "reasoning_budget": found.reasoning_budget,
    }
    defaults = {"context": DEFAULT_CONTEXT, "parallel": DEFAULT_PARALLEL, "build": "",
                "draft": "", "spec": "", "spec_n_max": None, "spec_p_min": None, "kv": "",
                "draft_kv": "", "mmproj": "", "reasoning_budget": None}
    took: list[str] = []
    for dest, value in wanted.items():
        if value in (None, "", ()) or getattr(args, dest, None) != defaults[dest]:
            continue
        setattr(args, dest, value)
        took.append(f"{dest} {value}")
    if found.extra_args:
        took.append(" ".join(found.extra_args))
    if serving.note:
        took.append(serving.note)
    return found, took


_EVENT_LINES = {
    "loading": lambda e: f"loading {e.get('model', '')} ({e.get('slots', 1)} slot(s))",
    "ready": lambda e: (
        f"ready in {e['load_s']:.1f}s" if e.get("load_s") is not None else "ready"),
    "escalating": lambda e: (
        f"escalating {e.get('from_slots')} -> {e.get('to_slots')} slot(s) by "
        f"{e.get('mode')} -- {e.get('reason', '')}"),
    "saving": lambda e: f"saving slot {e.get('slot')} ({e.get('tokens', 0):,} tokens)",
    "summarizing": lambda e: f"summarising slot {e.get('slot')} ({e.get('tokens', 0):,} tokens)",
    "summarized": lambda e: (
        f"summarised slot {e.get('slot')} to {e.get('tokens', 0):,} words: "
        f"{e.get('summary', '')}"),
    "stopping": lambda e: "stopping the old server",
    "restoring": lambda e: f"restoring slot {e.get('slot')} ({e.get('mode', 'cache')})",
    "done": lambda e: f"now serving {e.get('slots')} slot(s)",
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
        spec_p_min=getattr(args, "spec_p_min", None),
        spec_tree=getattr(args, "spec_tree", None),
        spec_draft_ngl=getattr(args, "draft_ngl", None),
        lookup_dynamic=str(getattr(args, "lookup_cache", "") or "") or None,
        override_tensor=tuple(getattr(args, "on_cpu", []) or ()),
        reasoning_budget=getattr(args, "reasoning_budget", None),
        # llama-server's own flags, which only a profile carries: `up` has no flag of its
        # own for `-ub 2048`
        extra_args=extra,
        cpu_moe=bool(getattr(args, "cpu_moe", False)))


OPTIONS_UP = [
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
    flag("--root", default=None,
         help="the fleet root whose beacon to announce in, when there is one "
              "(default: traind under the state root)"),
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
    flag("--spec-p-min", type=float, default=None, metavar="P",
         help="the draft's minimum probability to be accepted (server default 0.00)"),
    flag("--spec-tree", type=int, default=None, metavar="W",
         help="guess ahead in a tree: expand W branches per depth, each proposing W "
              "children, up to --spec-n-max nodes, all verified in one pass. Needs "
              "--parallel 1 and a build carrying the tree patch"),
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
    flag("--profile", action=argparse.BooleanOptionalAction, default=True,
         help="fill every flag not given from this model's measured profile -- the "
              "build, head, cache, thinking and llama-server flags that answered best "
              "(`ml-stack-serve profile MODEL` prints it). A flag given wins over the "
              "record. --no-profile serves the model bare, at this command's own "
              "defaults rather than the ones that measured best"),
    flag("--for", dest="workload", default=ASK, choices=sorted(WORKLOADS),
         metavar="WORKLOAD",
         help=f"which workload the profile is for: "
              f"{'; '.join(f'{k}, {v}' for k, v in WORKLOADS.items())} "
              f"(default: {ASK})"),
    flag("--anyway", action="store_true",
         help="start the server even while a measurement holds this card; both its "
              "timings and anything measured through this server are then two models "
              "sharing a GPU"),
]


def cmd_up(args: argparse.Namespace) -> int:
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
    resolved_spec = ops.resolve_spec(_asked_spec(args, model, extra), manager=manager)
    for note in resolved_spec.notes:
        warn(note)
    spec = resolved_spec.spec

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


OPTIONS_DOWN = [
    option("port", default=DEFAULT_PORT,
           help=f"port of the server to stop (default: {DEFAULT_PORT})"),
    flag("--orphans", action="store_true",
         help="instead of a port: stop every recorded server whose leasing process has "
              "gone, and nothing else"),
    flag("--root", default=None,
         help="the fleet root to withdraw it from (default: traind under the state root)"),
]


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


OPTIONS_ESCALATE = [
    option("port", default=DEFAULT_PORT,
           help=f"port of the server to grow (default: {DEFAULT_PORT})"),
    flag("--add", type=int, default=1, metavar="N",
         help="how many more slots to ask for (default: 1)"),
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
]


def cmd_escalate(args: argparse.Namespace) -> int:
    """``ml-stack-serve escalate`` -- grow a running server's slots in place."""
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


OPTIONS_BUILD = [
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
]


def cmd_build(args: argparse.Namespace) -> int:
    return build.cmd_build(args)
