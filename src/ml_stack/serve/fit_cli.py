"""``ml-stack-serve fit``: how many people fit on this machine, at what context -- from
measured KV numbers, not a formula."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

from ml_stack import home
from ml_stack.command import flag, option
from ml_stack.log import say, warn
from ml_stack.serve import charts, ops
from ml_stack.serve import fit as fit_mod
from ml_stack.serve.backend import ServerSpec, parse_context
from ml_stack.serve.ops import Refused
from ml_stack.serve.serving import split_cache_type

__all__ = ["OPTIONS", "cmd_fit"]

_SPEC = ServerSpec(model="")
DEFAULT_PORT = _SPEC.port

# The per-user contexts `fit` tabulates unless told otherwise; named here so --help can
# say them without importing the module that measures.
FIT_PER_USER = (4096, 8192, 16384, 32768, 65536, 131072)

OPTIONS = [
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
]


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


def cmd_fit(args: argparse.Namespace) -> int:
    """``ml-stack-serve fit`` -- how many people fit on this machine, at what context.

    Reads the measured records rather than a formula. `--measure` serves a model once at
    `-lv 4` and writes what llama.cpp says it allocated; `--tensors` reads the GGUF header
    and needs nothing running.
    """
    from ml_stack.hub import room as machine_room

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

    contexts = per_user or list(FIT_PER_USER)
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
            drawn = charts.plot(rows, picture,
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
