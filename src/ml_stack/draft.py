"""``ml-stack-draft`` -- what a draft head is worth for one model, measured arm by arm.

One argument is the whole experiment: find the head, pick the workload, check the machine
is quiet, serve the model once per arm, and print what each arm cost with its units.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml_stack.bench import home_dir
from ml_stack.bench.keep import save
from ml_stack.bench.measure import sample as sampled
from ml_stack.bench.quiet import look
from ml_stack.bench.score import Row
from ml_stack.bench.serve import served, up
from ml_stack.command import Group, flag, option
from ml_stack.graph.community import QUESTIONS
from ml_stack.graph.community import graph as invented
from ml_stack.hub import choose_head, located, spec_for
from ml_stack.ingest.extract import extract_unit, schema
from ml_stack.log import say, warn
from ml_stack.serve.backend import LlamaServerBackend
from ml_stack.serve.binary import find_binary
from ml_stack.serve.profile import profile_for, profiles
from ml_stack.serve.shape import Run, Shape, Talking
from ml_stack.sources import pdf

__all__ = [
    "COMMANDS",
    "DEPTHS",
    "Arm",
    "Measured",
    "arms_for",
    "main",
    "recommendation",
    "table",
]

#: The draft depths measured unless `--depth` says otherwise.
DEPTHS: tuple[int, ...] = (2, 4, 8, 16)
#: The head's own cache types measured when `--draft-kv` asks for the extra arms.
CACHES: tuple[str, ...] = ("q8_0", "q5_1", "q4_0")
#: How many questions each arm is asked unless `--sample` says otherwise.
SAMPLE = 12
#: How many a `--smoke` run asks, to prove the path before spending the card on it.
SMOKE = 2

@dataclass(frozen=True)
class Arm:
    """One serving setting to measure: what to call it and what to lay over the run."""

    label: str
    over: Mapping[str, Any]

    @property
    def undrafted(self) -> bool:
        return not self.over.get("draft") and not self.over.get("spec_type")


@dataclass(frozen=True)
class Measured:
    """What one arm cost. Every figure is summed over the rows it measured."""

    label: str
    rows: tuple[Mapping[str, Any], ...] = ()
    quiet: bool = True

    def _total(self, key: str) -> float | None:
        said = [float(r[key]) for r in self.rows if r.get(key) is not None]
        return sum(said) if said else None

    @property
    def seconds(self) -> float:
        return self._total("seconds") or 0.0

    @property
    def written(self) -> float:
        return self._total("completion_tokens") or 0.0

    @property
    def tokens_per_second(self) -> float | None:
        """Written tokens per second of wall clock. Higher is better."""
        return self.written / self.seconds if self.seconds > 0 else None

    @property
    def acceptance(self) -> float | None:
        """Share of the head's guesses the model kept. Higher is better; None for no head."""
        offered = self._total("draft_tokens")
        return (self._total("draft_taken") or 0.0) / offered if offered else None

    @property
    def per_pass(self) -> float | None:
        """Tokens written per verification pass. 1.0 with no head; higher is better."""
        passes = self._total("verify_n")
        if passes:
            return self.written / passes
        return 1.0 if self.written and self.acceptance is None else None

    @property
    def drafting_share(self) -> float | None:
        """Share of generation time spent in the head rather than the model. Diagnostic:
        a large share with a small speedup is a head whose passes cost too much."""
        drafting, checking = self._total("draft_ms"), self._total("verify_ms")
        if drafting is None or checking is None or drafting + checking <= 0:
            return None
        return drafting / (drafting + checking)

    @property
    def ms_per_drafted_token(self) -> float | None:
        """Milliseconds the head spent per token it offered. Lower is better."""
        drafting, offered = self._total("draft_ms"), self._total("draft_tokens")
        return drafting / offered if drafting is not None and offered else None

    @property
    def errors(self) -> int:
        return sum(1 for r in self.rows if r.get("error") or r.get("timed_out"))


# ---------------------------------------------------------------- what to measure

def arms_for(head: str, *, depths: Sequence[int], caches: Sequence[str] = (),
             spec_type: str = "") -> list[Arm]:
    """The arms to measure: no head at all, then the head at each depth.

    ``caches`` adds one arm per cache type for the head's own KV cache; those are decided
    at the best depth, so they are added by `_cache_arms` once the depths have run.
    """
    out = [Arm("none", {"draft": "", "spec_type": "", "draft_n_max": None})]
    for depth in depths:
        out.append(Arm(f"head@n{depth}",
                       {"draft": head, "spec_type": spec_type, "draft_n_max": int(depth)}))
    return out


def _cache_arms(head: str, *, depth: int, caches: Sequence[str], spec_type: str) -> list[Arm]:
    """One arm per cache type for the head's own KV cache, all at ``depth``."""
    return [Arm(f"head@n{depth}-kv-{kind}",
                {"draft": head, "spec_type": spec_type, "draft_n_max": int(depth),
                 "draft_cache_type": kind})
            for kind in caches]


# ---------------------------------------------------------------- saying it

def _f(value: float | None, fmt: str = ".1f", unit: str = "") -> str:
    return "-" if value is None else f"{float(value):{fmt}}{unit}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * float(value):.0f}%"


def table(measured: Sequence[Measured]) -> list[str]:
    """One row per arm, fastest first, with the baseline's speedup beside each."""
    if not measured:
        return ["nothing measured"]
    base = next((one for one in measured if one.label == "none"), None)
    theirs = base.tokens_per_second if base is not None else None
    head = (f"  {'arm':<20} {'tok/s':>7} {'accept':>7} {'tok/pass':>9} {'drafting':>9} "
            f"{'ms/draft':>9} {'wall':>8} {'vs none':>8} {'quiet':>6}")
    out = [head, "  " + "-" * (len(head) - 2)]
    order = sorted(measured, key=lambda one: -(one.tokens_per_second or 0.0))
    for one in order:
        mine = one.tokens_per_second
        faster = (mine / theirs) if mine and theirs else None
        out.append(
            f"  {one.label:<20} {_f(mine, '.1f'):>7} {_pct(one.acceptance):>7} "
            f"{_f(one.per_pass, '.2f'):>9} {_pct(one.drafting_share):>9} "
            f"{_f(one.ms_per_drafted_token, '.2f'):>9} "
            f"{_f(one.seconds, '.0f', 's'):>8} {_f(faster, '.2f', 'x'):>8} "
            f"{('yes' if one.quiet else 'NO'):>6}")
    out += [
        "",
        "  tok/s     written tokens per second of wall clock -- higher is better",
        "  accept    of the head's guesses, how many the model kept -- higher is better",
        "  tok/pass  tokens written per verification pass; 1.00 with no head -- higher is "
        "better",
        "  drafting  share of generation time spent in the head rather than the model. A "
        "large",
        "            share with a small speedup is a head whose own passes cost too much",
        "  ms/draft  milliseconds the head spent per token it offered -- lower is better",
        "  wall      seconds over the whole arm -- lower is better",
        "  vs none   tok/s against the no-head arm -- above 1.00 is a head worth serving",
    ]
    drafted = [one for one in measured if one.acceptance is not None]
    if drafted and all(one.drafting_share is None for one in drafted):
        out.append("  drafting, tok/pass and ms/draft are '-' because the build serving "
                   "does not time the")
        out.append("  head's passes apart from the model's. A build made by "
                   "`ml-stack-serve build` carries")
        out.append("  patches/llama.cpp/0002-speculative-timings.patch, which reports "
                   "them per request.")
    if any(not one.quiet for one in measured):
        out.append("  quiet     NO means something else held this machine while the arm "
                   "ran; its seconds are not this model's")
    return out


def recommendation(measured: Sequence[Measured], *, model: str, head: str,
                   build: str = "") -> list[str]:
    """The arm to serve, as a line to paste, or the case for serving no head at all."""
    base = next((one for one in measured if one.label == "none"), None)
    theirs = base.tokens_per_second if base is not None else None
    drafted = [one for one in measured if one.label != "none" and one.tokens_per_second]
    if not drafted or theirs is None:
        return ["no baseline to recommend against: measure the no-head arm too"]
    best = max(drafted, key=lambda one: one.tokens_per_second or 0.0)
    faster = (best.tokens_per_second or 0.0) / theirs
    name = Path(model).name
    if faster <= 1.0:
        return [f"serve no head: the best arm, {best.label}, runs at {faster:.2f}x the "
                f"no-head arm.",
                f"  ml-stack-serve up {name}"]
    depth = str(best.label.split("@n")[-1].split("-")[0])
    kind = best.label.split("-kv-")[-1] if "-kv-" in best.label else ""
    line = (f"  ml-stack-serve up {name} --draft {Path(head).name} --spec-n-max {depth}"
            + (f" --draft-kv {kind}" if kind else "")
            + (f" --build {build}" if build else ""))
    return [f"serve {best.label}: {faster:.2f}x the no-head arm at "
            f"{_f(best.tokens_per_second)} tok/s, "
            f"{_pct(best.acceptance)} accepted, {_f(best.per_pass, '.2f')} tokens per "
            f"verification pass.", line]


# ---------------------------------------------------------------- running one arm

def _run_for(model: str, args: argparse.Namespace, *, workload: str) -> Any:
    """The run every arm is measured against: this model's measured shape, one seat."""
    port, context = int(args.port), int(args.context or 0)
    build, kv = str(args.build or ""), str(args.kv or "")
    found = profile_for(model, workload=workload)
    if found is not None:
        run = found.run(port=port, seats=1)
        run = run.over(seat_context=context) if context else run
    else:
        run = Run(shape=Shape(model=model, port=port, seats=1,
                              seat_context=context or 32768),
                  talking=Talking(n_predict=4096, timeout=300.0))
    if build:
        run = run.over(build=build)
    if kv:
        run = run.over(cache_type=kv)
    # every arm decides its own head, so the profile's is not carried into the baseline
    return run.over(draft="", spec_type="", draft_n_max=None)


def _asked(sample: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The questions each arm is asked and the graph they are about."""
    return list(sampled(QUESTIONS, sample)), invented()


def measure_arm(run: Any, arm: Arm, work: Any, *, kept: str, quiet: bool) -> Measured:
    """Serve the model with this arm's setting, do the work, take it down again."""
    say(f"\n--- {arm.label}")
    rows = work(run.over(**dict(arm.over)), arm.label, kept)
    return Measured(label=arm.label, quiet=quiet,
                    rows=tuple(asdict(r) if hasattr(r, "__dataclass_fields__") else dict(r)
                               for r in rows))


def _asking_work(questions: Sequence[Mapping[str, Any]], graph: Mapping[str, Any], *,
                 binary: str, store: str) -> Any:
    """A callable that serves one arm's run and asks it the graph questions."""
    def work(run: Any, label: str, kept: str) -> list[Any]:
        return served(run, questions, graph, label=label, binary=binary, kept=kept,
                      store=store or None)

    return work


def _row_of(read: Any, label: str) -> Any:
    """One unit's `Read` as the row a bench run is kept as, its call counters summed."""
    def total(key: str) -> float | None:
        said = [float(c[key]) for c in read.calls if c.get(key) is not None]
        return sum(said) if said else None

    return Row(label=label, question=f"{read.section or read.unit} {read.title}".strip(),
               seconds=float(read.seconds), calls=len(read.calls),
               prompt_tokens=int(total("prompt_tokens") or 0),
               processed_tokens=int(total("prompt_n") or 0),
               completion_tokens=int(total("completion_tokens") or 0),
               draft_tokens=_as_int(total("draft_n")),
               draft_taken=_as_int(total("draft_n_accepted")),
               draft_ms=total("draft_ms"), verify_ms=total("verify_ms"),
               verify_n=_as_int(total("verify_n")),
               timed_out=bool(read.timed_out), error=str(read.error))


def _as_int(value: float | None) -> int | None:
    return None if value is None else int(value)


def units_of(docs: Sequence[str], *, chapter: str, sample: int) -> list[Any]:
    """The document sections each arm reads, in the order the document has them."""
    out: list[Any] = []
    for one in docs:
        where = Path(one).expanduser()
        if not where.is_file():
            raise FileNotFoundError(f"no such document: {where}")
        document = pdf.read(where, chapter=chapter or None)
        found = pdf.units(document)
        say(f"{document.title}: {len(document.chapters)} chapter(s), {len(found)} "
            f"section(s) over {document.page_count} pages")
        out += found
    return out[:sample]


def _reading_work(units: Sequence[Any], *, binary: str) -> Any:
    """A callable that serves one arm's run and reads the document sections with it."""
    shape = schema()

    def work(run: Any, label: str, kept: str) -> list[Any]:
        rows = []
        with up(run, binary=binary, name=label) as (server, held):
            client = run.client(server.base_url)
            for unit in units:
                read = extract_unit(client, unit, shape)
                rows.append(_row_of(read, label))
                say(f"  {read.section or read.unit:<10} {read.seconds:6.1f}s  "
                    f"{read.concepts:>3}c {read.relations:>3}r"
                    + (f"  {read.error}" if read.error else ""))
            held.pop("loaded", None)
            held.pop("baseline", None)
            if kept:
                say(f"  kept as {save(kept, rows, held=held, label=label, workload='ingest')}")
        return rows

    return work


# ---------------------------------------------------------------- the command

def build_for(model: str, asked: str) -> str:
    """The named llama.cpp build to serve this model on: the one asked for, else the one
    its measured profile names."""
    if asked:
        return asked
    for one in profiles():
        if one.build and Path(one.model).name.lower() == Path(model).name.lower():
            return str(one.build)
    return ""


def _head_for(model: str, asked: str, build: str) -> tuple[str, str]:
    """``(head, why)`` -- the head to measure with this model, or "" and why there is none.

    A head that borrows its target's embeddings loads only on a named fork build, so the
    build serving is what decides whether one is offered.
    """
    if asked:
        return asked, "named on the command line"
    binary = None
    if build:
        try:
            binary = LlamaServerBackend(build=build).binary
        except (OSError, ValueError) as why:
            warn(f"build {build} is not on this machine ({why}); reading the head against "
                 f"the default build instead")
    chosen = choose_head(model, binary=binary if binary is not None else find_binary())
    return chosen.path, chosen.why


def _quiet_enough(anyway: bool) -> bool | None:
    """True when the machine is quiet, False when it is not and ``anyway`` says go, None
    when the run should be refused."""
    found = look()
    say("this machine, before anything is measured:")
    for line in found.lines():
        say(line)
    if found.ok:
        return True
    say("")
    for line in found.refusal():
        warn(line)
    if not anyway:
        return None
    warn("\n--anyway: measuring regardless. Every row below is marked as taken on a busy "
         "machine, and its seconds are not this model's alone.")
    return False


def _measure(args: argparse.Namespace, model: str, head: str, spec_type: str) -> int:
    """Every arm in turn, then the table and the line to serve with."""
    quiet = _quiet_enough(bool(args.anyway))
    if quiet is None:
        return 3

    depths = [int(n) for n in (args.depth or DEPTHS)]
    sample = SMOKE if args.smoke else int(args.sample)
    kept = str(args.kept or (home_dir() / "runs.ladybug"))
    docs = list(getattr(args, "docs", []) or [])
    workload = str(getattr(args, "workload", "") or "") or ("ingest" if docs else "ask")
    if workload == "ingest" and not docs:
        warn("error: --for ingest needs a document to read; name one, or leave --for out "
             "and the graph questions are asked instead")
        return 2

    if workload == "ingest":
        units = units_of(docs, chapter=str(getattr(args, "chapter", "") or ""),
                         sample=sample)
        if not units:
            warn("error: no sections to read in that document")
            return 2
        each, work = len(units), _reading_work(units, binary=str(args.binary or ""))
    else:
        questions, graph = _asked(sample)
        each = len(questions)
        work = _asking_work(questions, graph, binary=str(args.binary or ""),
                            store=str(args.store or ""))
    say(f"\n{Path(model).name} doing {workload}: {each} "
        f"{'section' if workload == 'ingest' else 'question'}(s) an arm, "
        f"{1 + len(depths)} arm(s) to start"
        + (f", head {Path(head).name}" if head else ""))

    run = _run_for(model, args, workload=workload)

    measured = [measure_arm(run, arm, work, kept=kept, quiet=bool(quiet))
                for arm in arms_for(head, depths=depths, spec_type=spec_type)]
    caches = [k for k in (args.draft_kv or ()) if k]
    if caches:
        drafted = [one for one in measured if one.label != "none" and one.tokens_per_second]
        if drafted:
            best = max(drafted, key=lambda one: one.tokens_per_second or 0.0)
            depth = int(best.label.split("@n")[-1].split("-")[0])
            say(f"\nthe head's own cache, at the best depth so far (n{depth})")
            measured += [measure_arm(run, arm, work, kept=kept, quiet=bool(quiet))
                         for arm in _cache_arms(head, depth=depth, caches=caches,
                                                spec_type=spec_type)]

    say("")
    for line in table(measured):
        say(line)
    say("")
    for line in recommendation(measured, model=model, head=head,
                               build=str(args.build or "")):
        say(line)
    say(f"\nevery arm is kept in {kept}; `ml-stack-bench show` reads them back.")
    return 0


def cmd_draft(args: argparse.Namespace) -> int:
    """``ml-stack-draft MODEL``: what the draft head is worth, arm by arm."""
    model = str(located(args.model) or args.model)
    args.build = build_for(model, str(args.build or ""))
    if args.build:
        say(f"build: {args.build} -- the shape this model measured in")
    head, why = _head_for(model, str(args.draft or ""), str(args.build or ""))
    if not head:
        warn(f"no draft head for {Path(model).name}: {why}.")
        warn("There is nothing to measure. `ml-stack-models files REPO` lists what a "
             "repository ships; name one with --draft.")
        return 1
    say(f"draft head: {Path(head).name} -- {why}")
    return _measure(args, model, head, spec_for(head))


OPTIONS = (
    flag("model", help="the model to measure: a name, a path, or an hf: reference"),
    flag("docs", nargs="*", default=[], metavar="DOC.pdf",
         help="with --for ingest: the document(s) to read. Without one the graph "
              "questions are asked, which need no file on disk"),
    flag("--for", dest="workload", default="", choices=("ask", "ingest"),
         metavar="WORKLOAD",
         help="what the arms measure: ask, the graph questions; ingest, reading a "
              "document. Default: ingest when a document is named, ask otherwise"),
    flag("--chapter", default="", metavar="N",
         help="with --for ingest: read only this chapter of each document"),
    flag("--draft", default="", metavar="HEAD",
         help="the head to measure, instead of the one shipped with the model"),
    flag("--depth", action="append", type=int, default=[], metavar="N",
         help=f"a draft depth to measure; repeat for each (default: "
              f"{', '.join(str(n) for n in DEPTHS)})"),
    flag("--draft-kv", action="append", default=[], metavar="TYPE",
         help=f"also measure the head's own KV cache stored as this, at the best depth; "
              f"repeat for each. The head's cache is small next to what a drafted token "
              f"costs in latency, so this is off unless asked for. Try "
              f"{', '.join(CACHES)}"),
    flag("--sample", type=int, default=SAMPLE, metavar="N",
         help=f"questions to ask each arm (default: {SAMPLE}). A head cannot change an "
              f"answer -- the model checks every token -- so what is measured is "
              f"acceptance and the clock"),
    flag("--smoke", action="store_true",
         help=f"ask only {SMOKE} questions of each arm, to prove the whole path before "
              f"spending the card on it"),
    flag("--anyway", action="store_true",
         help="measure even when something else holds this machine. Every row is marked, "
              "because its seconds are not this model's"),
    flag("--kv", default="", metavar="TYPE",
         help="the model's own KV cache type for every arm (default: its measured shape's)"),
    flag("--build", default="", metavar="NAME",
         help="serve with a named build (default: the model's measured shape's)"),
    flag("--binary", default="", metavar="PATH",
         help="the llama-server to serve with, when the one on PATH cannot read this model"),
    flag("--store", default="", metavar="PATH",
         help="a graph store with the word index and vectors, so look_up searches as the "
              "application does"),
    flag("--kept", default="", metavar="PATH",
         help="where each arm is kept (default: the bench's own runs store)"),
    option("port", default=8099, help="the port to serve each arm on (default: 8099)"),
    option("context", default=0,
           help="context for the one seat (default: the model's measured shape's)"),
)

COMMANDS = Group(
    "ml-stack-draft",
    "What a draft head is worth for one model: no head, then the head at each depth, "
    "measured on the same work and printed side by side.",
    options=OPTIONS, run=cmd_draft)
main = COMMANDS.run
