"""The command line: the parser, the subcommands, the lock, and running in the background.

`_parser` is every flag; `_run` is every subcommand after the parse; `main` is what
`ml-stack-bench` calls -- the self-check and the prefetch before the lock, the lock itself,
SIGTERM taken as an exit so a served model comes down, and `--detach` re-running the
command in its own session with `status`, `tail` and `stop` reading the same file. The
ways one served model is asked (`_ways`, `halves`, `_asked`) and the sampler overrides read
off the command line (`sampling_from`, `with_card`) are here because they read argv.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.served`,
# `bench.find_model`, `bench.HOME` -- so anything patchable is looked up there at call
# time, never bound here at import.
from ml_stack.graph import bench
from ml_stack.graph.bench.keep import (
    SHORT,
    SMOKE,
    _commit,
    empties,
    forget,
    read_back,
    resumable,
    save,
)
from ml_stack.graph.bench.measure import (
    PER_QUESTION,
    _how_many,
    _idle,
    concurrent,
    read_questions,
    sample,
)
from ml_stack.graph.bench.score import NOISE, _which, export, ranking
from ml_stack.graph.bench.serve import SmokeFailed, drafts, references_in, smoked
from ml_stack.graph.bench.show import compare, missed, plot, rates, shape, table
from ml_stack.graph.vectors import MARGIN


def _ways(args: Any) -> list[dict[str, Any]]:
    """The askings to make of one served model: what was asked for, plus each --also.

    Separating these from the serving is where the time goes. A model load is minutes; an
    asking is minutes too, and repeating the load for a question about the *asking* pays it
    twice for nothing.
    """
    first: dict[str, Any] = {"terse": bool(getattr(args, "terse", False)),
                             **sampling_from(args)}
    out = [first]
    for also in getattr(args, "also", []) or []:
        if also == "terse":
            out.append({"label": "terse", "terse": True, **sampling_from(args)})
        elif also == "greedy":
            out.append({"label": "greedy", "terse": first["terse"], "temperature": 0.0})
        elif also == "card":
            # the card's own settings are read from the served model at ask time
            out.append({"label": "card", "terse": first["terse"], "_card": True})
        elif also == "rich":
            # look_up results carry a score and why they matched, and a topic hit brings
            # the people joined to it -- a question about the asking, so one load
            out.append({"label": "rich", "terse": first["terse"], "rich": True,
                        **sampling_from(args)})
        elif also == "tight":
            # show is told to light only what answers the question, and what is lit is
            # capped -- for the model that finds everything and then lights it all
            out.append({"label": "tight", "terse": first["terse"], "tight": True,
                        **sampling_from(args)})
    return out


def halves(args: Any, model: str = "") -> list[tuple[str, int]]:
    """The ``(suffix, shortlist)`` halves a sweep asks of one model: plain, and shortlisted.

    Every model gets its plain half. The shortlist half goes to every model too, unless
    ``--shortlist-for`` names substrings of the models that should have it -- `e2b,e4b` --
    in which case a model matching none of them is measured plain only. ``--plain-only``
    still means no shortlist half for anything. Matched case aside, against the name the
    model was asked for by and the file it resolved to, so `e2b` finds `gemma-4-E2B-it`.
    """
    if getattr(args, "plain_only", False):
        return [("plain", 0)]
    wanted = [w.strip().lower() for w in str(getattr(args, "shortlist_for", "") or "").split(",")
              if w.strip()]
    if wanted and not any(w in str(model).lower() for w in wanted):
        return [("plain", 0)]
    return [("plain", 0), ("shortlist", int(getattr(args, "shortlist", 0) or 0))]


def _asked(args: Any, parts: Sequence[tuple[str, int]]) -> list[dict[str, Any]]:
    """Every way one served model is asked, both halves in one load: each half of ``parts``
    crossed with each `_ways` variant, labelled ``plain``, ``plain-terse``, ``shortlist``...

    Loading the model once per half was how the sweep began, and a load is minutes that
    say nothing about the asking. Whether a shortlist is handed over is a question about
    the asking, so it rides on the way like `terse` does and the server is put up once.
    """
    out: list[dict[str, Any]] = []
    for suffix, shortlist in parts:
        for way in bench._ways(args):
            tag = str(way.get("label", "") or "")
            out.append({**way, "label": f"{suffix}-{tag}" if tag else suffix,
                        "shortlist": shortlist})
    return out


def sampling_from(args: Any) -> dict[str, Any]:
    """The sampler overrides asked for on the command line, and nothing else.

    A setting not given is left out entirely rather than defaulted here, so the client falls
    through to the model's own card. Sweeping them is the point: "is gemma-4 better at the
    temperature its publisher asks for than at 0?" is a question about this graph and these
    questions, and nobody else can answer it for you.
    """
    named = {"n_predict": getattr(args, "n_predict", None),
             "temperature": getattr(args, "temperature", None),
             "top_p": getattr(args, "top_p", None), "top_k": getattr(args, "top_k", None),
             "min_p": getattr(args, "min_p", None)}
    return {k: v for k, v in named.items() if v is not None}


def with_card(client: Any, args: Any) -> Any:
    """The same client, asking with what its model's card recommends, when --card was given.

    This is the only place a card is ever applied. A publisher's advice is a hypothesis about
    a task they have not seen; making it easy to test and impossible to ship by accident is
    the whole arrangement.
    """
    if not getattr(args, "card", False):
        return client
    asked = dict(client.card)
    asked.update(sampling_from(args))          # an explicit flag still beats the card
    if not asked:
        print(f"note: {client.base_url} serves a model whose card names no sampler settings",
              file=sys.stderr)
        return client
    return type(client)(client.base_url, **asked)


def checking(one: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The two flags every measuring subcommand has for the two checks it makes before it
    measures: the self-check on no GPU, and the smoke on the real one."""
    one.add_argument("--no-selfcheck", action="store_true",
                     help="skip the dry run made first, before the lock is taken: this exact "
                          "command through the whole path with a scripted model, no server "
                          "and no GPU, into a scratch store read back -- what catches a flag "
                          "the client does not take before a load is paid for it. For a run "
                          "you are deliberately repeating, whose path the last one proved. "
                          "Read before the rest of the line is parsed, like --no-queue")
    one.add_argument("--no-smoke", action="store_true",
                     help=f"skip the {SMOKE}-question smoke a real run makes first on the "
                          f"real server and the real store, read back, before its own "
                          f"questions. Without this every run that is not itself --smoke "
                          f"smokes first -- on the same load, where the model is served -- "
                          f"and a smoke that fails ends the run before anything else starts")
    return one


def _parser() -> argparse.ArgumentParser:
    """The command line of ``ml-stack-bench``, built once per call and shared with `detach`,
    which needs a label out of an argv before handing it to the child."""
    # allow_abbrev=False on every parser here: a flag that is documented but not defined
    # must be refused by name, not bound by prefix to whichever neighbour shares its
    # first letters -- `--short` became `--shortlist` that way and the error blamed the
    # wrong flag. The usage line's list of subcommands is generated, not written, so it
    # cannot go stale the way "{prepare,run,sweep,show}" did when `drafts` arrived.
    ap = argparse.ArgumentParser(
        prog="ml-stack-bench", allow_abbrev=False,
        description="Time a set of questions through a graph, and compare two runs.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", allow_abbrev=False,
                         help="ask every question once and keep what it cost")
    run.add_argument("label", help="what this run is, e.g. with-shortlist")
    run.add_argument("--kept", default=str(bench.HOME / "runs.ladybug"),
                     help="where to keep the run (default: %(default)s)")
    run.add_argument("--base-url", default="http://127.0.0.1:8080",
                     help="the model answering (default: %(default)s)")
    run.add_argument("--graph", default="",
                     help="a graph as JSON (default: the invented community that ships here)")
    run.add_argument("--questions", default="",
                     help="one per line: a question, or {\"q\":..., \"expect\":[ids]} "
                          "(default: the ones that go with the invented community)")
    run.add_argument("--shortlist", type=int, default=0, metavar="N",
                     help="hand the model the N likeliest entries before it starts, found by "
                          "search rather than by asking it to look (default: 0, it looks)")
    run.add_argument("--store", default=bench.prepared(),
                     help="a graph store with the word index and vectors: look_up searches it "
                          "as the application does, and --shortlist reads it first "
                          "(default: what `prepare` built, when it has been)")
    run.add_argument("--embed-url", default="",
                     help="a server that embeds, for --shortlist to search by meaning")
    run.add_argument("--embed-model", default="", help="the model that embedded the graph")
    run.add_argument("--margin", type=float, default=MARGIN,
                     help="how far the best match must stand above the rest before a "
                          "shortlist is worth handing over (default: %(default)s)")
    run.add_argument("--ask", default="",
                     help="module:function taking (question, client) — for asking some other "
                          "way than the ordinary one")
    run.add_argument("--client", default="",
                     help="module:function returning the model client, instead of --base-url")

    heads = sub.add_parser("drafts", allow_abbrev=False,
                           help="serve one model with each draft head in turn "
                                "and measure what each is worth")
    heads.add_argument("model", help="the model to serve: a name, a path, or an hf: "
                                        "reference. A name is looked up on this machine")
    heads.add_argument("--draft", action="append", default=[], metavar="PATH",
                       help="a draft head to measure; repeat for each. Pass '' for the "
                            "baseline with no draft, which every other row must beat")
    heads.add_argument("--n-max", action="append", type=int, default=[], metavar="N",
                       help="how many tokens a head guesses ahead per pass "
                            "(--spec-draft-n-max); repeat to serve each head once per "
                            "value, labelled draft:<head>@nN. Without it, once at the "
                            "build's own default")
    heads.add_argument("--serve-kv", default="", metavar="TYPE",
                       help="quantise the KV cache of the served model (q8_0, q4_0); the "
                            "label ends -kv-TYPE and the table's ctx column shows it")
    heads.add_argument("--port", type=int, default=8099)
    heads.add_argument("--context", type=int, default=32768)
    heads.add_argument("--parallel", type=int, default=1)
    heads.add_argument("--binary", default="", help="a llama-server that reads this model")
    heads.add_argument("--kept", default=str(bench.HOME / "runs.ladybug"))
    heads.add_argument("--questions", default="")
    heads.add_argument("--sample", type=int, default=SHORT, metavar="N",
                       help="how many questions to ask each head (default: %(default)s). "
                            "A draft cannot change an answer -- the large model verifies "
                            "every token -- so what is being measured is acceptance and "
                            "wall clock, and a full run spends most of itself proving a "
                            "score that must come out the same")
    heads.add_argument("--smoke", action="store_true",
                       help=f"ask only {SMOKE} questions of each head, to prove the whole "
                            f"path -- serve, ask, save, and read the run back -- before "
                            f"spending the GPU on it")

    conc = sub.add_parser("concurrent", allow_abbrev=False,
                          help="ask N conversations of T turns each at the same time, and "
                               "see what the waiting, the memory and the accuracy cost")
    conc.add_argument("label", help="what this run is, e.g. e2b-4x3")
    conc.add_argument("--conversations", type=int, default=4, metavar="N",
                      help="how many conversations are in flight together (default: "
                           "%(default)s). More than the server has slots, and the turns "
                           "queue -- which is the thing worth measuring")
    conc.add_argument("--turns", type=int, default=3, metavar="T",
                      help="how many questions each conversation asks in turn, the earlier "
                           "ones carried (default: %(default)s)")
    conc.add_argument("--kept", default=str(bench.HOME / "runs.ladybug"),
                      help="where to keep the run (default: %(default)s)")
    conc.add_argument("--base-url", default="http://127.0.0.1:8080",
                      help="the model answering (default: %(default)s)")
    conc.add_argument("--graph", default="",
                      help="a graph as JSON (default: the invented community that ships here)")
    conc.add_argument("--questions", default="",
                      help="one per line, as for run (default: the invented community's)")
    conc.add_argument("--store", default=bench.prepared(),
                      help="a graph store with the word index and vectors, so look_up "
                           "searches as the application does (default: what `prepare` "
                           "built, when it has been)")
    conc.add_argument("--embed-url", default="", help="a server that embeds, for the store")
    conc.add_argument("--embed-model", default="", help="the model that embedded the graph")
    conc.add_argument("--client", default="",
                      help="module:function returning the model client, instead of --base-url")

    ready = sub.add_parser("prepare", allow_abbrev=False,
                           help="put a graph in a store and index and embed it")
    ready.add_argument("--store", default=str(bench.HOME / "graph.ladybug"),
                       help="the store to build (default: %(default)s)")
    ready.add_argument("--graph", default="",
                       help="a graph as JSON (default: the invented community)")
    ready.add_argument("--embed-url", default="",
                       help="a server that embeds; without one only the word index is built")
    ready.add_argument("--embed-model", default="", help="what to file the vectors under")

    sweep = sub.add_parser("sweep", allow_abbrev=False,
                           help="run every model, with and without a shortlist")
    sweep.add_argument("--on", action="append", metavar="NAME=URL", default=[],
                       help="a model to measure, e.g. e4b=http://127.0.0.1:8083; repeatable")
    sweep.add_argument("--kept", default=str(bench.HOME / "runs.ladybug"),
                       help="where to keep the runs (default: %(default)s)")
    sweep.add_argument("--graph", default="", help="a graph as JSON (default: the invented one)")
    sweep.add_argument("--questions", default="", help="(default: the ones that go with it)")
    sweep.add_argument("--shortlist", type=int, default=8, metavar="N",
                       help="how many to hand over in the second run (default: %(default)s)")
    sweep.add_argument("--store", default=bench.prepared(),
                       help="the indexed and embedded graph, for look_up and the shortlist "
                            "(default: what `prepare` built, when it has been)")
    sweep.add_argument("--embed-url", default="", help="a server that embeds, for --shortlist")
    sweep.add_argument("--embed-model", default="", help="the model that embedded the graph")
    sweep.add_argument("--margin", type=float, default=MARGIN)
    sweep.add_argument("--serve", action="append", default=[], metavar="MODEL",
                       help="a model to put up, measure and take down again, one at a time: "
                            "a name, a path, or an hf: reference. A name is looked up on "
                            "this machine. "
                            "Repeat for each. Without this, --on measures servers somebody "
                            "else started -- which leaves the starting, stopping and waiting "
                            "to a shell loop that dies with its terminal")
    sweep.add_argument("--context", type=int, default=0, metavar="N",
                       help="total context for a --serve'd model (default: 32768 per slot)")
    sweep.add_argument("--parallel", type=int, default=1, metavar="N",
                       help="slots for a --serve'd model (default: %(default)s)")
    sweep.add_argument("--serve-port", type=int, default=8099,
                       help="the port each served model gets (default: %(default)s)")
    sweep.add_argument("--serve-draft", action="append", default=[], metavar="PATH_OR_AUTO",
                       help="a draft head for the matching --serve, positionally; 'auto' "
                            "finds the one shipped with it, '' serves without")
    sweep.add_argument("--binary", default="", metavar="PATH",
                       help="a llama-server that reads these models")
    sweep.add_argument("--serve-kv", default="", metavar="TYPE",
                       help="quantise the KV cache of each --serve'd model (q8_0, q4_0): "
                            "every label ends -kv-TYPE and the table's ctx column shows "
                            "it, since a run with a quantised cache is another "
                            "configuration and not another model")
    sweep.add_argument("--reasoning-budget", type=int, default=None, metavar="N",
                       help="tokens each --serve'd model may spend thinking before it must "
                            "answer (llama-server --reasoning-budget; -1 is unlimited). A "
                            "ceiling (--n-predict) cuts the answer; this is the budget that "
                            "stops the thinking. Every label ends -rbN and the table's ctx "
                            "column shows /rb, since it is another configuration")
    sweep.add_argument("--plain-only", action="store_true",
                       help="skip the shortlist half, just measure each model as it is")
    sweep.add_argument("--shortlist-for", default="", metavar="A,B",
                       help="substrings of the models that get the shortlist half as well "
                            "(e2b,e4b); the rest are measured plain only. Both halves of "
                            "a model are asked of one load")
    sweep.add_argument("--resume", action="store_true",
                       help="skip any model and way already kept since --since with this "
                            "many questions at this context and these slots, so a sweep "
                            "killed on its third model costs the third model and not all "
                            "three. Says which it skipped and when each was kept")
    sweep.add_argument("--since", default="", metavar="WHEN",
                       help="with --resume, how old a kept run may be and still count: an "
                            "ISO date or date-time (default: the start of today)")
    sweep.add_argument("--fleet", action="store_true",
                       help="spread the --serve models over the fleet instead of this "
                            "machine: one job per model, this same line with that one "
                            "--serve, planned over the peers, dispatched, waited for and "
                            "gathered into --kept, then shown. Every peer must be on this "
                            "checkout's commit; a peer that is not is refused, here and by "
                            "its daemon")
    sweep.add_argument("--peers", default="", metavar="NAME,...",
                       help="with --fleet, only these peers (default: whichever the fleet "
                            "plans over)")

    for one in (run, sweep, conc):
        one.add_argument("--sample", type=int, default=0, metavar="N",
                         help="ask only N of the questions, keeping every kind of answer. "
                              "For a comparison where accuracy is not the variable -- draft "
                              "heads, sampling, serving flags -- most of a full run is "
                              "spent re-establishing a score that cannot move")
        one.add_argument("--short", dest="short", action="store_true",
                         help=f"the same as --sample {SHORT}: every kind still asked about "
                              f"and the same mean number of answers expected, at about half "
                              f"the time. Each question is worth more, so a small difference "
                              f"is noise on a short run and signal on a full one")
        one.add_argument("--smoke", action="store_true",
                         help=f"ask only {SMOKE} questions, to prove the whole path works "
                              f"before spending the GPU on it. Serving, asking, scoring, "
                              f"measuring and saving all happen, so anything that would "
                              f"raise at the end raises here instead. The score means "
                              f"nothing at this size -- run it first, then run it properly")
        one.add_argument("--temperature", type=float, default=None,
                         help="override the sampling temperature; the default is whatever "
                              "the model's own card asks for (gemma-4: 1.0)")
        one.add_argument("--top-p", type=float, default=None, help="override top_p")
        one.add_argument("--top-k", type=int, default=None, help="override top_k")
        one.add_argument("--min-p", type=float, default=None, help="override min_p")
        one.add_argument("--n-predict", type=int, default=16384,
                         help="tokens each turn may write -- thinking, tool calls and the "
                              "answer together. A thinking model spends most of a turn "
                              "reasoning, so a low ceiling truncates the answer rather than "
                              "the thought (default: %(default)s)")
        one.add_argument("--anyway", action="store_true",
                         help="measure even when the server is already busy; the wall clock "
                              "will then be two runs sharing a GPU, not one run")
        one.add_argument("--card", action="store_true",
                         help="ask with what the model's own card recommends, to see whether "
                              "it suits this task -- it is not what a client sends otherwise")
    for one in (run, sweep):
        one.add_argument("--also", action="append", default=[],
                         choices=("terse", "card", "greedy", "rich", "tight"),
                         help="ask the same served model another way as well. Whether the "
                              "tools are described briefly, what sampling is used, "
                              "whether look_up says why it matched (rich), and whether "
                              "show is told to light only what answers the question and "
                              "capped at six (tight) are questions about the asking, not "
                              "the serving, so measuring five of them costs one model load "
                              "rather than five. Repeatable")

    for one in (run, sweep, heads, conc):
        one.add_argument("--per-question", type=float, default=PER_QUESTION,
                         metavar="SECONDS",
                         help="the most one question may take before it is recorded as "
                              "timed out -- no answer, scored wrong, the cap as its wall "
                              "clock -- and the next is asked (default: %(default)s). The "
                              "table counts them under t/o and --detail names them. "
                              "Measured: a 26B thinking model spent 505 s on one question "
                              "under a 16k ceiling, and a run that waits for that is not "
                              "a run")
        one.add_argument("--no-queue", action="store_true",
                         help="fail at once if another measurement holds the GPU, rather "
                              "than queue behind it. Read before the rest of the line is "
                              "parsed, and listed here so that --help says it exists")
        one.add_argument("--detach", action="store_true",
                         help="run this in the background, owned by nobody's terminal: the "
                              "command re-runs itself in a new session with its output in "
                              f"a log under {bench.HOME / 'logs'}, prints the log's path and "
                              "returns at once. `status` says what is measuring, `tail -f` "
                              "follows the log, `stop` ends it and takes its server down. "
                              "Read before the rest of the line is parsed, like --no-queue")
        one.add_argument("--no-prefetch", action="store_true",
                         help="do not download the hf: models and heads named here before "
                              "the measuring lock is taken. Without this every reference "
                              "is fetched first, one line each with its size, because a "
                              "download inside the timed window is a timing of the network")
        checking(one)

    from ml_stack.graph.bench.extract import add_arguments as extracting

    checking(extracting(sub))

    show = sub.add_parser("show", allow_abbrev=False,
                          help="compare two runs, or list what is kept")
    show.add_argument("--kept", default=str(bench.HOME / "runs.ladybug"),
                      help="the store the runs are in (default: %(default)s)")
    show.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), default=None)
    show.add_argument("--extract", action="store_true",
                      help="only the extraction runs (ml-stack-bench extract), in their own "
                           "table; without it they print under the answering table")
    show.add_argument("--detail", nargs="?", const="", default=None, metavar="LABEL",
                      help="the questions themselves, not the totals: what each one wanted, "
                           "what it showed, and what it missed. A label narrows it to one run")
    show.add_argument("--all", action="store_true",
                      help="with --detail, every question and not only the ones that missed")
    show.add_argument("--shape", action="store_true",
                      help="what the question set is made of -- which kinds its answers "
                           "want, and how many want no person -- so its bias is visible")
    show.add_argument("--rates", action="store_true",
                      help="what each run cost to be right -- accuracy over time, tokens and "
                           "memory -- with the Pareto frontier marked")
    show.add_argument("--anyway-export", action="store_true", dest="export_anyway",
                      help="with --export, include runs measured over some other graph. "
                           "Not into a repository: those may carry a real community's "
                           "questions")
    show.add_argument("--export", default="", metavar="FILE.json",
                      help="write every run as JSON. The store lives under ~/.ml-stack and "
                           "nothing backs it up: a day of measuring is on one disk, and the "
                           "results are all of an invented community, so they can be kept "
                           "beside the code that produced them")
    show.add_argument("--rank", default="", metavar="FILE.md",
                      help="write which model answers best, as a conclusion rather than as "
                           "evidence: one line per model -- accuracy from its largest run, "
                           "cost per question from its fastest run that held that accuracy, "
                           "a draft head or a fork included -- and which run that was. This "
                           "is the part worth keeping in a repository -- the raw runs are "
                           "not, since they describe one machine and one build")
    show.add_argument("--noise", type=float, default=NOISE * 100, metavar="PTS",
                      help="how far a run's F1 may fall under its model's largest run and "
                           "still supply the cost, in points (default: %(default)s -- one "
                           "question of a short run). A run outside it is listed as rejected")
    show.add_argument("--plot", default="", metavar="FILE.html",
                      help="write the runs as a scatter of accuracy against --cost, with "
                           "the frontier joined; opens with no network and no packages")
    show.add_argument("--cost", default="seconds",
                      choices=("seconds", "paid_tokens", "kv_bytes"),
                      help="which cost the frontier is drawn against (default: %(default)s)")

    gone = sub.add_parser("forget", allow_abbrev=False,
                          help="delete kept runs: the empty ones, or every run of one label")
    gone.add_argument("label", nargs="?", default="",
                      help="delete every run kept under this label (needs --yes)")
    gone.add_argument("--empty", action="store_true",
                      help="delete every run that reads back as nothing")
    gone.add_argument("--yes", action="store_true",
                      help="really delete a label's runs; without it they are only listed")
    gone.add_argument("--kept", default=str(bench.HOME / "runs.ladybug"),
                      help="the store the runs are in (default: %(default)s)")

    sub.add_parser("status", allow_abbrev=False,
                   help="whether something is measuring, since when, with what, and where "
                        "its log is. Exits 0 either way")
    following = sub.add_parser("tail", allow_abbrev=False,
                               help="the log of the current measurement, or the latest")
    following.add_argument("-n", type=int, default=20, metavar="N",
                           help="how many lines from the end (default: %(default)s)")
    following.add_argument("-f", action="store_true", dest="follow",
                           help="keep printing as the log grows, until the measurement ends")
    sub.add_parser("stop", allow_abbrev=False,
                   help="end the detached measurement: SIGTERM to its pid, so it takes down "
                        "any server it put up, then wait up to a minute. Never by name")

    from ml_stack.graph.bench.history import add_arguments as remembering

    remembering(sub.add_parser("history", allow_abbrev=False,
                               help="every measurement the logs remember: when, how long, "
                                    "how it ended, the estimate beside the actual, and the "
                                    "runs it kept"))
    return ap


def _main(argv: list[str] | None = None) -> int:
    """``ml-stack-bench`` -- what a change to the asking costs, and whether it was worth it."""
    args = _parser().parse_args(argv)
    # the line as given, for `sweep --fleet` to hand each peer the same line with one model
    args._argv = list(argv if argv is not None else sys.argv[1:])
    return _run(args)


def wants_smoke(args: Any) -> bool:
    """Whether a run smokes first: it is a real run, not itself ``--smoke``, and not told
    ``--no-smoke``."""
    return (getattr(args, "cmd", "") in MEASURING and not getattr(args, "smoke", False)
            and not getattr(args, "no_smoke", False))


def smoke_first(args: Any) -> None:
    """The same command as a smoke, before the run proper: the real server, the real
    store, the runs read back, and `SmokeFailed` -- the run never starts -- when it kept
    nothing or every question failed. For the commands that serve nothing themselves;
    a served model smokes inside `served`, on the one load."""
    trial = argparse.Namespace(**vars(args))
    trial.smoke = True
    before = {r["key"] for r in bench._kept(args.kept)}
    print(f"smoke: the same {args.cmd} on {SMOKE} question(s) first -- ask, score, save, "
          f"read back -- before the run proper")
    code = _run(trial)
    if code != 0:
        raise SmokeFailed(f"{args.cmd} --smoke returned {code}")
    smoked([r for r in bench._kept(args.kept) if r["key"] not in before],
           f"{args.cmd} smoke")
    print("smoke: ok\n")


def _run(args: Any) -> int:
    """`_main` after the parse, so a dry run can hand in a namespace it has rewritten."""
    if args.cmd == "history":
        from ml_stack.graph.bench.history import run as remembered

        return remembered(args)
    if args.cmd == "status":
        print(status())
        return 0
    if args.cmd == "tail":
        return tail(lines=args.n, follow=args.follow)
    if args.cmd == "stop":
        print(stop())
        return 0
    if args.cmd == "forget":
        if not args.empty and not args.label:
            print("error: say what to forget: --empty, or a label", file=sys.stderr)
            return 2
        if args.empty:
            went = forget(args.kept, empty=True)
            print(f"{len(went)} empty run(s) removed" if went else "no empty runs")
        if args.label:
            if not args.yes:
                would = [r["key"] for r in bench.runs(args.kept, args.label)]
                print("\n".join(would) if would else f"no run labelled {args.label!r}")
                if would:
                    print(f"{len(would)} run(s) would go; pass --yes to delete them")
                return 0
            went = forget(args.kept, label=args.label)
            print(f"{len(went)} run(s) labelled {args.label!r} removed")
        return 0
    if args.cmd == "sweep":
        from ml_stack.client import Client
        from ml_stack.graph.community import QUESTIONS, graph as invented

        named = []
        for one in args.on:
            name, _, url = one.partition("=")
            if not name or not url:
                print(f"error: --on wants NAME=URL, got {one!r}", file=sys.stderr)
                return 2
            named.append((name, url))
        if not named and not getattr(args, "serve", []):
            print("error: nothing to measure; pass --on NAME=URL for a server that is "
                  "already up, or --serve MODEL to put one up", file=sys.stderr)
            return 2
        if getattr(args, "fleet", False):
            return _fleet_sweep(args)
        everything = read_questions(args.questions) if args.questions else QUESTIONS
        questions = sample(everything, _how_many(args))
        graph = (json.loads(Path(args.graph).expanduser().read_text())
                 if args.graph else invented())
        # the smoke: two questions first, of every model. The servers somebody else
        # started are smoked as a sweep of their own before anything is served, and each
        # served model smokes as it comes up, so a load is paid once
        smoking = wants_smoke(args)
        if smoking and named:
            standing = argparse.Namespace(**vars(args))
            standing.serve = []
            smoke_first(standing)
        saved: list[str] = []
        total_context = args.context or 32768 * max(1, args.parallel)
        already = (resumable(args.kept, questions=len(questions), context=total_context,
                             parallel=getattr(args, "parallel", 1), since=args.since)
                   if args.resume else None)
        # `wanted`, not `named`: the loop variable was `named` once, which rebound the
        # (name, url) list built from --on to the last model's name, and the summary below
        # then unpacked its characters. Every `sweep --serve` answered its questions and
        # crashed while summarising, and the smoke run is what caught it.
        for n, wanted in enumerate(getattr(args, "serve", []) or []):
            model = bench.find_model(wanted)
            heads = getattr(args, "serve_draft", []) or []
            head = heads[n] if n < len(heads) else ""
            if head.lower() == "auto":
                # the one resolver (`hub.choose_head`): told which binary will serve, so
                # a head that borrows its target's embeddings is withheld from mainline
                # rather than found out at the far end of an 87G load
                from ml_stack import hub

                chosen = hub.choose_head(model, binary=args.binary or None)
                head = chosen.path
                print(f"    draft head: {head or 'none'} -- {chosen.why}"
                      + (f"\n      {chosen.note}" if chosen.note else ""))
            stem = str(model).rsplit("/", 1)[-1].removesuffix(".gguf")[:14]
            # Both halves -- plain, and shortlisted where `--shortlist-for` allows it --
            # and every `--also` of each, asked of one load. Loading twice per model was
            # how this began, and the second load measured nothing about the asking.
            parts = halves(args, f"{wanted} {model}")
            print(f"\n{stem}: " + ", ".join(suffix for suffix, _ in parts))
            # A port nothing answers on is exactly what --serve expects, so the
            # "would not say whether it is busy" note is noise here. Only a port
            # somebody is actually using should stop us.
            if bench.busy(f"http://127.0.0.1:{args.serve_port}") > 0 and not _idle(
                    f"http://127.0.0.1:{args.serve_port}", args):
                return 3
            # `--context` is the total across slots, which is what `-c` takes and what
            # ServerSpec means by it. Dividing by the slot count served a model at a
            # quarter of the context every other run had, and the only thing that said
            # so was the `ctx` column reading 8k where the rest read 32k.
            before = {r["key"] for r in bench._kept(args.kept)}
            from ml_stack.serve.backend import ServerFailed

            try:
                bench.served(model, questions, graph, label=stem, draft=head,
                       ways=_asked(args, parts),
                       port=args.serve_port,
                       context=total_context,
                       parallel=getattr(args, "parallel", 1), binary=args.binary,
                       kept=args.kept,
                       store=args.store or None, embed_url=args.embed_url,
                       embed_model=args.embed_model, terse=getattr(args, "terse", False),
                       already=already, cache_type=getattr(args, "serve_kv", "") or "",
                       per_question=args.per_question,
                       reasoning_budget=getattr(args, "reasoning_budget", None),
                       smoke=sample(everything, SMOKE) if smoking else (),
                       **sampling_from(args))
            except ServerFailed as why:
                # A model that will not load -- a head the build cannot read, a tensor it
                # does not know -- ends that model, not the sweep. Measured 2026-09-01: one
                # such load took gpt-oss-120b's measurement down with it, twice.
                print(f"    {stem} did not load; moving on:\n"
                      + "\n".join(f"      {line}" for line in str(why).splitlines()[:6]))
                continue
            saved += [r["key"] for r in bench._kept(args.kept) if r["key"] not in before]

        for name, url in named:
            for suffix, shortlist in halves(args, name):
                label = f"{name}-{suffix}"
                if already is not None and already(label):
                    print(f"skipping {label}: kept at {already(label).get('at', '?')}")
                    continue
                ask = bench.asking(graph, shortlist=shortlist, store=args.store or None,
                             embed_url=args.embed_url, embed_model=args.embed_model,
                             terse=getattr(args, "terse", False), margin=args.margin)
                print(f"\n{label} on {url}, look_up by {ask.finder}")
                if not _idle(url, args):
                    return 3
                asking_with = with_card(Client(url, timeout=args.per_question,
                                               **sampling_from(args)), args)
                # what it will actually send, card and overrides together: a run measured at
                # one temperature against a run at another is two measurements, and the only
                # way to know later is to write it down now
                used = dict(asking_with.sampling)
                rows = bench.measure(ask, questions, label=label, client=asking_with,
                                     log=print,
                               graph=graph, per_question=args.per_question)
                saved.append(save(args.kept, rows,
                                  held={**bench.footprint(url), "sampling": used,
                                        "graph": _which(graph), "finder": ask.finder}))
        print()
        table(read_back(args.kept, saved) if args.smoke else bench._kept(args.kept))
        return 0
    if args.cmd == "prepare":
        from ml_stack.graph.community import graph as invented
        from ml_stack.graph.store import replace
        from ml_stack.graph.vectors import remember

        graph = json.loads(Path(args.graph).expanduser().read_text()) if args.graph else invented()
        counted = replace(args.store, graph)          # writing builds the word index
        print(f"{args.store}: {counted['nodes']} nodes, {counted['edges']} edges, word index built")
        if not args.embed_url:
            print("  no --embed-url, so no vectors: search will be words only")
            return 0
        from ml_stack.graph.store import GraphStore

        texts = {n["id"]: (n["label"] + " — " + " ".join(
            (graph.get("messages", {}).get(m) or {}).get("text", "")
            for m in (n.get("messages") or [])))[:1400] for n in graph["nodes"]}
        with GraphStore(args.store) as held:
            written = remember(held, texts, base_url=args.embed_url,
                               model=args.embed_model or "embed", log=print)
        print(f"  {written} embedded")
        return 0
    if args.cmd == "drafts":
        from ml_stack.graph.community import QUESTIONS, graph as invented

        everything = read_questions(args.questions) if args.questions else QUESTIONS
        asked = sample(everything, SMOKE if getattr(args, "smoke", False) else args.sample)
        before = {r["key"] for r in bench._kept(args.kept)}
        rows = drafts(bench.find_model(args.model), args.draft or [""], asked, invented(),
                      port=args.port,
                      context=args.context, parallel=args.parallel, binary=args.binary,
                      kept=args.kept, store=bench.prepared() or None,
                      n_max=list(getattr(args, "n_max", []) or []) or [None],
                      cache_type=getattr(args, "serve_kv", "") or "",
                      per_question=args.per_question,
                      smoke=sample(everything, SMOKE) if wants_smoke(args) else ())
        print()
        if getattr(args, "smoke", False):
            saved = [r["key"] for r in bench._kept(args.kept) if r["key"] not in before]
            table(read_back(args.kept, saved))
        else:
            table(bench._kept(args.kept))
        return 0 if rows else 1

    if args.cmd == "concurrent":
        from ml_stack.graph.community import QUESTIONS, graph as invented

        if wants_smoke(args):
            smoke_first(args)
        questions = sample(read_questions(args.questions) if args.questions else QUESTIONS,
                           _how_many(args))
        if not questions:
            print(f"error: no questions in {args.questions}", file=sys.stderr)
            return 2
        graph = (json.loads(Path(args.graph).expanduser().read_text())
                 if args.graph else invented())
        if args.client:
            client = bench.ask_from(args.client)()
        else:
            from ml_stack.client import Client

            if not _idle(args.base_url, args):
                return 3
            client = with_card(Client(args.base_url, timeout=args.per_question,
                                      **sampling_from(args)), args)
        # a smoke run proves the path -- two conversations really overlapping, one turn
        # each -- and its numbers mean nothing, as with every other --smoke
        many, long = (2, 1) if args.smoke else (args.conversations, args.turns)
        ask = bench.asking(graph, store=args.store or None, embed_url=args.embed_url,
                     embed_model=args.embed_model)
        where = args.graph or "the invented community"
        print(f"{args.label}: {many} conversations of {long} turn(s) at once over {where}, "
              f"look_up by {ask.finder}")
        rows, held = concurrent(ask, questions, conversations=many, turns=long,
                                label=args.label, client=client, graph=graph,
                                base_url="" if args.client else args.base_url, log=print,
                                per_question=args.per_question)
        at = held["concurrency"]
        slots = at.get("slots") or 0
        print(f"  {at['seconds']:.1f}s for all of it"
              + (f", {at['queued']:.1f}s of that queued" if slots and many > slots else "")
              + (f", {slots} slot(s)" if slots > 0 else ""))
        key = save(args.kept, rows,
                   held={**held, "sampling": dict(getattr(client, "sampling", {}) or {}),
                         "graph": _which(graph), "finder": ask.finder})
        print(f"kept as {key}")
        if args.smoke:
            table(read_back(args.kept, [key]))
        return 0

    if args.cmd == "extract":
        from ml_stack.graph.bench.extract import main as extracting

        return extracting(args)

    if args.cmd == "show":
        from ml_stack.graph.bench import extract as bench_extract

        # an extraction run is kept in the same store and is not an answering run: it has
        # no questions to score, and its table is its own
        everything = bench.runs(args.kept) if Path(args.kept).expanduser().exists() else []
        extracted = bench_extract.only(everything)
        answering = [r for r in everything if r.get("kind") != bench_extract.KIND]
        if getattr(args, "extract", False):
            bench_extract.table(extracted)
            return 0
        if args.compare:
            print(compare(args.kept, *args.compare))
            return 0
        if args.rank:
            ranking(answering, args.rank, noise=args.noise / 100)
            print(args.rank)
            return 0
        if args.export:
            print(export(answering, args.export,
                         anyway=getattr(args, "export_anyway", False)))
            return 0
        if args.shape:
            from ml_stack.graph.community import QUESTIONS, graph as invented

            questions = read_questions(args.questions) if getattr(args, "questions", "") \
                else QUESTIONS
            shape(questions, invented())
            return 0
        if args.plot:
            print(plot(answering, args.plot, cost=args.cost, noise=args.noise / 100))
            return 0
        if args.rates:
            rates(answering, cost=args.cost, noise=args.noise / 100)
            return 0
        if args.detail is not None:
            missed([r for r in answering if not args.detail or r.get("label") == args.detail],
                   everything=args.all, among=answering)
            return 0
        table(answering)
        if extracted:
            print()
            bench_extract.table(extracted)
        hollow = empties(args.kept)
        if hollow:
            print(f"{len(hollow)} empty run(s) skipped -- ml-stack-bench forget --empty "
                  f"removes them")
        return 0

    from ml_stack.graph.community import QUESTIONS, graph as invented

    if wants_smoke(args):
        smoke_first(args)
    questions = sample(read_questions(args.questions) if args.questions else QUESTIONS,
                       _how_many(args))
    if not questions:
        print(f"error: no questions in {args.questions}", file=sys.stderr)
        return 2
    graph = json.loads(Path(args.graph).expanduser().read_text()) if args.graph else invented()
    if args.client:
        client = bench.ask_from(args.client)()
    else:
        from ml_stack.client import Client

        if not _idle(args.base_url, args):
            return 3
        client = with_card(Client(args.base_url, **sampling_from(args)), args)
    ask = bench.ask_from(args.ask) if args.ask else bench.asking(
        graph, shortlist=args.shortlist, store=args.store or None,
        embed_url=args.embed_url, embed_model=args.embed_model, margin=args.margin)
    where = args.graph or "the invented community"
    found = getattr(ask, "finder", "")
    print(f"{args.label}: {len(questions)} questions over {where}"
          + (f", look_up by {found}" if found else "")
          + (f", {args.shortlist} handed to it first" if args.shortlist else ""))
    rows = bench.measure(ask, questions, label=args.label, client=client, log=print,
                         graph=graph,
                   per_question=args.per_question)
    key = save(args.kept, rows,
               held={**bench.footprint(args.base_url), "sampling": client.sampling,
                     "graph": _which(graph), "finder": found})
    print(f"kept as {key}")
    if args.smoke:
        table(read_back(args.kept, [key]))
    return 0


# The flags `sweep --fleet` takes off the line before handing it to a peer: what is about
# this machine's session, not about the measuring.
_NOT_FOR_A_PEER = ("--fleet", "--detach", "--no-queue")
_NOT_FOR_A_PEER_VALUED = ("--peers", "--serve", "--serve-draft")


def fleet_jobs(argv: Sequence[str], models: Sequence[str], *, commit: str) -> list[dict[str, Any]]:
    """One job per model: this same command line with that one ``--serve`` (and its
    positional ``--serve-draft``, when one was given), on ``commit``.

    Every other flag rides along unchanged -- the questions, the store, the sample, every
    ``--also`` -- so a peer measures exactly what this machine would have. ``commit`` is
    the short sha, and ``dirty`` says whether the tree had changes: a peer on another
    commit is measuring other code, and both ends refuse it.
    """
    rest: list[str] = []
    heads: list[str] = []
    skip = False
    for word in argv:
        if skip:
            skip = False
            continue
        if word in _NOT_FOR_A_PEER:
            continue
        flag, sep, value = word.partition("=")
        if flag in _NOT_FOR_A_PEER_VALUED:
            if not sep:
                skip = True
            continue
        rest.append(word)
    heads = list(_values_of(argv, "--serve-draft"))
    sha, _, dirtiness = commit.partition(" ")
    out = []
    for n, model in enumerate(models):
        line = [*rest, "--serve", model]
        if n < len(heads):
            line += ["--serve-draft", heads[n]]
        out.append({"model": model, "argv": line, "commit": sha, "dirty": bool(dirtiness)})
    return out


def _values_of(argv: Sequence[str], flag: str) -> list[str]:
    """Every value ``flag`` was given on ``argv``, ``--flag V`` and ``--flag=V`` alike."""
    out: list[str] = []
    words = list(argv)
    for n, word in enumerate(words):
        if word == flag and n + 1 < len(words):
            out.append(words[n + 1])
        elif word.startswith(flag + "="):
            out.append(word.partition("=")[2])
    return out


def _planned(plan: Any) -> list[dict[str, Any]]:
    """The fleet's plan as one record per job, whatever shape `plan` gave it: a list of
    mappings as they are, a mapping of model to peer as ``{"model", "peer"}`` records."""
    if isinstance(plan, Mapping):
        return [{"model": str(k), "peer": v} for k, v in plan.items()]
    return [dict(one) if isinstance(one, Mapping) else {"model": str(one)}
            for one in (plan or ())]


def _fleet_sweep(args: Any) -> int:
    """`sweep --fleet`: the jobs, the plan, dispatch, wait, gather, show.

    The fleet side is `ml_stack.fleet.bench` -- `plan(models, peers)`, `dispatch(jobs)`,
    `wait(handles)`, `gather(handles, into=store)` -- imported by name here so this
    machine's sweep needs none of it. The plan is printed before anything is dispatched,
    and a peer the plan says is on another commit ends the sweep before it starts: the
    daemon refuses too, but finding out from four peers' logs is later than from one line.
    """
    models = [str(m) for m in (getattr(args, "serve", []) or [])]
    if not models:
        print("error: --fleet spreads --serve models over the fleet; pass --serve MODEL for "
              "each", file=sys.stderr)
        return 2
    mine = _commit()
    if not mine:
        print("error: --fleet needs to know this checkout's commit, and git would not say",
              file=sys.stderr)
        return 2
    fleet = importlib.import_module("ml_stack.fleet.bench")
    missing = [name for name in ("plan", "dispatch", "wait", "gather") if not hasattr(fleet, name)]
    if missing:
        print(f"error: ml_stack.fleet.bench has no {', '.join(missing)}; the fleet side of "
              f"the bench is not in this build", file=sys.stderr)
        return 2
    jobs = fleet_jobs(list(getattr(args, "_argv", None) or []), models, commit=mine)
    peers = [p.strip() for p in str(getattr(args, "peers", "") or "").split(",") if p.strip()]
    planned = _planned(fleet.plan(models, peers or None))
    sha = mine.partition(" ")[0]
    print(f"plan: {len(jobs)} job(s) on commit {mine}" + (f" over {', '.join(peers)}" if peers
                                                          else ""))
    for one in planned:
        theirs = str(one.get("commit") or "")
        print(f"  {one.get('model', '?')} -> {one.get('peer') or one.get('host') or '?'}"
              + (f" ({theirs})" if theirs else ""))
        if theirs and theirs.partition(" ")[0] != sha:
            print(f"error: {one.get('peer') or one.get('host') or 'a peer'} is on commit "
                  f"{theirs}, this checkout is on {mine}; a peer measuring other code is "
                  f"refused, and its daemon would refuse too", file=sys.stderr)
            return 2
    where = {str(one.get("model")): one for one in planned if one.get("model")}
    for job in jobs:
        peer = (where.get(job["model"]) or {}).get("peer")
        if peer is not None:
            job["peer"] = peer
    handles = fleet.dispatch(jobs)
    fleet.wait(handles)
    fleet.gather(handles, into=args.kept)
    print()
    table(bench._kept(args.kept))
    return 0


# Which subcommands put load on the GPU, and so must never overlap with each other.
MEASURING = ("run", "sweep", "drafts", "concurrent", "extract")


# Windows has no sessions; a child that survives its parent's console is asked for by flag.
_WINDOWS_DETACHED = 0x00000200 | 0x00000008     # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS


def measuring_file() -> Path:
    """Where the detached measurement's pid, argv, log and start time are written."""
    return bench.HOME / "measuring.json"


def measuring() -> dict[str, Any] | None:
    """The detached measurement still running, or None. Read from `measuring_file`; a
    record whose pid has gone is a measurement that finished, not one that is running."""
    from ml_stack.serve.process import pid_exists

    try:
        held = json.loads(measuring_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(held, dict) or not pid_exists(held.get("pid")):
        return None
    return held


def _named_in(argv: Sequence[str]) -> str:
    """What to call a detached run's log: its label, or the first model it measures."""
    try:
        args = _parser().parse_args(list(argv))
    except SystemExit:
        return "bench"
    named = (getattr(args, "label", "") or next(iter(getattr(args, "serve", []) or []), "")
             or next(iter(getattr(args, "on", []) or []), "").partition("=")[0]
             or getattr(args, "model", "") or "bench")
    return re.sub(r"[^\w.-]+", "-", str(named).rsplit("/", 1)[-1].removesuffix(".gguf"))[:40]


def detach(argv: Sequence[str]) -> Path:
    """Run ``ml-stack-bench argv`` in the background, owned by no terminal, and return its log.

    A measurement is hours, and a child of a shell -- `nohup`, `&`, a hand-made redirect
    into a scratch directory -- dies with the shell, or with the agent that opened it.
    A ranking sweep was killed that way, thirty minutes in. So the command re-runs itself
    in a new session with its output in a log under the bench's own home, writes down the
    pid it started and what it started, and gives the shell back at once. `status`,
    `tail -f` and `stop` read the same file.
    """
    rest = [a for a in argv if a != "--detach"]
    logs = bench.HOME / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cmd = next((a for a in rest if a in MEASURING), "bench")
    log = logs / f"{cmd}-{_named_in(rest)}-{time.strftime('%Y%m%dT%H%M%S')}.log"
    command = [sys.executable, "-m", "ml_stack.graph.bench", *rest]
    extra: dict[str, Any] = ({"creationflags": _WINDOWS_DETACHED}
                             if platform.system() == "Windows" else {"start_new_session": True})
    started = time.strftime("%FT%T")
    commit = _commit()
    with log.open("ab") as out:
        # the log's first lines are its record -- `history` reads them back when
        # `measuring.json` has moved on to the next run
        out.write((f"argv: {' '.join(rest)}\nstarted: {started}\n"
                   + (f"commit: {commit}\n" if commit else "")).encode("utf-8"))
        out.flush()
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out,
                                 stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1"}, **extra)
    measuring_file().write_text(json.dumps({
        "pid": child.pid, "argv": list(rest), "log": str(log),
        "started": started, "commit": commit}, indent=1), encoding="utf-8")
    return log


def _last_line(log: Path) -> str:
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return next((ln for ln in reversed(lines) if ln.strip()), "")


def status() -> str:
    """What is measuring, or that nothing is. Exit 0 either way: a question, not a check."""
    held = measuring()
    if held is None:
        try:
            last = json.loads(measuring_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "nothing is measuring"
        return (f"nothing is measuring; the last one -- ml-stack-bench "
                f"{' '.join(last.get('argv') or ())} -- started {last.get('started', '?')} and "
                f"has ended.\n  log: {last.get('log', '?')}\n  last: {_last_line(Path(str(last.get('log', ''))))}")
    return (f"measuring since {held.get('started', '?')} (pid {held.get('pid')}):\n"
            f"  ml-stack-bench {' '.join(held.get('argv') or ())}\n"
            f"  log: {held.get('log', '?')}\n"
            f"  last: {_last_line(Path(str(held.get('log', ''))))}")


def _latest_log() -> Path | None:
    held = measuring()
    if held and held.get("log"):
        return Path(str(held["log"]))
    try:
        last = json.loads(measuring_file().read_text(encoding="utf-8"))
        if last.get("log") and Path(str(last["log"])).exists():
            return Path(str(last["log"]))
    except (OSError, ValueError):
        pass
    logs = sorted((bench.HOME / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime) \
        if (bench.HOME / "logs").exists() else []
    return logs[-1] if logs else None


def tail(*, lines: int = 20, follow: bool = False, every: float = 0.5) -> int:
    """Print the end of the current (or latest) log; ``follow`` keeps printing until the
    measurement's pid has gone and the log has been drained."""
    from ml_stack.serve.process import pid_exists

    log = _latest_log()
    if log is None or not log.exists():
        print("no log yet: nothing has been detached", file=sys.stderr)
        return 1
    with log.open("rb") as fh:
        text = fh.read().decode("utf-8", "replace")
        shown = text.splitlines()[-lines:] if lines > 0 else []
        if shown:
            print("\n".join(shown))
        if not follow:
            return 0
        held = measuring() or {}
        pid = held.get("pid")
        try:
            while True:
                more = fh.read().decode("utf-8", "replace")
                if more:
                    print(more, end="", flush=True)
                elif not pid_exists(pid):
                    break
                else:
                    time.sleep(every)
        except KeyboardInterrupt:
            pass
    return 0


def stop(*, wait: float = 60.0) -> str:
    """SIGTERM to the detached measurement -- by pid, never by name -- and wait for it.

    The child's handler turns the signal into a `SystemExit`, so the `serve` block it is
    in runs its exit and takes its model down; `pkill llama-server` would not, and would
    take somebody else's server with it.
    """
    from ml_stack.serve.process import pid_exists

    held = measuring()
    if held is None:
        return "nothing is measuring"
    pid = int(held["pid"])
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return f"pid {pid} had already gone"
    began = time.monotonic()
    while pid_exists(pid) and time.monotonic() - began < wait:
        time.sleep(0.25)
    if pid_exists(pid):
        return (f"asked pid {pid} to stop; it is still running after {wait:.0f}s. Its log: "
                f"{held.get('log', '?')}")
    return f"stopped pid {pid} after {time.monotonic() - began:.1f}s; its log: {held.get('log', '?')}"


def _stop_on_sigterm(signum: int, frame: Any) -> None:
    """Turn SIGTERM into an exception, so every `with` on the way out runs its exit.

    Says ``[killed]`` first, so the log tells a run that was stopped from one that crashed
    or one that finished -- `history` reads that word."""
    print(f"[killed] SIGTERM ({signum}): stopping, taking any served model down", flush=True)
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    """Measure one thing at a time, waiting for whoever is already measuring.

    Two runs sharing a GPU produce timings that belong to neither, and the old way of
    arranging that -- a `pgrep` loop in the shell before the command -- could not work and
    said nothing when it did not. Waiting belongs here, where it can be announced.

    A measuring command also takes SIGTERM as an exception rather than as death: `stop`
    sends it, and a server put up inside a `with serve(...)` comes down on the way out
    instead of staying up under nobody.
    """
    from ml_stack.lock import Busy, only_one

    known = {*MEASURING, "show", "prepare", "forget", "status", "tail", "stop", "history"}
    cmd = next((a for a in (argv if argv is not None else sys.argv[1:]) if a in known), "")
    if cmd not in MEASURING:
        return _main(argv)

    rest = list(argv if argv is not None else sys.argv[1:])
    if "--detach" in rest:
        log = detach(rest)
        print(f"measuring in the background; log: {log}\n"
              f"  ml-stack-bench status   -- what is measuring, and its last line\n"
              f"  ml-stack-bench tail -f  -- follow the log\n"
              f"  ml-stack-bench stop     -- end it, taking its server down")
        return 0
    refuse = "--no-queue" in rest
    rest = [a for a in rest if a != "--no-queue"]
    if "--no-selfcheck" not in rest:
        # Before the prefetch and before the lock, on purpose: this is the run itself,
        # with the model and the machine faked, and what it catches -- a flag the client
        # does not take, a way that never reaches the store -- cost an 87G load the day
        # it was left to a person to remember (2026-09-02). It needs no GPU and holds
        # nobody up.
        from ml_stack.graph.bench.selfcheck import SelfCheckFailed, selfcheck

        began = time.monotonic()
        try:
            proved = selfcheck(rest)
        except SelfCheckFailed as why:
            print(f"selfcheck: FAILED -- this command would not get through with a "
                  f"scripted model, so nothing was loaded:\n{why}", file=sys.stderr)
            print("error: the self-check failed; fix it, or pass --no-selfcheck to "
                  "repeat a run whose path the last one proved", file=sys.stderr)
            return 4
        print(f"selfcheck: ok ({time.monotonic() - began:.1f} s) -- {proved}", flush=True)
    if "--no-prefetch" not in rest:
        # Before the lock, on purpose: a download is minutes of network and no GPU, and
        # holding the measuring lock through it makes the next run wait for the Hub.
        bench.prefetch(references_in(_parser().parse_args(rest)))
    previous = None
    try:
        previous = signal.signal(signal.SIGTERM, _stop_on_sigterm)
    except ValueError:
        pass                                 # not the main thread: nothing to hand a signal
    try:
        with only_one(bench.HOME / "measuring.lock", wait=not refuse,
                      announce=lambda line: print(line, file=sys.stderr)):
            return _main(rest)
    except Busy as why:
        print(f"error: {why}. Another measurement is running; wait for it, or pass "
              f"--no-queue to fail fast rather than queue.", file=sys.stderr)
        return 3
    except SmokeFailed as why:
        print(f"error: smoke failed, so the run did not start: {why}", file=sys.stderr)
        return 1
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
