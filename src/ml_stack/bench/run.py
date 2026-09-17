"""``ml-stack-bench`` -- the subcommands, the measuring lock, and running in the background.

Every subcommand parses its arguments and prints; the work is in `ml_stack.bench.ops` and
the modules beside it, and its flags are in `ml_stack.bench.options`. `main` is what the
console script calls -- the self-check and the prefetch before the lock, the lock itself,
SIGTERM taken as an exit so a served model comes down, and ``--detach`` re-running the
command in its own session with `status`, `tail` and `stop` reading the same file.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import signal
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.served`,
# `bench.home_dir()` -- so anything patchable is looked up there at call
# time, never bound here at import.
from ml_stack import bench, hub, jobs
from ml_stack.bench import ops, options
from ml_stack.bench.askings import _asked, asking_from, halves, sampling_from, with_card
from ml_stack.bench.detail import missed, shape
from ml_stack.bench.estimate import estimate
from ml_stack.bench.frontier import plot, rates
from ml_stack.bench.holding import _idle
from ml_stack.bench.keep import SMOKE, empties, forget, read_back, resumable, save
from ml_stack.bench.measure import concurrent
from ml_stack.bench.ops import Refused
from ml_stack.bench.progress import note_beside_the_run, status, stop, tail
from ml_stack.bench.questions import _how_many, read_questions, sample
from ml_stack.bench.score import _which, export, ranking
from ml_stack.bench.serve import SmokeFailed, drafts, references_in, smoked
from ml_stack.bench.show import compare, table
from ml_stack.bench.underway import MEASURING, detach, ended, remember
from ml_stack.command import Group
from ml_stack.log import say, warn
from ml_stack.serve.profile import ASK

__all__ = ["COMMANDS", "HANDED_OVER", "main"]

# allow_abbrev=False on every parser here: a flag that is documented but not defined must
# be refused by name, not bound by prefix to whichever neighbour shares its first letters
# -- `--short` became `--shortlist` that way and the error blamed the wrong flag.
COMMANDS = Group("ml-stack-bench",
                 "Time a set of questions through a graph, and compare two runs.",
                 allow_abbrev=False)

# The subcommands whose module has a `main(argv)` of its own: `ml-stack-bench NAME ARGS`
# hands ARGS to it as they were typed. `standard` takes the measuring lock itself, so
# neither is a MEASURING command here, and neither goes through this parser's lock.
HANDED_OVER = ("standard", "animate")


class _NotIdle(Exception):
    """A server a sweep would measure is busy; `_idle` has already said which."""


def _module(name: str) -> Any:
    """The ``ml_stack.bench`` module ``name``, for a subcommand that owns its own flags."""
    return importlib.import_module(f"ml_stack.bench.{name}")


def _parser() -> argparse.ArgumentParser:
    """The command line of ``ml-stack-bench``, built once per call and shared with `detach`,
    which needs a label out of an argv before handing it to the child."""
    return COMMANDS.parser()


def _after(argv: Sequence[str], cmd: str) -> list[str]:
    """The words after ``cmd`` on ``argv``: what a handed-over module's main is given."""
    words = list(argv)
    return words[words.index(cmd) + 1:] if cmd in words else []


def _main(argv: list[str] | None = None) -> int:
    """``ml-stack-bench`` -- what a change to the asking costs, and whether it was worth it."""
    words = list(argv if argv is not None else sys.argv[1:])
    args = _parser().parse_args(words)
    # the line as given, for `sweep --fleet` to hand each peer the same line with one model
    args._argv = words
    return _run(args)


def _run(args: Any) -> int:
    """`_main` after the parse, so a dry run can hand in a namespace it has rewritten."""
    return int(args.run(args) or 0)


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
    say(f"smoke: the same {args.cmd} on {SMOKE} question(s) first -- ask, score, save, "
        f"read back -- before the run proper")
    code = _run(trial)
    if code != 0:
        raise SmokeFailed(f"{args.cmd} --smoke returned {code}")
    smoked([r for r in bench._kept(args.kept) if r["key"] not in before],
           f"{args.cmd} smoke")
    say("smoke: ok\n")


def _questions_and_graph(args: Any, *, everything: Any = None) -> tuple[Any, Any, Any]:
    """The questions a run asks, the sample of them it will ask, and the graph they are
    about: the invented community unless ``--questions`` and ``--graph`` name others."""
    from ml_stack.graph.community import QUESTIONS
    from ml_stack.graph.community import graph as invented

    asked = everything if everything is not None else (
        read_questions(args.questions) if args.questions else QUESTIONS)
    graph = (json.loads(Path(args.graph).expanduser().read_text())
             if args.graph else invented())
    return asked, sample(asked, _how_many(args)), graph


@COMMANDS.command("run", help="ask every question once and keep what it cost",
                  options=options.run_options, allow_abbrev=False)
def cmd_run(args: Any) -> int:
    if wants_smoke(args):
        smoke_first(args)
    _, questions, graph = _questions_and_graph(args)
    if not questions:
        warn(f"error: no questions in {args.questions}")
        return 2
    if args.client:
        client = bench.ask_from(args.client)()
    else:
        from ml_stack.client import Client

        if not _idle(args.base_url, args):
            return 3
        client = with_card(Client(args.base_url, **sampling_from(args)), args)
    ask = bench.ask_from(args.ask) if args.ask else bench.asking(
        graph, how=asking_from(args), shortlist=args.shortlist, store=args.store or None,
        embed_url=args.embed_url, embed_model=args.embed_model, margin=args.margin)
    where = args.graph or "the invented community"
    found = getattr(ask, "finder", "")
    say(f"{args.label}: {len(questions)} questions over {where}"
        + (f", look_up by {found}" if found else "")
        + (f", {args.shortlist} handed to it first" if args.shortlist else ""))
    rows = bench.measure(ask, questions, label=args.label, client=client, log=print,
                         trace=getattr(args, "trace", None),
                         graph=graph,
                         per_question=args.per_question)
    key = save(args.kept, rows,
               server={**bench.footprint(args.base_url), "sampling": client.sampling,
                       "graph": _which(graph), "finder": found},
               asking=getattr(ask, "asking", None), workload=ASK)
    say(f"kept as {key}")
    if args.smoke:
        table(read_back(args.kept, [key]))
    return 0


@COMMANDS.command("drafts", help="serve one model with each draft head in turn "
                                 "and measure what each is worth",
                  options=options.drafts_options, allow_abbrev=False)
def cmd_drafts(args: Any) -> int:
    from ml_stack.graph.community import QUESTIONS
    from ml_stack.graph.community import graph as invented

    everything = read_questions(args.questions) if args.questions else QUESTIONS
    asked = sample(everything, SMOKE if getattr(args, "smoke", False) else args.sample)
    before = {r["key"] for r in bench._kept(args.kept)}
    model = str(hub.located(args.model, loose=True) or args.model)
    rows = drafts(ops.swept(args, model, None, context=args.context, head=None,
                            port=args.port),
                  args.draft or [""], asked, invented(),
                  binary=args.binary,
                  kept=args.kept, store=args.store or None,
                  embed_url=args.embed_url, embed_model=args.embed_model,
                  n_max=list(getattr(args, "n_max", []) or []) or [None],
                  per_request=False if getattr(args, "server_per_depth", False) else None,
                  smoke=sample(everything, SMOKE) if wants_smoke(args) else ())
    say()
    if getattr(args, "smoke", False):
        saved = [r["key"] for r in bench._kept(args.kept) if r["key"] not in before]
        table(read_back(args.kept, saved))
    else:
        table(bench._kept(args.kept))
    return 0 if rows else 1


@COMMANDS.command("concurrent",
                  help="ask N conversations of T turns each at the same time, and "
                       "see what the waiting, the memory and the accuracy cost",
                  options=options.concurrent_options, allow_abbrev=False)
def cmd_concurrent(args: Any) -> int:
    if wants_smoke(args):
        smoke_first(args)
    _, questions, graph = _questions_and_graph(args)
    if not questions:
        warn(f"error: no questions in {args.questions}")
        return 2
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
    ask = bench.asking(graph, how=asking_from(args), store=args.store or None,
                       embed_url=args.embed_url, embed_model=args.embed_model)
    where = args.graph or "the invented community"
    say(f"{args.label}: {many} conversations of {long} turn(s) at once over {where}, "
        f"look_up by {ask.finder}")
    rows, measured = concurrent(ask, questions, conversations=many, turns=long,
                                label=args.label, client=client, graph=graph,
                                base_url="" if args.client else args.base_url, log=print,
                                per_question=args.per_question)
    at = measured["concurrency"]
    slots = at.get("slots") or 0
    say(f"  {at['seconds']:.1f}s for all of it"
        + (f", {at['queued']:.1f}s of that queued"
           if slots and many > slots and at.get("queued") is not None else "")
        + (f", {slots} slot(s)" if slots > 0 else ""))
    key = save(args.kept, rows,
               server={**measured, "sampling": dict(getattr(client, "sampling", {}) or {}),
                       "graph": _which(graph), "finder": ask.finder},
               asking=getattr(ask, "asking", None), workload=ASK)
    say(f"kept as {key}")
    if args.smoke:
        table(read_back(args.kept, [key]))
    return 0


@COMMANDS.command("prepare", help="put a graph in a store and index and embed it",
                  options=options.prepare_options, allow_abbrev=False)
def cmd_prepare(args: Any) -> int:
    from ml_stack.graph.community import graph as invented

    graph = json.loads(Path(args.graph).expanduser().read_text()) if args.graph else invented()
    if getattr(args, "mix", False):
        from ml_stack.bench.questions import mix
        from ml_stack.graph.community import QUESTIONS

        everything = read_questions(args.questions) if args.questions else QUESTIONS
        counts = mix(everything, graph)
        scored = sum(1 for q in everything if q.get("expect"))
        say(f"{len(everything)} asked, {scored} scored")
        for kind, how_many in counts.items():
            say(f"  {kind:12} {how_many:4}  {how_many / len(everything):6.1%}")
        return 0
    counted = ops.prepare(args.store, graph, embed_url=args.embed_url,
                          embed_model=args.embed_model)
    say(f"{args.store}: {counted['nodes']} nodes, {counted['edges']} edges, word index built")
    if counted["embedded"] is None:
        say("  no --embed-url, so no vectors: search will be words only")
        return 0
    say(f"  {counted['embedded']} embedded")
    return 0


def _fleet_sweep(args: Any) -> int:
    """``sweep --fleet``: the plan said, the jobs dispatched, waited for and gathered."""
    peers = [p.strip() for p in str(getattr(args, "peers", "") or "").split(",") if p.strip()]
    try:
        planned = ops.fleet_planned(list(getattr(args, "_argv", None) or []),
                                    [str(m) for m in (getattr(args, "serve", []) or [])],
                                    peers=peers)
    except Refused as why:
        for line in why.said:
            say(line)
        warn(why.error)
        return 2
    for line in planned.lines:
        say(line)
    ops.fleet_measure(planned.jobs, into=args.kept)
    say()
    table(bench._kept(args.kept))
    return 0


def _served_by_the_sweep(args: Any, questions: Any, graph: Any, already: Any,
                         smoke: Any) -> list[str]:
    """Every ``--serve``'d model put up, asked both halves on one load, and taken down;
    the keys of the runs it kept."""
    from ml_stack.serve.backend import ServerFailed

    saved: list[str] = []
    total_context = args.context or 32768 * max(1, args.parallel)
    # `wanted`, not `named`: the loop variable was `named` once, which rebound the
    # (name, url) list built from --on to the last model's name, and the summary below
    # then unpacked its characters. Every `sweep --serve` answered its questions and
    # crashed while summarising, and the smoke run is what caught it.
    for n, wanted in enumerate(getattr(args, "serve", []) or []):
        model = str(hub.located(wanted, loose=True) or wanted)
        heads = getattr(args, "serve_draft", []) or []
        head = heads[n] if n < len(heads) else ""
        if head.lower() == "auto":
            # the one resolver (`hub.choose_head`): told which binary will serve, so
            # a head that borrows its target's embeddings is withheld from mainline
            # rather than found out at the far end of an 87G load
            chosen = hub.choose_head(model, binary=args.binary or None)
            head = chosen.path
            say(f"    draft head: {head or 'none'} -- {chosen.why}"
                + (f"\n      {chosen.note}" if chosen.note else ""))
        # the label's stem: the model's file, or what --serve-label says it is; then
        # -nodraft for a model served without its head, and the suffix asked for
        stem = ((str(getattr(args, "serve_label", "") or "")
                 or str(model).rsplit("/", 1)[-1].removesuffix(".gguf")[:14])
                + ("-nodraft" if getattr(args, "no_draft", False) else "")
                + str(getattr(args, "label_suffix", "") or ""))
        # Both halves -- plain, and shortlisted where `--shortlist-for` allows it --
        # and every `--also` of each, asked of one load. Loading twice per model was
        # how this began, and the second load measured nothing about the asking.
        parts = halves(args, f"{wanted} {model}")
        say(f"\n{stem}: " + ", ".join(suffix for suffix, _ in parts))
        # A port nothing answers on is exactly what --serve expects, so the
        # "would not say whether it is busy" note is noise here. Only a port
        # somebody is actually using should stop us.
        if bench.busy(f"http://127.0.0.1:{args.serve_port}") > 0 and not _idle(
                f"http://127.0.0.1:{args.serve_port}", args):
            raise _NotIdle
        # `--context` is the total across slots, which is what `-c` takes and what
        # ServerSpec means by it. Dividing by the slot count served a model at a
        # quarter of the context every other run had, and the only thing that said
        # so was the `ctx` column reading 8k where the rest read 32k.
        before = {r["key"] for r in bench._kept(args.kept)}
        # The settings that scored best fill every flag this sweep did not set: the head
        # at the length that measured best, the build that loads it, the cache type,
        # the thinking budget, the raw flags, and the asking. Adam: "if a model
        # has a drafting head that speeds it up at some config, always use it at that
        # config (be sure to report it)". --no-profile serves it bare.
        chosen = ops.swept(args, model, ops.measured_run(args, model, head, heads, n),
                           context=total_context, port=args.serve_port,
                           head=head if n < len(heads) else None)
        try:
            bench.served(chosen, questions, graph, label=stem,
                         askings=_asked(args, parts),
                         binary=args.binary or "",
                         kept=args.kept,
                         store=args.store or None, embed_url=args.embed_url,
                         embed_model=args.embed_model,
                         already=already,
                         trace=getattr(args, "trace", None),
                         smoke=smoke)
        except ServerFailed as why:
            # A model that will not load -- a head the build cannot read, a tensor it
            # does not know -- ends that model, not the sweep. Measured 2026-09-01: one
            # such load took gpt-oss-120b's measurement down with it, twice.
            say(f"    {stem} did not load; moving on:\n"
                + "\n".join(f"      {line}" for line in str(why).splitlines()[:6]))
            continue
        saved += [r["key"] for r in bench._kept(args.kept) if r["key"] not in before]
    return saved


def _measured_on(args: Any, named: Sequence[tuple[str, str]], questions: Any, graph: Any,
                 already: Any) -> list[str]:
    """Every ``--on`` server measured both halves; the keys of the runs it kept."""
    from ml_stack.bench.backends import client_for, http_of

    saved: list[str] = []
    total_context = args.context or 32768 * max(1, args.parallel)
    for name, url in named:
        for suffix, shortlist in halves(args, name):
            label = f"{name}-{suffix}"
            if already is not None and already(label):
                say(f"skipping {label}: kept at {already(label).get('at', '?')}")
                continue
            ask = bench.asking(graph, how=asking_from(args), shortlist=shortlist,
                               store=args.store or None, embed_url=args.embed_url,
                               embed_model=args.embed_model, margin=args.margin)
            say(f"\n{label} on {url}, look_up by {ask.finder}")
            if not _idle(http_of(url), args):
                raise _NotIdle
            # the client for whatever program the URL names -- a llama-server, Ollama,
            # an OpenAI-style server -- at the sweep's context
            asking_with = with_card(client_for(url, timeout=args.per_question,
                                               context=total_context,
                                               **sampling_from(args)), args)
            # what it will actually send, card and overrides together: a run measured at
            # one temperature against a run at another is two measurements, and the only
            # way to know later is to write it down now
            used = dict(asking_with.sampling)
            rows = bench.measure(ask, questions, label=label, client=asking_with,
                                 trace=getattr(args, "trace", None),
                                 log=print,
                                 graph=graph, per_question=args.per_question)
            saved.append(save(args.kept, rows,
                              server={**bench.footprint(url), "sampling": used,
                                      "graph": _which(graph), "finder": ask.finder},
                              asking=getattr(ask, "asking", None), workload=ASK))
    return saved


@COMMANDS.command("sweep", help="run every model, with and without a shortlist",
                  options=options.sweep_options, allow_abbrev=False)
def cmd_sweep(args: Any) -> int:
    from ml_stack.bench.backends import parse_on

    named = []
    for one in args.on:
        try:
            name, url, _ = parse_on(one)
        except ValueError as why:
            warn(f"error: {why}")
            return 2
        named.append((name, url))
    if not named and not getattr(args, "serve", []):
        warn("error: nothing to measure; pass --on NAME=URL for a server that is "
             "already up, or --serve MODEL to put one up")
        return 2
    if getattr(args, "fleet", False):
        return _fleet_sweep(args)
    everything, questions, graph = _questions_and_graph(args)
    # the smoke: two questions first, of every model. The servers somebody else
    # started are smoked as a sweep of their own before anything is served, and each
    # served model smokes as it comes up, so a load is paid once
    smoking = wants_smoke(args)
    if smoking and named:
        standing = argparse.Namespace(**vars(args))
        standing.serve = []
        smoke_first(standing)
    already = (resumable(args.kept, questions=len(questions),
                         context=args.context or 32768 * max(1, args.parallel),
                         parallel=getattr(args, "parallel", 1), since=args.since)
               if args.resume else None)
    try:
        saved = _served_by_the_sweep(args, questions, graph, already,
                                     sample(everything, SMOKE) if smoking else ())
        saved += _measured_on(args, named, questions, graph, already)
    except _NotIdle:
        return 3
    say()
    table(read_back(args.kept, saved) if args.smoke else bench._kept(args.kept))
    return 0


COMMANDS.borrow(lambda sub: _module("speed").add_arguments(sub),
                lambda args: _module("speed").main(args),
                options=lambda: (*options.measuring_options(), *options.checking()))
COMMANDS.borrow(lambda sub: _module("extract").add_arguments(sub),
                lambda args: _module("extract").main(args),
                options=options.checking)


@COMMANDS.command("show", help="compare two runs, or list what is kept",
                  options=options.show_options, allow_abbrev=False)
def cmd_show(args: Any) -> int:
    # an extraction run is kept in the same store and is not an answering run: it has
    # no questions to score, and its table is its own
    from ml_stack.bench import extract as bench_extract
    from ml_stack.bench import speed as bench_speed

    kept = ops.kept_for(args.kept, last=int(getattr(args, "last", 0) or 0),
                        since=str(getattr(args, "since", "") or ""))
    if getattr(args, "speed", False):
        bench_speed.speed_table(kept.speed)
        return 0
    answering = kept.answering
    if getattr(args, "trace", None) is not None:
        bench.transcript(answering, args.trace, getattr(args, "question", "") or "")
        return 0
    if getattr(args, "by", "") == "serving":
        bench.by_serving(answering)
        return 0
    if getattr(args, "extract", False):
        bench_extract.table(kept.extracted)
        return 0
    if args.compare:
        say(compare(args.kept, *args.compare))
        return 0
    if args.rank:
        ranking(answering, args.rank, noise=args.noise / 100)
        say(args.rank)
        return 0
    if args.export:
        say(export(answering, args.export,
                   anyway=getattr(args, "export_anyway", False)))
        return 0
    if args.shape:
        from ml_stack.graph.community import QUESTIONS
        from ml_stack.graph.community import graph as invented

        questions = read_questions(args.questions) if getattr(args, "questions", "") \
            else QUESTIONS
        shape(questions, invented())
        return 0
    if args.plot:
        say(plot(answering, args.plot, cost=args.cost, noise=args.noise / 100))
        return 0
    if args.rates:
        rates(answering, cost=args.cost, noise=args.noise / 100)
        return 0
    if args.detail is not None:
        missed([r for r in answering if not args.detail or r.get("label") == args.detail],
               everything=args.all, among=answering)
        return 0
    table(answering)
    if kept.extracted:
        say()
        bench_extract.table(kept.extracted)
    hollow = empties(args.kept)
    if hollow:
        say(f"{len(hollow)} empty run(s) skipped -- ml-stack-bench forget --empty "
            f"removes them")
    return 0


@COMMANDS.command("report",
                  help="everything measured so far as one document: how each "
                       "model was asked, what a draft head was worth, how much "
                       "memory it wants, and what to serve",
                  options=options.report_options, allow_abbrev=False)
def cmd_report(args: Any) -> int:
    from ml_stack.bench.report import main as reporting

    return reporting(args)


@COMMANDS.command("forget",
                  help="delete kept runs: the empty ones, or every run of one label",
                  options=options.forget_options, allow_abbrev=False)
def cmd_forget(args: Any) -> int:
    if not args.empty and not args.label:
        warn("error: say what to forget: --empty, or a label")
        return 2
    if args.empty:
        went = forget(args.kept, empty=True)
        say(f"{len(went)} empty run(s) removed" if went else "no empty runs")
    if args.label:
        if not args.yes:
            would = [r["key"] for r in bench.runs(args.kept, args.label)]
            say("\n".join(would) if would else f"no run labelled {args.label!r}")
            if would:
                say(f"{len(would)} run(s) would go; pass --yes to delete them")
            return 0
        went = forget(args.kept, label=args.label)
        say(f"{len(went)} run(s) labelled {args.label!r} removed")
    return 0


@COMMANDS.command("status",
                  help="whether something is measuring, since when, with what, and where "
                       "its log is. Exits 0 either way", allow_abbrev=False)
def cmd_status(args: Any) -> int:
    say(status())
    return 0


@COMMANDS.command("tail", help="the log of the current measurement, or the latest",
                  options=options.tail_options, allow_abbrev=False)
def cmd_tail(args: Any) -> int:
    return tail(lines=args.n, follow=args.follow)


@COMMANDS.command("stop",
                  help="end the detached measurement: SIGTERM to its pid, so it takes down "
                       "any server it put up, then wait up to a minute. Never by name",
                  allow_abbrev=False)
def cmd_stop(args: Any) -> int:
    say(stop())
    return 0


@COMMANDS.command("wait",
                  help="block until the detached measurement has ended, saying so "
                       "every minute -- so the next command can follow it "
                       "(ml-stack-bench wait && ml-stack-bench report --profile)",
                  options=options.wait_options, allow_abbrev=False)
def cmd_wait(args: Any) -> int:
    return jobs.wait("bench", every=args.every, home=bench.home_dir() / "jobs")


# The positional is `label` rather than `file`: `_named_in` reads it, so a detached
# queue's log is named after the queue file instead of "bench".
@COMMANDS.command("queue",
                  help="run an evening of measurements from a file: one "
                       "ml-stack-bench line per step, smoke:/then: pairs, "
                       "set VAR= and ${VAR}, one at a time through the "
                       "measuring lock",
                  options=options.queue_options, allow_abbrev=False)
def cmd_queue(args: Any) -> int:
    # The queue holds no lock: each of its steps is its own `ml-stack-bench`, and takes
    # the measuring lock itself, so a step of a queue and a run started by hand still
    # wait for each other.
    from ml_stack.bench.queue import QueueError, run_queue

    if args.detach:
        log = detach(getattr(args, "_argv", None) or sys.argv[1:])
        say(f"the queue is running in the background; log: {log}\n"
            f"  ml-stack-bench status   -- the step it is on, and what is left\n"
            f"  ml-stack-bench tail -f  -- follow the log\n"
            f"  ml-stack-bench stop     -- end the queue and the step inside it")
        return 0
    try:
        return run_queue(args.label, dry_run=args.dry_run, resume=args.resume,
                         yes=args.yes, ceiling=args.ceiling)
    except QueueError as why:
        warn(f"error: {why}")
        return 2


COMMANDS.borrow(lambda sub: _module("comparison").add_arguments(sub),
                lambda args: _module("comparison").main(args))


def _handed_over(args: Any) -> int:
    """A subcommand whose module has a ``main(argv)`` of its own, given the words after it."""
    return int(_module(args.cmd).main(
        _after(list(getattr(args, "_argv", None) or []), args.cmd)))


COMMANDS.add("standard", _handed_over,
             help="the standard sets -- GSM8K, MMLU-Pro, IFEval, HumanEval -- through "
                  "lm-evaluation-harness against a chat endpoint, one JSON per "
                  "configuration; takes the measuring lock itself",
             allow_abbrev=False, conflict_handler="resolve",
             parents=[_module("standard")._parser()])
COMMANDS.add("tree", lambda args: int(_module("tree").run(args)),
             help="tree speculative decoding on MLX: the lossless witness against plain "
                  "greedy decoding, and speed by drafter beside llama.cpp; takes the "
                  "measuring lock itself",
             options=options.tree_options, allow_abbrev=False)
COMMANDS.add("animate", _handed_over,
             help="a comparison document as an animated graphic, with manim",
             options=options.animate_options, allow_abbrev=False)

COMMANDS.borrow(
    lambda sub: _module("history").add_arguments(
        sub.add_parser("history", allow_abbrev=False,
                       help="every measurement the logs remember: when, how long, "
                            "how it ended, the estimate beside the actual, and the "
                            "runs it kept")),
    lambda args: _module("history").run(args))


def _estimated(rest: Sequence[str]) -> int:
    """Print what ``ml-stack-bench rest`` should take -- one ``estimate:`` line per model
    and the total, read from the runs kept in its store -- and return 5 when that is over
    the ceiling and ``--yes`` was not given, with the refusal on stderr; 0 otherwise."""
    args = _parser().parse_args([a for a in rest if a not in ("--detach", "--no-queue")])
    guess = estimate(args, bench._kept(args.kept))
    for line in guess.lines():
        say(line, flush=True)
    if guess.over and not getattr(args, "yes", False):
        warn(guess.refusal())
        return 5
    return 0


def _stop_on_sigterm(signum: int, frame: Any) -> None:
    """Turn SIGTERM into an exception, so every `with` on the way out runs its exit.

    Says ``[killed]`` first, so the log tells a run that was stopped from one that crashed
    or one that finished -- `history` reads that word."""
    say(f"[killed] SIGTERM ({signum}): stopping, taking any served model down", flush=True)
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

    # every subcommand there is, so a *value* that happens to read like one -- `report
    # --model run` -- is not mistaken for the command and sent through the lock
    known = {*MEASURING, *HANDED_OVER, "show", "report", "prepare", "forget", "status",
             "tail", "stop", "wait", "history", "compare"}
    cmd = next((a for a in (argv if argv is not None else sys.argv[1:]) if a in known), "")
    if cmd not in MEASURING:
        return _main(argv)

    rest = list(argv if argv is not None else sys.argv[1:])
    if "--detach" in rest:
        # estimated and, over the ceiling, refused here in the terminal: a refusal at the
        # top of a log nobody is watching is not a refusal. The child says it again into
        # the log, which is where `history` reads it beside the actual.
        refused = _estimated(rest)
        if refused:
            return refused
        log = detach(rest)
        say(f"measuring in the background; log: {log}\n"
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
        from ml_stack.bench.selfcheck import SelfCheckFailed, selfcheck

        began = time.monotonic()
        try:
            proved = selfcheck(rest)
        except SelfCheckFailed as why:
            warn(f"selfcheck: FAILED -- this command would not get through with a "
                 f"scripted model, so nothing was loaded:\n{why}")
            warn("error: the self-check failed; fix it, or pass --no-selfcheck to "
                 "repeat a run whose path the last one proved")
            return 4
        say(f"selfcheck: ok ({time.monotonic() - began:.1f} s) -- {proved}", flush=True)
    # After the self-check and before the prefetch and the lock: what this will cost, from
    # what is kept, and a refusal over the ceiling before a download or a load is paid
    refused = _estimated(rest)
    if refused:
        return refused
    if "--no-prefetch" not in rest:
        # Before the lock, on purpose: a download is minutes of network and no GPU, and
        # holding the measuring lock through it makes the next run wait for the Hub.
        bench.prefetch(references_in(_parser().parse_args(rest)))
    previous = None
    with contextlib.suppress(ValueError):    # not the main thread: nothing to hand a signal
        previous = signal.signal(signal.SIGTERM, _stop_on_sigterm)
    try:
        with only_one(bench.home_dir() / "measuring.lock", wait=not refuse,
                      announce=lambda line: warn(line)):
            remember(rest, pid=os.getpid())
            note_beside_the_run()
            try:
                return _main(rest)
            finally:
                ended()
    except Busy as why:
        warn(f"error: {why}. Another measurement is running; wait for it, or pass "
             f"--no-queue to fail fast rather than queue.")
        return 3
    except SmokeFailed as why:
        warn(f"error: smoke failed, so the run did not start: {why}")
        return 1
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
