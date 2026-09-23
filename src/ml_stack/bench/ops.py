"""What ``ml-stack-bench`` does, as functions that take values and return them.

The `Config` a ``--serve``'d model is measured in (`measured_run`, `swept`,
`serving_fields`), the store a graph is prepared in (`prepare`), what a store holds for
`show` to print (`kept_for`), and the jobs a ``sweep --fleet`` spreads over the peers
(`fleet_jobs`, `fleet_planned`, `fleet_measure`). `ml_stack.bench.run` parses and prints;
`Refused` carries what a command says when it will not act.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.drafted_by()` --
# so anything patchable is looked up there at call time, never bound here at import.
from ml_stack import bench
from ml_stack.bench.askings import sampling_from
from ml_stack.bench.keep import _commit
from ml_stack.bench.counting import PER_QUESTION
from ml_stack.bench.record import of
from ml_stack.bench.score import derived
from ml_stack.bench.show import NOT_ANSWERING
from ml_stack.log import say
from ml_stack.serve.serving import Config, Serving

__all__ = ["Fleeted", "Kept", "Refused", "fleet_jobs", "fleet_measure", "fleet_planned",
           "kept_for", "measured_run", "newest", "prepare", "serving_fields",
           "summarised", "swept"]


class Refused(Exception):
    """A command that will not act: the lines to say first, and the error that stopped it."""

    def __init__(self, error: str, *said: str) -> None:
        super().__init__(error)
        self.error = error
        self.said = tuple(said)


@dataclass(frozen=True, slots=True)
class Kept:
    """What a runs store holds for `show`: every run, and the three kinds it tables apart.

    ``answering`` and ``speed`` are narrowed to ``--last`` and ``--since``; ``everything``
    and ``extracted`` are the store as it stands.
    """

    everything: list[dict[str, Any]] = field(default_factory=list)
    answering: list[dict[str, Any]] = field(default_factory=list)
    extracted: list[dict[str, Any]] = field(default_factory=list)
    speed: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Fleeted:
    """The jobs a ``sweep --fleet`` would dispatch, one `fleet.measuring.Job` per peer, and
    the plan lines that say where each model goes."""

    jobs: dict[Any, Any] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)


def newest(kept: list[dict[str, Any]], *, last: int = 0, since: str = "") -> list[dict[str, Any]]:
    """``kept`` narrowed to what was kept at or after ``since`` and then to the newest
    ``last`` -- the two ways `show` is asked for what just happened."""
    rows = [r for r in kept if not since or str(r.get("at", "")) >= since]
    if last:
        rows = sorted(rows, key=lambda r: str(r.get("at", "")))[-last:]
    return rows


def kept_for(store: str | Path, *, last: int = 0, since: str = "") -> Kept:
    """What ``store`` holds, split into the tables `show` prints: the answering runs, the
    extractions and the speed grids."""
    from ml_stack.bench import extract as bench_extract
    from ml_stack.bench import speed as bench_speed

    everything = bench.runs(store) if Path(store).expanduser().exists() else []
    answering = [r for r in everything if r.get("kind") not in NOT_ANSWERING]
    return Kept(everything=everything,
                answering=newest(answering, last=last, since=since),
                extracted=bench_extract.only(everything),
                speed=newest(bench_speed.only(everything), last=last, since=since))


def summarised(kept: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One record per run with what the table shows -- what was served, how it was asked,
    what it scored and what it cost -- and none of the per-question rows."""
    out = []
    for one in kept:
        got = of(one)
        out.append({"label": got.label, "at": got.at, "key": got.key, "kind": got.kind,
                    "workload": got.workload, "model": got.model, "build": got.build,
                    "head": got.head, "head_ahead": got.head_ahead,
                    "context": got.context, "slots": got.slots,
                    "cache_type": got.cache_type, "reasoning_budget": got.reasoning_budget,
                    "graph": got.graph, "finder": got.finder,
                    "asking": dict(got.asking or {}), "sampling": dict(got.sampling),
                    "host": got.host, "commit": got.commit, "seconds": got.seconds,
                    "questions": len(got.rows), **derived(one)})
    return out


def prepare(store: str | Path, graph: Mapping[str, Any], *, embed_url: str = "",
            embed_model: str = "") -> dict[str, Any]:
    """Write ``graph`` into ``store`` with its word index, and embed it when ``embed_url``
    says where; the nodes, edges and vectors written."""
    from ml_stack.graph.rebuild import replace
    from ml_stack.ingest.embed import embed_store

    counted = dict(replace(str(store), dict(graph)))     # writing builds the word index
    counted["embedded"] = (embed_store(str(store), base_url=embed_url,
                                       model=embed_model or "embed", log=print)
                           if embed_url else None)
    return counted


def measured_run(args: Any, model: str, head: str, heads: Sequence[str], n: int) -> Any:
    """The `Config` the model's profile measured best in, or None with --no-profile or no
    record. Reports the settings it took.

    The head is left out when ``--serve-draft`` named one for this model, since a flag
    beats a record; everything else the record says is on the run, and the sweep's own
    flags are laid over it by `Config.over` afterwards.
    """
    if not getattr(args, "profile", True):
        return None
    try:
        from ml_stack.serve.profile import profile_for
    except ImportError:
        return None
    found = profile_for(str(model))
    if found is None:
        return None
    config = found.config(port=int(getattr(args, "serve_port", 8099) or 8099),
                    slots=int(getattr(args, "parallel", 1) or 1))
    serving = config.serving
    said: dict[str, Any] = {}
    if n >= len(heads) and serving.draft:
        said["draft"] = str(serving.draft)
    if serving.build:
        said["build"] = serving.build
    if serving.cache_type:
        said["cache_type"] = serving.cache_type
    if serving.reasoning_budget is not None:
        said["reasoning_budget"] = int(serving.reasoning_budget)
    if serving.draft_n_max:
        said["draft_n_max"] = int(serving.draft_n_max)
    rest = {k: v for k, v in (("extra_args", tuple(serving.extra_args)),
                              ("mmproj", serving.mmproj)) if v}
    asking = config.asking.said()
    say("    scored best with: " + ", ".join(f"{k}={v}" for k, v in said.items())
        + (f"; also {rest}" if rest else "")
        + (f"; asking {asking}" if asking else ""))
    return config


def swept(args: Any, model: str, measured: Any, *, context: int, head: str | None,
          port: int) -> Any:
    """The `Config` one ``--serve``'d model is measured in: what a record measured, with this
    sweep's own flags laid over it.

    One object rather than twenty keyword arguments, and one place that lays a flag over a
    record, so the lease `served` takes, the asking it asks with and the client it asks with
    cannot say different things. ``context`` is the total across the slots, which is what
    ``-c`` takes; a `Serving` holds it as every slot's share.

    ``head`` is what ``--serve-draft`` named for this model -- ``""`` for the bare model it
    asked for outright -- and None when it named nothing, which is where a record's own
    head stands.
    """
    slots = max(1, int(getattr(args, "parallel", 1) or 1))
    config = measured if measured is not None else Config(serving=Serving(model=str(model)))
    config = config.over(model=str(model), port=int(port), slots=slots,
                   slot_context=max(1, int(context) // slots),
                   timeout=float(getattr(args, "per_question", PER_QUESTION)),
                   terse=bool(getattr(args, "terse", False)),
                   **sampling_from(args))
    if head is not None:
        # a head named on the command line beats the one a record measured, and its method
        # is read off its own name rather than kept from the record's
        config = bench.drafted_by(config, head)
    if getattr(args, "no_draft", False):
        # the profile's serving minus its head: what the head is worth is this run against
        # the drafted one, two labels apart
        config = bench.drafted_by(config, "")
    if getattr(args, "serve_kv", ""):
        config = config.over(cache_type=str(args.serve_kv))
    if getattr(args, "serve_kv_unified", None) is not None:
        config = config.over(kv_unified=bool(args.serve_kv_unified))
    if getattr(args, "reasoning_budget", None) is not None:
        config = config.over(reasoning_budget=int(args.reasoning_budget))
    length = getattr(args, "n_max", None)
    if isinstance(length, int) and length:
        # `drafts` takes a list of them, one served configuration each, put on its own arm
        # there; a sweep takes one number for the whole run
        config = config.over(draft_n_max=length)
    return config.over(**serving_fields(args))


def serving_fields(args: Any) -> dict[str, Any]:
    """The ServerSpec fields a sweep's --serve-* flags name, and nothing when none do."""
    out: dict[str, Any] = {}
    raw = list(getattr(args, "serve_arg", []) or [])
    if raw:
        out["extra_args"] = tuple(raw)
    if getattr(args, "serve_mlock", False):
        out["mlock"] = True
    if getattr(args, "serve_no_flash_attn", False):
        out["flash_attn"] = False
    mmproj = str(getattr(args, "serve_mmproj", "") or "")
    if mmproj:
        out["mmproj"] = mmproj
    return out


# The flags `sweep --fleet` takes off the line before handing it to a peer: what is about
# this machine's session, not about the measuring -- a peer keeps its own runs in its own
# store, under its own detached process, so `--kept`, like `--detach` and `--no-queue`,
# is never on a job's argv (`fleet.measuring.Job` refuses it).
_NOT_FOR_A_PEER = ("--fleet", "--detach", "--no-queue")
_NOT_FOR_A_PEER_VALUED = ("--peers", "--serve", "--serve-draft", "--kept")


def _stripped(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """``argv`` minus what a peer never sees -- ``--fleet``, ``--detach``, ``--no-queue``,
    every ``--peers``, ``--serve`` and ``--serve-draft`` -- and the ``--serve-draft``
    values it carried, in the order they were given."""
    rest: list[str] = []
    skip = False
    for word in argv:
        if skip:
            skip = False
            continue
        if word in _NOT_FOR_A_PEER:
            continue
        flag, sep, _value = word.partition("=")
        if flag in _NOT_FOR_A_PEER_VALUED:
            if not sep:
                skip = True
            continue
        rest.append(word)
    return rest, list(_values_of(argv, "--serve-draft"))


def fleet_jobs(argv: Sequence[str], models: Sequence[str], *, commit: str) -> list[dict[str, Any]]:
    """One job per model: this same command line with that one ``--serve`` (and its
    positional ``--serve-draft``, when one was given), on ``commit``.

    Every other flag rides along unchanged -- the questions, the store, the sample, every
    ``--also`` -- so a peer measures exactly what this machine would have. ``commit`` is
    the short sha, and ``dirty`` says whether the tree had changes: a peer on another
    commit is measuring other code, and both ends refuse it.
    """
    rest, heads = _stripped(argv)
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


def _discovered(peers: Sequence[str]) -> dict[str, Any]:
    """Every peer on this machine's cluster(s), by name, narrowed to ``peers`` when named.

    `Refused` when a name in ``peers`` answered to no daemon, or when discovery found
    nobody at all.
    """
    join = importlib.import_module("ml_stack.fleet.join")
    pausing = importlib.import_module("ml_stack.fleet.pausing")

    clients = pausing.peer_clients(join.peers())
    if peers:
        wanted = {str(p) for p in peers}
        named = {name: client for name, client in clients.items() if name in wanted}
        absent = sorted(wanted - set(named))
        if absent:
            raise Refused(f"error: --peers named {', '.join(absent)}, which no daemon on "
                          f"this cluster answered to"
                          + (f" (found: {', '.join(sorted(clients))})" if clients else ""))
        clients = named
    if not clients:
        raise Refused("error: no peer answered discovery; is another machine running "
                      "'ml-stack-fleet join'?")
    return clients


def fleet_planned(argv: Sequence[str], models: Sequence[str], *,
                  peers: Sequence[str] = ()) -> Fleeted:
    """The fleet's plan for ``models``, and the jobs it dispatches.

    Peers are this machine's own discovery (`ml_stack.fleet.join.peers`, turned into
    `remote.Peer` clients the way `ml_stack.fleet.pausing.peer_clients` does), narrowed to
    ``peers`` by name when given. `ml_stack.fleet.sweeps.plan` then places each model,
    largest first, and `ml_stack.fleet.measuring.jobs_from` turns the placement into one
    `Job` per peer -- reached by name, both of them, so this machine's sweep needs neither
    module until a fleet run actually asks for one. A peer already on another commit is
    refused before anything is dispatched: the daemon refuses too, but finding out from
    four peers' logs is later than from one line.
    """
    if not models:
        raise Refused("error: --fleet spreads --serve models over the fleet; pass --serve "
                      "MODEL for each")
    mine = _commit()
    if not mine:
        raise Refused("error: --fleet needs to know this checkout's commit, and git would "
                      "not say")
    fleet = importlib.import_module("ml_stack.fleet.sweeps")
    missing = [name for name in ("plan", "dispatch", "wait", "gather")
               if not hasattr(fleet, name)]
    if missing:
        raise Refused(f"error: ml_stack.fleet.sweeps has no {', '.join(missing)}; the fleet "
                      f"side of the bench is not in this build")
    measuring = importlib.import_module("ml_stack.fleet.measuring")

    clients = _discovered(peers)
    ordered = sorted(clients.items())
    by_peer = {peer: name for name, peer in ordered}
    planned = fleet.plan(models, [client for _, client in ordered])
    lines = [f"plan: {len(models)} model(s) on commit {mine} over "
             + ", ".join(name for name, _ in ordered)]
    mismatched: list[str] = []
    for peer, assigned in planned.items():
        if not assigned:
            continue
        name = by_peer.get(peer, getattr(peer, "name", str(peer)))
        theirs = str(fleet._health_of(peer).get("bench_commit") or "")
        for model in assigned:
            lines.append(f"  {model} -> {name}" + (f" ({theirs})" if theirs else ""))
        if theirs and not measuring.same_commit(mine, theirs):
            mismatched.append(f"{name} is on commit {theirs}, this checkout is on {mine}")
    for model, why in planned.unplaced:
        lines.append(f"  {model} -> unplaced: {why}")
    if mismatched:
        raise Refused("error: " + "; ".join(mismatched) + "; a peer measuring other code "
                      "is refused, and its daemon would refuse too", *lines)
    rest, heads = _stripped(argv)
    drafts = {model: heads[n] for n, model in enumerate(models) if n < len(heads)}
    jobs = measuring.jobs_from(planned, rest, commit=mine, drafts=drafts)
    return Fleeted(jobs=jobs, lines=lines)


def fleet_measure(jobs: Mapping[Any, Any], *, into: str | Path) -> None:
    """Dispatch ``jobs`` (one per peer) over the fleet, wait for them, and gather their
    runs into ``into``."""
    fleet = importlib.import_module("ml_stack.fleet.sweeps")
    handles = fleet.dispatch(dict(jobs))
    fleet.wait(handles)
    fleet.gather(handles, into=into)
