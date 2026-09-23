"""Putting a model up for a run: served, asked every way on one load, taken down again.

`served` preflights the load, smokes every way first on the same server, asks the
questions, keeps each run and reads it back; `drafts` does that once per draft head and
says which head to serve. `prefetch` brings every `hf:` reference down before any of it.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# `ml_stack.serve.serve`, `hub.fetch`, `hub.room`, `binary.find_binary` and
# `checks.Preflight` are patched by the tests and `selfcheck`, so each is read off its module.
import ml_stack.serve
from ml_stack import bench, hub
from ml_stack.bench.backends import DRAFT_OBEYED, draft_depth_support
from ml_stack.bench.keep import read_back, save
from ml_stack.bench.measure import found as finder_of
from ml_stack.bench.score import Row, _which
from ml_stack.bench.show import drafted
from ml_stack.log import say, warn
from ml_stack.serve import binary as binaries
from ml_stack.serve import mlx_tree
from ml_stack.serve import preflight as checks
from ml_stack.serve.backend import LlamaServerBackend, ServerSpec
from ml_stack.serve.binary import BinaryNotFound
from ml_stack.serve.manager import ServerManager
from ml_stack.serve.ops import alongside
from ml_stack.serve.profile import ASK
from ml_stack.serve.serving import DEFAULT_CACHE
from ml_stack.serve.weights import weight_of


@dataclass(frozen=True, slots=True)
class Loading:
    """What every load of a measuring command shares: the build, the timeout, where runs
    are kept, the smoke asked first, and the store `look_up` reads."""

    binary: str = ""
    serve_timeout: float = 900.0
    kept: str | Path = ""
    host: str = ""
    smoke: Sequence[Mapping[str, Any]] = ()
    store: str | Path | None = None
    embed_url: str = ""
    embed_model: str = ""
    trace: bool | None = None


@dataclass(frozen=True, slots=True)
class Ways:
    """The ways one load is asked: the label, each asking laid over the config with
    `Config.over`, the default shortlist, and ``already(label)`` for a way kept before."""

    label: str = ""
    askings: Sequence[Mapping[str, Any]] = ()
    shortlist: int = 0
    already: Callable[[str], Mapping[str, Any] | None] | None = None
    needs_draft_depth: bool = False


@dataclass(frozen=True, slots=True)
class Heads:
    """The draft heads `drafts` serves in turn ("" for none, `EMBEDDED` for the one in the
    weights), the depths each drafts to, and whether the depths share a load (None: find out)."""

    heads: Sequence[str] = ("",)
    n_max: Sequence[int | None] = (None,)
    per_request: bool | None = None


BARE = Loading()
ONE_WAY = Ways()


def references_in(args: Any) -> list[str]:
    """Every ``hf:`` reference a measuring command names, models before heads, each once."""
    out: list[str] = []
    named = [getattr(args, "model", ""), *(getattr(args, "serve", None) or []),
             *(getattr(args, "serve_draft", None) or []), *(getattr(args, "draft", None) or [])]
    for one in named:
        if isinstance(one, str) and one.startswith("hf:") and one not in out:
            out.append(one)
    return out


def prefetch(references: Sequence[str], log: Callable[[str], None] = say) -> list[tuple[str, int]]:
    """Download every reference into the Hub cache: ``(reference, bytes)`` for each fetched."""
    out: list[tuple[str, int]] = []
    for ref in references:
        try:
            where = hub.fetch(ref)
        except (OSError, ValueError) as exc:
            warn(f"could not fetch {ref}: {exc}")
            continue
        size = weight_of(where)
        log(f"fetched {ref}: {size / 2**30:.2f}G at {where}")
        out.append((ref, size))
    return out


class SmokeFailed(RuntimeError):
    """The smoke kept nothing, kept no rows, or every question in it failed."""


def smoked(kept: Sequence[Mapping[str, Any]], what: str) -> None:
    """Raise `SmokeFailed` for no run kept, a run with no rows, or every row failed."""
    if not kept:
        raise SmokeFailed(f"{what}: no run was kept")
    rows = [r for one in kept for r in (one.get("rows") or ())]
    if not rows:
        raise SmokeFailed(f"{what}: {len(kept)} run(s) kept with no rows")
    if all(r.get("error") or r.get("timed_out") for r in rows):
        raise SmokeFailed(f"{what}: every question failed -- {rows[0].get('error') or 'timed out'}")


# A head inside the weights: served with --spec-type draft-mtp and no -md.
EMBEDDED = "embedded"


def drafted_by(config: Any, head: str) -> Any:
    """``config`` serving ``head``: a path or ``hf:`` reference, "" for none, or `EMBEDDED`."""
    if head == EMBEDDED:
        return config.over(draft="", spec_type="draft-mtp")
    return config.over(draft=str(head or ""), spec_type="")


class NotLoaded(RuntimeError):
    """The preflight refused the model, so nothing was loaded; the message is its report."""


def refused(label: str, why: Exception) -> str:
    """What a measuring command says of ``label`` when ``why`` kept it from loading."""
    return (f"    preflight refused {label}; not loaded:\n"
            + "\n".join(f"      {line}" for line in str(why).splitlines()))


def _with_projector(config: Any) -> Any:
    """``config`` with an ``mmproj`` of "auto" resolved to the projector beside the model."""
    if str(config.serving.mmproj or "").lower() != "auto":
        return config
    found = alongside(str(config.model), "auto", "mmproj-", best=True)
    if not found:
        say("no vision projector is shipped beside that model; serving without one")
    return config.over(mmproj=str(found or ""))


def _manager(config: Any, binary: str) -> tuple[Any, str]:
    """The manager for ``binary`` or the profile's named build (None for the default), and
    the build it runs. Raises `NotLoaded` when the named build is not on this machine."""
    if binary:
        return ServerManager(LlamaServerBackend(binary=binary)), binary
    try:
        if config.serving.build:
            manager = config.serving.manager()
            return manager, str(manager.backend.binary)
        return None, str(binaries.find_binary() or "llama-server")
    except BinaryNotFound as exc:
        raise NotLoaded(str(exc)) from exc


def _preflighted(spec: Any, build: str) -> Any:
    """The preflight report for ``spec`` on ``build``; raises `NotLoaded` when it refuses."""
    report = (mlx_tree.report_for(spec, limit_bytes=hub.room()) if mlx_tree.is_mlx(spec.model)
              else checks.Preflight(spec, binary=build, limit_bytes=hub.room()))
    if not report.ok:
        raise NotLoaded(report.said())
    return report


def _said_up(server: Any, loaded: float, report: Any) -> None:
    load_s = getattr(server, "load_s", None)
    warmup_s = getattr(server, "warmup_s", None)
    timed = ""
    if load_s is not None:
        warm = f", warm-up {float(warmup_s):.1f}s" if warmup_s is not None else ""
        timed = f" (load {float(load_s):.1f}s{warm})"
    say(f"    up in {loaded:.0f}s{timed}")
    say("\n".join(f"      {line}" for line in report.said().splitlines()))


def _held(config: Any, server: Any, report: Any, build: str) -> dict[str, Any]:
    """The record every run on this load carries: preflight, timings, build, head, cache."""
    serving = config.serving
    record: dict[str, Any] = {
        "preflight": {"kv_estimate_bytes": int(report.kv_estimate_bytes),
                      "weights_bytes": int(report.weights_bytes), "ok": bool(report.ok)},
        "load_s": getattr(server, "load_s", None), "warmup_s": getattr(server, "warmup_s", None),
        "binary": build, "build": str(serving.build or "")}
    if serving.draft or serving.spec_type:
        record["draft_model"] = str(serving.draft).rsplit("/", 1)[-1] if serving.draft else EMBEDDED
        if serving.draft_n_max is not None:
            record["spec_draft_max"] = int(serving.draft_n_max)
    if serving.cache_type:
        record["cache_type"] = serving.cache_type
    if serving.reasoning_budget is not None:
        record["reasoning_budget"] = int(serving.reasoning_budget)
    return record


@contextlib.contextmanager
def up(config: Any, *, binary: str = "", serve_timeout: float = 900.0) -> Any:
    """Preflight, serve and take down ``config``'s model; yields ``(server, held)``, the
    lease's `ServerInfo` and the record its runs carry (with ``baseline`` and ``loaded``)."""
    config = _with_projector(config)
    # the system prompt and tool schemas prefix every question, so KV shifting reuses them
    extra: dict[str, Any] = {**config.lease(), "cache_reuse": 256, "warmup": False}
    manager, build = _manager(config, binary)
    report = _preflighted(ServerSpec(model=config.model, **extra), build)
    if manager is not None:
        extra["manager"] = manager
    began = time.time()
    before_load = bench.machine_memory()
    with ml_stack.serve.serve(config.model, timeout=serve_timeout, **extra) as server:
        loaded = time.time() - began
        _said_up(server, loaded, report)
        yield server, {**_held(config, server, report, build),
                       "baseline": before_load, "loaded": loaded}


class DraftDepthIgnored(RuntimeError):
    """A server that drops ``speculative.n_max`` instead of drafting to it."""


@dataclass(frozen=True, slots=True)
class _Load:
    """One served model and everything each way asked of it needs."""

    config: Any
    graph: Mapping[str, Any]
    loading: Loading
    ways: Ways
    server: Any
    held: Mapping[str, Any]
    baseline: Any
    loaded: float
    named: Callable[[Mapping[str, Any]], str]


def _suffix(config: Any) -> str:
    """``-kv-TYPE`` for a cache other than the default, ``-rbN`` for a thinking budget."""
    kv = config.serving.cache_type
    budget = config.serving.reasoning_budget
    return ((f"-kv-{kv}" if kv and kv != DEFAULT_CACHE else "")
            + (f"-rb{budget}" if budget is not None else ""))


def _labeller(name: str, suffix: str) -> Callable[[Mapping[str, Any]], str]:
    """A way's label: the name, its tag (joined directly when it begins with @), the suffix."""
    def labelled(way: Mapping[str, Any]) -> str:
        tag = str(way.get("label", "") or "")
        if not tag:
            return name + suffix
        return f"{name}{'' if tag.startswith('@') else '-'}{tag}" + suffix
    return labelled


def _not_yet_kept(ways: Ways, labelled: Callable[[Mapping[str, Any]], str]
                  ) -> list[Mapping[str, Any]]:
    """The askings with no run kept yet under their label."""
    every = list(ways.askings) or [{}]
    if ways.already is None:
        return every
    todo = []
    for way in every:
        kept_as = ways.already(labelled(way))
        if kept_as:
            say(f"skipping {labelled(way)}: kept at {kept_as.get('at', '?')}")
        else:
            todo.append(way)
    return todo


def _ask_way(load: _Load, way: Mapping[str, Any],
             questions: Sequence[Mapping[str, Any]]) -> tuple[list[Row], str]:
    """``questions`` asked one way on ``load``: the rows, and the key they were kept under."""
    asked = dict(way)
    here = load.named(asked)
    asked.pop("label", None)
    first = int(asked.pop("shortlist", load.ways.shortlist) or 0)
    wants_card = bool(asked.pop("_card", False))
    this = load.config.over(**asked)
    client = this.client(load.server.base_url)
    if wants_card:
        this = this.over(**client.card)
        client = this.client(load.server.base_url)
    loading = load.loading
    ask = bench.asking(load.graph, how=this.asking, shortlist=first, store=loading.store,
                       embed_url=loading.embed_url, embed_model=loading.embed_model)
    got = bench.measure(ask, questions, label=here, client=client, trace=loading.trace,
                        log=print, baseline=load.baseline, graph=load.graph,
                        per_question=float(load.config.talking.timeout))
    for row in got:
        row.steps = f"{row.steps}; server up in {load.loaded:.0f}s".strip("; ")
    record = {**bench.footprint(load.server.base_url), "graph": _which(load.graph),
              "finder": getattr(ask, "finder", ""), **load.held}
    if loading.host:
        record["host"] = loading.host
    if client.request.spec_draft_max is not None:
        record["spec_draft_max_asked"] = int(client.request.spec_draft_max)
    key = ""
    if loading.kept:
        key = save(loading.kept, got, server={**record, "sampling": dict(client.sampling)},
                   asking=getattr(ask, "asking", None), workload=ASK)
    return got, key


def _ask_every(load: _Load, every: Sequence[Mapping[str, Any]],
               questions: Sequence[Mapping[str, Any]], *, smoking: bool
               ) -> tuple[list[Row], list[str]]:
    """Every way asked ``questions``: the rows, and the keys of the runs kept."""
    rows: list[Row] = []
    keys: list[str] = []
    for way in every:
        if len(every) > 1 or smoking:
            say(f"\n  --- {load.named(way)}" + (" (smoke)" if smoking else ""))
        got, key = _ask_way(load, way, questions)
        rows += got
        if key:
            keys.append(key)
    return rows, keys


def _checked_depth(config: Any, server: Any) -> None:
    """Raise `DraftDepthIgnored` unless the server drafts to a per-request depth."""
    reading = draft_depth_support(config.client(server.base_url))
    say(f"      per-request draft depth: {reading}")
    if reading != DRAFT_OBEYED:
        raise DraftDepthIgnored(reading)


def served(config: Any, questions: Sequence[Mapping[str, Any]], graph: Mapping[str, Any],
           loading: Loading = BARE, ways: Ways = ONE_WAY) -> list[Row]:
    """Put ``config``'s model up, smoke then ask every way, take it down: the rows.

    Raises `NotLoaded` when a preflight refuses, `SmokeFailed` when the smoke proves nothing."""
    name = ways.label or str(config.model).rsplit("/", 1)[-1].removesuffix(".gguf")
    suffix = _suffix(config)
    named = _labeller(name, suffix)
    every = _not_yet_kept(ways, named)
    if not every:
        return []
    try:
        with up(config, binary=loading.binary,
                serve_timeout=loading.serve_timeout) as (server, held):
            finder, why = finder_of(loading.store, loading.embed_url, loading.embed_model)
            say(f"      look_up by {finder}" + (f" ({why})" if why else ""))
            loaded = float(held.pop("loaded", 0.0))
            baseline = held.pop("baseline", None)
            if ways.needs_draft_depth:
                _checked_depth(config, server)
            load = _Load(config=config, graph=graph, loading=loading, ways=ways, server=server,
                         held=held, baseline=baseline, loaded=loaded, named=named)
            if loading.smoke:
                say(f"\n  smoke: {len(loading.smoke)} question(s) through every way first")
                proved, keys = _ask_every(load, every, loading.smoke, smoking=True)
                smoked(read_back(loading.kept, keys) if loading.kept
                       else [{"rows": [asdict(r) for r in proved]}], f"{name}{suffix} smoke")
                say("  smoke: ok")
            return _ask_every(load, every, questions, smoking=False)[0]
    except checks.PreflightFailed as why:
        raise NotLoaded(str(why)) from why


def _head_name(head: str) -> str:
    if not head:
        return "none"
    if head == EMBEDDED:
        return "embedded-mtp"
    return str(head).rsplit("/", 1)[-1].removesuffix(".gguf")


def drafts(config: Any, heads: Heads, questions: Sequence[Mapping[str, Any]],
           graph: Mapping[str, Any], loading: Loading = BARE) -> list[Row]:
    """Serve ``config`` with each head at each depth and measure it: the rows; with runs kept,
    ends by printing `drafted`, each (head, depth)'s speedup over the undrafted run."""
    out: list[Row] = []
    lengths = list(heads.n_max) or [None]
    before = {r.get("key") for r in bench._kept(loading.kept)} if loading.kept else set()
    shared = heads.per_request
    for head in heads.heads:
        name = _head_name(head)
        depths = list(lengths) if head else [None]
        if head and shared is not False and len(depths) > 1 and None not in depths:
            say(f"\n--- draft: {name}, {len(depths)} depths on one load")
            askings = [{"label": f"@n{d}", "spec_draft_max": d} for d in depths]
            try:
                out += bench.served(
                    drafted_by(config, head).over(draft_n_max=max(depths)), questions, graph,
                    loading, Ways(label=f"draft:{name}", askings=askings,
                                  needs_draft_depth=shared is None))
                shared = True
                continue
            except NotLoaded as why:
                say(refused(f"draft:{name}", why))
                continue
            except DraftDepthIgnored as reading:
                shared = False
                say(f"      this build {reading} the per-request depth; "
                    f"a server per depth instead")
        for length in depths:
            tagged = f"{name}@n{length}" if length is not None else name
            say(f"\n--- draft: {tagged}")
            try:
                out += bench.served(drafted_by(config, head).over(draft_n_max=length),
                                    questions, graph, loading, Ways(label=f"draft:{tagged}"))
            except NotLoaded as why:
                say(refused(f"draft:{tagged}", why))
    if loading.kept and out:
        # the smoke each load kept has fewer rows than the questions, so it is left out
        everything = bench._kept(loading.kept)
        mine = [r for r in everything if r.get("key") not in before
                and len(r.get("rows") or ()) == len(questions)]
        say("\n" + drafted(mine, among=everything))
    return out
