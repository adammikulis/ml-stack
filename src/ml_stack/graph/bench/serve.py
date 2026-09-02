"""Putting a model up for a run: served, asked every way on one load, taken down again.

`served` preflights the load, smokes every way first on the same server, asks the
questions, keeps each run and reads it back; `drafts` does that once per draft head and
says which head to serve. Before any of it, `find_model` turns a name into a path and
`prefetch` brings every `hf:` reference down outside the timed window.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.measure`,
# `bench.footprint`, `bench.served` -- so anything patchable is looked up there at call
# time, never bound here at import.
from ml_stack.graph import bench
from ml_stack.graph.bench.keep import read_back, save
from ml_stack.graph.bench.measure import PER_QUESTION, finding
from ml_stack.graph.bench.score import Row, _which
from ml_stack.graph.bench.show import drafted


def find_model(named: str) -> str:
    """A model by name, path or `hf:` reference -- whichever the caller has to hand.

    `fleet.models` has known where the files are all along, and looking one up by hand with
    `find ~/.cache/... -name '*.gguf'` was done six times in an afternoon before this
    existed. A name that matches nothing is returned unchanged, so a path still works and a
    typo still fails where it would have anyway.
    """
    if not named or named.startswith("hf:") or "/" in named:
        return named
    try:
        from ml_stack.fleet.models import Models, default_roots

        home = Path("~/.ml-stack").expanduser()
        found = Models(roots=default_roots(home), store=home).find(named)
    except Exception:  # noqa: BLE001 - a machine that cannot look is not a failed run
        return named
    return str(found.path) if found else named


def references_in(args: Any) -> list[str]:
    """Every ``hf:`` reference a measuring command would otherwise download inside the
    timed window: the models ``--serve`` names, the heads ``--serve-draft`` and ``--draft``
    name, and the model `drafts` is given. Models first, then heads, each once: the
    weights are what a preflight sizes."""
    out: list[str] = []
    named = [getattr(args, "model", ""), *(getattr(args, "serve", None) or []),
             *(getattr(args, "serve_draft", None) or []), *(getattr(args, "draft", None) or [])]
    for one in named:
        if isinstance(one, str) and one.startswith("hf:") and one not in out:
            out.append(one)
    return out


def prefetch(references: Sequence[str], log: Callable[[str], None] = print) -> list[tuple[str, int]]:
    """Download every reference into the Hub cache before the lock is taken, one line each.

    A download inside the timed window is a timing of the network: the first model of a
    sweep once showed a load three times the second's, and the difference was the fetch.
    `hub.fetch` brings down every shard of a build, so a preflight afterwards finds the
    weights complete. A reference that cannot be fetched is said and left -- the preflight
    on that model is what refuses it, with the shard named.
    """
    from ml_stack import hub
    from ml_stack.serve.manager import weight_of

    out: list[tuple[str, int]] = []
    for ref in references:
        try:
            where = hub.fetch(ref)
        except Exception as exc:  # noqa: BLE001 - the Hub is somebody else's machine
            print(f"could not fetch {ref}: {exc}", file=sys.stderr)
            continue
        size = weight_of(where)
        log(f"fetched {ref}: {size / 2**30:.2f}G at {where}")
        out.append((ref, size))
    return out


class SmokeFailed(RuntimeError):
    """The two-question pass a real run makes first did not get through, so the run did
    not start: nothing kept, nothing read back, or every question failed."""


def smoked(kept: Sequence[Mapping[str, Any]], what: str) -> None:
    """Refuse a smoke that proved nothing: no run kept, a run with no rows, or every row
    an error or a timeout. A model that fails two questions fails twenty, and the point of
    asking two first is that finding out costs a minute."""
    if not kept:
        raise SmokeFailed(f"{what}: no run was kept")
    rows = [r for one in kept for r in (one.get("rows") or ())]
    if not rows:
        raise SmokeFailed(f"{what}: {len(kept)} run(s) kept with no rows")
    if all(r.get("error") or r.get("timed_out") for r in rows):
        raise SmokeFailed(f"{what}: every question failed -- {rows[0].get('error') or 'timed out'}")


# A head that lives inside the weights: `--draft embedded` serves the model with
# --spec-type draft-mtp and no -md. Qwen3.8-27B ships its nextn layers in the main GGUF.
EMBEDDED = "embedded"


def served(model: str, questions: Sequence[Mapping[str, Any]], graph: Mapping[str, Any], *,
           label: str = "", draft: str = "", port: int = 8099, context: int = 32768,
           parallel: int = 1, binary: str = "", kept: str | Path = "", shortlist: int = 0,
           store: str | Path | None = None, embed_url: str = "", embed_model: str = "",
           terse: bool = False, ways: Sequence[Mapping[str, Any]] = (),
           serve_timeout: float = 900.0,
           already: Callable[[str], Mapping[str, Any] | None] | None = None,
           spec_draft_max: int | None = None, cache_type: str = "",
           per_question: float = PER_QUESTION, reasoning_budget: int | None = None,
           smoke: Sequence[Mapping[str, Any]] = (), host: str = "",
           spec_type: str = "",
           **making: Any) -> list[Row]:
    """Put one model up, ask it the questions, take it down again.

    ``host`` is what the kept runs say measured them, when this machine's name is not the
    answer -- a peer running a fleet's job records the name the plan gave it. Empty, the
    hostname is written.

    ``smoke`` is the questions to ask first, of every way, on the same load -- two of them
    -- kept, read back, and refused with `SmokeFailed` when every one of them failed. A
    real run does this before its questions unless told not to, and it is done here
    rather than around the call so that the load is paid once: the smoke is the first
    thing the served model is asked, and its own questions follow on the same server.

    The piece that was missing. `sweep` measures servers somebody else started, so anything
    comparing several models meant hand-rolling starts, stops and waits -- which is a shell
    loop that dies with the terminal, and which was written twice before becoming this.

    One model at a time is not a limitation, it is the point: two servers sharing a GPU
    produce timings that belong to neither.

    ``ways`` asks the *same* server several times, which is most of the saving available
    here. Whether the tools are described briefly, what sampling is used, and whether a
    shortlist is handed over first are questions about the asking and not about the
    serving -- so measuring four of them costs one load and not four. Only a change the
    server itself must be told about, a draft head or a context, needs putting it up
    again. Each way is ``{"label": ..., "terse": ..., "shortlist": ...}`` plus anything a
    client takes; a way without ``shortlist`` takes the ``shortlist`` given here.

    ``already(label)`` is the run a way is already kept as, when it is -- `sweep --resume`
    passes it -- and a way that has one is skipped before the model is loaded, so a sweep
    killed on its third model costs the third model to re-run and not the first two.

    ``spec_draft_max`` is how many tokens the draft head guesses ahead, when one is served;
    ``cache_type`` quantises the KV cache (``q8_0``), and every label gets ``-kv-q8_0`` on
    its end, because a run with a quantised cache is another configuration and the label is
    what the table shows. ``reasoning_budget`` is the same again: bound at start, on the
    label as ``-rbN``, and it stops a thinking model's thinking where `n_predict` would
    have cut its answer. ``per_question`` caps each question -- see `_ask_once`.

    **The load is preflighted first** -- shards present, architecture read by this build,
    weights plus an estimated KV cache under what this machine may use, every flag one the
    build accepts -- and the report is printed under the `up in` line, the KV estimate
    beside what `kv+run` then measures. A refused preflight is printed and the model is
    skipped, nothing loaded: a sweep of five must not end on the one that does not fit.
    """
    from ml_stack import hub
    from ml_stack.client import Client
    from ml_stack.serve import preflight as checks
    from ml_stack.serve import serve
    from ml_stack.serve.backend import ServerFailed, ServerSpec
    from ml_stack.serve.binary import find_binary

    name = label or str(model).rsplit("/", 1)[-1].removesuffix(".gguf")
    suffix = ((f"-kv-{cache_type}" if cache_type else "")
              + (f"-rb{reasoning_budget}" if reasoning_budget is not None else ""))

    def labelled(way: Mapping[str, Any]) -> str:
        tag = str(way.get("label", "") or "")
        return (f"{name}-{tag}" if tag else name) + suffix

    every = list(ways) or [{}]
    if already is not None:
        todo = []
        for way in every:
            kept_as = already(labelled(way))
            if kept_as:
                print(f"skipping {labelled(way)}: kept at {kept_as.get('at', '?')}")
            else:
                todo.append(way)
        if not todo:
            return []
        every = todo
    extra: dict[str, Any] = {"parallel": parallel}
    if draft == EMBEDDED:
        # the head is inside the weights (Qwen3.8-27B ships its nextn layers in the main
        # GGUF): no -md, only the speculative type, and the draft length if asked
        draft = ""
        extra["spec_type"] = "draft-mtp"
    if draft:
        from ml_stack.hub import spec_for

        extra["draft"] = draft
        kind = spec_for(draft)
        if kind:
            extra["spec_type"] = kind
    if spec_type:
        extra["spec_type"] = spec_type
    if (draft or extra.get("spec_type")) and spec_draft_max is not None:
        extra["spec_draft_max"] = int(spec_draft_max)
    if cache_type:
        extra["cache_type_k"] = extra["cache_type_v"] = cache_type
    if reasoning_budget is not None:
        extra["reasoning_budget"] = int(reasoning_budget)
    # Every question sends the same system prompt and the same tool schemas ahead of itself.
    # Reusing that prefix by KV shifting, rather than reprocessing it twenty times a run, is
    # free accuracy-wise: the tokens are identical, so the cache is valid.
    extra.setdefault("cache_reuse", 256)
    extra.setdefault("warmup", False)

    # Asked of the spec `serve` is about to build, with the binary it will start -- or, with
    # none named, the one `find_binary` would; a name no build answers to gives the flag and
    # architecture checks no opinion rather than a wrong one. `room()` is what this machine
    # may wire for a model, not what happens to be free.
    spec = ServerSpec(model=model, port=port, context=context, **extra)
    build = str(binary or find_binary() or "llama-server")
    report = checks.Preflight(spec, binary=build, limit_bytes=hub.room())
    if not report.ok:
        print(f"    preflight refused {name}{suffix}; not loaded:\n"
              + "\n".join(f"      {line}" for line in report.said().splitlines()))
        return []
    checked = {"kv_estimate_bytes": int(report.kv_estimate_bytes),
               "weights_bytes": int(report.weights_bytes), "ok": bool(report.ok)}

    if binary:
        from ml_stack.serve.backend import LlamaServerBackend
        from ml_stack.serve.manager import ServerManager

        extra["manager"] = ServerManager(LlamaServerBackend(binary=binary))

    rows: list[Row] = []
    began = time.time()
    try:
        with serve(model, port=port, context=context, timeout=serve_timeout, **extra) as server:
            # `load_s` is the lease's own clock, process start to health; the stopwatch
            # here also holds an adopted server's nothing and a warm-up's something.
            loaded = time.time() - began
            load_s = getattr(server, "load_s", None)
            warmup_s = getattr(server, "warmup_s", None)
            print(f"    up in {loaded:.0f}s"
                  + (f" (load {float(load_s):.1f}s" + (f", warm-up {float(warmup_s):.1f}s"
                                                       if warmup_s is not None else "") + ")"
                     if load_s is not None else "")
                  + f", look_up by {finding(store, embed_url)}")
            print("\n".join(f"      {line}" for line in report.said().splitlines()))

            def ask_every(asking_these: Sequence[Mapping[str, Any]],
                          *, smoking: bool) -> tuple[list[Row], list[str]]:
                """Every way, asked ``asking_these``, each kept: the rows and the keys."""
                got_all: list[Row] = []
                keys: list[str] = []
                for way in every:
                    asked = dict(way)
                    here = labelled(asked)
                    asked.pop("label", None)
                    how = bool(asked.pop("terse", terse))
                    first = int(asked.pop("shortlist", shortlist) or 0)
                    if len(every) > 1 or smoking:
                        print(f"\n  --- {here}" + (" (smoke)" if smoking else ""))
                    wants_card = bool(asked.pop("_card", False))
                    # the cap is the client's timeout too, so a call past it is cut off
                    # there and the connection closed, rather than waited on
                    # what is about the asking, not the client -- popped before the client
                    # is built: `tight` reached Client.__init__ and took an 87G load down
                    # with it (measured 2026-09-02); `rich` would have too, on the first
                    # real run
                    richly = bool(asked.pop("rich", False))
                    tightly = bool(asked.pop("tight", False))
                    client = Client(server.base_url,
                                    **{"timeout": per_question, **making, **asked})
                    if wants_card:
                        # what the model itself recommends, read from the GGUF it is serving
                        client = Client(server.base_url,
                                        **{"timeout": per_question, **making, **client.card})
                    ask = bench.asking(graph, shortlist=first, store=store,
                                       embed_url=embed_url,
                                 embed_model=embed_model, terse=how,
                                 rich=richly,
                                 tight=tightly)
                    got = bench.measure(ask, asking_these, label=here, client=client,
                                        log=print,
                                  graph=graph, per_question=per_question)
                    for row in got:
                        row.steps = f"{row.steps}; server up in {loaded:.0f}s".strip("; ")
                    # `binary` is the llama-server this ran on, so a run on a fork is told
                    # from one on mainline when the ranking takes its cost
                    held = {**bench.footprint(server.base_url), "graph": _which(graph),
                            "finder": getattr(ask, "finder", ""), "preflight": dict(checked),
                            "load_s": load_s, "warmup_s": warmup_s, "binary": build}
                    if draft:
                        held["draft_model"] = str(draft).rsplit("/", 1)[-1]
                        if spec_draft_max is not None:
                            held["spec_draft_max"] = int(spec_draft_max)
                    if cache_type:
                        held["cache_type"] = cache_type
                    if reasoning_budget is not None:
                        held["reasoning_budget"] = int(reasoning_budget)
                    if host:
                        held["host"] = host
                    if kept:
                        keys.append(save(kept, got,
                                         held={**held, "sampling": dict(client.sampling)}))
                    got_all += got
                return got_all, keys

            if smoke:
                # first, on this load: every way through the whole path on two questions,
                # kept and read back, before the questions that cost the GPU
                print(f"\n  smoke: {len(smoke)} question(s) through every way first")
                proved, keys = ask_every(smoke, smoking=True)
                smoked(read_back(kept, keys) if kept
                       else [{"rows": [asdict(r) for r in proved]}], f"{name}{suffix} smoke")
                print("  smoke: ok")
            rows += ask_every(questions, smoking=False)[0]
    except checks.PreflightFailed as why:
        # The backend's own preflight, which can refuse what this one passed -- a draft
        # head resolved to a file this could not size, say. Same answer: say it, move on.
        print(f"    preflight refused {name}{suffix}; not loaded:\n"
              + "\n".join(f"      {line}" for line in str(why).splitlines()))
    return rows


def drafts(model: str, heads: Sequence[str], questions: Sequence[Mapping[str, Any]],
           graph: Mapping[str, Any], *, port: int = 8099, context: int = 32768,
           parallel: int = 1, binary: str = "", kept: str | Path = "",
           store: str | Path | None = None, embed_url: str = "", embed_model: str = "",
           serve_timeout: float = 900.0, n_max: Sequence[int | None] = (None,),
           cache_type: str = "", per_question: float = PER_QUESTION,
           smoke: Sequence[Mapping[str, Any]] = (), host: str = "",
           **making: Any) -> list[Row]:
    """Serve one model with each draft head in turn and measure what each is worth.

    ``smoke`` and ``host`` go to `served` as they are: the two questions each load is asked
    first, and the name the kept runs carry.

    A draft head only *proposes*; the large model verifies every token, so a quantised head
    cannot make an answer wrong -- it can only be right less often, and each wrong guess
    costs a verification pass. Whether the extra precision pays for its memory is therefore
    an empirical question and not an arguable one: it depends on this model, this workload,
    and how often the head happens to be right about it.

    Pass "" as a head to measure the model with no draft at all, which is the baseline
    every other row has to beat.

    ``n_max`` is how many tokens a head guesses ahead per pass, one served configuration
    per value -- `--spec-draft-n-max` is bound when the server starts, like the head -- and
    the run is labelled ``draft:<head>@n8`` so the table shows acceptance and wall clock
    per (head, n-max). ``None`` is the build's own default and adds nothing to the label.
    The baseline with no head is measured once: there is nothing to guess ahead with.

    When the runs are ``kept``, it ends by printing `drafted`: one row per (head, n-max)
    with its speedup over the baseline as a number, and which configuration to serve.
    """
    # The base model is loaded again for every head, because `-md` is bound when the server
    # starts and llama.cpp has no runtime swap: N configurations is N servers. It costs much
    # less than the first load -- the weights are mmapped and the pages are still cached --
    # but it is not free, so `served` times it and prints it rather than waving it away.
    out: list[Row] = []
    lengths = list(n_max) or [None]
    before = {r.get("key") for r in bench._kept(kept)} if kept else set()
    for head in heads:
        name = "none" if not head else "embedded-mtp" if head == EMBEDDED else \
            str(head).rsplit("/", 1)[-1].removesuffix(".gguf")
        for length in (lengths if head else [None]):
            tagged = f"{name}@n{length}" if length is not None else name
            print(f"\n--- draft: {tagged}")
            out += bench.served(model, questions, graph, label=f"draft:{tagged}",
                                draft=head,
                          port=port, context=context, parallel=parallel, binary=binary,
                          kept=kept, store=store, embed_url=embed_url,
                          embed_model=embed_model, serve_timeout=serve_timeout,
                          spec_draft_max=length, cache_type=cache_type,
                          per_question=per_question, smoke=smoke, host=host, **making)
    if kept and out:
        # the speedup as a number, against the baseline this call measured -- or, given
        # only heads, the newest undrafted run of this model and size already kept. The
        # smoke each load made first is kept too and is not one of these rows: two
        # questions say nothing about a head
        everything = bench._kept(kept)
        mine = [r for r in everything if r.get("key") not in before
                and len(r.get("rows") or ()) == len(questions)]
        print("\n" + drafted(mine, among=everything))
    return out
