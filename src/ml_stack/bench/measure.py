"""Asking a graph its questions, and what a set of them cost.

`asking` is the ordinary way to ask this graph one question and `ask_from` imports someone
else's; `found` and `finding` say which `look_up` that measures. `_ask_once` puts one
question through the client with its bill kept, `measure` asks a set of them one at a
time, and `concurrent` asks N conversations of T turns at once.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.slot_count` -- so
# anything patchable is looked up there at call time, never bound here at import.
from ml_stack import bench
from ml_stack.bench.counting import PER_QUESTION, Counting, wants_trace
from ml_stack.bench.holding import _Peak, watching
from ml_stack.bench.score import Row, prefix_kept, unread_named
from ml_stack.client.embed import MARGIN, stands_out


def found(store: str | Path | None, embed_url: str = "",
          embed_model: str = "") -> tuple[str, str]:
    """``(finder, why)``: which look_up a run measures, and why it is not the one asked for.

    ``chars``: no store, so `look_up` matches characters and nothing else. ``words``: a
    store's word index votes as well, so "compilers" finds "compiler". ``meaning``: and its
    vectors, through the embedder at ``embed_url`` -- only when the store holds vectors for
    ``embed_model`` (any model's, when none is named), read from the store itself; an
    embedder named beside a store with none is ``words``, and ``why`` says so. Recorded on
    every run and printed in the table, for the same reason `ctx` is: a comparison across
    finders is two measurements.
    """
    if store is None or not str(store):
        return "chars", ""
    if not embed_url:
        return "words", ""
    from ml_stack.graph.store import GraphStore
    from ml_stack.graph.vectors import embedded

    try:
        with GraphStore(store, read_only=True) as held:
            vectors = embedded(held, model=embed_model)
    except Exception:  # noqa: BLE001 - a store that will not open holds no vectors
        vectors = 0
    if vectors:
        return "meaning", ""
    return "words", f"no vectors{f' for {embed_model}' if embed_model else ''} in the store"


def finding(store: str | Path | None, embed_url: str = "", embed_model: str = "") -> str:
    """Which look_up a run measures -- ``chars``, ``words`` or ``meaning`` -- see `found`."""
    return found(store, embed_url, embed_model)[0]

def _ask_once(ask: Callable[..., Any], one: Mapping[str, Any], *, label: str, client: Any,
              graph: Mapping[str, Any] | None = None, turns: Sequence[Mapping[str, str]] = (),
              conversation: int = 0, turn: int = 0,
              per_question: float = PER_QUESTION, trace: bool = False) -> tuple[Row, str]:
    """One question through ``ask(question, client)``, and what it cost; with the answer's
    text, which a conversation carries into its next turn.

    ``per_question`` is the most it may take. Past that the row is kept as timed out: no
    answer, the cap as its wall clock, scored wrong -- and the next question is asked. A
    question that hangs is a result, not a reason for the run to.

    ``trace`` keeps the transcript on the row as well as the totals -- see `Counting` and
    `wants_trace`. A question that failed or timed out keeps the trace it got to: that is
    the transcript worth having, since it says where it went wrong.
    """
    began = time.time()
    counting = Counting(client, deadline=began + per_question if per_question else None,
                        trace=trace)
    row = Row(label=label, question=str(one.get("q") or ""),
              expected=[str(i) for i in (one.get("expect") or ())],
              conversation=conversation, turn=turn)
    said = ""
    try:
        out = ask(row.question, counting, **({"turns": list(turns)} if turns else {}))
        # an Answer, or the payload a project sends its own page; both say the same things
        read = out.get if isinstance(out, Mapping) else lambda k, d=None: getattr(out, k, d)
        row.steps = read("why", "") or ""
        said = str(read("content", "") or "")
        row.answer_chars = len(said)
        row.shown = list(read("show", None) or read("ids", None) or [])
        if graph is not None:
            touched = [str(i) for key in ("found", "read", "path", "show", "ids")
                       for i in (read(key, None) or ())]
            row.unread = unread_named(said, graph, touched)
            row.unread_named = len(row.unread)
    except Exception as exc:  # noqa: BLE001 - a failure is a result, not the end of the run
        row.error = f"{type(exc).__name__}: {exc}"[:200]
    row.seconds = round(time.time() - began, 2)
    if counting.timed_out:
        # Whatever `ask` made of the timeout -- raised it, or caught it and answered with
        # what it had -- the question was not answered inside the cap, and is scored so.
        row.timed_out = True
        row.error = f"timed out after {per_question:.0f}s"
        row.seconds = float(per_question)
        row.shown, row.unread, row.unread_named, row.answer_chars, said = [], [], 0, 0, ""
    # None for every figure the program did not report: not measured is not 0
    row.first_token = counting.first_token
    row.queued = (round(max(0.0, row.seconds - counting.generating_ms / 1000), 2)
                  if counting.generating_ms is not None else None)
    row.calls = counting.calls
    row.prompt_tokens = counting.prompt_tokens
    row.cached_tokens = counting.cached_tokens
    row.processed_tokens = counting.processed_tokens
    row.completion_tokens = counting.completion_tokens
    row.draft_tokens = counting.draft_tokens
    row.draft_taken = counting.draft_taken
    row.draft_ms = counting.draft_ms
    row.verify_ms = counting.verify_ms
    row.verify_n = counting.verify_n
    row.cache_calls = [[c, p] for c, p in counting.per_call]
    row.prompts = counting.prompts
    row.trace = counting.trace
    if any(c is not None for c, _ in counting.per_call):
        row.prefix_kept, row.prefix_turns = prefix_kept(counting.per_call)
        row.prefix_hits = row.prefix_kept / row.prefix_turns if row.prefix_turns else None
    else:
        row.prefix_kept = row.prefix_turns = row.prefix_hits = None
    return row, said


def measure(ask: Callable[[str, Any], Any], questions: Sequence[dict[str, Any]], *,
            label: str, client: Any, log: Callable[[str], None] | None = None,
            graph: Mapping[str, Any] | None = None,
            per_question: float = PER_QUESTION, trace: bool | None = None,
            baseline: Mapping[str, int] | None = None) -> list[Row]:
    """Ask each question once through ``ask(question, client)`` and record what it cost.

    Given the ``graph``, each row also counts the entries the answer named that no tool
    call produced -- see `unread_named`. ``per_question`` caps each question; see
    `_ask_once`. ``trace`` keeps the whole transcript on each row as well, and unset is
    `wants_trace`'s answer for a run of this many questions.

    While the questions are asked a `Watching` samples what the server holds every
    `SAMPLE_EVERY` seconds and keeps the maxima, which `footprint` folds into the run's
    record afterwards; ``baseline`` is what a caller read before the server came up.
    """
    from contextlib import nullcontext

    traced = wants_trace(len(questions), trace)
    rows = []
    # what the server holds while it is answering, sampled -- see `Watching`. The server is
    # found from the client's own base_url, so nothing above has to carry a pid; a client
    # with none, or a URL this machine does not own, samples nothing and says so.
    at = str(getattr(client, "base_url", "") or "")
    with watching(at, baseline=baseline, client=client) if at else nullcontext():
        for one in questions:
            row, _ = _ask_once(ask, one, label=label, client=client, graph=graph,
                               per_question=per_question, trace=traced)
            rows.append(row)
            if log:
                log(f"  {row.seconds:5.1f}s {row.calls:3} calls  {row.question[:56]}"
                    + ("  TIMED OUT" if row.timed_out else ""))
    return rows

def concurrent(ask: Callable[..., Any], questions: Sequence[Mapping[str, Any]], *,
               conversations: int, turns: int, label: str, client: Any,
               graph: Mapping[str, Any] | None = None, base_url: str = "",
               log: Callable[[str], None] | None = None,
               per_question: float = PER_QUESTION,
               trace: bool | None = None) -> tuple[list[Row], dict[str, Any]]:
    """N conversations of T turns each, asked of one server at the same time.

    Each conversation is a chain of questions from the set with the earlier turns carried,
    run on its own thread, so N are in flight together. Each row carries the turn's wall
    clock, the time until the server began generating, and what it spent queueing. Returns
    the rows and what to keep beside them as the run's ``server``: the most the server held
    while it ran, and ``concurrency`` -- the conversations, turns, slots, the whole run's
    wall clock and the queueing summed over the turns.
    """
    from concurrent.futures import ThreadPoolExecutor

    asked = [dict(q) for q in questions]
    if not asked:
        raise ValueError("no questions to converse about")
    if conversations < 1 or turns < 1:
        raise ValueError("at least one conversation of one turn")
    # each conversation takes its own stretch of the set, so two of them do not ask the
    # same question at the same moment and share a prompt cache the real thing would not
    chains = [[asked[(c * turns + t) % len(asked)] for t in range(turns)]
              for c in range(conversations)]
    # every turn of every conversation is a question, and that is what decides whether the
    # transcripts are worth their kilobytes -- see `wants_trace`
    traced = wants_trace(conversations * turns, trace)

    def one_conversation(c: int) -> list[Row]:
        prior: list[dict[str, str]] = []
        rows: list[Row] = []
        for t, question in enumerate(chains[c]):
            row, said = _ask_once(ask, question, label=label, client=client, graph=graph,
                                  turns=prior, conversation=c, turn=t,
                                  per_question=per_question, trace=traced)
            rows.append(row)
            prior += [{"role": "user", "content": row.question},
                      {"role": "assistant", "content": said}]
            if log:
                log(f"  c{c} t{t} {row.seconds:5.1f}s  first token {_clock(row.first_token)}"
                    f"  queued {_clock(row.queued)}  {row.question[:40]}")
        return rows

    slots = bench.slot_count(base_url) if base_url else -1
    watching = _Peak(base_url) if base_url else None
    began = time.time()
    with ThreadPoolExecutor(max_workers=conversations) as pool:
        got = list(pool.map(one_conversation, range(conversations)))
    wall = round(time.time() - began, 2)
    rows = [row for chain in got for row in chain]
    held = watching.stop() if watching else {}
    # None when no turn reported what the server spent: not measured is not 0
    waited = [float(r.queued) for r in rows if r.queued is not None]
    held["concurrency"] = {"conversations": conversations, "turns": turns, "slots": slots,
                           "seconds": wall, "queued": round(sum(waited), 2) if waited else None}
    return rows, held


def _clock(value: float | None) -> str:
    """``1.2s`` for a clock, ``-`` for one nothing read."""
    return f"{float(value):4.1f}s" if value is not None else "   -"

def ask_from(spec: str) -> Callable[[str, Any], Any]:
    """Import ``module:function``. It takes ``(question, client)`` and returns an Answer."""
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"expected module:function, got {spec!r}")
    from importlib import import_module

    return getattr(import_module(module), name)


def asking(graph: Mapping[str, Any], *, how: Any = None, shortlist: int = 0,
           store: str | Path | None = None,
           embed_url: str = "", embed_model: str = "",
           margin: float = MARGIN) -> Callable[..., Any]:
    """The ordinary way to ask this graph a question, with or without a search run first.

    ``how`` is the :class:`~ml_stack.graph.Asking`. ``shortlist`` lets a cheap embedder
    suggest where to look before the large model starts, and ``margin`` is how far the
    nearest vector must stand out before it is offered at all. With a ``store``, `look_up`
    is the hybrid the application ships -- characters, the word index and, given
    ``embed_url``, vectors, fused -- and without one, characters alone.

    The returned callable carries ``.finder`` (see `finding`) and ``.asking``, the way as
    `keep.save` writes it beside the rows. It also takes ``turns=``, the earlier turns of
    a conversation, so `concurrent` can carry one on.
    """
    from ml_stack.asking import Asking
    from ml_stack.graph.conversation import converse
    from ml_stack.graph.looking import tools_for
    from ml_stack.graph.search import hybrid

    how = how if how is not None else Asking()

    finder_name = finding(store, embed_url, embed_model)

    def embedded(text: str) -> list[float] | None:
        # no vectors to search means no vector to search with
        if not embed_url or finder_name != "meaning":
            return None
        from ml_stack.client.embed import embed
        from ml_stack.client.embed import QUERY

        try:
            return embed([QUERY + text], base_url=embed_url, model=embed_model)[0]
        except Exception:  # noqa: BLE001 - the words still vote
            return None

    def likely(question: str, held: Any) -> list[str]:
        if not shortlist:
            return []
        vector = embedded(question)
        if vector is not None and margin > 0:
            near = held.similar(vector, model=embed_model, limit=max(shortlist, 8))
            if not stands_out([r["similarity"] for r in near], margin=margin):
                return []            # nothing here stands out: "hi" is not a search
        found = hybrid(graph, question, store=held, vector=vector, model=embed_model)
        return [r["id"] for r in found][:shortlist]

    def converse_with(question: str, client: Any, finder: Any, opening: Sequence[str],
                      turns: Sequence[Mapping[str, str]]) -> Any:
        # `finder` goes to both, because `converse` swaps look_up's callable in whichever
        # tools it is handed, and the terse set is handed in rather than chosen inside
        tools = (tools_for(graph, terse=True, finder=finder, **how.tools())
                 if how.terse else None)
        return converse(question, graph, client, asking=how, opening=opening,
                        finder=finder, turns=list(turns), tools=tools)

    def ask(question: str, client: Any, *, turns: Sequence[Mapping[str, str]] = ()) -> Any:
        if store is None or not str(store):
            return converse_with(question, client, None, [], turns)
        from ml_stack.graph.store import GraphStore

        # one handle for the whole conversation: a question makes several look_ups, and
        # each is a hybrid search over the store, embedded the same way the question is
        with GraphStore(store, read_only=True) as held:
            def finder(text: str) -> list[dict[str, str]]:
                return hybrid(graph, text, store=held, vector=embedded(text),
                              model=embed_model)

            return converse_with(question, client, finder, likely(question, held), turns)

    ask.finder = finder_name  # type: ignore[attr-defined]
    # Only what was asked for: a way that asked for nothing carries `{"tight": True}` and
    # no other key.
    ask.asking = {  # type: ignore[attr-defined]
        **how.said(), **({"shortlist": int(shortlist)} if shortlist else {}),
    }
    return ask
