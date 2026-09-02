"""Asking the questions and counting what each cost.

The questions themselves (`read_questions`, `sample` -- a short run that still asks about
every kind), the ordinary way to ask a graph one (`asking`), one question through the
client with the bill kept (`Counting`, `_ask_once`, the `--per-question` cap), a set of
them one at a time (`measure`) or N conversations at once (`concurrent`), and what the
server holding the model costs and whether it is free to be timed (`footprint`, `busy`,
`slot_count`, `_idle`).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.footprint`,
# `bench.busy` -- so anything patchable is looked up there at call time, never bound here
# at import.
from ml_stack.graph import bench
from ml_stack.graph.bench.keep import SHORT, SMOKE
from ml_stack.graph.bench.score import Row, unread_named
from ml_stack.graph.vectors import MARGIN, stands_out


def finding(store: str | Path | None, embed_url: str = "") -> str:
    """Which look_up a run measures, named so two runs are never read against each other.

    ``chars``: no store, so `look_up` matches characters and nothing else -- what the bench
    measured for months while the application shipped something better. ``words``: a store's
    word index votes as well, so "compilers" finds "compiler". ``meaning``: and its vectors,
    through the embedder at ``embed_url``. Recorded on every run and printed in the table,
    for the same reason `ctx` is: a comparison across finders is two measurements.
    """
    if store is None or not str(store):
        return "chars"
    return "meaning" if embed_url else "words"


class Counting:
    """A client that answers exactly as the real one does, and keeps the bill.

    Wrapping is the only way to count honestly: the tokens are on the reply the server sent,
    and nothing between here and there is going to add them up for you.
    """

    def __init__(self, client: Any, *, deadline: float | None = None) -> None:
        self.client = client
        # When this question must be over, as `time.time()` reads it. Each call is given the
        # time left, so a question of three calls gets one cap and not three; a call that
        # ends at the deadline with an error is the timeout, whatever it was wrapped as.
        self.deadline = deadline
        self.timed_out = False
        self.calls = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.processed_tokens = 0
        self.completion_tokens = 0
        self.draft_tokens = 0
        self.draft_taken = 0
        # What the server itself spent reading and generating, so that the difference
        # between it and the wall clock -- time spent waiting for a slot -- is a number.
        self.generating_ms = 0.0
        self.first_token: float | None = None

    def chat(self, messages: Any, **kw: Any) -> Any:
        self.calls += 1
        sent = time.time()
        if self.deadline is not None:
            left = self.deadline - sent
            if left <= 0:
                self.timed_out = True
                raise QuestionTimedOut(f"no time left for call {self.calls}")
            # The real client takes `timeout` per call and closes the connection when it
            # expires -- urllib closes the socket on the exception it raises, and llama.cpp's
            # server polls `is_connection_closed` while a non-streamed result is pending and
            # cancels the slot's tasks when it is, so the slot stops generating rather than
            # finishing a reply nobody is waiting for. A client without a `timeout` of its
            # own is only held to the deadline between calls.
            if hasattr(self.client, "timeout"):
                kw.setdefault("timeout", max(0.1, left))
        try:
            reply = self.client.chat(messages, **kw)
        except Exception as exc:
            if self.deadline is not None and time.time() >= self.deadline - 0.5:
                self.timed_out = True
                raise QuestionTimedOut(f"call {self.calls}: {exc}") from exc
            raise
        took = time.time() - sent
        raw = getattr(reply, "raw", None) or {}
        usage = raw.get("usage") or {}
        timings = raw.get("timings") or {}
        prompt_ms = float(timings.get("prompt_ms") or 0)
        predicted_ms = float(timings.get("predicted_ms") or 0)
        self.generating_ms += prompt_ms + predicted_ms
        if self.first_token is None:
            # Nothing here streams, so the first token is not seen arriving. What is known
            # is how long the server spent generating; everything before that -- waiting
            # for a slot, then reading the prompt -- is what the first token waited for.
            self.first_token = round(max(0.0, took - predicted_ms / 1000), 3)
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        # A conversation re-sends everything every turn, so the prompt total counts the same
        # words over and over. What the machine actually pays for is what it had to read:
        # `timings.prompt_n`, with `cache_n` the part it kept from the turn before.
        self.cached_tokens += int(timings.get("cache_n")
                                  or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                                  or 0)
        self.processed_tokens += int(timings.get("prompt_n") or 0)
        # A draft model guesses ahead and the large one checks the guesses in one pass, so
        # what decides whether it was worth serving is not that it ran but how often it was
        # right. Both are zero on a server without one, which is how the table tells them
        # apart without being told.
        self.draft_tokens += int(timings.get("draft_n") or 0)
        self.draft_taken += int(timings.get("draft_n_accepted") or 0)
        return reply

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


class QuestionTimedOut(RuntimeError):
    """A question ran past its `--per-question` cap; the row records it and the run goes on."""


# What one question may take before it is recorded as timed out and the run moves on.
PER_QUESTION = 300.0


def _how_many(args: Any) -> int:
    """How many questions to ask: --sample wins, then --short, then all of them."""
    asked = int(getattr(args, "sample", 0) or 0)
    if getattr(args, "smoke", False):
        return SMOKE
    return asked or (SHORT if getattr(args, "short", False) else 0)


def sample(questions: Sequence[Mapping[str, Any]], n: int,
           graph: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """``n`` questions that still cover every kind of answer, or all of them.

    Not the first n, and not an even stride either. A question set is written in groups --
    finding people, then joining them, then the ones whose answer is nobody -- so the first
    eight are eight of one kind, and an even stride over a set that is two-thirds people
    returns two-thirds people and drops whole kinds. Both measure one thing well and call it
    a shorter benchmark.

    So it takes one of each kind first -- person, org, place, topic, opportunity, event, and
    the questions whose right answer is nobody -- and only then fills up in proportion. A
    short run that has stopped asking about places is not a short run, it is a different one.

    Deterministic, so two short runs compare with each other. They do not compare with a
    full run, which is what the `n` column on every line is for.
    """
    scored = [dict(q) for q in questions]
    if n <= 0 or n >= len(scored):
        return scored

    if graph is None:
        from ml_stack.graph.community import graph as invented

        graph = invented()
    kind = {str(node.get("id")): str(node.get("kind") or "") for node in
            (graph.get("nodes") or ())}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for q in scored:
        want = q.get("expect") or ()
        # a question is filed under the rarest kind it asks for, so a kind that appears in
        # only one question is never crowded out by one that appears in twenty
        kinds = {kind.get(str(e), "?") for e in want} or {"nobody"}
        grouped.setdefault(min(kinds, key=lambda k: sum(
            1 for other in scored
            if k in ({kind.get(str(e), "?") for e in (other.get("expect") or ())}
                     or {"nobody"}))), []).append(q)

    taken: list[dict[str, Any]] = []
    order = sorted(grouped, key=lambda k: len(grouped[k]))
    for group in order:                       # one of every kind, rarest first
        if len(taken) < n:
            taken.append(grouped[group][0])
    for group in order:                       # then in proportion, evenly within each
        rest = grouped[group][1:]
        share = max(0, round((n - len(order)) * len(grouped[group]) / len(scored)))
        step = (len(rest) / share) if share else 0
        for i in range(min(share, len(rest))):
            if len(taken) < n:
                taken.append(rest[int(i * step)])
    for q in scored:                          # and top up in order if rounding left room
        if len(taken) >= n:
            break
        if q not in taken:
            taken.append(q)
    return [q for q in scored if q in taken][:n]


def read_questions(path: str | Path) -> list[dict[str, Any]]:
    """One question per line: a bare string, or ``{"q": ..., "expect": [ids]}``."""
    out = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            row = {"q": line}
        out.append(row if isinstance(row, dict) else {"q": str(row)})
    return out


def _ask_once(ask: Callable[..., Any], one: Mapping[str, Any], *, label: str, client: Any,
              graph: Mapping[str, Any] | None = None, turns: Sequence[Mapping[str, str]] = (),
              conversation: int = 0, turn: int = 0,
              per_question: float = PER_QUESTION) -> tuple[Row, str]:
    """One question through ``ask(question, client)``, and what it cost; with the answer's
    text, which a conversation carries into its next turn.

    ``per_question`` is the most it may take. Past that the row is kept as timed out: no
    answer, the cap as its wall clock, scored wrong -- and the next question is asked. A
    question that hangs is a result, not a reason for the run to.
    """
    began = time.time()
    counting = Counting(client, deadline=began + per_question if per_question else None)
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
    row.first_token = counting.first_token or 0.0
    row.queued = round(max(0.0, row.seconds - counting.generating_ms / 1000), 2)
    row.calls = counting.calls
    row.prompt_tokens = counting.prompt_tokens
    row.cached_tokens = counting.cached_tokens
    row.processed_tokens = counting.processed_tokens
    row.completion_tokens = counting.completion_tokens
    row.draft_tokens = counting.draft_tokens
    row.draft_taken = counting.draft_taken
    return row, said


def measure(ask: Callable[[str, Any], Any], questions: Sequence[dict[str, Any]], *,
            label: str, client: Any, log: Callable[[str], None] | None = None,
            graph: Mapping[str, Any] | None = None,
            per_question: float = PER_QUESTION) -> list[Row]:
    """Ask each question once through ``ask(question, client)`` and record what it cost.

    Given the ``graph``, each row also counts the entries the answer named that no tool
    call produced -- see `unread_named`. ``per_question`` caps each question; see `_ask_once`.
    """
    rows = []
    for one in questions:
        row, _ = _ask_once(ask, one, label=label, client=client, graph=graph,
                           per_question=per_question)
        rows.append(row)
        if log:
            log(f"  {row.seconds:5.1f}s {row.calls:3} calls  {row.question[:56]}"
                + ("  TIMED OUT" if row.timed_out else ""))
    return rows


def slot_count(base_url: str) -> int:
    """How many slots that server has, read from ``/slots`` as `busy` reads it.

    -1 when it will not say. It is what decides whether concurrent conversations queue:
    more of them than slots, and a turn waits for a slot before a token is read.
    """
    from ml_stack.client.http import request_json

    try:
        slots = request_json(f"{base_url.rstrip('/')}/slots", timeout=5.0, method="GET")
    except Exception:  # noqa: BLE001 - a server that will not answer has an unknown count
        return -1
    return len(slots) if isinstance(slots, list) else -1


class _Peak:
    """The most a server held while a run was going.

    `footprint` reads resident memory once, after the fact, and a cache that was full while
    four conversations were in flight has been let go by then. Sampled every second on a
    thread; ``stop`` returns the footprint with the largest resident figure seen.
    """

    def __init__(self, base_url: str, every: float = 1.0) -> None:
        self.base_url = base_url
        self.every = every
        self.most = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        while not self._stop.is_set():
            self.most = max(self.most,
                            int(bench.footprint(self.base_url).get("resident_bytes") or 0))
            self._stop.wait(self.every)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=30)
        out = bench.footprint(self.base_url)
        if self.most > int(out.get("resident_bytes") or 0):
            out["resident_bytes"] = self.most
            for derived_key in ("kv_and_run_bytes", "bytes_per_1k_context", "mmapped"):
                out.pop(derived_key, None)
            try:
                out = beyond_weights(out)
            except Exception:  # noqa: BLE001 - a summary line is never worth a run's answers
                pass
        return out


def concurrent(ask: Callable[..., Any], questions: Sequence[Mapping[str, Any]], *,
               conversations: int, turns: int, label: str, client: Any,
               graph: Mapping[str, Any] | None = None, base_url: str = "",
               log: Callable[[str], None] | None = None,
               per_question: float = PER_QUESTION) -> tuple[list[Row], dict[str, Any]]:
    """N conversations of T turns each, asked of one server at the same time.

    Every other measurement here asks one question at a time, which is right for timing a
    model and wrong for the question a server is actually asked: how many people can talk
    to it at once, and what that costs each of them. Each conversation is a chain of
    questions from the set with the earlier turns carried, run on its own thread, so N of
    them are in flight together. Per turn: the wall clock, the time until the server began
    generating, and what it spent waiting -- the wall clock less what the server itself
    reports reading and generating, which is the queueing when N exceeds the slots. For the
    run: the wall clock of the whole thing, which is not the sum of the turns, the most the
    server held while it ran, and F1 as usual, so a setting that answers faster by
    answering worse is visible.

    Returns the rows and what to keep beside them as the run's ``server``.
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

    def one_conversation(c: int) -> list[Row]:
        prior: list[dict[str, str]] = []
        rows: list[Row] = []
        for t, question in enumerate(chains[c]):
            row, said = _ask_once(ask, question, label=label, client=client, graph=graph,
                                  turns=prior, conversation=c, turn=t,
                                  per_question=per_question)
            rows.append(row)
            prior += [{"role": "user", "content": row.question},
                      {"role": "assistant", "content": said}]
            if log:
                log(f"  c{c} t{t} {row.seconds:5.1f}s  first token {row.first_token:4.1f}s"
                    f"  queued {row.queued:4.1f}s  {row.question[:40]}")
        return rows

    slots = bench.slot_count(base_url) if base_url else -1
    watching = _Peak(base_url) if base_url else None
    began = time.time()
    with ThreadPoolExecutor(max_workers=conversations) as pool:
        got = list(pool.map(one_conversation, range(conversations)))
    wall = round(time.time() - began, 2)
    rows = [row for chain in got for row in chain]
    held = watching.stop() if watching else {}
    held["concurrency"] = {"conversations": conversations, "turns": turns, "slots": slots,
                           "seconds": wall, "queued": round(sum(r.queued for r in rows), 2)}
    return rows, held


def footprint(base_url: str) -> dict[str, Any]:
    """What the server holding this model costs to keep up.

    How many conversations a machine can hold at once is decided by the KV cache, not by the
    weights: the weights are paid for once and the cache is paid for per slot per token. This
    build of llama.cpp does not print a "KV self size" line and `/props` does not carry one,
    so what is measured is the server's resident memory against the weights on disk — the
    difference is the cache and the runtime around it. Said that way rather than called KV
    exactly, because it is not exactly KV.
    """
    from ml_stack.client.http import request_json

    out: dict[str, Any] = {"base_url": base_url}
    try:
        props = request_json(f"{base_url.rstrip('/')}/props", timeout=5.0, method="GET") or {}
        out["model"] = str(props.get("model_path") or "").rsplit("/", 1)[-1]
        out["slots"] = int(props.get("total_slots") or 0)
        out["context"] = int((props.get("default_generation_settings") or {}).get("n_ctx") or 0)
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil

        port = int(base_url.rsplit(":", 1)[-1].strip("/"))
        for process in psutil.process_iter(["pid", "cmdline"]):
            line = " ".join(process.info.get("cmdline") or ())
            if "llama-server" in line and f"--port {port}" in line:
                out["resident_bytes"] = int(process.memory_info().rss)
                break
    except Exception:  # noqa: BLE001 - a number we could not get is not a failed run
        pass
    # The weights come from /props, not the command line: a model served by `hf:` reference
    # is on the command line as a repository, and only the server knows where it landed.
    try:
        where = Path(str(props.get("model_path") or ""))
        if where.exists():
            shards = sorted(where.parent.glob(where.name.replace("00001", "*"))) or [where]
            out["weights_bytes"] = sum(s.stat().st_size for s in shards if s.is_file())
    except Exception:  # noqa: BLE001 - a number we could not get is not a failed run
        pass
    # Resident minus weights, when that means anything. It does not always: llama.cpp mmaps
    # the weights, so a page is resident only once it has been touched, and an MoE that uses
    # ten experts of five hundred never touches most of them. Qwen3.8-Flash-Next sat at 63G
    # resident against 87G of weights on disk -- the subtraction goes negative, clamps to
    # zero and prints as a dash, which reads as "not measured" rather than "not meaningful".
    #
    # So it is only reported when the model really is fully resident, and `resident_bytes`
    # is carried either way, because what the process actually holds is the number that
    # decides how many of them fit.
    try:
        return beyond_weights(out)
    except Exception:  # noqa: BLE001 - a summary line is never worth a run's answers
        return out


def beyond_weights(out: dict[str, Any]) -> dict[str, Any]:
    """Split what the process holds into the weights and everything else.

    Its own function because it is the part that can be wrong without a server: reading
    `kv_and_run_bytes` when a model is mmapped raised at the *end* of a run, after every
    question had been answered, and threw away a quarter hour of GPU for a summary line.
    """
    if "resident_bytes" in out and "weights_bytes" in out:
        beyond = out["resident_bytes"] - out["weights_bytes"]
        if beyond > 0:
            out["kv_and_run_bytes"] = beyond
        else:
            # Less resident than the weights on disk means llama.cpp mapped the file and
            # never paged all of it in -- an MoE's unused experts, most often. The
            # subtraction then says nothing about the cache, so no number is better than a
            # wrong one.
            out["mmapped"] = True
        # What one more conversation costs, which is the question a number like this is
        # asked for. Held tokens are the context times the slots holding one each; dividing
        # by them makes two models comparable however each happened to be configured.
        held = (out.get("context") or 0) * (out.get("slots") or 0)
        if held and "kv_and_run_bytes" in out:
            out["bytes_per_1k_context"] = int(out["kv_and_run_bytes"] / (held / 1024))
    return out


def _idle(url: str, args: Any) -> bool:
    """Refuse to time a server somebody else is using, unless told not to care."""
    working = bench.busy(url)
    if working <= 0:
        if working < 0:
            print(f"note: {url} would not say whether it is busy; timings may not be alone",
                  file=sys.stderr)
        return True
    print(f"error: {url} is already working on {working} request(s). A timing taken while "
          f"another run has the same GPU is not a timing.\n"
          f"       Wait for it, or pass --anyway to measure regardless.", file=sys.stderr)
    return bool(getattr(args, "anyway", False))


def busy(base_url: str) -> int:
    """How many requests that server is already working on.

    A timing taken while somebody else is using the same GPU is not a timing. This is the
    cheapest way to know: llama.cpp's /slots says what each slot is doing, and anything
    above zero means the number about to be measured belongs to two callers at once.

    -1 when the server will not say, which is not the same as idle and is not treated as it.
    """
    from ml_stack.client.http import request_json

    try:
        slots = request_json(f"{base_url.rstrip('/')}/slots", timeout=5.0, method="GET")
    except Exception:  # noqa: BLE001 - a server that will not answer is not known to be idle
        return -1
    if not isinstance(slots, list):
        return -1
    return sum(1 for one in slots if isinstance(one, Mapping) and one.get("is_processing"))


def ask_from(spec: str) -> Callable[[str, Any], Any]:
    """Import ``module:function``. It takes ``(question, client)`` and returns an Answer."""
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"expected module:function, got {spec!r}")
    from importlib import import_module

    return getattr(import_module(module), name)


def asking(graph: Mapping[str, Any], *, shortlist: int = 0, store: str | Path | None = None,
           embed_url: str = "", embed_model: str = "", terse: bool = False,
           margin: float = MARGIN, rich: bool = False,
           tight: bool = False) -> Callable[..., Any]:
    """The ordinary way to ask this graph a question, with or without a search run first.

    Nothing here is any project's: it is `converse` over the graph you handed in. Two
    choices, both about where the looking happens. Whether a cheap embedder gets to suggest
    where to look before the large model starts (``shortlist``), which is the thing most
    worth measuring. And what `look_up` is when the model calls it: with a ``store``, the
    same hybrid the application ships -- characters, the word index and, given
    ``embed_url``, vectors, fused -- and without one, characters alone. For months the bench
    had no store on this path and every ranking it wrote measured a look_up nobody ran.

    ``rich`` asks with `converse(..., rich=True)`: look_up results carry a score and why
    they matched, and a topic hit brings the people joined to it. ``tight`` asks with
    `converse(..., tight=True)`: show is told to light only what answers the question,
    and what is lit is capped.

    The returned callable carries ``.finder`` -- see `finding` -- so a run can write down
    which one it measured, and takes ``turns=`` -- the earlier turns of a conversation, as
    `converse` does -- so `concurrent` can carry one on.
    """
    from ml_stack.graph.ask import converse, tools_for
    from ml_stack.graph.search import hybrid

    def embedded(text: str) -> list[float] | None:
        if not embed_url:
            return None
        from ml_stack.client.embed import embed
        from ml_stack.graph.vectors import QUERY

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
        extra: dict[str, Any] = (
            {"tools": tools_for(graph, terse=True, finder=finder, tight=tight)} if terse else {})
        if rich:
            extra["rich"] = True
        if tight:
            extra["tight"] = True
        return converse(question, graph, client, opening=opening, finder=finder,
                        turns=list(turns), **extra)

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

    ask.finder = finding(store, embed_url)  # type: ignore[attr-defined]
    return ask
