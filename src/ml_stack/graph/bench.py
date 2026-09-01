"""What a change to the asking costs and whether it was worth it.

A graph answers questions through a large model, and every tool call it makes is a whole
round trip. Any change to that — a different prompt, a search run before the model instead of
by it — has to be shown to be an improvement rather than asserted, on wall clock, on tokens,
and on whether the answers were right. Runs are kept, so two of them can be compared later.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from ml_stack.paths import repo_root
from ml_stack.graph.vectors import MARGIN, stands_out

__all__ = ["Counting", "HOME", "Row", "SHORT", "SMOKE", "beyond_weights", "export", "ranking",
           "ask_from", "asking", "compare", "concurrent", "detach", "empties", "finding",
           "footprint", "forget", "main", "measure", "measuring", "prepared", "read_questions",
           "runs", "save", "slot_count", "status", "stop", "table", "tail", "unread_named"]

# Runs are worth keeping: the point of one is to compare it with another, later, and a
# benchmark written to a temporary directory answers no question a week from now.
HOME = Path("~/.ml-stack/bench").expanduser()


def _plain(value: Any) -> Any:
    """``value`` as nothing but dicts, lists, strings, numbers, booleans and None.

    What `save` writes has to come back, and the store keeps JSON. Keys that are not
    strings become strings, dataclasses become their fields, sets and tuples become lists,
    and anything else becomes its ``str`` -- nothing is dropped, because a run that lost
    a field is not the run that was measured. What comes out is fed to ``json.dumps``
    with no ``default``, so anything this missed raises before the store sees it.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {(k if isinstance(k, str) else str(k)): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in (sorted(value, key=str) if isinstance(value, (set, frozenset))
                                    else value)]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def prepared() -> str:
    """The store `prepare` builds by default, when it has been built, else "".

    `run` and `sweep` take it as their `--store` default so that a machine that has run
    `prepare` measures the look_up that ships -- characters, the word index and vectors fused
    -- without being told to. Without one, the bench measures character matching alone, and
    every run recorded that way ranked something nobody runs.
    """
    where = HOME / "graph.ladybug"
    return str(where) if where.exists() else ""


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

# How many questions a short run asks. Chosen by measuring what survives, not by feel: at
# twenty, every kind of answer is still asked about and the mean number of answers expected
# is 2.2 -- the same as the whole set, so the difficulty is preserved and not only the
# variety. Below about fourteen the rarer kinds start to go, and a shorter benchmark that
# has stopped asking about places is not a shorter benchmark but a different one.
#
# The cost is granularity: eighteen scored questions make each one worth 5.6 points of F1
# against 2.9 on the full set, so a small difference is noise on a short run and signal on
# a full one. The `n` column on every line is what keeps the two from being read together.
SHORT = 20
# Enough to walk the whole path -- serve, ask, score, measure the server, save, summarise --
# and short enough that finding out it is broken costs a minute. A sweep once answered every
# question and then raised while writing a summary line, losing all of it.
SMOKE = 2


@dataclass
class Row:
    """One question, asked once, and everything it cost."""

    label: str
    question: str
    seconds: float = 0.0
    calls: int = 0                 # round trips through the large model
    prompt_tokens: int = 0         # everything the model was shown
    cached_tokens: int = 0         # of those, what it had already seen and did not reread
    processed_tokens: int = 0      # of those, what it actually had to read this time
    completion_tokens: int = 0
    steps: str = ""
    answer_chars: int = 0
    draft_tokens: int = 0          # guessed ahead by a draft model, when one is served
    draft_taken: int = 0           # of those, how many the large model kept
    shown: list[str] = field(default_factory=list)
    expected: list[str] = field(default_factory=list)
    # Entries the answer's prose names that no tool call found, read, traversed or showed
    # -- a plausible name the model made up, which F1 cannot see. See `unread_named`.
    unread: list[str] = field(default_factory=list)
    unread_named: int = 0
    # Which conversation and turn this was, when several were asked at once.
    conversation: int = 0
    turn: int = 0
    first_token: float = 0.0       # seconds until the server began generating the first reply
    queued: float = 0.0            # wall clock the turn spent not being read or generated
    error: str = ""

    @property
    def hit(self) -> float:
        """How well what was shown matches what was wanted: F1, -1 when nothing is expected.

        Recall alone is not a score. It was, for one afternoon, and it said a 2B model was
        more accurate than a 120B -- because showing more is free under it and the small
        model showed six entries where 1.7 were wanted, while the large one showed two. A
        model that lit every entry in the graph on every question scored 100%.

        Precision alone is no better: showing nothing is perfect by it. F1 is the pair held
        together, and it is also what the page actually needs -- lighting up people who have
        nothing to do with the question is the complaint this whole thing began with.
        """
        return _score(self.expected, self.shown)[2]

    @property
    def recall(self) -> float:
        """How much of what was wanted was shown."""
        return _score(self.expected, self.shown)[0]

    @property
    def precision(self) -> float:
        """How much of what was shown was wanted."""
        return _score(self.expected, self.shown)[1]


class Counting:
    """A client that answers exactly as the real one does, and keeps the bill.

    Wrapping is the only way to count honestly: the tokens are on the reply the server sent,
    and nothing between here and there is going to add them up for you.
    """

    def __init__(self, client: Any) -> None:
        self.client = client
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
        reply = self.client.chat(messages, **kw)
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
    return out


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


# A label or an answer reduced to its words: lower-cased, anything that is not a letter or
# a digit made a space, and a space at each end so that a whole-word match is a substring
# match. The page's `namedIn` does the same in JS, so the two agree on what an answer names.
_NOT_A_WORD = re.compile(r"[^\w]+|_+")


def _words(text: Any) -> str:
    return " " + " ".join(_NOT_A_WORD.sub(" ", str(text or "").casefold()).split()) + " "


def unread_named(text: str, graph: Mapping[str, Any],
                 touched: Iterable[str] = ()) -> list[str]:
    """The entries an answer names that none of its tool calls produced.

    F1 scores what was lit, and nothing else scored the prose. An answer can name a person
    the model never found, read or showed -- a plausible name, made up or half-remembered
    from an earlier question -- and light the right entries around it, and F1 is none the
    wiser. This is the count that catches it: every entry whose label appears in the
    answer, as whole words and regardless of case, and whose id is in none of ``touched``
    (what `look_up` found, `look_at` read, `path_between` walked, `show` lit).

    Labels shorter than three characters are not matched, as on the page: "Al" is in
    "already". A label another entry shares counts as read if either was touched.
    """
    said = _words(text)
    if not said.strip():
        return []
    seen = {str(i) for i in touched}
    by_label: dict[str, tuple[str, list[str]]] = {}
    for node in graph.get("nodes") or ():
        label = str(node.get("label") or "")
        key = _words(label)
        if len(label) < 3 or not key.strip():
            continue
        by_label.setdefault(key, (label, []))[1].append(str(node.get("id")))
    return sorted(label for key, (label, ids) in by_label.items()
                  if key in said and not any(i in seen for i in ids))


def _ask_once(ask: Callable[..., Any], one: Mapping[str, Any], *, label: str, client: Any,
              graph: Mapping[str, Any] | None = None, turns: Sequence[Mapping[str, str]] = (),
              conversation: int = 0, turn: int = 0) -> tuple[Row, str]:
    """One question through ``ask(question, client)``, and what it cost; with the answer's
    text, which a conversation carries into its next turn."""
    counting = Counting(client)
    row = Row(label=label, question=str(one.get("q") or ""),
              expected=[str(i) for i in (one.get("expect") or ())],
              conversation=conversation, turn=turn)
    said = ""
    began = time.time()
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
            graph: Mapping[str, Any] | None = None) -> list[Row]:
    """Ask each question once through ``ask(question, client)`` and record what it cost.

    Given the ``graph``, each row also counts the entries the answer named that no tool
    call produced -- see `unread_named`.
    """
    rows = []
    for one in questions:
        row, _ = _ask_once(ask, one, label=label, client=client, graph=graph)
        rows.append(row)
        if log:
            log(f"  {row.seconds:5.1f}s {row.calls:3} calls  {row.question[:56]}")
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
            self.most = max(self.most, int(footprint(self.base_url).get("resident_bytes") or 0))
            self._stop.wait(self.every)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=30)
        out = footprint(self.base_url)
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
               log: Callable[[str], None] | None = None) -> tuple[list[Row], dict[str, Any]]:
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
                                  turns=prior, conversation=c, turn=t)
            rows.append(row)
            prior += [{"role": "user", "content": row.question},
                      {"role": "assistant", "content": said}]
            if log:
                log(f"  c{c} t{t} {row.seconds:5.1f}s  first token {row.first_token:4.1f}s"
                    f"  queued {row.queued:4.1f}s  {row.question[:40]}")
        return rows

    slots = slot_count(base_url) if base_url else -1
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


class RunNotKept(RuntimeError):
    """A run was written and did not come back the way `runs` reads it."""


def save(store: str | Path, rows: Sequence[Row], *, held: dict[str, Any] | None = None) -> str:
    """Keep a run where it can be compared with another one, later, by anybody.

    Then read it back the way `runs` will, on a fresh handle, and refuse to return until
    what came back is what went in. Twelve runs -- half an hour of GPU -- were once kept
    as nothing: the store took them, a scan of it returned an empty string for each, and
    the sweep printed its summary from memory, so nobody knew until the next morning. The
    read-back is the only proof that a run exists, and it is cheap next to the run.
    """
    from ml_stack.graph.store import GraphStore

    server = held
    stem = f"bench:{rows[0].label}:{time.strftime('%Y%m%dT%H%M%S')}" if rows else "bench:empty"
    record = _plain({"at": time.strftime("%FT%T"), "label": rows[0].label if rows else "",
                     "server": server or {}, "rows": [asdict(r) for r in rows],
                     "unread_named": sum(r.unread_named for r in rows)})
    record = json.loads(json.dumps(record))      # no default=: anything left raises here
    with GraphStore(store) as writer:
        # Two runs of one label inside a second used to land on the same key and the later
        # one silently replaced the earlier. A run took minutes when that was written; with
        # answers cached it can take no time at all, so the collision is real now.
        key, n = stem, 1
        while writer.get_doc(key) is not None:
            key, n = f"{stem}-{n}", n + 1
        writer.put_doc(key, record)
    back = next((r for r in runs(store) if r.get("key") == key), None)
    if back is None:
        raise RunNotKept(f"{key} was written to {store} and did not come back")
    back = {k: v for k, v in back.items() if k != "key"}
    if back != record:
        differs = sorted(k for k in set(back) | set(record) if back.get(k) != record.get(k))
        raise RunNotKept(f"{key} came back from {store} changed: {', '.join(differs)} differ")
    return key


def runs(store: str | Path, label: str = "") -> list[dict[str, Any]]:
    """Every run kept in ``store``, newest last, optionally only one label's.

    Each carries its ``key``. A doc that reads back empty is left out -- it is not a run,
    and a row of dashes in the table said nothing about why -- and `empties` names them.
    """
    from ml_stack.graph.store import GraphStore

    with GraphStore(store, read_only=True) as held:
        kept = held.docs()
    found = [{**kept[k], "key": k} for k in sorted(kept)
             if k.startswith("bench:") and isinstance(kept[k], dict) and kept[k]]
    return [r for r in found if not label or r.get("label") == label]


def empties(store: str | Path) -> list[str]:
    """The keys of runs that read back as nothing, which `forget --empty` removes."""
    from ml_stack.graph.store import GraphStore

    if not Path(store).expanduser().exists():
        return []
    with GraphStore(store, read_only=True) as held:
        kept = held.docs()
    return sorted(k for k, v in kept.items()
                  if k.startswith("bench:") and not (isinstance(v, dict) and v))


def forget(store: str | Path, *, label: str = "", empty: bool = False) -> list[str]:
    """Delete runs: every empty one, or every run of one label. Returns what went."""
    from ml_stack.graph.store import GraphStore

    going = empties(store) if empty else [r["key"] for r in runs(store, label)] if label else []
    if not going:
        return []
    with GraphStore(store) as held:
        for key in going:
            held.delete_doc(key)
    return going


def _total(rows: Sequence[dict[str, Any]], key: str) -> float:
    return sum(float(r.get(key) or 0) for r in rows)


def compare(store: str | Path, first: str, second: str) -> str:
    """The two labels side by side, and the difference between them."""
    sides = []
    for label in (first, second):
        kept = runs(store, label)
        if not kept:
            return f"no run labelled {label!r} in {store}"
        sides.append((label, kept[-1]["rows"]))
    lines = [f"{'':22} {first:>16} {second:>16}   difference"]
    scored = [[r for r in rows if r.get("expected")] for _, rows in sides]

    def row(name: str, a: float, b: float, unit: str = "", better_lower: bool = True) -> str:
        gap = b - a
        way = "" if not a else f"  {gap / a * +100:+.0f}%"
        return f"{name:22} {a:>16.1f}{unit} {b:>16.1f}{unit}{way}"

    a, b = (rows for _, rows in sides)
    lines.append(row("wall clock (s)", _total(a, "seconds"), _total(b, "seconds")))
    lines.append(row("model calls", _total(a, "calls"), _total(b, "calls")))
    lines.append(row("prompt tokens (shown)", _total(a, "prompt_tokens"),
                     _total(b, "prompt_tokens")))
    lines.append(row("  of those, cached", _total(a, "cached_tokens"), _total(b, "cached_tokens")))
    lines.append(row("  of those, read", _total(a, "processed_tokens"),
                     _total(b, "processed_tokens")))
    lines.append(row("completion tokens", _total(a, "completion_tokens"),
                     _total(b, "completion_tokens")))
    lines.append(row("paid for (read+written)",
                     _total(a, "processed_tokens") + _total(a, "completion_tokens"),
                     _total(b, "processed_tokens") + _total(b, "completion_tokens")))
    lines.append(row("answered (chars)", _total(a, "answer_chars"), _total(b, "answer_chars")))
    if scored[0] and scored[1]:
        def hits(rows: Sequence[dict[str, Any]]) -> float:
            got = [len(set(r["expected"]) & set(r["shown"])) / len(r["expected"])
                   for r in rows if r.get("expected")]
            return 100 * sum(got) / len(got) if got else 0.0
        lines.append(row("expected shown (%)", hits(a), hits(b), unit="%"))
    failed = [sum(1 for r in rows if r.get("error")) for _, rows in sides]
    lines.append(row("failures", failed[0], failed[1]))
    return "\n".join(lines)


def _shown(label: Any, width: int = 28) -> str:
    """A label that fits the column with its *end* intact: the end is where a variant lives.

    `gemma-4-E2B-it-plain-terse` cut to 20 characters read as `gemma-4-E2B-it-plain`, and
    three runs that differed only in the asking printed as one -- measured, and mistaken
    for a labelling bug before it was seen to be the column.
    """
    text = str(label or "")
    return text if len(text) <= width else "…" + text[-(width - 1):]


def table(kept: Sequence[dict[str, Any]]) -> None:
    """Every run, one per line. Two runs compare; more than two want seeing at once."""
    if not kept:
        print("nothing kept yet")
        return
    # The context each slot gets is on every line on purpose: a model measured at 8k against
    # one at 32k is not being compared with it, and the KV cache — the number that decides
    # how many conversations fit at once — scales with exactly that.
    #
    # So is `n`, the number of scored questions, and for the same reason: adding a question
    # to the set changes every score after it, and a run of ten against a run of nine is two
    # different measurements sitting on adjacent lines looking like one. That happened here —
    # a question was added mid-afternoon and the next run read 85% against an earlier 72%,
    # which meant nothing at all. A column is cheaper than remembering.
    #
    # `conc` is the same lesson again: four conversations at once against one at a time is
    # two measurements, and the wall clock of the first is the run's, not the turns' sum.
    # `made` is what F1 cannot see -- entries the prose named that nothing found or read.
    head = (f"{'run':28} {'ctx':>7} {'n':>3} {'wall':>7} {'calls':>6} {'read':>8} "
            f"{'written':>8} {'cached':>8} {'draft':>6} {'find':>7} {'conc':>5} "
            f"{'resident':>9} {'kv+run':>8} {'per 1k':>8} {'F1':>5} {'rec':>5} {'prec':>5} "
            f"{'made':>5}  {'sampling'}")
    print(head)
    print("-" * len(head))
    for one in kept:
        rows = one.get("rows") or []
        server = one.get("server") or {}
        scored = [r for r in rows if r.get("expected")]
        def mean(f: Callable[[Mapping[str, Any]], float]) -> str:
            return f"{100 * sum(f(r) for r in scored) / len(scored):.0f}%" if scored else "-"
        right, rec, prec = mean(_hit), mean(_recall), mean(_precision)
        ctx = server.get("context") or 0
        slots = server.get("slots") or 0
        beyond = server.get("kv_and_run_bytes")
        per1k = server.get("bytes_per_1k_context")
        rss = server.get("resident_bytes")
        print(f"{_shown(one.get('label', '')):28} "
              f"{(f'{ctx // 1024}k x{slots}' if ctx else '-'):>7} "
              f"{len(scored):>3} "
              f"{wall_of(one):>6.0f}s {_total(rows, 'calls'):>6.0f} "
              f"{_total(rows, 'processed_tokens'):>8.0f} "
              f"{_total(rows, 'completion_tokens'):>8.0f} "
              f"{_total(rows, 'cached_tokens'):>8.0f} "
              f"{drafting(rows):>6} "
              f"{str(server.get('finder') or '-'):>7} "
              f"{at_once(server):>5} "
              f"{(f'{rss / 2**30:.2f}G' if rss else '-'):>9} "
              f"{(f'{beyond / 2**30:.2f}G' if beyond else ('mmap' if server.get('mmapped') else '-')):>8} "
              f"{(f'{per1k / 2**20:.1f}M' if per1k else '-'):>8} "
              f"{right:>5} {rec:>5} {prec:>5} {made(one):>5}  {sampled(server)}")


def wall_of(one: Mapping[str, Any]) -> float:
    """What a run took: its turns added up, or, asked at once, the clock over all of them."""
    at_once_ = (one.get("server") or {}).get("concurrency") or {}
    if at_once_.get("seconds"):
        return float(at_once_["seconds"])
    return _total(one.get("rows") or [], "seconds")


def at_once(server: Mapping[str, Any]) -> str:
    """``4x3`` for four conversations of three turns asked together; "" for one at a time."""
    held = server.get("concurrency") or {}
    if not isinstance(held, Mapping) or not held.get("conversations"):
        return ""
    return f"{held['conversations']}x{held.get('turns') or 1}"


def made(one: Mapping[str, Any]) -> str:
    """How many entries a run's answers named without ever finding or reading them, over
    the scored questions; "" for a run from before this was counted."""
    rows = [r for r in (one.get("rows") or []) if r.get("expected")]
    if not any("unread_named" in r for r in rows):
        return ""
    return str(int(_total(rows, "unread_named")))


def missed(kept: Sequence[Mapping[str, Any]], *, everything: bool = False) -> None:
    """Question by question: what was wanted, what the answer showed, and what it cost.

    A score is a number to act on only when you can see which questions made it. A run that
    scores 17% has failed in some particular way — no tool calls, an empty answer, the right
    people found and the wrong ones shown — and the aggregate cannot tell you which, so this
    prints the rows themselves. Only the misses by default; ``everything`` for all of them.
    """
    if not kept:
        print("nothing kept yet")
        return
    for one in kept:
        rows = [r for r in (one.get("rows") or []) if r.get("expected")]
        shortfall = [r for r in rows if not everything and _hit(r) < 1.0] if not everything else rows
        server = one.get("server") or {}
        found = str(server.get("finder") or "-")
        together = at_once(server)
        print(f"\n{one.get('label', '')}  ({one.get('at', '')}, find {found}"
              + (f", {together} at once" if together else "") + ")")
        if not shortfall:
            print("  every question answered in full")
            continue
        for r in shortfall:
            got, want = set(r.get("shown") or ()), set(r.get("expected") or ())
            print(f"  {_hit(r) * 100:3.0f}%  {r.get('question', '')}")
            print(f"        wanted  {', '.join(sorted(want)) or '-'}")
            print(f"        showed  {', '.join(sorted(got)) or '(nothing)'}")
            if want - got:
                print(f"        missed  {', '.join(sorted(want - got))}")
            if r.get("unread"):
                # what F1 cannot see: a name in the prose that no tool call produced
                print(f"        made    {', '.join(r['unread'])}  (named, never found or read)")
            note = (f"{r.get('calls', 0)} calls, {r.get('answer_chars', 0)} chars"
                    + (f", ERROR {r['error']}" if r.get("error") else ""))
            if together:
                note += (f"; conversation {r.get('conversation', 0)} turn {r.get('turn', 0)}, "
                         f"first token {r.get('first_token', 0):.1f}s, "
                         f"queued {r.get('queued', 0):.1f}s")
            print(f"        {note}")


def _score(expected: Sequence[str], shown: Sequence[str]) -> tuple[float, float, float]:
    """``(recall, precision, f1)`` for one answer. All -1 when nothing was expected."""
    want, got = set(expected or ()), set(shown or ())
    if not want:
        return (-1.0, -1.0, -1.0)
    hit = len(want & got)
    recall = hit / len(want)
    precision = hit / len(got) if got else 0.0
    f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
    return (recall, precision, f1)


def _hit(row: Mapping[str, Any]) -> float:
    """How well a kept row's answer matched what was wanted, as F1."""
    return _score(row.get("expected") or (), row.get("shown") or ())[2]


def _recall(row: Mapping[str, Any]) -> float:
    return _score(row.get("expected") or (), row.get("shown") or ())[0]


def _precision(row: Mapping[str, Any]) -> float:
    return _score(row.get("expected") or (), row.get("shown") or ())[1]


def shape(questions: Sequence[Mapping[str, Any]], graph: Mapping[str, Any]) -> None:
    """What a question set is made of, so its bias is visible without counting by hand.

    A set that is nine-tenths person-shaped rewards anything that prefers people, whether
    or not that rule is right. That was the state of this one, and it flattered a filter
    measured against it -- so the shape of the set is printed rather than assumed.
    """
    kinds = {str(n.get("id")): str(n.get("kind") or "") for n in (graph.get("nodes") or ())}
    scored = [q for q in questions if q.get("expect")]
    if not scored:
        print("no scored questions")
        return
    counted: dict[str, int] = {}
    for q in scored:
        for kind in {kinds.get(str(e), "?") for e in q["expect"]}:
            counted[kind] = counted.get(kind, 0) + 1
    peopleless = sum(1 for q in scored
                     if not any(kinds.get(str(e)) == "person" for e in q["expect"]))
    print(f"{len(questions)} questions, {len(scored)} scored, "
          f"{len(questions) - len(scored)} whose right answer is nobody")
    print(f"graph: {len(graph.get('nodes') or ())} entries, "
          f"{len(graph.get('edges') or ())} links")
    for kind, n in sorted(counted.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3} question(s) want a {kind}")
    print(f"  {peopleless:>3} question(s) want no person at all "
          f"({100 * peopleless / len(scored):.0f}%)")
    print(f"mean entries expected: {sum(len(q['expect']) for q in scored) / len(scored):.1f}")
    missing = sorted({str(e) for q in scored for e in q["expect"] if str(e) not in kinds})
    if missing:
        print(f"\nEXPECTED IDS THAT DO NOT EXIST IN THE GRAPH: {missing}")


def sampled(server: Mapping[str, Any]) -> str:
    """The sampling a run used, short enough for a column: "t1.0 p.95 k64"."""
    held = server.get("sampling") or {}
    if not isinstance(held, Mapping) or not held:
        return "-"
    bits = []
    for key, tag in (("temperature", "t"), ("top_p", "p"), ("top_k", "k"), ("min_p", "m")):
        if key in held:
            value = held[key]
            bits.append(f"{tag}{value:g}" if not isinstance(value, float) or value >= 1
                        else f"{tag}{value:g}".replace("0.", "."))
    return " ".join(bits) or "-"


def _idle(url: str, args: Any) -> bool:
    """Refuse to time a server somebody else is using, unless told not to care."""
    working = busy(url)
    if working <= 0:
        if working < 0:
            print(f"note: {url} would not say whether it is busy; timings may not be alone",
                  file=sys.stderr)
        return True
    print(f"error: {url} is already working on {working} request(s). A timing taken while "
          f"another run has the same GPU is not a timing.\n"
          f"       Wait for it, or pass --anyway to measure regardless.", file=sys.stderr)
    return bool(getattr(args, "anyway", False))


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


def derived(one: Mapping[str, Any]) -> dict[str, float]:
    """What a run cost per unit of getting the answer right.

    A score on its own cannot choose between two models: one is more accurate and one is
    cheaper, and which to serve depends on what is scarce. These put accuracy over each of
    the three scarcities in turn -- time, tokens, and the memory a conversation holds -- so
    the trade is a number rather than an argument.

    Right-per-second and right-per-1k are rates: twice the figure is twice the accuracy for
    the same cost. `per_right` inverts them into what one right answer cost, which is the
    easier one to feel.
    """
    rows = [r for r in (one.get("rows") or []) if r.get("expected")]
    if not rows:
        return {}
    got = sum(_hit(r) for r in rows) / len(rows)
    recall = sum(_recall(r) for r in rows) / len(rows)
    precision = sum(_precision(r) for r in rows) / len(rows)
    shown = sum(len(r.get("shown") or ()) for r in rows) / len(rows)
    wanted = sum(len(r.get("expected") or ()) for r in rows) / len(rows)
    seconds = wall_of(one)
    paid = _total(rows, "processed_tokens") + _total(rows, "completion_tokens")
    server = one.get("server") or {}
    memory = float(server.get("kv_and_run_bytes") or 0)
    # `right` is F1. Recall and precision are kept beside it because the pair is what says
    # *how* a run was wrong: a model that lights everything has high recall and no precision,
    # and under recall alone it looked like the most accurate model there was.
    out = {"right": got, "recall": recall, "precision": precision,
           "shown_per_question": shown, "wanted_per_question": wanted,
           "seconds": seconds, "paid_tokens": paid, "calls": _total(rows, "calls"),
           "kv_bytes": memory, "questions": float(len(rows))}
    # Rates, guarded: a run that took no time or paid nothing has nothing to divide by, and
    # a zero score is a real answer rather than a missing one.
    if seconds > 0:
        out["right_per_minute"] = got * 60.0 / seconds
    if paid > 0:
        out["right_per_1k"] = got * 1000.0 / paid
    if memory > 0:
        out["right_per_gb"] = got / (memory / 2**30)
    if got > 0:
        out["seconds_per_right"] = seconds / (got * len(rows))
        out["tokens_per_right"] = paid / (got * len(rows))
    return out


def pareto(kept: Sequence[Mapping[str, Any]], *,
           cost: str = "seconds") -> list[Mapping[str, Any]]:
    """The runs nothing else beats on both accuracy and ``cost``.

    A run is dominated when another is at least as accurate *and* costs no more; those are
    the ones there is never a reason to choose. What is left is the frontier — every point
    on it is the best available at some budget, and choosing among them is choosing a budget
    rather than choosing a better run.

    ``cost`` is any key `derived` produces: seconds, paid_tokens, kv_bytes.
    """
    scored = [(one, derived(one)) for one in kept]
    scored = [(one, d) for one, d in scored if d and d.get(cost) is not None]
    front = []
    for one, mine in scored:
        beaten = any(
            other is not one
            and theirs["right"] >= mine["right"] and theirs[cost] <= mine[cost]
            and (theirs["right"] > mine["right"] or theirs[cost] < mine[cost])
            for other, theirs in scored)
        if not beaten:
            front.append(one)
    return sorted(front, key=lambda one: derived(one)[cost])


def _which(graph: Mapping[str, Any]) -> str:
    """A fingerprint of the graph a run was asked of, so an export can tell them apart."""
    from ml_stack.graph.cache import digest

    try:
        return digest(graph)
    except Exception:  # noqa: BLE001 - a graph that will not hash is not the invented one
        return ""


def ranking(kept: Sequence[Mapping[str, Any]], where: str | Path | None = None) -> str:
    """Which model to choose, as a conclusion rather than as evidence.

    The raw runs are not committed: they describe one machine and one llama.cpp build, go
    stale with the next model release, and may have been asked of a real community. What
    survives all of that is the *ranking* -- which model answers best, what it costs, and
    which draft head and sampling were chosen for it -- because that is what the defaults in
    this library are set from, and a default with no recorded reason is a default nobody can
    argue with.

    One line per model: its best run, and the questions and sampling that run used, because
    a score without them is not comparable to anything.
    """
    best: dict[str, Mapping[str, Any]] = {}
    too_few = 0
    for row in _exportable(kept)[0]:
        # A smoke run asks two questions to prove the path works; its score is meaningless
        # by construction, and it would otherwise rank a model on a coin toss. Anything
        # below a short run is evidence that something ran, not evidence of how well.
        if (row.get("questions") or 0) < SHORT:
            too_few += 1
            continue
        name = str(row.get("model") or "?")
        if name not in best or (row.get("f1") or 0) > (best[name].get("f1") or 0):
            best[name] = row

    order = sorted(best.values(), key=lambda r: -(r.get("f1") or 0))
    lines = ["# Which model answers best",
             "",
             "Measured over the invented community that ships with this package, by",
             "`ml-stack-bench`. A conclusion, not evidence: the runs behind it are not in this",
             "repository. Re-measure after any model release -- none of this survives one.",
             "",
             "| model | F1 | recall | precision | questions | seconds | resident | sampling "
             "| find | made |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in order:
        gb = row.get("resident_bytes")
        temp = (row.get("sampling") or {}).get("temperature")
        lines.append(
            f"| `{row.get('model')}` "
            f"| {(row.get('f1') or 0) * 100:.0f}% "
            f"| {(row.get('recall') or 0) * 100:.0f}% "
            f"| {(row.get('precision') or 0) * 100:.0f}% "
            f"| {row.get('questions') or '-'} "
            f"| {row.get('seconds') or 0:.0f} "
            f"| {f'{gb / 2**30:.1f}G' if gb else '-'} "
            f"| {'greedy' if temp == 0 else (f'temp {temp}' if temp is not None else '-')} "
            f"| {row.get('finder') or '-'} "
            f"| {'-' if row.get('unread_named') is None else row['unread_named']} |")
    if too_few:
        lines += ["", f"*{too_few} run(s) not ranked: fewer than {SHORT} questions, which is "
                      f"a smoke run proving the path works rather than a measurement.*"]
    body = "\n".join(lines) + "\n"
    if where is not None:
        Path(where).expanduser().write_text(body, encoding="utf-8")
    return body


def invented_digest() -> str:
    """The fingerprint of the community that ships with this package."""
    from ml_stack.graph.cache import digest
    from ml_stack.graph.community import graph as invented

    return digest(invented())


def _exportable(kept: Sequence[Mapping[str, Any]], *,
                anyway: bool = False) -> tuple[list[dict[str, Any]], int]:
    """The runs that may leave this machine, and how many were held back.

    Shared by `export` and `ranking` so the gate cannot be enforced in one and forgotten in
    the other: only runs whose recorded graph fingerprint is the community that ships with
    this package, and never a run from before that marker existed -- not knowing which graph
    a run read is not the same as knowing it was invented.
    """
    mine = "" if anyway else invented_digest()
    out: list[dict[str, Any]] = []
    skipped = 0
    for one in kept:
        rows = [r for r in (one.get("rows") or []) if r.get("expected")]
        if not rows:
            continue
        if mine and str((one.get("server") or {}).get("graph") or "") != mine:
            skipped += 1
            continue
        got = derived(one)
        server = one.get("server") or {}
        out.append({
            "at": one.get("at", ""), "label": one.get("label", ""),
            "questions": len(rows),
            "f1": round(got.get("right", 0), 4),
            "recall": round(got.get("recall", 0), 4),
            "precision": round(got.get("precision", 0), 4),
            "lit_per_question": round(got.get("shown_per_question", 0), 2),
            "seconds": round(got.get("seconds", 0)),
            "calls": int(got.get("calls", 0)),
            "read_tokens": int(_total(rows, "processed_tokens")),
            "written_tokens": int(_total(rows, "completion_tokens")),
            "draft_offered": int(_total(rows, "draft_tokens")),
            "draft_kept": int(_total(rows, "draft_taken")),
            "context": server.get("context"), "slots": server.get("slots"),
            "model": server.get("model", ""), "draft_model": server.get("draft_model", ""),
            "resident_bytes": server.get("resident_bytes"),
            "kv_and_run_bytes": server.get("kv_and_run_bytes"),
            "mmapped": bool(server.get("mmapped")),
            "sampling": server.get("sampling") or {},
            "finder": str(server.get("finder") or ""),
            # None, not 0, for a run from before this was counted: not counted is not none
            "unread_named": (int(_total(rows, "unread_named"))
                             if any("unread_named" in r for r in rows) else None),
            "concurrency": dict(server.get("concurrency") or {}) or None,
        })
    return out, skipped


def export(kept: Sequence[Mapping[str, Any]], where: str | Path, *,
           anyway: bool = False) -> str:
    """Write every run as JSON, so a day of measuring is not in one place on one machine.

    The store lives under ~/.ml-stack and nothing backs it up: it dies with the disk, and a
    comparison a week from now has nothing to compare against. Everything measured here is
    asked of an invented community, so the results carry no one's details and can sit in a
    repository beside the code that produced them.

    Kept small on purpose -- the totals and the server, not every question's row -- because
    a file nobody will open is a file nobody will notice going stale.

    **Only runs over the community that ships with this package.** `run --graph` takes any
    graph, so a run may have been asked of a real one, and this file is meant for a public
    repository. Omitting the questions and the entry ids is what the current field list
    happens to do; refusing a run that was not over the invented community is what stops the
    next field from leaking. A run from before the marker existed is refused for the same
    reason -- not knowing which graph it read is not the same as knowing it was invented.

    ``anyway`` exports everything, for a store that never left the machine.
    """
    out, skipped = _exportable(kept, anyway=anyway)
    out.sort(key=lambda r: (r["label"], r["at"]))
    target = Path(where).expanduser()
    repo = repo_root(target.parent)
    if repo and not anyway:
        raise ValueError(
            f"{target} is inside the git repository at {repo}. These numbers describe one "
            f"machine and one llama.cpp build, they go stale with the next model release, "
            f"and a run may have been asked of a real community -- so they are backed up, "
            f"not committed. Write it somewhere outside a repository.")
    target.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if skipped:
        print(f"{skipped} run(s) left out: not measured over the community that ships with "
              f"this package, so they may carry a real one's questions. --anyway to include "
              f"them, but not into a repository.", file=sys.stderr)
    return str(target)


def rates(kept: Sequence[Mapping[str, Any]], *, cost: str = "seconds") -> None:
    """Every run by what it cost to be right, frontier marked."""
    if not kept:
        print("nothing kept yet")
        return
    on_front = {id(one) for one in pareto(kept, cost=cost)}
    head = (f"{'run':28} {'n':>3} {'F1':>5} {'rec':>5} {'prec':>5} {'lit/q':>6} "
            f"{'F1/min':>8} {'F1/1k tok':>10} {'F1/GB':>7} {'s per':>7} {'tok per':>8}")
    print(head)
    print("-" * len(head))
    for one in sorted(kept, key=lambda o: -(derived(o).get("right") or 0)):
        d = derived(one)
        if not d:
            continue
        def num(key: str, fmt: str) -> str:
            return format(d[key], fmt) if key in d else "-"
        mark = " *" if id(one) in on_front else "  "
        print(f"{str(one.get('label',''))[:18]:18}{mark} {d['questions']:>3.0f} "
              f"{100 * d['right']:>4.0f}% {100 * d['recall']:>4.0f}% "
              f"{100 * d['precision']:>4.0f}% {d['shown_per_question']:>6.1f} "
              f"{num('right_per_minute', '8.2f')} {num('right_per_1k', '10.4f')} "
              f"{num('right_per_gb', '7.3f')} {num('seconds_per_right', '7.1f')} "
              f"{num('tokens_per_right', '8.0f')}")
    print(f"\n* on the frontier for accuracy against {cost}: nothing is both more accurate "
          f"and cheaper.")


AXES = {"seconds": "wall clock (s)", "paid_tokens": "tokens paid for (read + written)",
        "kv_bytes": "KV cache and runtime (GB)"}


def plot(kept: Sequence[Mapping[str, Any]], where: str | Path, *,
         cost: str = "seconds") -> str:
    """Write accuracy against cost as a self-contained HTML scatter, frontier joined.

    Plain SVG built here rather than a plotting library: this has to open on a machine with
    no network and no packages, and a chart of a dozen points is a dozen circles. The
    frontier is drawn as a line through the runs nothing beats on both axes, so the shape of
    the trade is visible rather than inferred from a column of numbers.
    """
    points = [(one, derived(one)) for one in kept]
    points = [(one, d) for one, d in points if d and cost in d and d[cost] > 0]
    if not points:
        raise ValueError("nothing to plot")
    front = {id(one) for one in pareto([one for one, _ in points], cost=cost)}

    wide, tall, pad = 900, 520, 70
    costs = [d[cost] for _, d in points]
    lo, hi = 0.0, max(costs) * 1.08
    best = max(d["right"] for _, d in points)
    top = min(1.0, best * 1.15)

    def x(v: float) -> float:
        return pad + (v - lo) / (hi - lo or 1) * (wide - 2 * pad)

    def y(v: float) -> float:
        return tall - pad - (v / (top or 1)) * (tall - 2 * pad)

    marks, dots = [], []
    for one, d in sorted(points, key=lambda kv: kv[1][cost]):
        cx, cy = x(d[cost]), y(d["right"])
        on = id(one) in front
        label = _shown(one.get("label", ""), 28)
        dots.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{6 if on else 4.5}" '
            f'class="{"front" if on else "dot"}"><title>{label}\n'
            f'{100 * d["right"]:.0f}% right, {d[cost]:.0f} {cost}\n'
            f'{d["questions"]:.0f} questions</title></circle>')
        if on:
            marks.append((cx, cy))
            dots.append(f'<text x="{cx + 9:.1f}" y="{cy - 8:.1f}" class="tag">{label}</text>')
    line = ("<polyline class=\"edge\" points=\""
            + " ".join(f"{a:.1f},{b:.1f}" for a, b in sorted(marks)) + "\"/>") if marks else ""

    ticks = []
    for n in range(5):
        v = lo + (hi - lo) * n / 4
        ticks.append(f'<line class="grid" x1="{x(v):.1f}" y1="{y(0):.1f}" '
                     f'x2="{x(v):.1f}" y2="{y(top):.1f}"/>'
                     f'<text class="ax" x="{x(v):.1f}" y="{y(0) + 20:.1f}" '
                     f'text-anchor="middle">{v:.0f}</text>')
        r = top * n / 4
        ticks.append(f'<line class="grid" x1="{x(lo):.1f}" y1="{y(r):.1f}" '
                     f'x2="{x(hi):.1f}" y2="{y(r):.1f}"/>'
                     f'<text class="ax" x="{x(lo) - 10:.1f}" y="{y(r) + 4:.1f}" '
                     f'text-anchor="end">{100 * r:.0f}%</text>')

    out = Path(where)
    out.write_text(f"""<title>Answering the graph: accuracy against {AXES.get(cost, cost)}</title>
<style>
  :root {{ --ink:#1b1b1f; --thin:#d8d8de; --front:#1a6b4a; --dot:#8a8a95; --paper:#fbfbfd; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --ink:#e9e9ef; --thin:#33333c; --front:#5fd3a0; --dot:#6f6f7c; --paper:#141418; }} }}
  :root[data-theme="dark"] {{ --ink:#e9e9ef; --thin:#33333c; --front:#5fd3a0;
    --dot:#6f6f7c; --paper:#141418; }}
  body {{ background: var(--paper); color: var(--ink); margin: 0; padding: 24px;
          font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif; }}
  h1 {{ font-size: 17px; margin: 0 0 4px; }}
  p  {{ margin: 0 0 18px; color: var(--dot); max-width: 62ch; }}
  .wrap {{ overflow-x: auto; }}
  .grid {{ stroke: var(--thin); stroke-width: 1; }}
  .ax   {{ fill: var(--dot); font-size: 11px; }}
  .dot  {{ fill: var(--dot); }}
  .front{{ fill: var(--front); }}
  .edge {{ fill: none; stroke: var(--front); stroke-width: 2; stroke-dasharray: 5 4; }}
  .tag  {{ fill: var(--ink); font-size: 11px; }}
</style>
<h1>Answering the graph: accuracy against {AXES.get(cost, cost)}</h1>
<p>Each point is one benchmark run. Green points are the Pareto frontier &mdash; nothing is
both more accurate and cheaper &mdash; so choosing among them is choosing a budget, not
choosing a better run. Hover a point for its numbers.</p>
<div class="wrap"><svg width="{wide}" height="{tall}" viewBox="0 0 {wide} {tall}"
  role="img" aria-label="accuracy against {cost}">
  {"".join(ticks)}
  {line}
  {"".join(dots)}
  <text class="ax" x="{wide / 2:.0f}" y="{tall - 14}" text-anchor="middle">
    {AXES.get(cost, cost)} &mdash; lower is better</text>
</svg></div>
""", encoding="utf-8")
    return str(out)


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


def served(model: str, questions: Sequence[Mapping[str, Any]], graph: Mapping[str, Any], *,
           label: str = "", draft: str = "", port: int = 8099, context: int = 32768,
           parallel: int = 1, binary: str = "", kept: str | Path = "", shortlist: int = 0,
           store: str | Path | None = None, embed_url: str = "", embed_model: str = "",
           terse: bool = False, ways: Sequence[Mapping[str, Any]] = (),
           serve_timeout: float = 900.0,
           already: Callable[[str], Mapping[str, Any] | None] | None = None,
           **making: Any) -> list[Row]:
    """Put one model up, ask it the questions, take it down again.

    The piece that was missing. `sweep` measures servers somebody else started, so anything
    comparing several models meant hand-rolling starts, stops and waits -- which is a shell
    loop that dies with the terminal, and which was written twice before becoming this.

    One model at a time is not a limitation, it is the point: two servers sharing a GPU
    produce timings that belong to neither.

    ``ways`` asks the *same* server several times, which is most of the saving available
    here. Whether the tools are described briefly, and what sampling is used, are questions
    about the asking and not about the serving -- so measuring four of them costs one load
    and not four. Only a change the server itself must be told about, a draft head or a
    context, needs putting it up again. Each way is ``{"label": ..., "terse": ..., }`` plus
    anything a client takes.

    ``already(label)`` is the run a way is already kept as, when it is -- `sweep --resume`
    passes it -- and a way that has one is skipped before the model is loaded, so a sweep
    killed on its third model costs the third model to re-run and not the first two.
    """
    from ml_stack.client import Client
    from ml_stack.serve import serve

    name = label or str(model).rsplit("/", 1)[-1].removesuffix(".gguf")
    every = list(ways) or [{}]
    if already is not None:
        todo = []
        for way in every:
            tag = str(way.get("label", "") or "")
            here = f"{name}-{tag}" if tag else name
            kept_as = already(here)
            if kept_as:
                print(f"skipping {here}: kept at {kept_as.get('at', '?')}")
            else:
                todo.append(way)
        if not todo:
            return []
        every = todo
    extra: dict[str, Any] = {"parallel": parallel}
    if draft:
        from ml_stack.hub import spec_for

        extra["draft"] = draft
        kind = spec_for(draft)
        if kind:
            extra["spec_type"] = kind
    # Every question sends the same system prompt and the same tool schemas ahead of itself.
    # Reusing that prefix by KV shifting, rather than reprocessing it twenty times a run, is
    # free accuracy-wise: the tokens are identical, so the cache is valid.
    extra.setdefault("cache_reuse", 256)
    extra.setdefault("warmup", False)
    if binary:
        from ml_stack.serve.backend import LlamaServerBackend
        from ml_stack.serve.manager import ServerManager

        extra["manager"] = ServerManager(LlamaServerBackend(binary=binary))

    began = time.time()
    with serve(model, port=port, context=context, timeout=serve_timeout, **extra) as server:
        loaded = time.time() - began
        print(f"    up in {loaded:.0f}s, look_up by {finding(store, embed_url)}")
        rows = []
        for way in every:
            asked = dict(way)
            tag = str(asked.pop("label", "") or "")
            how = bool(asked.pop("terse", terse))
            here = f"{name}-{tag}" if tag else name
            if len(every) > 1:
                print(f"\n  --- {here}")
            wants_card = bool(asked.pop("_card", False))
            client = Client(server.base_url, **{**making, **asked})
            if wants_card:
                # what the model itself recommends, read from the GGUF it is serving
                client = Client(server.base_url, **{**making, **client.card})
            ask = asking(graph, shortlist=shortlist, store=store, embed_url=embed_url,
                         embed_model=embed_model, terse=how,
                         rich=bool(asked.pop("rich", False)))
            got = measure(ask, questions, label=here, client=client, log=print, graph=graph)
            for row in got:
                row.steps = f"{row.steps}; server up in {loaded:.0f}s".strip("; ")
            held = {**footprint(server.base_url), "graph": _which(graph),
                    "finder": getattr(ask, "finder", "")}
            if draft:
                held["draft_model"] = str(draft).rsplit("/", 1)[-1]
            if kept:
                save(kept, got, held={**held, "sampling": dict(client.sampling)})
            rows += got
    return rows


def drafts(model: str, heads: Sequence[str], questions: Sequence[Mapping[str, Any]],
           graph: Mapping[str, Any], *, port: int = 8099, context: int = 32768,
           parallel: int = 1, binary: str = "", kept: str | Path = "",
           store: str | Path | None = None, embed_url: str = "", embed_model: str = "",
           serve_timeout: float = 900.0, **making: Any) -> list[Row]:
    """Serve one model with each draft head in turn and measure what each is worth.

    A draft head only *proposes*; the large model verifies every token, so a quantised head
    cannot make an answer wrong -- it can only be right less often, and each wrong guess
    costs a verification pass. Whether the extra precision pays for its memory is therefore
    an empirical question and not an arguable one: it depends on this model, this workload,
    and how often the head happens to be right about it.

    Pass "" as a head to measure the model with no draft at all, which is the baseline
    every other row has to beat.
    """
    # The base model is loaded again for every head, because `-md` is bound when the server
    # starts and llama.cpp has no runtime swap: N configurations is N servers. It costs much
    # less than the first load -- the weights are mmapped and the pages are still cached --
    # but it is not free, so `served` times it and prints it rather than waving it away.
    out: list[Row] = []
    for head in heads:
        name = "none" if not head else str(head).rsplit("/", 1)[-1].removesuffix(".gguf")
        print(f"\n--- draft: {name}")
        out += served(model, questions, graph, label=f"draft:{name}", draft=head, port=port,
                      context=context, parallel=parallel, binary=binary, kept=kept,
                      store=store, embed_url=embed_url, embed_model=embed_model,
                      serve_timeout=serve_timeout, **making)
    return out


def drafting(rows: Sequence[Mapping[str, Any]]) -> str:
    """How much of what a draft model guessed was kept, as a percentage, or '-' for none.

    The count of guesses is not the interesting number and the wall clock already has the
    benefit in it. What this says is *why*: a draft accepted 76% of the time is earning its
    place, and one accepted 20% of the time is costing a pass to be told it was wrong.
    """
    guessed = _total(rows, "draft_tokens")
    return f"{100 * _total(rows, 'draft_taken') / guessed:.0f}%" if guessed else "-"


def ask_from(spec: str) -> Callable[[str, Any], Any]:
    """Import ``module:function``. It takes ``(question, client)`` and returns an Answer."""
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"expected module:function, got {spec!r}")
    from importlib import import_module

    return getattr(import_module(module), name)


def asking(graph: Mapping[str, Any], *, shortlist: int = 0, store: str | Path | None = None,
           embed_url: str = "", embed_model: str = "", terse: bool = False,
           margin: float = MARGIN, rich: bool = False) -> Callable[..., Any]:
    """The ordinary way to ask this graph a question, with or without a search run first.

    Nothing here is any project's: it is `converse` over the graph you handed in. Two
    choices, both about where the looking happens. Whether a cheap embedder gets to suggest
    where to look before the large model starts (``shortlist``), which is the thing most
    worth measuring. And what `look_up` is when the model calls it: with a ``store``, the
    same hybrid the application ships -- characters, the word index and, given
    ``embed_url``, vectors, fused -- and without one, characters alone. For months the bench
    had no store on this path and every ranking it wrote measured a look_up nobody ran.

    ``rich`` asks with `converse(..., rich=True)`: look_up results carry a score and why
    they matched, and a topic hit brings the people joined to it.

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
        extra: dict[str, Any] = {"tools": tools_for(graph, terse=True, finder=finder)} if terse else {}
        if rich:
            extra["rich"] = True
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
    run.add_argument("--kept", default=str(HOME / "runs.ladybug"),
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
    run.add_argument("--store", default=prepared(),
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
    heads.add_argument("--port", type=int, default=8099)
    heads.add_argument("--context", type=int, default=32768)
    heads.add_argument("--parallel", type=int, default=1)
    heads.add_argument("--binary", default="", help="a llama-server that reads this model")
    heads.add_argument("--kept", default=str(HOME / "runs.ladybug"))
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
    conc.add_argument("--kept", default=str(HOME / "runs.ladybug"),
                      help="where to keep the run (default: %(default)s)")
    conc.add_argument("--base-url", default="http://127.0.0.1:8080",
                      help="the model answering (default: %(default)s)")
    conc.add_argument("--graph", default="",
                      help="a graph as JSON (default: the invented community that ships here)")
    conc.add_argument("--questions", default="",
                      help="one per line, as for run (default: the invented community's)")
    conc.add_argument("--store", default=prepared(),
                      help="a graph store with the word index and vectors, so look_up "
                           "searches as the application does (default: what `prepare` "
                           "built, when it has been)")
    conc.add_argument("--embed-url", default="", help="a server that embeds, for the store")
    conc.add_argument("--embed-model", default="", help="the model that embedded the graph")
    conc.add_argument("--client", default="",
                      help="module:function returning the model client, instead of --base-url")

    ready = sub.add_parser("prepare", allow_abbrev=False,
                           help="put a graph in a store and index and embed it")
    ready.add_argument("--store", default=str(HOME / "graph.ladybug"),
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
    sweep.add_argument("--kept", default=str(HOME / "runs.ladybug"),
                       help="where to keep the runs (default: %(default)s)")
    sweep.add_argument("--graph", default="", help="a graph as JSON (default: the invented one)")
    sweep.add_argument("--questions", default="", help="(default: the ones that go with it)")
    sweep.add_argument("--shortlist", type=int, default=8, metavar="N",
                       help="how many to hand over in the second run (default: %(default)s)")
    sweep.add_argument("--store", default=prepared(),
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
    sweep.add_argument("--plain-only", action="store_true",
                       help="skip the shortlist half, just measure each model as it is")
    sweep.add_argument("--resume", action="store_true",
                       help="skip any model and way already kept since --since with this "
                            "many questions at this context and these slots, so a sweep "
                            "killed on its third model costs the third model and not all "
                            "three. Says which it skipped and when each was kept")
    sweep.add_argument("--since", default="", metavar="WHEN",
                       help="with --resume, how old a kept run may be and still count: an "
                            "ISO date or date-time (default: the start of today)")

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
                         choices=("terse", "card", "greedy", "rich"),
                         help="ask the same served model another way as well. Whether the "
                              "tools are described briefly, what sampling is used, and "
                              "whether look_up says why it matched (rich) are questions "
                              "about the asking, not the serving, so measuring four of them "
                              "costs one model load rather than four. Repeatable")

    for one in (run, sweep, heads, conc):
        one.add_argument("--no-queue", action="store_true",
                         help="fail at once if another measurement holds the GPU, rather "
                              "than queue behind it. Read before the rest of the line is "
                              "parsed, and listed here so that --help says it exists")
        one.add_argument("--detach", action="store_true",
                         help="run this in the background, owned by nobody's terminal: the "
                              "command re-runs itself in a new session with its output in "
                              f"a log under {HOME / 'logs'}, prints the log's path and "
                              "returns at once. `status` says what is measuring, `tail -f` "
                              "follows the log, `stop` ends it and takes its server down. "
                              "Read before the rest of the line is parsed, like --no-queue")

    show = sub.add_parser("show", allow_abbrev=False,
                          help="compare two runs, or list what is kept")
    show.add_argument("--kept", default=str(HOME / "runs.ladybug"),
                      help="the store the runs are in (default: %(default)s)")
    show.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), default=None)
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
                           "evidence: one line per model, its best run, and what that run "
                           "cost. This is the part worth keeping in a repository -- the raw "
                           "runs are not, since they describe one machine and one build")
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
    gone.add_argument("--kept", default=str(HOME / "runs.ladybug"),
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
    return ap


def _main(argv: list[str] | None = None) -> int:
    """``ml-stack-bench`` -- what a change to the asking costs, and whether it was worth it."""
    args = _parser().parse_args(argv)
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
                would = [r["key"] for r in runs(args.kept, args.label)]
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
        questions = sample(read_questions(args.questions) if args.questions else QUESTIONS,
                           _how_many(args))
        graph = (json.loads(Path(args.graph).expanduser().read_text())
                 if args.graph else invented())
        ways = [("plain", 0)] if args.plain_only else [("plain", 0), ("shortlist", args.shortlist)]
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
            model = find_model(wanted)
            heads = getattr(args, "serve_draft", []) or []
            head = heads[n] if n < len(heads) else ""
            if head.lower() == "auto":
                from ml_stack.serve.cli import drafted

                head = drafted(model, "auto")
            for suffix, shortlist in ways:
                stem = str(model).rsplit("/", 1)[-1].removesuffix(".gguf")[:14]
                print(f"\n{stem}-{suffix}")
                # A port nothing answers on is exactly what --serve expects, so the
                # "would not say whether it is busy" note is noise here. Only a port
                # somebody is actually using should stop us.
                if busy(f"http://127.0.0.1:{args.serve_port}") > 0 and not _idle(
                        f"http://127.0.0.1:{args.serve_port}", args):
                    return 3
                # `--context` is the total across slots, which is what `-c` takes and what
                # ServerSpec means by it. Dividing by the slot count served a model at a
                # quarter of the context every other run had, and the only thing that said
                # so was the `ctx` column reading 8k where the rest read 32k.
                before = {r["key"] for r in runs(args.kept)} if Path(args.kept).expanduser().exists() else set()
                served(model, questions, graph, label=f"{stem}-{suffix}", draft=head,
                       ways=_ways(args),
                       port=args.serve_port,
                       context=total_context,
                       parallel=getattr(args, "parallel", 1), binary=args.binary,
                       kept=args.kept, shortlist=shortlist,
                       store=args.store or None, embed_url=args.embed_url,
                       embed_model=args.embed_model, terse=getattr(args, "terse", False),
                       already=already, **sampling_from(args))
                saved += [r["key"] for r in runs(args.kept) if r["key"] not in before]

        for name, url in named:
            for suffix, shortlist in ways:
                label = f"{name}-{suffix}"
                if already is not None and already(label):
                    print(f"skipping {label}: kept at {already(label).get('at', '?')}")
                    continue
                ask = asking(graph, shortlist=shortlist, store=args.store or None,
                             embed_url=args.embed_url, embed_model=args.embed_model,
                             terse=getattr(args, "terse", False), margin=args.margin)
                print(f"\n{label} on {url}, look_up by {ask.finder}")
                if not _idle(url, args):
                    return 3
                asking_with = with_card(Client(url, **sampling_from(args)), args)
                # what it will actually send, card and overrides together: a run measured at
                # one temperature against a run at another is two measurements, and the only
                # way to know later is to write it down now
                used = dict(asking_with.sampling)
                rows = measure(ask, questions, label=label, client=asking_with, log=print,
                               graph=graph)
                saved.append(save(args.kept, rows,
                                  held={**footprint(url), "sampling": used,
                                        "graph": _which(graph), "finder": ask.finder}))
        print()
        table(read_back(args.kept, saved) if args.smoke else runs(args.kept))
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

        asked = sample(read_questions(args.questions) if args.questions else QUESTIONS,
                       SMOKE if getattr(args, "smoke", False) else args.sample)
        before = {r["key"] for r in runs(args.kept)} if Path(args.kept).expanduser().exists() else set()
        rows = drafts(find_model(args.model), args.draft or [""], asked, invented(),
                      port=args.port,
                      context=args.context, parallel=args.parallel, binary=args.binary,
                      kept=args.kept, store=prepared() or None)
        print()
        if getattr(args, "smoke", False):
            saved = [r["key"] for r in runs(args.kept) if r["key"] not in before]
            table(read_back(args.kept, saved))
        else:
            table(runs(args.kept))
        return 0 if rows else 1

    if args.cmd == "concurrent":
        from ml_stack.graph.community import QUESTIONS, graph as invented

        questions = sample(read_questions(args.questions) if args.questions else QUESTIONS,
                           _how_many(args))
        if not questions:
            print(f"error: no questions in {args.questions}", file=sys.stderr)
            return 2
        graph = (json.loads(Path(args.graph).expanduser().read_text())
                 if args.graph else invented())
        if args.client:
            client = ask_from(args.client)()
        else:
            from ml_stack.client import Client

            if not _idle(args.base_url, args):
                return 3
            client = with_card(Client(args.base_url, **sampling_from(args)), args)
        # a smoke run proves the path -- two conversations really overlapping, one turn
        # each -- and its numbers mean nothing, as with every other --smoke
        many, long = (2, 1) if args.smoke else (args.conversations, args.turns)
        ask = asking(graph, store=args.store or None, embed_url=args.embed_url,
                     embed_model=args.embed_model)
        where = args.graph or "the invented community"
        print(f"{args.label}: {many} conversations of {long} turn(s) at once over {where}, "
              f"look_up by {ask.finder}")
        rows, held = concurrent(ask, questions, conversations=many, turns=long,
                                label=args.label, client=client, graph=graph,
                                base_url="" if args.client else args.base_url, log=print)
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

    if args.cmd == "show":
        if args.compare:
            print(compare(args.kept, *args.compare))
            return 0
        if args.rank:
            ranking(runs(args.kept), args.rank)
            print(args.rank)
            return 0
        if args.export:
            print(export(runs(args.kept), args.export,
                         anyway=getattr(args, "export_anyway", False)))
            return 0
        if args.shape:
            from ml_stack.graph.community import QUESTIONS, graph as invented

            questions = read_questions(args.questions) if getattr(args, "questions", "") \
                else QUESTIONS
            shape(questions, invented())
            return 0
        if args.plot:
            print(plot(runs(args.kept), args.plot, cost=args.cost))
            return 0
        if args.rates:
            rates(runs(args.kept), cost=args.cost)
            return 0
        if args.detail is not None:
            missed(runs(args.kept, args.detail), everything=args.all)
            return 0
        table(runs(args.kept))
        hollow = empties(args.kept)
        if hollow:
            print(f"{len(hollow)} empty run(s) skipped -- ml-stack-bench forget --empty "
                  f"removes them")
        return 0

    from ml_stack.graph.community import QUESTIONS, graph as invented

    questions = sample(read_questions(args.questions) if args.questions else QUESTIONS,
                       _how_many(args))
    if not questions:
        print(f"error: no questions in {args.questions}", file=sys.stderr)
        return 2
    graph = json.loads(Path(args.graph).expanduser().read_text()) if args.graph else invented()
    if args.client:
        client = ask_from(args.client)()
    else:
        from ml_stack.client import Client

        if not _idle(args.base_url, args):
            return 3
        client = with_card(Client(args.base_url, **sampling_from(args)), args)
    ask = ask_from(args.ask) if args.ask else asking(
        graph, shortlist=args.shortlist, store=args.store or None,
        embed_url=args.embed_url, embed_model=args.embed_model, margin=args.margin)
    where = args.graph or "the invented community"
    found = getattr(ask, "finder", "")
    print(f"{args.label}: {len(questions)} questions over {where}"
          + (f", look_up by {found}" if found else "")
          + (f", {args.shortlist} handed to it first" if args.shortlist else ""))
    rows = measure(ask, questions, label=args.label, client=client, log=print, graph=graph)
    key = save(args.kept, rows,
               held={**footprint(args.base_url), "sampling": client.sampling,
                     "graph": _which(graph), "finder": found})
    print(f"kept as {key}")
    if args.smoke:
        table(read_back(args.kept, [key]))
    return 0


def read_back(store: str | Path, keys: Sequence[str]) -> list[dict[str, Any]]:
    """The runs under ``keys``, read from the store the way `show` reads them.

    What a smoke run exists to prove: the whole path, and the last step of the path is
    the store giving the run back. Summarising from memory proved everything but that,
    and a sweep passed its smoke and then kept twelve runs as nothing.
    """
    kept = {r["key"]: r for r in runs(store)}
    lost = [k for k in keys if k not in kept]
    if lost:
        raise RunNotKept(f"{len(lost)} run(s) saved to {store} did not come back: "
                         + ", ".join(lost))
    return [kept[k] for k in keys]


def resumable(store: str | Path, *, questions: int, context: int, parallel: int,
              since: str = "") -> Callable[[str], Mapping[str, Any] | None]:
    """``already(label)``: the run kept under ``label`` that makes measuring it again a
    waste, or None.

    A kept run counts when it asked this many questions at this context per slot and this
    many slots -- a run at another context is another measurement, as the `ctx` column
    says -- and is no older than ``since``, which defaults to the start of today. A run
    from last week is what the sweep was started to replace.
    """
    floor = since or time.strftime("%FT00:00:00")
    per_slot = int(context) // max(1, int(parallel))
    kept = runs(store) if Path(store).expanduser().exists() else []

    def already(label: str) -> Mapping[str, Any] | None:
        for one in reversed(kept):
            server = one.get("server") or {}
            if (one.get("label") == label
                    and str(one.get("at", "")) >= floor
                    and len(one.get("rows") or ()) == questions
                    and int(server.get("context") or 0) == per_slot
                    and int(server.get("slots") or 0) == parallel):
                return one
        return None

    return already


# Which subcommands put load on the GPU, and so must never overlap with each other.
MEASURING = ("run", "sweep", "drafts", "concurrent")

# Windows has no sessions; a child that survives its parent's console is asked for by flag.
_WINDOWS_DETACHED = 0x00000200 | 0x00000008     # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS


def measuring_file() -> Path:
    """Where the detached measurement's pid, argv, log and start time are written."""
    return HOME / "measuring.json"


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
    logs = HOME / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cmd = next((a for a in rest if a in MEASURING), "bench")
    log = logs / f"{cmd}-{_named_in(rest)}-{time.strftime('%Y%m%dT%H%M%S')}.log"
    command = [sys.executable, "-m", "ml_stack.graph.bench", *rest]
    extra: dict[str, Any] = ({"creationflags": _WINDOWS_DETACHED}
                             if platform.system() == "Windows" else {"start_new_session": True})
    with log.open("ab") as out:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out,
                                 stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1"}, **extra)
    measuring_file().write_text(json.dumps({
        "pid": child.pid, "argv": list(rest), "log": str(log),
        "started": time.strftime("%FT%T")}, indent=1), encoding="utf-8")
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
    logs = sorted((HOME / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime) \
        if (HOME / "logs").exists() else []
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
    """Turn SIGTERM into an exception, so every `with` on the way out runs its exit."""
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

    known = {*MEASURING, "show", "prepare", "forget", "status", "tail", "stop"}
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
    previous = None
    try:
        previous = signal.signal(signal.SIGTERM, _stop_on_sigterm)
    except ValueError:
        pass                                 # not the main thread: nothing to hand a signal
    try:
        with only_one(HOME / "measuring.lock", wait=not refuse,
                      announce=lambda line: print(line, file=sys.stderr)):
            return _main(rest)
    except Busy as why:
        print(f"error: {why}. Another measurement is running; wait for it, or pass "
              f"--no-queue to fail fast rather than queue.", file=sys.stderr)
        return 3
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
