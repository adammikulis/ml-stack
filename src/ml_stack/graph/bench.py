"""What a change to the asking costs and whether it was worth it.

A graph answers questions through a large model, and every tool call it makes is a whole
round trip. Any change to that — a different prompt, a search run before the model instead of
by it — has to be shown to be an improvement rather than asserted, on wall clock, on tokens,
and on whether the answers were right. Runs are kept, so two of them can be compared later.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.graph.vectors import MARGIN, stands_out

__all__ = ["Counting", "HOME", "Row", "SHORT", "ask_from", "asking", "compare", "footprint",
           "main", "measure", "read_questions", "runs", "save", "table"]

# Runs are worth keeping: the point of one is to compare it with another, later, and a
# benchmark written to a temporary directory answers no question a week from now.
HOME = Path("~/.ml-stack/bench").expanduser()

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

    def chat(self, messages: Any, **kw: Any) -> Any:
        self.calls += 1
        reply = self.client.chat(messages, **kw)
        raw = getattr(reply, "raw", None) or {}
        usage = raw.get("usage") or {}
        timings = raw.get("timings") or {}
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


def _how_many(args: Any) -> int:
    """How many questions to ask: --sample wins, then --short, then all of them."""
    asked = int(getattr(args, "sample", 0) or 0)
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


def measure(ask: Callable[[str, Any], Any], questions: Sequence[dict[str, Any]], *,
            label: str, client: Any, log: Callable[[str], None] | None = None) -> list[Row]:
    """Ask each question once through ``ask(question, client)`` and record what it cost."""
    rows = []
    for one in questions:
        counting = Counting(client)
        row = Row(label=label, question=str(one.get("q") or ""),
                  expected=[str(i) for i in (one.get("expect") or ())])
        began = time.time()
        try:
            out = ask(row.question, counting)
            # an Answer, or the payload a project sends its own page; both say the same things
            read = out.get if isinstance(out, Mapping) else lambda k, d=None: getattr(out, k, d)
            row.steps = read("why", "") or ""
            row.answer_chars = len(read("content", "") or "")
            row.shown = list(read("show", None) or read("ids", None) or [])
        except Exception as exc:  # noqa: BLE001 - a failure is a result, not the end of the run
            row.error = f"{type(exc).__name__}: {exc}"[:200]
        row.seconds = round(time.time() - began, 2)
        row.calls = counting.calls
        row.prompt_tokens = counting.prompt_tokens
        row.cached_tokens = counting.cached_tokens
        row.processed_tokens = counting.processed_tokens
        row.completion_tokens = counting.completion_tokens
        row.draft_tokens = counting.draft_tokens
        row.draft_taken = counting.draft_taken
        rows.append(row)
        if log:
            log(f"  {row.seconds:5.1f}s {row.calls:3} calls  {row.question[:56]}")
    return rows


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
    if "resident_bytes" in out and "weights_bytes" in out:
        beyond = out["resident_bytes"] - out["weights_bytes"]
        if beyond > 0:
            out["kv_and_run_bytes"] = beyond
        else:
            out["mmapped"] = True
        # What one more conversation costs, which is the question a number like this is
        # asked for. Held tokens are the context times the slots holding one each; dividing
        # by them makes two models comparable however each happened to be configured.
        held = (out.get("context") or 0) * (out.get("slots") or 0)
        if held:
            out["bytes_per_1k_context"] = int(out["kv_and_run_bytes"] / (held / 1024))
    return out


def save(store: str | Path, rows: Sequence[Row], *, held: dict[str, Any] | None = None) -> str:
    """Keep a run where it can be compared with another one, later, by anybody."""
    from ml_stack.graph.store import GraphStore

    server = held
    stem = f"bench:{rows[0].label}:{time.strftime('%Y%m%dT%H%M%S')}" if rows else "bench:empty"
    with GraphStore(store) as held:
        # Two runs of one label inside a second used to land on the same key and the later
        # one silently replaced the earlier. A run took minutes when that was written; with
        # answers cached it can take no time at all, so the collision is real now.
        key, n = stem, 1
        while held.get_doc(key) is not None:
            key, n = f"{stem}-{n}", n + 1
        held.put_doc(key, {"at": time.strftime("%FT%T"), "label": rows[0].label if rows else "",
                           "server": server or {}, "rows": [asdict(r) for r in rows]})
    return key


def runs(store: str | Path, label: str = "") -> list[dict[str, Any]]:
    """Every run kept in ``store``, newest last, optionally only one label's."""
    from ml_stack.graph.store import GraphStore

    with GraphStore(store, read_only=True) as held:
        kept = held.docs()
    found = [kept[k] for k in sorted(kept) if k.startswith("bench:")]
    return [r for r in found if isinstance(r, dict) and (not label or r.get("label") == label)]


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
    head = (f"{'run':20} {'ctx':>7} {'n':>3} {'wall':>7} {'calls':>6} {'read':>8} "
            f"{'written':>8} {'cached':>8} {'draft':>6} {'resident':>9} {'kv+run':>8} "
            f"{'per 1k':>8} {'F1':>5} {'rec':>5} {'prec':>5}  {'sampling'}")
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
        print(f"{str(one.get('label', ''))[:20]:20} "
              f"{(f'{ctx // 1024}k x{slots}' if ctx else '-'):>7} "
              f"{len(scored):>3} "
              f"{_total(rows, 'seconds'):>6.0f}s {_total(rows, 'calls'):>6.0f} "
              f"{_total(rows, 'processed_tokens'):>8.0f} "
              f"{_total(rows, 'completion_tokens'):>8.0f} "
              f"{_total(rows, 'cached_tokens'):>8.0f} "
              f"{drafting(rows):>6} "
              f"{(f'{rss / 2**30:.2f}G' if rss else '-'):>9} "
              f"{(f'{beyond / 2**30:.2f}G' if beyond else ('mmap' if server.get('mmapped') else '-')):>8} "
              f"{(f'{per1k / 2**20:.1f}M' if per1k else '-'):>8} "
              f"{right:>5} {rec:>5} {prec:>5}  {sampled(server)}")


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
        print(f"\n{one.get('label', '')}  ({one.get('at', '')})")
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
            note = (f"{r.get('calls', 0)} calls, {r.get('answer_chars', 0)} chars"
                    + (f", ERROR {r['error']}" if r.get("error") else ""))
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
    seconds = _total(rows, "seconds")
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


def rates(kept: Sequence[Mapping[str, Any]], *, cost: str = "seconds") -> None:
    """Every run by what it cost to be right, frontier marked."""
    if not kept:
        print("nothing kept yet")
        return
    on_front = {id(one) for one in pareto(kept, cost=cost)}
    head = (f"{'run':20} {'n':>3} {'F1':>5} {'rec':>5} {'prec':>5} {'lit/q':>6} "
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
        label = str(one.get("label", ""))[:22]
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
           terse: bool = False,
           serve_timeout: float = 900.0, **making: Any) -> list[Row]:
    """Put one model up, ask it the questions, take it down again.

    The piece that was missing. `sweep` measures servers somebody else started, so anything
    comparing several models meant hand-rolling starts, stops and waits -- which is a shell
    loop that dies with the terminal, and which was written twice before becoming this.

    One model at a time is not a limitation, it is the point: two servers sharing a GPU
    produce timings that belong to neither.
    """
    from ml_stack.client import Client
    from ml_stack.serve import serve

    name = label or str(model).rsplit("/", 1)[-1].removesuffix(".gguf")
    extra: dict[str, Any] = {"parallel": parallel}
    if draft:
        from ml_stack.hub import spec_for

        extra["draft"] = draft
        kind = spec_for(draft)
        if kind:
            extra["spec_type"] = kind
    if binary:
        from ml_stack.serve.backend import LlamaServerBackend
        from ml_stack.serve.manager import ServerManager

        extra["manager"] = ServerManager(LlamaServerBackend(binary=binary))

    began = time.time()
    with serve(model, port=port, context=context, timeout=serve_timeout, **extra) as server:
        loaded = time.time() - began
        print(f"    up in {loaded:.0f}s")
        ask = asking(graph, shortlist=shortlist, store=store, embed_url=embed_url,
                     embed_model=embed_model, terse=terse)
        rows = measure(ask, questions, label=name,
                       client=Client(server.base_url, **making), log=print)
        for row in rows:
            row.steps = f"{row.steps}; server up in {loaded:.0f}s".strip("; ")
        held = {**footprint(server.base_url)}
        if draft:
            held["draft_model"] = str(draft).rsplit("/", 1)[-1]
    if kept:
        save(kept, rows, held=held)
    return rows


def drafts(model: str, heads: Sequence[str], questions: Sequence[Mapping[str, Any]],
           graph: Mapping[str, Any], *, port: int = 8099, context: int = 32768,
           parallel: int = 1, binary: str = "", kept: str | Path = "",
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
           margin: float = MARGIN) -> Callable[[str, Any], Any]:
    """The ordinary way to ask this graph a question, with or without a search run first.

    Nothing here is any project's: it is `converse` over the graph you handed in. The only
    choice is whether a cheap embedder gets to suggest where to look before the large model
    starts, which is the thing most worth measuring.
    """
    from ml_stack.graph.ask import converse
    from ml_stack.graph.search import hybrid

    def likely(question: str) -> list[str]:
        if not shortlist or store is None:
            return []
        from ml_stack.graph.store import GraphStore

        vector = None
        if embed_url:
            from ml_stack.client.embed import embed

            from ml_stack.graph.vectors import QUERY
            try:
                vector = embed([QUERY + question], base_url=embed_url, model=embed_model)[0]
            except Exception:  # noqa: BLE001 - the words still vote
                vector = None
        with GraphStore(store, read_only=True) as held:
            if vector is not None and margin > 0:
                near = held.similar(vector, model=embed_model, limit=max(shortlist, 8))
                if not stands_out([r["similarity"] for r in near], margin=margin):
                    return []            # nothing here stands out: "hi" is not a search
            found = hybrid(graph, question, store=held, vector=vector, model=embed_model)
        return [r["id"] for r in found][:shortlist]

    def ask(question: str, client: Any) -> Any:
        from ml_stack.graph.ask import tools_for

        extra = {"tools": tools_for(graph, terse=True)} if terse else {}
        return converse(question, graph, client, opening=likely(question), **extra)

    return ask


def main(argv: list[str] | None = None) -> int:
    """``ml-stack-bench`` -- what a change to the asking costs, and whether it was worth it."""
    ap = argparse.ArgumentParser(
        prog="ml-stack-bench",
        description="Time a set of questions through a graph, and compare two runs.")
    sub = ap.add_subparsers(dest="cmd", required=True,
                            metavar="{prepare,run,sweep,show}")

    run = sub.add_parser("run", help="ask every question once and keep what it cost")
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
    run.add_argument("--store", default="",
                     help="a graph store with the word index and vectors, for --shortlist")
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

    heads = sub.add_parser("drafts", help="serve one model with each draft head in turn "
                                          "and measure what each is worth")
    heads.add_argument("model", help="the model to serve, a path or an hf: reference")
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

    ready = sub.add_parser("prepare", help="put a graph in a store and index and embed it")
    ready.add_argument("--store", default=str(HOME / "graph.ladybug"),
                       help="the store to build (default: %(default)s)")
    ready.add_argument("--graph", default="",
                       help="a graph as JSON (default: the invented community)")
    ready.add_argument("--embed-url", default="",
                       help="a server that embeds; without one only the word index is built")
    ready.add_argument("--embed-model", default="", help="what to file the vectors under")

    sweep = sub.add_parser("sweep", help="run every model, with and without a shortlist")
    sweep.add_argument("--on", action="append", metavar="NAME=URL", default=[],
                       help="a model to measure, e.g. e4b=http://127.0.0.1:8083; repeatable")
    sweep.add_argument("--kept", default=str(HOME / "runs.ladybug"),
                       help="where to keep the runs (default: %(default)s)")
    sweep.add_argument("--graph", default="", help="a graph as JSON (default: the invented one)")
    sweep.add_argument("--questions", default="", help="(default: the ones that go with it)")
    sweep.add_argument("--shortlist", type=int, default=8, metavar="N",
                       help="how many to hand over in the second run (default: %(default)s)")
    sweep.add_argument("--store", default=str(HOME / "graph.ladybug"),
                       help="the indexed and embedded graph, for the shortlist")
    sweep.add_argument("--embed-url", default="", help="a server that embeds, for --shortlist")
    sweep.add_argument("--embed-model", default="", help="the model that embedded the graph")
    sweep.add_argument("--margin", type=float, default=MARGIN)
    sweep.add_argument("--serve", action="append", default=[], metavar="MODEL",
                       help="a model to put up, measure and take down again, one at a time. "
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

    for one in (run, sweep):
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

    show = sub.add_parser("show", help="compare two runs, or list what is kept")
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
    show.add_argument("--plot", default="", metavar="FILE.html",
                      help="write the runs as a scatter of accuracy against --cost, with "
                           "the frontier joined; opens with no network and no packages")
    show.add_argument("--cost", default="seconds",
                      choices=("seconds", "paid_tokens", "kv_bytes"),
                      help="which cost the frontier is drawn against (default: %(default)s)")

    args = ap.parse_args(argv)
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
        for n, model in enumerate(getattr(args, "serve", []) or []):
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
                served(model, questions, graph, label=f"{stem}-{suffix}", draft=head,
                       port=args.serve_port, context=args.context // max(1, args.parallel)
                       if getattr(args, "context", 0) else 32768,
                       parallel=getattr(args, "parallel", 1), binary=args.binary,
                       kept=args.kept, shortlist=shortlist,
                       store=args.store or None, embed_url=args.embed_url,
                       embed_model=args.embed_model, terse=getattr(args, "terse", False),
                       **sampling_from(args))

        for name, url in named:
            for suffix, shortlist in ways:
                label = f"{name}-{suffix}"
                ask = asking(graph, shortlist=shortlist, store=args.store or None,
                             embed_url=args.embed_url, embed_model=args.embed_model,
                             terse=getattr(args, "terse", False), margin=args.margin)
                print(f"\n{label} on {url}")
                if not _idle(url, args):
                    return 3
                asking_with = with_card(Client(url, **sampling_from(args)), args)
                # what it will actually send, card and overrides together: a run measured at
                # one temperature against a run at another is two measurements, and the only
                # way to know later is to write it down now
                used = dict(asking_with.sampling)
                rows = measure(ask, questions, label=label, client=asking_with, log=print)
                save(args.kept, rows,
                     held={**footprint(url), "sampling": used})
        print()
        table(runs(args.kept))
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
                       args.sample)
        rows = drafts(args.model, args.draft or [""], asked, invented(), port=args.port,
                      context=args.context, parallel=args.parallel, binary=args.binary,
                      kept=args.kept)
        print()
        table(runs(args.kept))
        return 0 if rows else 1

    if args.cmd == "show":
        if args.compare:
            print(compare(args.kept, *args.compare))
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
    print(f"{args.label}: {len(questions)} questions over {where}"
          + (f", {args.shortlist} handed to it first" if args.shortlist else ""))
    rows = measure(ask, questions, label=args.label, client=client, log=print)
    key = save(args.kept, rows,
               held={**footprint(args.base_url), "sampling": client.sampling})
    print(f"kept as {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
