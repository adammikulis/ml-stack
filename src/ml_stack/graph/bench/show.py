"""Printing what was measured: the table, the questions behind a score, the rates and the
frontier, the plot, and the `drafts` summary.

Everything here reads kept runs and writes to stdout or a file; the numbers it prints are
`score`'s. The columns are the lessons -- `ctx`, `n`, `conc` and `find` are each on every
line because a comparison across any of them is two measurements read as one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.runs` -- so
# anything patchable is looked up there at call time, never bound here at import.
from ml_stack.graph import bench
from ml_stack.graph.bench.score import (
    NOISE,
    _head_of,
    _hit,
    _precision,
    _recall,
    _times,
    _total,
    baseline,
    composed,
    derived,
    host_of,
    hosts_of,
    per_question,
    speedup,
    wall_of,
)


def kv_short(cache_type: str) -> str:
    """``q8_0`` as the table shows it beside the context: ``q8``. A trailing ``_0`` is the
    common case and says nothing; ``q5_1`` keeps its ``_1`` because ``q5_0`` also exists,
    and two cache types printed as one is exactly what the column is there to prevent."""
    return str(cache_type or "").removesuffix("_0")


def compare(store: str | Path, first: str, second: str) -> str:
    """The two labels side by side, and the difference between them."""
    sides = []
    for label in (first, second):
        kept = bench.runs(store, label)
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
    #
    # `ctx` carries the KV cache type when the run's server record names one (`32k x1/q8`):
    # a run with a quantised cache against one at f16 is two configurations, not two
    # models. `load` is what the server took to come up, from the lease itself and not a
    # stopwatch around it; blank for a run kept before the lease recorded one.
    #
    # `speed` is what the draft head was worth: the same model's newest undrafted run of
    # the same size on the same build, per question, over this run's -- `speedup` -- as
    # `1.42x`. Acceptance says why a head is earning its place; this says whether it is.
    # Blank for an undrafted run, or one whose baseline is not among the runs shown.
    # `host` only when more than one machine measured: a fleet's store holds runs from
    # several, and a column nobody needs is noise on a single one.
    # `pfx` is the prompt cache per turn: of the calls after a question's first, the share
    # that found the previous call's whole prompt still cached. `cached` is a total and
    # cannot see a change to the asking that breaks the prefix; blank for a run kept
    # before it was counted.
    several = len(hosts_of(kept)) > 1
    head = (f"{'run':28} " + (f"{'host':>10} " if several else "")
            + f"{'ctx':>10} {'n':>3} {'wall':>7} {'load':>5} {'calls':>6} {'read':>8} "
            f"{'written':>8} {'cached':>8} {'pfx':>4} {'draft':>6} {'speed':>6} {'find':>7} {'conc':>5} "
            f"{'resident':>9} {'kv+run':>8} {'per 1k':>8} {'F1':>5} {'rec':>5} {'prec':>5} "
            f"{'made':>5} {'t/o':>4}  {'sampling'}")
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
        kv = kv_short(str(server.get("cache_type") or ""))
        # `/rb`: served with a reasoning budget, which is another configuration again --
        # the label carries the number, this says the run's thinking was stopped
        budgeted = "/rb" if server.get("reasoning_budget") is not None else ""
        beyond = server.get("kv_and_run_bytes")
        per1k = server.get("bytes_per_1k_context")
        rss = server.get("resident_bytes")
        load = server.get("load_s")
        print(f"{_shown(one.get('label', '')):28} "
              + (f"{_shown(host_of(one) or '-', 10):>10} " if several else "")
              + f"{(f'{ctx // 1024}k x{slots}' + (f'/{kv}' if kv else '') + budgeted if ctx else '-'):>10} "
              f"{len(scored):>3} "
              f"{wall_of(one):>6.0f}s "
              f"{(f'{float(load):.0f}s' if load is not None else ''):>5} "
              f"{_total(rows, 'calls'):>6.0f} "
              f"{_total(rows, 'processed_tokens'):>8.0f} "
              f"{_total(rows, 'completion_tokens'):>8.0f} "
              f"{_total(rows, 'cached_tokens'):>8.0f} "
              f"{prefixed(server):>4} "
              f"{drafting(rows):>6} "
              f"{_times(speedup(one, kept)):>6} "
              f"{str(server.get('finder') or '-'):>7} "
              f"{at_once(server):>5} "
              f"{(f'{rss / 2**30:.2f}G' if rss else '-'):>9} "
              f"{(f'{beyond / 2**30:.2f}G' if beyond else ('mmap' if server.get('mmapped') else '-')):>8} "
              f"{(f'{per1k / 2**20:.1f}M' if per1k else '-'):>8} "
              f"{right:>5} {rec:>5} {prec:>5} {made(one):>5} {timeouts(one):>4}  "
              f"{sampled(server)}")


def timeouts(one: Mapping[str, Any]) -> str:
    """How many of a run's questions ran past `--per-question`; "" for none, so the eye
    goes to the runs that did. Each is scored wrong and wall-clocked at the cap."""
    n = sum(1 for r in (one.get("rows") or []) if r.get("timed_out"))
    return str(n) if n else ""


def prefixed(server: Mapping[str, Any]) -> str:
    """The run's prompt-cache share as ``75%``; "" for a run kept before it was counted,
    or with no turn to judge -- not counted is not 0%."""
    got = server.get("prefix_hits")
    return f"{100 * float(got):.0f}%" if got is not None else ""


def cache_turns(row: Mapping[str, Any]) -> str:
    """``cache 3/4 turns``: of a question's calls after the first, how many found the
    previous call's prompt still cached; "" for a row from before it was counted or a
    question of one call."""
    turns = int(row.get("prefix_turns") or 0)
    return f"cache {int(row.get('prefix_kept') or 0)}/{turns} turns" if turns else ""


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


def missed(kept: Sequence[Mapping[str, Any]], *, everything: bool = False,
           among: Sequence[Mapping[str, Any]] = ()) -> None:
    """Question by question: what was wanted, what the answer showed, and what it cost.

    A score is a number to act on only when you can see which questions made it. A run that
    scores 17% has failed in some particular way — no tool calls, an empty answer, the right
    people found and the wrong ones shown — and the aggregate cannot tell you which, so this
    prints the rows themselves. Only the misses by default; ``everything`` for all of them.

    ``among`` is every run kept, so a drafted run's line can say what its head was worth
    against the undrafted baseline `baseline` finds there; ``kept`` alone is one label's.
    """
    if not kept:
        print("nothing kept yet")
        return
    for one in kept:
        base = baseline(one, among or kept)
        faster = speedup(one, among or kept)
        rows = [r for r in (one.get("rows") or []) if r.get("expected")]
        shortfall = [r for r in rows if not everything and _hit(r) < 1.0] if not everything else rows
        server = one.get("server") or {}
        found = str(server.get("finder") or "-")
        together = at_once(server)
        load = server.get("load_s")
        late = timeouts(one)
        print(f"\n{one.get('label', '')}  ({one.get('at', '')}, find {found}"
              + (f", {together} at once" if together else "")
              + (f", load {float(load):.0f}s" if load is not None else "")
              + (f", {late} timed out" if late else "") + ")"
              + (f"  speedup {_times(faster)} over draft:none ({base.get('label', '')})"
                 if faster is not None and base is not None else ""))
        if not shortfall:
            print("  every question answered in full")
            continue
        for r in shortfall:
            got, want = set(r.get("shown") or ()), set(r.get("expected") or ())
            print(f"  {_hit(r) * 100:3.0f}%  {r.get('question', '')}"
                  + (f"   [timed out at {float(r.get('seconds') or 0):.0f}s]"
                     if r.get("timed_out") else ""))
            print(f"        wanted  {', '.join(sorted(want)) or '-'}")
            print(f"        showed  {', '.join(sorted(got)) or '(nothing)'}")
            if want - got:
                print(f"        missed  {', '.join(sorted(want - got))}")
            if r.get("unread"):
                # what F1 cannot see: a name in the prose that no tool call produced
                print(f"        made    {', '.join(r['unread'])}  (named, never found or read)")
            turns = cache_turns(r)
            note = (f"{r.get('calls', 0)} calls, {r.get('answer_chars', 0)} chars"
                    + (f", {turns}" if turns else "")
                    + (f", ERROR {r['error']}" if r.get("error") else ""))
            if together:
                note += (f"; conversation {r.get('conversation', 0)} turn {r.get('turn', 0)}, "
                         f"first token {r.get('first_token', 0):.1f}s, "
                         f"queued {r.get('queued', 0):.1f}s")
            print(f"        {note}")


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


def drafted(kept: Sequence[Mapping[str, Any]], *, among: Sequence[Mapping[str, Any]] = (),
            noise: float = NOISE) -> str:
    """The `drafts` summary: what each (head, n-max) was worth, and which to serve.

    One row per drafted run in ``kept`` -- acceptance, seconds per question, speedup over
    the baseline `baseline` finds in ``among`` (``kept`` itself when not given), F1 and how
    far it moved from the baseline's -- fastest first, the rows with no baseline last. Then
    the recommendation: the fastest whose F1 held within ``noise`` of its baseline, since a
    head cannot change an answer and one that did has changed something else. A head that
    fell outside the noise is on the table, not in the recommendation.
    """
    pool = list(among) or list(kept)
    rows = []
    for one in kept:
        if not _head_of(one):
            continue
        base = baseline(one, pool)
        mine = derived(one)
        rows.append((one, base, speedup(one, pool), mine,
                     derived(base) if base is not None else {}))
    if not rows:
        return "no drafted run to summarise"
    rows.sort(key=lambda r: -(r[2] if r[2] is not None else -1.0))
    head = (f"{'draft':28} {'accept':>6} {'s/q':>6} {'speed':>6} {'F1':>5} {'dF1':>5}  "
            f"against")
    lines = [head, "-" * len(head)]
    for one, base, faster, mine, theirs in rows:
        got = mine.get("right")
        delta = ((got - theirs["right"]) * 100
                 if got is not None and theirs.get("right") is not None else None)
        lines.append(
            f"{_shown(one.get('label', '')):28} "
            f"{drafting(one.get('rows') or []):>6} "
            f"{per_question(one):>6.1f} "
            f"{_times(faster):>6} "
            f"{(f'{got * 100:.0f}%' if got is not None else '-'):>5} "
            f"{(f'{delta:+.0f}' if delta is not None else '-'):>5}  "
            f"{base.get('label', '') if base is not None else 'no baseline'}")
    held = [(one, faster) for one, base, faster, mine, theirs in rows
            if faster is not None and theirs.get("right") is not None
            and theirs["right"] - mine.get("right", 0.0) <= noise + 1e-9]
    # a head that held its F1 but is slower than no head is worth nothing: gpt-oss's eagle3
    # head accepted 65% and still ran at 0.82x, and was recommended (2026-09-02)
    won = [pair for pair in held if pair[1] > 1.0]
    pts = noise * 100
    if won:
        best, faster = max(won, key=lambda pair: pair[1])
        lines.append(f"serve {best.get('label', '')}: fastest whose F1 held within "
                     f"{pts:g} points of its baseline, {_times(faster)}")
    elif held:
        best, faster = max(held, key=lambda pair: pair[1])
        lines.append(f"serve no head: the best that held its F1, {best.get('label', '')}, "
                     f"is slower than none at {_times(faster)}")
    elif any(faster is not None for _, _, faster, _, _ in rows):
        lines.append(f"serve no head: none held its baseline's F1 within {pts:g} points")
    else:
        lines.append("no baseline to recommend against: measure draft:none too "
                     "(pass \"\" as a head)")
    return "\n".join(lines)


def rates(kept: Sequence[Mapping[str, Any]], *, cost: str = "seconds",
          noise: float = NOISE) -> None:
    """Every run by what it cost to be right, frontier marked -- and each model once more
    as `composed` composes it, marked ``=``, so the frontier holds what a model can do at
    the speed of the configuration that held its accuracy."""
    if not kept:
        print("nothing kept yet")
        return
    points = list(kept) + composed(kept, noise=noise)
    on_front = {id(one) for one in pareto(points, cost=cost)}
    several = len(hosts_of(points)) > 1
    head = (f"{'run':28} {'n':>3} {'F1':>5} {'rec':>5} {'prec':>5} {'lit/q':>6} "
            f"{'F1/min':>8} {'F1/1k tok':>10} {'F1/GB':>7} {'s per':>7} {'tok per':>8}")
    print(head)
    print("-" * len(head))
    for one in sorted(points, key=lambda o: -(derived(o).get("right") or 0)):
        d = derived(one)
        if not d:
            continue
        def num(key: str, fmt: str) -> str:
            return format(d[key], fmt) if key in d else "-"
        mark = ("*" if id(one) in on_front else " ") + ("=" if one.get("composed") else " ")
        # by host when several measured: the frontier is one clock, and a point from
        # another machine is on it only by name
        named = (_shown(f"{one.get('label', '')}@{host_of(one) or '?'}", 18) if several
                 else str(one.get("label", ""))[:18])
        print(f"{named:18}{mark} {d['questions']:>3.0f} "
              f"{100 * d['right']:>4.0f}% {100 * d['recall']:>4.0f}% "
              f"{100 * d['precision']:>4.0f}% {d['shown_per_question']:>6.1f} "
              f"{num('right_per_minute', '8.2f')} {num('right_per_1k', '10.4f')} "
              f"{num('right_per_gb', '7.3f')} {num('seconds_per_right', '7.1f')} "
              f"{num('tokens_per_right', '8.0f')}")
    print(f"\n* on the frontier for accuracy against {cost}: nothing is both more accurate "
          f"and cheaper.")
    print(f"= a model composed: accuracy from its largest run, cost from its fastest run "
          f"within {noise * 100:g} points of it, scaled to the same number of questions.")


AXES = {"seconds": "wall clock (s)", "paid_tokens": "tokens paid for (read + written)",
        "kv_bytes": "KV cache and runtime (GB)"}


def plot(kept: Sequence[Mapping[str, Any]], where: str | Path, *,
         cost: str = "seconds", noise: float = NOISE) -> str:
    """Write accuracy against cost as a self-contained HTML scatter, frontier joined.

    Plain SVG built here rather than a plotting library: this has to open on a machine with
    no network and no packages, and a chart of a dozen points is a dozen circles. The
    frontier is drawn as a line through the runs nothing beats on both axes, so the shape of
    the trade is visible rather than inferred from a column of numbers. Each model's
    `composed` point is drawn as a ring beside its runs.
    """
    points = [(one, derived(one)) for one in list(kept) + composed(kept, noise=noise)]
    points = [(one, d) for one, d in points if d and cost in d and d[cost] > 0]
    if not points:
        raise ValueError("nothing to plot")
    front = {id(one) for one in pareto([one for one, _ in points], cost=cost)}
    several = len(hosts_of([one for one, _ in points])) > 1

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
        label = _shown(f"{one.get('label', '')}@{host_of(one) or '?'}" if several
                       else one.get("label", ""), 28)
        kind = "composed" if one.get("composed") else ("front" if on else "dot")
        note = (f'\ncomposed: cost from {one.get("from") or "its own run"}'
                if one.get("composed") else "")
        if several:
            note += f'\nhost: {host_of(one) or "?"}'
        dots.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{6 if on else 4.5}" '
            f'class="{kind}{" front" if on and kind == "composed" else ""}"><title>{label}\n'
            f'{100 * d["right"]:.0f}% right, {d[cost]:.0f} {cost}\n'
            f'{d["questions"]:.0f} questions{note}</title></circle>')
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
  .composed {{ fill: none; stroke: var(--ink); stroke-width: 2; }}
  .composed.front {{ stroke: var(--front); }}
  .edge {{ fill: none; stroke: var(--front); stroke-width: 2; stroke-dasharray: 5 4; }}
  .tag  {{ fill: var(--ink); font-size: 11px; }}
</style>
<h1>Answering the graph: accuracy against {AXES.get(cost, cost)}</h1>
<p>Each point is one benchmark run; a ring is one model composed &mdash; accuracy from its
largest run, cost from its fastest run that held that accuracy. Green points are the Pareto
frontier &mdash; nothing is both more accurate and cheaper &mdash; so choosing among them is
choosing a budget, not choosing a better run. Hover a point for its numbers.</p>
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


def drafting(rows: Sequence[Mapping[str, Any]]) -> str:
    """How much of what a draft model guessed was kept, as a percentage, or '-' for none.

    The count of guesses is not the interesting number and the wall clock already has the
    benefit in it. What this says is *why*: a draft accepted 76% of the time is earning its
    place, and one accepted 20% of the time is costing a pass to be told it was wrong.
    """
    guessed = _total(rows, "draft_tokens")
    return f"{100 * _total(rows, 'draft_taken') / guessed:.0f}%" if guessed else "-"
