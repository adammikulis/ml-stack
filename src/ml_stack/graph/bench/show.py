"""Printing what was measured: the table, the questions behind a score, the rates and the
frontier, the plot, and the `drafts` summary.

Everything here reads kept runs and writes to stdout or a file; the numbers it prints are
`score`'s. The columns are the lessons -- `ctx`, `n`, `conc` and `find` are each on every
line because a comparison across any of them is two measurements read as one.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.runs` -- so
# anything patchable is looked up there at call time, never bound here at import.
from ml_stack.graph import bench
from ml_stack.graph.bench.backends import short
from ml_stack.graph.bench.score import (
    COSTS,
    NOISE,
    _head_of,
    _hit,
    _precision,
    _recall,
    _times,
    _total,
    band,
    baseline,
    composed,
    derived,
    half_band,
    held_up,
    host_of,
    hosts_of,
    per_question,
    separated,
    speedup,
    wall_of,
)


# The kinds of run the answering table leaves out: each has a table of its own.
NOT_ANSWERING = ("extract", "speed")


def kv_short(cache_type: str) -> str:
    """``q8_0`` as the table shows it beside the context: ``q8``. A trailing ``_0`` is the
    common case and says nothing; ``q5_1`` keeps its ``_1`` because ``q5_0`` also exists,
    and two cache types printed as one is exactly what the column is there to prevent."""
    return str(cache_type or "").removesuffix("_0")


def _flag(args: Sequence[Any], *names: str) -> str:
    """The value ``--ubatch 2048`` was given as, read out of a run's ``extra_args``, or "".

    A server flag that no field of the record names still decided the measurement, and
    `ServerSpec.extra_args` is where those arrive. Both spellings are read --
    ``--ubatch 2048`` and ``--ubatch=2048`` -- because both are what somebody typed.
    """
    words = [str(a) for a in args or ()]
    for n, word in enumerate(words):
        for name in names:
            if word == name and n + 1 < len(words):
                return words[n + 1]
            if word.startswith(f"{name}="):
                return word.split("=", 1)[1]
    return ""


def _num(value: Any) -> str:
    """A number as a shape says it: ``2048``, ``.5``, ``1.2`` -- no trailing zeros, and no
    leading one either, since a shape is read at a glance and ``0.5`` costs a character."""
    try:
        got = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{got:g}"
    return text[1:] if text.startswith("0.") else text


def head_short(name: Any) -> str:
    """A draft head as a shape names it: ``mtp``, ``eagle3``, ``ngram``, else its own stem.

    The file is `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` and what tells one run from another is
    the *kind* of head; which model it drafts for is the run's model, already on the line.
    """
    plain = str(name or "").rsplit("/", 1)[-1].removesuffix(".gguf")
    low = plain.lower()
    for known in ("eagle3", "eagle", "mtp", "ngram"):
        if known in low:
            return known
    return plain[:10]


def shape_of(one: Mapping[str, Any], *, ctx_shown: bool = False) -> str:
    """How a run was *served*, as one field: ``q8/rb0/ub2048/pmin.5/mtp@4``.

    Every one of these decided the measurement and none of them had a column. They lived in
    the end of the label instead -- `Qwen3.8-Flash--v2-plain-batch-kv-q8_0-rb0` -- where the
    micro-batch, the speculative p-min, the projector and the lock were not written down at
    all, so two runs that differed in one of them read as a repeat. A label is typed by a
    person; a record is written by the code that set the flag.

    In the order a person changes them: the KV cache type, the reasoning budget, the
    micro-batch, the draft's p-min, the head and how far ahead it guesses, then the
    projector and the lock. ``-`` for a run kept before any of it was recorded, and a field
    the record does not carry is left out rather than guessed at.

    ``ctx_shown`` leaves out the cache type and the reasoning budget, for `table`, whose
    ``ctx`` column already prints both -- ``32k x1/q8/rb``. Dropping what is on the line
    twice is what keeps the field narrow enough to print whole; cutting the front off
    instead lost the micro-batch and left ``2048/pmin.5`` reading as nothing.
    """
    server = one.get("server") or {}
    args = server.get("extra_args") or ()
    bits = []
    kv = kv_short(str(server.get("cache_type") or server.get("cache_type_k") or ""))
    if kv and not ctx_shown:
        bits.append(kv)
    if server.get("reasoning_budget") is not None and not ctx_shown:
        bits.append(f"rb{int(server['reasoning_budget'])}")
    micro = (server.get("n_ubatch") or server.get("ubatch")
             or _flag(args, "-ub", "--ubatch", "--ubatch-size"))
    if micro:
        bits.append(f"ub{_num(micro)}")
    pmin = server.get("spec_p_min", server.get("draft_p_min"))
    if pmin is None:
        pmin = _flag(args, "--draft-p-min", "--spec-p-min", "-spec-p-min") or None
    if pmin is not None:
        bits.append(f"pmin{_num(pmin)}")
    head = head_short(server.get("draft_model") or server.get("draft") or "")
    if head:
        ahead = server.get("spec_draft_max")
        bits.append(f"{head}@{int(ahead)}" if ahead is not None else head)
    elif server.get("spec_draft_max") is not None:
        bits.append(f"n{int(server['spec_draft_max'])}")
    if server.get("mmproj"):
        bits.append("mmproj")
    if server.get("mlock"):
        bits.append("mlock")
    return "/".join(bits) or "-"


# The asking, in the order it is thought about: what the tools looked like, then what the
# model was allowed to do with them. `tight` is the default asking now, so it says nothing
# and its absence -- `loose` -- says everything.
# (`report` has a `WAYS` of its own, the words a *label* can carry; this is the record's
# flags in the order a shape names them, and the two must not share a name in `bench`.)
SHOWN_WAYS = (("terse", "terse"), ("rich", "rich"), ("batch", "batch"),
              ("single", "single"), ("few", "few"),
              ("kinds", "kinds"), ("summary", "summary"), ("constrain_ids", "ids"))


def asked_as(one: Mapping[str, Any]) -> str:
    """How a run *asked*, as one field: ``+batch+kinds``, ``tight``, ``+loose+reach8k``.

    Read from the run's `asking` record -- the keywords `measure.asking` handed `converse`
    -- and never from the label, which is where these used to live and where a suffix said
    `batch` because somebody typed it rather than because anything did. ``-`` for a run
    kept before the record existed: not recorded is not "asked plainly".

    Only what differs from the plain asking is named, so a way shows what a person chose.
    A run that chose nothing reads ``tight``, which is the plain asking said out loud, and
    is how a plain run is told from one that kept no record at all.
    """
    asking = one.get("asking")
    if not isinstance(asking, Mapping) or not asking:
        return "-"
    bits = []
    if not asking.get("tight", True):
        bits.append("+loose")
    for key, name in SHOWN_WAYS:
        if asking.get(key):
            bits.append(f"+{name}")
    reach = int(asking.get("reach") or 0)
    if reach:
        bits.append(f"+reach{reach // 1000}k" if reach % 1000 == 0 else f"+reach{reach}")
    if asking.get("rounds"):
        # how many tool-calling turns a question had: `few` and `single` both want more of
        # them than `batch` does, so two runs that differ only here are two measurements
        bits.append(f"+rounds{int(asking['rounds'])}")
    if asking.get("shortlist"):
        bits.append(f"+list{int(asking['shortlist'])}")
    return "".join(bits) or "tight"


def shaped(one: Mapping[str, Any], *, ctx_shown: bool = False) -> str:
    """The whole configuration as one field: what was served, then how it was asked --
    ``q8/rb0/mtp@4+batch+kinds``. What `by_shape` groups on, sampling beside it.

    One word, with no space in it, because every column of the table is read by eye and by
    ``split()`` -- the tests count words along a line, and a field that is sometimes one
    word and sometimes two makes every column after it move. The asking's parts already
    begin with ``+``, so the join needs no separator; a run that kept no record of its
    asking carries the served shape alone, and one that recorded the plain asking says
    ``+tight`` rather than nothing, since "asked plainly" and "did not say" are not the
    same fact. ``ctx_shown`` is `shape_of`'s: leave out what the ``ctx`` column says.
    """
    form, way = shape_of(one, ctx_shown=ctx_shown), asked_as(one)
    if way == "-":
        return form
    return f"{'' if form == '-' else form}{'+tight' if way == 'tight' else way}" or "-"


def band_of(one: Mapping[str, Any], key: str = "right", *, unit: str = "pts") -> str:
    """``±6`` for a run's 95% interval on ``key``, "" for a run that carries none.

    Points for a score, seconds for a clock: half the interval in whatever the mean is in,
    so ``F1 70% ±6`` and ``10.4 s/q ±1.2`` each read without a conversion.
    """
    half = half_band(one, key)
    if half is None:
        return ""
    return f"±{half * 100:.0f}" if unit == "pts" else f"±{half:.1f}"


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

    def row(name: str, a: float | None, b: float | None, unit: str = "",
            better_lower: bool = True) -> str:
        # not measured is not 0: a side that reported nothing gets no percentage
        if a is None or b is None:
            missing = [label for label, value in ((first, a), (second, b)) if value is None]
            return (f"{name:22} {('-' if a is None else f'{a:.1f}{unit}'):>16} "
                    f"{('-' if b is None else f'{b:.1f}{unit}'):>16}"
                    f"  not measured on {', '.join(missing)}")
        gap = b - a
        way = "" if not a else f"  {gap / a * +100:+.0f}%"
        return f"{name:22} {a:>16.1f}{unit} {b:>16.1f}{unit}{way}"

    a, b = (rows for _, rows in sides)
    lines.append(row("wall clock (s)", _total(a, "seconds"), _total(b, "seconds")))
    lines.append(row("model calls", _total(a, "calls"), _total(b, "calls")))
    lines.append(row("prompt tokens (shown)", _total(a, "prompt_tokens"),
                     _total(b, "prompt_tokens")))
    lines.append(row("  of those, cached", measured(a, "cached_tokens"),
                     measured(b, "cached_tokens")))
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


def measured(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """The total of ``key`` over the rows that carry a figure for it, None when none does
    -- a run on a program that never reports the figure, or one kept before it was."""
    said = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(said) if said else None


def peak(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """The most a slot held for any one call of the run: cached plus read tokens, the
    largest of every call of every question. With a rolling window and recall, this --
    not the conversation's length -- is what the slot's context must hold, and what
    decides 16k against 32k per slot. None for a run whose calls reported nothing, or
    one kept before calls were counted."""
    most: int | None = None
    for row in rows:
        for call in row.get("cache_calls") or ():
            try:
                if call[0] is None and call[1] is None:
                    continue
                held = int(call[0] or 0) + int(call[1] or 0)
            except (TypeError, ValueError, IndexError):
                continue
            most = held if most is None else max(most, held)
    return most


def _k(n: int | None) -> str:
    return "-" if not n else (f"{n / 1024:.1f}k" if n < 10240 else f"{n // 1024}k")


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
    #
    # `real`, `mem` and `wired` are what the machine was asked for *while it was answering*,
    # sampled every couple of seconds by `measure.Watching` and kept at their worst -- not
    # read once after the last answer, which is how Flash-Next came to be described as
    # "about 90G" with nobody able to say whether that was its peak or its trough.
    #
    #   `real`  -- Activity Monitor's **Real Mem**: the resident set, shared and mapped
    #              pages included. Eviction is felt here.
    #   `mem`   -- Activity Monitor's **Memory**: the phys_footprint, what the process is
    #              charged for. Memory pressure is charged here, and for an mmapped 87G
    #              model it is far below `real`.
    #   `wired` -- the whole machine's wired memory at its peak, with `+N` beside it for
    #              how much of that arrived after the server did, where a baseline was
    #              taken before the load.
    #
    # A run against a `--base-url` this machine does not own samples nothing and prints
    # `-`: not sampled is not zero. `kv+run` is computed from the peak where there is one.
    #
    # `shape` is everything else that decided the run and had nowhere to live: the cache
    # type, the reasoning budget, the micro-batch, the draft's p-min and head, the
    # projector and the lock -- then the asking, `+batch+kinds`. All of it used to be
    # readable only as a guess at the end of a label, and half of it was not in the label
    # either. `-` where a run kept no record of it: see `shape_of` and `asked_as`.
    #
    # `F1` carries the interval its own questions put around it -- `70% ±6` -- because a
    # mean over twenty questions moves five points between identical runs, and a table that
    # prints the mean alone invites a comparison the questions cannot support.
    kept = [one for one in kept if str(one.get("kind") or "") not in NOT_ANSWERING]
    if not kept:
        print("nothing kept yet")
        return
    several = len(hosts_of(kept)) > 1
    # `served` is what served the run -- program, runtime or format, quant -- so two runs
    # of one model on two programs read apart: `llama.cpp·gguf·Q4_K_XL` beside
    # `ollama·mlx·nvfp4`. `-` for a run kept before it was recorded.
    head = (f"{'run':28} " + (f"{'host':>10} " if several else "")
            + f"{'ctx':>10} {'n':>3} {'shape':32} {'wall':>7} {'load':>5} {'calls':>6} {'read':>8} "
            f"{'written':>8} {'cached':>8} {'peak':>6} {'pfx':>4} {'draft':>6} {'speed':>6} {'find':>7} {'conc':>5} "
            f"{'real':>9} {'mem':>9} {'wired':>8} {'kv+run':>8} {'per 1k':>8} {'F1':>8} "
            f"{'rec':>5} {'prec':>5} "
            f"{'made':>5} {'t/o':>4}  {'sampling':14} {'served'}")
    print(head)
    print("-" * len(head))
    for one in kept:
        rows = one.get("rows") or []
        server = one.get("server") or {}
        scored = [r for r in rows if r.get("expected")]
        def mean(f: Callable[[Mapping[str, Any]], float]) -> str:
            return f"{100 * sum(f(r) for r in scored) / len(scored):.0f}%" if scored else "-"
        right, rec, prec = mean(_hit), mean(_recall), mean(_precision)
        spread = band_of(one, "right")
        right = f"{right} {spread}" if spread else right
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
        cached = measured(rows, "cached_tokens")
        print(f"{_shown(one.get('label', '')):28} "
              + (f"{_shown(host_of(one) or '-', 10):>10} " if several else "")
              + f"{(f'{ctx // 1024}k x{slots}' + (f'/{kv}' if kv else '') + budgeted if ctx else '-'):>10} "
              f"{len(scored):>3} "
              f"{_shown(shaped(one, ctx_shown=True), 32):32} "
              f"{wall_of(one):>6.0f}s "
              f"{(f'{float(load):.0f}s' if load is not None else ''):>5} "
              f"{_total(rows, 'calls'):>6.0f} "
              f"{_total(rows, 'processed_tokens'):>8.0f} "
              f"{_total(rows, 'completion_tokens'):>8.0f} "
              f"{(f'{cached:.0f}' if cached is not None else '-'):>8} "
              f"{_k(peak(rows)):>6} "
              f"{prefixed(server):>4} "
              f"{drafting(rows):>6} "
              f"{_times(speedup(one, kept)):>6} "
              f"{str(server.get('finder') or '-'):>7} "
              f"{at_once(server):>5} "
              f"{_gb(rss):>9} {_gb(server.get('footprint_peak')):>9} {wired_of(server):>8} "
              f"{(f'{beyond / 2**30:.2f}G' if beyond else ('mmap' if server.get('mmapped') else '-')):>8} "
              f"{(f'{per1k / 2**20:.1f}M' if per1k else '-'):>8} "
              f"{right:>8} {rec:>5} {prec:>5} {made(one):>5} {timeouts(one):>4}  "
              f"{sampled(server):14} {short(server.get('served_by'))}")


def by_shape(kept: Sequence[Mapping[str, Any]]) -> None:
    """One line per configuration rather than per run: the runs that were the same thing
    measured twice, gathered, with the mean and the spread of what they measured.

    A sweep leaves a table of forty lines in which the same shape appears four times, and
    reading the difference between two of them is reading noise -- ten questions moved 15%
    in wall clock and five points of F1 between identical runs (2026-09-02). Gathered, the
    question becomes the one worth asking: this shape against that one, with each side's
    band beside it, and ``n`` runs saying how much of the difference is the draw.

    A group is a model, a shape (`shape_of`), an asking (`asked_as`) and a sampling: change
    any of them and it is another measurement, which is what every column of `table` is
    there to say. The band is the questions' own -- pooled over the group's rows, so two
    runs of twenty are read as forty -- and the runs' spread is printed beside it as the
    range of their means, which is the part a band over one run cannot see.
    """
    if not kept:
        print("nothing kept yet")
        return
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for one in kept:
        if not derived(one):
            continue
        model = str((one.get("server") or {}).get("model") or "?")
        grouped.setdefault((model, shaped(one), sampled(one.get("server") or {})),
                           []).append(one)
    if not grouped:
        print("nothing scored yet")
        return
    head = (f"{'model':22} {'shape':40} {'sampling':14} {'runs':>4} {'n':>4} {'F1':>10} "
            f"{'spread':>8} {'rec':>5} {'prec':>5} {'s/q':>10}")
    print(head)
    print("-" * len(head))
    for (model, form, how), group in sorted(grouped.items()):
        rows = [r for one in group for r in (one.get("rows") or []) if r.get("expected")]
        pooled = {"rows": rows}
        got = derived(pooled)
        means = sorted(derived(one)["right"] for one in group)
        # what the runs themselves did, which a band over pooled questions cannot show: two
        # runs of the same shape 8 points apart is the answer to "is this worth re-running"
        drift = f"{(means[-1] - means[0]) * 100:.0f}" if len(means) > 1 else "-"
        right = f"{100 * got['right']:.0f}% {band_of(pooled)}".strip()
        each = f"{got['seconds_per_question']:.1f} " \
               + band_of(pooled, "seconds_per_question", unit="s")
        print(f"{_shown(model, 22):22} {form:40} {_shown(how, 14):14} "
              f"{len(group):>4} {len(rows):>4} {right:>10} {drift:>8} "
              f"{100 * got['recall']:>4.0f}% {100 * got['precision']:>4.0f}% "
              f"{each.strip():>10}")
    print("\nOne line per shape: runs of the same model, server shape, asking and sampling, "
          "read together.")
    print("F1's ± is the 95% interval over the group's questions; spread is how far the "
          "runs' own means lie apart, in points.")


def _gb(value: Any) -> str:
    """``2.00G``, or ``-`` for a figure nothing measured. Not ``0.00G``: a run against a
    server this machine does not own sampled nothing, and nothing is not none."""
    return f"{int(value) / 2**30:.2f}G" if value else "-"


def wired_of(server: Mapping[str, Any]) -> str:
    """The machine's wired memory at its peak, and how much of it the server brought.

    ``93.4G+8.1`` when a baseline was read before the load, so the second number is the
    server's own wired cost; ``93.4G`` when the baseline was taken with it already up and
    the difference would be a subtraction of the wrong thing. ``-`` for a run that sampled
    nothing at all.
    """
    peak = server.get("wired_peak")
    if not peak:
        return "-"
    base = server.get("wired_baseline")
    if base and server.get("wired_baseline_before_load"):
        return f"{int(peak) / 2**30:.1f}G+{(int(peak) - int(base)) / 2**30:.1f}"
    return f"{int(peak) / 2**30:.1f}G"


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
        # what these questions could and could not settle, before the questions themselves:
        # a reader about to explain a five-point difference should see the band first
        got = derived(one)
        if band(one) is not None:
            print(f"  F1 {100 * got['right']:.0f}% {band_of(one)}, "
                  f"recall {100 * got['recall']:.0f}% {band_of(one, 'recall')}, "
                  f"precision {100 * got['precision']:.0f}% {band_of(one, 'precision')}, "
                  f"{got['seconds_per_question']:.1f} s/q "
                  f"{band_of(one, 'seconds_per_question', unit='s')}"
                  f"   (95% over its own {got['questions']:.0f} questions)")
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
                         f"first token {_s(r.get('first_token'))}, "
                         f"queued {_s(r.get('queued'))}")
            print(f"        {note}")


def _s(value: Any) -> str:
    """``1.2s``, or ``-`` for a clock nothing read."""
    return f"{float(value):.1f}s" if value is not None else "-"


def _args_line(args: Mapping[str, Any], width: int = 60) -> str:
    """A tool call's arguments on one line, cut to ``width``."""
    text = json.dumps(args or {}, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= width else text[:width - 1] + "…"


def transcript(kept: Sequence[Mapping[str, Any]], label: str = "",
               question: str = "") -> None:
    """A traced question read back as a conversation: one line per call, in order.

    The table says a question took three calls; this says which three. Per call: the round,
    the tool and the arguments it was called with, how much came back and how many entries
    were in it, what the server spent reading and writing, and how much of the draft head's
    guessing was accepted -- the numbers `Spent` totals per answer, before they are totalled.

    It is where a wrong answer is diagnosed and where a training example is read before
    thousands of them are written: `ml-stack-train-tools from-bench` turns exactly these
    turns into a dataset, and a turn that reads wrong here trains wrong there.

    ``label`` narrows to the runs whose label contains it, ``question`` to the questions
    whose text contains it. Only rows that were traced -- see `wants_trace`.
    """
    from ml_stack.graph.bench.measure import TRACE_ENV

    wanted = [one for one in kept if not label or label in str(one.get("label") or "")]
    shown = 0
    for one in wanted:
        rows = [r for r in (one.get("rows") or []) if r.get("trace")
                and (not question or question.casefold() in str(r.get("question") or "").casefold())]
        for r in rows:
            shown += 1
            print(f"\n{one.get('label', '')}  {r.get('question', '')}"
                  f"   ({_hit(r) * 100:.0f}%, {r.get('calls', 0)} calls, "
                  f"{float(r.get('seconds') or 0):.1f}s"
                  + (", TIMED OUT" if r.get("timed_out") else "")
                  + (f", ERROR {r['error']}" if r.get("error") else "") + ")")
            for entry in r.get("trace") or []:
                for line in _trace_lines(entry):
                    print(f"  {line}")
    if not shown:
        print(f"no traced question found"
              + (f" for {label!r}" if label else "")
              + (f" matching {question!r}" if question else "")
              + f". A run of {bench.SHORT} questions or fewer traces by default; "
                f"{TRACE_ENV}=1 traces one of any size.")


def _trace_lines(entry: Mapping[str, Any]) -> list[str]:
    """One trace entry as the lines `transcript` prints for it."""
    role = str(entry.get("role") or "")
    if role == "tools":
        names = [str((t.get("function") or {}).get("name") or t.get("name") or "")
                 for t in (entry.get("tools") or ())]
        return [f"     tools  {', '.join(n for n in names if n)}"]
    if role == "tool":
        return [f"     <-  {entry.get('name', '?')}  {int(entry.get('chars') or 0)} chars, "
                f"{int(entry.get('ids') or 0)} ids"
                + ("  (cut)" if entry.get("cut") else "")]
    if role != "assistant":
        return [f"     {role:6} {_shown(str(entry.get('content') or '').replace(chr(10), ' '), 88)}"]
    timings = entry.get("timings") or {}
    read = (f"read {int(timings.get('prompt_n') or 0)}"
            f"+{int(timings.get('cache_n') or 0)} cached in "
            f"{float(timings.get('prompt_ms') or 0):.0f}ms")
    wrote = (f"wrote {int(timings.get('predicted_n') or 0)} in "
             f"{float(timings.get('predicted_ms') or 0):.0f}ms")
    drafted = ""
    if int(timings.get("draft_n") or 0):
        taken, guessed = int(timings.get("draft_n_accepted") or 0), int(timings["draft_n"])
        drafted = f", accepted {taken}/{guessed}"
    head = f"{int(entry.get('call') or 0):3}"
    lines = []
    for call in entry.get("tool_calls") or []:
        lines.append(f"{head}  -> {call.get('name', '?')}({_args_line(call.get('args') or {})})")
        head = "   "
    if not lines:
        lines.append(f"{head}  -- answered {int(entry.get('chars') or 0)} chars")
    thought = int(entry.get("thinking_chars") or 0)
    lines.append(f"        {read}, {wrote}{drafted}"
                 + (f", thought {thought} chars" if thought else "")
                 + (f", {entry['finish']}" if entry.get("finish") else ""))
    return lines


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

    ``cost`` is a name in `COSTS` -- seconds, paid_tokens, kv_bytes -- compared on the key
    it maps to: per question for time and tokens, a total for memory.
    """
    key = COSTS.get(cost, cost)
    scored = [(one, derived(one)) for one in kept]
    scored = [(one, d) for one, d in scored if d and d.get(key) is not None]
    front = []
    for one, mine in scored:
        beaten = any(
            other is not one
            and theirs["right"] >= mine["right"] and theirs[key] <= mine[key]
            and (theirs["right"] > mine["right"] or theirs[key] < mine[key])
            for other, theirs in scored)
        if not beaten:
            front.append(one)
    return sorted(front, key=lambda one: derived(one)[key])


def drafted(kept: Sequence[Mapping[str, Any]], *, among: Sequence[Mapping[str, Any]] = (),
            noise: float = NOISE) -> str:
    """The `drafts` summary: what each (head, n-max) was worth, and which to serve.

    One row per drafted run in ``kept`` -- acceptance, seconds per question, speedup over
    the baseline `baseline` finds in ``among`` (``kept`` itself when not given), F1 with the
    interval its questions put around it, and how far it moved from the baseline's --
    fastest first, the rows with no baseline last. Then the recommendation: the fastest
    whose F1 held up against its baseline's, since a head cannot change an answer and one
    that did has changed something else.

    Held up is `held_up`: not separated -- the two 95% intervals overlapping -- rather than
    within a fixed ``noise``, which is what decides where either run carries an interval. A
    head five points down over twenty questions was called a regression and is a coin toss,
    and the recommendation now says which of the two it is looking at. Where the head's own
    s/q is not separated from the baseline's either, the line says that too: a speedup
    inside the clock's own spread is not a speedup.
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
    head = (f"{'draft':28} {'accept':>6} {'s/q':>6} {'speed':>6} {'F1':>9} {'dF1':>5}  "
            f"against")
    lines = [head, "-" * len(head)]
    for one, base, faster, mine, theirs in rows:
        got = mine.get("right")
        delta = ((got - theirs["right"]) * 100
                 if got is not None and theirs.get("right") is not None else None)
        spread = band_of(one, "right")
        lines.append(
            f"{_shown(one.get('label', '')):28} "
            f"{drafting(one.get('rows') or []):>6} "
            f"{per_question(one):>6.1f} "
            f"{_times(faster):>6} "
            f"{(f'{got * 100:.0f}% {spread}'.strip() if got is not None else '-'):>9} "
            f"{(f'{delta:+.0f}' if delta is not None else '-'):>5}  "
            f"{base.get('label', '') if base is not None else 'no baseline'}")
    held = [(one, faster, base) for one, base, faster, mine, theirs in rows
            if faster is not None and base is not None and theirs.get("right") is not None
            and held_up(one, base, noise=noise)]
    # a head that held its F1 but is slower than no head is worth nothing: gpt-oss's eagle3
    # head accepted 65% and still ran at 0.82x, and was recommended (2026-09-02)
    won = [trio for trio in held if trio[1] > 1.0]
    pts = noise * 100
    if won:
        best, faster, base = max(won, key=lambda trio: trio[1])
        apart = separated(best, base, "right")
        # "held" stays the first words of this line whichever rule decided it: `report`'s
        # `recommended_head` reads the recommendation back out of here rather than keeping
        # a second copy of the rule, and a reworded line reads to it as "no head at all"
        why = ("held -- not separated from its baseline's" if apart is False
               else f"held within {pts:g} points of its baseline")
        # ...and whether the speedup itself is more than the clock's own spread
        clock = separated(best, base, "seconds_per_question")
        lines.append(f"serve {best.get('label', '')}: fastest whose F1 {why}, "
                     f"{_times(faster)}"
                     + (" -- but its s/q is not separated from the baseline's either, so "
                        "that speedup is inside the noise" if clock is False else ""))
    elif held:
        best, faster, base = max(held, key=lambda trio: trio[1])
        lines.append(f"serve no head: the best that held its F1, {best.get('label', '')}, "
                     f"is slower than none at {_times(faster)}")
    elif any(faster is not None for _, _, faster, _, _ in rows):
        lines.append("serve no head: every head's F1 fell clear of its baseline's -- "
                     f"separated, or, carrying no interval, more than {pts:g} points")
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
    print(f"\n* on the frontier for accuracy against {AXES.get(cost, cost)}: nothing is "
          f"both more accurate and cheaper.")
    print(f"= a model composed: accuracy from its largest run, cost from its fastest run "
          f"within {noise * 100:g} points of it, scaled to the same number of questions.")


AXES = {"seconds": "wall clock per question (s)",
        "paid_tokens": "tokens paid for per question (read + written)",
        "kv_bytes": "KV cache and runtime (GB)"}
UNITS = {"seconds": "s", "paid_tokens": "tokens", "kv_bytes": "bytes"}


def plot(kept: Sequence[Mapping[str, Any]], where: str | Path, *,
         cost: str = "seconds", noise: float = NOISE) -> str:
    """Write accuracy against cost as a self-contained HTML scatter, frontier joined.

    Plain SVG built here rather than a plotting library: this has to open on a machine with
    no network and no packages, and a chart of a dozen points is a dozen circles. The
    frontier is drawn as a line through the runs nothing beats on both axes, so the shape of
    the trade is visible rather than inferred from a column of numbers. Each model's
    `composed` point is drawn as a ring beside its runs.
    """
    key = COSTS.get(cost, cost)
    points = [(one, derived(one)) for one in list(kept) + composed(kept, noise=noise)]
    points = [(one, d) for one, d in points if d and d.get(key) and d[key] > 0]
    if not points:
        raise ValueError("nothing to plot")
    front = {id(one) for one in pareto([one for one, _ in points], cost=cost)}
    several = len(hosts_of([one for one, _ in points])) > 1
    each = f" {UNITS.get(cost, cost)}" + (" per question" if key != cost else "")

    wide, tall, pad = 900, 520, 70
    costs = [d[key] for _, d in points]
    lo, hi = 0.0, max(costs) * 1.08
    best = max(d["right"] for _, d in points)
    top = min(1.0, best * 1.15)

    def x(v: float) -> float:
        return pad + (v - lo) / (hi - lo or 1) * (wide - 2 * pad)

    def y(v: float) -> float:
        return tall - pad - (v / (top or 1)) * (tall - 2 * pad)

    marks, dots = [], []
    for one, d in sorted(points, key=lambda kv: kv[1][key]):
        cx, cy = x(d[key]), y(d["right"])
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
            f'{100 * d["right"]:.0f}% right, {d[key]:.1f}{each}\n'
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
    """How much of what a draft model guessed was kept, as a percentage; ``none`` for a
    llama-server that drafted nothing, and ``-`` for a program that does not say.

    The count of guesses is not the interesting number and the wall clock already has the
    benefit in it. What this says is *why*: a draft accepted 76% of the time is earning its
    place, and one accepted 20% of the time is costing a pass to be told it was wrong.
    """
    guessed = measured(rows, "draft_tokens")
    if guessed is None:
        return "-"
    if not guessed:
        return "none"
    return f"{100 * (measured(rows, 'draft_taken') or 0) / guessed:.0f}%"
