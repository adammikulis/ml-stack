"""Scoring: what one answer was worth, what a run cost per right answer, and which model
to choose.

F1 over what was lit against what was wanted (`Row`), the names an answer made up that no
tool call produced (`unread_named`), the rates that put accuracy over each scarcity
(`derived`), the undrafted baseline a drafted run is read against (`baseline`, `speedup`),
and the conclusion drawn from all of it (`choices`, `composed`, `ranking`, `export`).
Nothing here serves a model or reads a store: it takes kept runs and returns numbers and
text.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.graph.bench.keep import SHORT
from ml_stack.paths import repo_root


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
    # Per call, ``[cached, processed]`` as the server reported them, and from those
    # whether the prompt cache's prefix survived from each call to the next -- see
    # `prefix_kept`. `prefix_hits` is kept over turns, None for a question of one call
    # or a server that reports nothing; the run's is under ``server["prefix_hits"]``.
    cache_calls: list[list[int]] = field(default_factory=list)
    prefix_kept: int = 0
    prefix_turns: int = 0
    prefix_hits: float | None = None
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
    # The question ran past `--per-question`: no answer, `seconds` is the cap, scored wrong.
    # Measured 2026-09-01: gemma-4-26B-A4B took 252 s and 505 s on two questions, all of it
    # in the thinking channel under a 16k ceiling; a run that waits for that is not a run.
    timed_out: bool = False
    # What was said, call by call -- see `bench.measure.Counting.trace`. Every other field
    # here is a total; this is the transcript those totals are of, and the only field a
    # fine-tune can be built from: the conversation up to each model turn, and the call the
    # model made from it. Empty unless the run was traced (`wants_trace`), because it is the
    # one field measured in kilobytes rather than in numbers.
    trace: list[dict[str, Any]] = field(default_factory=list)

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


def _total(rows: Sequence[dict[str, Any]], key: str) -> float:
    return sum(float(r.get(key) or 0) for r in rows)


# How many tokens short of the previous call's whole prompt a cache may fall and still
# count as the prefix kept: a chat template re-rendering the turn boundary re-reads a few.
# A broken prefix is not a few tokens, it is the system prompt and every tool schema.
PREFIX_SLACK = 8


def prefix_kept(per_call: Sequence[Sequence[int]], *, slack: int = PREFIX_SLACK) -> tuple[int, int]:
    """``(kept, turns)``: of the calls after the first, how many found the previous call's
    whole prompt still in the cache -- its cached tokens at least the previous call's
    cached plus processed, less ``slack`` -- and how many were judged.

    A conversation re-sends everything every turn, so the second call of a question
    should pay for the tool result and the model's reply and nothing before them; when it
    pays for the system prompt and the tool schemas again, a change to the asking has
    broken the prefix, and the totals alone cannot see it. A transition is judged only
    when the previous call reported reading something -- a server that reports no
    timings says nothing about its cache, and nothing is not a hit.
    """
    kept = turns = 0
    for before, after in zip(per_call, per_call[1:]):
        cached_before, processed_before = int(before[0]), int(before[1])
        if cached_before + processed_before <= 0:
            continue
        turns += 1
        if int(after[0]) >= cached_before + processed_before - slack:
            kept += 1
    return kept, turns


def prefix_hits(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """A run's share of judged turns whose prefix survived, over every row that carries
    the count; None for a run kept before it was counted, or one with no turn to judge."""
    if not any("prefix_turns" in r for r in rows):
        return None
    turns = _total(rows, "prefix_turns")
    return _total(rows, "prefix_kept") / turns if turns else None


def wall_of(one: Mapping[str, Any]) -> float:
    """What a run took: its turns added up, or, asked at once, the clock over all of them."""
    at_once_ = (one.get("server") or {}).get("concurrency") or {}
    if at_once_.get("seconds"):
        return float(at_once_["seconds"])
    return _total(one.get("rows") or [], "seconds")


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


# How wide the sampling is, and how it is drawn. A run's score is a mean over the questions
# it happened to ask, and the questions are a sample: two identical configurations measured
# on ten questions moved 15% in wall clock and five points of F1 (2026-09-02), and nothing
# in the table said which of that was the change and which was the draw.
#
# So every mean carries the interval the questions themselves put around it: resample the
# run's own questions with replacement, take the mean again, a thousand times, and keep the
# 2.5th and 97.5th percentiles. It measures the only spread there is evidence for -- across
# these questions, on this run -- and needs no second run to do it.
#
# Seeded, because a band that moves when nothing was measured is a band nobody can quote.
# One resample serves all four figures, so F1, recall, precision and s/q are drawn from the
# same questions each time, which is what they were measured on.
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260902

# Bands cost a thousand resamples and `derived` is called several times per run by every
# table, ranking and frontier, so each is computed once. Keyed on the per-question figures
# themselves rather than on the run's identity: two dicts holding the same questions are
# the same measurement, and a dict has no identity worth trusting across a read-back.
_BANDS: dict[Any, dict[str, tuple[float, float]]] = {}
_BANDS_CAP = 512


def _pct(sorted_values: Sequence[float], share: float) -> float:
    """The value at ``share`` of the way through values already sorted."""
    last = len(sorted_values) - 1
    return sorted_values[min(last, max(0, int(round(share * last))))]


def bands(rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP,
          seed: int = BOOTSTRAP_SEED) -> dict[str, tuple[float, float]]:
    """The 95% bootstrap interval each of a run's means sits in, over its own questions.

    ``{"right": (lo, hi), "recall": ..., "precision": ..., "seconds_per_question": ...}``,
    from resampling ``rows`` with replacement ``draws`` times. Empty for a run of one
    question: one question has no spread, and an interval of zero width would claim a
    precision that is not there.

    The s/q band is over the per-question seconds whatever `wall_of` says, because it is
    the per-question spread being asked about; a run asked concurrently has a wall clock
    that is not the sum of its questions, and its band still describes the questions.
    """
    per = tuple((round(_hit(r), 6), round(_recall(r), 6), round(_precision(r), 6),
                 round(float(r.get("seconds") or 0.0), 6)) for r in rows)
    n = len(per)
    if n < 2:
        return {}
    key = (draws, seed, per)
    got = _BANDS.get(key)
    if got is not None:
        return got
    import random

    rand = random.Random(seed)
    means: tuple[list[float], ...] = ([], [], [], [])
    pool = range(n)
    for _ in range(draws):
        picked = [per[i] for i in rand.choices(pool, k=n)]
        for axis in range(4):
            means[axis].append(sum(p[axis] for p in picked) / n)
    out = {}
    for axis, name in enumerate(("right", "recall", "precision", "seconds_per_question")):
        drawn = sorted(means[axis])
        out[name] = (_pct(drawn, 0.025), _pct(drawn, 0.975))
    if len(_BANDS) >= _BANDS_CAP:
        _BANDS.clear()
    _BANDS[key] = out
    return out


def band(one: Mapping[str, Any], key: str = "right") -> tuple[float, float] | None:
    """A run's 95% interval for ``key``, or None where there is none -- a run of one
    question, or a `composed` point, which is two runs and has no questions of its own."""
    got = derived(one)
    lo, hi = got.get(f"{key}_lo"), got.get(f"{key}_hi")
    return (float(lo), float(hi)) if lo is not None and hi is not None else None


def _pm(one: Mapping[str, Any], key: str = "right") -> str:
    """`` ±6``, the half-interval in points, or "" for a run that carries none."""
    half = half_band(one, key)
    return f" ±{half * 100:.0f}" if half is not None else ""


def half_band(one: Mapping[str, Any], key: str = "right") -> float | None:
    """Half the width of a run's 95% interval for ``key``: the ``±`` a mean is printed with.

    Half, because a bootstrap interval is not symmetric about the mean and printing both
    ends everywhere would cost three columns; ``70% ±6`` says how far the questions leave
    the number free to move, which is the only thing a reader does with it.
    """
    got = band(one, key)
    return (got[1] - got[0]) / 2 if got is not None else None


def separated(first: Mapping[str, Any], second: Mapping[str, Any],
              key: str = "right") -> bool | None:
    """Whether two runs differ by more than the questions can account for: True when their
    95% intervals for ``key`` do not overlap, False when they do, None where either has no
    interval and the fixed `NOISE` has to answer instead.

    Not overlapping is the claim worth making. Overlapping is not evidence that two runs
    are the same -- only that this many questions cannot tell them apart, which is why the
    word everywhere is "not separated" and never "equal".
    """
    a, b = band(first, key), band(second, key)
    if a is None or b is None:
        return None
    return a[1] < b[0] or b[1] < a[0]


def derived(one: Mapping[str, Any]) -> dict[str, float]:
    """What a run cost per unit of getting the answer right.

    A score on its own cannot choose between two models: one is more accurate and one is
    cheaper, and which to serve depends on what is scarce. These put accuracy over each of
    the three scarcities in turn -- time, tokens, and the memory a conversation holds -- so
    the trade is a number rather than an argument.

    Right-per-second and right-per-1k are rates: twice the figure is twice the accuracy for
    the same cost. `per_right` inverts them into what one right answer cost, which is the
    easier one to feel.

    A point that carries ``derived`` already -- one `composed` built, with accuracy from one
    run and cost from another -- is returned as it is.
    """
    if "derived" in one:
        return dict(one["derived"])
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
    out = _with_rates({"right": got, "recall": recall, "precision": precision,
                       "shown_per_question": shown, "wanted_per_question": wanted,
                       "seconds": seconds, "paid_tokens": paid, "calls": _total(rows, "calls"),
                       "kv_bytes": memory, "questions": float(len(rows)),
                       "seconds_per_question": _total(rows, "seconds") / len(rows)})
    # ...and what the questions can and cannot tell apart. `<key>_lo`/`<key>_hi` rather
    # than a pair, so `derived` stays a flat mapping of numbers that `rates` and `_flat`
    # can read a key at a time. Absent for a run of one question -- see `bands`.
    for key, (lo, hi) in bands(rows).items():
        out[f"{key}_lo"], out[f"{key}_hi"] = lo, hi
    return out


def _with_rates(out: dict[str, float]) -> dict[str, float]:
    """The rates, from the totals. Guarded: a run that took no time or paid nothing has
    nothing to divide by, and a zero score is a real answer rather than a missing one."""
    got, seconds, paid, memory = out["right"], out["seconds"], out["paid_tokens"], out["kv_bytes"]
    if seconds > 0:
        out["right_per_minute"] = got * 60.0 / seconds
    if paid > 0:
        out["right_per_1k"] = got * 1000.0 / paid
    if memory > 0:
        out["right_per_gb"] = got / (memory / 2**30)
    if got > 0:
        out["seconds_per_right"] = seconds / (got * out["questions"])
        out["tokens_per_right"] = paid / (got * out["questions"])
    return out


def _which(graph: Mapping[str, Any]) -> str:
    """A fingerprint of the graph a run was asked of, so an export can tell them apart."""
    from ml_stack.graph.cache import digest

    try:
        return digest(graph)
    except Exception:  # noqa: BLE001 - a graph that will not hash is not the invented one
        return ""


# How far a run's F1 may fall under its model's accuracy run and still be the same answer,
# as a fraction: 0.05 is five points. On a twenty-question run one question is worth five
# points, so anything tighter rejects the noise of one question.
NOISE = 0.05


def held_up(one: Mapping[str, Any], against: Mapping[str, Any], *,
            noise: float = NOISE) -> bool:
    """Whether ``one``'s F1 stands beside ``against``'s rather than under it.

    It held if it did not fall at all; if it fell, it held unless the fall is bigger than
    the questions can account for -- the two 95% intervals not overlapping. Where either
    run carries no interval, the fixed ``noise`` answers, as it did before there were any.
    """
    fell = derived(against).get("right", 0.0) - derived(one).get("right", 0.0)
    if fell <= 0:
        return True
    apart = separated(one, against, "right")
    return fell <= noise + 1e-9 if apart is None else not apart


@dataclass
class Choice:
    """One model's row in the ranking: the run its accuracy came from, the run its cost came
    from, and the runs that were faster but did not hold the accuracy."""

    model: str
    accuracy: Mapping[str, Any]
    cost: Mapping[str, Any]
    rejected: list[tuple[Mapping[str, Any], float]]     # (run, how far F1 fell, a fraction)
    # runs measured on another host than the accuracy run: never a cost, whatever their F1
    elsewhere: list[Mapping[str, Any]] = field(default_factory=list)

    @property
    def own(self) -> bool:
        """The cost is the accuracy run's own: nothing faster held its score."""
        return self.cost is self.accuracy


def host_of(one: Mapping[str, Any]) -> str:
    """The machine a run was measured on, "" for a run from before that was recorded."""
    return str((one.get("server") or {}).get("host") or "")


def hosts_of(kept: Sequence[Mapping[str, Any]]) -> set[str]:
    """Every host named among ``kept``; more than one means every table says which."""
    return {host_of(one) for one in kept if host_of(one)}


def per_question(one: Mapping[str, Any]) -> float:
    """What one question took, so a twenty-question run compares with a thirty-four."""
    got = derived(one)
    return got["seconds"] / got["questions"] if got.get("questions") else 0.0


def _head_of(one: Mapping[str, Any]) -> str:
    """The draft head a run was served with, "" for none."""
    server = one.get("server") or {}
    return str(server.get("draft_model") or server.get("draft") or "")


def baseline(one: Mapping[str, Any],
             kept: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The undrafted run a drafted one is measured against, or None.

    The newest run among ``kept`` of the same model, on the same llama-server (or neither
    naming one), over the same number of scored questions, served with no draft head. The
    same size because a twenty-question run and a thirty-four are two measurements; the
    same build because a fork's speed against mainline's is the fork's, not the head's.
    None for a run with no head: it is its own baseline, and a speedup of 1.00x would say
    nothing.
    """
    if not _head_of(one):
        return None
    server = one.get("server") or {}
    mine = derived(one)
    if not mine.get("questions"):
        return None
    wanted = (str(server.get("model") or ""), str(server.get("binary") or ""),
              mine["questions"])
    found = [(n, o) for n, o in enumerate(kept)
             if o is not one and not _head_of(o)
             and (str((o.get("server") or {}).get("model") or ""),
                  str((o.get("server") or {}).get("binary") or ""),
                  derived(o).get("questions")) == wanted]
    if not found:
        return None
    return max(found, key=lambda pair: (str(pair[1].get("at") or ""), pair[0]))[1]


def speedup(one: Mapping[str, Any], kept: Sequence[Mapping[str, Any]]) -> float | None:
    """What a draft head was worth: the baseline's seconds per question over this run's.

    The wall clock already has the benefit in it and `drafting` says how often the head was
    right; this is the number a reader was dividing out by hand -- `1.42x` -- and it only
    exists against the run `baseline` finds, so it is None for an undrafted run, for one
    with no baseline among ``kept``, and for one that took no time at all.
    """
    base = baseline(one, kept)
    if base is None:
        return None
    mine = per_question(one)
    return per_question(base) / mine if mine > 0 else None


def _times(ratio: float | None) -> str:
    """``1.42x``, or "" for no ratio."""
    return f"{ratio:.2f}x" if ratio is not None else ""


def choices(kept: Sequence[Mapping[str, Any]], *,
            noise: float = NOISE) -> tuple[list[Choice], int]:
    """Per model, where its accuracy comes from and where its cost comes from.

    A draft head cannot change an answer -- the target verifies every token -- only the wall
    clock and the memory. So a model's accuracy is its largest run, the full sweep, and its
    cost is the fastest configuration that kept that accuracy: a `drafts` run of twenty
    questions with a head and a draft length, possibly on another build, if its F1 held
    within ``noise``. Taking both from one run ranked a drafted model at its undrafted
    speed, or not at all when the drafted run was short.

    Accuracy: the run with the most questions, the best F1 among those, the newest on a
    full tie. Cost: the fastest per
    question of the model's runs of at least `SHORT` questions whose F1 held up against the
    accuracy run's -- `held_up`: it did not fall, or the fall is inside what these questions
    can account for, the two 95% intervals still overlapping. Where a run carries no
    interval the fixed ``noise`` decides, as it did before there were any. The accuracy run
    itself when nothing faster held. ``rejected`` holds the rest, so a head that hurt
    accuracy is seen rather than skipped -- and it now holds only the runs the questions
    really did separate, rather than every run a fraction of a question below a fixed line.

    Never across hosts: a cost run from another machine than the accuracy run's is a
    different GPU, and goes under ``elsewhere`` -- listed by the ranking as "other host"
    -- however well its F1 held.

    Returns the choices, most accurate first, and how many runs were too short to count.
    """
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    too_few = 0
    for one in kept:
        got = derived(one)
        if not got:
            continue
        # A smoke run asks two questions to prove the path works; its score is meaningless
        # by construction, and it would otherwise rank a model on a coin toss. Anything
        # below a short run is evidence that something ran, not evidence of how well --
        # and not of what it cost either.
        if got["questions"] < SHORT:
            too_few += 1
            continue
        by_model.setdefault(str((one.get("server") or {}).get("model") or "?"), []).append(one)
    out = []
    for model, mine in by_model.items():
        # the most questions; among those, the best F1 -- the askings (plain, terse, card)
        # are configurations a person chooses between, and the newest was the card run,
        # which ranked E4B at 48% when plain had 49% (measured 2026-09-01); newest last
        accuracy = max(mine, key=lambda o: (derived(o)["questions"], derived(o)["right"],
                                            str(o.get("at") or "")))
        top = derived(accuracy)["right"]
        here = [o for o in mine if host_of(o) == host_of(accuracy)]
        elsewhere = [o for o in mine if o not in here]
        held = [o for o in here if held_up(o, accuracy, noise=noise)]
        cost = min(held, key=lambda o: (per_question(o), o is not accuracy))
        if per_question(cost) >= per_question(accuracy):
            cost = accuracy
        rejected = sorted(((o, top - derived(o)["right"]) for o in here if o not in held),
                          key=lambda pair: -pair[1])
        out.append(Choice(model, accuracy, cost, rejected, elsewhere))
    return sorted(out, key=lambda c: -derived(c.accuracy)["right"]), too_few


def composed(kept: Sequence[Mapping[str, Any]], *, noise: float = NOISE) -> list[dict[str, Any]]:
    """Each model as one point: accuracy from its largest run, cost from its fastest run
    that held it, scaled to the accuracy run's question count so both are one run's worth.

    `derived` reads one back as it is, so `pareto`, `rates` and `plot` take them beside the
    runs; ``composed`` marks them, ``from`` names the run the cost came from.
    """
    out = []
    for choice in choices(kept, noise=noise)[0]:
        a, c = derived(choice.accuracy), derived(choice.cost)
        scale = a["questions"] / c["questions"] if c.get("questions") else 1.0
        point = {"right": a["right"], "recall": a["recall"], "precision": a["precision"],
                 "shown_per_question": a["shown_per_question"],
                 "wanted_per_question": a["wanted_per_question"], "questions": a["questions"],
                 "seconds": c["seconds"] * scale, "paid_tokens": c["paid_tokens"] * scale,
                 "calls": c["calls"] * scale, "kv_bytes": c["kv_bytes"]}
        out.append({"label": choice.model, "composed": True, "model": choice.model,
                    "from": str(choice.cost.get("label") or ""), "own": choice.own,
                    "server": {"host": host_of(choice.accuracy)},
                    "derived": _with_rates(point)})
    return out


def _build(binary: Any) -> str:
    """Which llama-server a run was served by, as the last two path segments: a managed
    build is ``<name>/llama-server``, so a fork's run says so and mainline's says current."""
    parts = Path(str(binary)).parts if binary else ()
    return "/".join(parts[-2:])


def ranking(kept: Sequence[Mapping[str, Any]], where: str | Path | None = None, *,
            noise: float = NOISE) -> str:
    """Which model to choose, as a conclusion rather than as evidence.

    The raw runs are not committed: they describe one machine and one llama.cpp build, go
    stale with the next model release, and may have been asked of a real community. What
    survives all of that is the *ranking* -- which model answers best, what it costs, and
    which draft head and sampling were chosen for it -- because that is what the defaults in
    this library are set from, and a default with no recorded reason is a default nobody can
    argue with.

    One line per model, composed as `choices` composes it: accuracy from its largest run,
    cost -- per question, so a short drafted run compares with a full one -- from its
    fastest run that held that accuracy, and the last column saying which run and which
    build that was. A run that was rejected for cost because its F1 fell is listed under
    the table by name, so a head that hurt accuracy is visible rather than silently ignored.
    """
    over, skipped = _over_invented(kept)
    chosen, too_few = choices(over, noise=noise)
    pts = noise * 100
    # which machine, only when more than one measured: a column nobody needs is noise
    several = len(hosts_of(over)) > 1
    lines = ["# Which model answers best",
             "",
             "Measured over the invented community that ships with this package, by",
             "`ml-stack-bench`. A conclusion, not evidence: the runs behind it are not in this",
             "repository. Re-measure after any model release -- none of this survives one.",
             "",
             "Accuracy is each model's largest run -- the most questions, the newest on a tie --",
             "since a draft head cannot change an answer, only the clock. Cost is the model's",
             f"fastest run of at least {SHORT} questions whose F1 was not separated from that",
             "-- the two 95% bootstrap intervals over their own questions overlapping, or,",
             f"for a run carrying no interval, within {pts:g} points -- per question, whatever",
             "head, draft length or build it ran on; the last column says which run that was.",
             "Every F1 below carries the interval its questions put around it: a difference",
             "smaller than the interval is a difference these questions did not measure.",
             "",
             "| model | F1 | recall | precision | questions | s/question | load | resident "
             "| kv+run | sampling | find | made |" + (" host |" if several else "")
             + " cost from |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
             + (" --- |" if several else "") + " --- |"]
    for choice in chosen:
        a, c = _flat(choice.accuracy, over), _flat(choice.cost, over)
        gb, kv, load = c.get("resident_bytes"), c.get("kv_and_run_bytes"), c.get("load_s")
        temp = (a.get("sampling") or {}).get("temperature")
        build = _build(c.get("binary"))
        source = ("its own run" if choice.own
                  else f"`{c.get('label')}`") + (f" on {build}" if build else "")
        if not choice.own:
            # and what the head was worth, when its baseline was measured: `1.42x`
            faster = c.get("speedup")
            # ...and whether it is really faster. Two runs whose s/q intervals overlap are
            # not separated by these questions, however the means fell, and a recommendation
            # that does not say so is read as a measurement when it is a coin toss.
            apart = separated(choice.cost, choice.accuracy, "seconds_per_question")
            source += (f" ({c.get('questions')} q"
                       + (f", {_times(faster)}" if faster is not None else "")
                       + (", s/q not separated" if apart is False else "") + ")")
        lines.append(
            f"| `{choice.model}` "
            f"| {(a.get('f1') or 0) * 100:.0f}%{_pm(choice.accuracy, 'right')} "
            f"| {(a.get('recall') or 0) * 100:.0f}% "
            f"| {(a.get('precision') or 0) * 100:.0f}% "
            f"| {a.get('questions') or '-'} "
            f"| {per_question(choice.cost):.1f} "
            f"| {f'{float(load):.0f}s' if load is not None else '-'} "
            f"| {f'{gb / 2**30:.1f}G' if gb else '-'} "
            f"| {f'{kv / 2**30:.1f}G' if kv else '-'} "
            f"| {'greedy' if temp == 0 else (f'temp {temp}' if temp is not None else '-')} "
            f"| {a.get('finder') or '-'} "
            f"| {'-' if a.get('unread_named') is None else a['unread_named']} "
            + (f"| {a.get('host') or '-'} " if several else "")
            + f"| {source} |")
    refused = [(choice, run, fell) for choice in chosen for run, fell in choice.rejected]
    if refused:
        lines += ["", "Runs whose F1 fell clear of their model's -- separated, their 95% "
                      "intervals not", f"overlapping, or, carrying none, more than {pts:g} "
                      "points down -- so their cost was", "not taken. A head cannot change "
                      "an answer, so look at what else these changed:", ""]
        for choice, run, fell in refused:
            lines.append(f"- `{choice.model}` rejected: `{run.get('label')}` F1 "
                         f"-{fell * 100:.0f} pts ({derived(run)['questions']:.0f} q, "
                         f"{per_question(run):.1f} s/question)")
    away = [(choice, run) for choice in chosen for run in choice.elsewhere]
    if away:
        lines += ["", "Runs measured on another host than their model's accuracy run, so "
                      "their cost was not taken -- a different machine is a different clock:",
                  ""]
        for choice, run in away:
            lines.append(f"- `{choice.model}` rejected: other host -- "
                         f"`{run.get('label')}` on {host_of(run) or '?'} "
                         f"({derived(run)['questions']:.0f} q, "
                         f"{per_question(run):.1f} s/question)")
    notes = []
    if too_few:
        notes.append(f"{too_few} run(s) not ranked: fewer than {SHORT} questions, which is a "
                     f"smoke run proving the path works rather than a measurement -- it "
                     f"supplies neither accuracy nor cost.")
    if skipped:
        notes.append(f"{skipped} run(s) not ranked: not measured over the community that "
                     f"ships with this package.")
    if notes:
        lines += ["", "*" + " ".join(notes) + "*"]
    body = "\n".join(lines) + "\n"
    if where is not None:
        Path(where).expanduser().write_text(body, encoding="utf-8")
    return body


def invented_digest() -> str:
    """The fingerprint of the community that ships with this package."""
    from ml_stack.graph.cache import digest
    from ml_stack.graph.community import graph as invented

    return digest(invented())


def _over_invented(kept: Sequence[Mapping[str, Any]], *,
                   anyway: bool = False) -> tuple[list[Mapping[str, Any]], int]:
    """The runs that may leave this machine, and how many were held back.

    Shared by `export` and `ranking` so the gate cannot be enforced in one and forgotten in
    the other: only scored runs whose recorded graph fingerprint is the community that ships
    with this package, and never a run from before that marker existed -- not knowing which
    graph a run read is not the same as knowing it was invented.
    """
    mine = "" if anyway else invented_digest()
    out: list[Mapping[str, Any]] = []
    skipped = 0
    for one in kept:
        if not any(r.get("expected") for r in (one.get("rows") or [])):
            continue
        if mine and str((one.get("server") or {}).get("graph") or "") != mine:
            skipped += 1
            continue
        out.append(one)
    return out, skipped


def _flat(one: Mapping[str, Any], among: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """One run as the totals and the server, and no question or entry: what `export` writes
    and `ranking` reads. ``among`` is where `speedup` looks for the run's baseline."""
    rows = [r for r in (one.get("rows") or []) if r.get("expected")]
    got = derived(one)
    server = one.get("server") or {}
    faster = speedup(one, among)
    return {
        "at": one.get("at", ""), "label": one.get("label", ""),
        "questions": len(rows),
        "f1": round(got.get("right", 0), 4),
        "recall": round(got.get("recall", 0), 4),
        "precision": round(got.get("precision", 0), 4),
        "lit_per_question": round(got.get("shown_per_question", 0), 2),
        # what these questions could and could not tell apart: the 95% bootstrap interval
        # around F1 and around s/q. None for a run of one question -- see `bands`
        "f1_band": [round(v, 4) for v in band(one, "right")] if band(one) else None,
        "seconds_per_question_band": ([round(v, 3)
                                       for v in band(one, "seconds_per_question")]
                                      if band(one, "seconds_per_question") else None),
        "seconds": round(got.get("seconds", 0)),
        "calls": int(got.get("calls", 0)),
        "read_tokens": int(_total(rows, "processed_tokens")),
        "written_tokens": int(_total(rows, "completion_tokens")),
        "draft_offered": int(_total(rows, "draft_tokens")),
        "draft_kept": int(_total(rows, "draft_taken")),
        # None for an undrafted run, or one whose baseline is not among the runs
        "speedup": round(faster, 3) if faster is not None else None,
        "timed_out": sum(1 for r in rows if r.get("timed_out")),
        # None for a run kept before the cache was counted per turn, or with no turn to judge
        "prefix_hits": (round(float(server["prefix_hits"]), 4)
                        if server.get("prefix_hits") is not None else None),
        "context": server.get("context"), "slots": server.get("slots"),
        "cache_type": str(server.get("cache_type") or ""),
        "reasoning_budget": server.get("reasoning_budget"),
        "model": server.get("model", ""), "draft_model": server.get("draft_model", ""),
        # the llama-server that served it, so a fork's run is told from mainline's
        "binary": str(server.get("binary") or ""),
        # which machine and which code measured it; "" for a run from before either was
        "host": str(server.get("host") or ""),
        "commit": str(server.get("commit") or ""),
        # None for a run kept before the lease recorded it: not recorded is not instant
        "load_s": server.get("load_s"),
        "resident_bytes": server.get("resident_bytes"),
        # what the machine was asked for while it answered -- see `measure.Watching`.
        # None for a run kept before it was sampled, or one against a server this machine
        # does not own: not sampled is not zero
        "resident_peak": server.get("resident_peak"),
        "footprint_peak": server.get("footprint_peak"),
        "wired_peak": server.get("wired_peak"),
        "wired_baseline": server.get("wired_baseline"),
        "available_low": server.get("available_low"),
        "kv_and_run_bytes": server.get("kv_and_run_bytes"),
        "mmapped": bool(server.get("mmapped")),
        "sampling": server.get("sampling") or {},
        "finder": str(server.get("finder") or ""),
        # None, not 0, for a run from before this was counted: not counted is not none
        "unread_named": (int(_total(rows, "unread_named"))
                         if any("unread_named" in r for r in rows) else None),
        "concurrency": dict(server.get("concurrency") or {}) or None,
    }


def _exportable(kept: Sequence[Mapping[str, Any]], *,
                anyway: bool = False) -> tuple[list[dict[str, Any]], int]:
    """`_over_invented`, flattened by `_flat`: what `export` writes."""
    out, skipped = _over_invented(kept, anyway=anyway)
    return [_flat(one, out) for one in out], skipped


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
