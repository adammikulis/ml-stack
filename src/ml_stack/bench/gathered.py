"""The kept runs read into the shapes a report tables.

`by_model` groups the answering runs by the model file they were served from, `answering`
drops the ones too short to read, and `across` is the best run of each model. `asking_of`,
`thinking_of` and `cache_of` say how one was asked and served. `extractions`,
`read_messages` and `best_extractor` are the same three questions for the extraction runs,
and `recommended_head` and `fit_for` are what a run's draft head and a fit record say.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ml_stack.bench.record import of
from ml_stack.bench.score import derived
from ml_stack.bench.show import drafted, kv_short

# The words a sweep puts in a label for how it asked (`bench.halves`, `bench._askings`).
# ``shortlist`` is here beside ``plain`` although it is a half rather than a way: without
# it `shortlist-terse` and `plain-terse` read as the same row, which is two measurements
# printed as one -- the mistake every column in `show`'s table exists to prevent.
ASKINGS = ("plain", "shortlist", "terse", "card", "greedy", "rich", "tight", "reach")

_WORD = re.compile(r"[-_:@.]+")

# `drafted`'s recommendation, whose label may itself contain a colon
_RECOMMENDED = re.compile(r"^serve (.+?): fastest whose F1 held")


def model_of(one: Mapping[str, Any]) -> str:
    """The model file a run was served from -- what groups runs into tables. "?" when the
    run was kept before the server record named one."""
    return str((one.get("server") or {}).get("model") or "?")


def asking_of(label: Any) -> str:
    """The way a run was asked, read out of its label: ``plain+terse``, ``shortlist``.

    The label is where this lives and the only place it lives -- `served` composes it from
    the half and the ``--also`` and keeps no separate record -- so it is read back the same
    way, by whole word, never by substring: a model called ``tightfit`` is not a ``tight``
    asking.
    """
    words = [w for w in _WORD.split(str(label or "").lower()) if w in ASKINGS]
    return "+".join(dict.fromkeys(words)) or "-"


def thinking_of(server: Mapping[str, Any]) -> str:
    """``on`` when nothing bound the model's thinking, ``off`` at a budget of zero, else
    the budget itself. A run served with a reasoning budget is another configuration, and
    a budget of 0 is not the same measurement as no budget at all."""
    budget = (server or {}).get("reasoning_budget")
    if budget is None:
        return "on"
    return "off" if int(budget) == 0 else str(int(budget))


def cache_of(server: Mapping[str, Any]) -> str:
    """The KV cache type when it was quantised, "-" at f16. A quantised cache against an
    f16 one is two configurations, not two models."""
    kind = str((server or {}).get("cache_type") or "")
    return kv_short(kind) if kind and kind != "f16" else "-"


def _pct(value: float | None) -> str:
    return f"{value * 100:.0f}%" if value is not None else "-"


def by_model(kept: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """The scored runs grouped by the model file they were served from, newest last.

    A run with no scored question is not an answering run and is left out: it has no F1,
    and a row of dashes said nothing about why.
    """
    out: dict[str, list[Mapping[str, Any]]] = {}
    for one in kept:
        if not derived(one):
            continue
        out.setdefault(model_of(one), []).append(one)
    return out


def answering(kept: Sequence[Mapping[str, Any]], *, min_n: int = 6
              ) -> dict[str, tuple[list[Mapping[str, Any]], int]]:
    """Per model: the runs long enough to read, best F1 first, and how many were too short.

    A run of two questions is a smoke run proving the path works; its score is a coin toss
    by construction, and put in the table beside a full run it is read as a measurement.
    So it is counted in a footnote instead of printed.
    """
    out: dict[str, tuple[list[Mapping[str, Any]], int]] = {}
    for model, mine in by_model(kept).items():
        long_enough = [o for o in mine if derived(o)["questions"] >= min_n]
        long_enough.sort(key=lambda o: (-derived(o)["right"], str(o.get("at") or "")))
        out[model] = (long_enough, len(mine) - len(long_enough))
    return out


def across(kept: Sequence[Mapping[str, Any]], *, full_n: int = 0
           ) -> list[tuple[str, Mapping[str, Any]]]:
    """The best run of each model, most accurate first -- the ranking, as data.

    "Best" is the best F1 among the model's *longest* runs, because a score is only
    comparable with another over the same questions: twenty scored questions make each one
    worth five points of F1 and fifty make it two, so the short run and the full one are
    two measurements that must not be sorted against each other. ``full_n`` fixes the floor
    across every model; unset, each model is read at the largest run it has.
    """
    out = []
    for model, mine in by_model(kept).items():
        floor = full_n or max(derived(o)["questions"] for o in mine)
        pool = [o for o in mine if derived(o)["questions"] >= floor]
        if not pool:
            continue
        out.append((model, max(pool, key=lambda o: (derived(o)["right"],
                                                    str(o.get("at") or "")))))
    return sorted(out, key=lambda pair: -derived(pair[1])["right"])


# ---------------------------------------------------------------- the extraction runs

# How many messages an extraction run reads before its scores are read as a measurement,
# and `min_n`'s opposite number for the other half of the bench. `extract.SMOKE_MESSAGES`
# is three, and three messages fix every coverage to a third: a run that missed one thing
# reads 67%, which is not a rate but an arithmetic accident of how few it was asked.
MIN_MESSAGES = 10


def extract_model_of(one: Mapping[str, Any]) -> str:
    """The model an extraction run read with, "?" for a run that names none.

    Its own top-level ``model`` first, because that is where `extract.save` writes it --
    already the file's basename with the ``.gguf`` off -- and the server record only after.
    An answering run keeps the same fact under ``server.model`` and `model_of` reads it
    there; the two are separate functions rather than one that guesses, since a run that
    named neither would otherwise be grouped under whatever the other kind happened to say.
    """
    return str(one.get("model") or (one.get("server") or {}).get("model") or "?")


def read_messages(one: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The rows of an extraction run whose gold is exact -- the messages its scores were
    measured over.

    `extract.measure` scores the template-written messages and puts the model-written ones
    in ``lower_bound``, and `extract.table` counts the run's messages the same way. So does
    this: a run's `s/msg` counted over rows its coverage was not measured over is two
    numbers over two different sets printed as one row.
    """
    return [r for r in (one.get("rows") or ()) if r.get("exact", True)]


def _scores(one: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return (one.get("scores") or {}).get(key) or {}


def extractions(kept: Sequence[Mapping[str, Any]], *, min_msgs: int = MIN_MESSAGES
                ) -> tuple[list[Mapping[str, Any]], int]:
    """The extraction runs among ``kept``, newest first, and how many were too short.

    Newest first rather than best first, unlike `answering`: which model read best is
    `best_extractor`. A run of fewer than ``min_msgs`` messages is counted rather than
    listed, as a smoke run is in `answering`. The key breaks a tie between two runs kept
    inside the same second: its tail is the run's stamp and the suffix `save` adds when one
    second held two.
    """
    from ml_stack.bench.extract import only

    mine = only(kept)
    long_enough = [one for one in mine if len(read_messages(one)) >= min_msgs]
    long_enough.sort(key=lambda one: (str(one.get("at") or ""), str(one.get("key") or "")),
                     reverse=True)
    return long_enough, len(mine) - len(long_enough)


def best_extractor(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The run that read a graph out of messages best: the highest relation F1 among those
    that read the most messages. None for nothing to read.

    By relations rather than by nodes because naming the right people and joining none of
    them is the failure this half of the bench exists to catch -- a model can list every
    name in a message and state no relation at all, and its node F1 will not say so.

    Among the longest runs only, for `across`'s reason: a coverage over ten messages and one
    over forty are not the same measurement, and sorting them against each other rewards
    whichever was asked less. Ties go to the later run, which is the one measured against
    whatever changed last.
    """
    if not rows:
        return None
    floor = max(len(read_messages(one)) for one in rows)
    pool = [one for one in rows if len(read_messages(one)) >= floor]
    return max(pool, key=lambda one: (float(_scores(one, "relations").get("f1") or 0.0),
                                      str(one.get("at") or "")))


def recommended_head(mine: Sequence[Mapping[str, Any]],
                     among: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The drafted run `drafted` recommends serving, or None for "serve no head".

    Read out of `drafted`'s own last line rather than re-derived here: the rule -- the
    fastest head whose F1 held within the noise of its baseline, and only if it beat no
    head at all -- belongs in one place, and a second copy of it would be a second thing to
    keep in step. A summary that stops saying "serve LABEL" makes this return None, which
    reads as "not measured" rather than as a wrong recommendation.
    """
    if not any(of(o).head for o in mine):
        return None
    said = _RECOMMENDED.match(drafted(mine, among=among).splitlines()[-1])
    if not said:
        return None
    # matched to the end of the label rather than to its first colon: a drafts label is
    # `draft:mtp-alder@n4`, and splitting on the colon recommended a run called "draft"
    return next((o for o in mine if str(o.get("label") or "") == said.group(1)), None)


def fit_for(model: str, fits: Sequence[Any]) -> Any | None:
    """The fit record measured for this model file, or None.

    By file name first and by substring after, because the two sides name a model from
    different ends: a run records what ``/props`` called the file it served, and a fit
    record is keyed on the basename it was measured under.
    """
    name = str(model).lower()
    for one in fits:
        if str(getattr(one, "model", "")).lower() == name:
            return one
    for one in fits:
        held = str(getattr(one, "model", "")).lower()
        if held and (held in name or name in held):
            return one
    return None
