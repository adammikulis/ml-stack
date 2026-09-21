"""The serving record a kept run sets.

`measured_best` is the row a model's record is written from -- the fastest whose F1 held
-- `flags_of` reads the asking a run recorded as the fields of a profile, `profile_of`
builds the record, `workload_of` says which slot it goes in, and `write_profiles` writes
one per model and workload.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ml_stack.bench.gathered import _WORD, across, by_model
from ml_stack.bench.keep import SHORT
from ml_stack.bench.record import of
from ml_stack.bench.score import derived, held_up, host_of, per_question
from ml_stack.serve.profile import ASK

# Every word a label can carry about the asking, and what it means to `converse`. Wider
# than `ASKINGS`, which is only what the tables print: `batch`, `kinds` and `summary` ride
# on an asking rather than naming one, and a profile has to carry them or a model measured with
# all three would be served with none.
FLAGS = ("tight", "batch", "single", "few", "kinds", "summary", "rich", "terse",
        "constrain_ids", "reach", "rounds")


def flags_of(one: Mapping[str, Any]) -> dict[str, Any]:
    """The asking a run records, as the fields of a profile.

    A run kept since `asked_with` carries ``asking`` -- the keywords `converse` was actually
    handed -- and that is taken as it is: it is the record, and reading a label instead
    would be inferring what is already written down.

    Older runs have only the label, so it is read by whole word, never by substring -- a
    model called ``tightfit`` is not a ``tight`` asking. ``loose`` is the one word that
    means the *absence* of a way: it is the control the ranking runs were measured with,
    and it is how ``tight=False`` is said.
    """
    said = one.get("asking")
    if isinstance(said, Mapping) and said:
        out: dict[str, Any] = {"tight": bool(said.get("tight", True))}
        for way in ("batch", "kinds", "summary", "rich", "terse", "single", "few",
                    "constrain_ids"):
            out[way] = bool(said.get(way, False))
        if said.get("reach"):
            out["reach"] = int(said["reach"])
        if said.get("rounds"):
            out["rounds"] = int(said["rounds"])
        return out
    words = {w for w in _WORD.split(str(one.get("label") or "").lower()) if w}
    out = {"tight": "loose" not in words}
    for way in ("batch", "kinds", "summary", "rich", "terse", "single", "few"):
        out[way] = way in words
    out["constrain_ids"] = False
    if "reach" in words:
        from ml_stack.bench.askings import REACH

        # the label says a run reached; it does not say how far, and `--also reach` is the
        # only thing that puts the word there, so its own figure is what was measured
        out["reach"] = int(REACH)
    return out


def measured_best(mine: Sequence[Mapping[str, Any]], *, full_n: int = 0
                  ) -> Mapping[str, Any] | None:
    """The run one model's record should be written from: the fastest whose F1 held.

    Held is `score.held_up`: it did not fall at all, or the fall is inside what the
    questions can account for. Compared only among the model's longest runs, since a score
    means nothing beside a score over a different number of questions; ties go to the
    higher F1, then to the later run.

    None for a model whose longest run is under `SHORT` questions -- a record is never set
    from a smoke run -- and `main` says so.
    """
    pool = [one for one in mine if derived(one)]
    if not pool:
        return None
    floor = full_n or max(derived(one)["questions"] for one in pool)
    if floor < SHORT:
        return None
    pool = [one for one in pool if derived(one)["questions"] >= floor]
    if not pool:
        return None
    best = max(pool, key=lambda o: (derived(o)["right"], str(o.get("at") or "")))
    held = [one for one in pool if held_up(one, best)] or [best]
    return min(held, key=lambda o: (per_question(o), -derived(o)["right"],
                                    str(o.get("at") or "")))


def workload_of(one: Mapping[str, Any]) -> str:
    """The workload a run measured. A run kept before workloads measured the graph asking,
    which is what every answering run in a store is."""
    return str(of(one).workload or ASK)


def profile_of(model: str, one: Mapping[str, Any]) -> Any:
    """The settings a run records, as a `ml_stack.serve.profile.Profile`.

    One row sets one record, and the record says which row: settings composed from the
    accuracy of one run and the speed of another is a configuration nobody ever served.
    Nothing is guessed -- a field the run does not carry is left at its default, and `add`
    keeps whatever the older record knew about the two fields a kept run cannot see (the
    extra llama-server flags and the vision projector).
    """
    from ml_stack.hub import spec_for
    from ml_stack.serve.profile import record

    server = one.get("server") or {}
    got = derived(one)
    kept = of(one)
    head = kept.head
    slots = int(server.get("slots") or 0) or 1
    context = int(server.get("context") or 0)
    asked = one.get("asking") if isinstance(one.get("asking"), Mapping) else {}
    sampling = asked.get("sampling") or server.get("sampling")
    return record(
        model,
        workload=workload_of(one),
        build=kept.build,
        draft=head,
        spec_type=spec_for(head) if head else "",
        spec_draft_max=(int(server["spec_draft_max"])
                        if server.get("spec_draft_max") is not None else None),
        spec_p_min=(float(server["spec_p_min"])
                    if server.get("spec_p_min") is not None else None),
        cache_type=str(server.get("cache_type") or ""),
        draft_cache_type=str(server.get("draft_cache_type") or ""),
        reasoning_budget=(int(server["reasoning_budget"])
                          if server.get("reasoning_budget") is not None else None),
        slot_context=(context // slots) if context else 32768,
        parallel=slots,
        sampling=dict(sampling) if isinstance(sampling, Mapping) else {},
        measured_at=str(one.get("at") or "")[:10],
        label=str(one.get("label") or ""),
        questions=int(got.get("questions") or 0),
        right=float(got.get("right") or 0.0),
        recall=float(got.get("recall") or 0.0),
        precision=float(got.get("precision") or 0.0),
        seconds_per_question=float(per_question(one)),
        host=host_of(one),
        note=(f"set from the fastest row whose F1 held: `{one.get('label') or '?'}`, "
              f"{int(got.get('questions') or 0)} question(s) at "
              f"{float(got.get('right') or 0.0) * 100:.0f}% F1, "
              f"{float(per_question(one)):.1f} s/question"),
        **flags_of(one))


def write_profiles(kept: Sequence[Mapping[str, Any]], *, full_n: int = 0,
                   path: Path | None = None) -> list[tuple[Any, Path]]:
    """Write one record per model and workload, from the row `across` ranks it by. Returns
    what it wrote.

    The ranking fixes the *order* and `measured_best` fixes the *row*. They are not the
    same question: the ranking asks which model answers best, and a record asks how this
    model should be asked, where two askings the questions cannot tell apart should be
    settled by the seconds rather than by a hundredth of an F1. Both read only a model's
    longest runs, so a profile is never settings chosen by a coin toss over two questions.
    """
    grouped = by_model(kept)
    out = []
    for model, _ranked in across(kept, full_n=full_n):
        mine = grouped.get(model) or []
        for workload in sorted({workload_of(r) for r in mine}):
            one = measured_best([r for r in mine if workload_of(r) == workload],
                                full_n=full_n)
            if one is None:
                continue
            out.append(_written(model, workload, one, path))
    return out


def _written(model: str, workload: str, one: Mapping[str, Any],
             path: Path | None) -> tuple[Any, Path]:
    """One record, written from ``one`` into the ``model`` and ``workload`` slot."""
    from dataclasses import replace

    from ml_stack.serve.profile import FLAGS, add, profile_for, records_in, writable_file

    made_one = profile_of(model, one)
    if not asked_recorded(one):
        # a run whose label is all `flags_of` could read keeps the asking the record holds
        older = profile_for(model, workload=workload,
                            records=records_in(path or writable_file()))
        if older is not None and older.workload == workload:
            asked = {flag: getattr(older, flag) for flag in FLAGS}
            asked.update(reach=older.reach, rounds=older.rounds)
            made_one = replace(made_one, **asked,
                               note=(made_one.note + " -- asked as the record already "
                                     "said: this run predates asking records"))
    return made_one, add(made_one, path=path)


def asked_recorded(one: Mapping[str, Any]) -> bool:
    """Whether a run carries the asking record `asked_with` keeps -- the keywords
    `converse` was handed -- rather than only a label to read words from."""
    said = one.get("asking")
    return isinstance(said, Mapping) and bool(said)
