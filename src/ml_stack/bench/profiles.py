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

# Every word a label can carry about the asking; wider than `ASKINGS`, which the tables print.
FLAGS = ("tight", "batch", "single", "few", "kinds", "summary", "rich", "terse",
        "constrain_ids", "reach", "rounds")


def flags_of(one: Mapping[str, Any]) -> dict[str, Any]:
    """The asking a run records, as the fields of a profile.

    Read from the run's ``asking`` record when it has one, otherwise from its label by
    whole word; ``loose`` in a label means ``tight=False``.
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

        # a label says `reach` without a distance; `--also reach` uses REACH
        out["reach"] = int(REACH)
    return out


def measured_best(mine: Sequence[Mapping[str, Any]], *, full_n: int = 0
                  ) -> Mapping[str, Any] | None:
    """The run one model's record is written from: the fastest whose F1 held.

    Held is `score.held_up`, compared among the model's longest runs; ties go to the
    higher F1, then the later run. None when the longest run is under `SHORT` questions.
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
    """The settings one run records, as a `ml_stack.serve.profile.Profile`; a field the
    run does not carry is left at its default."""
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
    """Write one record per model and workload, in `across` order, each from the row
    `measured_best` picks. Returns what it wrote."""
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

    from ml_stack.serve.profile import FLAGS, _plain, add, profile_for, records_in, writable_file

    made_one = profile_of(model, one)
    if not asked_recorded(one):
        # a run whose label is all `flags_of` could read keeps the asking its own record holds
        older = profile_for(model, workload=workload,
                            records=records_in(path or writable_file()))
        if (older is not None and older.workload == workload
                and _plain(older.model) == _plain(model)):
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
