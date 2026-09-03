"""The "Extraction" section of ``ml-stack-bench report``.

Extraction runs are kept in the same store as answering runs and used to be dropped by the
report before it read them, so the document could say nothing about reading a graph *out
of* messages -- which is where the day's cause-then-fix record lived: topics at 19%
precision and relations at 0% F1 with 26% invented ids, then the instructions given the
topic and relation vocabulary, then 67% / 62% / 7%. What is tested here is the arranging,
not the scoring: that the runs table newest first, that a three-message smoke run is
footnoted rather than tabled beside a full one, that the best model by relation F1 is named
at the most messages, and that a window holding no extraction run prints no section at all.

Everything is built in ``tmp_path`` -- extraction runs through `extract.save` with
hand-written `MessageRow` fixtures and score records, answering runs through `bench.save`.
Nothing here reads ``~/.ml-stack``, serves a model or touches a GPU.
"""

from __future__ import annotations

import json
import pathlib
import time

import pytest

from ml_stack.graph import bench
from ml_stack.graph.bench import Row
from ml_stack.graph.bench import extract as bx
from ml_stack.graph.bench.report import (
    MIN_MESSAGES,
    best_extractor,
    extract_model_of,
    extractions,
    read_messages,
    report,
)

GIB = 1024 ** 3

# Two invented readers over the invented foundry world the rest of the bench tests use.
KESTREL = "kestrel-8B-UD-Q4_K_XL"
EMBER = "ember-2B-Q4_K_M"

_STRFTIME = time.strftime


def _clock(monkeypatch: pytest.MonkeyPatch, minutes: int) -> None:
    """Move the clock on before the next run is kept.

    Both savers stamp a run with `time.strftime` as they are called, so a fixture that
    writes five runs inside one second gives them all the same ``at`` -- and then "newest
    first" would be decided by `bench.runs`, which sorts by key and so by label. A real
    afternoon spaces its runs out; this does the same, so the order under test is the
    timestamp's and not the alphabet's.
    """
    when = time.localtime(time.mktime((2026, 9, 2, 9, 0, 0, 0, 0, -1)) + minutes * 60)
    monkeypatch.setattr(time, "strftime", lambda fmt, _t=None: _STRFTIME(fmt, when))


def _messages(n: int, *, seconds: float = 2.5, tokens: int = 500,
              exact: bool = True) -> list[bx.MessageRow]:
    """``n`` messages read, each costing the same, so a run's s/msg and tok/msg are exact."""
    return [bx.MessageRow(id=f"msg-{i}", sender="Marisol Quen", channel="foundry-floor",
                          seconds=seconds, prompt_tokens=tokens - 100,
                          completion_tokens=100, exact=exact)
            for i in range(n)]


def _scored(*, node_f1: float, rel_cov: float, rel_prec: float, rel_f1: float,
            top_prec: float, invented: float) -> dict:
    """A score record of the shape `extract.score` writes, with the numbers this file
    asserts on named rather than measured -- the report composes, it does not score."""
    return {
        "by_kind": {
            "people": {"coverage": 0.90, "precision": 0.95, "f1": 0.92, "of": 20,
                       "found": 18, "said": 19, "invented": 1},
            "orgs": {"coverage": 0.80, "precision": 0.90, "f1": 0.85, "of": 5,
                     "found": 4, "said": 4, "invented": 0},
            "topics": {"coverage": 0.60, "precision": top_prec, "f1": 0.50, "of": 10,
                       "found": 6, "said": 9, "invented": 3},
            "places": {"coverage": 0.70, "precision": 0.80, "f1": 0.75, "of": 4,
                       "found": 3, "said": 3, "invented": 0},
        },
        "nodes": {"coverage": 0.80, "precision": 0.88, "f1": node_f1, "of": 39,
                  "found": 31, "said": 35, "invented": 4},
        "relations": {"coverage": rel_cov, "precision": rel_prec, "f1": rel_f1,
                      "of": 30, "found": 20, "said": 24, "invented": 4},
        "invented": {"count": 2, "of": 23, "rate": invented},
        "attrs": {"org": {"right": 7, "stated": 8}, "place": {"right": 4, "stated": 5}},
        "topology": {"extracted": {"nodes": 35, "edges": 24, "components": 3,
                                   "largest_share": 0.77},
                     "gold": {"nodes": 39, "edges": 30, "components": 1,
                              "largest_share": 1.0}},
        "conformance": {"relations": {"in_vocabulary": 22, "of": 24},
                        "entities": {"in_schema": 35, "of": 35}, "off_schema": 0},
        "survival": {"mean": 0.71, "messages": 24},
        "resolution": {"splits": 1.08, "merges": 1.00},
    }


def _extraction(store: str, label: str, *, model: str, messages: int, seconds: float = 2.5,
                node_f1: float = 0.84, rel_cov: float = 0.70, rel_prec: float = 0.83,
                rel_f1: float = 0.76, top_prec: float = 0.86, invented: float = 0.07,
                resident: int = 6 * GIB) -> str:
    return bx.save(store, _messages(messages, seconds=seconds), label=label, model=model,
                   world={"kind": "foundry", "size": "small", "seed": 3},
                   sample={"n": messages, "seed": 3},
                   scores=_scored(node_f1=node_f1, rel_cov=rel_cov, rel_prec=rel_prec,
                                  rel_f1=rel_f1, top_prec=top_prec, invented=invented),
                   held={"model": model, "resident_bytes": resident})


def _answering(store: str, label: str, *, model: str = "kestrel.gguf", questions: int = 20,
               hits: int = 15, seconds: float = 200.0) -> str:
    rows = [Row(label=label, question=f"who runs the kiln, question {n}?",
                expected=["person:marisol-quen"],
                shown=["person:marisol-quen"] if n < hits else [],
                seconds=seconds / questions, calls=3, answer_chars=200,
                processed_tokens=500, completion_tokens=100)
            for n in range(questions)]
    return bench.save(store, rows, held={"model": model, "context": 32768, "slots": 1,
                                         "binary": "/builds/current/llama-server",
                                         "graph": "invented"})


@pytest.fixture()
def store(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """One store holding both halves of the bench: two answering runs, two extraction runs
    of two different readers, and one three-message extraction smoke run, an hour apart.

    The older reader is the one measured before the extraction instructions named a topic
    and a relation vocabulary, so its topic precision and relation F1 are the low pair and
    the newer reader's are the high one -- the shape of the record the section exists for.
    """
    pytest.importorskip("ladybug")
    where = str(tmp_path / "runs.ladybug")
    _clock(monkeypatch, 0)
    _answering(where, "kestrel-plain", questions=20, hits=15, seconds=200.0)
    _clock(monkeypatch, 60)
    _answering(where, "ember-plain", model="ember.gguf", questions=20, hits=11,
               seconds=90.0)
    _clock(monkeypatch, 120)
    _extraction(where, "kestrel-extract-smoke", model=KESTREL, messages=3, seconds=2.5)
    _clock(monkeypatch, 180)
    _extraction(where, "ember-extract-v1", model=EMBER, messages=24, seconds=1.5,
                node_f1=0.55, rel_cov=0.00, rel_prec=0.00, rel_f1=0.00, top_prec=0.19,
                invented=0.26, resident=3 * GIB)
    _clock(monkeypatch, 240)
    _extraction(where, "kestrel-extract-v2", model=KESTREL, messages=24, seconds=2.5,
                node_f1=0.84, rel_cov=0.70, rel_prec=0.83, rel_f1=0.76, top_prec=0.86,
                invented=0.07, resident=6 * GIB)
    monkeypatch.setattr(time, "strftime", _STRFTIME)
    return where


@pytest.fixture()
def split(store: str) -> tuple[list, list]:
    """``(answering, extraction)`` out of one store, the way `report`'s subcommand splits
    them before it hands them to the document."""
    kept = bench.runs(store)
    return ([r for r in kept if r.get("kind") != bx.KIND], bx.only(kept))


@pytest.fixture()
def measured_fit(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The memory records the subcommand reads: an empty file in ``tmp_path``, never
    ``~/.ml-stack`` and never the one that ships with the package."""
    from ml_stack.serve import fit as fit_mod

    shipped = tmp_path / "shipped.json"
    shipped.write_text("[]", encoding="utf-8")
    mine = tmp_path / "fit.json"
    mine.write_text(json.dumps([]), encoding="utf-8")
    monkeypatch.setattr(fit_mod, "package_file", lambda: shipped)
    monkeypatch.setenv("MLSTACK_FIT_FILE", str(mine))
    monkeypatch.setattr("ml_stack.hub.room", lambda: 32 * GIB)
    monkeypatch.setattr(bench, "HOME", tmp_path / "bench")


# -- reading one run ---------------------------------------------------------------------

def test_the_model_is_read_off_the_runs_own_record_before_the_servers():
    """`extract.save` writes the file it read with at the top of the run; `model_of`'s
    ``server.model`` is only the fallback, and a run naming neither says so."""
    assert extract_model_of({"model": EMBER, "server": {"model": "other.gguf"}}) == EMBER
    assert extract_model_of({"server": {"model": "other.gguf"}}) == "other.gguf"
    assert extract_model_of({}) == "?"


def test_only_the_messages_whose_gold_is_exact_are_counted(store):
    """`extract.measure` scores the template-written messages and puts the model-written
    ones in ``lower_bound``, so a run's msgs and s/msg are counted over the same set its
    coverage was."""
    one = dict(next(r for r in bx.only(bench.runs(store))
                    if r["label"] == "kestrel-extract-v2"))
    assert len(read_messages(one)) == 24
    one["rows"] = [*one["rows"][:20], *({**r, "exact": False} for r in one["rows"][20:])]
    assert len(read_messages(one)) == 20


# -- which runs are read -----------------------------------------------------------------

def test_the_runs_come_back_newest_first_with_the_short_ones_counted(split):
    _kept, extracted = split
    rows, short = extractions(extracted, min_msgs=MIN_MESSAGES)
    assert [r["label"] for r in rows] == ["kestrel-extract-v2", "ember-extract-v1"]
    assert short == 1                                    # the three-message smoke run


def test_the_floor_decides_what_is_short_enough_to_leave_out(split):
    _kept, extracted = split
    assert extractions(extracted, min_msgs=25) == ([], 3)
    rows, short = extractions(extracted, min_msgs=3)
    assert len(rows) == 3 and short == 0


def test_the_best_reader_is_the_best_relation_f1_among_the_longest_runs(split):
    """A smoke run cannot be the best reader however it scored: three messages and
    twenty-four are two measurements, and the shorter one is not ranked against the longer."""
    _kept, extracted = split
    rows, _short = extractions(extracted, min_msgs=3)
    best = best_extractor(rows)
    assert best is not None and best["label"] == "kestrel-extract-v2"
    assert best_extractor([]) is None


# -- the section -------------------------------------------------------------------------

def test_both_runs_are_tabled_newest_first_with_every_column(split):
    kept, extracted = split
    body = report(kept, extracted=extracted)
    assert "## Extraction" in body
    head = ("| run | model | msgs | s/msg | tok/msg | n-F1 | r-F1 | top-prec | rel-cov "
            "| rel-prec | invented | resident |")
    assert head in body
    # the newer reader is also the best by relation F1, so its label is the marked row
    assert ("| **kestrel-extract-v2** | `kestrel-8B (Q4_K_XL)` | 24 | 2.5 | 500 | 84% | "
            "76% | 86% | 70% | 83% | 7% | 6.00G |") in body
    assert ("| ember-extract-v1 | `ember-2B (Q4_K_M)` | 24 | 1.5 | 500 | 55% | 0% | 19% | "
            "0% | 0% | 26% | 3.00G |") in body
    lines = body.splitlines()
    newer = next(n for n, one in enumerate(lines) if "kestrel-extract-v2" in one)
    older = next(n for n, one in enumerate(lines) if "ember-extract-v1" in one)
    assert newer < older


def test_the_smoke_run_is_footnoted_and_never_tabled(split):
    kept, extracted = split
    body = report(kept, extracted=extracted)
    assert "kestrel-extract-smoke" not in body
    assert "1 smoke and short extraction run(s) left out" in body
    assert f"fewer than {MIN_MESSAGES} messages read" in body


def test_the_best_model_by_relation_f1_is_named_at_the_most_messages(split):
    kept, extracted = split
    body = report(kept, extracted=extracted)
    assert ("Best at relations: **kestrel-8B (Q4_K_XL)** at 76% relation F1 over 24 "
            "message(s) (`kestrel-extract-v2`)") in body


def test_the_topology_and_the_conformance_are_one_line_under_each_run(split):
    kept, extracted = split
    body = report(kept, extracted=extracted)
    said = [one for one in body.splitlines() if one.startswith("- `") and "topology:" in one]
    assert len(said) == 2
    for one in said:
        assert "extracted 35 nodes, 24 edges, 3 components" in one
        assert "conformance: 22/24 relations in the world's vocabulary" in one
    assert said[0].startswith("- `kestrel-extract-v2` — ")


def test_a_window_with_no_extraction_run_prints_no_section(split):
    """A heading over an empty table reads as a model that scored nothing, so a window
    holding only answering runs has no Extraction section at all."""
    kept, _extracted = split
    body = report(kept)
    assert "## Extraction" not in body
    assert "Best at relations" not in body
    assert "## Answering, per model" in body           # the rest of the document is intact


def test_extraction_runs_alone_still_make_a_document(split):
    """A store with nothing but extraction runs is not an empty store: it says what was
    read, and says the answering half was not measured rather than saying nothing."""
    _kept, extracted = split
    body = report([], extracted=extracted)
    assert "Nothing kept yet" not in body
    assert "0 run(s) and 3 extraction run(s)" in body
    assert "## Extraction" in body and "kestrel-extract-v2" in body


def test_every_extraction_run_too_short_says_so_rather_than_tabling_one(split):
    _kept, extracted = split
    body = report([], extracted=extracted, min_msgs=25)
    assert "## Extraction" in body
    assert "Nothing read at 25 message(s) or more." in body
    assert "3 smoke and short extraction run(s) left out" in body


def test_the_text_rendering_carries_the_same_numbers_without_pipes(split):
    kept, extracted = split
    body = report(kept, extracted=extracted, md=False)
    assert "|" not in body
    assert "EXTRACTION" in body
    assert "kestrel-extract-v2 *" in body               # the marked row, said in text
    assert "Best at relations: kestrel-8B (Q4_K_XL) at 76% relation F1" in body


# -- the subcommand ----------------------------------------------------------------------

def test_the_subcommand_reads_the_extraction_runs_out_of_the_same_store(store,
                                                                       measured_fit,
                                                                       capsys):
    assert bench.main(["report", "--kept", store]) == 0
    said = capsys.readouterr().out
    assert "## Extraction" in said
    assert "kestrel-extract-v2" in said and "ember-extract-v1" in said
    assert "2 run(s) and 3 extraction run(s)" in said
    # and the answering half is still the document it was
    assert "## Answering, per model" in said and "## What to serve" in said


def test_the_subcommand_narrows_the_extraction_runs_by_model_too(store, measured_fit,
                                                                 capsys):
    assert bench.main(["report", "--kept", store, "--model", "kestrel"]) == 0
    said = capsys.readouterr().out
    assert "kestrel-extract-v2" in said
    assert "ember-extract-v1" not in said


def test_a_model_with_no_extraction_run_prints_no_section(tmp_path, measured_fit, capsys):
    pytest.importorskip("ladybug")
    where = str(tmp_path / "answering.ladybug")
    _answering(where, "kestrel-plain")
    assert bench.main(["report", "--kept", where]) == 0
    said = capsys.readouterr().out
    assert "## Extraction" not in said
    assert "## Answering, per model" in said
