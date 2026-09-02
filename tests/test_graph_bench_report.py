"""``ml-stack-bench report``: every measurement arranged as one document.

The report is composed rather than measured, so what is tested here is the arranging: that
runs group by the model they were served from, that a run served with a bound reasoning
budget is told from one served without, that a smoke run is counted in a footnote instead of
tabled beside a full run, and that the serving line at the end says "not measured" for a
part nothing measured rather than filling it in.

Everything is built in ``tmp_path``: runs through `bench.save` with `Row` fixtures, memory
records through the `fit` seams. Nothing here reads ``~/.ml-stack``, serves a model or
touches a GPU.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ml_stack.graph import bench
from ml_stack.graph.bench import Row
from ml_stack.graph.bench.report import (
    across,
    answering,
    asking_of,
    cache_of,
    head_of,
    model_of,
    recommended_head,
    report,
    thinking_of,
)
from ml_stack.serve.fit import Fit

GIB = 1024 ** 3


def _rows(label: str, *, questions: int, hits: int, seconds: float,
          guessed: int = 0, taken: int = 0) -> list[Row]:
    """``hits`` of ``questions`` answered in full, the rest showing nothing, over
    ``seconds`` altogether -- so the F1 of a run is ``hits / questions`` exactly."""
    out = []
    for n in range(questions):
        one = Row(label=label, question=f"who runs the kiln, question {n}?",
                  expected=["person:marisol-quen"],
                  shown=["person:marisol-quen"] if n < hits else [],
                  seconds=seconds / questions, calls=3, answer_chars=200,
                  processed_tokens=500, completion_tokens=100)
        one.draft_tokens, one.draft_taken = guessed, taken
        out.append(one)
    return out


def _keep(store, label, *, model="flash.gguf", questions=20, hits=12, seconds=200.0,
          draft="", ahead=0, cache="", budget=None, guessed=0, taken=0) -> str:
    held = {"model": model, "binary": "/builds/current/llama-server",
            "context": 32768, "slots": 1, "graph": "invented"}
    if draft:
        held["draft_model"] = draft
        held["spec_draft_max"] = ahead
    if cache:
        held["cache_type"] = cache
    if budget is not None:
        held["reasoning_budget"] = budget
    return bench.save(store, _rows(label, questions=questions, hits=hits, seconds=seconds,
                                   guessed=guessed, taken=taken), held=held)


@pytest.fixture()
def store(tmp_path: pathlib.Path) -> str:
    """One store with two models in it: a flash measured five ways, one of them with a
    draft head and one of them a smoke run, and a smaller one measured two ways."""
    where = str(tmp_path / "runs.ladybug")
    _keep(where, "flash-plain", questions=24, hits=18, seconds=240.0)
    _keep(where, "flash-plain-terse", questions=24, hits=15, seconds=180.0)
    _keep(where, "flash-plain-kv-q8_0", questions=24, hits=14, seconds=170.0, cache="q8_0")
    _keep(where, "draft:none", questions=20, hits=12, seconds=200.0)
    _keep(where, "draft:mtp-alder@n4", questions=20, hits=12, seconds=140.0,
          draft="mtp-alder.gguf", ahead=4, guessed=100, taken=76)
    _keep(where, "flash-plain-smoke", questions=2, hits=1, seconds=20.0)
    _keep(where, "tiny-plain", model="tiny.gguf", questions=20, hits=10, seconds=100.0,
          budget=0)
    _keep(where, "tiny-plain-card", model="tiny.gguf", questions=20, hits=8, seconds=90.0,
          budget=512)
    return where


@pytest.fixture()
def fits() -> list[Fit]:
    """Two measured models, in a 32 GiB room."""
    return [Fit(model="flash.gguf", weights=4 * GIB, compute=GIB // 2, room=32 * GIB,
                per_token=64 * 1024, per_seq=1024),
            Fit(model="tiny.gguf", weights=GIB, compute=GIB // 4, room=32 * GIB,
                per_token=16 * 1024, per_seq=1024)]


# -- reading one run ---------------------------------------------------------------------

@pytest.mark.parametrize(("label", "expect"), [
    ("flash-plain", "plain"),
    ("gemma-4-E2B-it-plain-terse", "plain+terse"),
    ("thing-shortlist-card", "shortlist+card"),
    ("draft:mtp-alder@n4", "-"),
    ("flash-plain-kv-q8_0", "plain"),
    ("tightfit-plain", "plain"),          # by whole word: `tightfit` is not `tight`
])
def test_the_asking_is_read_out_of_the_label_by_whole_word(label, expect):
    assert asking_of(label) == expect


def test_thinking_is_on_off_or_the_budget_itself():
    """A budget of zero is thinking turned off; no budget at all is thinking left alone,
    and the two are not the same configuration."""
    assert thinking_of({}) == "on"
    assert thinking_of({"reasoning_budget": 0}) == "off"
    assert thinking_of({"reasoning_budget": 512}) == "512"


def test_the_cache_column_is_blank_at_f16_and_short_otherwise():
    assert cache_of({}) == "-"
    assert cache_of({"cache_type": "f16"}) == "-"
    assert cache_of({"cache_type": "q8_0"}) == "q8"


def test_a_head_carries_how_far_it_guessed():
    assert head_of({"server": {}}) == "-"
    assert head_of({"server": {"draft_model": "mtp-alder.gguf",
                               "spec_draft_max": 4}}) == "mtp-alder.gguf@n4"


def test_a_run_is_grouped_by_the_model_it_was_served_from():
    assert model_of({"server": {"model": "flash.gguf"}}) == "flash.gguf"
    assert model_of({}) == "?"


# -- the tables --------------------------------------------------------------------------

def test_runs_group_by_model_and_the_best_f1_is_first(store):
    kept = bench.runs(store)
    tables = answering(kept, min_n=6)
    assert sorted(tables) == ["flash.gguf", "tiny.gguf"]
    rows, short = tables["flash.gguf"]
    assert [r["label"] for r in rows][0] == "flash-plain"      # 75%, the best
    assert short == 1                                          # the two-question smoke run
    assert all(r["label"] != "flash-plain-smoke" for r in rows)


def test_a_short_run_is_counted_in_a_footnote_and_not_tabled(store):
    body = report(bench.runs(store), min_n=6)
    assert "flash-plain-smoke" not in body
    assert "1 smoke and short run(s) of flash.gguf left out" in body


def test_min_n_decides_what_is_short_enough_to_leave_out(store):
    """At a floor of 21 the twenty-question runs go to the footnote too -- which is the
    flag's whole purpose: read only what was asked at length."""
    tables = answering(bench.runs(store), min_n=21)
    rows, short = tables["flash.gguf"]
    assert {r["label"] for r in rows} == {"flash-plain", "flash-plain-terse",
                                          "flash-plain-kv-q8_0"}
    assert short == 3
    assert answering(bench.runs(store), min_n=21)["tiny.gguf"] == ([], 2)


def test_across_models_takes_each_models_longest_run(store):
    """flash's longest run is 24 questions, so its twenty-question drafted run cannot be
    what it is ranked by -- twenty scored questions and twenty-four are two measurements."""
    ranked = across(bench.runs(store))
    assert [model for model, _ in ranked] == ["flash.gguf", "tiny.gguf"]
    assert ranked[0][1]["label"] == "flash-plain"
    assert ranked[1][1]["label"] == "tiny-plain"


def test_full_n_holds_every_model_to_the_same_number_of_questions(store):
    """At a floor of 24 the smaller model has no run long enough and drops out rather than
    being ranked on a shorter one."""
    ranked = across(bench.runs(store), full_n=24)
    assert [model for model, _ in ranked] == ["flash.gguf"]


def test_the_answering_table_names_every_way_it_was_asked(store):
    body = report(bench.runs(store), min_n=6)
    assert "## Answering, per model" in body
    assert "### `flash.gguf`" in body and "### `tiny.gguf`" in body
    # the way, the thinking, the cache and the head are each a column
    assert "| plain | on | q8 | - | 24 |" in body
    assert "| - | on | - | mtp-alder.gguf@n4 | 20 |" in body
    assert "| **plain** | off | - | - | 20 |" in body       # tiny, thinking at zero
    assert "| plain+card | 512 | - | - | 20 |" in body


def test_the_best_row_of_a_model_is_marked(store):
    body = report(bench.runs(store), min_n=6)
    assert "| **plain** | on | - | - | 24 | 10.0 | 75% |" in body


# -- draft heads -------------------------------------------------------------------------

def test_the_drafts_block_is_the_drafts_summary_for_the_model_that_has_one(store):
    body = report(bench.runs(store), min_n=6)
    assert "## Draft heads, per model" in body
    assert "draft:mtp-alder@n4" in body
    assert "76%" in body                                   # what the head guessed and kept
    assert "serve draft:mtp-alder@n4" in body
    # the model with no head measured gets no block of its own
    heads = body.split("## Draft heads, per model", 1)[1].split("## Memory", 1)[0]
    assert "`flash.gguf`" in heads and "`tiny.gguf`" not in heads


def test_no_draft_head_measured_says_how_to_measure_one(tmp_path):
    where = str(tmp_path / "runs.ladybug")
    _keep(where, "flash-plain")
    body = report(bench.runs(where), min_n=6)
    assert "No draft head measured" in body


def test_the_recommended_head_is_the_one_the_drafts_summary_names(store):
    kept = bench.runs(store)
    flash = [r for r in kept if model_of(r) == "flash.gguf"]
    chosen = recommended_head(flash, kept)
    assert chosen is not None and chosen["label"] == "draft:mtp-alder@n4"
    assert recommended_head([r for r in kept if model_of(r) == "tiny.gguf"], kept) is None


# -- memory ------------------------------------------------------------------------------

def test_the_memory_block_says_who_fits_at_the_context_asked_about(store, fits):
    body = report(bench.runs(store), fits=fits, at=32768, room="32.0G")
    assert "## Memory" in body
    assert "| users at 32,768 |" in body
    # 32G room less 4.5G loaded, 2G per user at 32k
    assert f"| {fits[0].users(32768)} |" in body
    assert "per token of context" in body                   # the full record, rendered


def test_another_room_is_answered_beside_this_machine(store, fits):
    body = report(bench.runs(store), fits=fits,
                  elsewhere=[("16.0G", [f.at_room(16 * GIB) for f in fits])], at=32768)
    assert "### A machine with 16.0G" in body
    assert str(fits[0].at_room(16 * GIB).users(32768)) in body


def test_nothing_measured_for_memory_says_how_to_measure_it(store):
    body = report(bench.runs(store))
    assert "ml-stack-serve fit MODEL --measure" in body


# -- the serving line --------------------------------------------------------------------

def test_the_serving_line_composes_the_asking_the_head_and_the_memory(store, fits):
    body = report(bench.runs(store), fits=fits, at=32768)
    line = next(one for one in body.splitlines() if one.startswith("- `flash.gguf`"))
    assert "serve with: plain" in line
    assert "thinking on" in line
    assert "head mtp-alder.gguf@n4" in line
    assert "cache f16" in line
    # the cost is the drafted run's -- 140s over 20 questions -- and the accuracy the
    # longest run's, which is how `choices` composes a model too
    assert "expect ~7.0 s/q at F1 75% (n=24)" in line
    assert f"{fits[0].users(32768)} users at 32,768 tokens on this machine" in line


def test_the_serving_line_says_not_measured_rather_than_guessing(store):
    body = report(bench.runs(store))
    line = next(one for one in body.splitlines() if one.startswith("- `tiny.gguf`"))
    assert "thinking off" in line
    assert "head not measured" in line
    assert "users at 32,768 tokens: not measured" in line


# -- the two renderings ------------------------------------------------------------------

def test_the_text_rendering_has_no_pipes_and_the_same_numbers(store, fits):
    body = report(bench.runs(store), fits=fits, md=False)
    assert "|" not in body
    assert "ANSWERING, PER MODEL" in body
    assert "flash-plain" not in body.split("WHAT TO SERVE")[0].split("DRAFT HEADS")[0]
    assert "expect ~7.0 s/q at F1 75% (n=24)" in body


def test_an_empty_store_says_what_to_run(tmp_path):
    body = report([])
    assert "Nothing kept yet" in body
    assert "ml-stack-bench sweep" in body


# -- the subcommand ----------------------------------------------------------------------

@pytest.fixture()
def measured_fit(tmp_path, monkeypatch, fits):
    """The fit records the subcommand reads, in ``tmp_path`` -- never ``~/.ml-stack`` and
    never the file that ships with the package."""
    from ml_stack.serve import fit as fit_mod

    shipped = tmp_path / "shipped.json"
    shipped.write_text("[]", encoding="utf-8")
    mine = tmp_path / "fit.json"
    mine.write_text(json.dumps([f.as_dict() for f in fits]), encoding="utf-8")
    monkeypatch.setattr(fit_mod, "package_file", lambda: shipped)
    monkeypatch.setenv("MLSTACK_FIT_FILE", str(mine))
    monkeypatch.setattr("ml_stack.hub.room", lambda: 32 * GIB)
    monkeypatch.setattr(bench, "HOME", tmp_path / "bench")
    return mine


def test_the_subcommand_prints_the_document(store, measured_fit, capsys):
    assert bench.main(["report", "--kept", store]) == 0
    said = capsys.readouterr().out
    assert "# What has been measured" in said
    assert "## What to serve" in said
    assert "users at 32,768 tokens on this machine" in said


def test_the_subcommand_narrows_to_one_model(store, measured_fit, capsys):
    assert bench.main(["report", "--kept", store, "--model", "tiny"]) == 0
    said = capsys.readouterr().out
    assert "tiny.gguf" in said and "flash.gguf" not in said


def test_md_writes_the_file_and_open_opens_it(store, measured_fit, tmp_path, capsys,
                                              monkeypatch):
    opened = []
    monkeypatch.setattr("ml_stack.platform.open_path",
                        lambda where: opened.append(str(where)) or "open")
    out = tmp_path / "written" / "report.md"
    assert bench.main(["report", "--kept", store, "--md", str(out), "--open"]) == 0
    assert "# What has been measured" in out.read_text()
    assert opened == [str(out)]
    assert f"wrote {out}" in capsys.readouterr().out


def test_text_renders_without_markdown(store, measured_fit, capsys):
    assert bench.main(["report", "--kept", store, "--text"]) == 0
    said = capsys.readouterr().out
    assert "|" not in said
    assert "WHAT HAS BEEN MEASURED" in said


def test_a_room_that_cannot_be_read_is_refused_rather_than_guessed(store, measured_fit,
                                                                   capsys):
    assert bench.main(["report", "--kept", store, "--room", "twenty-four"]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_a_room_is_answered_beside_this_machine(store, measured_fit, capsys):
    assert bench.main(["report", "--kept", store, "--room", "16G"]) == 0
    assert "### A machine with 16.0G" in capsys.readouterr().out


def test_report_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as left:
        bench.main(["report", "--help"])
    assert left.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_a_model_named_run_is_not_mistaken_for_the_run_subcommand(store, measured_fit):
    """``--model run`` used to be seen as the ``run`` subcommand by the word-scan that
    decides whether to take the measuring lock, which would have served a model."""
    assert bench.main(["report", "--kept", store, "--model", "run"]) == 0
