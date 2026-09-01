"""What a run kept says when it is read back.

Every fixture here is invented. Nothing reads a real store, a real graph, or a real server.
"""

from __future__ import annotations

import pathlib

import pytest

from ml_stack.graph.bench import Row, _hit, missed, runs, save, table


def a_row(question: str, *, expected: list[str], shown: list[str], calls: int = 3,
          chars: int = 200, error: str = "") -> Row:
    return Row(label="tried", question=question, expected=expected, shown=shown,
               calls=calls, answer_chars=chars, error=error)


def test_hit_is_how_much_of_what_was_wanted_was_shown():
    assert _hit({"expected": ["person:iris"], "shown": ["person:iris"]}) == 1.0
    assert _hit({"expected": ["person:iris", "person:otto"], "shown": ["person:iris"]}) == 0.5
    assert _hit({"expected": ["person:iris"], "shown": []}) == 0.0
    assert _hit({"expected": ["person:iris"], "shown": ["topic:welding"]}) == 0.0
    # showing more than was asked for is not punished: the question is whether it was found
    assert _hit({"expected": ["person:iris"], "shown": ["person:iris", "org:pellard"]}) == 1.0
    assert _hit({"expected": [], "shown": ["person:iris"]}) == -1.0    # nothing to score


def test_row_hit_and_the_kept_form_agree():
    """`Row.hit` and `_hit` read the same row, one live and one out of a store."""
    row = a_row("who welds?", expected=["person:iris", "person:otto"], shown=["person:otto"])
    from dataclasses import asdict

    assert row.hit == _hit(asdict(row)) == 0.5


def test_missed_prints_the_questions_that_fell_short_and_not_the_others(capsys):
    kept = [{"label": "tried", "at": "2026-01-01T00:00:00", "rows": [
        {"question": "who welds?", "expected": ["person:iris"], "shown": ["person:iris"],
         "calls": 4, "answer_chars": 300},
        {"question": "who sells?", "expected": ["person:otto"], "shown": [],
         "calls": 2, "answer_chars": 90},
    ]}]
    missed(kept)
    said = capsys.readouterr().out
    assert "who sells?" in said
    assert "who welds?" not in said              # a question answered in full is not a miss
    assert "person:otto" in said and "(nothing)" in said
    assert "2 calls, 90 chars" in said           # the cost is what explains the miss

    missed(kept, everything=True)
    assert "who welds?" in capsys.readouterr().out


def test_missed_says_so_when_a_run_answered_everything(capsys):
    missed([{"label": "tried", "rows": [
        {"question": "who welds?", "expected": ["person:iris"], "shown": ["person:iris"]}]}])
    assert "every question answered in full" in capsys.readouterr().out


def test_missed_shows_an_error_and_skips_unscored_rows(capsys):
    missed([{"label": "tried", "rows": [
        {"question": "hello", "expected": [], "shown": []},          # not a scored question
        {"question": "who welds?", "expected": ["person:iris"], "shown": [],
         "error": "TimeoutError: no reply"},
    ]}])
    said = capsys.readouterr().out
    assert "hello" not in said
    assert "TimeoutError: no reply" in said


def test_missed_on_nothing(capsys):
    missed([])
    assert "nothing kept yet" in capsys.readouterr().out


def test_a_run_kept_reads_back_as_it_was_written(tmp_path):
    store = tmp_path / "runs.ladybug"
    rows = [a_row("who welds?", expected=["person:iris"], shown=["person:iris"]),
            a_row("who sells?", expected=["person:otto"], shown=[])]
    key = save(store, rows, held={"context": 32768, "slots": 2})
    assert key.startswith("bench:tried:")

    back = runs(store)
    assert len(back) == 1
    assert back[0]["label"] == "tried"
    assert back[0]["server"]["context"] == 32768
    assert [r["question"] for r in back[0]["rows"]] == ["who welds?", "who sells?"]
    assert [_hit(r) for r in back[0]["rows"]] == [1.0, 0.0]

    assert runs(store, "tried") == back
    assert runs(store, "never-run") == []        # a label nobody used is empty, not an error


def test_table_reads_a_kept_run(tmp_path, capsys):
    store = tmp_path / "runs.ladybug"
    save(store, [a_row("who welds?", expected=["person:iris"], shown=["person:iris"])],
         held={"context": 32768, "slots": 2, "kv_and_run_bytes": 2 * 2**30,
               "bytes_per_1k_context": 32 * 2**20})
    table(runs(store))
    said = capsys.readouterr().out
    assert "tried" in said
    assert "32k x2" in said                      # the context is on the line, always
    assert "100%" in said
    assert "2.00G" in said and "32.0M" in said


def test_the_table_says_how_many_questions_a_run_answered(tmp_path, capsys):
    """A run of ten against a run of nine is two measurements, not one comparison.

    Adding a question changes every score after it. Without the count on the line the two
    sit adjacent looking comparable, which is how an 85% got read against an earlier 72%
    when the only difference was a question that had not existed before.
    """
    store = tmp_path / "runs.ladybug"
    save(store, [a_row(f"q{n}?", expected=["person:iris"], shown=["person:iris"])
                 for n in range(9)], held={"context": 32768, "slots": 2})
    save(store, [a_row(f"q{n}?", expected=["person:iris"], shown=["person:iris"])
                 for n in range(10)], held={"context": 32768, "slots": 2})

    table(runs(store))
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("tried")]
    assert len(lines) == 2
    # "tried  32k x2  9 ..." -- the context field carries a space, so n is the fourth word
    assert lines[0].split()[3] == "9"
    assert lines[1].split()[3] == "10"


def test_table_on_nothing(capsys):
    table([])
    assert "nothing kept yet" in capsys.readouterr().out


@pytest.mark.parametrize("scores,stands", [([0.9, 0.5], True), ([0.9, 0.89], False)])
def test_the_bench_gate_is_the_shared_one(scores, stands):
    """`ml-stack-bench` must not grow a second copy of the margin test."""
    from ml_stack.graph.bench import MARGIN, stands_out
    from ml_stack.graph.vectors import MARGIN as SOURCE, stands_out as origin

    assert MARGIN is SOURCE and stands_out is origin
    assert stands_out(scores) is stands


def test_two_runs_in_one_second_do_not_replace_each_other(tmp_path):
    """A run took minutes when the key was a timestamp. Cached, it takes no time at all."""
    store = tmp_path / "runs.ladybug"
    first = save(store, [a_row("q?", expected=["person:iris"], shown=["person:iris"])])
    second = save(store, [a_row("q?", expected=["person:iris"], shown=[])])
    assert first != second
    kept_runs = runs(store, "tried")
    assert len(kept_runs) == 2
    assert [_hit(r["rows"][0]) for r in kept_runs] == [1.0, 0.0]     # both, in order


def test_the_table_records_the_sampling_and_the_draft_a_run_used(tmp_path, capsys):
    """A run at one temperature against a run at another is two measurements.

    Same lesson as `ctx` and `n`: the only way to know later is to write it down now, and a
    column is cheaper than remembering which afternoon a setting changed.
    """
    store = tmp_path / "runs.ladybug"
    row = a_row("who?", expected=["person:iris"], shown=["person:iris"])
    row.draft_tokens, row.draft_taken = 54, 41
    save(store, [row], held={"context": 32768, "slots": 4,
                             "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}})
    table(runs(store))
    said = capsys.readouterr().out
    assert "t1 p.95 k64" in said
    assert "76%" in said                       # 41 of 54 guesses kept
    assert "sampling" in said                  # and the column is named


def test_a_run_with_no_draft_and_no_recorded_sampling_says_so(tmp_path, capsys):
    from ml_stack.graph.bench import drafting, sampled

    assert drafting([{"draft_tokens": 0, "draft_taken": 0}]) == "-"
    assert drafting([{"draft_tokens": 10, "draft_taken": 5}]) == "50%"
    assert sampled({}) == "-"
    assert sampled({"sampling": {}}) == "-"
    assert sampled({"sampling": {"temperature": 0.0}}) == "t0"


def test_sampling_overrides_are_only_what_was_asked_for():
    """Anything not given falls through to the model's own card, rather than being defaulted
    here — which would quietly overrule a publisher."""
    from argparse import Namespace

    from ml_stack.graph.bench import sampling_from

    assert sampling_from(Namespace(temperature=None, top_p=None, top_k=None, min_p=None)) == {}
    assert sampling_from(Namespace(temperature=0.0, top_p=None, top_k=None, min_p=None)) \
        == {"temperature": 0.0}                # nought is a choice, not an absence
    assert sampling_from(Namespace(temperature=1.0, top_p=0.95, top_k=64, min_p=None)) \
        == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}


def test_derived_puts_accuracy_over_each_scarcity():
    """A score alone cannot choose between a model that is better and one that is cheaper."""
    from ml_stack.graph.bench import derived

    run = {"server": {"kv_and_run_bytes": 2 * 2**30},
           "rows": [{"expected": ["a"], "shown": ["a"], "seconds": 10.0, "calls": 2,
                     "processed_tokens": 800, "completion_tokens": 200},
                    {"expected": ["b"], "shown": [], "seconds": 10.0, "calls": 2,
                     "processed_tokens": 800, "completion_tokens": 200}]}
    d = derived(run)
    assert d["right"] == 0.5 and d["questions"] == 2
    assert d["seconds"] == 20.0 and d["paid_tokens"] == 2000
    assert d["right_per_minute"] == pytest.approx(0.5 * 60 / 20)
    assert d["right_per_1k"] == pytest.approx(0.5 * 1000 / 2000)
    assert d["right_per_gb"] == pytest.approx(0.25)
    # one right answer out of two questions took all twenty seconds and all two thousand
    assert d["seconds_per_right"] == pytest.approx(20.0)
    assert d["tokens_per_right"] == pytest.approx(2000.0)


def test_derived_on_a_run_that_got_nothing_right_divides_by_nothing():
    from ml_stack.graph.bench import derived

    d = derived({"rows": [{"expected": ["a"], "shown": [], "seconds": 5.0,
                           "processed_tokens": 100, "completion_tokens": 10}]})
    assert d["right"] == 0.0
    assert d["right_per_minute"] == 0.0            # a rate of nothing is still a rate
    assert "seconds_per_right" not in d            # but the inverse has no meaning
    assert derived({"rows": []}) == {}


def test_the_frontier_keeps_only_what_nothing_beats_on_both_axes():
    from ml_stack.graph.bench import pareto

    def run(label, right, seconds):
        rows = [{"expected": ["a"], "shown": ["a"] if n < right else [], "seconds": seconds / 10,
                 "processed_tokens": 10, "completion_tokens": 1} for n in range(10)]
        return {"label": label, "rows": rows, "server": {}}

    fast = run("fast", 5, 10.0)          # 50% in 10s
    good = run("good", 9, 100.0)         # 90% in 100s
    both = run("both", 7, 50.0)          # 70% in 50s -- beaten by neither
    worse = run("worse", 5, 80.0)        # 50% in 80s -- fast is as good and quicker

    on = [str(o["label"]) for o in pareto([fast, good, both, worse], cost="seconds")]
    assert on == ["fast", "both", "good"]          # cheapest first
    assert "worse" not in on


def test_the_frontier_can_be_drawn_against_tokens_instead_of_time():
    from ml_stack.graph.bench import pareto

    def run(label, right, tokens):
        return {"label": label, "server": {},
                "rows": [{"expected": ["a"], "shown": ["a"] if n < right else [],
                          "seconds": 1.0, "processed_tokens": tokens // 10,
                          "completion_tokens": 0} for n in range(10)]}

    thrifty, lavish = run("thrifty", 8, 1000), run("lavish", 8, 9000)
    on = [str(o["label"]) for o in pareto([thrifty, lavish], cost="paid_tokens")]
    assert on == ["thrifty"]             # same accuracy, nine times the tokens


def test_the_plot_is_self_contained_and_names_what_it_drew(tmp_path):
    """It has to open on a machine with no network and no packages."""
    from ml_stack.graph.bench import plot

    where = plot([{"label": "tried", "server": {"kv_and_run_bytes": 2**30},
                   "rows": [{"expected": ["a"], "shown": ["a"], "seconds": 5.0,
                             "processed_tokens": 100, "completion_tokens": 20}]}],
                 tmp_path / "f.html")
    said = pathlib.Path(where).read_text()
    assert "<svg" in said and "tried" in said
    assert "http://" not in said and "https://" not in said     # nothing to fetch
    assert "<script" not in said
