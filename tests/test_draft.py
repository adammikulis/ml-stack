"""``ml-stack-draft``: the arms it measures, what it prints, and what it refuses.

Nothing here serves a model. The work each arm does is a stand-in that returns rows of the
shape a real run keeps, so the arithmetic, the table and the recommendation are exercised
against numbers a person can check by hand.
"""

from __future__ import annotations

import pytest

from ml_stack import draft
from ml_stack.draft import Arm, Measured, arms_for, recommendation, table

#: How many rows one arm's totals are split over, so a sum is what is being tested.
EACH = 4


def rows(seconds: float, written: int, *, drafted: tuple[int, int] = (0, 0),
         timed: tuple[float, float] | None = None,
         passes: int | None = None) -> tuple[dict, ...]:
    """`EACH` question rows, the totals split evenly between them.

    ``drafted`` is ``(offered, kept)`` and ``timed`` is ``(drafting ms, checking ms)``;
    None for a build that does not time the two apart.
    """
    n = EACH
    offered, kept = drafted
    return tuple({
        "seconds": seconds / n, "completion_tokens": written // n,
        "draft_tokens": offered // n, "draft_taken": kept // n,
        "draft_ms": None if timed is None else timed[0] / n,
        "verify_ms": None if timed is None else timed[1] / n,
        "verify_n": None if passes is None else passes // n,
    } for _ in range(n))


class TestWhatOneArmCost:
    def test_tokens_per_second_is_written_over_wall_clock(self):
        one = Measured("head@n4", rows(20.0, 1000))
        assert one.seconds == pytest.approx(20.0)
        assert one.tokens_per_second == pytest.approx(50.0)

    def test_acceptance_is_what_the_model_kept_of_what_was_offered(self):
        one = Measured("head@n4", rows(20.0, 1000,
                                       drafted=(800, 600)))
        assert one.acceptance == pytest.approx(0.75)

    def test_no_head_has_no_acceptance_and_one_token_a_pass(self):
        one = Measured("none", rows(20.0, 1000))
        assert one.acceptance is None
        assert one.per_pass == pytest.approx(1.0)

    def test_tokens_per_pass_comes_from_the_verification_count(self):
        one = Measured("head@n4", rows(20.0, 1000, drafted=(800, 600), passes=400))
        assert one.per_pass == pytest.approx(2.5)

    def test_the_drafting_share_is_the_headline(self):
        one = Measured("head@n4", rows(20.0, 1000, drafted=(800, 600), timed=(4000, 6000)))
        assert one.drafting_share == pytest.approx(0.4)
        assert one.ms_per_drafted_token == pytest.approx(5.0)

    def test_a_build_that_does_not_time_the_two_apart_says_so_rather_than_zero(self):
        one = Measured("head@n4", rows(20.0, 1000, drafted=(800, 600)))
        assert one.drafting_share is None
        assert one.ms_per_drafted_token is None
        assert one.per_pass is None, "not measured is not one token a pass"


class TestTheArms:
    def test_no_head_is_always_the_first_arm(self):
        every = arms_for("mtp-head.gguf", depths=(2, 4))
        assert every[0].label == "none"
        assert every[0].undrafted
        assert [one.label for one in every[1:]] == ["head@n2", "head@n4"]

    def test_each_depth_reaches_the_run_as_both_halves(self):
        every = arms_for("mtp-head.gguf", depths=(8,), spec_type="draft-mtp")
        assert every[1].over == {"draft": "mtp-head.gguf", "spec_type": "draft-mtp",
                                 "draft_n_max": 8}

    def test_the_baseline_takes_the_head_away_rather_than_leaving_it_unsaid(self):
        every = arms_for("mtp-head.gguf", depths=(2,))
        assert every[0].over["draft"] == ""
        assert every[0].over["draft_n_max"] is None

    def test_the_heads_own_cache_is_measured_at_one_depth(self):
        every = draft._cache_arms("mtp-head.gguf", depth=4, caches=("q8_0", "q4_0"),
                                  spec_type="draft-mtp")
        assert [one.label for one in every] == ["head@n4-kv-q8_0", "head@n4-kv-q4_0"]
        assert every[0].over["draft_cache_type"] == "q8_0"
        assert every[0].over["draft_n_max"] == 4

    def test_an_arm_lays_over_a_run_without_touching_the_rest_of_it(self):
        from ml_stack.serve.shape import Run, Shape

        run = Run(shape=Shape(model="m.gguf", seat_context=32768, cache_type="q8_0"))
        arm = Arm("head@n8", {"draft": "h.gguf", "spec_type": "draft-mtp",
                              "draft_n_max": 8})
        over = run.over(**dict(arm.over))
        assert over.shape.draft == "h.gguf"
        assert over.shape.seat_context == 32768
        assert over.shape.cache_type == "q8_0"


class TestTheTable:
    def _measured(self):
        return [
            Measured("none", rows(40.0, 1000)),
            Measured("head@n4", rows(20.0, 1000, drafted=(800, 600), timed=(4000, 6000),
                                     passes=400)),
        ]

    def test_every_column_names_its_unit_and_which_way_is_better(self):
        said = "\n".join(table(self._measured()))
        for word in ("tok/s", "accept", "tok/pass", "drafting", "ms/draft", "wall",
                     "vs none"):
            assert word in said
        assert said.count("higher is better") >= 3
        assert said.count("lower is better") >= 2

    def test_the_fastest_arm_is_first_and_carries_its_speedup(self):
        lines = table(self._measured())
        assert lines[2].strip().startswith("head@n4")
        assert "2.00x" in lines[2]
        assert "1.00x" in lines[3]

    def test_a_busy_machine_marks_every_row_and_says_what_that_means(self):
        busy = [Measured(one.label, one.rows, quiet=False) for one in self._measured()]
        said = "\n".join(table(busy))
        assert said.count(" NO") >= 2
        assert "not this model's" in said

    def test_a_build_that_cannot_time_the_split_says_why_the_column_is_empty(self):
        blind = [Measured("none", rows(40.0, 1000)),
                 Measured("head@n4", rows(20.0, 1000, drafted=(800, 600)))]
        said = "\n".join(table(blind))
        assert "0002-speculative-timings.patch" in said

    def test_nothing_measured_says_so(self):
        assert table([]) == ["nothing measured"]


class TestWhatToServe:
    def test_the_fastest_arm_becomes_a_line_to_paste(self):
        said = recommendation(
            [Measured("none", rows(40.0, 1000)),
             Measured("head@n8", rows(20.0, 1000, drafted=(800, 600)))],
            model="/models/marrowgate-Q4.gguf", head="/models/mtp-marrowgate.gguf")
        assert "serve head@n8" in said[0]
        assert said[1].strip() == ("ml-stack-serve up marrowgate-Q4.gguf --draft "
                                   "mtp-marrowgate.gguf --spec-n-max 8")

    def test_a_head_that_is_slower_than_none_is_not_recommended(self):
        said = recommendation(
            [Measured("none", rows(20.0, 1000)),
             Measured("head@n8", rows(40.0, 1000, drafted=(800, 100)))],
            model="/models/marrowgate-Q4.gguf", head="/models/mtp-marrowgate.gguf")
        assert said[0].startswith("serve no head")
        assert said[1].strip() == "ml-stack-serve up marrowgate-Q4.gguf"

    def test_the_heads_cache_type_reaches_the_line(self):
        said = recommendation(
            [Measured("none", rows(40.0, 1000)),
             Measured("head@n4-kv-q4_0", rows(20.0, 1000, drafted=(800, 600)))],
            model="/models/marrowgate-Q4.gguf", head="/models/mtp-marrowgate.gguf",
            build="unsloth")
        assert "--spec-n-max 4 --draft-kv q4_0 --build unsloth" in said[1]

    def test_without_a_baseline_it_recommends_nothing(self):
        said = recommendation([Measured("head@n4", rows(20.0, 1000))],
                              model="m.gguf", head="h.gguf")
        assert "no baseline" in said[0]


class TestTheCommand:
    def test_a_model_with_no_head_stops_rather_than_measuring_nothing(self, monkeypatch,
                                                                     capsys):
        monkeypatch.setattr(draft, "located", lambda m: None)
        monkeypatch.setattr(draft, "build_for", lambda model, asked: "")
        monkeypatch.setattr(draft, "_head_for",
                            lambda model, asked, build: ("", "no head shipped"))
        assert draft.main(["marrowgate-Q4.gguf"]) == 1
        said = capsys.readouterr()
        assert "no draft head" in said.out + said.err
        assert "nothing to measure" in said.out + said.err

    def test_a_busy_machine_refuses_before_anything_is_served(self, monkeypatch, capsys):
        from ml_stack.bench.quiet import Quiet

        served = []
        monkeypatch.setattr(draft, "located", lambda m: None)
        monkeypatch.setattr(draft, "build_for", lambda model, asked: "")
        monkeypatch.setattr(draft, "_head_for",
                            lambda model, asked, build: ("mtp-head.gguf", "beside it"))
        monkeypatch.setattr(draft, "look",
                            lambda: Quiet(reasons=("something else holds the card",)))
        monkeypatch.setattr(draft, "measure_arm",
                            lambda *a, **k: served.append(a) or Measured("x"))
        assert draft.main(["marrowgate-Q4.gguf"]) == 3
        assert served == [], "nothing was served"
        printed = capsys.readouterr()
        said = printed.out + printed.err
        assert "not quiet" in said and "something else holds the card" in said

    def test_anyway_measures_and_marks_every_row(self, monkeypatch, capsys):
        from ml_stack.bench.quiet import Quiet

        asked = []

        def one_arm(run, arm, work, *, kept, quiet):
            asked.append((arm.label, quiet))
            return Measured(arm.label, rows(10.0, 200), quiet=quiet)

        monkeypatch.setattr(draft, "located", lambda m: None)
        monkeypatch.setattr(draft, "build_for", lambda model, asked_: "")
        monkeypatch.setattr(draft, "_head_for",
                            lambda model, asked_, build: ("mtp-head.gguf", "beside it"))
        monkeypatch.setattr(draft, "look",
                            lambda: Quiet(reasons=("something else holds the card",)))
        monkeypatch.setattr(draft, "measure_arm", one_arm)
        monkeypatch.setattr(draft, "_asking_work", lambda *a, **k: (lambda *b: []))
        assert draft.main(["marrowgate-Q4.gguf", "--anyway", "--depth", "4"]) == 0
        assert [label for label, _ in asked] == ["none", "head@n4"]
        assert all(quiet is False for _, quiet in asked)
        assert " NO" in capsys.readouterr().out

    def test_for_ingest_without_a_document_says_what_to_do(self, monkeypatch, capsys):
        from ml_stack.bench.quiet import Quiet

        monkeypatch.setattr(draft, "located", lambda m: None)
        monkeypatch.setattr(draft, "build_for", lambda model, asked: "")
        monkeypatch.setattr(draft, "_head_for",
                            lambda model, asked, build: ("mtp-head.gguf", "beside it"))
        monkeypatch.setattr(draft, "look", lambda: Quiet())
        assert draft.main(["marrowgate-Q4.gguf", "--for", "ingest"]) == 2
        assert "needs a document" in capsys.readouterr().err
