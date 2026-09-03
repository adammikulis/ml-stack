"""The row a model's profile is written from: the fastest whose F1 held, where held is a
claim about the difference the questions measured -- and never a smoke."""

from ml_stack.graph.bench import invented_digest, runs, save
from ml_stack.graph.bench.keep import SHORT
from ml_stack.graph.bench.report import measured_best
from ml_stack.graph.bench.score import held_up, separated
from tests.test_graph_bench import scored_rows


def _run(store, label, *, questions, hits, seconds, model="flash.gguf"):
    rows = scored_rows(label, questions=questions, hits=hits, seconds=seconds)
    return save(store, rows, held={"graph": invented_digest(), "model": model})


def test_fifteen_points_over_a_hundred_questions_is_a_measurement(tmp_path):
    """80% against 65% over a hundred right-or-wrong questions: each band is about nine
    points wide either side, the two overlap, and the difference is still measured --
    its own interval is the two spreads in quadrature, and fifteen points is outside it.

    This is the pick that wrote a shape ten points worse (80% ±6 against 70% ±6 on
    per-question F1, whose bands are narrower than these) into a profile the evening the
    intervals were compared by overlap."""
    store = tmp_path / "runs.ladybug"
    _run(store, "best", questions=100, hits=80, seconds=2700.0)
    _run(store, "faster-worse", questions=100, hits=65, seconds=2500.0)
    by_label = {r["label"]: r for r in runs(store)}
    assert separated(by_label["faster-worse"], by_label["best"]) is True
    assert not held_up(by_label["faster-worse"], by_label["best"])
    assert measured_best(list(by_label.values()))["label"] == "best", \
        "the cheaper row did not hold, so the record is set from the accurate one"


def test_a_difference_the_questions_cannot_see_still_goes_to_the_cheaper_row(tmp_path):
    store = tmp_path / "runs.ladybug"
    _run(store, "best", questions=100, hits=80, seconds=2700.0)
    _run(store, "faster-close", questions=100, hits=78, seconds=2400.0)
    by_label = {r["label"]: r for r in runs(store)}
    assert separated(by_label["faster-close"], by_label["best"]) is False
    assert measured_best(list(by_label.values()))["label"] == "faster-close"


def test_a_profile_is_never_set_from_a_smoke(tmp_path):
    """A 27B once got its record from a two-question row: whichever way the coin fell."""
    store = tmp_path / "runs.ladybug"
    _run(store, "smoke", questions=2, hits=1, seconds=180.0, model="big-27B.gguf")
    _run(store, "short", questions=SHORT - 1, hits=10, seconds=900.0, model="big-27B.gguf")
    mine = [r for r in runs(store) if r["server"]["model"] == "big-27B.gguf"]
    assert measured_best(mine) is None
    _run(store, "long-enough", questions=SHORT, hits=12, seconds=1000.0, model="big-27B.gguf")
    mine = [r for r in runs(store) if r["server"]["model"] == "big-27B.gguf"]
    assert measured_best(mine)["label"] == "long-enough"


def test_a_run_without_an_asking_record_keeps_the_asking_the_record_already_says(tmp_path):
    """The hundred-question row asked with batch, kinds and summary carried no asking
    record (it predates them), and a rewrite from it set all three to false."""
    from ml_stack.graph.bench.report import write_profiles
    from ml_stack.serve.profile import add, profile_for, record, records_in

    where = tmp_path / "profiles.json"
    add(record("flash.gguf", tight=True, batch=True, kinds=True, summary=True, rounds=6,
               questions=10, right=0.8), path=where)
    store = tmp_path / "runs.ladybug"
    _run(store, "flash--all-plain", questions=100, hits=80, seconds=2700.0)
    written = write_profiles(runs(store), path=where)
    assert [w.model for w, _ in written] == ["flash.gguf"]
    got = profile_for("flash.gguf", records=records_in(where))
    assert (got.batch, got.kinds, got.summary, got.rounds) == (True, True, True, 6)
    assert got.questions == 100 and "predates asking records" in got.note

    # a run that does record its asking is taken as it is, even when that means fewer ways
    save(store, scored_rows("flash--bare", questions=100, hits=82, seconds=2600.0),
         held={"graph": invented_digest(), "model": "flash.gguf"},
         asking={"tight": True, "batch": False, "kinds": False, "summary": False})
    write_profiles(runs(store), path=where)
    got = profile_for("flash.gguf", records=records_in(where))
    assert got.label == "flash--bare" and got.batch is False and got.summary is False
    assert "predates" not in got.note
