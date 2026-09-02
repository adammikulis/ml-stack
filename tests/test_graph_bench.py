"""What a run kept says when it is read back.

Every fixture here is invented. Nothing reads a real store, a real graph, or a real server.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ml_stack.graph.bench import Row, _hit, missed, runs, save, table
from ml_stack.graph.bench.selfcheck import ScriptedModel


def a_row(question: str, *, expected: list[str], shown: list[str], calls: int = 3,
          chars: int = 200, error: str = "") -> Row:
    return Row(label="tried", question=question, expected=expected, shown=shown,
               calls=calls, answer_chars=chars, error=error)


def test_hit_is_how_well_what_was_shown_matched_what_was_wanted():
    """F1, not recall. Recall alone made a 2B model look more accurate than a 120B, because
    it showed six entries where fewer than two were wanted and was charged nothing for it."""
    assert _hit({"expected": ["person:iris"], "shown": ["person:iris"]}) == 1.0
    assert _hit({"expected": ["person:iris"], "shown": []}) == 0.0
    assert _hit({"expected": ["person:iris"], "shown": ["topic:welding"]}) == 0.0
    assert _hit({"expected": [], "shown": ["person:iris"]}) == -1.0    # nothing to score

    # half of what was wanted, and all of what was shown was wanted
    assert _hit({"expected": ["person:iris", "person:otto"],
                 "shown": ["person:iris"]}) == pytest.approx(2 / 3)
    # everything wanted was found, but half of what was shown was noise
    assert _hit({"expected": ["person:iris"],
                 "shown": ["person:iris", "org:pellard"]}) == pytest.approx(2 / 3)


def test_showing_everything_does_not_score_well():
    """The test that was missing. A model that lights the whole graph on every question had
    a perfect score under recall, which is how a gameable metric goes unnoticed for a day."""
    from ml_stack.graph.bench import _precision, _recall

    graph_ids = [f"n{i}" for i in range(17)]
    everything = {"expected": ["n3", "n9"], "shown": graph_ids}

    assert _recall(everything) == 1.0            # it did find them, technically
    assert _precision(everything) == pytest.approx(2 / 17)
    assert _hit(everything) < 0.25               # and it is not a good answer

    # a precise answer that misses one beats a complete answer that shows everything
    careful = {"expected": ["n3", "n9"], "shown": ["n3"]}
    assert _recall(careful) < _recall(everything)
    assert _hit(careful) > _hit(everything)


def test_showing_nothing_does_not_score_well_either():
    """Precision alone is the opposite trap: saying nothing is perfect by it."""
    from ml_stack.graph.bench import _precision

    silent = {"expected": ["n3"], "shown": []}
    assert _precision(silent) == 0.0
    assert _hit(silent) == 0.0


def test_recall_and_precision_are_kept_beside_the_score():
    """The pair is what says *how* a run was wrong; the single number cannot."""
    from ml_stack.graph.bench import derived

    spraying = {"rows": [{"expected": ["n1"], "shown": ["n1", "n2", "n3", "n4"],
                          "seconds": 1.0, "processed_tokens": 10, "completion_tokens": 1}],
                "server": {}}
    d = derived(spraying)
    assert d["recall"] == 1.0
    assert d["precision"] == pytest.approx(0.25)
    assert d["shown_per_question"] == 4.0 and d["wanted_per_question"] == 1.0
    assert d["right"] == pytest.approx(0.4)


def test_row_hit_and_the_kept_form_agree():
    """`Row.hit` and `_hit` read the same row, one live and one out of a store."""
    row = a_row("who welds?", expected=["person:iris", "person:otto"], shown=["person:otto"])
    from dataclasses import asdict

    assert row.hit == _hit(asdict(row)) == pytest.approx(2 / 3)
    assert row.recall == 0.5 and row.precision == 1.0


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


def _server(handler):
    """A tiny HTTP server answering /slots, for the busy check."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = _json.dumps(handler(self.path)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_busy_counts_the_slots_that_are_working():
    from ml_stack.graph.bench import busy

    srv, url = _server(lambda p: [{"is_processing": True}, {"is_processing": False},
                                  {"is_processing": True}])
    try:
        assert busy(url) == 2
    finally:
        srv.shutdown()

    srv, url = _server(lambda p: [{"is_processing": False}, {"is_processing": False}])
    try:
        assert busy(url) == 0
    finally:
        srv.shutdown()


def test_a_server_that_will_not_say_is_not_treated_as_idle():
    """Unknown is not idle. Guessing idle is how the guard would fail open."""
    from ml_stack.graph.bench import busy

    assert busy("http://127.0.0.1:9") == -1          # nothing listening
    srv, url = _server(lambda p: {"not": "a list"})
    try:
        assert busy(url) == -1
    finally:
        srv.shutdown()


def test_a_busy_server_is_refused_and_anyway_overrides(capsys):
    """A timing taken while another run has the same GPU is not a timing.

    This happened: several sweeps were left running in the background against one server,
    and every wall clock measured during the overlap was two runs sharing a machine.
    """
    from argparse import Namespace

    from ml_stack.graph.bench import _idle

    srv, url = _server(lambda p: [{"is_processing": True}])
    try:
        assert _idle(url, Namespace(anyway=False)) is False
        said = capsys.readouterr().err
        assert "already working on 1 request" in said
        assert "--anyway" in said                     # and how to proceed on purpose

        assert _idle(url, Namespace(anyway=True)) is True
    finally:
        srv.shutdown()

    srv, url = _server(lambda p: [{"is_processing": False}])
    try:
        assert _idle(url, Namespace(anyway=False)) is True
    finally:
        srv.shutdown()

    # a server that will not say lets the run proceed, but says so rather than staying quiet
    assert _idle("http://127.0.0.1:9", Namespace(anyway=False)) is True
    assert "would not say whether it is busy" in capsys.readouterr().err


def test_a_short_run_still_asks_about_everything():
    """A shorter benchmark that has stopped asking about places is not a shorter benchmark,
    it is a different one. The first n are all of one kind because the set is written in
    groups; an even stride over a set that is two-thirds people returns two-thirds people."""
    from ml_stack.graph.bench import SHORT, sample
    from ml_stack.graph.community import QUESTIONS, graph

    kind = {n["id"]: n["kind"] for n in graph()["nodes"]}
    whole = {kind[e] for q in QUESTIONS for e in q["expect"]}
    assert len(whole) >= 6, "the full set should cover every kind the page draws"

    for n in (SHORT, 8, 10, 14, 24):
        short = sample(QUESTIONS, n)
        assert len(short) == n
        covered = {kind[e] for q in short for e in q["expect"]}
        assert covered == whole, f"sample({n}) dropped {whole - covered}"
        assert any(not q["expect"] for q in short), \
            f"sample({n}) asks nothing whose right answer is nobody"

    # the naive versions this replaced, shown failing
    first = QUESTIONS[:8]
    assert {kind[e] for q in first for e in q["expect"]} != whole
    stride = [QUESTIONS[int(i * len(QUESTIONS) / 8)] for i in range(8)]
    assert {kind[e] for q in stride for e in q["expect"]} != whole


def test_a_short_run_is_the_same_short_run_twice():
    """Two runs of a short set have to be comparable with each other, or the shortening has
    bought speed by giving up the only thing a benchmark is for."""
    from ml_stack.graph.bench import sample
    from ml_stack.graph.community import QUESTIONS

    assert sample(QUESTIONS, 9) == sample(QUESTIONS, 9)
    assert [q["q"] for q in sample(QUESTIONS, len(QUESTIONS) + 1)] == [q["q"] for q in QUESTIONS]
    assert sample(QUESTIONS, 0) == [dict(q) for q in QUESTIONS]


def test_drafts_counts_the_client_it_is_measuring():
    """`measure` wraps the client it is given and hands *that* to the asking. An asking that
    closes over a client of its own is never counted: every token and call comes back zero
    while the wall clock says otherwise, and the table reads as though nothing happened.
    That shipped once."""
    import ml_stack.graph.bench as bench

    seen = {}

    class Reply:
        content = "an answer"
        raw = {"usage": {"prompt_tokens": 100, "completion_tokens": 20},
               "timings": {"prompt_n": 90, "cache_n": 10, "draft_n": 8,
                           "draft_n_accepted": 5}}
        tool_calls = None

    class Model:
        def chat(self, messages, **kw):
            return Reply()

    def fake_ask(graph, **kw):
        def ask(question, client):
            seen["client"] = client
            client.chat([{"role": "user", "content": question}])
            return type("A", (), {"content": "an answer", "show": [], "ids": [],
                                  "read": [], "found": [], "path": [], "why": ""})()
        return ask

    rows = bench.measure(fake_ask(None), [{"q": "who?", "expect": ["n1"]}],
                         label="tried", client=Model())
    assert type(seen["client"]).__name__ == "Counting", \
        "the asking must be handed the counted client"
    assert rows[0].calls == 1
    assert rows[0].prompt_tokens == 100 and rows[0].completion_tokens == 20
    assert rows[0].draft_tokens == 8 and rows[0].draft_taken == 5


def test_what_a_server_holds_is_reported_even_when_the_derived_number_is_not(tmp_path, capsys):
    """`resident - weights` assumes every weight is resident. llama.cpp mmaps them, so a
    page is resident only once touched, and an MoE using ten experts of five hundred never
    touches most. Flash-Next sat at 70G resident against 87G on disk: the subtraction goes
    negative, clamps to zero, and prints as a dash that reads as "not measured" rather than
    "not meaningful"."""
    store = tmp_path / "runs.ladybug"
    row = a_row("who?", expected=["person:iris"], shown=["person:iris"])
    save(store, [row], held={"context": 32768, "slots": 2,
                             "resident_bytes": 70 * 2**30, "mmapped": True})
    save(store, [row], held={"context": 32768, "slots": 2,
                             "resident_bytes": 12 * 2**30,
                             "kv_and_run_bytes": 3 * 2**30,
                             "bytes_per_1k_context": 30 * 2**20})
    table(runs(store))
    said = capsys.readouterr().out
    assert "70.00G" in said, "what it actually holds is always reported"
    assert "mmap" in said, "and why the derived number is missing, rather than a bare dash"
    assert "3.00G" in said and "30.0M" in said, "a fully resident model still reports both"


def test_the_footprint_does_not_invent_a_negative_cost(monkeypatch):
    """Clamping to zero produced a dash; the honest answer is that the subtraction does not
    apply, and that has to be distinguishable from never having looked."""
    import ml_stack.graph.bench as bench

    def props(url, **kw):
        return {"model_path": "/models/thing-00001-of-00003.gguf", "total_slots": 2,
                "default_generation_settings": {"n_ctx": 32768}}

    monkeypatch.setattr(bench, "request_json", props, raising=False)

    # mmapped: less resident than the weights on disk
    out = dict(resident_bytes=70 * 2**30, weights_bytes=87 * 2**30)
    beyond = out["resident_bytes"] - out["weights_bytes"]
    assert beyond < 0

    # what the code must do with that: no kv_and_run, but say why
    held = {}
    if beyond > 0:
        held["kv_and_run_bytes"] = beyond
    else:
        held["mmapped"] = True
    assert "kv_and_run_bytes" not in held and held["mmapped"] is True


def test_a_short_run_keeps_the_difficulty_and_not_only_the_variety():
    """Every kind still asked about is half of it. The other half is that the questions are
    as hard: a short run made only of one-answer questions would score higher for a reason
    that has nothing to do with the model."""
    from ml_stack.graph.bench import SHORT, sample
    from ml_stack.graph.community import QUESTIONS

    whole = [q for q in QUESTIONS if q["expect"]]
    short = [q for q in sample(QUESTIONS, SHORT) if q["expect"]]

    def mean(qs):
        return sum(len(q["expect"]) for q in qs) / len(qs)

    assert abs(mean(short) - mean(whole)) < 0.35, "the short run must be about as hard"
    assert any(len(q["expect"]) == 1 for q in short), "questions with one right answer"
    assert any(len(q["expect"]) > 2 for q in short), "and questions with several"
    assert len(short) < len(whole)


def test_how_many_prefers_an_explicit_count():
    from argparse import Namespace

    from ml_stack.graph.bench import SHORT, _how_many

    assert _how_many(Namespace(sample=0, short=True)) == SHORT
    assert _how_many(Namespace(sample=12, short=True)) == 12, "--sample is the explicit one"
    assert _how_many(Namespace(sample=0, short=False)) == 0, "and neither means all of them"


def test_runs_can_be_written_out_so_they_are_not_on_one_disk(tmp_path):
    """The store lives under ~/.ml-stack and nothing backs it up. A day of measuring sits on
    one machine, and a comparison a week from now has nothing to compare against."""
    import json

    from ml_stack.graph.bench import export, invented_digest

    store = tmp_path / "runs.ladybug"
    row = a_row("who?", expected=["person:iris", "person:otto"], shown=["person:iris"])
    row.draft_tokens, row.draft_taken = 54, 41
    save(store, [row], held={"context": 32768, "slots": 4, "model": "thing.gguf",
                             "resident_bytes": 12 * 2**30,
                             "graph": invented_digest(),
                             "sampling": {"temperature": 0.0}})

    where = export(runs(store), tmp_path / "out.json")
    got = json.loads(pathlib.Path(where).read_text())
    assert len(got) == 1
    one = got[0]
    assert one["label"] == "tried" and one["questions"] == 1
    assert one["recall"] == 0.5 and one["precision"] == 1.0
    assert one["f1"] == pytest.approx(2 / 3, abs=1e-3)
    assert one["draft_offered"] == 54 and one["draft_kept"] == 41
    assert one["context"] == 32768 and one["slots"] == 4
    assert one["sampling"] == {"temperature": 0.0}
    # the point is that it opens without this package, so nothing exotic may be in it
    assert json.dumps(got)


def test_an_export_skips_runs_with_nothing_to_score(tmp_path):
    from ml_stack.graph.bench import export

    store = tmp_path / "runs.ladybug"
    save(store, [Row(label="chatter", question="hi", expected=[], shown=[])])
    assert json.loads(pathlib.Path(export(runs(store), tmp_path / "o.json")).read_text()) == []


def test_only_runs_over_the_invented_community_are_exported(tmp_path, capsys):
    """`run --graph` takes any graph, so a run may have been asked of a real community, and
    this file is meant for a public repository. Omitting the questions and entry ids is what
    the current field list happens to do; refusing a run that was not over the invented
    community is what stops the next field added from leaking."""
    from ml_stack.graph.bench import export, invented_digest

    store = tmp_path / "runs.ladybug"
    row = a_row("who?", expected=["person:iris"], shown=["person:iris"])
    save(store, [row], held={"graph": invented_digest(), "model": "thing.gguf"})
    save(store, [row], held={"graph": "some-other-graph", "model": "thing.gguf"})
    save(store, [row], held={"model": "thing.gguf"})          # from before the marker

    got = json.loads(pathlib.Path(export(runs(store), tmp_path / "o.json")).read_text())
    assert len(got) == 1, "only the one whose graph is known to be the invented one"
    said = capsys.readouterr().err
    assert "2 run(s) left out" in said
    assert "not into a repository" in said

    everything = json.loads(
        pathlib.Path(export(runs(store), tmp_path / "all.json", anyway=True)).read_text())
    assert len(everything) == 3, "--anyway is for a store that never left the machine"


def test_an_export_carries_no_question_and_no_entry(tmp_path):
    """Whatever else changes, the words a community said must not be in here."""
    from ml_stack.graph.bench import export, invented_digest

    store = tmp_path / "runs.ladybug"
    row = a_row("who surveys land in Calderwick?", expected=["person:iris"],
                shown=["person:iris", "org:brayfield"])
    save(store, [row], held={"graph": invented_digest()})

    text = pathlib.Path(export(runs(store), tmp_path / "o.json")).read_text()
    assert "Calderwick" not in text and "surveys" not in text
    assert "person:iris" not in text and "org:brayfield" not in text
    assert "recall" in text, "the totals are the point, and they are there"


def test_a_mmapped_model_still_measures():
    """An mmapped model has no kv_and_run_bytes -- the weights are not all resident, so the
    subtraction says nothing. Reading it anyway raised at the *end* of a run, after every
    question had been answered, and threw away fourteen minutes of GPU for a summary line."""
    from ml_stack.graph import bench

    got = bench.beyond_weights({"resident_bytes": 4 * 2**30, "weights_bytes": 60 * 2**30,
                                "context": 65536, "slots": 2})
    assert got["mmapped"] is True
    assert "bytes_per_1k_context" not in got
    assert got["resident_bytes"] == 4 * 2**30, "what it holds is still reported"

    fully = bench.beyond_weights({"resident_bytes": 70 * 2**30, "weights_bytes": 60 * 2**30,
                                  "context": 32768, "slots": 2})
    assert fully["kv_and_run_bytes"] == 10 * 2**30
    assert fully["bytes_per_1k_context"] > 0 and "mmapped" not in fully


# A graph small enough to read at a glance. "compiler" is the point: the word index stems
# it, so "compilers" finds it, and character matching does not -- a lexical miss the store
# catches, which is the difference the bench was not measuring.
TINY = {
    "nodes": [
        {"id": "topic:compiler", "kind": "topic", "label": "compiler", "mentions": 3,
         "attrs": {}},
        {"id": "person:ada", "kind": "person", "label": "Ada Quill", "mentions": 1, "attrs": {}},
    ],
    "edges": [{"source": "person:ada", "target": "topic:compiler", "rel": "interested_in",
               "weight": 1}],
}


class _Scripted(ScriptedModel):
    """The runner's own scripted model -- calls look_up for one text, then answers, keeps
    every message it was shown -- given the text to look up. One fake, shared with
    `bench.selfcheck`, so that what the tests let through the runner lets through."""

    def __init__(self, text: str = "compilers") -> None:
        super().__init__()
        self.text = text


def _tiny_store(tmp_path):
    from ml_stack.graph.store import GraphStore

    where = tmp_path / "graph.ladybug"
    with GraphStore(where) as store:
        store.write(TINY)
    return where


def test_finding_names_what_a_run_measured():
    from ml_stack.graph.bench import finding

    assert finding(None) == "chars"
    assert finding("") == "chars"
    assert finding("some.ladybug") == "words"
    assert finding("some.ladybug", "http://127.0.0.1:8081") == "meaning"


def test_a_run_given_a_store_looks_up_as_the_application_does(tmp_path):
    """The bench measured character matching for months while the application shipped
    the hybrid -- characters, the word index and vectors fused -- so every ranking it wrote
    ranked a look_up nobody ran. Given a store, the model's look_up is the shipped one."""
    pytest.importorskip("ladybug", reason="the store needs ml-stack[store]")
    from ml_stack.graph.bench import asking

    where = _tiny_store(tmp_path)

    with_store = _Scripted()
    ask = asking(TINY, store=where)
    assert ask.finder == "words"
    ask("who works on compilers?", with_store)
    assert "topic:compiler" in with_store.told(), "the word index stems; characters do not"

    without = _Scripted()
    plain = asking(TINY)
    assert plain.finder == "chars"
    plain("who works on compilers?", without)
    assert "topic:compiler" not in without.told()
    assert "Nothing in the graph matches" in without.told()


def test_the_terse_tools_look_up_through_the_store_too(tmp_path):
    """`tools_for(terse=True)` is built here rather than inside converse, so the finder has
    to be handed to it as well or the terse run measures a different look_up again."""
    pytest.importorskip("ladybug", reason="the store needs ml-stack[store]")
    from ml_stack.graph.bench import asking

    terse = _Scripted()
    asking(TINY, store=_tiny_store(tmp_path), terse=True)("who?", terse)
    assert "topic:compiler" in terse.told()


def test_a_run_writes_down_which_finder_it_measured(tmp_path, monkeypatch, capsys):
    """Like `ctx`: a run with one finder against a run with another is two measurements,
    and the only way to know later is to write it down now."""
    pytest.importorskip("ladybug", reason="the store needs ml-stack[store]")
    import ml_stack.graph.bench as bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")     # never ~/.ml-stack
    monkeypatch.setattr(bench, "footprint", lambda url: {"base_url": url})
    monkeypatch.setattr(bench, "ask_from", lambda spec: _Scripted)
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps(TINY))
    asked = tmp_path / "q.jsonl"
    asked.write_text(json.dumps({"q": "who works on compilers?", "expect": ["topic:compiler"]})
                     + "\n")
    kept = tmp_path / "runs.ladybug"
    store = _tiny_store(tmp_path)
    common = ["--kept", str(kept), "--graph", str(graph), "--questions", str(asked),
              "--client", "fake:client", "--no-smoke"]

    assert bench._main(["run", "with-words", "--store", str(store), *common]) == 0
    assert "with-words: 1 questions" in capsys.readouterr().out.splitlines()[0]
    assert bench._main(["run", "by-chars", "--store", "", *common]) == 0
    assert "look_up by chars" in capsys.readouterr().out.splitlines()[0]

    back = {r["label"]: r for r in runs(kept)}
    assert back["with-words"]["server"]["finder"] == "words"
    assert back["by-chars"]["server"]["finder"] == "chars"


def test_the_first_line_a_run_prints_says_which_finder(tmp_path, monkeypatch, capsys):
    pytest.importorskip("ladybug", reason="the store needs ml-stack[store]")
    import ml_stack.graph.bench as bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench, "footprint", lambda url: {"base_url": url})
    monkeypatch.setattr(bench, "ask_from", lambda spec: _Scripted)
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps(TINY))
    asked = tmp_path / "q.jsonl"
    asked.write_text(json.dumps({"q": "who?", "expect": ["topic:compiler"]}) + "\n")
    bench._main(["run", "tried", "--kept", str(tmp_path / "runs.ladybug"),
                 "--graph", str(graph), "--questions", str(asked), "--client", "fake:client",
                 "--store", str(_tiny_store(tmp_path)), "--no-smoke"])
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("tried: 1 questions over") and "look_up by words" in first


def test_the_store_prepare_built_is_the_default_once_it_exists(tmp_path, monkeypatch):
    """A machine that has run `prepare` measures the shipped finder without another flag;
    one that has not is not pointed at a file that is not there."""
    import ml_stack.graph.bench as bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    assert bench.prepared() == ""
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "graph.ladybug").write_bytes(b"")
    assert bench.prepared() == str(tmp_path / "home" / "graph.ladybug")


def test_the_table_says_which_finder_a_run_used_and_still_prints_an_old_one(tmp_path, capsys):
    from ml_stack.graph.bench import missed

    store = tmp_path / "runs.ladybug"
    row = a_row("who?", expected=["person:iris"], shown=["person:iris"])
    save(store, [row], held={"context": 32768, "slots": 2, "finder": "meaning"})
    save(store, [row], held={"context": 32768, "slots": 2})        # from before the column
    table(runs(store))
    said = capsys.readouterr().out
    head, lines = said.splitlines()[0], [ln for ln in said.splitlines() if ln.startswith("tried")]
    assert "find" in head
    assert head.split().index("speed") == head.split().index("draft") + 1, "beside draft"
    assert head.split().index("find") == head.split().index("speed") + 1, "then find"
    # "tried  32k x2  1 ..." -- the context field carries a space, so find is the eleventh word
    assert lines[0].split()[10] == "meaning"
    assert lines[1].split()[10] == "-"

    missed(runs(store), everything=True)
    said = capsys.readouterr().out
    assert "find meaning" in said and "find -" in said


def test_an_export_and_the_ranking_carry_the_finder(tmp_path):
    from ml_stack.graph.bench import SHORT, export, invented_digest, ranking

    store = tmp_path / "runs.ladybug"
    rows = [a_row(f"q{n}?", expected=["person:iris"], shown=["person:iris"])
            for n in range(SHORT)]
    save(store, rows, held={"graph": invented_digest(), "model": "thing.gguf",
                            "finder": "words"})
    got = json.loads(pathlib.Path(export(runs(store), tmp_path / "o.json")).read_text())
    assert got[0]["finder"] == "words"
    said = ranking(runs(store))
    assert "| find |" in said and "| words |" in said


# -- what F1 cannot see: a name the answer made up ---------------------------------------

NAMED = {
    "nodes": [
        {"id": "person:iris", "kind": "person", "label": "Iris Tamsin", "attrs": {}},
        {"id": "person:otto", "kind": "person", "label": "Otto Brayfield", "attrs": {}},
        {"id": "org:pellard", "kind": "org", "label": "Pellard", "attrs": {}},
        {"id": "person:al", "kind": "person", "label": "Al", "attrs": {}},
    ],
    "edges": [],
}


def test_an_answer_naming_an_entry_that_was_never_read_counts_one():
    """F1 scores what was lit. An answer can light the right people and still name one the
    model never found, read or showed -- a plausible name it made up -- and F1 is none
    the wiser."""
    from ml_stack.graph.bench import unread_named

    said = "Iris Tamsin welds; ask Otto Brayfield about the rest."
    assert unread_named(said, NAMED, touched=["person:iris"]) == ["Otto Brayfield"]
    assert unread_named(said, NAMED, touched=[]) == ["Iris Tamsin", "Otto Brayfield"]


def test_an_answer_naming_only_what_it_read_counts_nothing():
    from ml_stack.graph.bench import unread_named

    said = "Iris Tamsin welds, and Pellard hires."
    assert unread_named(said, NAMED, touched=["person:iris", "org:pellard"]) == []
    # any of found, read, path or show is a tool having produced it; ids from any count
    assert unread_named(said, NAMED, touched=["org:pellard", "person:iris"]) == []


def test_a_label_inside_a_longer_word_does_not_count():
    """Whole words, as the page's `namedIn` matches: "Pellard" is not in "Pellardsville"."""
    from ml_stack.graph.bench import unread_named

    assert unread_named("the Pellardsville fair", NAMED) == []
    assert unread_named("the Pellard fair", NAMED) == ["Pellard"]
    assert unread_named("PELLARD, again", NAMED) == ["Pellard"], "case and punctuation aside"
    # a two-letter label is in every "already" and "also": not matched, as on the page
    assert unread_named("it is already done, also", NAMED) == []
    assert unread_named("", NAMED) == []


def test_measure_counts_what_the_answer_named_but_never_touched():
    import ml_stack.graph.bench as bench

    class Model:
        def chat(self, messages, **kw):
            return type("R", (), {"content": "x", "raw": {}, "tool_calls": None})()

    def ask(question, client):
        client.chat([])
        return type("A", (), {"content": "Iris Tamsin and Otto Brayfield.", "show": ["person:iris"],
                              "ids": ["person:iris"], "read": ["person:iris"], "found": [],
                              "path": [], "why": ""})()

    rows = bench.measure(ask, [{"q": "who?", "expect": ["person:iris"]}], label="t",
                         client=Model(), graph=NAMED)
    assert rows[0].unread == ["Otto Brayfield"] and rows[0].unread_named == 1
    without = bench.measure(ask, [{"q": "who?", "expect": ["person:iris"]}], label="t",
                            client=Model())
    assert without[0].unread_named == 0, "no graph, nothing counted -- not an error"


def test_the_table_the_detail_and_the_ranking_carry_what_was_made_up(tmp_path, capsys):
    from ml_stack.graph.bench import SHORT, export, invented_digest, missed, ranking
    from ml_stack.graph.store import GraphStore

    store = tmp_path / "runs.ladybug"
    rows = [a_row(f"q{n}?", expected=["person:iris"], shown=["person:iris"])
            for n in range(SHORT)]
    rows[0].unread, rows[0].unread_named = ["Otto Brayfield"], 1
    rows[1].unread, rows[1].unread_named = ["Pellard", "Otto Brayfield"], 2
    save(store, rows, held={"graph": invented_digest(), "model": "thing.gguf",
                            "finder": "words"})
    # a run kept before the count existed has rows without the key
    with GraphStore(store) as held:
        held.put_doc("bench:older:20260101T000000", {
            "at": "2026-01-01T00:00:00", "label": "older", "server": {},
            "rows": [{"question": "q?", "expected": ["person:iris"], "shown": ["person:iris"]}]})

    table(runs(store))
    said = capsys.readouterr().out
    head = said.splitlines()[0]
    assert "made" in head
    assert head.split().index("made") == head.split().index("prec") + 1, "beside the scores"
    by_label = {ln.split()[0]: ln for ln in said.splitlines() if ln.startswith(("tried", "older"))}
    # made, then t/o (blank: nothing timed out), then the sampling
    assert by_label["tried"].rstrip().endswith("100%     3       -"), "the total over the run"
    assert by_label["older"].rstrip().endswith("100%" + " " * 13 + "-"), \
        "blank for a run from before, not 0"

    missed(runs(store, "tried"), everything=True)
    said = capsys.readouterr().out
    assert "made    Pellard, Otto Brayfield" in said and "never found or read" in said

    got = json.loads(pathlib.Path(export(runs(store), tmp_path / "o.json")).read_text())
    assert {r["label"]: r["unread_named"] for r in got} == {"tried": 3}
    assert "| made |" in ranking(runs(store)) and "| words | 3 |" in ranking(runs(store))


# -- --also rich --------------------------------------------------------------------------

def test_rich_is_asked_for_the_way_terse_is():
    from argparse import Namespace

    from ml_stack.graph.bench import _ways

    ways = _ways(Namespace(also=["rich"], terse=False, temperature=0.0))
    assert ways[1]["label"] == "rich" and ways[1]["rich"] is True
    assert ways[1]["temperature"] == 0.0, "the run's sampling, as terse carries it"
    assert "rich" not in ways[0], "the first way is what was asked for, unchanged"


def test_rich_reaches_converse_as_a_keyword(monkeypatch):
    """`converse(..., rich=True)` is another agent's keyword; this only has to hand it on."""
    import ml_stack.graph.ask as ask_module
    from ml_stack.graph.bench import asking

    reached = {}

    def fake_converse(question, graph, client, **kw):
        reached.update(kw)
        return type("A", (), {"content": "", "show": [], "ids": [], "why": ""})()

    monkeypatch.setattr(ask_module, "converse", fake_converse)
    asking(TINY, rich=True)("who?", _Scripted())
    assert reached.get("rich") is True
    reached.clear()
    asking(TINY)("who?", _Scripted())
    assert "rich" not in reached, "not asked for, not sent -- the default is converse's own"


# -- several conversations at once --------------------------------------------------------

class _Overlapping:
    """Answers after a pause and keeps the interval of every call, so a test can see
    whether two calls were ever in flight together. Looks up one thing that matches
    nothing, then names a person it never found. How many calls a turn takes after that
    is `converse`'s business -- it nudges an answer that lit nothing -- not this fake's."""

    def __init__(self, pause: float = 0.1) -> None:
        import threading

        self.pause = pause
        self.calls: list[tuple[float, float, list[dict]]] = []
        self._lock = threading.Lock()
        self.sampling: dict = {}

    def chat(self, messages, tools=None, **_):
        import time

        began = time.time()
        time.sleep(self.pause)
        with self._lock:
            self.calls.append((began, time.time(), list(messages)))
        raw = {"usage": {"prompt_tokens": 40, "completion_tokens": 8},
               "timings": {"prompt_n": 30, "cache_n": 10, "prompt_ms": 20.0,
                           "predicted_ms": 30.0}}
        if tools and not any(m.get("role") == "tool" for m in messages):
            return type("R", (), {"content": "", "thinking": None, "raw": raw, "tool_calls": [
                {"id": "c1", "function": {"name": "look_up",
                                          "arguments": json.dumps({"texts": ["zzqx"]})}}]})()
        return type("R", (), {"content": "Ada Quill works on compilers.", "thinking": None,
                              "raw": raw, "tool_calls": None})()

    def asked(self, question: str) -> list[dict]:
        """The messages the first call of the turn that asked ``question`` was shown."""
        return next(m for _, _, m in self.calls if m[-1].get("content") == question)

    def overlapped(self) -> bool:
        spans = [(a, b) for a, b, _ in self.calls]
        return any(a1 < b2 and a2 < b1
                   for i, (a1, b1) in enumerate(spans) for (a2, b2) in spans[i + 1:])


def test_conversations_really_run_at_the_same_time():
    from ml_stack.graph.bench import asking, concurrent

    model = _Overlapping()
    rows, held = concurrent(asking(TINY), [{"q": f"q{n}?", "expect": ["person:ada"]}
                                           for n in range(6)],
                            conversations=3, turns=2, label="t", client=model, graph=TINY)
    assert len(rows) == 6 and len(model.calls) == sum(r.calls for r in rows)
    assert all(r.calls >= 2 for r in rows), "a look_up and an answer at the least"
    assert model.overlapped(), "three conversations on threads must overlap"
    assert held["concurrency"]["seconds"] < sum(r.seconds for r in rows), \
        "the run's clock is over all of them, not their sum"


def test_a_conversation_carries_its_earlier_turns():
    from ml_stack.graph.bench import asking, concurrent

    model = _Overlapping(pause=0.0)
    rows, _ = concurrent(asking(TINY), [{"q": "first?"}, {"q": "second?"}, {"q": "third?"}],
                         conversations=1, turns=3, label="t", client=model, graph=TINY)
    assert [(r.conversation, r.turn, r.question) for r in rows] == [
        (0, 0, "first?"), (0, 1, "second?"), (0, 2, "third?")]
    last = model.asked("third?")
    said = [(m["role"], m["content"]) for m in last if m["role"] in ("user", "assistant")]
    assert said == [("user", "first?"), ("assistant", "Ada Quill works on compilers."),
                    ("user", "second?"), ("assistant", "Ada Quill works on compilers."),
                    ("user", "third?")]
    assert not any(m["role"] == "tool" for m in last), "the working is not carried, the answer is"


def test_each_conversation_asks_its_own_stretch_of_the_questions():
    from ml_stack.graph.bench import asking, concurrent

    rows, _ = concurrent(asking(TINY), [{"q": f"q{n}?"} for n in range(4)],
                         conversations=2, turns=2, label="t", client=_Overlapping(0.0))
    assert [(r.conversation, r.question) for r in rows] == [
        (0, "q0?"), (0, "q1?"), (1, "q2?"), (1, "q3?")]
    # fewer questions than turns: it wraps rather than stopping short
    rows, _ = concurrent(asking(TINY), [{"q": "only?"}], conversations=2, turns=2,
                         label="t", client=_Overlapping(0.0))
    assert [r.question for r in rows] == ["only?"] * 4


def test_a_turn_records_its_first_token_and_what_it_spent_waiting():
    """The server says what it spent reading and generating; the wall clock less that is
    the waiting, which is the queueing once there are more conversations than slots."""
    from ml_stack.graph.bench import asking, concurrent

    rows, held = concurrent(asking(TINY), [{"q": "q?", "expect": ["person:ada"]}],
                            conversations=1, turns=1, label="t", client=_Overlapping(0.1),
                            graph=TINY)
    row = rows[0]
    assert row.calls >= 2 and row.seconds >= 0.1 * row.calls - 1e-6  # three sleeps of 0.1 are 0.30000000000000004
    # 30ms of generating out of a 100ms pause: the first token waited about 70ms
    # real sleeps: a loaded machine (a benchmark, a parallel suite) stretches them, so the
    # bounds say only that the first token came after the wait and before the answer
    assert 0.05 <= row.first_token < row.seconds
    # every call is 20ms reading and 30ms generating by the server's account
    assert row.queued == pytest.approx(row.seconds - 0.05 * row.calls, abs=0.5)
    assert row.cached_tokens == 10 * row.calls and row.processed_tokens == 30 * row.calls
    assert row.unread == ["Ada Quill"], "named, never looked up"
    assert row.shown == [] and row.hit == 0.0, "and F1 is scored as ever"
    assert held["concurrency"]["queued"] == pytest.approx(row.queued)


def test_the_run_reads_the_slots_and_keeps_the_most_the_server_held(monkeypatch):
    import ml_stack.graph.bench as bench

    seen = iter([{"base_url": "u", "resident_bytes": 3 * 2**30, "weights_bytes": 2**30},
                 {"base_url": "u", "resident_bytes": 2**30, "weights_bytes": 2**30}])
    monkeypatch.setattr(bench, "footprint", lambda url: dict(next(seen)))
    srv, url = _server(lambda p: [{"is_processing": False}, {"is_processing": False}])
    try:
        assert bench.slot_count(url) == 2
        rows, held = bench.concurrent(bench.asking(TINY), [{"q": "q?"}], conversations=1,
                                      turns=1, label="t", client=_Overlapping(0.0),
                                      base_url=url)
    finally:
        srv.shutdown()
    assert held["concurrency"]["slots"] == 2
    assert held["resident_bytes"] == 3 * 2**30, "the peak, not what was left afterwards"
    assert held["kv_and_run_bytes"] == 2 * 2**30, "and the derived figure follows it"
    assert bench.slot_count("http://127.0.0.1:9") == -1


def test_a_concurrent_run_is_kept_and_shown_with_its_marker(tmp_path, capsys):
    from ml_stack.graph.bench import asking, concurrent, missed

    store = tmp_path / "runs.ladybug"
    rows, held = concurrent(asking(TINY), [{"q": f"q{n}?", "expect": ["person:ada"]}
                                           for n in range(6)],
                            conversations=2, turns=3, label="together",
                            client=_Overlapping(0.0), graph=TINY)
    save(store, rows, held={**held, "context": 32768, "slots": 2, "finder": "chars"})
    save(store, [a_row("q?", expected=["person:ada"], shown=["person:ada"])],
         held={"context": 32768, "slots": 2, "finder": "chars"})

    back = {r["label"]: r for r in runs(store)}
    kept = back["together"]["server"]["concurrency"]
    assert kept["conversations"] == 2 and kept["turns"] == 3 and kept["slots"] == -1
    assert kept["seconds"] >= 0 and kept["queued"] >= 0
    assert {(r["conversation"], r["turn"]) for r in back["together"]["rows"]} == {
        (c, t) for c in range(2) for t in range(3)}
    assert all("first_token" in r and "queued" in r for r in back["together"]["rows"])
    assert back["together"]["unread_named"] == 6

    table(runs(store))
    said = capsys.readouterr().out
    head = said.splitlines()[0]
    assert head.split().index("conc") == head.split().index("find") + 1, "beside find"
    lines = {ln.split()[0]: ln for ln in said.splitlines() if ln[:1].isalpha() and ln[:3] != "run"}
    # `together`'s client reports its cache, so its line has a pfx word and `tried`'s has
    # none: the marker is found by what it says, not by where it falls
    assert "2x3" in lines["together"].split()
    assert not any("x" in w and w[0].isdigit() for w in lines["tried"].split())

    missed(runs(store, "together"), everything=True)
    said = capsys.readouterr().out
    assert "2x3 at once" in said
    assert "conversation 1 turn 2" in said and "first token" in said and "queued" in said


def test_the_concurrent_subcommand_smokes_two_conversations_of_one_turn(tmp_path, monkeypatch, capsys):
    import ml_stack.graph.bench as bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")     # never ~/.ml-stack
    monkeypatch.setattr(bench, "ask_from", lambda spec: _Overlapping)
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps(TINY))
    asked = tmp_path / "q.jsonl"
    asked.write_text("\n".join(json.dumps({"q": f"q{n}?", "expect": ["person:ada"]})
                               for n in range(5)) + "\n")
    kept = tmp_path / "runs.ladybug"
    assert bench.main(["concurrent", "two-at-once", "--smoke", "--conversations", "9",
                       "--turns", "9", "--kept", str(kept), "--graph", str(graph),
                       "--questions", str(asked), "--client", "fake:client",
                       "--store", "", "--no-selfcheck"]) == 0
    said = capsys.readouterr().out
    first = next(ln for ln in said.splitlines() if not ln.startswith("estimate:"))
    assert first.startswith("two-at-once: 2 conversations of 1 turn(s) at once")
    assert "kept as bench:two-at-once:" in said

    back = runs(kept)[0]
    assert back["server"]["concurrency"]["conversations"] == 2
    assert back["server"]["concurrency"]["turns"] == 1
    assert back["server"]["finder"] == "chars"
    assert len(back["rows"]) == 2


def test_concurrent_is_a_measuring_subcommand_and_takes_the_lock():
    """Two of anything on the GPU at once is two measurements of nothing; the lock is what
    `main` takes for every subcommand in MEASURING, so this one has to be in it."""
    from ml_stack.graph.bench import MEASURING

    assert "concurrent" in MEASURING


# -- sweep --serve, which had no test through _main -----------------------------------------

# The client `served` builds for each way: takes a base_url and the way's sampling -- and
# only what the real `Client` takes -- and has the card a `--also card` way reads. It is
# the runner's own fake; a fake of the tests' own with **kwargs let `tight` through to an
# 87G load on 2026-09-02.
_ServedModel = ScriptedModel


def _preflight_ok(monkeypatch, *, refuse=(), kv_estimate=3 * 2**30, weights=5 * 2**30):
    """A preflight that reads nothing: every check passes, unless the model's name holds
    one of ``refuse``, in which case the shards check fails the way a missing file would.
    `room` is faked too, so no test asks sysctl what this machine may wire."""
    import ml_stack.hub
    import ml_stack.serve.preflight as preflight
    from ml_stack.serve.preflight import Check, Report

    seen = []

    def fake_preflight(spec, *, binary, limit_bytes=0):
        seen.append(spec)
        bad = any(word in str(spec.model) for word in refuse)
        return Report(checks=[
            Check("shards", not bad, "missing or empty: " + str(spec.model) if bad else "complete"),
            Check("architecture", True, "gemma4"),
            Check("fit", True, f"{(weights + kv_estimate) / 2**30:.1f}G estimated fits under "
                               f"{limit_bytes / 2**30:.1f}G"),
            Check("flags", True, "every flag this spec would emit is one this build accepts"),
        ], weights_bytes=weights, kv_estimate_bytes=kv_estimate)

    monkeypatch.setattr(preflight, "Preflight", fake_preflight)
    monkeypatch.setattr(ml_stack.hub, "room", lambda: 110 * 2**30)
    return seen


def test_a_sweep_that_serves_summarises_one_row_per_variant(tmp_path, monkeypatch, capsys):
    """This path had no test. The `--serve` loop's variable was `named`, the same name as
    the (name, url) list `--on` builds, so after serving, the summary unpacked the last
    model's name a character at a time and every `sweep --serve` crashed after answering
    everything. A smoke run caught it; this is the test that should have."""
    from contextlib import contextmanager

    import ml_stack.client
    import ml_stack.serve
    import ml_stack.graph.bench as bench

    @contextmanager
    def fake_serve(model, **kw):
        yield type("Up", (), {"base_url": "http://127.0.0.1:1"})()

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench, "footprint", lambda url: {"base_url": url})
    monkeypatch.setattr(bench, "find_model", lambda named: named)
    monkeypatch.setattr(ml_stack.serve, "serve", fake_serve)
    monkeypatch.setattr(ml_stack.client, "Client", _ServedModel)
    _preflight_ok(monkeypatch)
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps(TINY))
    asked = tmp_path / "q.jsonl"
    asked.write_text(json.dumps({"q": "who works on compilers?", "expect": ["topic:compiler"]})
                     + "\n")
    kept = tmp_path / "runs.ladybug"

    assert bench._main(["sweep", "--serve", "tiny.gguf", "--plain-only", "--also", "terse",
                        "--also", "card", "--kept", str(kept), "--graph", str(graph),
                        "--questions", str(asked), "--store", "", "--serve-port", "1",
                        "--no-smoke"]) == 0
    said = capsys.readouterr().out
    labels = [r["label"] for r in runs(kept)]
    assert sorted(labels) == sorted(["tiny-plain", "tiny-plain-terse", "tiny-plain-card"])
    after_rule = said.split("\n---", 1)[1].splitlines()[1:]
    summary = [ln for ln in after_rule if ln.startswith("tiny-plain")]
    assert len(summary) == 3, said
    assert any("t1 k64" in ln for ln in summary), "the card way was asked with the card"


# -- never lose a run -----------------------------------------------------------------------

def test_plain_makes_a_record_the_store_can_keep_without_dropping_anything():
    from dataclasses import dataclass

    from ml_stack.graph.bench import _plain

    @dataclass
    class Held:
        slots: int
        where: pathlib.Path

    record = {"server": {"per_turn": {(0, 1): 2.0, 3: "x"}, "held": Held(2, pathlib.Path("/m"))},
              "rows": ({"shown": {"b", "a"}},), "n": None, "ok": True}
    plain = _plain(record)
    assert plain == {"server": {"per_turn": {"(0, 1)": 2.0, "3": "x"},
                                "held": {"slots": 2, "where": "/m"}},
                     "rows": [{"shown": ["a", "b"]}], "n": None, "ok": True}
    assert json.loads(json.dumps(plain)) == plain     # no default= needed any more


def test_a_run_with_a_field_the_store_could_not_take_is_kept_whole(tmp_path):
    """The twelve runs that were kept as nothing. Whatever is in the record, it is made
    plain before the store sees it, and read back before `save` returns."""
    store = tmp_path / "runs.ladybug"
    row = a_row("who welds?", expected=["person:iris"], shown=["person:iris"])
    key = save(store, [row], held={"context": 32768, "slots": 2,
                                   "concurrency": {"per_turn": {(0, 1): 2.0}},
                                   "where": pathlib.Path("/models/thing.gguf")})
    back = runs(store)
    assert [r["key"] for r in back] == [key]
    assert back[0]["server"]["concurrency"] == {"per_turn": {"(0, 1)": 2.0}}
    assert back[0]["server"]["where"] == "/models/thing.gguf"
    assert back[0]["rows"][0]["question"] == "who welds?"


def test_save_refuses_to_return_a_run_that_did_not_come_back(tmp_path, monkeypatch):
    """The store took twelve runs and gave back nothing for each. What `save` returns is a
    key that `runs` reads, or it is an error."""
    import ml_stack.graph.bench as bench

    store = tmp_path / "runs.ladybug"
    row = a_row("who welds?", expected=["person:iris"], shown=["person:iris"])

    monkeypatch.setattr(bench, "runs", lambda where, label="": [])
    with pytest.raises(bench.RunNotKept, match="did not come back"):
        bench.save(store, [row])

    honest = runs

    def changed(where, label=""):
        return [{**r, "rows": []} for r in honest(where, label)]

    monkeypatch.setattr(bench, "runs", changed)
    with pytest.raises(bench.RunNotKept, match="changed: rows differ"):
        bench.save(store, [row])


def test_a_served_sweep_with_a_store_keeps_every_way_and_reads_each_back(tmp_path, monkeypatch,
                                                                       capsys):
    """The shape that was lost: `sweep --serve` with `--also terse --also card`, a store
    for the finder, a smoke run. Every run comes back from the store whole, and the
    summary the smoke prints is read from the store, not from memory."""
    from contextlib import contextmanager

    import ml_stack.client
    import ml_stack.serve
    import ml_stack.graph.bench as bench

    @contextmanager
    def fake_serve(model, **kw):
        yield type("Up", (), {"base_url": "http://127.0.0.1:1"})()

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench, "footprint", lambda url: {"base_url": url, "context": 32768,
                                                          "slots": 1, "model": "tiny.gguf"})
    monkeypatch.setattr(bench, "find_model", lambda named: named)
    monkeypatch.setattr(ml_stack.serve, "serve", fake_serve)
    monkeypatch.setattr(ml_stack.client, "Client", _ServedModel)
    _preflight_ok(monkeypatch)
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps(TINY))
    asked = tmp_path / "q.jsonl"
    asked.write_text("\n".join(json.dumps({"q": q, "expect": ["topic:compiler"]})
                               for q in ("who works on compilers?", "who else?", "and?")) + "\n")
    kept = tmp_path / "runs.ladybug"

    assert bench.main(["sweep", "--serve", "tiny.gguf", "--plain-only", "--also", "terse",
                       "--also", "card", "--smoke", "--kept", str(kept), "--graph", str(graph),
                       "--questions", str(asked), "--store", str(_tiny_store(tmp_path)),
                       "--serve-port", "1"]) == 0
    said = capsys.readouterr().out
    back = {r["label"]: r for r in runs(kept)}
    assert sorted(back) == ["tiny-plain", "tiny-plain-card", "tiny-plain-terse"]
    for one in back.values():
        assert len(one["rows"]) == 2, "a smoke run, read back with both its questions"
        assert one["server"]["finder"] == "words"
        assert one["server"]["sampling"]
    summary = [ln for ln in said.split("\n---", 1)[1].splitlines() if ln.startswith("tiny-plain")]
    assert len(summary) == 3, said


def test_a_smoke_run_whose_run_does_not_come_back_raises(tmp_path, monkeypatch):
    import ml_stack.graph.bench as bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench, "footprint", lambda url: {"base_url": url})
    monkeypatch.setattr(bench, "ask_from", lambda spec: _Scripted)
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps(TINY))
    asked = tmp_path / "q.jsonl"
    asked.write_text(json.dumps({"q": "who?", "expect": ["topic:compiler"]}) + "\n")
    honest, calls = runs, []

    def then_nothing(where, label=""):
        calls.append(1)
        return honest(where, label) if len(calls) == 1 else []     # save's read-back, then none

    monkeypatch.setattr(bench, "runs", then_nothing)
    with pytest.raises(bench.RunNotKept, match="did not come back"):
        bench.main(["run", "tried", "--smoke", "--kept", str(tmp_path / "runs.ladybug"),
                    "--graph", str(graph), "--questions", str(asked), "--client", "fake:client",
                    "--store", "", "--no-selfcheck"])


def test_empty_runs_are_skipped_named_and_forgotten(tmp_path, capsys):
    from ml_stack.graph.bench import empties, forget
    from ml_stack.graph.store import GraphStore

    store = tmp_path / "runs.ladybug"
    save(store, [a_row("who?", expected=["person:iris"], shown=["person:iris"])])
    with GraphStore(store) as held:
        held.put_doc("bench:hollow:20260901T174747", {})
        held.put_doc("bench:hollow:20260901T180039", {})

    assert [r["label"] for r in runs(store)] == ["tried"], "an empty doc is not a run"
    assert empties(store) == ["bench:hollow:20260901T174747", "bench:hollow:20260901T180039"]

    import ml_stack.graph.bench as bench

    assert bench._main(["show", "--kept", str(store)]) == 0
    assert "2 empty run(s) skipped -- ml-stack-bench forget --empty removes them" \
        in capsys.readouterr().out

    assert bench._main(["forget", "--empty", "--kept", str(store)]) == 0
    assert "2 empty run(s) removed" in capsys.readouterr().out
    assert empties(store) == [] and len(runs(store)) == 1
    assert forget(store, empty=True) == []


def test_forgetting_a_label_lists_first_and_deletes_only_with_yes(tmp_path, capsys):
    import ml_stack.graph.bench as bench

    store = tmp_path / "runs.ladybug"
    save(store, [a_row("who?", expected=["person:iris"], shown=["person:iris"])])
    save(store, [Row(label="other", question="hi")])

    assert bench._main(["forget", "tried", "--kept", str(store)]) == 0
    said = capsys.readouterr().out
    assert "bench:tried:" in said and "pass --yes" in said
    assert len(runs(store)) == 2, "listed, not deleted"

    assert bench._main(["forget", "tried", "--yes", "--kept", str(store)]) == 0
    assert "1 run(s) labelled 'tried' removed" in capsys.readouterr().out
    assert [r["label"] for r in runs(store)] == ["other"]

    assert bench._main(["forget", "--kept", str(store)]) == 2, "say what to forget"


# -- the measuring subcommands detach themselves -------------------------------------------

def test_detach_reruns_the_command_in_its_own_session_with_a_log_of_its_own(tmp_path, monkeypatch,
                                                                          capsys):
    """Not nohup, not &: a sweep started as a backgrounded shell command was killed the
    moment the agent that started it was resumed, and its log was a hand-made redirect
    into a scratch directory."""
    import subprocess
    import sys

    import ml_stack.graph.bench as bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench.platform, "system", lambda: "Darwin")
    started = {}

    class FakeChild:
        pid = 4242

    def fake_popen(command, **kw):
        started["command"], started["kw"] = command, kw
        return FakeChild()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    argv = ["sweep", "--serve", "models/tiny.gguf", "--detach", "--also", "terse", "--smoke"]
    assert bench.main(argv) == 0

    assert started["command"][:3] == [sys.executable, "-m", "ml_stack.graph.bench"]
    assert started["command"][3:] == [a for a in argv if a != "--detach"]
    kw = started["kw"]
    assert kw["start_new_session"] is True and "creationflags" not in kw
    assert kw["stdin"] is subprocess.DEVNULL and kw["stderr"] is subprocess.STDOUT
    log = pathlib.Path(kw["stdout"].name)
    assert log.parent == tmp_path / "home" / "logs"
    assert log.name.startswith("sweep-tiny-") and log.suffix == ".log"
    assert len(log.stem.split("-")[-1]) == 15                       # YYYYmmddTHHMMSS
    assert kw["env"]["PYTHONUNBUFFERED"] == "1"

    held = json.loads((tmp_path / "home" / "measuring.json").read_text())
    assert held["pid"] == 4242 and held["log"] == str(log)
    assert held["argv"] == [a for a in argv if a != "--detach"]
    assert held["started"]

    said = capsys.readouterr().out
    assert str(log) in said
    assert "ml-stack-bench status" in said and "ml-stack-bench tail -f" in said
    assert "ml-stack-bench stop" in said


def test_detach_on_windows_asks_for_a_detached_process_group(tmp_path, monkeypatch):
    import subprocess

    import ml_stack.graph.bench as bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench.platform, "system", lambda: "Windows")
    seen = {}
    monkeypatch.setattr(subprocess, "Popen",
                        lambda command, **kw: seen.update(kw) or type("C", (), {"pid": 7})())
    bench.main(["run", "tried", "--detach"])
    assert seen["creationflags"] == 0x200 | 0x8          # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
    assert "start_new_session" not in seen


def test_the_log_is_named_after_the_label_or_the_first_model():
    from ml_stack.graph.bench import _named_in

    assert _named_in(["run", "with-shortlist", "--smoke"]) == "with-shortlist"
    assert _named_in(["concurrent", "e2b-4x3"]) == "e2b-4x3"
    assert _named_in(["sweep", "--serve", "/m/gemma-tiny.gguf", "--serve", "other"]) == "gemma-tiny"
    assert _named_in(["sweep", "--on", "e4b=http://127.0.0.1:8083"]) == "e4b"
    assert _named_in(["drafts", "hf:someone/tiny/tiny.gguf"]) == "tiny"
    assert _named_in(["sweep"]) == "bench", "nothing to name it after is still a name"


def _measuring(tmp_path, pid, *, log_lines=("first", "second", "third")):
    home = tmp_path / "home"
    (home / "logs").mkdir(parents=True, exist_ok=True)
    log = home / "logs" / "sweep-tiny-20260901T180000.log"
    log.write_text("\n".join(log_lines) + "\n")
    (home / "measuring.json").write_text(json.dumps(
        {"pid": pid, "argv": ["sweep", "--serve", "tiny"], "log": str(log),
         "started": "2026-09-01T18:00:00"}))
    return log


def test_status_says_what_is_measuring_or_that_nothing_is(tmp_path, monkeypatch, capsys):
    import os

    import ml_stack.graph.bench as bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    assert bench.main(["status"]) == 0
    assert capsys.readouterr().out.strip() == "nothing is measuring"

    _measuring(tmp_path, os.getpid())                     # alive: this very process
    assert bench.main(["status"]) == 0
    said = capsys.readouterr().out
    assert f"measuring since 2026-09-01T18:00:00 (pid {os.getpid()})" in said
    assert "ml-stack-bench sweep --serve tiny" in said
    assert "sweep-tiny-20260901T180000.log" in said
    assert "last: third" in said

    _measuring(tmp_path, 2 ** 22 + 12345)                # a pid nobody has
    assert bench.main(["status"]) == 0, "exit 0 either way"
    said = capsys.readouterr().out
    assert said.startswith("nothing is measuring; the last one")
    assert "has ended" in said and "last: third" in said


def test_tail_prints_the_end_of_the_log_and_follows_until_the_pid_is_gone(tmp_path, monkeypatch,
                                                                         capsys):
    import ml_stack.graph.bench as bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    assert bench.main(["tail"]) == 1
    assert "nothing has been detached" in capsys.readouterr().err

    _measuring(tmp_path, 2 ** 22 + 12345, log_lines=("one", "two", "three", "four"))
    assert bench.main(["tail", "-n", "2"]) == 0
    assert capsys.readouterr().out == "three\nfour\n"

    # -f on a measurement that has ended drains what is there and returns
    assert bench.main(["tail", "-n", "1", "-f"]) == 0
    assert capsys.readouterr().out == "four\n"

    # with no measuring.json at all, the latest log under logs/ is the one
    (tmp_path / "home" / "measuring.json").unlink()
    assert bench.main(["tail", "-n", "1"]) == 0
    assert capsys.readouterr().out == "four\n"


def test_stop_signals_the_measuring_pid_and_never_a_name(tmp_path, monkeypatch, capsys):
    import subprocess
    import sys

    import ml_stack.graph.bench as bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    assert bench.main(["stop"]) == 0
    assert capsys.readouterr().out.strip() == "nothing is measuring"

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _measuring(tmp_path, child.pid)
        assert bench.main(["stop"]) == 0
        said = capsys.readouterr().out
        assert said.startswith(f"stopped pid {child.pid}")
        assert child.wait(timeout=10) != 0, "it was signalled, not left to finish"
    finally:
        if child.poll() is None:
            child.kill()
    assert bench.main(["status"]) == 0
    assert capsys.readouterr().out.startswith("nothing is measuring")


def test_a_measuring_command_takes_sigterm_as_an_exit_so_its_server_comes_down(tmp_path,
                                                                             monkeypatch):
    """`stop` sends SIGTERM. Left to the default, that kills the sweep between two lines
    and leaves the model it put up serving under nobody. As an exception, the `with
    serve(...)` runs its exit on the way out."""
    import os
    import signal
    from contextlib import contextmanager

    import ml_stack.client
    import ml_stack.serve
    import ml_stack.graph.bench as bench

    came_down = []

    @contextmanager
    def fake_serve(model, **kw):
        try:
            yield type("Up", (), {"base_url": "http://127.0.0.1:1"})()
        finally:
            came_down.append(model)

    def fake_measure(ask, questions, **kw):
        os.kill(os.getpid(), signal.SIGTERM)            # what `stop` does, from inside
        raise AssertionError("SIGTERM should have raised before this")

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench, "find_model", lambda named: named)
    monkeypatch.setattr(bench, "measure", fake_measure)
    monkeypatch.setattr(ml_stack.serve, "serve", fake_serve)
    monkeypatch.setattr(ml_stack.client, "Client", _ServedModel)
    _preflight_ok(monkeypatch)
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps(TINY))
    asked = tmp_path / "q.jsonl"
    asked.write_text(json.dumps({"q": "who?", "expect": ["topic:compiler"]}) + "\n")
    before = signal.getsignal(signal.SIGTERM)

    with pytest.raises(SystemExit) as left:
        bench.main(["sweep", "--serve", "tiny.gguf", "--plain-only", "--kept",
                    str(tmp_path / "runs.ladybug"), "--graph", str(graph),
                    "--questions", str(asked), "--store", "", "--serve-port", "1",
                    "--no-selfcheck"])
    assert left.value.code == 128 + signal.SIGTERM
    assert came_down == ["tiny.gguf"], "the server was taken down on the way out"
    assert signal.getsignal(signal.SIGTERM) is before, "and the handler was put back"
    from ml_stack.lock import only_one
    with only_one(tmp_path / "home" / "measuring.lock", wait=False):
        pass                                             # and the lock was let go


# -- a sweep resumes -------------------------------------------------------------------------

def test_a_resumed_sweep_measures_only_the_way_it_has_not_kept(tmp_path, monkeypatch, capsys):
    """A sweep killed on its third model, re-run with --resume, costs the third model."""
    import time
    from contextlib import contextmanager

    import ml_stack.client
    import ml_stack.serve
    import ml_stack.graph.bench as bench

    served_models = []

    @contextmanager
    def fake_serve(model, **kw):
        served_models.append(model)
        yield type("Up", (), {"base_url": "http://127.0.0.1:1"})()

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench, "footprint", lambda url: {"base_url": url})
    monkeypatch.setattr(bench, "find_model", lambda named: named)
    monkeypatch.setattr(ml_stack.serve, "serve", fake_serve)
    monkeypatch.setattr(ml_stack.client, "Client", _ServedModel)
    _preflight_ok(monkeypatch)
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps(TINY))
    asked = tmp_path / "q.jsonl"
    asked.write_text(json.dumps({"q": "who works on compilers?", "expect": ["topic:compiler"]})
                     + "\n")
    kept = tmp_path / "runs.ladybug"

    # two of three ways kept today at this context and slot count, one question each
    row = a_row("who works on compilers?", expected=["topic:compiler"], shown=["topic:compiler"])
    for label in ("tiny-plain", "tiny-plain-terse"):
        row.label = label
        save(kept, [row], held={"context": 32768, "slots": 1})
    # and a stale one for the third: a different question count, so it does not count
    row.label = "tiny-plain-card"
    save(kept, [row, a_row("and?", expected=[], shown=[])], held={"context": 32768, "slots": 1})
    # and a whole model kept, every way of it, which is then never loaded
    for label in ("other-plain", "other-plain-terse", "other-plain-card"):
        row.label = label
        save(kept, [row], held={"context": 32768, "slots": 1})

    argv = ["sweep", "--serve", "tiny.gguf", "--serve", "other.gguf", "--plain-only",
            "--also", "terse", "--also", "card", "--kept", str(kept), "--graph", str(graph),
            "--questions", str(asked), "--store", "", "--serve-port", "1", "--no-smoke"]
    assert bench._main([*argv, "--resume"]) == 0
    said = capsys.readouterr().out
    assert "skipping tiny-plain: kept at" in said
    assert "skipping tiny-plain-terse: kept at" in said
    assert "skipping other-plain: kept at" in said
    assert served_models == ["tiny.gguf"], "the model with a way still to measure, once"
    assert len(runs(kept, "tiny-plain-card")) == 2, "the third way was the only one measured"
    assert len(runs(kept, "tiny-plain")) == 1 and len(runs(kept, "other-plain")) == 1

    # --since after everything kept: nothing counts, and both models are served again
    served_models.clear()
    later = time.strftime("%FT%T", time.localtime(time.time() + 3600))
    assert bench._main([*argv, "--resume", "--since", later]) == 0
    assert served_models == ["tiny.gguf", "other.gguf"]
    assert "skipping" not in capsys.readouterr().out

    # and a kept run at another context is another measurement, not this one
    monkeypatch.setattr(bench, "footprint", lambda url: {"base_url": url, "context": 8192,
                                                          "slots": 1})
    already = bench.resumable(kept, questions=1, context=32768, parallel=1)
    assert already("tiny-plain") is not None
    assert bench.resumable(kept, questions=1, context=8192, parallel=1)("tiny-plain") is None
    assert bench.resumable(kept, questions=1, context=65536, parallel=2)("tiny-plain") is None
    assert already("never-run") is None


def test_a_long_label_keeps_its_end_in_the_table():
    """The end of a label is the variant (`-terse`, `-card`); cutting it made three runs
    print as one. Mutation: cut from the end instead of the front."""
    from ml_stack.graph.bench import _shown

    assert _shown("gemma-4-E2B-it-plain-terse", 20).endswith("plain-terse")
    assert _shown("short") == "short"


# -- a load is paid for once, checked first, and written down ------------------------------

def _serving(monkeypatch, tmp_path, *, load_s=12.5, warmup_s=1.2, fail_for=()):
    """Everything a `sweep --serve` or `drafts` needs faked, and what each fake saw.

    The lease yields a real `ServerInfo` carrying ``load_s`` and ``warmup_s``, as the
    backend records them; a model whose name holds one of ``fail_for`` has the lease raise
    `PreflightFailed` instead -- the backend's own preflight refusing what this one passed.
    """
    from contextlib import contextmanager

    import ml_stack.client
    import ml_stack.serve
    import ml_stack.graph.bench as bench
    from ml_stack.serve import ServerInfo
    from ml_stack.serve.preflight import PreflightFailed

    seen = {"models": [], "kwargs": [], "preflights": _preflight_ok(monkeypatch)}

    @contextmanager
    def fake_serve(model, **kw):
        seen["models"].append(model)
        seen["kwargs"].append(dict(kw))
        if any(word in str(model) for word in fail_for):
            raise PreflightFailed(f"FAIL  shards: not on this machine yet: {model}")
        yield ServerInfo(base_url="http://127.0.0.1:1", port=1, pid=None, backend="fake",
                         load_s=load_s, warmup_s=warmup_s)

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench, "footprint", lambda url: {"base_url": url, "context": 32768,
                                                          "slots": 1, "model": "tiny.gguf"})
    monkeypatch.setattr(bench, "find_model", lambda named: named)
    monkeypatch.setattr(ml_stack.serve, "serve", fake_serve)
    monkeypatch.setattr(ml_stack.client, "Client", _ServedModel)
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps(TINY))
    asked = tmp_path / "q.jsonl"
    asked.write_text(json.dumps({"q": "who works on compilers?", "expect": ["topic:compiler"]})
                     + "\n")
    seen["common"] = ["--kept", str(tmp_path / "runs.ladybug"), "--graph", str(graph),
                      "--questions", str(asked), "--store", "", "--serve-port", "1",
                      "--no-smoke"]
    seen["kept"] = tmp_path / "runs.ladybug"
    return seen


def test_every_hf_reference_is_fetched_before_the_lock_and_no_prefetch_skips_it(tmp_path,
                                                                                monkeypatch,
                                                                                capsys):
    """A download inside the timed window is a timing of the network. The fetch is also
    before the *lock*: minutes of Hub and no GPU are not a reason to make the next run wait."""
    import ml_stack.hub
    import ml_stack.graph.bench as bench
    from ml_stack.lock import only_one

    seen = _serving(monkeypatch, tmp_path)
    fetched = []

    def fake_fetch(reference):
        # the measuring lock must not be held while the Hub is being waited on
        with only_one(tmp_path / "home" / "measuring.lock", wait=False):
            pass
        fetched.append(reference)
        where = tmp_path / reference.rsplit("/", 1)[-1]
        where.write_bytes(b"x" * 2048)
        return where

    monkeypatch.setattr(ml_stack.hub, "fetch", fake_fetch)
    argv = ["sweep", "--serve", "hf:someone/tiny-GGUF/tiny.gguf", "--serve", "local.gguf",
            "--serve-draft", "hf:someone/tiny-GGUF/mtp-tiny.gguf", "--plain-only",
            *seen["common"], "--no-selfcheck"]
    assert bench.main(argv) == 0
    said = capsys.readouterr().out
    assert fetched == ["hf:someone/tiny-GGUF/tiny.gguf", "hf:someone/tiny-GGUF/mtp-tiny.gguf"], \
        "each hf: reference once, a local path never"
    # the estimate is said before the fetch -- a refusal after a download is a download
    # for nothing -- and the fetch lines are the first thing after it
    first_lines = [ln for ln in said.splitlines() if not ln.startswith("estimate:")][:2]
    assert first_lines[0].startswith("fetched hf:someone/tiny-GGUF/tiny.gguf: 0.00G at ")
    assert first_lines[1].startswith("fetched hf:someone/tiny-GGUF/mtp-tiny.gguf: 0.00G at ")

    fetched.clear()
    assert bench.main([*argv, "--no-prefetch"]) == 0
    assert fetched == [] and "fetched" not in capsys.readouterr().out


def test_references_are_read_from_every_measuring_subcommand():
    from argparse import Namespace

    from ml_stack.graph.bench import references_in

    assert references_in(Namespace(serve=["hf:a/b/c.gguf", "d.gguf"], serve_draft=["auto", ""])) \
        == ["hf:a/b/c.gguf"]
    assert references_in(Namespace(model="hf:a/b/c.gguf", draft=["hf:a/b/mtp.gguf", "",
                                                                  "hf:a/b/c.gguf"])) \
        == ["hf:a/b/c.gguf", "hf:a/b/mtp.gguf"], "the model, then each head, each once"
    assert references_in(Namespace(label="x")) == [], "run has nothing to fetch"


def test_a_fetch_that_fails_is_said_and_the_rest_still_come_down(capsys):
    import ml_stack.hub
    import ml_stack.graph.bench as bench

    def flaky(reference):
        if "gone" in reference:
            raise OSError("no such repository")
        return pathlib.Path("/dev/null")

    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(ml_stack.hub, "fetch", flaky)
        got = bench.prefetch(["hf:a/gone/x.gguf", "hf:a/b/c.gguf"])
    assert [ref for ref, _ in got] == ["hf:a/b/c.gguf"]
    assert "could not fetch hf:a/gone/x.gguf: no such repository" in capsys.readouterr().err


def test_the_preflight_is_printed_under_the_up_line_and_kept_beside_the_measured_cache(
        tmp_path, monkeypatch, capsys):
    """The estimate and the measurement on adjacent lines is the point: a KV estimate that
    reads 3G against a `kv+run` that measures 9G is a model whose runtime is not the cache."""
    import ml_stack.graph.bench as bench

    seen = _serving(monkeypatch, tmp_path)
    assert bench._main(["sweep", "--serve", "tiny.gguf", "--plain-only", *seen["common"]]) == 0
    said = capsys.readouterr().out.splitlines()
    up = next(i for i, ln in enumerate(said) if ln.strip().startswith("up in "))
    assert "load 12.5s" in said[up] and "warm-up 1.2s" in said[up]
    assert said[up + 1].strip() == "ok    shards: complete"
    assert said[up + 3].strip().startswith("ok    fit: 8.0G estimated fits under 110.0G")
    assert len(seen["preflights"]) == 1
    spec = seen["preflights"][0]
    assert spec.model == "tiny.gguf" and spec.context == 32768 and spec.port == 1

    back = runs(seen["kept"])[0]["server"]
    assert back["preflight"] == {"kv_estimate_bytes": 3 * 2**30, "weights_bytes": 5 * 2**30,
                                 "ok": True}


def test_a_refused_preflight_skips_the_model_and_the_sweep_goes_on(tmp_path, monkeypatch,
                                                                    capsys):
    """A sweep of five must not end on the one that does not fit. Two refusals, both
    caught per model: this preflight's own, and the backend's raised out of the lease."""
    import ml_stack.graph.bench as bench

    seen = _serving(monkeypatch, tmp_path, fail_for=("wrongbuild",))
    _preflight_ok(monkeypatch, refuse=("toobig",))
    assert bench._main(["sweep", "--serve", "toobig.gguf", "--serve", "wrongbuild.gguf",
                        "--serve", "tiny.gguf", "--plain-only", *seen["common"]]) == 0
    said = capsys.readouterr().out
    assert "preflight refused toobig; not loaded:" in said
    assert "FAIL  shards: missing or empty: toobig.gguf" in said
    assert "preflight refused wrongbuild; not loaded:" in said
    assert "not on this machine yet: wrongbuild.gguf" in said
    assert seen["models"] == ["wrongbuild.gguf", "tiny.gguf"], \
        "the one this refused was never leased; the one the lease refused did not stop the rest"
    assert [r["label"] for r in runs(seen["kept"])] == ["tiny-plain"]


def test_a_sweep_refused_everywhere_still_prints_its_table(tmp_path, monkeypatch, capsys):
    import ml_stack.graph.bench as bench

    seen = _serving(monkeypatch, tmp_path)
    _preflight_ok(monkeypatch, refuse=("tiny",))
    assert bench._main(["sweep", "--serve", "tiny.gguf", "--plain-only", *seen["common"]]) == 0
    assert "nothing kept yet" in capsys.readouterr().out
    assert not seen["kept"].exists()


def test_the_load_is_the_leases_own_clock_and_shows_everywhere_a_run_does(tmp_path, monkeypatch,
                                                                         capsys):
    """Not a stopwatch around `serve()`: that also holds an adopted server's nothing and a
    warm-up's something. Blank for a run kept before the lease recorded it, and `n` stays
    the fourth word of every row, old or new."""
    import ml_stack.graph.bench as bench
    from ml_stack.graph.bench import SHORT, export, invented_digest, missed, ranking

    seen = _serving(monkeypatch, tmp_path, load_s=41.6, warmup_s=None)
    assert bench._main(["sweep", "--serve", "tiny.gguf", "--plain-only", *seen["common"]]) == 0
    capsys.readouterr()
    back = runs(seen["kept"])[0]["server"]
    assert back["load_s"] == 41.6 and back["warmup_s"] is None

    # an older run beside it, kept before the lease said how long it took
    save(seen["kept"], [a_row("who?", expected=["person:iris"], shown=["person:iris"])],
         held={"context": 32768, "slots": 1})
    table(runs(seen["kept"]))
    said = capsys.readouterr().out
    head = said.splitlines()[0].split()
    assert head.index("load") == head.index("wall") + 1, "next to wall"
    new = next(ln for ln in said.splitlines() if ln.startswith("tiny-plain"))
    old = next(ln for ln in said.splitlines() if ln.startswith("tried"))
    assert new.split()[3] == "1" and old.split()[3] == "1", "n is the fourth word of both"
    assert new.split()[5] == "42s", "the load, rounded, after the wall clock"
    assert old.split()[5] != "42s" and not old.split()[5].endswith("s"), "blank, not 0s"

    missed(runs(seen["kept"], "tiny-plain"), everything=True)
    assert "load 42s" in capsys.readouterr().out.splitlines()[1]

    rows = [a_row(f"q{n}?", expected=["person:iris"], shown=["person:iris"])
            for n in range(SHORT)]
    save(seen["kept"], rows, held={"graph": invented_digest(), "model": "thing.gguf",
                                   "load_s": 41.6})
    got = json.loads(pathlib.Path(export(runs(seen["kept"]), tmp_path / "o.json")).read_text())
    assert {r["load_s"] for r in got} == {41.6}
    ranked = ranking(runs(seen["kept"]))
    assert "| load |" in ranked and "| 42s |" in ranked


def test_drafts_serves_each_head_once_per_n_max_and_the_baseline_once(tmp_path, monkeypatch,
                                                                     capsys):
    """`--spec-draft-n-max` is bound at start like the head, so N lengths is N servers --
    labelled so the table shows acceptance and wall per (head, n-max)."""
    import ml_stack.graph.bench as bench

    seen = _serving(monkeypatch, tmp_path)
    kept = tmp_path / "runs.ladybug"
    asked = tmp_path / "q.jsonl"
    asked.write_text(json.dumps({"q": "who works on compilers?", "expect": ["topic:compiler"]})
                     + "\n")
    assert bench._main(["drafts", "tiny.gguf", "--draft", "", "--draft", "mtp-tiny.gguf",
                        "--n-max", "4", "--n-max", "8", "--smoke", "--kept", str(kept),
                        "--questions", str(asked), "--port", "1"]) == 0
    said = capsys.readouterr().out
    assert sorted(r["label"] for r in runs(kept)) == sorted(
        ["draft:none", "draft:mtp-tiny@n4", "draft:mtp-tiny@n8"])
    assert seen["models"] == ["tiny.gguf"] * 3
    assert [kw.get("spec_draft_max") for kw in seen["kwargs"]] == [None, 4, 8]
    assert [kw.get("draft") for kw in seen["kwargs"]] == [None, "mtp-tiny.gguf", "mtp-tiny.gguf"]
    by_label = {r["label"]: r["server"] for r in runs(kept)}
    assert by_label["draft:mtp-tiny@n8"]["spec_draft_max"] == 8
    assert "spec_draft_max" not in by_label["draft:none"]
    assert "--- draft: mtp-tiny@n4" in said and "--- draft: mtp-tiny@n8" in said
    # the drafts summary comes first -- one row per (head, n-max), the baseline named --
    # and then the table of every run kept
    worth, then = said.split("\n" + "draft" + " " * 24, 1)[1].split("\nrun ", 1)
    heads = [ln for ln in worth.splitlines() if ln.startswith("draft:")]
    assert [ln.split()[0] for ln in heads] == ["draft:mtp-tiny@n4", "draft:mtp-tiny@n8"] \
        or [ln.split()[0] for ln in heads] == ["draft:mtp-tiny@n8", "draft:mtp-tiny@n4"]
    assert all(ln.rstrip().endswith("draft:none") for ln in heads), "against the baseline"
    summary = [ln for ln in then.splitlines() if ln.startswith("draft:")]
    assert len(summary) == 3, "one row per (head, n-max), and the baseline"

    # without --n-max: once, at the build's own default, and the label says nothing of it
    seen["models"].clear()
    seen["kwargs"].clear()
    assert bench._main(["drafts", "tiny.gguf", "--draft", "mtp-tiny.gguf", "--smoke",
                        "--kept", str(kept), "--questions", str(asked), "--port", "1"]) == 0
    capsys.readouterr()
    assert seen["models"] == ["tiny.gguf"] and "spec_draft_max" not in seen["kwargs"][0]
    assert len(runs(kept, "draft:mtp-tiny")) == 1


def test_a_quantised_cache_is_on_the_spec_the_label_and_the_ctx_column(tmp_path, monkeypatch,
                                                                       capsys):
    """A run with a q8 cache against one at f16 is two configurations. The label carries it
    where the variant lives, at the end, and `ctx` shows it beside the context it sizes."""
    import ml_stack.graph.bench as bench
    from ml_stack.graph.bench import kv_short

    seen = _serving(monkeypatch, tmp_path)
    assert bench._main(["sweep", "--serve", "tiny.gguf", "--plain-only", "--also", "terse",
                        "--serve-kv", "q8_0", *seen["common"]]) == 0
    capsys.readouterr()
    assert seen["kwargs"][0]["cache_type_k"] == "q8_0"
    assert seen["kwargs"][0]["cache_type_v"] == "q8_0"
    assert seen["preflights"][0].cache_type_k == "q8_0", "and the preflight sized that cache"
    assert sorted(r["label"] for r in runs(seen["kept"])) == \
        ["tiny-plain-kv-q8_0", "tiny-plain-terse-kv-q8_0"]
    assert all(r["server"]["cache_type"] == "q8_0" for r in runs(seen["kept"]))

    save(seen["kept"], [a_row("who?", expected=["person:iris"], shown=["person:iris"])],
         held={"context": 32768, "slots": 1})
    table(runs(seen["kept"]))
    said = capsys.readouterr().out
    quantised = next(ln for ln in said.splitlines() if ln.startswith("tiny-plain-kv"))
    plain = next(ln for ln in said.splitlines() if ln.startswith("tried"))
    assert quantised.split()[1:4] == ["32k", "x1/q8", "1"], "ctx says so, n still fourth"
    assert plain.split()[1:4] == ["32k", "x1", "1"]

    assert kv_short("q8_0") == "q8" and kv_short("q4_0") == "q4"
    assert kv_short("q5_1") == "q5_1", "q5_0 also exists; two types as one is the bug"
    assert kv_short("f16") == "f16" and kv_short("") == ""


def test_shortlist_for_gives_both_halves_to_the_models_named_in_one_load(tmp_path, monkeypatch,
                                                                         capsys):
    """One load per model, whatever it is asked: the shortlist half used to cost a second
    load that measured nothing about the asking. `--shortlist-for` narrows who gets it."""
    import ml_stack.graph.bench as bench

    seen = _serving(monkeypatch, tmp_path)
    assert bench._main(["sweep", "--serve", "gemma-E2B.gguf", "--serve", "other.gguf",
                        "--shortlist-for", "e2b,e4b", "--shortlist", "5",
                        *seen["common"]]) == 0
    said = capsys.readouterr().out
    assert seen["models"] == ["gemma-E2B.gguf", "other.gguf"], "each once, both halves inside"
    assert sorted(r["label"] for r in runs(seen["kept"])) == \
        ["gemma-E2B-plain", "gemma-E2B-shortlist", "other-plain"]
    assert "gemma-E2B: plain, shortlist" in said and "other: plain" in said

    # without --shortlist-for every model gets both halves, still in one load each
    seen["models"].clear()
    assert bench._main(["sweep", "--serve", "other.gguf", *seen["common"]]) == 0
    capsys.readouterr()
    assert seen["models"] == ["other.gguf"]
    assert len(runs(seen["kept"], "other-shortlist")) == 1

    # and --plain-only still means none, named or not
    seen["models"].clear()
    assert bench._main(["sweep", "--serve", "gemma-E2B.gguf", "--shortlist-for", "e2b",
                        "--plain-only", *seen["common"]]) == 0
    capsys.readouterr()
    assert len(runs(seen["kept"], "gemma-E2B-shortlist")) == 1, "no second shortlist run"
    assert len(runs(seen["kept"], "gemma-E2B-plain")) == 2


def test_halves_and_the_ways_they_cross():
    from argparse import Namespace

    from ml_stack.graph.bench import _asked, halves

    args = Namespace(plain_only=False, shortlist_for="e2b,E4B", shortlist=8, also=["terse"],
                     terse=False)
    assert halves(args, "gemma-4-E2B-it") == [("plain", 0), ("shortlist", 8)]
    assert halves(args, "/models/gemma-4-e4b-it.gguf") == [("plain", 0), ("shortlist", 8)]
    assert halves(args, "gpt-oss-120b") == [("plain", 0)]
    assert halves(Namespace(plain_only=False, shortlist_for="", shortlist=8), "anything") \
        == [("plain", 0), ("shortlist", 8)]
    assert halves(Namespace(plain_only=True, shortlist_for="e2b", shortlist=8), "e2b") \
        == [("plain", 0)]
    ways = _asked(args, halves(args, "e2b"))
    assert [(w["label"], w["shortlist"], w["terse"]) for w in ways] == [
        ("plain", 0, False), ("plain-terse", 0, True),
        ("shortlist", 8, False), ("shortlist-terse", 8, True)]


# -- the ranking composes accuracy and cost --------------------------------------------------
#
# Adam, 2026-09-01: "how can we properly rank flash-next if we haven't figured out the draft
# head?" A head cannot change an answer, only the clock, so a model's accuracy is its largest
# run and its cost is the fastest run that held that accuracy -- a short drafted run, on a
# fork, included -- rather than both from one run.

def _kept_run(store, label, *, model, questions, hits, seconds, binary="", **server):
    """A run of ``questions`` over the invented community, ``hits`` of them answered in
    full, taking ``seconds`` altogether, served by ``binary``."""
    from ml_stack.graph.bench import invented_digest

    rows = [a_row(f"q{n}?", expected=["person:iris"], shown=["person:iris"] if n < hits else [])
            for n in range(questions)]
    for r in rows:
        r.label, r.seconds = label, seconds / questions
    return save(store, rows, held={"graph": invented_digest(), "model": model,
                                   "binary": binary, **server})


def test_a_models_cost_comes_from_its_fastest_run_that_held_its_accuracy(tmp_path):
    """A 34-question undrafted run on mainline says how well flash answers; a 20-question
    drafted run on a fork, F1 held, says what it costs -- per question, so the two compare."""
    from ml_stack.graph.bench import ranking

    store = tmp_path / "runs.ladybug"
    _kept_run(store, "flash-plain", model="flash.gguf", questions=34, hits=20, seconds=500.0,
              binary="/builds/current/llama-server", resident_bytes=8 * 2**30,
              kv_and_run_bytes=3 * 2**30, load_s=30.0)
    _kept_run(store, "draft:mtp-tiny@n4", model="flash.gguf", questions=20, hits=12,
              seconds=200.0, binary="/builds/brayfork/llama-server", resident_bytes=9 * 2**30,
              kv_and_run_bytes=4 * 2**30, load_s=40.0)
    said = ranking(runs(store))
    row = next(ln for ln in said.splitlines() if ln.startswith("| `flash.gguf`"))
    assert "| 59% |" in row, "accuracy is the 34-question run's (20 of 34), not the drafted 60%"
    assert "| 34 |" in row and "| 10.0 |" in row, "cost is the drafted run's, per question"
    assert "| 40s | 9.0G | 4.0G |" in row, "load, resident and kv+run from the same run"
    assert row.endswith("| `draft:mtp-tiny@n4` on brayfork/llama-server (20 q) |")
    assert "rejected" not in said
    assert "| s/question |" in said and "| cost from |" in said
    assert said.startswith("# Which model answers best\n")
    assert "Accuracy is each model's largest run" in said and "Cost is the model's" in said


def test_a_drafted_run_whose_f1_fell_is_rejected_and_named(tmp_path):
    from ml_stack.graph.bench import ranking

    store = tmp_path / "runs.ladybug"
    _kept_run(store, "flash-plain", model="flash.gguf", questions=34, hits=20, seconds=500.0)
    _kept_run(store, "draft:mtp-tiny@n8", model="flash.gguf", questions=20, hits=10,
              seconds=120.0)
    said = ranking(runs(store))
    row = next(ln for ln in said.splitlines() if ln.startswith("| `flash.gguf`"))
    assert "| 14.7 |" in row and row.endswith("| its own run |"), \
        "the fast run did not hold the accuracy, so the cost is the accuracy run's own"
    assert "- `flash.gguf` rejected: `draft:mtp-tiny@n8` F1 -9 pts (20 q, 6.0 s/question)" in said
    # widen the noise and the same run supplies the cost
    wider = ranking(runs(store), noise=0.10)
    assert "rejected" not in wider and "| 6.0 |" in wider


def test_a_model_with_one_run_uses_it_and_says_so(tmp_path):
    from ml_stack.graph.bench import ranking

    store = tmp_path / "runs.ladybug"
    _kept_run(store, "only", model="one.gguf", questions=20, hits=10, seconds=100.0,
              binary="/builds/current/llama-server")
    said = ranking(runs(store))
    row = next(ln for ln in said.splitlines() if ln.startswith("| `one.gguf`"))
    assert "| 50% |" in row and "| 5.0 |" in row
    assert row.endswith("| its own run on current/llama-server |")


def test_a_smoke_run_never_supplies_cost_and_the_footnote_counts_it(tmp_path):
    from ml_stack.graph.bench import SHORT, ranking

    store = tmp_path / "runs.ladybug"
    _kept_run(store, "flash-plain", model="flash.gguf", questions=34, hits=20, seconds=500.0)
    _kept_run(store, "draft:mtp-tiny@n4", model="flash.gguf", questions=2, hits=2, seconds=2.0)
    said = ranking(runs(store))
    row = next(ln for ln in said.splitlines() if ln.startswith("| `flash.gguf`"))
    assert "| 14.7 |" in row and "its own run" in row, "two questions at a second each is not a cost"
    assert f"*1 run(s) not ranked: fewer than {SHORT} questions" in said
    assert "supplies neither accuracy nor cost" in said


def test_accuracy_is_the_largest_run_then_the_best_then_the_newest():
    from ml_stack.graph.bench import choices, invented_digest

    def run(label, at, hits, seconds):
        rows = [{"expected": ["a"], "shown": ["a"] if n < hits else [], "seconds": seconds / 20}
                for n in range(20)]
        return {"label": label, "at": at, "rows": rows,
                "server": {"model": "m.gguf", "graph": invented_digest()}}

    older, newer = run("older", "2026-08-01T00:00:00", 12, 100.0), \
        run("newer", "2026-09-01T00:00:00", 12, 150.0)
    chosen, too_few = choices([older, newer])
    assert too_few == 0 and len(chosen) == 1
    assert chosen[0].accuracy is newer, "the same size: the newest"
    assert chosen[0].cost is older, "and the older one, at the same F1, was faster"
    assert not chosen[0].own and chosen[0].rejected == []

    # the askings are configurations a person chooses between: the best F1 among the
    # largest runs wins, however old -- the newest was a card run once (2026-09-01)
    better = run("plain", "2026-07-01T00:00:00", 15, 120.0)
    chosen, _ = choices([older, newer, better])
    assert chosen[0].accuracy is better, "the best F1 among the same size, not the newest"


def test_the_composed_frontier_holds_the_composed_point(tmp_path, capsys):
    from ml_stack.graph.bench import composed, pareto, plot, rates

    store = tmp_path / "runs.ladybug"
    _kept_run(store, "flash-plain", model="flash.gguf", questions=34, hits=20, seconds=500.0)
    _kept_run(store, "draft:mtp-tiny@n4", model="flash.gguf", questions=20, hits=11,
              seconds=200.0)
    kept = runs(store)
    points = composed(kept)
    assert len(points) == 1 and points[0]["composed"] and points[0]["from"] == "draft:mtp-tiny@n4"
    got = points[0]["derived"]
    assert got["questions"] == 34 and got["seconds"] == pytest.approx(340.0), \
        "the drafted run's 10 s/question over the accuracy run's 34 questions"
    assert got["right"] == pytest.approx(20 / 34)
    # 59% at 340 s: the undrafted run is as accurate and slower, so it falls off; the
    # drafted run at 55% is cheaper and worse, so both it and the composed point stay
    on = pareto(kept + points, cost="seconds")
    assert any(o.get("composed") for o in on)
    assert not any(o.get("label") == "flash-plain" for o in on), "dominated by its own model"

    rates(kept)
    said = capsys.readouterr().out
    assert any(ln.startswith("flash.gguf") and "*=" in ln for ln in said.splitlines()), \
        "the composed point, marked as composed and on the frontier"
    assert "= a model composed" in said

    drawn = pathlib.Path(plot(kept, tmp_path / "f.html")).read_text()
    assert 'class="composed' in drawn and "cost from draft:mtp-tiny@n4" in drawn


def test_an_export_carries_the_binary_and_the_timeouts(tmp_path):
    from ml_stack.graph.bench import export

    store = tmp_path / "runs.ladybug"
    _kept_run(store, "only", model="one.gguf", questions=20, hits=10, seconds=100.0,
              binary="/builds/fork/llama-server")
    got = json.loads(pathlib.Path(export(runs(store), tmp_path / "o.json")).read_text())
    assert got[0]["binary"] == "/builds/fork/llama-server" and got[0]["timed_out"] == 0


# -- a question that runs past --per-question -------------------------------------------------

class _Stalling:
    """A client whose first call takes longer than the question is allowed and then fails
    the way urllib does at its socket timeout; every later call answers at once."""

    def __init__(self, stall: float) -> None:
        self.stall = stall
        self.timeout = 180.0            # the real client has one, so the cap is handed down
        self.given: list[float | None] = []
        self.calls = 0
        self.sampling: dict = {}

    def chat(self, messages, **kw):
        from ml_stack.client import ServerUnreachable

        self.calls += 1
        self.given.append(kw.get("timeout"))
        if self.calls == 1:
            import time

            time.sleep(self.stall)
            raise ServerUnreachable("cannot reach http://127.0.0.1:1/completion (timed out)")
        return type("R", (), {"content": "a compiler person", "thinking": None, "raw": {},
                              "tool_calls": None})()


def test_a_question_past_the_cap_is_kept_as_timed_out_and_the_run_moves_on(capsys):
    from ml_stack.graph.bench import measure

    client = _Stalling(stall=0.3)

    def ask(question, counting):
        return counting.chat([{"role": "user", "content": question}])

    rows = measure(ask, [{"q": "who?", "expect": ["person:iris"]},
                         {"q": "and who else?", "expect": ["person:iris"]}],
                   label="t", client=client, log=print, per_question=0.25)
    first, second = rows
    assert first.timed_out and first.seconds == 0.25 and first.shown == []
    assert first.error == "timed out after 0s" and first.hit == 0.0, "no answer, scored wrong"
    assert client.given[0] == pytest.approx(0.25, abs=0.05), "the cap is the call's timeout"
    assert not second.timed_out and second.error == "", "the next question was asked"
    assert "TIMED OUT" in capsys.readouterr().out


def test_a_deadline_already_spent_stops_the_next_call_before_it_is_made():
    """Three calls get one cap between them, not three."""
    from ml_stack.graph.bench import Counting, QuestionTimedOut

    class Prompt:
        def chat(self, messages, **kw):
            return type("R", (), {"raw": {}})()

    import time

    counting = Counting(Prompt(), deadline=time.time() - 1)
    with pytest.raises(QuestionTimedOut):
        counting.chat([])
    assert counting.timed_out


def test_an_ask_that_swallows_the_timeout_is_still_scored_wrong():
    from ml_stack.graph.bench import measure

    client = _Stalling(stall=0.3)

    def ask(question, counting):
        try:
            counting.chat([])
        except Exception:
            pass
        return {"content": "Iris Tamsin, probably", "show": ["person:iris"]}

    row, = measure(ask, [{"q": "who?", "expect": ["person:iris"]}], label="t", client=client,
                   per_question=0.25)
    assert row.timed_out and row.shown == [] and row.answer_chars == 0 and row.hit == 0.0


def test_the_table_counts_timeouts_and_the_detail_names_them(tmp_path, capsys):
    from ml_stack.graph.bench import missed

    store = tmp_path / "runs.ladybug"
    rows = [a_row("who?", expected=["person:iris"], shown=[]),
            a_row("and who else?", expected=["person:iris"], shown=["person:iris"])]
    rows[0].timed_out, rows[0].seconds, rows[0].error = True, 300.0, "timed out after 300s"
    save(store, rows, held={"context": 32768, "slots": 1})
    table(runs(store))
    said = capsys.readouterr().out
    head = said.splitlines()[0].split()
    assert head.index("t/o") == head.index("made") + 1
    line = next(ln for ln in said.splitlines() if ln.startswith("tried"))
    assert line.split()[-2] == "1", "one timed out, before the sampling"
    missed(runs(store))
    said = capsys.readouterr().out
    assert "1 timed out)" in said.splitlines()[1]
    assert "who?   [timed out at 300s]" in said and "ERROR timed out after 300s" in said


def test_per_question_reaches_the_client_and_the_measuring(tmp_path, monkeypatch, capsys):
    import ml_stack.graph.bench as bench

    seen = _serving(monkeypatch, tmp_path)
    assert bench._main(["sweep", "--serve", "tiny.gguf", "--plain-only", "--per-question", "42",
                        *seen["common"]]) == 0
    capsys.readouterr()
    back = runs(seen["kept"])[0]
    assert back["server"]["sampling"]["timeout"] == 42.0, "the served client was built with it"
    assert all(r["timed_out"] is False for r in back["rows"])
    assert "binary" in back["server"], "which llama-server served it is on the record"


# -- a reasoning budget on every served model ------------------------------------------------

def test_a_reasoning_budget_is_on_the_spec_the_label_and_the_ctx_column(tmp_path, monkeypatch,
                                                                       capsys):
    """A ceiling cuts the answer; a budget stops the thinking. Bound at start like the cache
    type, so it is on the label and beside the context it was served with."""
    import ml_stack.graph.bench as bench

    seen = _serving(monkeypatch, tmp_path)
    assert bench._main(["sweep", "--serve", "tiny.gguf", "--plain-only", "--also", "terse",
                        "--reasoning-budget", "2048", *seen["common"]]) == 0
    capsys.readouterr()
    assert seen["kwargs"][0]["reasoning_budget"] == 2048
    assert seen["preflights"][0].reasoning_budget == 2048, "and the preflight saw the flag"
    assert sorted(r["label"] for r in runs(seen["kept"])) == \
        ["tiny-plain-rb2048", "tiny-plain-terse-rb2048"]
    assert all(r["server"]["reasoning_budget"] == 2048 for r in runs(seen["kept"]))

    save(seen["kept"], [a_row("who?", expected=["person:iris"], shown=["person:iris"])],
         held={"context": 32768, "slots": 1})
    table(runs(seen["kept"]))
    said = capsys.readouterr().out
    budgeted = next(ln for ln in said.splitlines() if ln.startswith("tiny-plain-rb"))
    plain = next(ln for ln in said.splitlines() if ln.startswith("tried"))
    assert budgeted.split()[1:4] == ["32k", "x1/rb", "1"], "ctx says so, n still fourth"
    assert plain.split()[1:4] == ["32k", "x1", "1"]

    # without the flag nothing is bound and no label says so
    seen["kwargs"].clear()
    assert bench._main(["sweep", "--serve", "other.gguf", "--plain-only", *seen["common"]]) == 0
    capsys.readouterr()
    assert "reasoning_budget" not in seen["kwargs"][0]
    assert runs(seen["kept"], "other-plain")[0]["server"].get("reasoning_budget") is None


def test_a_model_that_will_not_load_ends_that_model_and_not_the_sweep(monkeypatch, tmp_path, capsys):
    """Measured 2026-09-01: Flash-Next's head failed to load on mainline and the crash took
    gpt-oss-120b's measurement down with it, twice. Mutation: drop the except."""
    from ml_stack.graph import bench
    from ml_stack.serve.backend import ServerFailed

    calls = []

    def fake_served(model, *a, **k):
        calls.append(model)
        if "bad" in model:
            raise ServerFailed("llama-server did not become healthy (exited 1)")
        return []

    monkeypatch.setattr(bench, "served", fake_served)
    monkeypatch.setattr(bench, "_kept", lambda kept: [])
    monkeypatch.setattr(bench, "runs", lambda *a, **k: [])
    monkeypatch.setattr(bench, "busy", lambda url: 0)
    monkeypatch.setattr(bench, "prepared", lambda: "")
    graph = tmp_path / "g.json"
    graph.write_text('{"nodes": [], "edges": []}')
    code = bench._main(["sweep", "--serve", "bad.gguf", "--serve", "good.gguf", "--plain-only",
                        "--graph", str(graph), "--kept", str(tmp_path / "runs.ladybug"),
                        "--no-prefetch", "--smoke"])
    said = capsys.readouterr().out
    assert calls == ["bad.gguf", "good.gguf"], said
    assert "did not load; moving on" in said


# -- what a draft head was worth, as a number ----------------------------------------------

def _measured(label, *, model="flash.gguf", questions=20, hits=12, seconds=200.0,
              binary="/builds/current/llama-server", draft="", at="2026-09-01T12:00:00",
              guessed=0, taken=0):
    """A kept run as `runs` hands it back, built by hand so ``at`` is chosen and not the
    clock's: ``hits`` of ``questions`` answered in full over ``seconds`` altogether."""
    from dataclasses import asdict

    rows = []
    for n in range(questions):
        r = a_row(f"q{n}?", expected=["person:iris"], shown=["person:iris"] if n < hits else [])
        r.label, r.seconds = label, seconds / questions
        r.draft_tokens, r.draft_taken = guessed, taken
        rows.append(asdict(r))
    server = {"model": model, "binary": binary, "context": 32768, "slots": 1}
    if draft:
        server["draft_model"] = draft
    return {"key": f"bench:{label}:{at}", "at": at, "label": label, "server": server,
            "rows": rows}


def test_speedup_is_the_newest_same_model_same_build_same_size_undrafted_run_over_this_one():
    """A drafted run at 7.04 s/question against its baseline's 10.0 is 1.42x -- against
    the *newest* undrafted run of the same model, build and size, and none other."""
    from ml_stack.graph.bench import baseline, speedup

    older = _measured("draft:none", seconds=300.0, at="2026-09-01T10:00:00")
    newest = _measured("draft:none", seconds=200.0, at="2026-09-01T11:00:00")
    fork = _measured("draft:none", seconds=100.0, binary="/builds/brayfork/llama-server")
    larger = _measured("flash-plain", questions=34, hits=20, seconds=170.0)
    other = _measured("draft:none", model="tiny.gguf", seconds=50.0)
    drafted = _measured("draft:mtp-flash@n4", seconds=140.8, draft="mtp-flash.gguf",
                        guessed=100, taken=76)
    kept = [older, fork, larger, other, drafted, newest]
    assert baseline(drafted, kept) is newest
    assert speedup(drafted, kept) == pytest.approx(200.0 / 140.8)
    assert f"{speedup(drafted, kept):.2f}x" == "1.42x"
    # an undrafted run is its own baseline, which is no baseline at all
    assert speedup(newest, kept) is None and baseline(newest, kept) is None
    # a fork's baseline says nothing about mainline's head, nor a larger run about a smaller
    assert speedup(drafted, [fork, larger, other, drafted]) is None
    # the same model on no named build pairs with a baseline on no named build
    bare = _measured("draft:mtp-flash@n4", seconds=100.0, draft="mtp-flash.gguf", binary="")
    assert speedup(bare, [newest, bare]) is None
    assert speedup(bare, [_measured("draft:none", binary=""), bare]) == pytest.approx(2.0)
    # a run that took no time has nothing to divide by
    still = _measured("draft:mtp-flash@n4", seconds=0.0, draft="mtp-flash.gguf")
    assert speedup(still, [newest, still]) is None


def test_the_table_prints_speed_after_draft_and_leaves_it_blank_without_a_baseline(tmp_path,
                                                                                    capsys):
    from ml_stack.graph.bench import invented_digest

    store = tmp_path / "runs.ladybug"
    _kept_run(store, "draft:none", model="flash.gguf", questions=20, hits=12, seconds=200.0,
              binary="/builds/current/llama-server")
    _kept_run(store, "draft:mtp-flash@n4", model="flash.gguf", questions=20, hits=12,
              seconds=140.8, binary="/builds/current/llama-server",
              draft_model="mtp-flash.gguf")
    _kept_run(store, "draft:mtp-flash@n8", model="flash.gguf", questions=20, hits=12,
              seconds=100.0, binary="/builds/brayfork/llama-server",
              draft_model="mtp-flash.gguf")
    table(runs(store))
    said = capsys.readouterr().out
    head = said.splitlines()[0].split()
    assert head.index("speed") == head.index("draft") + 1
    lines = {ln.split()[0]: ln for ln in said.splitlines() if ln.startswith("draft:")}
    assert "1.42x" in lines["draft:mtp-flash@n4"]
    assert not any(w.endswith("x") for w in lines["draft:none"].split()), "no head, no speed"
    assert not any(w.endswith("x") for w in lines["draft:mtp-flash@n8"].split()), \
        "the fork's run has no baseline on the fork"
    assert invented_digest()  # the runs above are over the invented community


def test_the_detail_says_the_speedup_on_the_run_line(tmp_path, capsys):
    import ml_stack.graph.bench as bench

    store = tmp_path / "runs.ladybug"
    _kept_run(store, "draft:none", model="flash.gguf", questions=20, hits=12, seconds=200.0)
    _kept_run(store, "draft:mtp-flash@n4", model="flash.gguf", questions=20, hits=12,
              seconds=140.8, draft_model="mtp-flash.gguf")
    assert bench._main(["show", "--detail", "draft:mtp-flash@n4", "--kept", str(store)]) == 0
    said = capsys.readouterr().out
    line = next(ln for ln in said.splitlines() if ln.startswith("draft:mtp-flash@n4"))
    assert line.endswith("speedup 1.42x over draft:none (draft:none)")
    assert "draft:none  (" not in said, "only the label asked for is detailed"
    # and --detail with no label details every run, the baseline saying nothing of speed
    assert bench._main(["show", "--detail", "--kept", str(store)]) == 0
    said = capsys.readouterr().out
    base = next(ln for ln in said.splitlines() if ln.startswith("draft:none"))
    assert "speedup" not in base and "speedup 1.42x" in said


def test_the_drafts_summary_is_sorted_by_speedup_and_recommends_the_fastest_that_held():
    """Three heads: the fastest lost twenty points of F1, so it is on the table and not in
    the recommendation; the next fastest held within the noise and is what to serve."""
    from ml_stack.graph.bench import NOISE, drafted

    base = _measured("draft:none", hits=12, seconds=200.0, at="2026-09-01T11:00:00")
    slow = _measured("draft:mtp-a@n4", hits=12, seconds=140.0, draft="mtp-a.gguf",
                     guessed=100, taken=70)
    fell = _measured("draft:mtp-a@n8", hits=8, seconds=100.0, draft="mtp-a.gguf",
                     guessed=100, taken=90)
    held = _measured("draft:mtp-b@n4", hits=11, seconds=125.0, draft="mtp-b.gguf",
                     guessed=100, taken=80)
    said = drafted([base, slow, fell, held])
    lines = said.splitlines()
    assert lines[0].split() == ["draft", "accept", "s/q", "speed", "F1", "dF1", "against"]
    rows = [ln for ln in lines if ln.startswith("draft:")]
    assert [ln.split()[0] for ln in rows] == ["draft:mtp-a@n8", "draft:mtp-b@n4",
                                              "draft:mtp-a@n4"], "fastest first"
    assert rows[0].split()[1:] == ["90%", "5.0", "2.00x", "40%", "-20", "draft:none"]
    assert rows[1].split()[1:] == ["80%", "6.2", "1.60x", "55%", "-5", "draft:none"]
    assert rows[2].split()[1:] == ["70%", "7.0", "1.43x", "60%", "+0", "draft:none"]
    assert lines[-1] == (f"serve draft:mtp-b@n4: fastest whose F1 held within "
                         f"{NOISE * 100:g} points of its baseline, 1.60x")
    # the baseline may be an older run already kept, found among everything
    assert drafted([held], among=[base, held]).splitlines()[-1].startswith("serve draft:mtp-b@n4")
    # no baseline: every row says so, and there is nothing to recommend against
    alone = drafted([slow, held])
    assert all(ln.endswith("no baseline") for ln in alone.splitlines() if ln.startswith("draft:"))
    assert alone.splitlines()[-1].startswith("no baseline to recommend against")
    # every head fell: serve none
    assert drafted([base, fell]).splitlines()[-1].startswith("serve no head")
    assert drafted([base]) == "no drafted run to summarise"


def test_the_ranking_and_the_export_carry_the_speedup(tmp_path):
    """The cost row names the drafted run, its size, and what the head was worth."""
    from ml_stack.graph.bench import export, ranking

    store = tmp_path / "runs.ladybug"
    _kept_run(store, "flash-plain", model="flash.gguf", questions=34, hits=20, seconds=500.0,
              binary="/builds/current/llama-server")
    _kept_run(store, "draft:none", model="flash.gguf", questions=20, hits=12, seconds=300.0,
              binary="/builds/unsloth/llama-server")
    _kept_run(store, "draft:mtp-flash@n4", model="flash.gguf", questions=20, hits=12,
              seconds=200.0, binary="/builds/unsloth/llama-server",
              draft_model="mtp-flash.gguf")
    said = ranking(runs(store))
    row = next(ln for ln in said.splitlines() if ln.startswith("| `flash.gguf`"))
    assert row.endswith("| `draft:mtp-flash@n4` on unsloth/llama-server (20 q, 1.50x) |")
    got = {r["label"]: r for r in
           json.loads(pathlib.Path(export(runs(store), tmp_path / "o.json")).read_text())}
    assert got["draft:mtp-flash@n4"]["speedup"] == 1.5
    assert got["draft:none"]["speedup"] is None and got["flash-plain"]["speedup"] is None


# -- --also tight -------------------------------------------------------------------------

def test_tight_is_asked_for_the_way_rich_is():
    from argparse import Namespace

    from ml_stack.graph.bench import _parser, _ways

    ways = _ways(Namespace(also=["tight"], terse=False, temperature=0.0))
    assert ways[1]["label"] == "tight" and ways[1]["tight"] is True
    assert ways[1]["temperature"] == 0.0, "the run's sampling, as rich carries it"
    assert "tight" not in ways[0], "the first way is what was asked for, unchanged"
    assert _parser().parse_args(["run", "x", "--also", "tight"]).also == ["tight"]


def test_tight_reaches_converse_as_a_keyword(monkeypatch):
    """`converse(..., tight=True)` is ask's; this only has to hand it on, and hand the
    terse set in already told, since that set is built here rather than chosen inside."""
    import ml_stack.graph.ask as ask_module
    from ml_stack.graph.ask import TIGHT_SHOW_TERSE
    from ml_stack.graph.bench import asking

    reached = {}

    def fake_converse(question, graph, client, **kw):
        reached.update(kw)
        return type("A", (), {"content": "", "show": [], "ids": [], "why": ""})()

    monkeypatch.setattr(ask_module, "converse", fake_converse)
    asking(TINY, tight=True)("who?", _Scripted())
    assert reached.get("tight") is True and "tools" not in reached
    reached.clear()
    asking(TINY, terse=True, tight=True)("who?", _Scripted())
    assert reached.get("tight") is True
    show = next(s for s, _ in reached["tools"] if s["function"]["name"] == "show")
    assert show["function"]["description"] == TIGHT_SHOW_TERSE
    reached.clear()
    asking(TINY)("who?", _Scripted())
    assert "tight" not in reached, "not asked for, not sent -- the default is converse's own"


def test_what_is_about_the_asking_never_reaches_the_client(monkeypatch):
    """`--also rich` and `--also tight` are questions about the asking; the client does not
    take them. `tight` reached Client.__init__ and took an 87G load down with it on
    2026-09-02, after a fake client with **kwargs had let it pass. Mutation: pop after
    Client is built. The strict fake is the runner's own `ScriptedModel`, bound against
    the real `Client.__init__`, so it stays strict as the client changes."""
    import ml_stack.client
    import ml_stack.serve
    from ml_stack.graph import bench

    built = []

    class Strict(ScriptedModel):
        def __init__(self, base_url, **settings):
            super().__init__(base_url, **settings)
            built.append(base_url)

    with pytest.raises(TypeError, match="nonsense"):
        Strict("http://127.0.0.1:1", nonsense=True)
    Strict("http://127.0.0.1:1", timeout=1.0, n_predict=4, temperature=0.0)
    built.clear()

    class Server:
        base_url = "http://127.0.0.1:1"

    class FakeServe:
        def __init__(self, *a, **k): pass
        def __enter__(self): return Server()
        def __exit__(self, *a): return False

    monkeypatch.setattr(ml_stack.serve, "serve", FakeServe)
    monkeypatch.setattr(ml_stack.client, "Client", Strict)
    monkeypatch.setattr(bench, "measure", lambda ask, questions, **k: [])
    monkeypatch.setattr(bench, "asking", lambda *a, **k: (lambda *x, **y: None))
    monkeypatch.setattr(bench, "find_model", lambda named: named)
    monkeypatch.setattr(bench, "footprint", lambda url: {"base_url": url})
    _preflight_ok(monkeypatch)
    ways = [{}, {"label": "rich", "rich": True}, {"label": "tight", "tight": True}]
    bench.served("tiny.gguf", [{"q": "who?", "expect": []}], {"nodes": [], "edges": []},
                 ways=ways, kept="")
    assert len(built) == 3, "one strict client per way, none refused"


# -- the smoke is the first step of a real run ----------------------------------------------

def _watching(monkeypatch):
    """Every client `served` builds, in the order it built them, so what each was asked
    can be read off afterwards."""
    import ml_stack.client

    built = []

    class Watched(_ServedModel):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            built.append(self)

    monkeypatch.setattr(ml_stack.client, "Client", Watched)
    return built


def _asked_of(client) -> int:
    """How many different questions a scripted client was asked: the first user turn of
    each conversation it was shown, distinct."""
    return len({next(m["content"] for m in turn if m.get("role") == "user")
                for turn in client.seen})


def test_a_served_sweep_smokes_every_way_first_on_the_one_load(tmp_path, monkeypatch, capsys):
    """Unless it is itself --smoke or told --no-smoke, a sweep asks two questions of every
    way as soon as the model is up -- kept, read back -- and only then its own questions,
    on the same server, so the load is paid once. The smoke used to be a step in a plan,
    and the day it was left out of one a bad way cost an 87G load."""
    import ml_stack.graph.bench as bench

    seen = _serving(monkeypatch, tmp_path)
    built = _watching(monkeypatch)
    asked = tmp_path / "q.jsonl"
    asked.write_text("\n".join(json.dumps({"q": q, "expect": ["topic:compiler"]})
                               for q in ("who works on compilers?", "who else?", "and?")) + "\n")
    common = [a for a in seen["common"] if a != "--no-smoke"]
    common[common.index("--questions") + 1] = str(asked)
    assert bench._main(["sweep", "--serve", "tiny.gguf", "--plain-only", "--also", "terse",
                        *common]) == 0
    said = capsys.readouterr().out
    assert seen["models"] == ["tiny.gguf"], "one load for the smoke and the sweep"
    assert [_asked_of(c) for c in built] == [2, 2, 3, 3], \
        "two questions of every way first, then the three of every way"
    by_label: dict = {}
    for r in runs(seen["kept"]):
        by_label.setdefault(r["label"], []).append(len(r["rows"]))
    assert by_label == {"tiny-plain": [2, 3], "tiny-plain-terse": [2, 3]}, "the smoke kept too"
    assert (said.index("smoke: 2 question(s) through every way first")
            < said.index("--- tiny-plain (smoke)") < said.index("--- tiny-plain-terse (smoke)")
            < said.index("smoke: ok") < said.index("\n  --- tiny-plain\n"))

    # --no-smoke: the questions alone
    built.clear()
    seen["models"].clear()
    assert bench._main(["sweep", "--serve", "tiny.gguf", "--plain-only", "--also", "terse",
                        *common, "--no-smoke"]) == 0
    capsys.readouterr()
    assert [_asked_of(c) for c in built] == [3, 3] and seen["models"] == ["tiny.gguf"]


def test_a_smoke_that_fails_ends_the_run_with_exit_1_and_nothing_else_starts(tmp_path,
                                                                            monkeypatch,
                                                                            capsys):
    """The self-check passes -- its model is scripted -- and then the real server answers
    nothing: every smoke question fails, the run stops there with the reason, and the
    questions that cost the GPU are never asked."""
    import ml_stack.client
    import ml_stack.graph.bench as bench

    seen = _serving(monkeypatch, tmp_path)
    built = _watching(monkeypatch)
    watched = ml_stack.client.Client

    class Silent(watched):
        def chat(self, messages, tools=None, **_):
            raise RuntimeError("no answer from the server")

    monkeypatch.setattr(ml_stack.client, "Client", Silent)
    common = [a for a in seen["common"] if a != "--no-smoke"]
    assert bench.main(["sweep", "--serve", "tiny.gguf", "--plain-only", "--also", "terse",
                       *common]) == 1
    said = capsys.readouterr()
    assert said.out.splitlines()[0].startswith("selfcheck: ok (")
    assert ("error: smoke failed, so the run did not start: tiny smoke: every question "
            "failed -- RuntimeError: no answer from the server") in said.err
    assert seen["models"] == ["tiny.gguf"] and len(built) == 2, "the smoke's clients, no more"
    assert sorted((r["label"], len(r["rows"])) for r in runs(seen["kept"])) == [
        ("tiny-plain", 1), ("tiny-plain-terse", 1)], "the smoke was kept; nothing else ran"
    assert all(r["error"] for one in runs(seen["kept"]) for r in one["rows"])


def test_a_run_on_a_standing_server_smokes_first_and_stops_on_a_failing_smoke(tmp_path,
                                                                             monkeypatch,
                                                                             capsys):
    import ml_stack.graph.bench as bench

    built = []

    class Watched(_Scripted):
        def __init__(self):
            super().__init__()
            built.append(self)

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench, "footprint", lambda url: {"base_url": url})
    monkeypatch.setattr(bench, "ask_from", lambda spec: Watched)
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps(TINY))
    asked = tmp_path / "q.jsonl"
    asked.write_text("\n".join(json.dumps({"q": q, "expect": ["topic:compiler"]})
                               for q in ("who works on compilers?", "who else?", "and?")) + "\n")
    kept = tmp_path / "runs.ladybug"
    common = ["--kept", str(kept), "--graph", str(graph), "--questions", str(asked),
              "--client", "fake:client", "--store", ""]
    assert bench._main(["run", "tried", *common]) == 0
    said = capsys.readouterr().out
    assert said.splitlines()[0].startswith("smoke: the same run on 2 question(s) first")
    assert [_asked_of(c) for c in built] == [2, 3]
    assert [len(r["rows"]) for r in runs(kept, "tried")] == [2, 3]
    assert said.index("smoke: ok") < said.index("tried: 3 questions over")

    class Silent(Watched):
        def chat(self, messages, tools=None, **_):
            raise RuntimeError("nothing")

    monkeypatch.setattr(bench, "ask_from", lambda spec: Silent)
    with pytest.raises(bench.SmokeFailed, match="run smoke: every question failed"):
        bench._main(["run", "again", *common])
    assert [len(r["rows"]) for r in runs(kept, "again")] == [2], "the smoke, and no more"


# -- the split: one package, one namespace, the old names still resolve ---------------------------

def test_the_old_module_names_still_import_and_are_the_same_objects():
    """`bench_extract`, `bench_selfcheck` and `bench_history` moved into the package; the
    shims at the old paths re-export, so an import written before the move still works."""
    from ml_stack.graph import bench_extract, bench_history, bench_selfcheck
    from ml_stack.graph.bench import extract, history, selfcheck

    assert bench_extract.main is extract.main
    assert bench_selfcheck.ScriptedModel is selfcheck.ScriptedModel
    assert bench_history.history is history.history


def test_patching_the_package_reaches_every_module(monkeypatch):
    """The package is the one namespace: `bench.runs` patched here is what `read_back` in
    `keep` and `compare` in `show` see, or a test's fake store would be read by nobody."""
    import ml_stack.graph.bench as bench
    from ml_stack.graph.bench import keep, show

    monkeypatch.setattr(bench, "runs",
                        lambda store, label="": [{"key": "k", "label": label, "rows": []}])
    assert keep.read_back("nowhere", ["k"]) == [{"key": "k", "label": "", "rows": []}]
    assert "wall clock" in show.compare("nowhere", "a", "b")


def test_serve_draft_auto_asks_the_one_resolver_and_says_why(tmp_path, monkeypatch, capsys):
    """`hub.choose_head` is the resolver every caller shares -- told which binary will
    serve, so a head that borrows is withheld from mainline. The bench had its own, which
    chose a BF16 MTP head for mainline twice (2026-09-01), 87G each time."""
    import ml_stack.hub
    import ml_stack.graph.bench as bench

    asked = []

    def fake_choose(model, *, binary=None, **k):
        asked.append((model, binary))
        return ml_stack.hub.Chosen("hf:someone/tiny-GGUF/mtp-tiny.gguf", "",
                                   "shipped beside the weights", False,
                                   "the README says it needs a fork")

    monkeypatch.setattr(ml_stack.hub, "choose_head", fake_choose)
    seen = _serving(monkeypatch, tmp_path)
    assert bench._main(["sweep", "--serve", "tiny.gguf", "--serve-draft", "auto",
                        "--plain-only", *seen["common"]]) == 0
    said = capsys.readouterr().out
    assert asked == [("tiny.gguf", None)], "no --binary means the build find_binary picks"
    assert seen["kwargs"][0]["draft"] == "hf:someone/tiny-GGUF/mtp-tiny.gguf"
    assert "draft head: hf:someone/tiny-GGUF/mtp-tiny.gguf -- shipped beside the weights" in said
    assert "the README says it needs a fork" in said

    # a head withheld: nothing served as the draft, and the reason printed
    asked.clear()
    seen["kwargs"].clear()
    monkeypatch.setattr(ml_stack.hub, "choose_head",
                        lambda model, **k: ml_stack.hub.Chosen("", "", "withheld: mainline", False))
    assert bench._main(["sweep", "--serve", "tiny.gguf", "--serve-draft", "auto",
                        "--plain-only", *seen["common"]]) == 0
    assert "draft head: none -- withheld: mainline" in capsys.readouterr().out
    assert "draft" not in seen["kwargs"][0]


def test_detach_writes_argv_started_and_commit_at_the_top_of_the_log(tmp_path, monkeypatch,
                                                                     capsys):
    """The log's first lines are its record: `history` reads them back once
    `measuring.json` has moved on to the next run, and `measuring.json` carries the commit
    while the run is going."""
    import pathlib
    import subprocess

    import ml_stack.graph.bench as bench
    from ml_stack.graph.bench import history
    from ml_stack.graph.bench import run as running

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(running, "_commit", lambda root=None: "0f1e2d3 (dirty)")
    monkeypatch.setattr(subprocess, "Popen",
                        lambda command, **kw: type("C", (), {"pid": 4242})())
    argv = ["sweep", "--serve", "models/tiny.gguf", "--detach", "--also", "terse", "--smoke"]
    assert bench.main(argv) == 0
    capsys.readouterr()
    held = json.loads((tmp_path / "home" / "measuring.json").read_text())
    assert held["commit"] == "0f1e2d3 (dirty)"
    lines = pathlib.Path(held["log"]).read_text().splitlines()
    assert lines[0] == "argv: sweep --serve models/tiny.gguf --also terse --smoke"
    assert lines[1] == f"started: {held['started']}"
    assert lines[2] == "commit: 0f1e2d3 (dirty)"

    (tmp_path / "home" / "measuring.json").unlink()
    entry = history.history(tmp_path / "home", alive=lambda pid: False)[0]
    assert entry.commit == "0f1e2d3 (dirty)"
    assert entry.argv == [a for a in argv if a != "--detach"]
    assert entry.started == held["started"]

    # no repository to ask: the header has no commit line and the record an empty one
    monkeypatch.setattr(running, "_commit", lambda root=None: "")
    assert bench.main(["run", "tried", "--detach"]) == 0
    capsys.readouterr()
    held = json.loads((tmp_path / "home" / "measuring.json").read_text())
    assert held["commit"] == ""
    assert pathlib.Path(held["log"]).read_text().splitlines() == \
        ["argv: run tried", f"started: {held['started']}"]


def test_commit_reads_the_short_sha_and_marks_a_dirty_tree(tmp_path):
    """Best effort against a repository made here: the sha, `(dirty)` once a file is left
    uncommitted, and "" where there is no repository at all."""
    import subprocess

    from ml_stack.graph.bench.run import _commit

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*words):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=quill",
                               "-c", "user.email=quill@invented.example", *words],
                              check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "first")
    sha = git("rev-parse", "--short", "HEAD")
    assert _commit(repo) == sha
    (repo / "note.txt").write_text("left over", encoding="utf-8")
    assert _commit(repo) == f"{sha} (dirty)"
    assert _commit(tmp_path / "nowhere") == ""


def test_a_sigterm_says_killed_before_it_raises(capsys):
    """`history` tells a stopped run from a crashed one by that word in its log."""
    import signal

    from ml_stack.graph.bench import history
    from ml_stack.graph.bench.run import _stop_on_sigterm

    with pytest.raises(SystemExit) as left:
        _stop_on_sigterm(signal.SIGTERM, None)
    assert left.value.code == 128 + signal.SIGTERM
    assert capsys.readouterr().out.startswith(history._KILLED)


# -- every run says which machine and which code measured it -------------------------------------

def test_every_run_carries_the_host_and_the_commit(tmp_path, monkeypatch):
    """Stamped by `save`, kept as given when a record arrives with them -- a run gathered
    from a peer names the peer, not the machine that gathered it."""
    import socket

    from ml_stack.graph.bench import keep

    store = tmp_path / "runs.ladybug"
    monkeypatch.setattr(keep, "_commit", lambda root=None: "0f1e2d3 (dirty)")
    save(store, [a_row("who?", expected=["person:iris"], shown=["person:iris"])],
         held={"context": 32768})
    save(store, [a_row("who?", expected=["person:iris"], shown=[])],
         held={"host": "lantern", "commit": "abc1234"})
    mine, theirs = runs(store)
    assert mine["server"]["host"] == socket.gethostname()
    assert mine["server"]["commit"] == "0f1e2d3 (dirty)" and mine["server"]["context"] == 32768
    assert theirs["server"] == {"host": "lantern", "commit": "abc1234"}


def test_the_table_names_the_host_only_when_more_than_one_measured(tmp_path, capsys):
    store = tmp_path / "runs.ladybug"
    _kept_run(store, "flash-plain", model="flash.gguf", questions=20, hits=10, seconds=100.0,
              host="quill-box")
    _kept_run(store, "flash-terse", model="flash.gguf", questions=20, hits=10, seconds=90.0,
              host="quill-box")
    table(runs(store))
    said = capsys.readouterr().out
    assert "host" not in said.splitlines()[0], "one machine: no column"
    _kept_run(store, "flash-plain", model="flash.gguf", questions=20, hits=10, seconds=80.0,
              host="lantern")
    table(runs(store))
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:2] == ["run", "host"]
    # `runs` is by key, so the second flash-plain sits between the two quill-box runs
    assert [ln.split()[1] for ln in lines[2:]] == ["quill-box", "lantern", "quill-box"]


def test_a_cost_run_from_another_host_is_never_taken_and_is_named(tmp_path, capsys):
    """A different machine is a different clock: however well its F1 held, a run from
    another host supplies no cost, and the ranking says so rather than skipping it."""
    from ml_stack.graph.bench import choices, composed, plot, ranking, rates

    store = tmp_path / "runs.ladybug"
    _kept_run(store, "flash-plain", model="flash.gguf", questions=34, hits=20, seconds=500.0,
              host="quill-box")
    _kept_run(store, "draft:mtp-tiny@n4", model="flash.gguf", questions=20, hits=12,
              seconds=200.0, host="lantern")
    kept = runs(store)
    chosen, _ = choices(kept)
    assert chosen[0].own and chosen[0].rejected == []
    assert [r["label"] for r in chosen[0].elsewhere] == ["draft:mtp-tiny@n4"]
    said = ranking(kept)
    row = next(ln for ln in said.splitlines() if ln.startswith("| `flash.gguf`"))
    assert "| 14.7 |" in row and "| quill-box |" in row and row.endswith("| its own run |")
    assert "| host |" in said
    assert ("- `flash.gguf` rejected: other host -- `draft:mtp-tiny@n4` on lantern "
            "(20 q, 10.0 s/question)") in said
    assert composed(kept)[0]["server"] == {"host": "quill-box"}

    rates(kept)
    out = capsys.readouterr().out
    # `_shown` keeps the end of a label, which is where the host now is
    assert any(ln.startswith("…h-plain@quill-box") for ln in out.splitlines()), out
    assert any(ln.startswith("…p-tiny@n4@lantern") for ln in out.splitlines())
    assert any(ln.startswith("…sh.gguf@quill-box =") for ln in out.splitlines()), \
        "the composed point carries its accuracy run's host"
    drawn = pathlib.Path(plot(kept, tmp_path / "f.html")).read_text()
    assert "host: lantern" in drawn and "@quill-box" in drawn

    # the same run on the same host is taken as before
    _kept_run(store, "draft:mtp-tiny@n8", model="flash.gguf", questions=20, hits=12,
              seconds=200.0, host="quill-box")
    again = ranking(runs(store))
    row = next(ln for ln in again.splitlines() if ln.startswith("| `flash.gguf`"))
    assert "| 10.0 |" in row and "`draft:mtp-tiny@n8`" in row

    # one host: no host column at all
    alone = ranking([kept[0]])
    assert "| host |" not in alone


# -- sweep --fleet ----------------------------------------------------------------------------------

def test_fleet_jobs_are_the_same_line_with_one_serve_each():
    from ml_stack.graph.bench.run import fleet_jobs

    argv = ["sweep", "--fleet", "--peers", "quill,lantern", "--serve", "a.gguf", "--serve",
            "b.gguf", "--serve-draft", "ha.gguf", "--also", "terse", "--kept", "/k",
            "--no-queue", "--sample", "5"]
    jobs = fleet_jobs(argv, ["a.gguf", "b.gguf"], commit="0f1e2d3 (dirty)")
    assert [j["argv"] for j in jobs] == [
        ["sweep", "--also", "terse", "--kept", "/k", "--sample", "5", "--serve", "a.gguf",
         "--serve-draft", "ha.gguf"],
        ["sweep", "--also", "terse", "--kept", "/k", "--sample", "5", "--serve", "b.gguf"]]
    assert [(j["model"], j["commit"], j["dirty"]) for j in jobs] == \
        [("a.gguf", "0f1e2d3", True), ("b.gguf", "0f1e2d3", True)]
    assert fleet_jobs(["sweep", "--peers=quill", "--serve=x.gguf"], ["x.gguf"],
                      commit="abc1234")[0] == {"model": "x.gguf", "argv": ["sweep", "--serve",
                                                                            "x.gguf"],
                                               "commit": "abc1234", "dirty": False}


def _fake_fleet(monkeypatch, *, plan):
    """`ml_stack.fleet.bench` with the four functions faked, recording every call; `gather`
    keeps one invented run in the store it is told to."""
    import sys
    import types

    calls = {}
    fake = types.ModuleType("ml_stack.fleet.bench")

    def planning(models, peers):
        calls["plan"] = (list(models), peers)
        return plan

    def dispatching(jobs):
        calls["dispatch"] = [dict(j) for j in jobs]
        return [f"handle:{j['model']}" for j in jobs]

    def waiting(handles):
        calls["wait"] = list(handles)

    def gathering(handles, *, into):
        calls["gather"] = (list(handles), into)
        _kept_run(into, "a-plain", model="a.gguf", questions=20, hits=10, seconds=100.0,
                  host="quill")

    fake.plan, fake.dispatch, fake.wait, fake.gather = planning, dispatching, waiting, gathering
    monkeypatch.setitem(sys.modules, "ml_stack.fleet.bench", fake)
    return calls


def test_sweep_fleet_plans_prints_dispatches_waits_and_gathers(tmp_path, monkeypatch, capsys):
    import ml_stack.graph.bench as bench
    from ml_stack.graph.bench import run as running

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(running, "_commit", lambda root=None: "0f1e2d3")
    calls = _fake_fleet(monkeypatch, plan=[{"model": "a.gguf", "peer": "quill", "commit": "0f1e2d3"},
                                           {"model": "b.gguf", "peer": "lantern"}])
    kept = tmp_path / "runs.ladybug"
    argv = ["sweep", "--fleet", "--peers", "quill,lantern", "--serve", "a.gguf", "--serve",
            "b.gguf", "--plain-only", "--kept", str(kept), "--store", ""]
    assert bench._main(argv) == 0
    said = capsys.readouterr().out
    assert calls["plan"] == (["a.gguf", "b.gguf"], ["quill", "lantern"])
    assert [j["argv"] for j in calls["dispatch"]] == [
        ["sweep", "--plain-only", "--kept", str(kept), "--store", "", "--serve", "a.gguf"],
        ["sweep", "--plain-only", "--kept", str(kept), "--store", "", "--serve", "b.gguf"]]
    assert [(j["peer"], j["commit"]) for j in calls["dispatch"]] == \
        [("quill", "0f1e2d3"), ("lantern", "0f1e2d3")]
    assert calls["wait"] == ["handle:a.gguf", "handle:b.gguf"]
    assert calls["gather"] == (["handle:a.gguf", "handle:b.gguf"], str(kept))
    assert "plan: 2 job(s) on commit 0f1e2d3 over quill, lantern" in said
    assert "  a.gguf -> quill (0f1e2d3)" in said and "  b.gguf -> lantern" in said
    assert said.index("plan:") < said.index("a-plain"), "the plan first, then the table"
    assert [r["label"] for r in runs(kept)] == ["a-plain"], "gathered into --kept"


def test_sweep_fleet_refuses_a_peer_on_another_commit_before_dispatching(tmp_path, monkeypatch,
                                                                         capsys):
    import ml_stack.graph.bench as bench
    from ml_stack.graph.bench import run as running

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(running, "_commit", lambda root=None: "0f1e2d3 (dirty)")
    calls = _fake_fleet(monkeypatch, plan={"a.gguf": "quill", "b.gguf": "lantern"})
    kept = tmp_path / "runs.ladybug"
    assert bench._main(["sweep", "--fleet", "--serve", "a.gguf", "--serve", "b.gguf",
                        "--kept", str(kept)]) == 0, "a plan that names no commit is trusted"
    assert [j["peer"] for j in calls["dispatch"]] == ["quill", "lantern"]
    capsys.readouterr()

    calls = _fake_fleet(monkeypatch, plan=[{"model": "a.gguf", "peer": "quill",
                                            "commit": "deadbee"}])
    assert bench._main(["sweep", "--fleet", "--serve", "a.gguf", "--kept", str(kept)]) == 2
    said = capsys.readouterr()
    assert "quill is on commit deadbee, this checkout is on 0f1e2d3 (dirty)" in said.err
    assert "dispatch" not in calls and "gather" not in calls
    assert bench._main(["sweep", "--fleet", "--on", "x=http://127.0.0.1:1",
                        "--kept", str(kept)]) == 2
    assert "pass --serve MODEL" in capsys.readouterr().err


# -- the prompt cache, per turn --------------------------------------------------------------

class _Reporting:
    """A client whose every reply carries the ``(cache_n, prompt_n)`` the script says the
    server reported for that call, and nothing else a bench needs."""

    def __init__(self, script):
        self.script = list(script)
        self.sampling: dict = {}

    def chat(self, messages, tools=None, **_):
        cached, processed = self.script.pop(0)
        raw = {"usage": {"prompt_tokens": cached + processed, "completion_tokens": 5},
               "timings": {"cache_n": cached, "prompt_n": processed}}
        return type("R", (), {"content": "a compiler person", "thinking": None, "raw": raw,
                              "tool_calls": None})()


def _asking_calls(n):
    """An `ask` of ``n`` round trips that then lights the compiler."""
    def ask(question, client, **_):
        for _ in range(n):
            client.chat([{"role": "user", "content": question}])
        return {"content": "a compiler person", "show": ["topic:compiler"]}
    return ask


def _cache_row(script, label="cached"):
    from ml_stack.graph.bench import _ask_once

    row, _ = _ask_once(_asking_calls(len(script)), {"q": "who?", "expect": ["topic:compiler"]},
                       label=label, client=_Reporting(script))
    return row


def test_a_prefix_that_survives_and_one_that_breaks_are_told_apart_per_turn():
    """The second call of a question should pay for the tool result and the reply and
    nothing before them. Cached tokens that grow past the previous call's whole prompt
    say the prefix was kept; cached tokens that fall back say the system prompt and the
    tool schemas were read again -- which the run's totals cannot see."""
    from ml_stack.graph.bench import prefix_kept

    grew = _cache_row([(0, 900), (900, 120), (1025, 80), (1100, 60)])
    assert grew.cache_calls == [[0, 900], [900, 120], [1025, 80], [1100, 60]]
    assert (grew.prefix_kept, grew.prefix_turns, grew.prefix_hits) == (3, 3, 1.0)
    assert grew.cached_tokens == 3025 and grew.processed_tokens == 1160, "the totals as before"

    reset = _cache_row([(0, 900), (0, 1020), (700, 400), (0, 1500)])
    assert (reset.prefix_kept, reset.prefix_turns, reset.prefix_hits) == (0, 3, 0.0)

    mixed = _cache_row([(0, 900), (900, 120), (0, 1100), (1100, 50)])
    assert (mixed.prefix_kept, mixed.prefix_turns) == (2, 3)
    assert mixed.prefix_hits == pytest.approx(2 / 3)

    # a template re-rendering the turn boundary re-reads a few tokens: still kept
    assert prefix_kept([(0, 900), (895, 130)]) == (1, 1)
    assert prefix_kept([(0, 900), (880, 130)]) == (0, 1), "more than a few is a break"
    # a server that reports nothing says nothing about its cache; one call has no turn
    silent = _cache_row([(0, 0), (0, 0), (0, 0)])
    assert (silent.prefix_turns, silent.prefix_hits) == (0, None)
    assert _cache_row([(0, 900)]).prefix_hits is None


def test_the_run_the_table_the_detail_and_the_export_carry_the_cache_per_turn(tmp_path, capsys):
    from ml_stack.graph.bench import _flat, cache_turns, missed, prefixed

    store = tmp_path / "runs.ladybug"
    grew = _cache_row([(0, 900), (900, 120), (1025, 80), (1100, 60)])
    mixed = _cache_row([(0, 900), (900, 120), (0, 1100), (1100, 50)])
    save(store, [grew, mixed], held={"context": 32768, "slots": 1})
    save(store, [a_row("q?", expected=["topic:compiler"], shown=["topic:compiler"])],
         held={"context": 32768, "slots": 1})               # as a run kept before this
    back = {r["label"]: r for r in runs(store)}
    assert back["cached"]["server"]["prefix_hits"] == pytest.approx(5 / 6)
    assert "prefix_hits" not in back["tried"]["server"], "no turn judged: no key, not 0"
    assert back["cached"]["rows"][1]["cache_calls"] == [[0, 900], [900, 120], [0, 1100],
                                                        [1100, 50]]

    table(runs(store))
    said = capsys.readouterr().out
    head = said.splitlines()[0].split()
    assert head.index("pfx") == head.index("cached") + 1
    new = next(ln for ln in said.splitlines() if ln.startswith("cached"))
    old = next(ln for ln in said.splitlines() if ln.startswith("tried"))
    assert "83%" in new.split() and "83%" not in old.split()
    assert prefixed(back["cached"]["server"]) == "83%" and prefixed(back["tried"]["server"]) == ""
    assert old.split()[3] == "1" and new.split()[3] == "2", "n is still the fourth word"

    missed(runs(store, "cached"), everything=True)
    said = capsys.readouterr().out
    assert "cache 3/3 turns" in said and "cache 2/3 turns" in said
    assert cache_turns(back["tried"]["rows"][0]) == ""
    missed(runs(store, "tried"), everything=True)
    assert "cache " not in capsys.readouterr().out

    assert _flat(back["cached"])["prefix_hits"] == pytest.approx(5 / 6, abs=1e-4)
    assert _flat(back["tried"])["prefix_hits"] is None


# -- the estimate, before anything is paid for -----------------------------------------------

def _stamped_run(label, *, model, per_question, questions=20, context=32768, at, load_s=None):
    """A kept run as `runs` hands it back, ``questions`` rows of ``per_question`` seconds."""
    from dataclasses import asdict

    rows = []
    for n in range(questions):
        r = a_row(f"q{n}?", expected=["person:iris"], shown=["person:iris"])
        r.label, r.seconds = label, per_question
        rows.append(asdict(r))
    server = {"model": model, "context": context, "slots": 1}
    if load_s is not None:
        server["load_s"] = load_s
    return {"key": f"bench:{label}:{at}", "at": at, "label": label, "server": server,
            "rows": rows}


def _twelve(tmp_path):
    asked = tmp_path / "twelve.jsonl"
    asked.write_text("\n".join(json.dumps({"q": f"q{n}?", "expect": ["person:ada"]})
                               for n in range(12)) + "\n")
    return str(asked)


def test_the_estimate_is_seconds_per_question_from_the_kept_run_at_the_same_context(tmp_path,
                                                                                    monkeypatch):
    """The newest run of that model at this context, over a newer one at another: a model
    at 8k answers faster than the same model at 32k, and the `ctx` column exists because
    the two are not the same measurement. Times the questions, the ways and a load."""
    import ml_stack.graph.bench as bench
    from ml_stack.graph.bench import estimate, history

    monkeypatch.setattr(bench, "find_model", lambda named: named)
    kept = [_stamped_run("quill-plain", model="quill.gguf", per_question=65.0, load_s=41.6,
                         at="2026-09-01T12:00:00"),
            _stamped_run("quill-plain", model="quill.gguf", per_question=2.0, context=8192,
                         at="2026-09-01T15:00:00"),
            _stamped_run("lantern-plain", model="lantern.gguf", per_question=9.0,
                         at="2026-09-01T16:00:00")]
    args = bench._parser().parse_args(["sweep", "--serve", "models/quill.gguf", "--also",
                                       "terse", "--questions", _twelve(tmp_path),
                                       "--no-smoke", "--kept", str(tmp_path / "none")])
    got = estimate(args, kept)
    assert len(got.models) == 1
    one = got.models[0]
    assert (one.questions, one.ways, one.per_question, one.load_s) == (12, 4, 65.0, 41.6)
    assert not one.guessed
    assert one.line() == ("estimate: 53 min (quill 12 q × 4 ways × 65 s/q + load 42 s; "
                          "from quill-plain kept 2026-09-01T12:00:00 at 32k)")
    assert got.seconds == pytest.approx(12 * 4 * 65 + 41.6)
    assert got.over and got.ceiling_min == 30
    assert got.lines()[-1] == "estimate: 53 min in all for 1 model (over the ceiling)"
    assert history.parse_duration(got.lines()[-1].split(":", 1)[1]) == 53 * 60, \
        "what history reads back is the total and nothing added to it"
    assert "over the 30 min ceiling" in got.refusal() and "--yes" in got.refusal()
    assert "--sample N" in got.refusal() and "MLSTACK_BENCH_CEILING" in got.refusal()

    # a real run smokes first: two more questions of every way; a server already up
    # (--on) has no load to pay and is matched by the name its labels start with
    args = bench._parser().parse_args(["sweep", "--serve", "models/quill.gguf", "--on",
                                       "lantern=http://127.0.0.1:1", "--plain-only",
                                       "--questions", _twelve(tmp_path)])
    got = estimate(args, kept)
    assert [(m.name, m.questions, m.ways, m.load_s) for m in got.models] == [
        ("quill", 14, 1, 41.6), ("lantern", 14, 1, 0.0)]
    assert got.models[1].per_question == 9.0 and not got.models[1].guessed
    assert got.lines()[-1] == "estimate: 18 min in all for 2 models"


def test_a_model_with_no_run_kept_is_guessed_from_its_weights_and_the_line_says_so(tmp_path,
                                                                                   monkeypatch):
    import ml_stack.graph.bench as bench
    from ml_stack.graph.bench import estimate

    monkeypatch.setattr(bench, "find_model", lambda named: named)
    sized = tmp_path / "sized.gguf"
    with sized.open("wb") as fh:
        fh.truncate(3_000_000_000)                       # sparse: 3G on paper, no disk
    args = bench._parser().parse_args(["sweep", "--serve", str(sized), "--serve",
                                       "models/absent.gguf", "--plain-only", "--no-smoke",
                                       "--questions", _twelve(tmp_path), "--sample", "4"])
    got = estimate(args, [])
    by_size, unknown = got.models
    assert by_size.guessed and by_size.per_question == pytest.approx(2.1)
    assert by_size.load_s == 30.0 and "a guess from 2.8G of weights" in by_size.line()
    assert unknown.guessed and unknown.per_question == 15.0
    assert unknown.line() == ("estimate: 2 min (absent 4 q × 1 way × 15 s/q + load 30 s; a "
                              "guess, no run of it kept and no weights on disk to size it by)")
    assert unknown.seconds == 90.0, "and 90 s reads as 2 min, the shape history reads"
    assert got.lines()[-1] == "estimate: 2 min in all for 2 models, 2 guessed with no run kept"
    assert not got.over


def test_every_measuring_subcommand_estimates_its_own_shape(tmp_path, monkeypatch):
    """`drafts` is a load per (head, n-max) and one for the baseline; `concurrent` asks
    conversations times turns; `extract` reads messages, twice with --twice; `run` is one
    way of one server. A --smoke is two questions and is never over the ceiling."""
    import ml_stack.graph.bench as bench
    from ml_stack.graph.bench import estimate

    monkeypatch.setattr(bench, "find_model", lambda named: named)
    parse = bench._parser().parse_args
    heads = estimate(parse(["drafts", "tiny.gguf", "--draft", "mtp-a.gguf", "--draft", "",
                            "--n-max", "4", "--n-max", "8", "--sample", "5"]), [])
    assert [(m.name, m.questions, m.load_s) for m in heads.models] == [
        ("tiny draft:mtp-a@n4", 7, 30.0), ("tiny draft:mtp-a@n8", 7, 30.0),
        ("tiny draft:none", 7, 30.0)]
    talk = estimate(parse(["concurrent", "four-by-three", "--conversations", "4", "--turns",
                           "3", "--no-smoke", "--ceiling", "1"]), [])
    assert [(m.name, m.questions, m.ways, m.load_s) for m in talk.models] == [
        ("four-by-three", 12, 1, 0.0)]
    assert talk.over and talk.ceiling_min == 1.0
    reading = estimate(parse(["extract", "read", "--world", str(tmp_path), "--serve",
                              "tiny.gguf", "--sample", "10", "--twice"]), [])
    assert [(m.name, m.questions, m.ways, m.load_s) for m in reading.models] == [
        ("tiny", 10, 2, 30.0)]
    alone = estimate(parse(["run", "plain", "--client", "fake:client", "--no-smoke",
                            "--questions", _twelve(tmp_path)]), [])
    assert [(m.name, m.questions, m.ways, m.load_s) for m in alone.models] == [
        ("plain", 12, 1, 0.0)]
    smoke = estimate(parse(["sweep", "--serve", "huge.gguf", "--also", "terse", "--also",
                            "rich", "--smoke", "--ceiling", "0.01"]), [])
    assert smoke.models[0].questions == 2 and not smoke.over
    assert smoke.lines()[-1].endswith("(a smoke run, never refused)")
    monkeypatch.setenv("MLSTACK_BENCH_CEILING", "0.5")
    assert bench._parser().parse_args(["run", "x"]).ceiling == 0.5


def test_main_refuses_over_the_ceiling_with_exit_5_and_serves_nothing_unless_yes(tmp_path,
                                                                                monkeypatch,
                                                                                capsys):
    """Adam, 2026-09-02: no more eight-hour tests. The rule is in the tool: over the
    ceiling, said and refused before a download, a lock or a load; --yes runs it; a
    --smoke is never refused."""
    import ml_stack.graph.bench as bench

    seen = _serving(monkeypatch, tmp_path)
    argv = ["sweep", "--serve", "tiny.gguf", "--plain-only", *seen["common"],
            "--no-selfcheck", "--ceiling", "0.5"]
    assert bench.main(argv) == 5
    said = capsys.readouterr()
    assert said.out.splitlines()[0].startswith("estimate: 45 s (tiny 1 q × 1 way × 15 s/q "
                                               "+ load 30 s; a guess")
    assert said.out.splitlines()[1] == ("estimate: 45 s in all for 1 model, 1 guessed with "
                                        "no run kept (over the ceiling)")
    assert said.err.strip() == ("error: estimated 45 s, over the 0.5 min ceiling -- no more "
                                "eight-hour tests. Ask fewer questions (--sample N, --short), "
                                "raise the ceiling (--ceiling MINUTES, or "
                                "MLSTACK_BENCH_CEILING), or pass --yes to run it anyway.")
    assert seen["models"] == [], "nothing was loaded"
    assert not (tmp_path / "home" / "measuring.lock").exists(), "the lock was never taken"

    # the environment sets the ceiling when the flag does not
    monkeypatch.setenv("MLSTACK_BENCH_CEILING", "0.25")
    assert bench.main([a for a in argv if a not in ("--ceiling", "0.5")]) == 5
    assert "over the 0.25 min ceiling" in capsys.readouterr().err
    assert seen["models"] == []
    monkeypatch.delenv("MLSTACK_BENCH_CEILING")
    assert bench.main([a for a in argv if a != "--no-smoke"] + ["--smoke", "--ceiling",
                       "0.01"]) == 0, "a smoke run is never refused"
    assert seen["models"] == ["tiny.gguf"]
    assert "(a smoke run, never refused)" in capsys.readouterr().out

    assert bench.main([*argv, "--yes"]) == 0
    assert seen["models"] == ["tiny.gguf", "tiny.gguf"]
    said = capsys.readouterr().out
    # the smoke kept a run of this model, so this estimate was measured, not guessed
    assert "s/q + load 12 s; from tiny-plain kept " in said.splitlines()[0]
    assert "guessed" not in said.splitlines()[1]


def test_a_detached_run_is_estimated_in_the_terminal_and_a_refusal_never_detaches(tmp_path,
                                                                                  monkeypatch,
                                                                                  capsys):
    """A refusal at the top of a log nobody is watching is not a refusal."""
    import subprocess

    import ml_stack.graph.bench as bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench.platform, "system", lambda: "Darwin")
    started = []
    monkeypatch.setattr(subprocess, "Popen",
                        lambda command, **kw: started.append(command) or type(
                            "C", (), {"pid": 4242})())
    argv = ["sweep", "--serve", "models/tiny.gguf", "--plain-only", "--detach", "--no-smoke",
            "--questions", _twelve(tmp_path), "--ceiling", "1"]
    assert bench.main(argv) == 5
    said = capsys.readouterr()
    assert started == [] and not (tmp_path / "home" / "measuring.json").exists()
    assert said.out.startswith("estimate: 4 min (tiny 12 q × 1 way × 15 s/q + load 30 s;")
    assert "over the 1 min ceiling" in said.err

    assert bench.main([*argv, "--yes"]) == 0
    said = capsys.readouterr().out
    assert len(started) == 1 and started[0][-1] == "--yes"
    assert said.startswith("estimate: 4 min (tiny") and "measuring in the background" in said


def test_an_embedded_head_serves_with_the_speculative_type_and_no_file(monkeypatch, tmp_path):
    """Qwen3.8-27B ships its nextn layers inside the main GGUF: `--draft embedded` means
    --spec-type draft-mtp and no -md, with the draft length if asked (Adam, 2026-09-02:
    "we need to test the mtp of the newest 27b"). Mutation: drop the EMBEDDED branch."""
    import ml_stack.client
    import ml_stack.serve
    from ml_stack.graph import bench
    from ml_stack.testing import FakeClient, FakeServe

    fake = FakeServe()
    monkeypatch.setattr(ml_stack.serve, "serve", fake)
    monkeypatch.setattr(ml_stack.client, "Client", FakeClient.scripted([]))
    monkeypatch.setattr(bench, "measure", lambda ask, questions, **k: [])
    monkeypatch.setattr(bench, "asking", lambda *a, **k: (lambda *x, **y: None))
    monkeypatch.setattr(bench, "find_model", lambda named: named)
    monkeypatch.setattr(bench, "footprint", lambda url: {"base_url": url})
    _preflight_ok(monkeypatch)
    bench.served("tiny.gguf", [{"q": "who?", "expect": []}], {"nodes": [], "edges": []},
                 draft=bench.EMBEDDED, spec_draft_max=2, kept="", smoke=())
    spec = fake.leased[-1]
    assert spec.spec_type == "draft-mtp" and not spec.draft and spec.spec_draft_max == 2

    seen = []
    monkeypatch.setattr(bench, "served", lambda model, *a, **k: seen.append((k.get("label"), k.get("draft"))) or [])
    bench.drafts("tiny.gguf", ["", bench.EMBEDDED], [{"q": "who?", "expect": []}],
                 {"nodes": [], "edges": []}, n_max=[2, 8], kept="")
    assert seen == [("draft:none", ""), ("draft:embedded-mtp@n2", "embedded"),
                    ("draft:embedded-mtp@n8", "embedded")]


def test_a_head_that_held_its_f1_but_runs_slower_than_none_is_not_recommended():
    """gpt-oss's eagle3 head: 65% accepted, F1 unchanged, 0.82x -- the summary recommended
    it (2026-09-02). Serving a head is only worth it when it is faster than no head."""
    from ml_stack.graph.bench import drafted

    base = _measured("draft:none", hits=12, seconds=120.0, at="2026-09-01T11:00:00")
    slower = _measured("draft:eagle3@n2", hits=12, seconds=146.0, draft="eagle3.gguf",
                       guessed=100, taken=65)
    slowest = _measured("draft:eagle3@n4", hits=12, seconds=160.0, draft="eagle3.gguf",
                        guessed=100, taken=46)
    last = drafted([base, slower, slowest]).splitlines()[-1]
    assert last.startswith("serve no head: the best that held its F1, draft:eagle3@n2, "
                           "is slower than none at 0.82x"), last
    # one that is faster is still recommended over the slower ones
    quick = _measured("draft:eagle3@n1", hits=12, seconds=100.0, draft="eagle3.gguf",
                      guessed=100, taken=80)
    assert drafted([base, slower, quick]).splitlines()[-1].startswith("serve draft:eagle3@n1")
