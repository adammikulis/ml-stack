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
    assert [q["q"] for q in sample(QUESTIONS, 40)] == [q["q"] for q in QUESTIONS]
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
