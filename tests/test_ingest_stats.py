"""The numbers `ml-stack-ingest status` reads off a run's own files: speed, drafting, and
how long it has been going.

Every source, unit and call here is invented, and nothing opens a port.
"""

from __future__ import annotations

import pytest
from test_ingest import a_part_read_source, a_read

from ml_stack import ingest
from ml_stack.ingest.stats import run_stats, status
from ml_stack.telemetry import Call

SLUG = "velthorne-open-texts"


def a_call(*, completion=1000, prompt=2000, draft=800, accepted=700):
    """One kept call, in the shape `Call.public` writes it: 20s writing, 4s reading."""
    return Call(model="thornwick-8b.gguf", tool="extract", prompt_ms=4000.0,
                predicted_ms=20000.0, prompt_n=prompt, predicted_n=completion,
                draft_n=draft, draft_n_accepted=accepted, prompt_tokens=prompt,
                completion_tokens=completion, seconds=24.0, cache_n=0).public()


def a_run(tmp_path, *, calls=(), rows=1):
    """A store's files with ``rows`` units read, each keeping ``calls``."""
    listed = [dict(a_read(f"{SLUG}:1:1.{n}", source=SLUG), calls=list(calls))
              for n in range(1, rows + 1)]
    return a_part_read_source(tmp_path, rows=listed, sections=4)


def test_a_kept_call_reads_back_as_the_call_that_was_written():
    call = Call(model="thornwick-8b.gguf", draft_n=8, draft_n_accepted=7, completion_tokens=9)

    assert Call.from_kept(call.public()) == call


def test_speed_and_drafting_come_from_the_calls_the_run_kept(tmp_path):
    stats = run_stats(a_run(tmp_path, calls=[a_call()], rows=2))

    assert stats.sections == 2
    assert stats.wanted == 4
    assert stats.remaining == 2
    assert stats.spent.calls == 2
    assert stats.spent.completion_tokens == 2000
    assert stats.spent.decode_tokens_per_second == pytest.approx(50.0)
    assert stats.spent.prompt_tokens_per_second == pytest.approx(500.0)
    assert stats.spent.acceptance == pytest.approx(0.875)
    assert stats.prompt_share == pytest.approx(4 / 24)


def test_tokens_per_pass_counts_one_token_for_every_pass(tmp_path):
    stats = run_stats(a_run(tmp_path, calls=[a_call(completion=100, accepted=80, draft=90)]))

    assert stats.passes == 20
    assert stats.tokens_per_pass == pytest.approx(5.0)


def test_a_run_with_no_draft_head_says_so_rather_than_zero(tmp_path):
    stats = run_stats(a_run(tmp_path, calls=[a_call(draft=0, accepted=0)]))

    assert stats.drafting is False
    assert stats.passes is None
    assert stats.tokens_per_pass is None
    assert stats.spent.acceptance is None


def test_a_run_whose_calls_never_reached_the_server_measures_nothing(tmp_path):
    stats = run_stats(a_run(tmp_path, calls=[]))

    assert stats.spent.calls == 0
    assert stats.spent.decode_tokens_per_second is None
    assert stats.tokens_per_pass is None


def test_the_estimate_left_is_the_rate_this_store_measured(tmp_path):
    stats = run_stats(a_run(tmp_path, calls=[a_call()], rows=2))

    assert stats.per_section == pytest.approx(86.0)
    assert stats.left_seconds == pytest.approx(172.0)


def test_status_says_the_speed_the_drafting_and_the_time(tmp_path, capsys):
    where = a_run(tmp_path, calls=[a_call()], rows=2)

    assert status(where) == 0
    said = capsys.readouterr().out
    assert "50.0 tokens/second written" in said
    assert "500 tokens/second read" in said
    assert "higher is better" in said
    assert "87.5% of drafted tokens kept" in said
    assert "tokens per verification pass" in said
    assert "left at this rate" in said


def test_status_says_there_is_no_draft_head_rather_than_printing_zeros(tmp_path, capsys):
    where = a_run(tmp_path, calls=[a_call(draft=0, accepted=0)])

    assert status(where) == 0
    said = capsys.readouterr().out
    assert "no draft head" in said
    assert "0.0% of drafted tokens kept" not in said


def test_status_says_nothing_was_measured_when_no_call_reached_the_server(tmp_path, capsys):
    where = a_run(tmp_path, calls=[])

    assert status(where) == 0
    assert "no call reached the server" in capsys.readouterr().out


def test_the_head_and_the_depth_come_from_the_run_the_store_kept(tmp_path):
    pytest.importorskip("ladybug")
    where = a_run(tmp_path, calls=[a_call()])
    ingest.write_run(where, {"id": "run:20260906T101500", "model": "/w/thornwick-8b.gguf",
                             "n_max": 6, "sampling": {"temperature": 0.1},
                             "started": "2026-09-06T10:15:00",
                             "serving": "serve with --context 32768 "
                                        "--draft /w/heads/mtp-thornwick-Q8_0.gguf --spec "
                                        "draft-mtp"})

    stats = run_stats(where)

    assert stats.model == "thornwick-8b.gguf"
    assert stats.head == "mtp-thornwick-Q8_0.gguf"
    assert stats.head_depth == 6
    assert stats.sampling == {"temperature": 0.1}
