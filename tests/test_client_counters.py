"""The server's speculative counters, read off /metrics and differenced across a block.

Every payload here is the text llama.cpp writes, copied in shape from `server-task.cpp`'s
own Prometheus writer: `# HELP`, `# TYPE`, a bare counter line, and one labelled series per
draft position. No server is started; the parser and its arithmetic are the whole point.
"""

from __future__ import annotations

import pytest

from ml_stack.client.counters import (
    GREEDY,
    Block,
    Ledger,
    Speculative,
    counting,
    read_speculative,
    sampling_named,
)

METRICS = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 3046
# HELP llamacpp:n_decode_total Total number of llama_decode() calls
# TYPE llamacpp:n_decode_total counter
llamacpp:n_decode_total 552
# HELP llamacpp:spec_decode_num_draft_tokens_total Speculative: Total draft tokens generated
# TYPE llamacpp:spec_decode_num_draft_tokens_total counter
llamacpp:spec_decode_num_draft_tokens_total 2160
# HELP llamacpp:spec_decode_num_accepted_tokens_total Speculative: Total draft tokens accepted
# TYPE llamacpp:spec_decode_num_accepted_tokens_total counter
llamacpp:spec_decode_num_accepted_tokens_total 2105
# HELP llamacpp:spec_decode_num_drafts_total Speculative: Total verification steps
# TYPE llamacpp:spec_decode_num_drafts_total counter
llamacpp:spec_decode_num_drafts_total 552
# HELP llamacpp:requests_processing Number of requests processing
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 0
# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft position
# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 548
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 540
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="2"} 525
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="3"} 492
"""

NO_DRAFT_HEAD = """\
# HELP llamacpp:n_decode_total Total number of llama_decode() calls
# TYPE llamacpp:n_decode_total counter
llamacpp:n_decode_total 552
# HELP llamacpp:requests_processing Number of requests processing
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 0
"""


@pytest.fixture
def served(monkeypatch):
    """A server whose /metrics answers with whatever the test last handed it."""
    body = {"text": METRICS}

    def _bytes(url, **_):
        assert url.endswith("/metrics"), url
        if body["text"] is None:
            from ml_stack.http import ServerError

            raise ServerError("501 metrics endpoint is disabled")
        return body["text"].encode()

    monkeypatch.setattr("ml_stack.client.counters.request_bytes", _bytes)
    return body


# -- reading one --------------------------------------------------------------------------
class TestReadingTheCounters:
    def test_it_reads_the_totals_and_the_per_position_series(self, served):
        got = read_speculative("http://127.0.0.1:8099")
        assert got is not None
        assert (got.drafts, got.drafted, got.accepted) == (552, 2160, 2105)
        assert got.per_position == (548, 540, 525, 492)

    def test_the_rates_are_what_the_counts_divide_to(self, served):
        got = read_speculative("http://127.0.0.1:8099")
        assert got is not None
        assert got.acceptance == pytest.approx(2105 / 2160)
        assert got.depth_offered == pytest.approx(2160 / 552)
        # one verification step yields the accepted draft tokens plus the free one
        assert got.tokens_per_draft == pytest.approx(1 + 2105 / 552)

    def test_survival_falls_off_with_position(self, served):
        got = read_speculative("http://127.0.0.1:8099")
        assert got is not None
        survival = got.survival
        assert survival == tuple(sorted(survival, reverse=True)), \
            "acceptance is a prefix, so a deeper position can never survive more often"
        assert survival[0] == pytest.approx(548 / 552)

    def test_a_server_with_no_draft_head_reports_nothing_rather_than_zero(self, served):
        served["text"] = NO_DRAFT_HEAD
        assert read_speculative("http://127.0.0.1:8099") is None

    def test_a_server_with_the_endpoint_off_reports_nothing(self, served):
        served["text"] = None
        assert read_speculative("http://127.0.0.1:8099") is None


# -- the difference across a block ---------------------------------------------------------
class TestOneBlockOfWork:
    def test_the_block_is_the_difference_and_not_the_running_total(self, served):
        with counting("http://127.0.0.1:8099", workload="ingest", sampling=GREEDY) as block:
            served["text"] = METRICS.replace(
                "spec_decode_num_drafts_total 552", "spec_decode_num_drafts_total 652"
            ).replace(
                "spec_decode_num_draft_tokens_total 2160",
                "spec_decode_num_draft_tokens_total 2560",
            ).replace(
                "spec_decode_num_accepted_tokens_total 2105",
                "spec_decode_num_accepted_tokens_total 2485",
            )
        moved = block.delta
        assert moved is not None
        assert (moved.drafts, moved.drafted, moved.accepted) == (100, 400, 380)
        assert moved.acceptance == pytest.approx(0.95)

    def test_a_block_on_a_server_that_counts_nothing_says_so(self, served):
        served["text"] = NO_DRAFT_HEAD
        with counting("http://127.0.0.1:8099", workload="ask", sampling=GREEDY) as block:
            pass
        assert not block.measured and block.delta is None
        assert "no speculative counters" in block.why
        assert "acceptance" not in block.public(), "an unmeasured block reports no rate"

    def test_a_server_restarted_mid_block_is_not_a_measurement(self, served):
        with counting("http://127.0.0.1:8099", workload="chat", sampling=GREEDY) as block:
            served["text"] = METRICS.replace(
                "spec_decode_num_drafts_total 552", "spec_decode_num_drafts_total 4"
            )
        assert not block.measured
        assert "restarted" in block.why

    def test_work_sharing_the_server_is_flagged_rather_than_folded_in(self, served):
        served["text"] = METRICS.replace("requests_processing 0", "requests_processing 1")
        with counting("http://127.0.0.1:8099", workload="ingest", sampling=GREEDY) as block:
            pass
        assert block.note and "another request" in block.note

    def test_the_block_carries_its_workload_and_its_sampling(self, served):
        with counting("http://127.0.0.1:8099", workload="chat", sampling="t1") as block:
            pass
        said = block.public()
        assert said["workload"] == "chat" and said["sampling"] == "t1"


# -- per workload and pooled ---------------------------------------------------------------
class TestTheLedger:
    def _block(self, workload, sampling, counts, per_pos=(0,)):
        drafts, drafted, accepted = counts
        return Block(workload=workload, sampling=sampling, before=Speculative(),
                     after=Speculative(drafts=drafts, drafted=drafted, accepted=accepted,
                                       per_position=per_pos))

    def test_blocks_add_up_per_workload_and_sampling_apart(self):
        led = Ledger()
        led.add(self._block("ingest", GREEDY, (100, 400, 380)))
        led.add(self._block("ingest", GREEDY, (100, 400, 384)))
        led.add(self._block("ingest", "t1", (100, 400, 300)))
        by = led.by_workload()
        assert by[("ingest", GREEDY)].accepted == 764
        assert by[("ingest", "t1")].accepted == 300, \
            "sampling changes what acceptance means, so the two never pool"

    def test_the_overall_line_pools_every_workload(self):
        led = Ledger()
        led.add(self._block("ingest", GREEDY, (100, 400, 380)))
        led.add(self._block("ask", GREEDY, (50, 200, 150)))
        pooled = led.overall()
        assert pooled is not None
        assert (pooled.drafts, pooled.drafted, pooled.accepted) == (150, 600, 530)

    def test_pooling_lines_up_the_positions_rather_than_truncating(self):
        led = Ledger()
        led.add(self._block("ingest", GREEDY, (10, 40, 38), per_pos=(10, 9, 9, 8)))
        led.add(self._block("ask", GREEDY, (10, 20, 15), per_pos=(9, 6)))
        pooled = led.overall()
        assert pooled is not None
        assert pooled.per_position == (19, 15, 9, 8), \
            "a shallower run contributes nothing to the positions it never drafted"

    def test_an_unmeasured_block_stays_out_of_every_total_and_is_named(self):
        led = Ledger()
        led.add(self._block("ingest", GREEDY, (100, 400, 380)))
        led.add(Block(workload="ask", sampling=GREEDY))
        assert led.overall() is not None and led.overall().drafts == 100
        assert any("not measured" in line for line in led.lines())

    def test_a_ledger_with_nothing_measured_has_no_overall_line(self):
        led = Ledger()
        led.add(Block(workload="ask", sampling=GREEDY))
        assert led.overall() is None
        assert led.public()["overall"] is None


# -- naming the regime ----------------------------------------------------------------------
def test_temperature_zero_is_greedy_and_anything_else_says_what_it_was():
    assert sampling_named(0) == GREEDY and sampling_named(0.0) == GREEDY
    assert sampling_named(1.0) == "t1"
    assert sampling_named(0.7) == "t0.7"
    assert sampling_named(1.0, top_p=0.9) == "t1/p0.9"
    assert sampling_named(None) == "unsaid", "a regime nobody recorded is not greedy"
