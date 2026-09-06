"""Whether this machine is quiet enough for a timing taken on it to mean anything.

Four agent processes at about 190% CPU each, and a second model merely holding memory,
cost 40% of throughput on 2026-09-06 and made a day of timings worthless. Acceptance
survived, because token counts do not care about load; seconds did not.
"""

from __future__ import annotations

from ml_stack.bench import quiet
from ml_stack.bench.quiet import BUSY_LOAD, BUSY_PERCENT, Quiet, look


def _idle(monkeypatch, *, servers=(), measuring=None, load=0.05, busy=4.0):
    monkeypatch.setattr(quiet, "beside_on_the_card", lambda: list(servers))
    monkeypatch.setattr(quiet, "measurement_on_the_card", lambda: measuring)
    monkeypatch.setattr(quiet, "_load_per_core", lambda: load)
    monkeypatch.setattr(quiet, "_busy_percent", lambda: busy)


SERVER = {"port": 8080, "pid": 4242, "model": "marrowgate-Q4.gguf",
          "bytes": 61 * 2 ** 30, "leased": False}


class TestAQuietMachine:
    def test_nothing_running_is_quiet(self, monkeypatch):
        _idle(monkeypatch)
        found = look()
        assert found.ok
        assert found.reasons == ()

    def test_it_says_what_it_found_even_when_all_is_well(self, monkeypatch):
        _idle(monkeypatch)
        said = "\n".join(look().lines())
        assert "no model server is running" in said
        assert "measuring lock: free" in said
        assert "load 0.05 per core" in said and "cpu 4% busy" in said


class TestWhatItRefuses:
    def test_a_second_model_holding_the_card_is_named_with_how_to_stop_it(self,
                                                                         monkeypatch):
        _idle(monkeypatch, servers=(SERVER,))
        found = look()
        assert not found.ok
        said = "\n".join(found.reasons)
        assert "port 8080" in said and "marrowgate-Q4.gguf" in said
        assert "61.0G" in said and "pid 4242" in said
        assert "ml-stack-serve down --port 8080" in said

    def test_a_measurement_holding_the_lock_is_named_by_its_command(self, monkeypatch):
        held = {"pid": 99, "argv": ["drafts", "--sample", "40"], "started": "2026-01-01"}
        _idle(monkeypatch, measuring=held)
        said = "\n".join(look().reasons)
        assert "ml-stack-bench drafts --sample 40" in said
        assert "pid 99" in said
        assert "ml-stack-bench stop" in said

    def test_a_loaded_machine_is_refused_even_with_nothing_serving(self, monkeypatch):
        _idle(monkeypatch, load=BUSY_LOAD + 0.1, busy=5.0)
        assert not look().ok

    def test_busy_cores_under_the_load_bar_are_still_busy(self, monkeypatch):
        """Four processes at 190% on sixteen cores is 0.47 per core: under the load bar."""
        _idle(monkeypatch, load=0.47, busy=47.0)
        found = look()
        assert not found.ok
        assert "47% busy" in "\n".join(found.reasons)
        assert BUSY_PERCENT < 47.0

    def test_the_refusal_says_what_to_stop_and_names_the_way_past_it(self, monkeypatch):
        _idle(monkeypatch, servers=(SERVER,))
        said = "\n".join(look().refusal())
        assert "not quiet" in said
        assert "--anyway" in said
        assert "marked as taken on a busy machine" in said

    def test_a_lock_the_lease_guard_does_not_count_is_not_a_measurement(self, monkeypatch):
        """`measurement_on_the_card` already leaves out this process and its parents."""
        _idle(monkeypatch, measuring=None)
        assert look().ok

    def test_a_measurement_that_wrote_no_record_is_still_named(self, monkeypatch):
        _idle(monkeypatch, measuring={"pid": 7, "argv": [], "started": ""})
        assert "wrote no record" in "\n".join(look().reasons)


def test_a_reading_that_cannot_be_taken_is_left_out_rather_than_called_zero(monkeypatch):
    _idle(monkeypatch, busy=None)
    found = look()
    assert found.busy is None
    assert "cpu" not in "\n".join(found.lines())


def test_an_empty_quiet_is_quiet():
    assert Quiet().ok
