"""A run says, and keeps, what else held the card while it was measured."""

from __future__ import annotations

from ml_stack.bench.keep import beside_the_run, note_beside, stamped
from ml_stack.bench.run import beside_on_the_card, note_beside_the_run


class _Processes:
    def __init__(self, found, leased):
        self.found = tuple(found)
        self.leased = frozenset(leased)
        self.strays = ()
        self.foreign = ()


def test_every_live_server_is_found_leased_or_not(monkeypatch):
    monkeypatch.setattr(
        "ml_stack.bench.run.processes",
        lambda: _Processes(
            [{"port": 8080, "pid": 11, "model": "/w/big.gguf", "rss": 2 ** 30, "defunct": False},
             {"port": 8081, "pid": 12, "model": "/w/small.gguf", "rss": 2 ** 29,
              "defunct": False},
             {"port": 8082, "pid": 13, "model": "", "rss": 0, "defunct": True}],
            [8080]))

    found = beside_on_the_card()
    assert [one["port"] for one in found] == [8080, 8081], "the defunct one holds nothing"
    assert found[0] == {"port": 8080, "pid": 11, "model": "big.gguf", "bytes": 2 ** 30,
                        "leased": True}
    assert found[1]["leased"] is False


def test_the_line_names_each_server_and_what_it_costs(monkeypatch):
    monkeypatch.setattr(
        "ml_stack.bench.run.beside_on_the_card",
        lambda: [{"port": 8080, "pid": 11, "model": "big.gguf", "bytes": 2 ** 30,
                  "leased": True},
                 {"port": 8081, "pid": 12, "model": "small.gguf", "bytes": 2 ** 29,
                  "leased": False}])
    said = note_beside_the_run()
    assert said.startswith("2 server(s) already hold this card: ")
    assert ":8080 big.gguf 1.0G" in said
    assert ":8081 small.gguf 512.0M, not leased" in said
    assert "in these timings" in said
    assert len(beside_the_run()) == 2, "kept for the runs this measurement writes"
    note_beside([])


def test_a_run_kept_alone_carries_no_beside():
    note_beside([])
    assert beside_the_run() == []
    assert "beside" not in stamped({"model": "big.gguf"})


def test_a_run_measured_with_company_keeps_it():
    note_beside([{"port": 8081, "pid": 12, "model": "small.gguf", "bytes": 2 ** 30,
                  "leased": False}])
    try:
        assert stamped({"model": "big.gguf"})["beside"] == [
            {"port": 8081, "pid": 12, "model": "small.gguf", "bytes": 2 ** 30,
             "leased": False}]
    finally:
        note_beside([])
