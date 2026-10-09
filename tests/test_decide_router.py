"""Choosing a backend, keeping it warm, and the pinned-file check."""

from __future__ import annotations

import json

import pytest
from decide_fakes import logprob_handler

from poolhouse.decide import Rule, RulesDecider, router
from poolhouse.decide.fetch import Pin, _verified
from poolhouse.decide.types import BackendUnavailable


def config(server, **kw):
    return router.Config(url=server.base_url, **kw)


def test_auto_takes_the_first_backend_in_order_that_is_available(server):
    fake = server(logprob_handler(lambda u: {"A": 0.2, "B": 0.8}))
    cfg = config(fake, order=("embed", "logprob", "rules"))
    got = router.decide("q", "s", ["x", "y"], config=cfg)
    assert (got.backend, got.choice) == ("logprob", "y")


def test_auto_skips_logprob_for_more_options_than_letters(server):
    fake = server(logprob_handler(lambda u: {"A": 1.0}))
    rules = RulesDecider([Rule("o3", contains=("three",))])
    cfg = config(fake, rules=rules, order=("logprob", "rules"))
    names = [f"o{i}" for i in range(30)]
    got = router.decide("q", "three", names, config=cfg)
    assert got.backend == "rules" and got.choice == "o3"


def test_nothing_available_says_why_for_each_backend():
    cfg = router.Config(url="http://127.0.0.1:9", order=("logprob", "embed", "rules"))
    with pytest.raises(BackendUnavailable) as err:
        router.decide("q", "s", ["x", "y"], config=cfg)
    text = str(err.value)
    assert "logprob: http://127.0.0.1:9 did not answer" in text
    assert "embed: needs an embedding server" in text and "rules: no rules" in text


def test_a_named_backend_is_used_even_when_others_come_first(server):
    fake = server(logprob_handler(lambda u: {"A": 0.9, "B": 0.1}))
    rules = RulesDecider([Rule("y", contains=("s",))])
    cfg = config(fake, backend="rules", rules=rules)
    assert router.decide("q", "s", ["x", "y"], config=cfg).backend == "rules"


def test_a_built_decider_is_reused_for_the_same_config(server):
    fake = server(logprob_handler(lambda u: {"A": 1.0}))
    cfg = config(fake)
    assert router.build("logprob", cfg) is router.build("logprob", cfg)
    assert router.build("logprob", cfg) is not router.build("logprob", config(fake))


def test_an_unknown_backend_is_an_error_naming_the_choices():
    assert "one of" in router.unavailable("quantum", router.Config())
    with pytest.raises(BackendUnavailable):
        router.build("quantum", router.Config())


def test_the_pointer_backend_reports_a_missing_download_without_fetching():
    why = router.unavailable("pointer", router.Config())
    assert why == "" or "poolhouse decide fetch" in why or "needs" in why


def test_a_pin_needs_full_length_hashes():
    with pytest.raises(ValueError, match="full hashes"):
        Pin("a/b", "abc", "f", "0" * 64, 1)


def test_a_file_is_verified_by_size_and_hash_and_the_result_is_remembered(tmp_path, monkeypatch):
    from poolhouse import home
    monkeypatch.setattr(home, "cache", lambda *parts: tmp_path.joinpath("cache", *parts))
    f = tmp_path / "w.bin"
    f.write_bytes(b"weights")
    import hashlib
    good = Pin("a/b", "1" * 40, "w.bin", hashlib.sha256(b"weights").hexdigest(), 7)
    assert _verified(good, f)
    marks = json.loads((tmp_path / "cache" / "decide" / "verified.json").read_text())
    assert good.sha256 in marks
    assert not _verified(Pin("a/b", "1" * 40, "w.bin", "0" * 64, 7), f)
    assert not _verified(Pin("a/b", "1" * 40, "w.bin", good.sha256, 8), f)
    f.write_bytes(b"WEIGHTS")
    assert not _verified(good, f)
