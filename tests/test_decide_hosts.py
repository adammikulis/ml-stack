"""A decider judges tool calls, so it talks only to this machine unless the operator names the host."""

from __future__ import annotations

import pytest

from ml_stack import httpguard
from ml_stack.decide import router
from ml_stack.decide.logprob import LogprobDecider, require_decider_host
from ml_stack.decide.types import DecideError

REMOTE = "http://203.0.113.7:8080"          # TEST-NET-3: never routable, never contacted by these tests


@pytest.mark.parametrize("url", ["http://127.0.0.1:8080", "http://localhost:9000", "http://[::1]:8080",
                                 "http://127.0.0.2:1"])
def test_this_machine_is_always_allowed(url, monkeypatch):
    monkeypatch.delenv(httpguard.ALLOW_ENV, raising=False)
    require_decider_host(url)
    assert LogprobDecider(url).base_url == url


@pytest.mark.parametrize("url", [REMOTE, "https://api.example.com/v1", "http://192.168.1.20:8080",
                                 "http://localhost.evil.example:8080", "http://127.0.0.1.evil.example"])
def test_any_other_host_is_refused_by_default(url, monkeypatch):
    monkeypatch.delenv(httpguard.ALLOW_ENV, raising=False)
    with pytest.raises(DecideError, match=httpguard.ALLOW_ENV):
        LogprobDecider(url)


def test_an_operator_named_host_is_allowed(monkeypatch):
    monkeypatch.setenv(httpguard.ALLOW_ENV, "192.168.1.20, other.lan")
    require_decider_host("http://192.168.1.20:8080")
    require_decider_host("http://other.lan/v1")
    with pytest.raises(DecideError):
        require_decider_host(REMOTE)


def test_the_router_does_not_even_probe_a_remote_decider(monkeypatch):
    """The reachability probe would send a request to the host; the socket seal in conftest fails
    the test if anything tries, so reaching the message proves nothing was sent."""
    monkeypatch.delenv(httpguard.ALLOW_ENV, raising=False)
    why = router._reachable(REMOTE, "")
    assert "not on this machine" in why
