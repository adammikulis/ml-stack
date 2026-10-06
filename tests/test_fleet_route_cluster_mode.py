"""Cluster enrollment forwards requested mode separately from the join action."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ml_stack.fleet import routes
from ml_stack.fleet.discovery import DiscoveryError


class Enrollment(routes.ClusterRoutes):
    method = "POST"

    def __init__(self, request, path):
        self.request = request
        self.answers = []
        self.ui = SimpleNamespace(cluster_key_path=path, discovery_port=1234,
                                  join_guard=nullcontext, rejoined=lambda: None)

    def body(self):
        return self.request

    def send(self, status, body, *args):
        self.answers.append((status, body))


@pytest.mark.parametrize("requested", [None, "dev", "prod"])
@pytest.mark.parametrize("action", [None, "join", "create"])
def test_enrollment_preserves_optional_cluster_mode(monkeypatch, tmp_path, requested, action):
    calls = []
    monkeypatch.setattr(routes, "join_by_passphrase", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(routes, "cluster_action", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(routes.recovery, "remember", lambda *args: None)
    monkeypatch.setattr(routes, "memberships", lambda *args: [])
    request = {"group": "lab", "passphrase": "cedar lantern meadow"}
    if requested is not None:
        request["cluster_mode"] = requested
    if action is not None:
        request["mode"] = action
    route = Enrollment(request, tmp_path / "cluster.key")
    assert route._clusters()
    assert route.answers[0][0] == 200
    assert calls == [{"port": 1234, "cluster_mode" if action else "mode": requested}]


def test_rejected_mode_does_not_remember_passphrase_or_rejoin(monkeypatch, tmp_path):
    def reject(*args, **kwargs):
        raise DiscoveryError("cluster mode does not match")

    def forbidden(*args):
        pytest.fail("a rejected enrollment changed local setup")

    monkeypatch.setattr(routes, "join_by_passphrase", reject)
    monkeypatch.setattr(routes.recovery, "remember", forbidden)
    route = Enrollment({"group": "lab", "passphrase": "cedar lantern meadow",
                        "cluster_mode": "prod"}, tmp_path / "cluster.key")
    route.ui.rejoined = forbidden
    assert route._clusters()
    assert route.answers == [(400, {"error": "cluster mode does not match"})]
