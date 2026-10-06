"""Automatic cluster selection at the Fleet command boundary."""

from types import SimpleNamespace

import pytest

from ml_stack.fleet import join as joining


@pytest.fixture
def isolated_join(monkeypatch):
    monkeypatch.setattr(joining, "checks", lambda *a, **kw: [])
    monkeypatch.setattr(joining, "already_running", lambda port: {"name": "larch"})
    monkeypatch.setattr(joining, "peers", lambda **kw: [])
    monkeypatch.setattr(joining.recovery, "remember", lambda *a, **kw: None)


@pytest.mark.parametrize("mode", [None, "dev", "prod"])
def test_no_passphrase_uses_automatic_cluster_without_enrollment(
        isolated_join, monkeypatch, tmp_path, mode):
    from ml_stack.fleet import automatic_clusters

    asked = []
    actual = mode or "dev"
    member = SimpleNamespace(group="orchard", mode=actual)
    monkeypatch.setattr(automatic_clusters, "ensure",
                        lambda path, **kw: asked.append((path, kw)) or member)
    key = tmp_path / "cluster.key"
    joined = joining.join_machine(root=tmp_path, cluster_key_path=key, port=9123,
                                  mode=mode, say=lambda text: None,
                                  enrol=lambda *a: pytest.fail("automatic enrollment used a passphrase"))
    assert joined.group == "orchard" and joined.mode == actual
    assert asked == [(key, {"mode": mode, "port": 9124})]
    assert not joined.started


def test_automatic_cluster_uses_explicit_discovery_port(isolated_join, monkeypatch, tmp_path):
    from ml_stack.fleet import automatic_clusters

    asked = []
    monkeypatch.setattr(automatic_clusters, "ensure", lambda path, **kw:
                        asked.append(kw) or SimpleNamespace(group="orchard", mode="dev"))
    joining.join_machine(root=tmp_path, discovery_port=9456, say=lambda text: None)
    assert asked[0]["port"] == 9456


@pytest.mark.parametrize("mode, expected", [(None, "prod"), ("dev", "dev"), ("prod", "prod")])
def test_passphrase_enrollment_preserves_selected_mode(isolated_join, monkeypatch, tmp_path,
                                                       mode, expected):
    monkeypatch.setattr(joining, "already_running", lambda port: None)
    monkeypatch.setattr(joining, "wait_for_health", lambda *a, **kw: {})
    asked = []
    monkeypatch.setattr(joining, "join_by_passphrase", lambda *a, **kw: asked.append((a, kw)))
    monkeypatch.setattr(joining, "memberships", lambda path: [])
    joined = joining.join_machine(root=tmp_path, passphrase="quince larch marlow",
                                  group="orchard", mode=mode, start=lambda *a: 42,
                                  say=lambda text: None)
    assert asked[0][1]["mode"] == expected
    assert joined.mode == expected


def test_no_argument_terminal_does_not_prompt_or_read_stdin(monkeypatch, tmp_path):
    class Terminal:
        def isatty(self):
            return True

        def read(self, *a):
            pytest.fail("automatic join read stdin")

    import sys

    monkeypatch.setattr(sys, "stdin", Terminal())
    monkeypatch.delenv("ML_STACK_PASSPHRASE", raising=False)
    monkeypatch.delenv("ML_STACK_CLUSTER", raising=False)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("automatic join prompted"))
    asked = []
    monkeypatch.setattr(joining, "join_machine", lambda **kw: asked.append(kw) or joining.Joined(name="larch", port=1, root=tmp_path, group=""))
    assert joining.main(["--root", str(tmp_path), "join"]) == 0
    assert asked[0]["passphrase"] == "" and asked[0]["group"] == ""
    assert asked[0]["mode"] is None


def test_automatic_admission_finishes_before_first_daemon_start(isolated_join, monkeypatch,
                                                               tmp_path):
    from ml_stack.fleet import automatic_clusters

    events = []
    monkeypatch.setattr(joining, "already_running", lambda port: None)
    monkeypatch.setattr(joining, "wait_for_health", lambda *a, **kw: {})
    monkeypatch.setattr(automatic_clusters, "ensure", lambda *a, **kw:
                        events.append("admitted") or SimpleNamespace(group="orchard", mode="dev"))
    joined = joining.join_machine(root=tmp_path, say=lambda text: None,
                                  start=lambda *a: events.append("started") or 42)
    assert events == ["admitted", "started"]
    assert joined.started and joined.daemon_pid == 42


def test_automatic_admission_failure_does_not_start_daemon(isolated_join, monkeypatch, tmp_path):
    from ml_stack.fleet import automatic_clusters
    from ml_stack.fleet.discovery import DiscoveryError

    def refuse(*a, **kw):
        raise DiscoveryError("Production mode needs an explicitly admitted Production cluster")

    monkeypatch.setattr(automatic_clusters, "ensure", refuse)
    with pytest.raises(DiscoveryError, match="explicitly admitted"):
        joining.join_machine(root=tmp_path, mode="prod", say=lambda text: None,
                             start=lambda *a: pytest.fail("started before admission"))
