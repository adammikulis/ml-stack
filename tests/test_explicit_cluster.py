"""Explicit cluster creation and failed joins preserve membership state."""

import pytest

from ml_stack.fleet import discovery, peers
from ml_stack.fleet.onboard import joining


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(joining, "find_joiners", lambda *_args, **_kwargs: [])


def test_join_failure_preserves_existing_membership(tmp_path):
    path = tmp_path / "cluster.key"
    first = discovery.mint_cluster("default", path)
    with pytest.raises(discovery.DiscoveryError, match="No machine"):
        joining.join_existing("quince larch marlow", "default", path)
    assert discovery.memberships(path) == [first]


def test_explicit_creation_mints_a_key_with_join_credential(tmp_path):
    path = tmp_path / "cluster.key"
    member = joining.create_by_passphrase("quince larch marlow", " default ", path)
    assert member.group == "default"
    assert member.join == joining.join_secret("quince larch marlow", "default")
    assert discovery.memberships(path) == [member]


def test_creation_refuses_an_existing_lan_cluster(monkeypatch, tmp_path):
    monkeypatch.setattr(joining, "find_joiners", lambda *_args, **_kwargs: [object()])
    with pytest.raises(discovery.DiscoveryError, match="Join it instead"):
        joining.create_by_passphrase("quince larch marlow", "default", tmp_path / "cluster.key")
    assert discovery.memberships(tmp_path / "cluster.key") == []


@pytest.mark.parametrize("mode", ["join", "create", "unknown"])
def test_cluster_action_refuses_missing_name_or_unknown_mode(mode, tmp_path):
    with pytest.raises(discovery.DiscoveryError):
        joining.cluster_action(mode, "quince larch marlow", "", tmp_path / "cluster.key")
    assert discovery.memberships(tmp_path / "cluster.key") == []


def test_creation_refuses_to_replace_an_existing_key(tmp_path):
    path = tmp_path / "cluster.key"
    first = joining.create_by_passphrase("quince larch marlow", "default", path)
    with pytest.raises(discovery.DiscoveryError, match="already belongs"):
        joining.create_by_passphrase("other words here", "default", path)
    assert discovery.memberships(path) == [first]


def test_cli_setup_never_creates_when_join_discovery_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(peers.sys.stdin, "isatty", lambda: False)
    path = tmp_path / "cluster.key"
    assert peers.main(["--cluster-key", str(path), "setup", "--group", "default",
                       "--passphrase", "quince larch marlow"]) == 2
    assert discovery.memberships(path) == []


def test_cli_requires_a_cluster_name(tmp_path, monkeypatch):
    monkeypatch.setattr(peers.sys.stdin, "isatty", lambda: False)
    path = tmp_path / "cluster.key"
    assert peers.main(["--cluster-key", str(path), "setup", "--passphrase", "quince larch marlow"]) == 2
    assert discovery.memberships(path) == []
