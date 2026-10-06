"""Persisted cluster profiles and explicit Production admission."""

import base64
import json

import pytest

from ml_stack.fleet import automatic_clusters, discovery, recovery
from ml_stack.fleet.onboard import joining

KEY = base64.urlsafe_b64encode(bytes(range(32))).rstrip(b"=")


def test_legacy_single_key_is_not_adopted(tmp_path):
    path = tmp_path / "cluster.key"
    path.write_text(KEY.decode())
    path.with_suffix(".group").write_text("laboratory")
    assert discovery.memberships(path) == []
    assert not discovery.clusters_path(path).exists()


def test_mode_less_membership_list_is_not_adopted(tmp_path):
    path = tmp_path / "cluster.key"
    discovery.clusters_path(path).write_text(json.dumps([
        {"group": "laboratory", "key": KEY.decode(), "join": "secret"}]))
    assert discovery.memberships(path) == []


@pytest.mark.parametrize("field,value", [
    ("group", 42), ("group", ""), ("key", 42), ("key", "invalid"),
    ("join", None), ("join", 42), ("mode", None), ("mode", "development"),
    ("mode", []),
])
def test_malformed_profile_is_skipped_without_shadowing_valid_member(tmp_path, field, value):
    path = tmp_path / "cluster.key"
    bad = {"group": "laboratory", "key": KEY.decode(), "join": "", "mode": "prod"}
    bad[field] = value
    good = {"group": "laboratory", "key": KEY.decode(), "join": "", "mode": "prod"}
    discovery.clusters_path(path).write_text(json.dumps([bad, good]))
    assert discovery.memberships(path) == [discovery.Membership("laboratory", KEY, mode="prod")]


@pytest.mark.parametrize("mode", ["dev", "prod"])
def test_recovery_round_trip_preserves_cluster_profile(tmp_path, mode):
    original = tmp_path / "original.key"
    restored = tmp_path / "restored.key"
    member = discovery.adopt(discovery.Membership("laboratory", KEY, mode=mode), original)
    document = tmp_path / "cluster.recovery"
    recovery.export_recovery(document, "laboratory", original)
    assert recovery.parse_recovery(document.read_text()) == member
    assert recovery.import_recovery(document, restored) == member
    assert discovery.memberships(restored) == [member]


def test_mode_less_recovery_is_refused():
    with pytest.raises(discovery.DiscoveryError, match="valid cluster recovery"):
        recovery.parse_recovery(json.dumps({"group": "laboratory", "key": KEY.decode()}))


@pytest.mark.parametrize("mode", [None, "development", "PROD", 42, []])
def test_invalid_recovery_mode_is_refused(mode):
    with pytest.raises(discovery.DiscoveryError, match="valid cluster recovery"):
        recovery.parse_recovery(json.dumps({"group": "laboratory", "key": KEY.decode(), "mode": mode}))


@pytest.mark.parametrize("mode", [None, "development", "PROD", 42, []])
def test_public_membership_refuses_invalid_mode(mode):
    with pytest.raises(ValueError, match="cluster mode"):
        discovery.Membership("laboratory", KEY, mode=mode)


def test_explicit_production_membership_cannot_be_downgraded(tmp_path, monkeypatch):
    path = tmp_path / "cluster.key"
    member = discovery.adopt(discovery.Membership("laboratory", KEY, mode="prod"), path)
    def refuse_discovery(*_args, **_kwargs):
        pytest.fail("Production membership must not discover automatic admission")
    monkeypatch.setattr(automatic_clusters, "offers", refuse_discovery)
    assert automatic_clusters.ensure(path) == member
    with pytest.raises(discovery.DiscoveryError, match="explicitly admitted Development"):
        automatic_clusters.ensure(path, mode="dev")
    assert discovery.memberships(path) == [member]


def test_production_server_refuses_automatic_admission_to_secondary_development_cluster():
    production = discovery.Membership("production", KEY, mode="prod")
    development = discovery.Membership("development", KEY, mode="dev")
    server = joining.Joining(lambda: [production, development], lambda: "a" * 64, log=lambda _s: None)
    status, answer = server.handle(f"{joining.API}/automatic", {
        "group": "development", "nonce": "a" * 32, "mode": "dev"},
        "192.0.2.10", transport_tls=True)
    assert status == 403
    assert "key" not in answer


@pytest.mark.parametrize("key", [KEY + b"=", b"x", base64.urlsafe_b64encode(bytes(31)).rstrip(b"=")])
def test_membership_key_requires_canonical_256_bit_encoding(key):
    with pytest.raises(ValueError):
        discovery.Membership("laboratory", key, mode="prod")


@pytest.mark.parametrize("requested", [None, "dev", "prod"])
def test_offline_membership_respects_explicit_admission_mode(tmp_path, monkeypatch, requested):
    path = tmp_path / "cluster.key"
    member = discovery.adopt(discovery.Membership("laboratory", KEY, mode="dev"), path)
    monkeypatch.setattr(joining, "find_joiners", lambda *_args, **_kwargs: [])
    options = joining.JoinOptions(mode=requested)
    if requested == "prod":
        with pytest.raises(discovery.DiscoveryError, match="differs from the requested mode"):
            joining.join_by_passphrase("cedar lantern meadow", "laboratory", path, options=options)
    else:
        assert joining.join_by_passphrase("cedar lantern meadow", "laboratory", path, options=options) == member
    assert discovery.memberships(path) == [member]
