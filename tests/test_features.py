"""Experimental features: the registry, its state in settings.json, the command, the audit row and the floors."""

from __future__ import annotations

import json

import pytest

from poolhouse import authority, features, features_cli
from poolhouse.fleet.settings import Settings


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A state root of its own, so the settings file is this test's."""
    monkeypatch.setenv("POOLHOUSE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv(features.ACTOR_ENV, raising=False)
    return tmp_path / "state" / "traind" / "settings.json"


@pytest.fixture
def scratch(monkeypatch):
    """The registry as it was, whatever a test registers."""
    monkeypatch.setattr(features, "FEATURES", dict(features.FEATURES))


def test_the_first_ones_are_registered_experimental_and_off(machine):
    assert {"test-runner-extras", "guard-change"} <= set(features.FEATURES)
    for name in ("test-runner-extras", "guard-change"):
        one = features.FEATURES[name]
        assert one.stage == "experimental" and one.about and one.risk
        assert features.enabled(name) is False


def test_an_unregistered_name_is_an_error_not_an_off(machine):
    with pytest.raises(features.UnknownFeature):
        features.enabled("windows-node")  # the Windows node has no feature: it is just how Windows runs
    with pytest.raises(features.UnknownFeature):
        features.enabled("test-runner-extra")
    with pytest.raises(features.UnknownFeature):
        features.switch("nothing-here", True)


def test_enabling_is_kept_in_the_settings_file_and_disabling_takes_it_back(machine):
    assert features.switch("test-runner-extras", True) is True
    assert features.enabled("test-runner-extras") is True
    assert json.loads(machine.read_text())["features"] == {"test-runner-extras": True}
    assert features.switch("test-runner-extras", True) is False
    features.switch("test-runner-extras", False)
    assert features.enabled("test-runner-extras") is False


def test_the_features_share_the_file_the_settings_already_use(machine):
    Settings(name="studio").save(machine)
    features.switch("test-runner-extras", True)
    kept = Settings.load(machine)
    assert kept.name == "studio" and kept.features == {"test-runner-extras": True}


def test_a_daemon_saving_its_stale_settings_does_not_undo_a_switch(machine):
    stale = Settings(name="studio")
    stale.save(machine)
    features.switch("test-runner-extras", True)
    stale.save(machine)
    assert features.enabled("test-runner-extras") is True


@pytest.mark.parametrize("held", ['{"features": ["test-runner-extras"]}', '{"features": {"test-runner-extras": "yes"}}',
                                  "not json", '{"features": null}'])
def test_a_settings_file_that_says_something_else_reads_as_off(machine, held):
    machine.parent.mkdir(parents=True)
    machine.write_text(held)
    assert features.enabled("test-runner-extras") is False


def test_a_switch_is_in_the_authority_audit_log_with_who_did_it(machine, monkeypatch):
    features.switch("test-runner-extras", True)
    monkeypatch.setenv(features.ACTOR_ENV, "claude-abc123")
    features.switch("test-runner-extras", False)
    rows = [r for r in authority.audit_rows() if r["event"] == "features.set"]
    assert [(r["by"], r["feature"], r["to"]) for r in rows] == [
        (authority.PERSON, "test-runner-extras", True), ("claude-abc123", "test-runner-extras", False)]


def test_the_command_lists_enables_and_disables(machine, capsys):
    assert features_cli.main(["list"]) == 0
    listed = capsys.readouterr().out
    assert "test-runner-extras  [experimental]  off" in listed and "risk:" in listed
    assert features_cli.main(["enable", "test-runner-extras"]) == 0
    assert features.enabled("test-runner-extras")
    assert json.loads((features_cli.main(["list", "--json"]), capsys.readouterr().out.split("\n", 1)[1])[1])
    assert features_cli.main(["disable", "test-runner-extras"]) == 0
    assert not features.enabled("test-runner-extras")


def test_the_command_refuses_a_name_nothing_registered(machine, capsys):
    assert features_cli.main(["enable", "push-to-main"]) == 2
    assert "no feature named" in capsys.readouterr().err
    assert not machine.exists()


def test_the_command_takes_a_root_for_a_daemon_kept_elsewhere(machine, tmp_path):
    other = tmp_path / "elsewhere"
    assert features_cli.main(["enable", "test-runner-extras", "--root", str(other)]) == 0
    assert json.loads((other / "settings.json").read_text())["features"] == {"test-runner-extras": True}
    assert features.enabled("test-runner-extras") is False and features.enabled("test-runner-extras", str(other)) is True


@pytest.mark.parametrize("name", [
    "push-to-main", "allow-push-main", "agent-is-human", "agents-as-person", "agent-human-identity",
    "keystore", "keystore-unlock", "share-the-gpu", "gpu-sharing", "two-gpu-jobs", "gpu-lease-bypass",
])
def test_a_safety_floor_is_never_registered(scratch, name):
    with pytest.raises(ValueError, match="safety floor"):
        features.register(name, "experimental", "Loosen it.", "Everything.")
    assert name not in features.FEATURES


def test_no_registered_feature_is_a_floor():
    assert [n for n in features.FEATURES if features._aimed_at(n)] == []


@pytest.mark.parametrize("args", [
    ("bad_name!", "experimental", "a", "b"), ("good-name", "alpha", "a", "b"),
    ("good-name", "experimental", "", "b"), ("good-name", "experimental", "a", ""),
    ("good-name", "experimental", "two\nlines", "b"), ("test-runner-extras", "experimental", "again", "b")])
def test_a_malformed_or_repeated_registration_is_refused(scratch, args):
    with pytest.raises(ValueError):
        features.register(*args)


def test_a_new_feature_starts_off(machine, scratch):
    features.register("tidy-things", "beta", "Tidies.", "Moves files.")
    assert features.enabled("tidy-things") is False
    assert [r["name"] for r in features.listing() if r["enabled"]] == []
