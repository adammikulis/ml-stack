"""Experimental features, attacked: hostile names, a hostile settings file, the route and the floors."""

from __future__ import annotations

import json

import pytest
from test_features_routes import daemon, the_passphrase_is_kept  # noqa: F401
from test_fleet_ui import a_keystore, counting  # noqa: F401

from ml_stack import features, features_cli


@pytest.fixture
def machine(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
    return tmp_path / "state" / "traind" / "settings.json"


HOSTILE = ["../../etc/passwd", "test-runner-extras; rm -rf /", "test-runner-extras\nkeystore", "$(touch pwned)", "x" * 100_000,
           "", " ", "TEST-RUNNER-EXTRAS", "test_runner_extras", "test-runner-extras ", "\u0000", "__proto__", "constructor"]


@pytest.mark.parametrize("name", HOSTILE)
def test_a_hostile_name_is_refused_and_writes_nothing(machine, capsys, name):
    assert features_cli.main(["enable", name]) == 2
    assert not machine.exists()
    with pytest.raises(features.UnknownFeature):
        features.enabled(name)


@pytest.mark.parametrize("name", ["push-to-main", "keystore", "agent-is-human", "share-the-gpu"])
def test_a_floor_cannot_be_switched_because_it_cannot_be_registered(machine, capsys, name):
    assert name not in features.FEATURES
    assert features_cli.main(["enable", name]) == 2
    assert not machine.exists()


@pytest.mark.parametrize("held", [
    '{"features": {"keystore": true, "push-to-main": true, "test-runner-extras": 1}}',
    '{"features": {"test-runner-extras": "true"}}', '{"features": [true]}', '[]', '"features"', "\x00\x01", "{" * 5000])
def test_a_settings_file_naming_other_things_turns_nothing_on(machine, held):
    machine.parent.mkdir(parents=True)
    machine.write_text(held)
    assert [row["name"] for row in features.listing() if row["enabled"]] == []


def test_enabling_one_feature_leaves_every_other_off(machine):
    features.switch("test-runner-extras", True)
    assert json.loads(machine.read_text())["features"] == {"test-runner-extras": True}
    assert features.enabled("guard-change") is False and features.enabled("remote-tests") is False


@pytest.mark.parametrize("path", ["/ui/features/../settings", "/ui/features/x", "/ui/features?name=test-runner-extras&enabled=1",
                                  "/ui/features%00"])
def test_the_route_takes_no_name_and_changes_nothing(daemon, path):  # noqa: F811
    daemon.call(path, method="POST", body={"name": "test-runner-extras", "enabled": True})
    assert features.enabled("test-runner-extras", str(daemon.ui.settings_path.parent)) is False


def test_the_route_needs_the_page_header(daemon):  # noqa: F811
    status, _, _ = daemon.call("/ui/features", ui_header=False)
    assert status == 403
