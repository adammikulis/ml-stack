"""``/ui/features``: the settings screen's list of experimental features, from a real daemon."""

from __future__ import annotations

import pytest
from test_fleet_ui import Serving, a_keystore, counting  # noqa: F401

from ml_stack import features


@pytest.fixture(autouse=True)
def the_passphrase_is_kept(a_keystore):  # noqa: F811
    """Joining stores the passphrase and signing in compares against it."""


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    from ml_stack.fleet.onboard import joining

    monkeypatch.setattr(joining, "find_joiners", lambda *a, **k: [])
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
    s = Serving(tmp_path)
    try:
        yield s
    finally:
        s.close()


def test_the_route_lists_every_feature_with_stage_risk_and_state(daemon):
    status, body, _ = daemon.call("/ui/features")
    assert status == 200
    by_name = {row["name"]: row for row in body["features"]}
    assert set(by_name) == set(features.FEATURES)
    assert by_name["windows-node"]["stage"] == "experimental" and by_name["windows-node"]["risk"]
    assert not any(row["enabled"] for row in body["features"])


def test_the_route_reads_the_daemons_own_settings_file(daemon):
    features.switch("windows-node", True, root=str(daemon.ui.settings_path.parent))
    _, body, _ = daemon.call("/ui/features")
    assert {row["name"] for row in body["features"] if row["enabled"]} == {"windows-node"}


def test_the_route_changes_nothing(daemon):
    status, _, _ = daemon.call("/ui/features", method="POST", body={"name": "windows-node", "enabled": True})
    assert status != 200 and not features.enabled("windows-node", str(daemon.ui.settings_path.parent))


def test_the_page_carries_the_component_on_the_settings_screen(daemon):
    from ml_stack.fleet.page import render

    page = render()
    assert page.count("<feature-flags>") == 1 and 'customElements.define("feature-flags"' in page
