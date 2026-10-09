"""The daemon proves to the desktop app that it is the daemon, over a real socket."""
from __future__ import annotations

import hashlib
import hmac
import stat

import pytest
from test_fleet_ui import Serving

from poolhouse.fleet import app_identity

CHALLENGE = "ab" * 16


@pytest.fixture
def daemon(tmp_path):
    server = Serving(tmp_path, secure=False)
    server.ui.root = tmp_path / "traind"
    try:
        yield server
    finally:
        server.close()


def health(server, challenge=None, **options):
    headers = {app_identity.HEADER: challenge} if challenge is not None else {}
    return server.call("/health", ui_header=False, headers=headers, **options)


def test_health_proves_the_challenge_with_a_secret_only_the_root_holds(daemon):
    status, body, _ = health(daemon, CHALLENGE)
    assert status == 200 and body["ok"] is True
    secret = (daemon.ui.root / app_identity.PROOF_KEY_FILE).read_text().strip()
    assert body["proof"] == hmac.new(bytes.fromhex(secret), CHALLENGE.encode(), hashlib.sha256).hexdigest()
    assert stat.S_IMODE((daemon.ui.root / app_identity.PROOF_KEY_FILE).stat().st_mode) == 0o600
    other = health(daemon, "cd" * 16)[1]["proof"]
    assert other != body["proof"] and health(daemon, CHALLENGE)[1]["proof"] == body["proof"]


@pytest.mark.parametrize("challenge", [None, "", "short", "AB" * 16, "zz" * 16, "ab" * 100, "ab" * 16 + " "])
def test_health_without_a_well_formed_challenge_carries_no_proof(daemon, challenge):
    status, body, _ = health(daemon, challenge)
    assert status == 200 and body["ok"] is True and "proof" not in body


def test_a_secret_that_is_a_link_is_refused(tmp_path):
    target = tmp_path / "elsewhere"
    target.write_text("00" * 32)
    (tmp_path / app_identity.PROOF_KEY_FILE).symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        app_identity.prove(tmp_path, CHALLENGE)
    assert target.read_text() == "00" * 32


def test_a_damaged_secret_is_replaced_rather_than_trusted(tmp_path):
    (tmp_path / app_identity.PROOF_KEY_FILE).write_text("short")
    proof = app_identity.prove(tmp_path, CHALLENGE)
    assert len((tmp_path / app_identity.PROOF_KEY_FILE).read_text().strip()) == 64
    assert proof == app_identity.prove(tmp_path, CHALLENGE)
