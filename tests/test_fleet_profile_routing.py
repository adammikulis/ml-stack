"""Selected Fleet profile launch arguments and private machine token storage."""

import base64
import hashlib
import stat
from types import SimpleNamespace

import pytest

from ml_stack.fleet import daemon, join
from ml_stack.fleet.discovery import derive_token

PRODUCTION = base64.urlsafe_b64encode(bytes(range(32))).rstrip(b"=")
DEVELOPMENT = base64.urlsafe_b64encode(bytes(reversed(range(32)))).rstrip(b"=")


def token_path(root, profile):
    return root / "machine-tokens" / hashlib.sha256(str(profile.resolve()).encode()).hexdigest()


def test_profile_token_preserves_legacy_production_token(tmp_path):
    root = tmp_path / "daemon"
    production = daemon.load_or_create_token(root, PRODUCTION)
    profile = tmp_path / "development.key"
    development = daemon.load_or_create_token(root, DEVELOPMENT, profile=profile)
    assert production == derive_token(PRODUCTION)
    assert development == derive_token(DEVELOPMENT)
    assert development != production
    assert (root / "token").read_text() == production
    assert token_path(root, profile).read_text() == development
    assert stat.S_IMODE((root / "machine-tokens").stat().st_mode) == 0o700
    assert stat.S_IMODE(token_path(root, profile).stat().st_mode) == 0o600


def test_profile_paths_normalize_and_rotate_only_the_selected_token(tmp_path):
    root = tmp_path / "daemon"
    production = daemon.load_or_create_token(root, PRODUCTION)
    profile = tmp_path / "profiles" / "development.key"
    alias = tmp_path / "profiles" / "unused" / ".." / "development.key"
    first = daemon.load_or_create_token(root, DEVELOPMENT, profile=profile)
    assert daemon.load_or_create_token(root, DEVELOPMENT, profile=alias) == first
    assert len(list((root / "machine-tokens").iterdir())) == 1
    rotated = daemon.load_or_create_token(root, PRODUCTION, profile=alias)
    assert rotated != first
    assert token_path(root, profile).read_text() == rotated
    assert (root / "token").read_text() == production


def test_profile_local_token_reload_restores_private_permissions(tmp_path):
    root = tmp_path / "daemon"
    profile = tmp_path / "development.key"
    token = daemon.load_or_create_token(root, profile=profile)
    token_path(root, profile).chmod(0o644)
    (root / "machine-tokens").chmod(0o755)
    assert daemon.load_or_create_token(root, profile=profile) == token
    assert stat.S_IMODE(token_path(root, profile).stat().st_mode) == 0o600
    assert stat.S_IMODE((root / "machine-tokens").stat().st_mode) == 0o700
    assert not (root / "token").exists()


def test_profile_token_refuses_symbolic_link_target(tmp_path):
    root = tmp_path / "daemon"
    profile = tmp_path / "development.key"
    target = tmp_path / "unrelated"
    target.write_text("unrelated state")
    selected = token_path(root, profile)
    selected.parent.mkdir(parents=True)
    selected.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic links"):
        daemon.load_or_create_token(root, DEVELOPMENT, profile=profile)
    assert target.read_text() == "unrelated state"


def test_maintained_launcher_records_selected_profile_and_mode(tmp_path, monkeypatch):
    captured = []
    def detach(module, argv, *, log):
        captured.append((module, argv, log))
        return SimpleNamespace(pid=42, command=[module, *argv], log=log, started=1)
    monkeypatch.setattr(join, "detach", detach)
    profile = tmp_path / "development.key"
    assert join.start_daemon(9123, tmp_path / "daemon", "device", cluster_key_path=profile, mode="dev") == 42
    assert captured[0][0] == "ml_stack.cli.daemon"
    assert captured[0][1] == ["--port", "9123", "--root", str(tmp_path / "daemon"),
                              "--name", "device", "--cluster-key", str(profile.resolve()), "--mode", "dev"]
    record = join.read_json(join.started_file(tmp_path / "daemon"), {})
    assert record["argv"] == [captured[0][0], *captured[0][1]]


def test_default_join_launcher_receives_effective_selected_profile(tmp_path, monkeypatch):
    launched = []
    responses = iter([None, {"name": "device", "machine": "device-id"}])
    monkeypatch.setattr(join, "already_running", lambda _port: next(responses))
    monkeypatch.setattr(join, "checks", lambda *args, **kwargs: [])
    monkeypatch.setattr(join, "peers", lambda **kwargs: [])
    monkeypatch.setattr(join, "wait_for_health", lambda *args, **kwargs: {})
    monkeypatch.setattr(join.automatic_clusters, "ensure", lambda *args, **kwargs:
                        SimpleNamespace(group="development", mode="dev"))
    monkeypatch.setattr(join, "start_daemon", lambda *args, **kwargs: launched.append((args, kwargs)) or 42)
    profile = tmp_path / "development.key"
    joined = join.join_machine(root=tmp_path / "daemon", port=9123, name="device",
                               cluster_key_path=profile, say=lambda _text: None)
    assert joined.started and joined.mode == "dev"
    assert launched == [((9123, tmp_path / "daemon", "device"),
                         {"cluster_key_path": profile, "mode": "dev"})]


def test_injected_launcher_keeps_three_argument_contract(tmp_path):
    called = []
    joined = join.Joined(name="device", port=9123, root=tmp_path, group="development", mode="dev")
    profile = tmp_path / "development.key"
    def injected(port, root, name):
        called.append((port, root, name))
        return 42
    assert join._start_selected(injected, joined, profile) == 42
    assert called == [(9123, tmp_path, "device")]
