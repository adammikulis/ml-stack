"""scripts/release-key: a key made, stored, rotated and loaded, against real ssh-keygen, a fake
keyring and a fake `gh` that records what it was given."""

from __future__ import annotations

import base64
import importlib.machinery
import importlib.util
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from poolhouse import credentials
from poolhouse.fleet import signing
from tests import memory_keys

ring = memory_keys.ring

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "release-key"
TTY = (True, True)
SIGNING_TEXT = 'NAMESPACE = "x"\nRELEASE_KEY = ""\nOTHER = 1\n'


@pytest.fixture
def rk(tmp_path, monkeypatch):
    for name in ("CLAUDECODE", "POOLHOUSE_AGENT", "POOLHOUSE_NONINTERACTIVE", "WIDGET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POOLHOUSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    tempfile.tempdir = None
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text('#!/bin/sh\nn=$(ls "$GH_LOG" | grep -c "^stdin")\n'
                  'cat > "$GH_LOG/stdin.$n"\necho "$@" > "$GH_LOG/argv.$n"\n'
                  '[ -z "$GH_FAIL" ] || { echo "gh: not logged in" >&2; exit 1; }\n')
    gh.chmod(0o755)
    log = tmp_path / "ghlog"
    log.mkdir()
    monkeypatch.setenv("GH_LOG", str(log))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    signing_py = tmp_path / "signing.py"
    signing_py.write_text(SIGNING_TEXT)
    loader = importlib.machinery.SourceFileLoader("release_key", str(SCRIPT))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("release_key", loader))
    loader.exec_module(module)
    monkeypatch.setattr(module, "SIGNING", signing_py)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    module.log, module.signing_py = log, signing_py
    yield module
    tempfile.tempdir = None


def _stdin(rk, n=0) -> bytes:
    return (rk.log / f"stdin.{n}").read_bytes()


def _stored(rk) -> bytes:
    return base64.b64decode(str(credentials.get(rk.ITEM)))


def _leftovers(tmp_path) -> list[str]:
    return [p.name for p in (tmp_path / "tmp").iterdir()]


def test_create_stores_the_key_and_sends_it_to_gh_on_stdin(rk, tmp_path, capsys):
    assert rk.main(["create"], TTY) == 0
    private = _stored(rk)
    assert private.startswith(b"-----BEGIN OPENSSH PRIVATE KEY-----")
    assert _stdin(rk) == private
    argv = (rk.log / "argv.0").read_text()
    assert argv.split() == ["secret", "set", "RELEASE_SIGNING_KEY", "--env", "release"]
    assert "PRIVATE KEY" not in argv
    out = capsys.readouterr()
    assert "PRIVATE KEY" not in out.out + out.err
    assert "ssh-ed25519 " in out.out
    assert rk.signing_py.read_text() == SIGNING_TEXT
    assert "RELEASE_KEY = " in out.out
    assert _leftovers(tmp_path) == []


def test_the_public_line_verifies_a_signature_made_with_the_stored_key(rk, tmp_path, capsys):
    rk.main(["create"], TTY)
    capsys.readouterr()
    assert rk.main(["show-public"], TTY) == 0
    public = capsys.readouterr().out.strip()
    key = tmp_path / "k"
    key.write_bytes(_stored(rk))
    key.chmod(0o600)
    asset = tmp_path / "asset.zip"
    asset.write_bytes(b"bundle")
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(key), "-n", signing.NAMESPACE, str(asset)],
                   check=True, capture_output=True)
    signature = (tmp_path / "asset.zip.sig").read_text()
    signing.verify_file(asset, signature, key=public)
    other = tmp_path / "other.zip"
    other.write_bytes(b"changed")
    with pytest.raises(signing.SignatureError):
        signing.verify_file(other, signature, key=public)


def test_write_sets_release_key_and_nothing_else(rk, capsys):
    assert rk.main(["create", "--write"], TTY) == 0
    public = capsys.readouterr().out.split("public key: ")[1].split("\n")[0]
    assert rk.signing_py.read_text() == (
        f'NAMESPACE = "x"\nRELEASE_KEY = "{public}"\nOTHER = 1\n')
    assert public.count(" ") == 1


def test_a_second_create_refuses_and_changes_nothing(rk):
    rk.main(["create"], TTY)
    before = _stored(rk)
    assert rk.main(["create"], TTY) == 1
    assert _stored(rk) == before
    assert not (rk.log / "stdin.1").exists()


def test_rotate_replaces_the_item_the_secret_and_the_line(rk, capsys):
    rk.main(["create", "--write"], TTY)
    first = _stored(rk)
    capsys.readouterr()
    assert rk.main(["rotate", "--write"], TTY) == 0
    second = _stored(rk)
    assert second != first
    assert _stdin(rk, 1) == second
    public = capsys.readouterr().out.split("public key: ")[1].split("\n")[0]
    assert f'RELEASE_KEY = "{public}"' in rk.signing_py.read_text()
    assert rk._public(first) not in rk.signing_py.read_text()


def test_rotate_says_earlier_releases_verify_against_the_old_key(rk, capsys):
    rk.main(["create"], TTY)
    capsys.readouterr()
    rk.main(["rotate"], TTY)
    assert "old public key" in capsys.readouterr().out


def test_gh_failing_leaves_the_keystore_and_signing_untouched(rk, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GH_FAIL", "1")
    assert rk.main(["create", "--write"], TTY) == 1
    assert credentials.get(rk.ITEM) is None
    assert rk.signing_py.read_text() == SIGNING_TEXT
    err = capsys.readouterr().err
    assert "not logged in" in err
    assert "unchanged" in err
    assert _leftovers(tmp_path) == []


def test_a_missing_gh_is_named(rk, monkeypatch, capsys, tmp_path):
    # Only ssh-keygen is on the path: the directory it lives in (/usr/bin on a CI runner) holds gh too.
    only = tmp_path / "only-ssh-keygen"
    only.mkdir()
    keygen = subprocess.run(["which", "ssh-keygen"], capture_output=True, text=True).stdout.strip()
    (only / "ssh-keygen").symlink_to(keygen)
    monkeypatch.setenv("PATH", str(only))
    assert rk.main(["create"], TTY) == 1
    assert "gh is not installed" in capsys.readouterr().err
    assert credentials.get(rk.ITEM) is None


@pytest.mark.parametrize("marker", ["CLAUDECODE", "POOLHOUSE_NONINTERACTIVE"])
def test_an_agent_marker_is_refused_and_nothing_runs(rk, monkeypatch, marker):
    monkeypatch.setenv(marker, "1")
    for command in (["create"], ["rotate"], ["show-public"], ["agent"]):
        assert rk.main(command, TTY) == 1
    assert credentials.get(rk.ITEM) is None
    assert list(rk.log.iterdir()) == []


def test_no_terminal_is_refused(rk):
    assert rk.main(["create"], (False, True)) == 1
    assert list(rk.log.iterdir()) == []


def test_show_public_with_no_key_says_so(rk, capsys):
    assert rk.main(["show-public"], TTY) == 1
    assert "no release key is stored" in capsys.readouterr().err


def test_agent_loads_the_key_and_prints_the_git_lines(rk, monkeypatch, tmp_path, capsys):
    rk.main(["create"], TTY)
    capsys.readouterr()
    rk.main(["show-public"], TTY)
    public = capsys.readouterr().out.strip()
    sock = Path(tempfile.mkdtemp(prefix="ag", dir="/tmp")) / "s"
    started = subprocess.run(["ssh-agent", "-a", str(sock)], capture_output=True, text=True, check=True)
    pid = int(started.stdout.split("SSH_AGENT_PID=")[1].split(";")[0])
    try:
        monkeypatch.setenv("SSH_AUTH_SOCK", str(sock))
        assert rk.main(["agent"], TTY) == 0
        out = capsys.readouterr().out
        assert "git config gpg.format ssh" in out
        assert f"git config user.signingkey 'key::{public}'" in out
        listed = subprocess.run(["ssh-add", "-L"], capture_output=True, text=True).stdout
        assert public in listed
        assert "PRIVATE KEY" not in out
    finally:
        os.kill(pid, 15)
        shutil.rmtree(sock.parent, ignore_errors=True)


def test_agent_without_an_ssh_agent_says_so(rk, monkeypatch, capsys):
    rk.main(["create"], TTY)
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    assert rk.main(["agent"], TTY) == 1
    assert "SSH_AUTH_SOCK" in capsys.readouterr().err


def test_the_stored_item_is_not_a_readable_file(rk, tmp_path):
    rk.main(["create"], TTY)
    held = credentials.wrapped_path()
    assert not stat.S_IMODE(held.stat().st_mode) & 0o077
    for path in (tmp_path / "home").rglob("*"):
        if path.is_file():
            assert b"OPENSSH PRIVATE" not in path.read_bytes()
