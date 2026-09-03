"""`scripts/encrypted-volume.sh`: an encrypted image for a directory, driven against fakes of
`hdiutil`, `security`, `mount` and `rsync` that write what they were asked to a log. No image
is made and nothing touches the keychain."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "encrypted-volume.sh"

FAKES = {
    "security": r'''#!/bin/sh
echo "security $*" >> "$FAKE_LOG"
case "$1" in
  find-generic-password) [ -f "$FAKE_HOME/keychain" ] && cat "$FAKE_HOME/keychain" || exit 44 ;;
  add-generic-password) while [ $# -gt 1 ]; do [ "$1" = -w ] && printf '%s\n' "$2" > "$FAKE_HOME/keychain"; shift; done ;;
esac
''',
    "hdiutil": r'''#!/bin/sh
echo "hdiutil $*" >> "$FAKE_LOG"
pw=$(cat)
echo "stdin:[$pw]" >> "$FAKE_LOG"
case "$1" in
  create) for last; do :; done; mkdir -p "$last" ;;
  attach) while [ $# -gt 1 ]; do [ "$1" = -mountpoint ] && echo "/dev/disk9 on $2 (apfs)" >> "$FAKE_HOME/mounts"; shift; done ;;
  detach) : > "$FAKE_HOME/mounts" ;;
esac
''',
    "mount": r'''#!/bin/sh
[ -f "$FAKE_HOME/mounts" ] && cat "$FAKE_HOME/mounts"
exit 0
''',
    "rsync": r'''#!/bin/sh
echo "rsync $*" >> "$FAKE_LOG"
cp -R "$2". "$3"
''',
}


@pytest.fixture
def fakes(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in FAKES.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    return tmp_path


def volume(fakes: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{fakes / 'bin'}:{os.environ['PATH']}",
           "FAKE_LOG": str(fakes / "log"), "FAKE_HOME": str(fakes), "USER": "nobody"}
    return subprocess.run(["sh", str(SCRIPT), *args], capture_output=True, text=True, env=env)


def log(fakes: Path) -> str:
    return (fakes / "log").read_text() if (fakes / "log").exists() else ""


@pytest.mark.slow
def test_setup_makes_a_passphrase_the_image_and_mounts_it(fakes):
    mount = fakes / "store" / "data"
    done = volume(fakes, "kilnbook", str(mount), "setup")
    assert done.returncode == 0, done.stderr
    said = log(fakes)
    assert "security add-generic-password -a nobody -s kilnbook-data -w " in said
    pw = (fakes / "keychain").read_text().strip()
    assert len(pw) == 40 and pw.isalnum()
    assert f"-encryption AES-256 -stdinpass -volname kilnbook-data {mount}.sparsebundle" in said
    assert f"stdin:[{pw}]" in said, "the passphrase goes to hdiutil on stdin, without a newline"
    assert f"hdiutil attach {mount}.sparsebundle -stdinpass -mountpoint {mount} -nobrowse" in said
    assert "kilnbook-data" in done.stdout


@pytest.mark.slow
def test_setup_with_a_source_moves_it_into_the_volume_and_leaves_a_link(fakes):
    mount = fakes / "store" / "data"
    source = fakes / "repo" / "data"
    source.mkdir(parents=True)
    (source / "graph.json").write_text("{}")
    done = volume(fakes, "kilnbook", str(mount), "setup", "--source", str(source))
    assert done.returncode == 0, done.stderr
    assert f"rsync -a {source}/ {mount}/" in log(fakes)
    assert source.is_symlink() and os.readlink(source) == str(mount)
    assert (mount / "graph.json").exists()


@pytest.mark.slow
def test_setup_twice_leaves_the_image_alone(fakes):
    mount = fakes / "store" / "data"
    assert volume(fakes, "kilnbook", str(mount), "setup").returncode == 0
    before = log(fakes)
    done = volume(fakes, "kilnbook", str(mount), "setup")
    assert done.returncode == 0
    assert "already set up" in done.stdout
    assert log(fakes) == before


@pytest.mark.slow
def test_mount_is_quiet_when_the_volume_is_attached_and_unmount_detaches(fakes):
    mount = fakes / "store" / "data"
    assert volume(fakes, "kilnbook", str(mount), "setup").returncode == 0
    before = log(fakes)
    assert volume(fakes, "kilnbook", str(mount), "mount").returncode == 0
    assert log(fakes) == before, "already attached: nothing asked of hdiutil"
    assert volume(fakes, "kilnbook", str(mount), "unmount").returncode == 0
    assert f"hdiutil detach {mount} -quiet" in log(fakes)
    assert volume(fakes, "kilnbook", str(mount), "mount").returncode == 0
    assert log(fakes).count("hdiutil attach") == 2


@pytest.mark.slow
def test_a_wrong_call_prints_the_usage(fakes):
    for args in ((), ("kilnbook",), ("kilnbook", "/x", "explode"), ("kilnbook", "/x", "mount", "--what")):
        done = volume(fakes, *args)
        assert done.returncode == 2, args
        assert "usage:" in done.stderr
