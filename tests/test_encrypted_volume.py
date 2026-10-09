"""`scripts/encrypted-volume.sh`: an encrypted image for a directory, driven against fakes of
`hdiutil`, `security`, `mount` and `rsync` that write what they were asked to a log. No image
is made and nothing touches the keychain."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "encrypted-volume.sh"

FAKES = {
    "security": r'''#!/bin/sh
echo "security $*" >> "$FAKE_LOG"
case "$1" in
  find-generic-password) [ -f "$FAKE_HOME/keychain" ] && cat "$FAKE_HOME/keychain" || exit 44 ;;
  add-generic-password) while [ $# -gt 1 ]; do [ "$1" = -w ] && printf '%s\n' "$2" > "$FAKE_HOME/keychain"; shift; done ;;
  -i) while read -r verb rest; do echo "security-stdin $verb $rest" >> "$FAKE_LOG"
        pw=$(printf '%s' "$rest" | sed -n 's/.* -w "\([^"]*\)".*/\1/p'); printf '%s\n' "$pw" > "$FAKE_HOME/keychain"; done ;;
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
    pw = (fakes / "keychain").read_text().strip()
    assert len(pw) == 40 and pw.isalnum()
    assert 'security-stdin add-generic-password -a "nobody" -s "kilnbook-data" -w "' in said
    assert not [line for line in said.splitlines() if line.startswith("security ") and pw in line], \
        "the passphrase never appears on the argv of `security`"
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


def test_a_name_with_a_quote_or_space_is_refused_before_security_runs(fakes):
    for name in ("bad name", 'bad"name'):
        done = volume(fakes, name, str(fakes / "m"), "setup")
        assert done.returncode == 2 and "unsupported character" in done.stderr
    assert "security" not in log(fakes)


def test_a_keychain_path_starting_with_a_dash_is_refused(fakes):
    env = {**os.environ, "PATH": f"{fakes / 'bin'}:{os.environ['PATH']}", "FAKE_LOG": str(fakes / "log"),
           "FAKE_HOME": str(fakes), "USER": "nobody", "ML_STACK_VOLUME_KEYCHAIN": "-evil"}
    done = subprocess.run(["sh", str(SCRIPT), "kilnbook", str(fakes / "m"), "setup"], capture_output=True, text=True, env=env)
    assert done.returncode == 2 and "must not start with" in done.stderr and "security" not in log(fakes)


def test_an_empty_passphrase_from_the_keychain_stops_before_hdiutil(fakes):
    (fakes / "keychain").write_text("")
    done = volume(fakes, "kilnbook", str(fakes / "m"), "setup")
    assert done.returncode == 1 and "no passphrase found" in done.stderr and "hdiutil" not in log(fakes)


@pytest.mark.skipif(sys.platform != "darwin", reason="the `security` CLI and keychains are macOS")
def test_a_real_throwaway_keychain_gets_the_passphrase_on_stdin_not_argv(tmp_path):
    """The real `security` against a keychain file made here (never the login keychain), with a
    shim that records argv and then runs the real tool; hdiutil is faked so no image is made."""
    real = shutil.which("security")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "security"
    shim.write_text(f'#!/bin/sh\necho "security $*" >> "{tmp_path}/argv"\nexec {real} "$@"\n')
    shim.chmod(0o755)
    for name in ("hdiutil", "mount"):
        (bin_dir / name).write_text(FAKES[name])
        (bin_dir / name).chmod(0o755)
    keychain = tmp_path / "throwaway.keychain-db"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_LOG": str(tmp_path / "log"),
           "FAKE_HOME": str(tmp_path), "USER": "mlstacktest", "ML_STACK_VOLUME_KEYCHAIN": str(keychain)}
    subprocess.run([real, "create-keychain", "-p", "throwaway", str(keychain)], check=True, capture_output=True)
    try:
        done = subprocess.run(["sh", str(SCRIPT), "tmpvol", str(tmp_path / "m"), "setup"], env=env,
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        got = subprocess.run([real, "find-generic-password", "-a", "mlstacktest", "-s", "tmpvol-data", "-w",
                              str(keychain)], capture_output=True, text=True).stdout.strip()
        assert len(got) == 40 and got.isalnum()
        argv = (tmp_path / "argv").read_text()
        assert "add-generic-password" not in argv and got not in argv
    finally:
        subprocess.run([real, "delete-keychain", str(keychain)], capture_output=True)
