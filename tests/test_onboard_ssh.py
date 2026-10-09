"""Installing over SSH: the target is checked character by character, the argv is exact and has
no shell, the host key is never taken on trust, and the remote script refuses anything that does
not verify. Fake ssh/ssh-keyscan on PATH record argv; the remote script runs for real, locally,
against a temporary HOME; a localhost sshd test is opt-in."""

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import keyring
import pytest
from onboard_support import FileKeyring, Recorder

from poolhouse.fleet.onboard import manifest as mf, ssh
from poolhouse.fleet.onboard.signing import SigningKeys
from poolhouse.safenames import Unsafe

WHEEL = b"PK-not-really-a-wheel" * 400
FINGERPRINT = "SHA256:" + "A" * 43


@pytest.fixture(autouse=True)
def needs_cryptography():
    pytest.importorskip("cryptography")


@pytest.fixture
def keystore(tmp_path, monkeypatch):
    monkeypatch.setenv("POOLHOUSE_TEST_KEYRING", str(tmp_path / "keystore.json"))
    before = keyring.get_keyring()
    keyring.set_keyring(FileKeyring())
    yield
    keyring.set_keyring(before)


@pytest.fixture
def share(tmp_path):
    d = tmp_path / "share"
    d.mkdir()
    (d / "poolhouse-0.2-py3-none-any.whl").write_bytes(WHEEL)
    return d


def signed_for(tmp_path, share):
    keys = SigningKeys(tmp_path / "state", bus=Recorder().bus)
    made = keys.attest(ssh.program_entries(share), serial=1, namespace=ssh.NAMESPACE)
    return keys, ssh.Signed(made["manifest"], made["sshsig"], made["allowed_signers"],
                            made["key_id"])


# -- the target ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,user,host", [
    ("pi@kitchen.local", "pi", "kitchen.local"), ("192.168.1.20", "", "192.168.1.20"),
    ("me_1@host-2", "me_1", "host-2"), ("fe80::1", "", "fe80::1"), ("a", "", "a")])
def test_plain_targets_are_accepted(text, user, host):
    t = ssh.parse_target(text, 2222)
    assert (t.user, t.host, t.port) == (user, host, 2222)


@pytest.mark.parametrize("bad", [
    "-oProxyCommand=touch /tmp/x", "-o ProxyCommand=x", "-host", "user@-host", "host;rm -rf /",
    "host && id", "host rm", "ho st", " host", "host ", "host\n", "host\nid", "$(id)",
    "`id`", "a|b", "a>b", "a'b", 'a"b', "a\\b", "user@host@other", "", "@host", "u ser@host",
    "host:22", "[::1]", "host/../x", "host\x00", "--", "-", "*", "~root@host", "host#x"])
def test_anything_that_is_not_user_at_host_is_refused(bad):
    with pytest.raises(ssh.SshRefused):
        ssh.parse_target(bad)


@pytest.mark.parametrize("port", [0, -1, 70000])
def test_a_port_outside_the_range_is_refused(port):
    with pytest.raises(ssh.SshRefused):
        ssh.parse_target("host", port)


# -- fake ssh tools on PATH -----------------------------------------------------------------
SHIM = """#!/bin/sh
{{ printf '%s\\n' "$0" "$@"; echo "---"; }} >> "$FAKE_LOG"
case "$(basename "$0")" in
  ssh-keygen)
    case "$1" in
      -F) exit "${{FAKE_KNOWN:-1}}";;
      -l) echo "256 {fp} host (ED25519)"; exit 0;;
    esac;;
  ssh-keyscan) echo "kitchen.local ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFAKEKEYFAKEKEYFAKEKEYFAKEKEYFAKEKEYFAKEKEYFAKE";;
  ssh)
    for last; do :; done
    case "$last" in
      *"tar -xf"*) cat > "$FAKE_PAYLOAD";;
      *) cat "$FAKE_REPLY";;
    esac;;
esac
"""


class Tools:
    def __init__(self, tmp_path, monkeypatch, *, known):
        self.dir = tmp_path / "fakebin"
        self.dir.mkdir()
        self.log, self.payload = tmp_path / "calls.log", tmp_path / "payload.tar"
        self.reply = tmp_path / "reply.txt"
        self.reply.write_text(json.dumps({"step": "verified"}) + "\n" + json.dumps(
            {"step": "listening", "port": 8772, "fingerprint": "ff" * 32, "ok": True}) + "\n")
        for name in ("ssh", "ssh-keygen", "ssh-keyscan"):
            f = self.dir / name
            f.write_text(SHIM.format(fp=FINGERPRINT))
            f.chmod(0o755)
        for key, value in (("FAKE_LOG", self.log), ("FAKE_PAYLOAD", self.payload),
                           ("FAKE_REPLY", self.reply), ("FAKE_KNOWN", "0" if known else "1")):
            monkeypatch.setenv(key, str(value))
        monkeypatch.setenv("PATH", f"{self.dir}{os.pathsep}{os.environ['PATH']}")

    def calls(self):
        text = self.log.read_text() if self.log.exists() else ""
        return [block.split("\n")[:-1] if block.endswith("\n") else block.split("\n")
                for block in text.split("---\n") if block.strip()]

    def ssh_calls(self):
        return [c[1:] for c in self.calls() if c[0].endswith("/ssh")]


def run_ssh(tmp_path, share, tools, typed, target="pi@kitchen.local"):
    _, signed = signed_for(tmp_path, share)
    rec = Recorder()
    done = ssh.bootstrap_over_ssh(ssh.parse_target(target), share, signed, lambda prints: typed,
                                  ssh.Runtime(timeout=30, bus=rec.bus))
    return done, rec


def test_a_known_host_gets_exactly_two_ssh_calls_with_the_fixed_argv(
        tmp_path, monkeypatch, keystore, share):
    tools = Tools(tmp_path, monkeypatch, known=True)
    done, rec = run_ssh(tmp_path, share, tools, typed="")
    unpack, run = tools.ssh_calls()
    t = ssh.parse_target("pi@kitchen.local")
    assert unpack == ssh.ssh_argv(t, ssh.REMOTE_UNPACK)[1:]
    assert run == ssh.ssh_argv(t, ssh.REMOTE_RUN)[1:]
    assert unpack[-3:] == ["--", "pi@kitchen.local", ssh.REMOTE_UNPACK]
    assert "BatchMode=yes" in unpack and "StrictHostKeyChecking=yes" in unpack
    assert not any("keyscan" in c[0] for c in tools.calls())      # a known host is not scanned
    assert done["listen_port"] == 8772 and done["fingerprint"] == "ff" * 32
    assert [e.kind for e in rec.events][:2] == ["onboard.ssh.started", "onboard.ssh.copied"]


def test_the_payload_is_plain_files_with_fixed_modes_and_the_script_that_is_printed(
        tmp_path, monkeypatch, keystore, share):
    tools = Tools(tmp_path, monkeypatch, known=True)
    run_ssh(tmp_path, share, tools, typed="")
    with tarfile.open(tools.payload) as tar:
        members = {m.name: m for m in tar.getmembers()}
        assert set(members) == {"manifest.json", "manifest.sig", "allowed_signers", "KEYID",
                                "install_remote.py", "poolhouse-0.2-py3-none-any.whl"}
        assert all(m.isreg() and m.uid == 0 and m.mtime == 0 for m in members.values())
        assert members["install_remote.py"].mode == 0o700 and members["KEYID"].mode == 0o600
        script = tar.extractfile("install_remote.py").read()
    assert script == ssh.REMOTE_SCRIPT.read_bytes()


def test_an_unknown_host_needs_its_fingerprint_typed_in_full_and_is_then_pinned_for_this_run(
        tmp_path, monkeypatch, keystore, share):
    tools = Tools(tmp_path, monkeypatch, known=False)
    seen = {}
    real_pin = ssh.host_key_pin

    def watch(target, run, offered, bus):
        extra, pin = real_pin(target, run, offered, bus)
        seen["extra"], seen["pin"] = extra, pin
        seen["content"] = pin.read_text() if pin else ""
        return extra, pin
    monkeypatch.setattr(ssh, "host_key_pin", watch)
    run_ssh(tmp_path, share, tools, typed=FINGERPRINT)
    unpack, _ = tools.ssh_calls()
    assert f"UserKnownHostsFile={seen['pin']}" in unpack and "GlobalKnownHostsFile=/dev/null" in unpack
    assert seen["content"].startswith("kitchen.local ssh-ed25519 AAAAC3")
    assert not seen["pin"].exists()                         # gone afterwards
    scan = next(c for c in tools.calls() if c[0].endswith("ssh-keyscan"))
    assert scan[1:4] == ["-T", "10", "-p"]


@pytest.mark.parametrize("typed", ["", "SHA256:" + "B" * 43, FINGERPRINT[:-1], "yes", FINGERPRINT + " "])
def test_a_fingerprint_that_is_not_the_one_presented_runs_nothing(
        tmp_path, monkeypatch, keystore, share, typed):
    tools = Tools(tmp_path, monkeypatch, known=False)
    with pytest.raises(ssh.SshRefused, match="nothing was run"):
        run_ssh(tmp_path, share, tools, typed=typed)
    assert tools.ssh_calls() == []


def test_no_argv_ever_turns_checking_off_or_asks_for_a_password(
        tmp_path, monkeypatch, keystore, share):
    for known in (True, False):
        tools = Tools(tmp_path, monkeypatch, known=known)
        run_ssh(tmp_path, share, tools, typed=FINGERPRINT)
        flat = " ".join(" ".join(c) for c in tools.calls())
        assert "StrictHostKeyChecking=no" not in flat and "accept-new" not in flat
        assert "PasswordAuthentication=yes" not in flat and "sshpass" not in flat
        assert all("BatchMode=yes" in c for c in tools.ssh_calls())
        shutil.rmtree(tools.dir)
        tools.log.unlink(missing_ok=True)


def test_a_remote_that_refuses_is_a_clean_failure_with_the_reason(
        tmp_path, monkeypatch, keystore, share):
    tools = Tools(tmp_path, monkeypatch, known=True)
    tools.reply.write_text(json.dumps({"ok": False, "refused": "the manifest has expired"}) + "\n")
    monkeypatch.setenv("FAKE_REPLY", str(tools.reply))
    with pytest.raises(ssh.SshRefused, match="the manifest has expired"):
        run_ssh(tmp_path, share, tools, typed="")


def test_output_is_bounded_and_a_hung_command_times_out(tmp_path):
    big = [sys.executable, "-c", "import sys; sys.stdout.write('x' * 500000)"]
    assert len(ssh.real_run(big, None, 30)[1]) == ssh.MOST_OUTPUT
    code, _, err = ssh.real_run([sys.executable, "-c", "import time; time.sleep(30)"], None, 0.5)
    assert code == 124 and b"timed out" in err
    assert ssh.real_run(["/no/such/tool"], None, 5)[0] == 127


def test_a_missing_openssh_client_is_a_clear_refusal(tmp_path, monkeypatch, keystore, share):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ssh.SshRefused, match="OpenSSH"):
        run_ssh(tmp_path, share, None, typed="")


# -- the command line ------------------------------------------------------------------------
def fleet(env, *args):
    return subprocess.run([sys.executable, "-m", "poolhouse.fleet.join", *args], env=env,
                          capture_output=True, text=True, timeout=120, check=False)


def cli_env(tmp_path, tools_dir=None):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path),
           "POOLHOUSE_HOME": str(tmp_path / "home"),
           "PYTHON_KEYRING_BACKEND": "onboard_support.FileKeyring",
           "POOLHOUSE_TEST_KEYRING": str(tmp_path / "keystore.json")}
    if tools_dir:
        env["PATH"] = f"{tools_dir}{os.pathsep}{env['PATH']}"
    return env


def test_the_dry_run_prints_the_script_in_full_with_its_hash_and_runs_nothing(
        tmp_path, monkeypatch, share):
    tools = Tools(tmp_path, monkeypatch, known=True)
    env = {**cli_env(tmp_path, tools.dir), "FAKE_LOG": str(tools.log)}
    out = fleet(env, "bootstrap", "--ssh", "pi@kitchen.local", "--share", str(share),
                "--dry-run", "--json", "--state", str(tmp_path / "state"))
    doc = json.loads(out.stdout)
    assert out.returncode == 0 and doc["runs_nothing"] is True
    assert doc["script"] == ssh.REMOTE_SCRIPT.read_text()
    import hashlib
    assert doc["script_sha256"] == hashlib.sha256(doc["script"].encode()).hexdigest()
    assert doc["commands"][0][-1] == ssh.REMOTE_UNPACK and doc["commands"][1][-1] == ssh.REMOTE_RUN
    assert doc["files"][0]["name"] == "poolhouse-0.2-py3-none-any.whl"
    assert tools.calls() == []                                # nothing contacted
    assert not (tmp_path / "state" / "signing.json").exists()    # and no key was made or used
    assert 80 < len(doc["script"].splitlines()) < 160


@pytest.mark.parametrize("bad", ["-oProxyCommand=touch /tmp/pwned", "host;touch /tmp/pwned",
                                 "-host", "host && id"])
def test_a_hostile_target_from_the_command_line_reaches_no_process(
        tmp_path, monkeypatch, share, bad):
    tools = Tools(tmp_path, monkeypatch, known=True)
    env = {**cli_env(tmp_path, tools.dir), "FAKE_LOG": str(tools.log)}
    out = fleet(env, "bootstrap", f"--ssh={bad}", "--share", str(share), "--json",
                "--state", str(tmp_path / "state"))
    assert out.returncode == 2 and "error" in json.loads(out.stdout)
    assert tools.calls() == [] and not Path("/tmp/pwned").exists()


# -- the remote script, run for real in a sandbox HOME ----------------------------------------
def unpack(tmp_path, tar_bytes):
    inbox = tmp_path / "home" / ".poolhouse-bootstrap" / "inbox"
    inbox.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        tar.extractall(inbox, filter="data")
    return inbox


def remote(tmp_path, inbox, *flags):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["HOME"] = str(tmp_path / "home")
    return subprocess.run([sys.executable, str(inbox / "install_remote.py"), *flags], env=env,
                          capture_output=True, text=True, timeout=300, check=False)


@pytest.fixture
def inbox(tmp_path, keystore, share):
    _, signed = signed_for(tmp_path, share)
    return unpack(tmp_path, ssh.build_payload(share, signed))


def test_a_good_payload_verifies_on_the_remote_side(tmp_path, inbox):
    done = remote(tmp_path, inbox, "--verify-only")
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout.splitlines()[-1])["step"] == "verified"


def refused(done, needle):
    assert done.returncode == 2, done.stdout + done.stderr
    reasons = [json.loads(line) for line in done.stdout.splitlines()]
    assert needle in reasons[-1]["refused"], reasons


def test_a_tampered_wheel_is_refused_before_anything_is_installed(tmp_path, inbox):
    wheel = inbox / "poolhouse-0.2-py3-none-any.whl"
    wheel.write_bytes(wheel.read_bytes()[:-1] + b"X")
    refused(remote(tmp_path, inbox, "--verify-only"), "does not match the signed manifest")
    assert not (tmp_path / "home" / ".poolhouse" / "venv").exists()


def test_a_wheel_of_the_wrong_size_or_a_swapped_wheel_is_refused(tmp_path, inbox):
    wheel = inbox / "poolhouse-0.2-py3-none-any.whl"
    wheel.write_bytes(wheel.read_bytes() + b"more")
    refused(remote(tmp_path, inbox, "--verify-only"), "does not match")


def test_an_edited_manifest_fails_the_signature(tmp_path, inbox):
    manifest = inbox / "manifest.json"
    doc = json.loads(manifest.read_text())
    doc["manifest"]["entries"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(doc, sort_keys=True))
    refused(remote(tmp_path, inbox, "--verify-only"), "signature does not verify")


def test_a_signature_by_another_key_is_refused_even_with_its_own_allowed_signers(
        tmp_path, inbox):
    thief = mf.Signer.generate()
    (inbox / "allowed_signers").write_text(thief.ssh_public_line() + "\n")
    refused(remote(tmp_path, inbox, "--verify-only"), "not the one the owner named")
    (inbox / "KEYID").write_text(thief.key_id + "\n")
    refused(remote(tmp_path, inbox, "--verify-only"), "signature does not verify")


def test_a_key_id_that_is_not_a_key_id_is_refused(tmp_path, inbox):
    (inbox / "KEYID").write_text("not-an-id\n")
    refused(remote(tmp_path, inbox, "--verify-only"), "KEYID")


def test_a_file_the_manifest_does_not_list_is_refused(tmp_path, inbox):
    (inbox / "extra.sh").write_text("echo hi")
    refused(remote(tmp_path, inbox, "--verify-only"), "exactly the files")


def test_an_expired_manifest_is_refused(tmp_path, keystore, share):
    signer = mf.Signer.generate()
    raw = signer.sign(ssh.program_entries(share), serial=1, now=time.time() - 10 * 86400)
    signed = ssh.Signed(raw, signer.sshsig(raw, ssh.NAMESPACE), signer.ssh_public_line() + "\n",
                        signer.key_id)
    refused(remote(tmp_path, unpack(tmp_path, ssh.build_payload(share, signed)), "--verify-only"),
            "expired")


def make_wheel(path: Path) -> None:
    dist = "poolhousefake-0.1.dist-info"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("poolhousefake.py", "VALUE = 1\n")
        z.writestr(f"{dist}/METADATA", "Metadata-Version: 2.1\nName: poolhousefake\nVersion: 0.1\n")
        z.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: true\n"
                                    "Tag: py3-none-any\n")
        z.writestr(f"{dist}/RECORD", "")


@pytest.mark.slow
def test_a_verified_wheel_is_installed_into_a_per_user_venv_with_no_sudo(
        tmp_path, keystore):
    share = tmp_path / "share"
    share.mkdir()
    make_wheel(share / "poolhousefake-0.1-py3-none-any.whl")
    _, signed = signed_for(tmp_path, share)
    inbox = unpack(tmp_path, ssh.build_payload(share, signed))
    done = remote(tmp_path, inbox, "--no-listen")
    assert done.returncode == 0, done.stdout + done.stderr
    venv = tmp_path / "home" / ".poolhouse" / "venv"
    out = subprocess.run([str(venv / "bin" / "python"), "-c", "import poolhousefake; "
                          "print(poolhousefake.VALUE)"], capture_output=True, text=True, check=False)
    assert out.stdout.strip() == "1"
    assert [json.loads(line)["step"] for line in done.stdout.splitlines()] == [
        "verified", "installed"]


@pytest.mark.skipif(not os.environ.get("POOLHOUSE_TEST_SSHD"),
                    reason="set POOLHOUSE_TEST_SSHD=1 with a localhost sshd and a key that logs "
                           "in without a prompt, to run the real thing")
@pytest.mark.slow
def test_against_a_real_localhost_sshd(tmp_path, keystore, share):
    _, signed = signed_for(tmp_path, share)
    target = ssh.parse_target(f"{os.environ.get('USER', '')}@localhost")
    done = ssh.bootstrap_over_ssh(target, share, signed,
                                  lambda prints: os.environ.get("POOLHOUSE_TEST_HOSTKEY", ""))
    assert done["listen_port"]


def test_a_manifest_naming_a_path_cannot_make_the_payload_read_outside_the_share(
        tmp_path, keystore, share):
    signer = mf.Signer.generate()
    row = ssh.program_entries(share)[0]
    evil = mf.Entry("../secret.whl", row.size, row.sha256, row.chunk_size, row.chunks, "wheel")
    raw = signer.sign([evil], serial=1)
    signed = ssh.Signed(raw, signer.sshsig(raw, ssh.NAMESPACE), signer.ssh_public_line() + "\n",
                        signer.key_id)
    with pytest.raises(Unsafe):
        ssh.build_payload(share, signed)


@pytest.mark.parametrize("kind,name", [("model", "big.gguf"), ("other", "notes.txt")])
def test_the_remote_script_installs_only_wheels_and_archives_even_when_the_signature_is_good(
        tmp_path, keystore, share, kind, name):
    (share / name).write_bytes(b"data" * 10)
    signer = mf.Signer.generate()
    raw = signer.sign([signer.entry(share / name, kind=kind)], serial=1)
    signed = ssh.Signed(raw, signer.sshsig(raw, ssh.NAMESPACE), signer.ssh_public_line() + "\n",
                        signer.key_id)
    refused(remote(tmp_path, unpack(tmp_path, ssh.build_payload(share, signed)), "--verify-only"),
            "not a program file")


def test_the_remote_script_refuses_a_name_that_is_not_plain_even_when_signed(
        tmp_path, keystore, share):
    (share / "has space.whl").write_bytes(b"data" * 10)
    signer = mf.Signer.generate()
    raw = signer.sign([signer.entry(share / "has space.whl", kind="wheel")], serial=1)
    signed = ssh.Signed(raw, signer.sshsig(raw, ssh.NAMESPACE), signer.ssh_public_line() + "\n",
                        signer.key_id)
    refused(remote(tmp_path, unpack(tmp_path, ssh.build_payload(share, signed)), "--verify-only"),
            "not a program file")
