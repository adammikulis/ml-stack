"""`ml-stack-cluster nearby | listen | pair | requests | accept | decline | revoke | bootstrap` as
separate processes on loopback with throwaway state: the owner's terminal and the new machine
are different processes sharing nothing but a TCP port."""

import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ml_stack.fleet import discovery
from tests.cluster_support import join as cluster_join

FP1 = "1" * 64

@pytest.fixture(autouse=True)
def needs_spake2():
    pytest.importorskip("spake2")



@pytest.fixture(autouse=True)
def needs_cryptography():
    pytest.importorskip("cryptography")


def env_for(root):
    root.mkdir(exist_ok=True)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "ML_STACK_HOME": str(root),
            "PYTHON_KEYRING_BACKEND": "onboard_support.FileKeyring",
            "ML_STACK_TEST_KEYRING": str(root / "keyring.json"), "DISPLAY": ":0",
            "ML_STACK_CLUSTER_KEY": str(root / "cluster.key"), "PYTHONUNBUFFERED": "1"}


def fleet(env, *args, stdin=None, timeout=60):
    return subprocess.run([sys.executable, "-m", "ml_stack.fleet.join", *args], env=env,
                          capture_output=True, text=True, timeout=timeout, input=stdin,
                          check=False)


def spawn(env, *args):
    return subprocess.Popen([sys.executable, "-m", "ml_stack.fleet.join", *args], env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def stop(proc):
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream:
            stream.close()


def free_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def owner(tmp_path):
    """A machine with a cluster, listening for pairing on loopback."""
    env = env_for(tmp_path / "owner")
    key_path = tmp_path / "owner" / "cluster.key"
    cluster_join("a-long-enough-passphrase", path=key_path)
    state = tmp_path / "owner" / "state"
    proc = spawn(env, "listen", "--json", "--state", str(state), "--host", "127.0.0.1",
                 "--port", "0", "--no-announce", "--for", "120s")
    started = json.loads(read_document(proc))
    yield type("Owner", (), {"env": env, "state": state, "port": started["port"],
                             "doc": started, "key_path": key_path})
    stop(proc)


def read_document(proc):
    """One pretty-printed JSON document from a process's output."""
    text = ""
    while True:
        line = proc.stdout.readline()
        if not line:
            raise AssertionError(f"process ended: {proc.stderr.read()}")
        text += line
        try:
            json.loads(text)
            return text
        except ValueError:
            continue


def test_the_whole_conversation_across_three_processes(owner, tmp_path):
    new_env = env_for(tmp_path / "newbox")
    new_state = tmp_path / "newbox" / "state"
    pair = spawn(new_env, "pair", "--host", "127.0.0.1", "--port", str(owner.port), "--json",
                 "--state", str(new_state), "--wait", "60s", "--name", "new-box")
    try:
        # the owner's terminal: wait for the request to show up, then accept it
        seen = []
        for _ in range(100):
            out = fleet(owner.env, "requests", "--json", "--state", str(owner.state))
            seen = json.loads(out.stdout)["requests"]
            if seen:
                break
            pair.poll()
        assert seen and seen[0]["name"] == "new-box" and "code" not in seen[0]
        accepted = fleet(owner.env, "accept", seen[0]["id"][:8], "--mine", "--json",
                         "--state", str(owner.state))
        assert accepted.returncode == 0, accepted.stderr
        doc = json.loads(accepted.stdout)
        # the new machine's person types the code
        pair.stdin.write(doc["code"] + "\n")
        pair.stdin.flush()
        out, err = pair.communicate(timeout=60)
        assert pair.returncode == 0, err
        result = json.loads(out)
        assert result["paired"] and result["joined_cluster"]
    finally:
        pair.kill()
    # the new machine now holds the cluster key the owner holds
    mine = discovery.memberships(tmp_path / "newbox" / "cluster.key")
    theirs = discovery.memberships(owner.key_path)
    assert mine and mine[0].key == theirs[0].key
    # and the owner can see it and revoke it
    devices = json.loads((owner.state / "devices.json").read_text())["devices"]
    assert devices[0]["name"] == "new-box" and devices[0]["status"] == "active"
    revoked = fleet(owner.env, "revoke", "new-box", "--json", "--state", str(owner.state))
    assert json.loads(revoked.stdout)["cluster_key_rotation_needed"] is True
    again = fleet(new_env, "pair", "--host", "127.0.0.1", "--port", str(owner.port), "--json",
                  "--state", str(new_state), "--wait", "2s")
    assert again.returncode == 2 and "revoked" in json.loads(again.stdout)["error"]


def test_a_wrong_code_on_the_command_line_says_how_many_tries_are_left(owner, tmp_path):
    new_env = env_for(tmp_path / "newbox")
    pair = spawn(new_env, "pair", "--host", "127.0.0.1", "--port", str(owner.port), "--json",
                 "--state", str(tmp_path / "newbox" / "state"), "--wait", "60s",
                 "--code", "000000")
    try:
        for _ in range(100):
            seen = json.loads(fleet(owner.env, "requests", "--json", "--state",
                                    str(owner.state)).stdout)["requests"]
            if seen:
                break
        code = json.loads(fleet(owner.env, "accept", seen[0]["id"][:8], "--mine", "--json", "--state",
                                str(owner.state)).stdout)["code"]
        out, _ = pair.communicate(timeout=60)
        doc = json.loads(out)
        if code == "000000":
            pytest.skip("the random code was the guess")
        assert pair.returncode == 2 and doc["tries_left"] == 2
    finally:
        pair.kill()
    assert not (tmp_path / "newbox" / "cluster.json").exists()


def test_decline_and_an_unknown_request_are_answered_in_json(owner, tmp_path):
    new_env = env_for(tmp_path / "newbox")
    pair = spawn(new_env, "pair", "--host", "127.0.0.1", "--port", str(owner.port), "--json",
                 "--state", str(tmp_path / "newbox" / "state"), "--wait", "60s")
    try:
        for _ in range(100):
            seen = json.loads(fleet(owner.env, "requests", "--json", "--state",
                                    str(owner.state)).stdout)["requests"]
            if seen:
                break
        gone = fleet(owner.env, "decline", seen[0]["id"][:8], "--json", "--state",
                     str(owner.state))
        assert json.loads(gone.stdout)["state"] == "declined"
        out, _ = pair.communicate(timeout=60)
        assert json.loads(out)["state"] == "declined" and pair.returncode == 1
    finally:
        pair.kill()
    nope = fleet(owner.env, "accept", "ffffffff", "--mine", "--json", "--state", str(owner.state))
    assert nope.returncode == 2 and "error" in json.loads(nope.stdout)


def test_nearby_hears_a_listener_that_announces_to_loopback(tmp_path):
    env = env_for(tmp_path / "owner")
    port = free_udp_port()
    listener = spawn(env, "listen", "--json", "--state", str(tmp_path / "s"), "--host",
                     "127.0.0.1", "--port", "0", "--announce-to", f"127.0.0.1:{port}",
                     "--for", "60s")
    try:
        started = json.loads(read_document(listener))
        out = fleet(env, "nearby", "--json", "--port", str(port), "--bind", "127.0.0.1",
                    "--timeout", "8")
        found = json.loads(out.stdout)["nearby"]
        assert out.returncode == 0 and found[0]["port"] == started["port"]
        assert found[0]["fingerprint"] == started["fingerprint"]
    finally:
        stop(listener)


def test_nearby_with_nobody_there_exits_nonzero(tmp_path):
    env = env_for(tmp_path / "x")
    out = fleet(env, "nearby", "--json", "--port", str(free_udp_port()), "--bind", "127.0.0.1",
                "--timeout", "0.5")
    assert out.returncode == 1 and json.loads(out.stdout) == {"nearby": []}


def test_revoke_with_no_such_device_and_ssh_without_files_say_so(tmp_path):
    env = env_for(tmp_path / "x")
    out = fleet(env, "revoke", "nobody", "--json", "--state", str(tmp_path / "x" / "s"))
    assert out.returncode == 2 and "error" in json.loads(out.stdout)
    ssh = fleet(env, "bootstrap", "--ssh", "pi@10.0.0.9", "--json", "--state",
                str(tmp_path / "x" / "s"))
    assert ssh.returncode == 2 and "--share" in json.loads(ssh.stdout)["error"]


def test_bootstrap_prints_an_offer_with_a_pinned_certificate(tmp_path):
    env = env_for(tmp_path / "x")
    share = tmp_path / "share"
    share.mkdir()
    (share / "ml_stack-0.2-py3-none-any.whl").write_bytes(b"wheel" * 100)
    proc = spawn(env, "bootstrap", "--share", str(share), "--valid", "60s", "--json",
                 "--state", str(tmp_path / "x" / "s"))
    try:
        doc = json.loads(read_document(proc))
        assert doc["url"].startswith("https://127.0.0.1:") and doc["url"].endswith(
            f"#fp={doc['certificate']}")
        assert doc["files"] == ["ml_stack-0.2-py3-none-any.whl"]
        assert "--pinnedpubkey" in doc["command"]
        state = tmp_path / "x" / "s"
        assert (state / "signing.json").stat().st_mode & 0o077 == 0
        assert not (state / "signing.key").exists() and not (state / "signing.key.enc").exists()
        assert json.loads((state / "signing.json").read_text())["store"] == "keystore"
    finally:
        stop(proc)
    refused = fleet(env, "bootstrap", "--json", "--state", str(tmp_path / "x" / "s"))
    assert refused.returncode == 2


def pair_up(owner, tmp_path):
    """The new machine pairs with the owner; returns its env and state directory."""
    env = env_for(tmp_path / "newbox")
    state = tmp_path / "newbox" / "state"
    pair = spawn(env, "pair", "--host", "127.0.0.1", "--port", str(owner.port), "--json",
                 "--state", str(state), "--wait", "60s", "--name", "new-box")
    try:
        for _ in range(100):
            seen = json.loads(fleet(owner.env, "requests", "--json", "--state",
                                    str(owner.state)).stdout)["requests"]
            if seen:
                break
        code = json.loads(fleet(owner.env, "accept", seen[0]["id"][:8], "--mine", "--json", "--state",
                                str(owner.state)).stdout)["code"]
        pair.stdin.write(code + "\n")
        pair.stdin.flush()
        _, err = pair.communicate(timeout=60)
        assert pair.returncode == 0, err
    finally:
        pair.kill()
    return env, state


def test_after_pairing_the_new_machine_fetches_files_from_the_owner(owner, tmp_path):
    env, state = pair_up(owner, tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    wheel = os.urandom(300_000)
    (shared / "ml_stack-0.2-py3-none-any.whl").write_bytes(wheel)
    (shared / "gated.gguf").write_bytes(b"GGUF" * 1000)
    sharing = spawn(owner.env, "share", "--port", "0", "--dir", str(shared),
                    "--sharing", "gated.gguf=never", "--licence", "gated.gguf=nocopy,https://x/l",
                    "--host", "127.0.0.1", "--json", "--state", str(owner.state), "--for", "120s")
    try:
        started = json.loads(read_document(sharing))
        assert {f["name"]: f["sharing"] for f in started["files"]} == {
            "ml_stack-0.2-py3-none-any.whl": "open", "gated.gguf": "never"}
        source = f"127.0.0.1:{started['port']}"
        got = fleet(env, "fetch", "ml_stack-0.2-py3-none-any.whl", "--from", source, "--json",
                    "--state", str(state))
        assert got.returncode == 0, got.stdout + got.stderr
        staged = json.loads(got.stdout)["staged"]
        assert Path(staged[0]).read_bytes() == wheel
        assert Path(staged[0]).parent == state / "staging"
        refused = fleet(env, "fetch", "gated.gguf", "--from", source, "--json", "--state",
                        str(state))
        assert refused.returncode == 2 and "may not be shared" in json.loads(refused.stdout)["error"]
        # a machine that never paired has no pinned key and no cluster, and is turned away
        stranger = env_for(tmp_path / "stranger")
        nope = fleet(stranger, "fetch", "ml_stack-0.2-py3-none-any.whl", "--from", source,
                     "--json", "--state", str(tmp_path / "stranger" / "state"))
        assert nope.returncode == 2
    finally:
        stop(sharing)


def test_an_older_manifest_than_one_already_seen_is_refused(owner, tmp_path):
    env, state = pair_up(owner, tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "a-0.1-py3-none-any.whl").write_bytes(b"w" * 70_000)
    sharing = spawn(owner.env, "share", "--port", "0", "--dir", str(shared), "--host", "127.0.0.1", "--json",
                    "--state", str(owner.state), "--for", "120s")
    try:
        source = f"127.0.0.1:{json.loads(read_document(sharing))['port']}"
        trust = json.loads((state / "trust.json").read_text())
        (state / "trust.json").write_text(json.dumps({**trust, "serial": 2_000_000_000_000}))
        old = fleet(env, "fetch", "a-0.1-py3-none-any.whl", "--from", source, "--json",
                    "--state", str(state))
        assert old.returncode == 2 and "older" in json.loads(old.stdout)["error"]
    finally:
        stop(sharing)


def test_share_answers_only_requests_signed_with_the_cluster_secret(owner, tmp_path):
    import http.client

    from ml_stack import macauth
    from ml_stack.fleet.onboard.pairing import unverified_context
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "a-0.1-py3-none-any.whl").write_bytes(b"w" * 70_000)
    sharing = spawn(owner.env, "share", "--port", "0", "--dir", str(shared), "--host", "127.0.0.1", "--json",
                    "--state", str(owner.state), "--for", "120s")
    try:
        port = json.loads(read_document(sharing))["port"]
        url = f"https://127.0.0.1:{port}/onboard/v1/manifest"
        key = discovery.memberships(owner.key_path)[0].key
        answers = []
        for headers in ({}, macauth.sign(macauth.derive(b"somebody else"), "GET", url, None),
                        macauth.sign(macauth.derive(key), "GET", url, None)):
            conn = http.client.HTTPSConnection("127.0.0.1", port, context=unverified_context(),
                                               timeout=10)
            conn.request("GET", "/onboard/v1/manifest", headers=headers)
            answers.append(conn.getresponse().status)
        assert answers == [401, 401, 200]
    finally:
        stop(sharing)


@pytest.mark.skipif(platform.system() != "Darwin", reason="the fake osascript answers a macOS dialog")
def test_a_click_on_accept_in_the_dialog_accepts_the_request_and_the_code_comes_in_a_second_dialog(
        tmp_path):
    """The owner's side with a fake `osascript` first on PATH: it answers the first dialog as a
    click on 'Accept as mine' and records the second, which carries the code."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "dialogs.txt"
    fake = bin_dir / "osascript"
    fake.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" >> "$FAKE_DIALOGS"\n'
        'case "$2" in *"set answer"*) echo "button returned:Accept as mine, gave up:false";; esac\n')
    fake.chmod(0o755)
    env = {**env_for(tmp_path / "owner"), "ML_STACK_NOTIFY": "system", "FAKE_DIALOGS": str(record),
           "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    key_path = tmp_path / "owner" / "cluster.key"
    cluster_join("a-long-enough-passphrase", path=key_path)
    state = tmp_path / "owner" / "state"
    listener = spawn(env, "listen", "--json", "--state", str(state), "--host", "127.0.0.1",
                     "--port", "0", "--no-announce", "--for", "120s")
    try:
        started = json.loads(read_document(listener))
        assert started["notifier"] == "macos"
        new_env = env_for(tmp_path / "newbox")
        pair = spawn(new_env, "pair", "--host", "127.0.0.1", "--port", str(started["port"]),
                     "--json", "--state", str(tmp_path / "newbox" / "state"), "--wait", "60s")
        try:
            for _ in range(300):                      # the owner clicks nothing: the dialog did
                if record.exists() and record.read_text().count("-e") >= 2:
                    break
                pair.poll()
                time.sleep(0.1)
            lines = record.read_text().split("\n")
            code_line = next(line for line in lines if line.startswith("Pairing code "))
            code = code_line.removeprefix("Pairing code ").replace(" ", "")
            shown = json.loads(fleet(env, "requests", "--json", "--state", str(state)).stdout)
            assert shown["requests"][0]["state"] == "accepted" and shown["requests"][0]["mine"]
            pair.stdin.write(code + "\n")
            pair.stdin.flush()
            out, err = pair.communicate(timeout=60)
            assert pair.returncode == 0, err
            assert json.loads(out)["paired"]
        finally:
            pair.kill()
        ledger = json.loads((state / "devices.json").read_text())["devices"]
        assert ledger[0]["mine"] is True
    finally:
        stop(listener)
