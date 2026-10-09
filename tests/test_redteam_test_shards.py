"""Hostile test-shard requests against a real stand-in device: nothing but a named test file may run.

Each refusal is checked on the wire (the real status over pinned TLS with signed, sealed requests)
and on the device (no claim, no scratch checkout, and the marker a shipped test would write is absent).
"""

from __future__ import annotations

import hashlib
import io
import os
import secrets
import struct
import tarfile
import time

import pytest
from shard_support import NAME

from ml_stack.fleet import shard_client, shard_tree
from ml_stack.fleet.remote import Peer, PeerError
from ml_stack.fleet.shard_spec import MOST_FILES, frame

pytest_plugins = ("test_fleet_test_shards",)
pytestmark = pytest.mark.redteam


def archive(members: list[tuple[str, bytes, bytes]]) -> bytes:
    """A gzip tar of ``(name, type, content)`` members; type is ``0`` file, ``2`` symlink, ``5`` directory."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name, kind, content in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(content) if kind == tarfile.REGTYPE else 0
            info.linkname = "/etc/passwd" if kind == tarfile.SYMTYPE else ""
            tar.addfile(info, io.BytesIO(content) if kind == tarfile.REGTYPE else None)
    return out.getvalue()


def header(tree: bytes, files: list[str], **more) -> dict:
    return {"id": secrets.token_hex(16), "tree_sha256": hashlib.sha256(tree).hexdigest(), "files": files,
            "timeout_s": 60, **more}


def post(peer: Peer, body: bytes) -> None:
    peer._request("POST", shard_client.ROUTE, data=body, headers={"Content-Type": "application/octet-stream"})


def refused(peer: Peer, body: bytes, status: int) -> None:
    with pytest.raises(PeerError) as caught:
        post(peer, body)
    assert caught.value.status == status, caught.value


def untouched(one) -> None:
    """The device took nothing: no claim, no scratch tree, nothing running."""
    time.sleep(0.2)
    assert list(one.host.folder.iterdir()) == [] and not one.host._running


@pytest.fixture
def device(standin):
    one = standin()
    return one, shard_client.device_peer(NAME, host="127.0.0.1", port=one.port)


GOOD = archive([("tests/test_ok.py", tarfile.REGTYPE, b"def test_a():\n    assert 1\n")])


@pytest.mark.parametrize("name", [
    "../outside.py", "tests/../../outside.py", "/etc/passwd", "scripts/test", "tests/sub/test_ok.py",
    "tests/test_ok.py::test_a", "tests\\test_ok.py", "tests/test_ok.py\0", "tests/.hidden.py", "tests/", "",
    "tests/test_ok.txt", "C:/tests/test_ok.py", "tests/test ok.py", "-k", "--rootdir=/", "tests/test_ok.py;ls"])
def test_a_path_outside_the_plain_test_files_is_refused(device, name):
    one, peer = device
    refused(peer, frame(header(GOOD, [name]), GOOD), 400)
    untouched(one)


@pytest.mark.parametrize("files", [[], None, "tests/test_ok.py", [1], [["tests/test_ok.py"]],
                                   ["tests/test_ok.py", "tests/test_ok.py"],
                                   [f"tests/t{n}.py" for n in range(MOST_FILES + 1)]])
def test_a_file_list_that_is_not_a_short_list_of_distinct_names_is_refused(device, files):
    one, peer = device
    refused(peer, frame(header(GOOD, files), GOOD), 400)
    untouched(one)


@pytest.mark.parametrize("field", ["argv", "env", "cwd", "command", "python", "pytest_args", "shell"])
def test_extra_fields_are_refused_whatever_they_say(device, field):
    one, peer = device
    refused(peer, frame(header(GOOD, ["tests/test_ok.py"], **{field: ["touch", "x"]}), GOOD), 400)
    untouched(one)


@pytest.mark.parametrize("seconds", [0, -1, 3601, "60", 1.5, True, None])
def test_a_time_limit_outside_the_allowed_range_is_refused(device, seconds):
    one, peer = device
    refused(peer, frame({**header(GOOD, ["tests/test_ok.py"]), "timeout_s": seconds}, GOOD), 400)
    untouched(one)


@pytest.mark.parametrize("bad", ["id", "tree_sha256"])
@pytest.mark.parametrize("value", ["", "A" * 32, "../" + "a" * 29, "g" * 64, None, 5])
def test_a_malformed_id_or_digest_is_refused(device, bad, value):
    one, peer = device
    refused(peer, frame({**header(GOOD, ["tests/test_ok.py"]), bad: value}, GOOD), 400)
    untouched(one)


def test_a_file_that_is_not_in_the_shipped_tree_is_refused(device):
    one, peer = device
    refused(peer, frame(header(GOOD, ["tests/test_missing.py"]), GOOD), 400)
    untouched(one)


def test_a_tree_that_does_not_match_its_digest_is_refused(device):
    one, peer = device
    other = archive([("tests/test_ok.py", tarfile.REGTYPE, b"import os\nos._exit(0)\n")])
    refused(peer, frame(header(GOOD, ["tests/test_ok.py"]), other), 400)
    refused(peer, frame(header(GOOD, ["tests/test_ok.py"]), GOOD[:-1]), 400)
    untouched(one)


@pytest.mark.parametrize("members", [
    [("tests/test_ok.py", tarfile.SYMTYPE, b"")],
    [("tests/test_ok.py", tarfile.REGTYPE, b"x"), ("../evil.py", tarfile.REGTYPE, b"x")],
    [("tests/test_ok.py", tarfile.REGTYPE, b"x"), ("/tmp/evil.py", tarfile.REGTYPE, b"x")],
    [("tests/test_ok.py", tarfile.REGTYPE, b"x"), (".git/hooks/pre-commit", tarfile.REGTYPE, b"x")],
    [("tests/test_ok.py", tarfile.REGTYPE, b"x"), ("tests", tarfile.DIRTYPE, b"")],
    [("tests/test_ok.py", tarfile.REGTYPE, b"x"), ("tests/..\\..\\evil.py", tarfile.REGTYPE, b"x")],
    [("tests/test_ok.py", tarfile.REGTYPE, b"x"), ("nowhere/evil.py", tarfile.REGTYPE, b"x")]])
def test_a_tree_with_anything_but_plain_shipped_files_is_refused(device, members):
    one, peer = device
    tree = archive(members)
    refused(peer, frame(header(tree, ["tests/test_ok.py"]), tree), 400)
    untouched(one)


def test_a_tree_that_inflates_past_the_limit_is_refused(device):
    one, peer = device
    bomb = archive([("tests/test_ok.py", tarfile.REGTYPE, b"\0" * (shard_tree.MOST_UNPACKED + 1))])
    assert len(bomb) < shard_tree.MOST_PACKED
    refused(peer, frame(header(bomb, ["tests/test_ok.py"]), bomb), 400)
    untouched(one)


def test_an_oversized_or_malformed_body_is_refused(device):
    one, peer = device
    big = os.urandom(shard_tree.MOST_PACKED + 1)
    refused(peer, frame(header(big, ["tests/test_ok.py"]), big), 400)
    refused(peer, b"", 400)
    refused(peer, b"\0\0", 400)
    refused(peer, struct.pack(">I", 1 << 30) + b"{}", 400)
    refused(peer, struct.pack(">I", 2) + b"[]" + GOOD, 400)
    refused(peer, struct.pack(">I", 3) + b"{x}" + GOOD, 400)
    refused(peer, struct.pack(">I", 2) + b"{}", 400)
    untouched(one)


def test_a_replayed_shard_id_is_refused_and_runs_once(standin, tree):
    one = standin()
    peer = shard_client.device_peer(NAME, host="127.0.0.1", port=one.port)
    packed, digest = shard_tree.pack(tree)
    body = frame({"id": secrets.token_hex(16), "tree_sha256": digest, "files": ["tests/test_ok.py"],
                  "timeout_s": 60}, packed)
    post(peer, body)
    refused(peer, body, 409)


def test_a_shard_past_the_running_cap_is_refused(standin, tree):
    one = standin()
    peer = shard_client.device_peer(NAME, host="127.0.0.1", port=one.port)
    sent = [shard_client.send(peer, tree, ["tests/test_sleep.py"], 20) for _ in range(2)]
    packed, digest = shard_tree.pack(tree)
    refused(peer, frame({"id": secrets.token_hex(16), "tree_sha256": digest,
                         "files": ["tests/test_sleep.py"], "timeout_s": 20}, packed), 429)
    assert len(sent) == 2


def test_a_device_with_shards_off_runs_nothing_and_says_so(standin, tree):
    one = standin(enabled=False)
    peer = shard_client.device_peer(NAME, host="127.0.0.1", port=one.port)
    with pytest.raises(PeerError) as caught:
        shard_client.send(peer, tree, ["tests/test_ok.py"], 60)
    assert caught.value.status == 403 and "off" in str(caught.value)
    untouched(one)


def test_a_paired_device_that_is_not_marked_as_the_owners_is_refused(standin, tree):
    one = standin(sender="other")
    peer = shard_client.device_peer(NAME, host="127.0.0.1", port=one.port)
    with pytest.raises(PeerError) as caught:
        shard_client.send(peer, tree, ["tests/test_ok.py"], 60)
    assert caught.value.status == 403
    untouched(one)


def test_a_peer_that_is_not_paired_is_refused_before_anything_is_read(standin, tree):
    one = standin(sender="none")
    peer = shard_client.device_peer(NAME, host="127.0.0.1", port=one.port)
    for call in (lambda: shard_client.capability(peer), lambda: shard_client.send(peer, tree, ["tests/test_ok.py"], 60)):
        with pytest.raises(PeerError) as caught:
            call()
        assert caught.value.status in {401, 403}
    untouched(one)


def test_the_cluster_secret_is_not_a_paired_device(standin, tree):
    one = standin()
    stranger = Peer(f"https://127.0.0.1:{one.port}", "cluster-token", timeout=30)
    shard_client.device_peer(NAME, host="127.0.0.1", port=one.port)
    for call in (lambda: shard_client.capability(stranger), lambda: shard_client.send(stranger, tree, ["tests/test_ok.py"], 60)):
        with pytest.raises(PeerError) as caught:
            call()
        assert caught.value.status in {400, 401, 403}
    untouched(one)


def test_an_unknown_or_hostile_shard_id_has_no_result(device):
    one, peer = device
    for name in ("a" * 32, "..%2f..%2fetc", "../" * 4, "A" * 32):
        with pytest.raises(PeerError) as caught:
            peer._json("GET", f"{shard_client.ROUTE}/{name}")
        assert caught.value.status in {400, 404}
    untouched(one)
