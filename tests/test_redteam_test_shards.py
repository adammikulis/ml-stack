"""Hostile test-shard requests against a real device: nothing but a named test file may run, and only the pool's members ask.

Two node processes on loopback, paired. Each refusal is checked where it happens (the answer the asker gets) and
on the device afterwards: no folder for the refused shard, no executor, and the marker a shipped test would
write is absent. What reaches the executor (the contents of the tree) is checked by the executor itself, so a
hostile tree is a finished shard with exit 70, never a run.
"""

from __future__ import annotations

import hashlib
import io
import secrets
import tarfile

import pytest
from testfarm_kit import enable, jobs_on, peer, until

from ml_stack import node_health
from ml_stack.board import session as board_session
from ml_stack.board.client import NodeError
from ml_stack.testfarm.client import ShardError, Shards

pytest_plugins = ("testfarm_kit",)
pytestmark = pytest.mark.redteam

GOOD_ID = "ab" * 16


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


def start_request(data: bytes, **more) -> dict:
    """The arguments of a well-formed `shard_start` for ``data``, with ``more`` changed."""
    return {"id": secrets.token_hex(16), "tree_sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "tier": "all",
            "files": ["tests/test_a.py"], "timeout_s": 60, **more}


def uploaded(shards: Shards, fp: str, data: bytes, **more) -> dict:
    """Upload ``data`` as a new shard and return the arguments to start it with ``more`` changed."""
    request = start_request(data, **more)
    for offset in range(0, len(data), 128 << 10):
        shards.call(fp, "shard_put", id=request["id"], offset=offset, data=data[offset:offset + (128 << 10)].hex())
    return request


def refused(shards: Shards, fp: str, op: str, match: str = "", **args) -> None:
    with pytest.raises(ShardError, match=match):
        shards.call(fp, op, **args)


def finished_with_70(shards: Shards, fp: str, request: dict) -> dict:
    """Start the shard and wait for it: a hostile tree ends as a refusal by the executor, not as a run."""
    shards.call(fp, "shard_start", **request)
    result = shards.wait(fp, request["id"], timeout_s=60, poll_s=0.2)
    assert result["exit"] == 70 and result["state"] == "failed" and "stub ran" not in result["output_tail"]
    return result


@pytest.mark.parametrize("name", ["scripts/test", "tests/../scripts/test.py", "tests/sub/test_a.py", "/etc/passwd", "tests\\test_a.py",
                                  "tests/.hidden.py", "tests/test_a.py::test_one", "--rootdir=/", "tests/test a.py", "", "tests/"])
def test_a_path_outside_the_plain_test_files_is_refused(pool, name):
    _a, b, shards, _tree = pool
    enable(b)
    fp = peer(_a)["fingerprint"]
    request = uploaded(shards, fp, b"x" * 64, files=[name])
    refused(shards, fp, "shard_start", **request)
    assert shards.call(fp, "shard_status", id=request["id"])["state"] == "uploading", "it never started"
    assert jobs_on(b) == 1 and not list((b.state / "shards" / request["id"]).glob("spec.json"))


@pytest.mark.parametrize("files", [[f"tests/test_{n}.py" for n in range(201)], ["tests/test_a.py"] * 2, "tests/test_a.py", None,
                                   [["tests/test_a.py"]], [1], {"tests/test_a.py": 1}])
def test_a_file_list_that_is_not_a_short_list_of_distinct_names_is_refused(pool, files):
    a, b, shards, _tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    request = uploaded(shards, fp, b"x" * 64, files=files)
    refused(shards, fp, "shard_start", **request)
    assert shards.call(fp, "shard_status", id=request["id"])["state"] == "uploading"


def test_a_file_that_is_not_in_the_shipped_tree_is_refused(pool):
    a, b, shards, _tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    data = archive([("tests/test_a.py", tarfile.REGTYPE, b"x = 1\n")])
    result = finished_with_70(shards, fp, uploaded(shards, fp, data, files=["tests/test_missing.py"]))
    assert "is not in the shipped tree" in result["output_tail"]


@pytest.mark.parametrize(("bad", "value"), [("id", "short"), ("id", "AB" * 16), ("id", "../" * 10 + "ab"), ("id", "g" * 32), ("id", 7),
                                            ("tree_sha256", "0" * 63), ("tree_sha256", "G" * 64), ("tree_sha256", "A" * 64), ("tree_sha256", None)])
def test_a_malformed_id_or_digest_is_refused(pool, bad, value):
    a, b, shards, _tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    request = uploaded(shards, fp, b"x" * 64)
    refused(shards, fp, "shard_start", **{**request, bad: value})
    assert shards.call(fp, "shard_status", id=request["id"])["state"] == "uploading", "the real shard is untouched"


@pytest.mark.parametrize("field", ["argv", "env", "cwd", "command", "shell", "python", "repo", "requester", "owner", "mine", "extra"])
def test_extra_fields_are_refused_whatever_they_say(pool, field):
    a, b, shards, _tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    request = uploaded(shards, fp, b"x" * 64, **{field: ["rm", "-rf", "/"] if field == "argv" else "1"})
    refused(shards, fp, "shard_start", **request)
    refused(shards, fp, "shard_put", id=secrets.token_hex(16), offset=0, data="00", **{field: "1"})
    refused(shards, fp, "shard_status", id=request["id"], **{field: "1"})
    assert jobs_on(b) == 1, "only the one upload exists"


@pytest.mark.parametrize("seconds", [0, -1, 3601, 10**9, True, "60", 1.5, None])
def test_a_time_limit_outside_the_allowed_range_is_refused(pool, seconds):
    a, b, shards, _tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    request = uploaded(shards, fp, b"x" * 64, timeout_s=seconds)
    refused(shards, fp, "shard_start", **request)
    assert shards.call(fp, "shard_status", id=request["id"])["state"] == "uploading"


def test_a_tree_that_does_not_match_its_digest_is_refused(pool):
    a, b, shards, _tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    request = uploaded(shards, fp, b"x" * 64, tree_sha256=hashlib.sha256(b"something else").hexdigest())
    refused(shards, fp, "shard_start", "does not match its digest", **request)
    refused(shards, fp, "shard_status", "no such shard", id=request["id"])
    assert jobs_on(b) == 0, "the mismatched upload was removed from the disk"


@pytest.mark.parametrize("members", [
    [("tests/../../escape.py", tarfile.REGTYPE, b"x = 1\n")],
    [("/tmp/escape.py", tarfile.REGTYPE, b"x = 1\n")],
    [("tests/link.py", tarfile.SYMTYPE, b"")],
    [("tests/", tarfile.DIRTYPE, b"")],
    [(".git/hooks/pre-commit", tarfile.REGTYPE, b"#!/bin/sh\ntouch /tmp/pwned\n")],
    [("C:/escape.py", tarfile.REGTYPE, b"x = 1\n")],
])
def test_a_tree_with_anything_but_plain_shipped_files_is_refused(pool, members):
    a, b, shards, _tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    data = archive([("tests/test_a.py", tarfile.REGTYPE, b"x\n"), *members])
    result = finished_with_70(shards, fp, uploaded(shards, fp, data))
    assert "stub ran" not in result["output_tail"]
    assert not list(b.root.rglob("escape.py")) and not (b.root.parent / "escape.py").exists()


def test_a_tree_that_inflates_past_the_limit_is_refused(pool):
    a, b, shards, _tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    bomb = archive([("tests/test_a.py", tarfile.REGTYPE, b"\0" * (300 << 20))])
    assert len(bomb) < 1 << 20
    result = finished_with_70(shards, fp, uploaded(shards, fp, bomb))
    assert "inflates" in result["output_tail"]


def test_an_oversized_or_malformed_body_is_refused(pool):
    a, b, shards, _tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    for bad in ("xyz", "0", "", "zz" * 8, "00" * (256 * 1024 + 1)):
        refused(shards, fp, "shard_put", id=GOOD_ID, offset=0, data=bad)
    assert jobs_on(b) == 0, "none of those opened an upload"
    shards.call(fp, "shard_put", id=GOOD_ID, offset=0, data="00" * 256 * 1024)
    refused(shards, fp, "shard_put", id=GOOD_ID, offset=5, data="00")
    refused(shards, fp, "shard_put", id=GOOD_ID, offset=0, data="00")
    request = start_request(b"\0" * 10, id=GOOD_ID, size=(24 << 20) + 1)
    refused(shards, fp, "shard_start", **request)
    refused(shards, fp, "shard_start", **{**request, "size": 9})


def test_a_replayed_shard_id_is_refused_and_runs_once(pool):
    a, b, shards, tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    shard = shards.send(fp, tree, "all", ["tests/test_a.py"], 60)
    assert shards.wait(fp, shard, timeout_s=60, poll_s=0.2)["exit"] == 0
    refused(shards, fp, "shard_put", id=shard, offset=0, data="00")
    refused(shards, fp, "shard_start", **start_request(b"x" * 64, id=shard))
    assert jobs_on(b) == 1 and shards.call(fp, "shard_status", id=shard)["state"] == "done"


def test_a_shard_past_the_running_cap_is_refused(pool):
    a, b, shards, tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    held = [shards.send(fp, tree, "all", ["tests/test_hang.py"], 600) for _ in range(2)]
    until("both to run", 30, lambda: shards.capability(fp)["free"] == 0)
    with pytest.raises(ShardError, match="2 shards are already running"):
        shards.send(fp, tree, "all", ["tests/test_a.py"], 60)
    for shard in held:
        shards.cancel(fp, shard)
    until("the slots to come back", 30, lambda: shards.capability(fp)["free"] == 2)


def test_a_device_with_shards_off_runs_nothing_and_says_so(pool):
    a, b, shards, tree = pool
    fp = peer(a)["fingerprint"]
    refused(shards, fp, "shard_put", "off on this device", id=GOOD_ID, offset=0, data="00")
    refused(shards, fp, "shard_start", "off on this device", **start_request(b"x"))
    with pytest.raises(ShardError, match="off on this device"):
        shards.send(fp, tree, "all", ["tests/test_a.py"], 60)
    assert shards.capability(fp)["accepts"] is False and jobs_on(b) == 0


def test_a_paired_device_that_is_not_marked_as_the_owners_is_refused(pool, third):
    """Being the owner's device is standing in the pool: a device put out of it asks nothing, however it got in."""
    a, b, _shards, _tree = pool
    enable(b)
    code = b.call("pair_accept", token=b.token)
    third.call("pair_start", token=third.token, host="127.0.0.1", port=code["port"], passphrase=code["code"])
    ours, fp_b = Shards(seat(third)), peer(a)["fingerprint"]
    assert ours.capability(fp_b)["accepts"] is True, "a member is answered"
    b.call("member_revoke", token=b.token, fingerprint=node_health.call(third.state, "pool_status")["fingerprint"])
    with pytest.raises(ShardError):
        ours.capability(fp_b)
    with pytest.raises(ShardError):
        ours.call(fp_b, "shard_put", id=GOOD_ID, offset=0, data="00")
    assert jobs_on(b) == 0


def test_a_peer_that_is_not_paired_is_refused_before_anything_is_read(pool, third):
    a, b, _shards, _tree = pool
    enable(b)
    stranger = Shards(seat(third))
    with pytest.raises(ShardError, match="no other active device"):
        stranger.call(peer(a)["fingerprint"], "shard_put", id=GOOD_ID, offset=0, data="00")
    assert jobs_on(b) == 0


def test_the_cluster_secret_is_not_a_paired_device(pool, third):
    """A session token of the pool, held by a device that is not in it, opens nothing: tokens belong to one node."""
    a, b, _shards, _tree = pool
    enable(b)
    with pytest.raises(NodeError):
        third.call("shard_call", token=a.token, device=peer(a)["fingerprint"], op="shard_caps", args={})
    with pytest.raises(NodeError):
        third.call("shard_consent", token=a.token, enabled=True, python="/usr/bin/python3", repo="/")
    assert jobs_on(b) == 0


@pytest.mark.parametrize("shard_id", ["../../etc/passwd", "ab" * 16, "AB" * 16, "", "x", "ab" * 32, "..", "ab/cd"])
def test_an_unknown_or_hostile_shard_id_has_no_result(pool, shard_id):
    a, b, shards, _tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    refused(shards, fp, "shard_status", id=shard_id)
    refused(shards, fp, "shard_cancel", id=shard_id)
    assert jobs_on(b) == 0


def seat(device):
    """A device's own registered session."""
    return board_session.Session(device.client, "demo", device.token)
