"""The activity log itself: schema bounds, redaction, encryption at rest, the hash chain and
what detects tampering, rotation and retention, concurrent writers, and failure that never
reaches the caller. Real files, a real keystore over a fake keyring."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ml_stack import keystore
from ml_stack.activity import schema, writer
from ml_stack.activity.log import ActivityLog, Limits, Unreadable
from ml_stack.activity.schema import Entry
from tests.activity_support import CANARY, entries, person, ring
from tests.keystore_support import counting

__all__ = ["counting", "person", "ring"]
KEY = bytes(range(32))
SRC = str(Path(__file__).resolve().parent.parent / "src")
TOKEN = "hf_" + "Zq8Lm2Xv" * 4


def small(tmp_path: Path, **kw) -> ActivityLog:
    return ActivityLog(tmp_path / "log", key=lambda: KEY, **kw)


def line(**kw):
    return schema.build(kw.pop("kind", "agent.tool_call"), ts=kw.pop("ts", 1_800_000_000.0),
                        actor=kw.pop("actor", "agent:a"), session="s1", **kw)


def disk(directory: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file())


# -- schema -------------------------------------------------------------------------------
def test_a_record_is_bounded_whatever_it_is_given():
    huge = {f"k{i}": "x" * 5000 for i in range(200)}
    payload = line(subject="s" * 9000, outcome="o" * 900, refs=huge, meta=huge)
    assert len(json.dumps(payload)) <= schema.MAX_BYTES
    assert len(payload["subject"]) <= schema.MAX_TEXT
    assert payload["truncated"] is True and payload["schema_version"] == schema.VERSION


def test_bodies_are_never_stored_under_any_of_their_names():
    meta = dict.fromkeys(schema.BODY_KEYS, "BODY-TEXT") | {"size": 12, "type": "status"}
    kept = line(meta=meta)["meta"]
    assert kept == {"size": 12, "type": "status"}


def test_a_kind_that_is_not_a_dotted_lowercase_name_is_recorded_as_invalid():
    assert line(kind="Not A Kind")["kind"] == "invalid"
    assert line(kind="../../etc")["kind"] == "invalid"
    assert line(kind="a.b.c.d.e")["kind"] == "invalid"
    assert line(kind="workspace.message")["kind"] == "workspace.message"


def test_secrets_are_removed_from_every_field_before_anything_is_written():
    payload = line(subject=f"fetch with {TOKEN}", outcome="Bearer abcdefgh12345678",
                   refs={"url": f"https://h.example/x?token={TOKEN}"},
                   meta={"password": "hunter2hunter2", "note": f"key {TOKEN}", "api_key": "abcd1234"})
    text = json.dumps(payload)
    assert TOKEN not in text and "hunter2" not in text and "abcdefgh12345678" not in text
    assert payload["meta"]["password"] == "<redacted>" and payload["meta"]["api_key"] == "<redacted>"


def test_control_and_bidi_characters_are_written_as_visible_escapes():
    payload = line(subject="ok\x1b[31m‮evil\n\x00end")
    assert "\x1b" not in payload["subject"] and "‮" not in payload["subject"]
    assert "\\x1b" in payload["subject"] and "\\u202e" in payload["subject"]


def test_control_characters_in_metadata_and_references_are_escaped_where_they_are_written():
    payload = line(refs={"to": "a\x1b[2Jb"}, meta={"note": "x\u202ey\x00", "n": 3})
    assert payload["refs"]["to"] == "a\\x1b[2Jb"
    assert payload["meta"]["note"] == "x\\u202ey\\x00" and payload["meta"]["n"] == 3


# -- encryption at rest ---------------------------------------------------------------------
def test_nothing_a_record_says_is_readable_in_any_file_under_the_directory(person):
    for n in range(5):
        assert writer.record("workspace.message", subject=CANARY, outcome=CANARY, refs={"to": CANARY},
                             meta={"note": CANARY, "n": n})
    on_disk = disk(writer.directory())
    assert on_disk and CANARY.encode() not in on_disk and b"workspace.message" not in on_disk
    assert b"agent" not in on_disk and b"tool_call" not in on_disk
    assert [e.subject for e in entries()] == [CANARY] * 5


def test_a_record_does_not_open_under_another_users_key(tmp_path):
    mine = small(tmp_path)
    mine.add(line(subject="secret-ish"))
    other = ActivityLog(tmp_path / "log", key=lambda: bytes(reversed(KEY)))
    rows = list(other.entries())
    assert len(rows) == 1 and isinstance(rows[0], Unreadable)
    assert other.verify().ok


def test_the_files_are_owner_only(person):
    writer.record("net.download", subject="host:a.example")
    assert writer.directory().stat().st_mode & 0o077 == 0
    for path in writer.directory().rglob("*"):
        if path.is_file() and path.suffix != ".lock":
            assert path.stat().st_mode & 0o077 == 0, path


# -- chain and tamper detection ---------------------------------------------------------------
def filled(tmp_path: Path, n: int = 6) -> ActivityLog:
    log = small(tmp_path)
    for i in range(n):
        log.add(line(subject=f"s{i}", ts=1_800_000_000.0 + i))
    return log


def test_an_untouched_log_verifies_and_reads_back_in_order(tmp_path):
    log = filled(tmp_path)
    done = log.verify()
    assert done.ok and done.records == 6
    assert [e.subject for e in log.entries()] == [f"s{i}" for i in range(6)]
    assert [e.seq for e in log.entries()] == list(range(6))


def rewrite(log: ActivityLog, change) -> None:
    rows = log.path.read_text().splitlines()
    log.path.write_text("\n".join(change(rows)) + "\n")


def test_an_edited_record_is_found_and_named(tmp_path):
    log = filled(tmp_path)

    def edit(rows):
        doc = json.loads(rows[2])
        doc["ct"] = doc["ct"][:-4] + "AAAA"
        return [*rows[:2], json.dumps(doc), *rows[3:]]
    rewrite(log, edit)
    done = log.verify()
    assert not done.ok and any("activity.log:3: edited" in p for p in done.problems)


def test_a_deleted_record_is_found(tmp_path):
    log = filled(tmp_path)
    rewrite(log, lambda rows: [*rows[:2], *rows[3:]])
    done = log.verify()
    assert not done.ok and any("activity.log:3" in p for p in done.problems)


def test_reordered_records_are_found(tmp_path):
    log = filled(tmp_path)
    rewrite(log, lambda rows: [rows[0], rows[2], rows[1], *rows[3:]])
    assert not log.verify().ok


def test_records_cut_off_the_end_are_found(tmp_path):
    log = filled(tmp_path)
    rewrite(log, lambda rows: rows[:-2])
    done = log.verify()
    assert not done.ok and any("cut off" in p for p in done.problems)


def test_a_head_anchor_kept_elsewhere_catches_a_log_rolled_back_with_its_head(tmp_path):
    log = filled(tmp_path)
    anchor = log.head()
    log.add(line(subject="later"))
    assert log.verify(anchor).ok is False


def test_the_verify_command_says_where_the_chain_breaks(person, capsys):
    from ml_stack.activity import cli
    for i in range(4):
        writer.record("net.download", subject=f"h{i}")
    path = writer.log().path
    rows = path.read_text().splitlines()
    doc = json.loads(rows[1])
    doc["ts"] += 1
    rows[1] = json.dumps(doc)
    path.write_text("\n".join(rows) + "\n")
    assert cli.main(["verify"]) == 1
    out = capsys.readouterr().out
    assert "BROKEN" in out and "activity.log:2" in out


# -- rotation and retention ---------------------------------------------------------------------
def test_rotation_keeps_one_chain_across_files(tmp_path):
    log = small(tmp_path, limits=Limits(max_bytes=700, keep=3))
    for i in range(40):
        log.add(line(subject=f"s{i}", ts=1_800_000_000.0 + i))
    assert len(log.files()) == 3
    done = log.verify()
    assert done.ok, done.problems
    seqs = [e.seq for e in log.entries() if isinstance(e, Entry)]
    assert seqs == list(range(seqs[0], 40)) and seqs[0] > 0
    assert sum(p.stat().st_size for p in log.files()) < 3 * 1200


def test_a_file_older_than_the_retention_is_removed_and_the_chain_still_holds(tmp_path):
    now = [1_800_000_000.0]
    log = small(tmp_path, limits=Limits(max_bytes=500, keep=100, retention_s=3600.0), clock=lambda: now[0])
    for i in range(12):
        now[0] += 10
        log.add(line(subject=f"old{i}", ts=now[0]))
    assert len(log.files()) > 3
    now[0] += 7200
    for i in range(30):
        now[0] += 10
        log.add(line(subject=f"new{i}", ts=now[0]))
    done = log.verify()
    assert done.ok, done.problems
    left = [e.subject for e in log.entries() if isinstance(e, Entry)]
    assert "old0" not in left and "new29" in left
    assert log._head.load().payload["base"] != "0" * 64


def test_a_record_does_not_expire_without_a_retention(tmp_path):
    now = [1_800_000_000.0]
    log = small(tmp_path, limits=Limits(max_bytes=500, keep=100), clock=lambda: now[0])
    for i in range(12):
        now[0] += 10
        log.add(line(subject=f"old{i}", ts=now[0]))
    now[0] += 10 * 86400
    for i in range(30):
        now[0] += 10
        log.add(line(subject=f"new{i}", ts=now[0]))
    assert log.verify().ok
    assert "old0" in [e.subject for e in log.entries() if isinstance(e, Entry)]


def test_an_active_file_older_than_the_age_limit_is_rotated(tmp_path):
    now = [1_800_000_000.0]
    log = small(tmp_path, limits=Limits(max_age_s=100.0, keep=4), clock=lambda: now[0])
    log.add(line(subject="a", ts=now[0]))
    now[0] += 500
    log.add(line(subject="b", ts=now[0]))
    assert len(log.files()) == 2 and log.verify().ok


# -- concurrent writers -----------------------------------------------------------------------------
WRITER = """
import sys, time
from pathlib import Path
from ml_stack.activity import schema
from ml_stack.activity.log import ActivityLog, Limits
key = bytes(range(32))
log = ActivityLog(Path(sys.argv[1]), key=lambda: key, limits=Limits(max_bytes=3000, keep=50))
for i in range(25):
    log.add(schema.build("net.download", ts=time.time(), actor="p" + sys.argv[2], session="s", subject=f"w{sys.argv[2]}-{i}"))
"""


def test_several_processes_appending_at_once_leave_a_chain_that_verifies(tmp_path):
    env = {**os.environ, "PYTHONPATH": SRC}
    kids = [subprocess.Popen([sys.executable, "-c", WRITER, str(tmp_path / "log"), str(n)], env=env)
            for n in range(4)]
    assert [k.wait(timeout=120) for k in kids] == [0, 0, 0, 0]
    log = small(tmp_path, limits=Limits(keep=50))
    done = log.verify()
    assert done.ok, done.problems
    rows = [e for e in log.entries() if isinstance(e, Entry)]
    assert len(rows) == 100 and sorted(e.seq for e in rows) == list(range(100))
    for n in range(4):
        mine = [e.subject for e in rows if e.actor == f"p{n}"]
        assert mine == [f"w{n}-{i}" for i in range(25)]


# -- failure never reaches the caller ------------------------------------------------------------------
def test_a_full_disk_drops_the_record_counts_it_and_the_next_one_says_so(person, monkeypatch):
    real = ActivityLog.add

    def full(self, payload):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(ActivityLog, "add", full)
    assert writer.record("net.download", subject="a") is False
    assert writer.record("net.download", subject="b") is False
    assert writer.drops()["dropped"] == 2 and writer.drops()["cause"] == "OSError"
    monkeypatch.setattr(ActivityLog, "add", real)
    assert writer.record("net.download", subject="c")
    kinds = [(e.kind, e.subject) for e in entries()]
    assert kinds[0][0] == "activity.gap" and entries()[0].meta["dropped"] == 2
    assert kinds[1] == ("net.download", "c")


def test_a_locked_keystore_drops_records_and_the_caller_goes_on(person, counting):
    counting.refuse = keystore.KeystoreDenied.__mro__[1] and __import__("keyring").errors.KeyringError
    assert writer.record("net.download", subject="a") is False
    assert writer.drops().get("dropped", 0) >= 1
    assert not list(writer.directory().glob("activity.log*"))


def test_record_never_raises_whatever_it_is_given(person):
    assert writer.record("net.download", subject=object(), meta={"x": object(), 3: 4}, refs=None) in (True, False)
    assert writer.record("", subject=None) in (True, False)
    assert writer.record("net.download", meta={"x": float("nan")})


def test_a_record_written_while_one_is_being_written_is_dropped_not_recursed(person, monkeypatch):
    seen = []
    real = ActivityLog.add

    def nested(self, payload):
        seen.append(writer.record("net.download", subject="inner"))
        return real(self, payload)
    monkeypatch.setattr(ActivityLog, "add", nested)
    assert writer.record("net.download", subject="outer")
    assert seen == [False] and [e.subject for e in entries()] == ["outer"]


# -- switching it off ------------------------------------------------------------------------------------
def test_off_by_a_person_is_logged_first_and_then_nothing_is(person, monkeypatch):
    writer.record("net.download", subject="before")
    monkeypatch.setenv(writer.ENV_OFF, "off")
    assert writer.record("net.download", subject="after") is False
    assert writer.record("net.download", subject="after2") is False
    got = [(e.kind, e.subject) for e in entries()]
    assert got == [("net.download", "before"), ("activity.off", "")]


@pytest.mark.parametrize("marker", ["CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE"])
def test_an_agent_cannot_switch_the_log_off_and_the_try_is_logged(person, monkeypatch, marker):
    monkeypatch.setenv(marker, "1")
    monkeypatch.setenv(writer.ENV_OFF, "off")
    assert writer.record("net.download", subject="still")
    got = [(e.kind, e.subject) for e in entries()]
    assert got == [("activity.off_refused", ""), ("net.download", "still")]


def test_the_session_and_the_actor_come_from_the_process(person, monkeypatch):
    writer.bind_session("chat-9")
    monkeypatch.setenv(writer.ENV_ACTOR, "agent:scout")
    writer.record("net.download", subject="a")
    writer.record("role.changed", actor="person", subject="reader")
    first, second = entries()
    assert (first.session, first.actor, second.actor) == ("chat-9", "agent:scout", "person")


def test_retention_defaults_to_ninety_days_and_follows_the_setting(person, monkeypatch):
    assert writer.log().retention_s == 90 * 86400
    writer.reset = None
    monkeypatch.setenv(writer.ENV_RETENTION, "3")
    from tests.activity_support import reset
    reset()
    assert writer.log().retention_s == 3 * 86400


def test_time_is_the_only_plaintext_field(tmp_path):
    log = filled(tmp_path, 1)
    doc = json.loads(log.path.read_text())
    assert set(doc) == {"v", "ts", "ct", "seq", "prev", "hash"} and doc["ts"] == pytest.approx(1_800_000_000.0)
    assert time.time() > 0
