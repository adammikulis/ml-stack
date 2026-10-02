"""The event log: redaction, rotation, the hash chain and each way of tampering with it."""

from __future__ import annotations

import json

import pytest

from ml_stack.sentinel.events import Bus, Event, EventLog, Severity
from ml_stack.sentinel.redaction import MASK, redact, redact_value


def _log(tmp_path, **kw) -> EventLog:
    return EventLog(tmp_path / "events.log", **kw)


def _fill(log: EventLog, n: int = 6) -> None:
    for i in range(n):
        log.append(Event("test.event", Severity.NOTICE, "test", f"model:m{i}", {"n": i}))


def _lines(log: EventLog) -> list[str]:
    return log.path.read_text().splitlines()


def test_a_secret_never_reaches_the_log(tmp_path):
    log = _log(tmp_path)
    token = "hf_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7"
    log.append(Event("x.y", Severity.INFO, "t", "peer:1", {
        "header": f"Authorization: Bearer {token}", "api_key": "plainvalue123",
        "note": f"token={token}", "sha256": "ab" * 32}))
    text = log.path.read_text()
    assert token not in text and "plainvalue123" not in text
    assert "ab" * 32 in text


@pytest.mark.parametrize("secret", [
    "hf_" + "Z" * 30, "mlsk1." + "q" * 30, "sk-" + "a" * 30, "ghp_" + "b" * 30,
    "AKIA" + "A" * 16, "Bearer " + "c" * 20, "password = hunter2hunter2",
    "https://x.example/path?token=abcdef123456", "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----",
])
def test_redact_masks_each_secret_shape(secret):
    assert MASK in redact(f"before {secret} after")
    assert secret.split()[-1] not in redact(secret) or secret.endswith("KEY-----")


def test_redact_keeps_digests_and_ordinary_text():
    text = "sha256 " + "0f" * 32 + " model qwen3-4b ok"
    assert redact(text) == text
    assert redact_value({"password": "x" * 9, "n": 3, "ok": True})["password"] == MASK


def test_the_chain_verifies_and_survives_a_reopen(tmp_path):
    log = _log(tmp_path)
    _fill(log)
    again = _log(tmp_path)
    again.append(Event("later", Severity.INFO, "t"))
    result = again.verify()
    assert result.ok and result.records == 7
    assert [e.kind for e in again.read()][-1] == "later"


def test_an_edited_line_is_found(tmp_path):
    log = _log(tmp_path)
    _fill(log)
    lines = _lines(log)
    record = json.loads(lines[2])
    record["severity"] = "info"
    lines[2] = json.dumps(record, sort_keys=True)
    log.path.write_text("\n".join(lines) + "\n")
    result = log.verify()
    assert not result.ok and any("edited" in p for p in result.problems)


def test_a_removed_line_is_found(tmp_path):
    log = _log(tmp_path)
    _fill(log)
    lines = _lines(log)
    del lines[3]
    log.path.write_text("\n".join(lines) + "\n")
    result = log.verify()
    assert not result.ok and any("does not follow" in p or "sequence" in p
                                 for p in result.problems)


def test_reordered_lines_are_found(tmp_path):
    log = _log(tmp_path)
    _fill(log)
    lines = _lines(log)
    lines[1], lines[2] = lines[2], lines[1]
    log.path.write_text("\n".join(lines) + "\n")
    assert not log.verify().ok


def test_a_log_cut_short_is_found(tmp_path):
    log = _log(tmp_path)
    _fill(log)
    log.path.write_text("\n".join(_lines(log)[:-2]) + "\n")
    result = log.verify()
    assert not result.ok and any("cut off" in p for p in result.problems)


def test_a_deleted_log_is_found(tmp_path):
    log = _log(tmp_path)
    _fill(log)
    log.path.unlink()
    assert not log.verify().ok


def test_a_forged_head_is_found(tmp_path):
    log = _log(tmp_path)
    _fill(log)
    head = log.path.with_name("events.log.head")
    doc = json.loads(head.read_text())
    doc["payload"]["count"] = 1
    head.write_text(json.dumps(doc))
    result = log.verify()
    assert not result.ok and any("seal" in p for p in result.problems)


def test_a_whole_rewrite_is_caught_by_an_outside_anchor(tmp_path):
    log = _log(tmp_path)
    _fill(log)
    anchor = log.head()
    other = tmp_path / "other"
    other.mkdir()
    forged = EventLog(other / "events.log")
    _fill(forged)
    log.path.write_text(forged.path.read_text())
    log.path.with_name("events.log.head").write_text(
        forged.path.with_name("events.log.head").read_text())
    log.path.with_name("events.log.head.key").write_bytes(
        forged.path.with_name("events.log.head.key").read_bytes())
    assert log.verify().ok
    assert not log.verify(anchor).ok


def test_rotation_keeps_the_chain_whole_and_the_log_bounded(tmp_path):
    log = _log(tmp_path, max_bytes=600, keep=3)
    _fill(log, 40)
    assert len(log.files()) <= 3
    assert sum(f.stat().st_size for f in log.files()) < 3 * 1200
    result = log.verify()
    assert result.ok, result.problems
    assert result.records < 40


def test_files_are_private(tmp_path):
    log = _log(tmp_path)
    _fill(log, 1)
    assert log.path.stat().st_mode & 0o777 == 0o600
    assert log.path.with_name("events.log.head").stat().st_mode & 0o777 == 0o600


def test_a_failing_subscriber_does_not_stop_the_others(tmp_path):
    bus = Bus(_log(tmp_path))
    seen = []

    def bad(_e):
        raise RuntimeError("boom")

    bus.subscribe(bad)
    stop = bus.subscribe(seen.append)
    bus.emit(Event("a.b", Severity.INFO, "t"))
    stop()
    bus.emit(Event("a.c", Severity.INFO, "t"))
    assert [e.kind for e in seen] == ["a.b"]
    assert len(bus.recent(kind="a.")) == 2


def test_the_anchor_file_records_each_head(tmp_path):
    log = _log(tmp_path, anchor=tmp_path / "anchor.log")
    _fill(log, 3)
    assert (tmp_path / "anchor.log").read_text().splitlines()[-1] == log.head()
