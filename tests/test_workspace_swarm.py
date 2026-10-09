"""The bus, the chained log and the claims under many agents: wake-ups, incremental reads, indexes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from ml_stack.workspace import ChainBroken, claims as claimlib, wake
from ml_stack.workspace.bus import Bus
from ml_stack.workspace.chain import ChainLog
from ml_stack.workspace.identity import Identity

SRC = str(Path(__file__).resolve().parents[1] / "src")


def msg(to: str, body: str = "x", sender: str = "s", **more) -> dict:
    return {"type": "note", "from": sender, "role": "agent", "to": to, "thread": 0, "reply_to": 0,
            "subject": "", "body": body, "held": "", "flags": [], "expires": 0.0, **more}


def after(seconds: float, action) -> threading.Thread:
    def run() -> None:
        time.sleep(seconds)
        action()
    thread = threading.Thread(target=run)
    thread.start()
    return thread


# -- wait ---------------------------------------------------------------------------------------


@pytest.mark.skipif(not wake.PIPES, reason="named pipes are not available here")
def test_a_waiter_wakes_on_the_signal_not_on_the_next_backoff_tick(tmp_path):
    bus = Bus(tmp_path)
    sender = after(1.5, lambda: Bus(tmp_path).append(msg("me")))
    start = time.monotonic()
    got = bus.wait("me", 20)
    woke = time.monotonic() - start
    sender.join()
    assert [m["to"] for m in got] == ["me"]
    assert 1.4 <= woke < 1.9, woke


@pytest.mark.skipif(not wake.PIPES, reason="named pipes are not available here")
def test_a_message_for_someone_else_does_not_wake_a_waiter(tmp_path):
    bus = Bus(tmp_path)
    other = Bus(tmp_path)
    sender = after(0.3, lambda: other.append(msg("them")))
    waiter = wake.Waiter(tmp_path / "wake", "me")
    try:
        start = time.monotonic()
        ready = wake.select.select([waiter.fd], [], [], 1.0)[0]
    finally:
        waiter.close()
        sender.join()
    assert ready == [] and time.monotonic() - start >= 0.95
    assert bus.inbox("me") == []


@pytest.mark.parametrize("pipes", [True, False])
def test_a_message_that_lands_between_the_check_and_the_sleep_is_not_missed(tmp_path, monkeypatch, pipes):
    monkeypatch.setattr(wake, "PIPES", pipes and wake.PIPES)
    sender = Bus(tmp_path)

    class Racy(Bus):
        raced = False

        def inbox(self, me, after=None, limit=50):
            found = super().inbox(me, after, limit)
            if not self.raced:
                self.raced = True
                sender.append(msg(me, "late"))
            return found

    start = time.monotonic()
    got = Racy(tmp_path).wait("me", 20)
    assert [m["body"] for m in got] == ["late"] and time.monotonic() - start < 1.0


def test_two_hundred_waits_each_racing_one_send_never_miss(tmp_path):
    bus = Bus(tmp_path)
    for i in range(200):
        me = f"agent{i % 7}"
        cursor = bus.cursor(me)
        sender = after(0.001 * (i % 5), lambda me=me, i=i: Bus(tmp_path).append(msg(me, f"m{i}")))
        got = bus.wait(me, 10)
        sender.join()
        assert got and got[-1]["body"] == f"m{i}"
        bus.ack(me, got[-1]["seq"])
        assert bus.cursor(me) > cursor


def test_wait_returns_empty_at_the_timeout_and_at_a_cancel(tmp_path):
    bus = Bus(tmp_path)
    start = time.monotonic()
    assert bus.wait("me", 0.4) == [] and 0.35 <= time.monotonic() - start < 1.0
    flag = threading.Event()
    stopper = after(0.3, flag.set)
    start = time.monotonic()
    assert bus.wait("me", 30, cancel=flag.is_set) == [] and time.monotonic() - start < 1.0
    stopper.join()


def test_a_broadcast_row_wakes_nobody_and_lands_in_no_inbox(tmp_path):
    results: dict[str, list] = {}
    names = [f"w{i}" for i in range(6)]
    threads = [threading.Thread(target=lambda n=n: results.update({n: Bus(tmp_path).wait(n, 2.5)}))
               for n in names]
    for t in threads:
        t.start()
    time.sleep(1.5)
    start = time.monotonic()
    Bus(tmp_path).append(msg("*", "all"))
    for t in threads:
        t.join(timeout=10)
    assert all(results[n] == [] for n in names)
    assert time.monotonic() - start > 0.5


@pytest.mark.skipif(not wake.PIPES, reason="named pipes are not available here")
def test_signals_never_follow_a_name_out_of_the_wake_folder_or_write_into_a_file(tmp_path):
    folder = tmp_path / "wake"
    folder.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("keep")
    (folder / "evil.fifo").symlink_to(victim)
    (folder / "plain.fifo").write_text("keep")
    assert wake.signal(folder, ["evil", "plain", "../victim", "a/b", ".hidden", ""]) == 0
    assert wake.signal(folder) == 0
    assert victim.read_text() == "keep" and (folder / "plain.fifo").read_text() == "keep"
    assert wake.Waiter(folder, "../escape").fd == -1 and not (tmp_path / "escape.fifo").exists()


def test_backoff_grows_from_a_tenth_to_two_seconds_with_jitter():
    steps = [wake.backoff(n) for n in range(8)]
    assert 0.07 <= steps[0] <= 0.13 and all(1.5 <= s <= 2.5 for s in steps[6:])
    assert len({wake.backoff(3) for _ in range(20)}) > 1


# -- the chained log -----------------------------------------------------------------------------


def test_a_reader_sees_rows_another_process_appended_and_reads_only_those(tmp_path):
    reader = ChainLog(tmp_path / "log.jsonl")
    writer = ChainLog(tmp_path / "log.jsonl")
    for i in range(50):
        writer.append({"n": i})
    assert [r["n"] for r in reader.rows()] == list(range(50))
    offset = reader._seen.offset
    writer.append({"n": 50})
    assert [r["seq"] for r in reader.after(50)] == [51]
    assert reader._seen.offset > offset and reader._seen.lines == 51


def test_an_edit_of_the_same_length_is_found_by_the_next_read_and_the_next_append(tmp_path):
    log = ChainLog(tmp_path / "log.jsonl")
    for i in range(5):
        log.append({"body": f"pay {i}"})
    path = tmp_path / "log.jsonl"
    path.write_text(path.read_text().replace("pay 1", "pay 7"))
    assert not log.verify().ok
    assert len(log.rows()) < 5
    with pytest.raises(ChainBroken):
        log.append({"body": "more"})


def test_a_same_length_edit_shows_in_a_plain_read_of_a_warm_process(tmp_path):
    log = ChainLog(tmp_path / "log.jsonl")
    for i in range(5):
        log.append({"body": f"pay {i}"})
    assert len(log.rows()) == 5
    path = tmp_path / "log.jsonl"
    path.write_text(path.read_text().replace("pay 3", "pay 8"))
    assert len(log.rows()) == 3


def test_an_edit_hidden_behind_another_processes_append_is_refused_by_the_next_append(tmp_path):
    mine, theirs = ChainLog(tmp_path / "log.jsonl"), ChainLog(tmp_path / "log.jsonl")
    for i in range(5):
        mine.append({"body": f"pay {i}"})
    mine.rows()
    theirs.append({"body": "other"})
    path = tmp_path / "log.jsonl"
    path.write_text(path.read_text().replace("pay 1", "pay 7"))
    with pytest.raises(ChainBroken):
        mine.append({"body": "more"})


def test_an_edit_that_changes_the_length_is_refused_by_an_append_in_a_warm_process(tmp_path):
    log = ChainLog(tmp_path / "log.jsonl")
    for i in range(5):
        log.append({"body": f"pay {i}"})
    log.rows()
    path = tmp_path / "log.jsonl"
    path.write_text(path.read_text().replace("pay 1", "pay one"))
    with pytest.raises(ChainBroken):
        log.append({"body": "more"})


def test_a_full_verify_always_walks_the_whole_file(tmp_path):
    log = ChainLog(tmp_path / "log.jsonl")
    for i in range(5):
        log.append({"body": f"pay {i}"})
    log.rows()
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    first = json.loads(lines[1])
    first["body"] = "pay 9"
    stat = (tmp_path / "log.jsonl").stat()
    lines[1] = json.dumps(first, separators=(",", ":"))
    (tmp_path / "log.jsonl").write_text("\n".join(lines) + "\n")
    os.utime(tmp_path / "log.jsonl", ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert not log.verify().ok


def test_a_pruned_log_is_re_read_by_a_process_that_held_the_old_one(tmp_path):
    clock = [100.0]
    a = ChainLog(tmp_path / "log.jsonl", lambda: clock[0])
    b = ChainLog(tmp_path / "log.jsonl", lambda: clock[0])
    for i in range(10):
        a.append({"n": i})
        clock[0] += 1
    assert len(b.rows()) == 10
    assert a.prune_prefix(lambda r: r["n"] < 6) == 6
    assert [r["n"] for r in b.rows()] == [6, 7, 8, 9] and [r["seq"] for r in b.after(8)] == [9, 10]
    b.append({"n": 10})
    assert a.verify().ok and a.verify().rows == 5


def test_a_second_read_of_a_big_log_is_much_cheaper_than_the_first(tmp_path):
    log = ChainLog(tmp_path / "log.jsonl")
    for i in range(1500):
        log.append({"n": i, "pad": "x" * 200})
    cold = ChainLog(tmp_path / "log.jsonl")
    start = time.perf_counter()
    cold.rows()
    first = time.perf_counter() - start
    log.append({"n": 1500})
    start = time.perf_counter()
    cold.after(1500)
    second = time.perf_counter() - start
    assert second * 10 < first


def test_twenty_writer_processes_leave_one_numbered_verifiable_chain(tmp_path):
    code = ("import sys\nfrom ml_stack.workspace.chain import ChainLog\n"
            "log = ChainLog(sys.argv[1])\n"
            "for i in range(15):\n    log.append({'who': sys.argv[2], 'i': i})\n")
    env = {**os.environ, "PYTHONPATH": SRC}
    procs = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "log.jsonl"), f"w{n}"], env=env)
             for n in range(20)]
    assert [p.wait(timeout=240) for p in procs] == [0] * 20
    log = ChainLog(tmp_path / "log.jsonl")
    rows = log.rows()
    assert [r["seq"] for r in rows] == list(range(1, 301)) and log.verify().ok
    for n in range(20):
        assert [r["i"] for r in rows if r["who"] == f"w{n}"] == list(range(15))


# -- bus indexes ---------------------------------------------------------------------------------


def test_thread_outbox_and_get_agree_with_a_scan_of_the_log(tmp_path):
    bus = Bus(tmp_path)
    root = bus.append(msg("b", "root", "a"))["seq"]
    for i in range(30):
        row = bus.append(msg("a" if i % 2 else "b", f"r{i}", "b" if i % 2 else "a",
                             thread=root if i % 3 else 0, reply_to=root))
        assert bus.get(row["seq"])["body"] == f"r{i}"
    for r in (root, 5, 9):
        scan = [x for x in bus.log.rows() if x["kind"] == "msg" and r in (x["seq"], x.get("thread"))]
        assert bus.thread(r) == scan
    scan = [x for x in bus.log.rows() if x["from"] == "a"][-7:]
    assert bus.outbox("a", 7) == scan and bus.get(9999) is None


def test_the_indexes_follow_a_prune_and_a_replaced_log(tmp_path):
    clock = [1000.0]
    bus = Bus(tmp_path, lambda: clock[0])
    for i in range(10):
        bus.append(msg("b", f"m{i}", "a"))
        clock[0] += 10
    assert len(bus.outbox("a", 100)) == 10
    assert bus.prune(45) == 6
    assert [m["body"] for m in bus.outbox("a", 100)] == [f"m{i}" for i in range(6, 10)]
    assert bus.thread(8)[0]["seq"] == 8 and bus.thread(2) == []
    (tmp_path / "board.db").rename(tmp_path / "board.old")
    bus.append(msg("a", "fresh", "b"))
    assert bus.outbox("a", 100) == [] and [m["body"] for m in bus.outbox("b", 100)] == ["fresh"]


# -- claims --------------------------------------------------------------------------------------

class Tick:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def table(tmp_path, clock, **hooks):
    return claimlib.Claims(tmp_path, 900.0, clock, **hooks)


def test_a_listing_flags_a_claim_that_is_about_to_expire(tmp_path):
    clock = Tick()
    claims, me = table(tmp_path, clock), Identity("a", "agent")
    claims.claim(me, "branch", "feat/x")
    (row,) = claims.listing()
    assert row["expiring_soon"] is False and row["expires_in_s"] == 900.0
    clock.now += 650
    (row,) = claims.listing()
    assert row["expiring_soon"] is True and row["expires_in_s"] == 250.0
    assert claims.heartbeat(me) == 1
    assert claims.listing()[0]["expiring_soon"] is False


def test_renew_extends_all_my_claims_and_none_of_anyone_elses(tmp_path):
    clock = Tick()
    claims = table(tmp_path, clock)
    a, b = Identity("a", "agent"), Identity("b", "agent")
    claims.claim(a, "branch", "one")
    claims.claim(a, "port", "8080")
    claims.claim(b, "branch", "two")
    clock.now += 600
    renewed = claims.renew(a)
    assert sorted(c["key"] for c in renewed) == ["8080", "one"]
    by_key = {c["key"]: c["expires_in_s"] for c in claims.listing()}
    assert by_key == {"one": 900.0, "8080": 900.0, "two": 300.0}


def test_a_renewal_is_capped_per_step_and_in_total_and_never_shortens(tmp_path):
    clock = Tick()
    claims, me = table(tmp_path, clock), Identity("a", "agent")
    made, _ = claims.claim(me, "branch", "x", {"ttl_s": 7200})
    (row,) = claims.renew(me, 10 * 86_400)
    assert row["expires"] == made["expires"]
    clock.now += 7000
    (row,) = claims.renew(me, 10 * 86_400)
    assert row["expires"] - clock.now == claimlib.MAX_RENEW_S and row["capped"] is False
    for _ in range(7):
        clock.now += 3000
        (row,) = claims.renew(me, 10 * 86_400)
    assert row["expires"] == made["since"] + claimlib.MAX_LIFETIME_S and row["capped"] is True
    clock.now = row["expires"] + 1
    assert claims.listing() == []


def test_a_claim_taken_over_an_expired_one_is_reported_with_both_owners(tmp_path):
    clock = Tick()
    stolen, swept = [], []
    claims = table(tmp_path, clock, on_swept=swept.append, on_stolen=lambda old, new: stolen.append((old, new)))
    a, b = Identity("a", "agent"), Identity("b", "agent")
    claims.claim(a, "branch", "x")
    clock.now += 901
    claims.claim(b, "branch", "x")
    assert [(o["owner"], n["owner"], o["reason"]) for o, n in stolen] == [("a", "b", "expired")]
    assert [c["reason"] for c in swept] == ["expired"]
    claims.claim(b, "branch", "y")
    clock.now += 901
    claims.claim(b, "branch", "y")
    assert len(stolen) == 1


def test_a_live_claim_is_never_reported_as_stolen(tmp_path):
    clock = Tick()
    stolen = []
    claims = table(tmp_path, clock, on_stolen=lambda o, n: stolen.append(o))
    claims.claim(Identity("a", "agent"), "branch", "x")
    with pytest.raises(claimlib.Conflict):
        claims.claim(Identity("b", "agent"), "branch", "x")
    assert stolen == []
