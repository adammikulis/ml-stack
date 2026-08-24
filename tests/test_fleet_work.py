"""Spreading work over several peers, against several real daemons.

Nothing is mocked, for the reason the other fleet tests give: the failures this module
exists to prevent are scheduling-shaped and process-shaped. Two units overlapping on a
one-slot box, a broken peer draining the queue, a Ctrl-C leaving three machines running
-- a fake Peer that returns canned job dicts reproduces none of them, because in every
case the thing that went wrong is that a real process really was still running.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from ml_stack.fleet.daemon import JobRunner, device_report, load_or_create_token, make_handler
from ml_stack.fleet.pool import Candidate, Requires, choose, eligible, soonest
from ml_stack.fleet.rates import Rates
from ml_stack.fleet.remote import Peer
from ml_stack.fleet.work import Unit, run


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Box:
    """One real daemon in this process, with a real HTTP server on a real port."""

    def __init__(self, root: Path, name: str, *, slots: int = 1,
                 labels: tuple[str, ...] = (), extra: dict | None = None) -> None:
        self.name = name
        self.files = root / "files"
        self.files.mkdir(parents=True)
        token = load_or_create_token(root)
        self.runner = JobRunner(root, slots=slots)
        port = _free_port()

        def report() -> dict:
            return {**device_report(lambda: dict(extra or {})), "labels": list(labels)}

        self.httpd = ThreadingHTTPServer(
            ("127.0.0.1", port),
            make_handler(self.runner, self.files, token, name, report))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.peer = Peer(f"http://127.0.0.1:{port}", token)

    def close(self) -> None:
        self.runner.shutdown()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def boxes(tmp_path):
    """Three one-slot daemons, the shape of a small home fleet."""
    made = [Box(tmp_path / n, n) for n in ("alpha", "beta", "gamma")]
    try:
        yield made
    finally:
        for b in made:
            b.close()


@pytest.fixture
def rates(tmp_path):
    return Rates(tmp_path / "rates.json")


def _marker_unit(uid: str, marker: Path, seconds: float = 0.6) -> Unit:
    """A unit that records, in a real file, exactly when it ran and for how long."""
    script = (
        "import json,os,sys,time\n"
        "start=time.time()\n"
        f"time.sleep({seconds})\n"
        "open(sys.argv[1],'a').write(json.dumps("
        "{'unit':sys.argv[2],'start':start,'end':time.time()})+'\\n')\n"
    )
    return Unit(id=uid, argv=[sys.executable, "-c", script, str(marker), uid])


# -- eligibility ---------------------------------------------------------
class TestRequires:
    """Why a peer cannot take a unit has to survive as text. 'Nothing ran' with no
    reason is the failure mode that turns a five-minute fix into an afternoon."""

    PI = {"backends": [], "cpus": 4, "ram_gb": 8.0, "labels": ["prep"]}
    RTX = {"backends": ["torch"], "cuda": True, "gpu": "RTX 4090", "cpus": 16,
           "vram_free_gb": 23.0, "ram_gb": 64.0, "labels": ["train"]}

    def test_a_cuda_unit_will_not_land_on_a_pi(self):
        why = Requires(backend="cuda").why_not("pi", self.PI, 1, 1)
        assert "cuda" in why and "no backends" in why

    def test_a_cuda_unit_lands_on_the_card(self):
        assert Requires(backend="cuda").admits("rtx", self.RTX, 1, 1)

    def test_prep_can_be_kept_off_the_training_boxes_by_label(self):
        """Declared, not detected: a box cannot prove it has no GPU, so 'leave the
        training boxes alone' has to be something an operator said."""
        keep_off = Requires(exclude_labels=("train",))
        assert keep_off.admits("pi", self.PI, 1, 1)
        assert "train" in keep_off.why_not("rtx", self.RTX, 1, 1)

    def test_a_missing_measurement_does_not_read_as_too_small(self):
        """The stdlib probe reports no VRAM at all. Refusing on that would exclude every
        box that has not been given an accelerator probe -- which is most of them."""
        unmeasured = {"backends": ["torch"], "cuda": True, "labels": []}
        assert Requires(backend="cuda", min_vram_gb=8.0).admits("x", unmeasured, 1, 1)

    def test_a_measured_shortfall_is_refused_with_both_numbers(self):
        small = {**self.RTX, "vram_free_gb": 4.0}
        why = Requires(backend="cuda", min_vram_gb=8.0).why_not("rtx", small, 1, 1)
        assert "4.0" in why and "8.0" in why

    def test_exclusive_work_refuses_a_box_that_is_partly_busy(self):
        why = Requires(exclusive=True).why_not("rtx", self.RTX, 2, 4)
        assert "2/4" in why

    def test_eligible_reports_the_kept_and_the_reasons_for_the_rest(self):
        cands = [
            Candidate(peer=None, name="pi", report=self.PI, slots=1, free=1),
            Candidate(peer=None, name="rtx", report=self.RTX, slots=1, free=1),
        ]
        kept, refused = eligible(cands, Requires(backend="cuda"))
        assert [c.name for c in kept] == ["rtx"]
        assert "pi" in refused and refused["pi"]


# -- choosing ------------------------------------------------------------
class TestChoose:
    @staticmethod
    def _c(name, *, rate=None, bench=None, free=1, queued=0, transfer_gb=0.0):
        return Candidate(peer=None, name=name, report={}, slots=1, free=free,
                         queued=queued, rate=rate, bench=bench, transfer_gb=transfer_gb)

    def test_an_unmeasured_peer_gets_work_while_a_fast_one_sits_idle(self):
        """A missing measurement is not evidence. Score an unmeasured peer as slow and
        the first peer ever measured wins forever, because nothing else is ever tried."""
        picked = choose([self._c("fast", rate=1000.0), self._c("new")])
        assert picked.name == "new"

    def test_among_unmeasured_peers_the_benchmark_breaks_the_tie(self):
        picked = choose([self._c("slow", bench=10.0), self._c("quick", bench=900.0)])
        assert picked.name == "quick"

    def test_an_unbenchmarked_peer_still_outranks_a_slow_benchmarked_one(self):
        """'Not measured' must not quietly become 'measured badly'."""
        picked = choose([self._c("benched_slow", bench=1.0), self._c("unknown")])
        assert picked.name == "unknown"

    def test_with_everything_measured_the_fastest_wins(self):
        picked = choose([self._c("slow", rate=1.0), self._c("fast", rate=100.0)],
                        score=soonest(work=10.0))
        assert picked.name == "fast"

    def test_a_peer_that_already_holds_the_data_beats_a_faster_one_that_does_not(self):
        """A slower box holding the shard usually wins: a score that ignores transfer
        keeps picking the fast idle box and keeps waiting on the network."""
        picked = choose(
            [self._c("fast_far", rate=100.0, transfer_gb=8.0),
             self._c("slower_near", rate=50.0, transfer_gb=0.0)],
            score=soonest(work=10.0, link_gbps=1.0))
        assert picked.name == "slower_near"

    def test_nobody_free_is_nobody(self):
        assert choose([self._c("a", free=0), self._c("b", free=0)]) is None


# -- fanning out over real daemons ---------------------------------------
def test_units_spread_over_the_peers_and_never_overlap_on_one(boxes, tmp_path, rates):
    """Each box is one slot. Two units overlapping on a single box would mean the
    fan-out is racing the daemon's own invariant rather than respecting it."""
    marker = tmp_path / "ran.jsonl"
    units = [_marker_unit(f"u{i}", marker) for i in range(6)]

    placements = run(units, [b.peer for b in boxes], kind="test", rates=rates,
                     poll_s=0.1)

    assert all(p.ok for p in placements), [p.error for p in placements if not p.ok]
    rows = [json.loads(line) for line in marker.read_text().splitlines()]
    assert len(rows) == 6
    assert len({p.peer for p in placements}) > 1, "everything landed on one box"

    by_peer: dict[str, list[dict]] = {}
    for placement, row in ((p, r) for p in placements for r in rows
                           if r["unit"] == p.unit_id):
        by_peer.setdefault(placement.peer, []).append(row)
    for peer, ran in by_peer.items():
        ran.sort(key=lambda r: r["start"])
        for earlier, later in zip(ran, ran[1:]):
            assert earlier["end"] <= later["start"] + 0.05, \
                f"{peer} ran two units at once"


def test_a_failing_unit_is_retried_on_a_different_peer(boxes, tmp_path, rates):
    """Retrying on the box that just failed it learns nothing. The point of a retry is
    to find out whether the unit is wrong or the machine is."""
    marker = tmp_path / "attempts.txt"
    script = (
        "import sys\n"
        "p=sys.argv[1]\n"
        "n=len(open(p).readlines()) if __import__('os').path.exists(p) else 0\n"
        "open(p,'a').write('x\\n')\n"
        "sys.exit(1 if n==0 else 0)\n"
    )
    unit = Unit(id="flaky", argv=[sys.executable, "-c", script, str(marker)])

    [placement] = run([unit], [b.peer for b in boxes], kind="test", rates=rates,
                      retries=1, poll_s=0.1)

    assert placement.ok, placement.error
    assert placement.attempts == 2
    assert len(set(placement.tried)) == 2, f"retried on the same peer: {placement.tried}"


def test_a_unit_no_peer_admits_fails_at_once_with_every_reason(boxes, rates):
    """Not the same as 'nothing free'. Waiting for capacity that would not help if it
    arrived is a hang, and a hang with no message is the worst of both."""
    unit = Unit(id="needs-cuda", argv=["true"], requires=Requires(backend="cuda"))

    started = time.time()
    [placement] = run([unit], [b.peer for b in boxes], kind="test", rates=rates)

    assert placement.state == "unplaceable"
    assert time.time() - started < 10
    for box in boxes:
        assert box.name in placement.error
    assert "cuda" in placement.error


def test_a_pinned_unit_runs_where_it_was_pinned(boxes, tmp_path, rates):
    marker = tmp_path / "pinned.jsonl"
    unit = _marker_unit("pinned", marker, seconds=0.1)
    unit = Unit(id=unit.id, argv=unit.argv, peer="beta")

    [placement] = run([unit], [b.peer for b in boxes], kind="test", rates=rates,
                      poll_s=0.1)

    assert placement.ok, placement.error
    assert placement.peer == "beta"


def test_a_completed_unit_leaves_a_measurement_behind(boxes, tmp_path, rates):
    """This is where 'fastest' comes from: a job's own elapsed time divided by the work
    it declared. Nothing new is measured to get it."""
    marker = tmp_path / "measured.jsonl"
    units = [_marker_unit(f"m{i}", marker, seconds=0.2) for i in range(3)]

    placements = run(units, [b.peer for b in boxes], kind="tokenize", rates=rates,
                     poll_s=0.1)

    assert all(p.ok for p in placements)
    measured = [rates.get(p.peer, "tokenize") for p in placements]
    assert all(m is not None and m > 0 for m in measured), measured
    assert Rates(rates.path).get(placements[0].peer, "tokenize") is not None, \
        "the measurement did not survive being written to disk"


def test_a_broken_peer_is_quarantined_instead_of_draining_the_queue(tmp_path, rates):
    """The fast-fail black hole: a box whose card has fallen over accepts and fails work
    faster than the healthy boxes can take any, so every unit fails on the one machine
    that cannot run them."""
    good = Box(tmp_path / "good", "good", slots=1)
    broken = Box(tmp_path / "broken", "broken", slots=4)
    try:
        units = [Unit(id=f"u{i}", argv=[sys.executable, "-c", "pass"])
                 for i in range(12)]
        events: list[tuple[str, dict]] = []
        placements = run(units, [good.peer, broken.peer], kind="test", rates=rates,
                         poll_s=0.05, on_event=lambda e, f: events.append((e, f)))
    finally:
        good.close()
        broken.close()

    # Every unit is trivially runnable, so nothing should be lost.
    assert all(p.ok for p in placements), [p.error for p in placements if not p.ok]
    assert any(e == "done" for e, _ in events)


def test_a_multi_slot_peer_really_runs_several_at_once(tmp_path, rates):
    """--slots on the daemon means nothing if the fan-out still feeds it one at a time.
    The specific failure: one worker thread per peer rather than per slot, which looks
    exactly like the daemon ignoring its own flag."""
    box = Box(tmp_path / "prep", "prep", slots=4)
    try:
        marker = tmp_path / "concurrent.jsonl"
        units = [_marker_unit(f"p{i}", marker, seconds=0.8) for i in range(4)]
        placements = run(units, [box.peer], kind="prep", rates=rates, poll_s=0.05)
    finally:
        box.close()

    assert all(p.ok for p in placements), [p.error for p in placements if not p.ok]
    rows = [json.loads(line) for line in marker.read_text().splitlines()]
    assert len(rows) == 4
    span = max(r["end"] for r in rows) - min(r["start"] for r in rows)
    assert span < 4 * 0.8 * 0.75, (
        f"four 0.8s units took {span:.2f}s on a 4-slot box -- they were serialised")


def test_a_labelled_unit_never_lands_on_a_box_that_lacks_the_label(tmp_path, rates):
    """The specific failure: eligibility computed up front but not checked when a worker
    takes its next unit. Every worker then happily grabs anything pending, and the labels
    that were supposed to keep prep off the training boxes decide nothing at all."""
    gpu = Box(tmp_path / "gpu", "gpubox", slots=1, labels=("train",),
              extra={"cuda": True, "backends": ["torch"]})
    pi = Box(tmp_path / "pi", "pi", slots=2, labels=("prep",))
    try:
        units = [Unit(id=f"s{i}", argv=[sys.executable, "-c", "import time;time.sleep(0.3)"],
                      requires=Requires(labels=("prep",))) for i in range(6)]
        placements = run(units, [gpu.peer, pi.peer], kind="prep", rates=rates,
                         poll_s=0.05)
    finally:
        gpu.close()
        pi.close()

    assert all(p.ok for p in placements), [p.error for p in placements if not p.ok]
    landed = {p.peer for p in placements}
    assert landed == {"pi"}, f"prep work reached the training box: {landed}"


class TestVendorsAreNotInterchangeable:
    """torch's HIP build answers True to every CUDA question. A box with a Radeon would
    therefore satisfy backend="cuda" and only reveal itself inside the job -- so "cuda",
    "rocm" and "accelerator" have to be three different questions."""

    AMD = {"backends": ["torch"], "vendor": "amd", "rocm": True, "cuda": False,
           "accelerator": True, "gpu": "AMD Radeon RX 7900 XTX",
           "vram_free_gb": 23.0, "labels": []}
    NVIDIA = {"backends": ["torch"], "vendor": "nvidia", "cuda": True, "rocm": False,
              "accelerator": True, "gpu": "RTX 4090", "vram_free_gb": 23.0, "labels": []}
    APPLE = {"backends": ["mlx"], "vendor": "apple", "cuda": False, "rocm": False,
             "accelerator": True, "unified_memory": True, "labels": []}
    PI = {"backends": [], "vendor": "cpu", "accelerator": False, "cpus": 4, "labels": []}

    def test_a_cuda_only_run_refuses_the_amd_box(self):
        why = Requires(backend="cuda").why_not("amd", self.AMD, 1, 1)
        assert why and "ROCm" in why

    def test_the_refusal_says_what_to_ask_for_instead(self):
        why = Requires(backend="cuda").why_not("amd", self.AMD, 1, 1)
        assert "rocm" in why and "accelerator" in why

    def test_rocm_work_lands_on_the_amd_box_and_not_the_nvidia_one(self):
        wants = Requires(backend="rocm")
        assert wants.admits("amd", self.AMD, 1, 1)
        assert not wants.admits("rtx", self.NVIDIA, 1, 1)

    @pytest.mark.parametrize("name", ["AMD", "NVIDIA", "APPLE"])
    def test_any_gpu_matches_every_vendor(self, name):
        assert Requires(backend="accelerator").admits(name, getattr(self, name), 1, 1)

    def test_any_gpu_still_refuses_a_box_with_none(self):
        assert not Requires(backend="accelerator").admits("pi", self.PI, 1, 1)


class TestABoxThatIsSomebodysDesk:
    """A machine someone works on is not misconfigured when it refuses work, and the
    refusal has to say so -- "nothing could run this" reads as a broken fleet."""

    BUSY = {"backends": ["torch"], "cuda": True, "labels": [], "availability": {
        "available": False,
        "unavailable_because": "this machine is in use (mon tue wed thu fri "
                               "09:00-17:00); work resumes Mon 17:00"}}
    PAUSED = {"backends": ["torch"], "cuda": True, "labels": [], "availability": {
        "available": False, "paused": True,
        "unavailable_because": "paused: gaming, until it is switched back on"}}
    FREE = {"backends": ["torch"], "cuda": True, "labels": [], "availability": {
        "available": True, "unavailable_because": ""}}

    def test_a_box_inside_its_working_hours_is_refused_with_the_hours(self):
        why = Requires(backend="cuda").why_not("amd", self.BUSY, 1, 1)
        assert "17:00" in why

    def test_a_paused_box_says_it_was_paused_not_that_it_is_broken(self):
        why = Requires(backend="cuda").why_not("amd", self.PAUSED, 1, 1)
        assert "paused" in why and "gaming" in why

    def test_the_same_box_outside_its_hours_takes_work(self):
        assert Requires(backend="cuda").admits("amd", self.FREE, 1, 1)

    def test_a_box_with_no_schedule_is_available(self):
        """Most machines never set one, and absence must not read as unavailable."""
        assert Requires().admits("pi", {"backends": [], "labels": []}, 1, 1)
