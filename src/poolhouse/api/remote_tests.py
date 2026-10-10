"""``ph.test``: run a project's tests on other devices of the pool, so the work lands where compute is free.

The tree of the checkout is sent to the device through this device's node, over TLS pinned to the device's
certificate; the device runs the tier with its own interpreter and answers with the result. Nothing runs on a
device until a person turns remote tests on there and names the devices it takes tests from
(docs/test-farm.md). `scripts/test` adds content-keyed reuse, `quick` selection and splitting a run between
devices on top of this.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from poolhouse import errors, features
from poolhouse.api import _node
from poolhouse.fleet.shard_spec import TEST_FILE
from poolhouse.testfarm import report
from poolhouse.testfarm.client import ShardError, Shards, choose, pool_devices
from poolhouse.testfarm.consent import FEATURE

__all__ = ["Device", "Result", "devices", "run"]

TIERS = ("fast", "full", "slow", "all", "gate")


@dataclass(frozen=True, slots=True)
class Device:
    """A pool device and what it says about taking tests."""

    name: str
    fingerprint: str
    connected: bool
    accepts: bool
    """Whether it takes tests from this device now."""
    reason: str
    """Why not, when it does not."""
    platform: str
    free_slots: int


@dataclass(frozen=True, slots=True)
class Result:
    """How a run on one device ended."""

    device: str
    tier: str
    exit_code: int
    passed: int
    failed: int
    skipped: int
    wall_s: float
    failures: tuple[str, ...]
    """The failing tests, one line each."""
    summary: str


def devices() -> list[Device]:
    """The other devices of the pool and whether each takes tests from this one."""
    seat = Shards(_node.session())
    out = []
    for one in pool_devices():
        try:
            caps = seat.capability(one["fingerprint"])
        except ShardError as exc:
            caps = {"accepts": False, "reason": f"did not answer: {exc}"}
        system = caps.get("platform", {})
        out.append(Device(one["name"], one["fingerprint"], one["connected"], bool(caps["accepts"]),
                          str(caps.get("reason", "")), str(system.get("system", "")), int(caps.get("free", 0))))
    return out


def run(tier: str, *, on: str, root: str | Path = ".", files: tuple[str, ...] = (), timeout_s: int = 1800) -> Result:
    """Run ``tier`` (fast, full, slow, all or gate) on the device ``on`` (its name or fingerprint) and wait.

    ``files`` names test files (``tests/test_x.py``) for the ``all`` tier. ``root`` is the checkout to send.
    Raises `Error` when the device is not in the pool, does not take tests, or the run could not finish; a run
    whose tests fail returns normally with a non-zero ``exit_code``.
    """
    if tier not in TIERS:
        raise errors.Error(f"the tier is one of {', '.join(TIERS)}, not {tier!r}")
    if any(not TEST_FILE.match(f) for f in files) or (tier == "gate" and files):
        raise errors.Error("name test files as tests/NAME.py; the gate takes none")
    if not features.enabled(FEATURE):
        raise errors.Error(f"remote tests are off here; a person turns them on with `poolhouse features enable {FEATURE}`")
    chosen = choose(on, pool_devices())[0]
    seat = Shards(_node.session())
    caps = seat.capability(chosen["fingerprint"])
    if not caps["accepts"]:
        raise errors.Error(f"{chosen['name']} does not take tests: {caps.get('reason', '')}")
    shard = seat.send(chosen["fingerprint"], Path(root).resolve(), tier, list(files), timeout_s)
    got = seat.wait(chosen["fingerprint"], shard, timeout_s=timeout_s + 180)
    n = report.counts(got)
    return Result(chosen["name"], tier, int(got["exit"]), n["passed"], n["failed"] + n["error"], n["skipped"],
                  float(got["wall_s"]), tuple(f["nodeid"] for f in got["failures"]), report.summary(got))
