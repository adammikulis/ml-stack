"""A short benchmark a peer runs once, when it first joins the swarm.

Placement scores an unmeasured peer at the median of the measured ones, which is the
right default but a blunt one: on a fleet holding both a Pi and a 4090, "typical" is a
number describing neither, and the first real unit sent to a new box is a coin-flip that
can cost an hour.

So a box that has never been measured is asked to do something small first. This is a
**measurement, not an assumption** -- the distinction this codebase keeps everywhere
else -- but it measures a proxy, and the docstring has to be honest about which:

* It is stdlib-only, because this package is device tier and cannot import a framework.
  So it measures CPU and memory throughput, and says nothing about a GPU.
* A CPU score does not predict training speed on a card. It predicts *prep* speed well,
  and orders unmeasured peers far better than nothing.
* It is therefore a **prior**, and the first genuine measurement of real work replaces
  it. ``Rates`` stores it under its own kind so it can never be mistaken for one.

Bounded by *time*, not by work, and that choice matters on this fleet specifically: a
work budget large enough to be meaningful on a Mac Studio is one a Raspberry Pi would
still be chewing on minutes later, and the boxes that most need measuring are the slow
ones. A fixed budget with throughput reported back is just as comparable between
machines and costs every machine the same couple of seconds.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .rates import Rates
    from .remote import Peer

__all__ = ["BENCH_KIND", "BUDGET_S", "argv", "calibrate", "main", "measure"]

BENCH_KIND = "_bench"
"""The ``Rates`` kind this is filed under. Leading underscore because it is a proxy for
real work and must never be read as a rate for any actual stage."""

BUDGET_S = 1.5
"""Seconds of work. Long enough that scheduler noise does not dominate, short enough
that joining a swarm is not something you plan around."""

BLOCK = b"ml-stack-fleet-bench" * 64
CHUNK = 200
"""Rounds between clock checks. Reading the clock every iteration would measure the
clock as much as the machine."""


def measure(budget_s: float = BUDGET_S) -> dict[str, Any]:
    """Hash and float throughput over a fixed time budget.

    Two loops because they exercise different things: hashing is integer and memory
    bandwidth, and the float loop is the only part that touches the FPU. A machine can
    be good at one and ordinary at the other, and averaging them into a single score
    keeps that from being invisible.
    """
    half = max(0.05, budget_s / 2)

    digest = hashlib.sha256()
    hashed = 0
    started = time.perf_counter()
    while time.perf_counter() - started < half:
        for _ in range(CHUNK):
            digest.update(BLOCK)
        hashed += CHUNK
    hash_s = time.perf_counter() - started

    acc = 0.0
    floats = 0
    fstart = time.perf_counter()
    while time.perf_counter() - fstart < half:
        for i in range(CHUNK):
            acc += (i * 1.000001) ** 0.5
        floats += CHUNK
    float_s = time.perf_counter() - fstart

    hash_rate = hashed / hash_s if hash_s > 0 else 0.0
    float_rate = floats / float_s if float_s > 0 else 0.0
    return {
        # Geometric mean: a box twice as fast at everything scores twice as high, and
        # one that is enormously good at a single thing cannot ride that alone.
        "score": round((hash_rate * float_rate) ** 0.5, 2),
        "hash_per_s": round(hash_rate, 2),
        "float_per_s": round(float_rate, 2),
        "seconds": round(hash_s + float_s, 3),
        "mb_hashed": round(hashed * len(BLOCK) / 2**20, 1),
        "cpus": os.cpu_count() or 1,
        "arch": platform.machine(),
        "python": platform.python_version(),
        "checksum": digest.hexdigest()[:16],
    }


def argv(budget_s: float = BUDGET_S) -> list[str]:
    """What to submit to a peer to have it benchmark itself.

    ``sys.executable`` is deliberately not used: it is the *coordinator's* interpreter
    path, which on a mixed fleet is not a path that exists on the peer.
    """
    return ["python3", "-m", "ml_stack.fleet.bench", "--budget", str(budget_s)]


def main(argv_: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="ml-stack-bench")
    ap.add_argument("--budget", type=float, default=BUDGET_S,
                    help="seconds to spend measuring (default 1.5)")
    a = ap.parse_args(argv_)
    result = measure(a.budget)
    # One JSON object on stdout, so the coordinator can read it back out of the job log
    # without a file to fetch or a manifest convention to agree on first.
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def calibrate(peers: "Sequence[Peer]", rates: "Rates", *,
              budget_s: float = BUDGET_S, timeout_s: float = 120.0,
              force: bool = False,
              on_event: "Callable[[str, dict[str, Any]], None] | None" = None,
              ) -> dict[str, float]:
    """Benchmark every peer that has never been benchmarked. Returns name -> score.

    Runs once per peer per rates file, which is what makes this a joining cost rather
    than a recurring one. A peer that cannot run it is skipped rather than failed: a box
    with no ``python3`` on its PATH can still be sent work, it just has to earn its
    measurement the slow way.
    """
    def emit(event: str, **fields: Any) -> None:
        if on_event is not None:
            try:
                on_event(event, fields)
            except Exception:                         # noqa: BLE001
                pass

    scores: dict[str, float] = {}
    for peer in peers:
        try:
            name = peer.health().get("name") or peer.name
        except Exception as exc:                      # noqa: BLE001
            emit("skip", peer=peer.base_url, why=str(exc))
            continue
        if not force and rates.get(name, BENCH_KIND) is not None:
            scores[name] = rates.get(name, BENCH_KIND)      # type: ignore[assignment]
            continue
        emit("bench", peer=name)
        try:
            job = peer.submit(argv(budget_s), name="ml-stack-bench")
            final = peer.wait(job["id"], poll_s=1.0, timeout_s=timeout_s)
            if final.get("state") != "done":
                emit("skip", peer=name, why=f"bench {final.get('state')}")
                continue
            result = json.loads(peer.log(job["id"], tail=5).strip().splitlines()[-1])
            score = float(result["score"])
        except Exception as exc:                      # noqa: BLE001
            emit("skip", peer=name, why=str(exc))
            continue
        # Recorded as a rate of one unit over the reciprocal, so it lands in the same
        # store as real measurements without pretending to be one -- BENCH_KIND keeps
        # them apart, and any genuine measurement of real work supersedes it.
        rates.record(name, BENCH_KIND, units=score, seconds=1.0)
        scores[name] = score
        emit("benched", peer=name, score=score)
    rates.save()
    return scores
