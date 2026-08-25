"""A short benchmark a peer runs once, when it first joins the swarm."""

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
"""The ``Rates`` kind this is filed under. Leading underscore because it is a proxy for"""

BUDGET_S = 1.5
"""Seconds of work. Long enough that scheduler noise does not dominate, short enough"""

BLOCK = b"ml-stack-fleet-bench" * 64
CHUNK = 200
"""Rounds between clock checks. Reading the clock every iteration would measure the"""


def measure(budget_s: float = BUDGET_S) -> dict[str, Any]:
    """Hash and float throughput over a fixed time budget."""
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
    """What to submit to a peer to have it benchmark itself."""
    return ["python3", "-m", "ml_stack.fleet.bench", "--budget", str(budget_s)]


def main(argv_: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="ml-stack-bench")
    ap.add_argument("--budget", type=float, default=BUDGET_S,
                    help="seconds to spend measuring (default 1.5)")
    a = ap.parse_args(argv_)
    result = measure(a.budget)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def calibrate(peers: "Sequence[Peer]", rates: "Rates", *,
              budget_s: float = BUDGET_S, timeout_s: float = 120.0,
              force: bool = False,
              on_event: "Callable[[str, dict[str, Any]], None] | None" = None,
              ) -> dict[str, float]:
    """Benchmark every peer that has never been benchmarked. Returns name -> score."""
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
        rates.record(name, BENCH_KIND, units=score, seconds=1.0)
        scores[name] = score
        emit("benched", peer=name, score=score)
    rates.save()
    return scores
