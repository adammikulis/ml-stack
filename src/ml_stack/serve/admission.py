"""Whether a machine can hold one more model server, judged from the lease registry.

Every server on the registry, and every unmanaged one found running, is charged what it
takes: the estimate recorded when it was admitted, else its weights on disk. The server asked
for is charged the same way, and the sum is rated against the memory this machine allows
(`ml_stack.hub.room`): green under 80 percent, yellow to 95, red beyond. Red waits for room
and then refuses. Servers are not counted or limited otherwise; any number may be up while
they fit.

Servers started with every layer on the accelerator share one pool, ``gpu``; servers with
none share ``cpu``. `ml_stack.gate` queues the requests of a pool.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.serve.backend import ServerFailed, ServerSpec
from ml_stack.serve.process import pid_exists
from ml_stack.serve.weights import weight_of

__all__ = ["ENV_WAIT", "AdmissionRefused", "RED_AT", "YELLOW_AT", "Verdict", "charge",
           "check", "compatible", "estimate_bytes", "live", "pool_of", "rate"]

ENV_WAIT = "ML_STACK_ADMISSION_WAIT_S"
"""Seconds a start waits for memory to come free before it is refused."""

DEFAULT_WAIT_S = 60.0
YELLOW_AT = 0.80
RED_AT = 0.95
WEIGHTS_OVERHEAD = 1.1
"""Share of a model's weights taken on top of them when nothing better is known."""


class AdmissionRefused(ServerFailed):
    """The servers up, and this one, would not fit in the memory this machine allows."""


@dataclass(frozen=True)
class Verdict:
    """The rating of a machine with one more server on it."""

    rating: str
    wanted: int
    committed: int
    budget: int
    holders: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return self.wanted + self.committed

    def said(self) -> str:
        gib = 1024 ** 3
        held = "; ".join(self.holders) or "nothing else"
        return (f"{self.rating}: this server needs {self.wanted / gib:.1f} GiB beside "
                f"{self.committed / gib:.1f} GiB already taken, {self.total / gib:.1f} of "
                f"{self.budget / gib:.1f} GiB this machine allows ({held})")


def wait_s() -> float:
    """How long a start waits for memory."""
    try:
        return float(os.environ.get(ENV_WAIT) or DEFAULT_WAIT_S)
    except ValueError:
        return DEFAULT_WAIT_S


def pool_of(spec: ServerSpec) -> str:
    """The accelerator pool ``spec`` computes in."""
    return "cpu" if str(spec.n_gpu_layers).strip() == "0" else "gpu"


def _size(ref: str | Path | None) -> int:
    return weight_of(ref) if ref else 0


def estimate_bytes(spec: ServerSpec) -> int:
    """What serving ``spec`` will take: weights, draft and projector files, the KV cache
    read off the GGUF header, and a fixed allowance for the runtime. Weights not on disk
    are unknown and count as 0."""
    from ml_stack.serve.preflight import RUNTIME_ALLOWANCE_BYTES, _kv_estimate_bytes, read_gguf_header

    weights = _size(spec.model)
    if not weights:
        return 0
    kv = 0
    try:
        kv = _kv_estimate_bytes(read_gguf_header(Path(str(spec.model))), int(spec.context),
                                spec.cache_type_k or "f16", spec.cache_type_v or "f16")
    except (OSError, ValueError, struct.error):
        kv = 0
    return weights + _size(spec.draft) + _size(spec.mmproj) + kv + RUNTIME_ALLOWANCE_BYTES


def charge(entry: Mapping[str, Any]) -> int:
    """What a registry entry holds: its recorded estimate, else its weights on disk."""
    recorded = entry.get("est_bytes")
    if isinstance(recorded, int) and recorded > 0:
        return recorded
    return int(_size(str(entry.get("model") or "")) * WEIGHTS_OVERHEAD)


def live(entry: Mapping[str, Any]) -> bool:
    """Whether a registry entry is a server that is up or about to be: its process exists,
    or it is a start in progress whose owner exists."""
    pid, owner = entry.get("pid"), entry.get("owner_pid")
    if isinstance(pid, int):
        return pid_exists(pid)
    return bool(entry.get("pending")) and isinstance(owner, int) and pid_exists(owner)


def rate(total: int, budget: int) -> str:
    """``green``, ``yellow`` or ``red`` for ``total`` bytes against ``budget``; green when
    the budget is unknown."""
    if budget <= 0:
        return "green"
    share = total / budget
    return "red" if share >= RED_AT else "yellow" if share >= YELLOW_AT else "green"


def check(spec: ServerSpec, records: Mapping[int, Mapping[str, Any]], *, budget: int,
          unmanaged: Iterable[Mapping[str, Any]] = ()) -> Verdict:
    """The rating with ``spec`` added to the live servers in ``records`` (keyed by port;
    ``spec``'s own port is replaced) and to the ``unmanaged`` servers running beside them."""
    holders: list[str] = []
    committed = 0
    for port, entry in records.items():
        if port == spec.port or not live(entry):
            continue
        took = charge(entry)
        committed += took
        holders.append(f"port {port} {Path(str(entry.get('model') or '')).name} "
                       f"{took / 1024 ** 3:.1f} GiB")
    for proc in unmanaged:
        took = int(proc.get("rss") or 0)
        committed += took
        holders.append(f"unmanaged pid {proc.get('pid')} port {proc.get('port')} "
                       f"{took / 1024 ** 3:.1f} GiB")
    wanted = estimate_bytes(spec)
    return Verdict(rate(wanted + committed, budget), wanted, committed, budget, tuple(holders))


def compatible(spec: ServerSpec, entry: Mapping[str, Any], mismatches: list[str]) -> bool:
    """Whether the server on ``entry`` can serve ``spec`` as it is, given the settings
    in which it differs (``mismatches``, from `serving_mismatch`)."""
    if mismatches:
        return False
    if bool(entry.get("embedding")) != bool(spec.embedding):
        return False
    return not (spec.mmproj and not entry.get("mmproj"))
