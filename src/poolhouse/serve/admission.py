"""Whether a machine can hold one more model server, judged from the lease registry.

Every server on the registry, and every unmanaged one found running, is charged what it
takes: the estimate recorded when it was admitted, else its weights on disk. The server asked
for is charged the same way, and the sum is rated against the memory this machine allows
(`poolhouse.hub.room`): green under 80 percent, yellow to 95, red beyond. Red waits for room
and then refuses. Servers are not counted or limited otherwise; any number may be up while
they fit.

A server runs on the device ``gpu``, or on ``cpu`` when it offloads no layer or says so. `poolhouse.gate` queues the requests of a device.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poolhouse.serve.backend import ServerFailed, ServerSpec
from poolhouse.serve.preflight import (
    RUNTIME_ALLOWANCE_BYTES,
    _kv_estimate_bytes,
    draft_kv_estimate_bytes,
    read_gguf_header,
    recurrent_state_bytes,
)
from poolhouse.serve.process import pid_exists
from poolhouse.serve.weights import weight_of
from poolhouse.ui.verdict import THRESHOLDS, verdict_of

__all__ = [
    "ENV_WAIT",
    "RED_AT",
    "YELLOW_AT",
    "AdmissionRefused",
    "Verdict",
    "charge",
    "check",
    "compatible",
    "device_of",
    "estimate_bytes",
    "live",
    "rate",
]

ENV_WAIT = "POOLHOUSE_ADMISSION_WAIT_S"
"""Seconds a start waits for memory to come free before it is refused."""

DEFAULT_WAIT_S = 60.0
YELLOW_AT = THRESHOLDS.yellow_at
RED_AT = THRESHOLDS.red_at
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


def device_of(settings: ServerSpec | Mapping[str, Any]) -> str:
    """The device, ``cpu`` or ``gpu``, a spec or its settings as a dict compute on: the
    ``device`` given, else ``cpu`` when no layer is offloaded. Raises ``ValueError`` on a
    device that is neither or that disagrees with ``n_gpu_layers``."""
    get = settings.get if isinstance(settings, Mapping) else lambda k, d=None: getattr(settings, k, d)
    given = str(get("device") or "").strip().lower()
    none_offloaded = str(get("n_gpu_layers") if get("n_gpu_layers") is not None else "auto").strip() == "0"
    if given not in ("", "cpu", "gpu"):
        raise ValueError(f"device is cpu or gpu, not {given!r}")
    if given == "gpu" and none_offloaded:
        raise ValueError("device gpu with n_gpu_layers 0")
    return given or ("cpu" if none_offloaded else "gpu")


def _size(ref: str | Path | None) -> int:
    return weight_of(ref) if ref else 0


def estimate_bytes(spec: ServerSpec) -> int:
    """What serving ``spec`` will take: weights, draft and projector files, the KV cache
    read off the GGUF header, and a fixed allowance for the runtime. Weights not on disk
    are unknown and count as 0."""
    weights = _size(spec.model)
    if not weights:
        return 0
    kv = 0
    try:
        meta = read_gguf_header(Path(str(spec.model)))
        kv = _kv_estimate_bytes(meta, int(spec.context),
                                spec.cache_type_k or "f16", spec.cache_type_v or "f16")
        kv += recurrent_state_bytes(meta) * max(1, spec.parallel)
    except (OSError, ValueError, struct.error):
        kv = 0
    return (weights + _size(spec.draft) + _size(spec.mmproj) + kv
            + draft_kv_estimate_bytes(spec) + RUNTIME_ALLOWANCE_BYTES)


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
    return verdict_of(total, budget) if budget > 0 else "green"


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


def pending_serves(spec: ServerSpec, entry: Mapping[str, Any]) -> bool:
    """Whether the server being started on ``entry`` will serve ``spec`` when it is up."""
    held = max(1, int(entry.get("parallel") or 1))
    return (int(entry.get("context") or 0) // held >= int(spec.context) // max(1, int(spec.parallel or 1))
            and held >= max(1, int(spec.parallel or 1))
            and bool(entry.get("embedding")) == bool(spec.embedding)
            and bool(entry.get("reranking")) == bool(spec.reranking)
            and not (spec.mmproj and not entry.get("mmproj")))


def compatible(spec: ServerSpec, entry: Mapping[str, Any], mismatches: list[str]) -> bool:
    """Whether the server on ``entry`` can serve ``spec`` as it is, given the settings
    in which it differs (``mismatches``, from `serving_mismatch`)."""
    if mismatches:
        return False
    if bool(entry.get("embedding")) != bool(spec.embedding):
        return False
    if bool(entry.get("reranking")) != bool(spec.reranking):
        return False
    if spec.mtp is not None and entry.get("mtp") is not spec.mtp:
        return False
    for field in ("cache_type_k", "cache_type_v", "spec_type"):
        asked = getattr(spec, field)
        if asked and entry.get(field) != asked:
            return False
    for field in ("draft", "chat_template_file"):
        asked = getattr(spec, field)
        loaded = entry.get(field)
        if asked and (not loaded or Path(asked).resolve() != Path(loaded).resolve()):
            return False
    return not (spec.mmproj and not entry.get("mmproj"))
