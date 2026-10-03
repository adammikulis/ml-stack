"""Holding the machine's GPU for a training run, through the Broker."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from ml_stack.decide.types import DecideError
from ml_stack.serve import broker_wire
from ml_stack.serve.broker import BrokerError

CLAIM = "gpu-training"


@contextlib.contextmanager
def hold(purpose: str, *, wait_s: float = 0.0, wire: Any = broker_wire) -> Iterator[None]:
    """Claim ``gpu-training`` at the Broker for the block, where it shows in
    ``ml-stack-serve status``.

    Raises `DecideError` when a model server is held by another process or another run
    already holds the claim for longer than ``wait_s``.
    """
    try:
        busy = [s for s in wire.status(start=False)["servers"] if s["holders"] or s["loading"]]
    except (BrokerError, OSError):
        busy = []
    if busy:
        who = "; ".join(f"{s['model']} on port {s['port']} held by "
                        f"{[h['label'] or h['pid'] for h in s['holders']]}" for s in busy)
        raise DecideError(f"the GPU is in use: {who}. Release it, or wait until it is free")
    got = wire.claim(CLAIM, {"purpose": purpose}, timeout=wait_s)
    if not got["granted"]:
        raise DecideError(f"another training run (pid {got['pid']}: "
                          f"{got['info'].get('purpose', '')}) holds the GPU")
    try:
        yield
    finally:
        with contextlib.suppress(BrokerError, OSError):
            wire.unclaim(CLAIM)
