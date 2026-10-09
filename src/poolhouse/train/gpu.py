"""Training errors around the shared serving GPU claim."""

import contextlib
from collections.abc import Iterator
from typing import Any

from poolhouse.decide.types import DecideError
from poolhouse.serve import broker_wire
from poolhouse.serve.broker import BrokerError
from poolhouse.serve.gpu import CLAIM, hold as gpu_hold

__all__ = ["CLAIM", "hold"]


@contextlib.contextmanager
def hold(purpose: str, *, wait_s: float = 0.0, wire: Any = broker_wire) -> Iterator[None]:
    """Preserve training's typed errors around the maintained broker claim."""
    try:
        with gpu_hold(purpose, wait_s=wait_s, wire=wire):
            yield
    except BrokerError as exc:
        raise DecideError(str(exc)) from exc
