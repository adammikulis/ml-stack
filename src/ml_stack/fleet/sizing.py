"""What a model will take to serve on one machine: `estimate` in bytes, from the file at
hand when there is one and from the Hub listing when there is not."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack.hub import draft_for, files, located

__all__ = ["estimate"]


def estimate(model: str, *, context: int = 32768, draft: str = "auto",
             preflight: Callable[..., Any] | None = None, binary: str = "") -> int:
    """Bytes ``model`` will take to serve: weights, its draft head, the KV cache at
    ``context`` and the runtime allowance -- `serve.preflight`'s own estimate when the
    file is at hand, else the weights' size from the Hub listing, else 0 for unknown.

    Unknown is not enormous: `plan` sends a model it could not size to the roomiest idle
    peer and lets that peer's preflight be the judge, as `weight_of` leaves it to the load.
    ``preflight`` is `serve.preflight.Preflight` unless a test hands in a fake.
    """
    from ml_stack.serve.preflight import RUNTIME_ALLOWANCE_BYTES
    from ml_stack.serve.weights import weight_of

    path = _at_hand(model)
    if path is None:
        return _hub_bytes(model, draft=draft)
    head: str | Path | None = None
    if draft == "auto":
        from .models import draft_beside

        head = draft_beside(path)
    elif draft:
        head = _at_hand(draft) or draft
    try:
        from ml_stack.serve.backend import ServerSpec

        if preflight is None:
            from ml_stack.serve.preflight import Preflight as preflight
        if not binary:
            from ml_stack.serve.binary import find_binary

            binary = str(find_binary() or "llama-server")
        report = preflight(ServerSpec(model=path, context=int(context), draft=head),
                           binary=binary, limit_bytes=0)
        weights = int(report.weights_bytes) or weight_of(path)
        kv = int(report.kv_estimate_bytes)
    except Exception:  # noqa: BLE001 - a preflight that cannot read it still leaves the weights
        weights, kv = weight_of(path), 0
    return weights + kv + (weight_of(head) if head else 0) + RUNTIME_ALLOWANCE_BYTES


def _at_hand(model: str) -> Path | None:
    """The file ``model`` names on this machine, or None: a path, a Hub-cached file by its
    exact name, or an ``hf:`` reference whose file has been fetched."""
    if not model:
        return None
    where = Path(model).expanduser()
    if where.is_file():
        return where
    if model.startswith("hf:"):
        parts = [p for p in model[3:].split("/") if p]
        return located("/".join(parts[2:])) if len(parts) > 2 else None
    if "/" in model:
        return None
    return located(model)


def _hub_bytes(model: str, *, draft: str = "auto") -> int:
    """The weights' size from the Hub listing for an ``hf:`` reference not yet fetched --
    its draft head's beside it when ``draft`` is ``auto`` -- or 0 when nothing answers."""
    if not model.startswith("hf:"):
        return 0
    parts = [p for p in model[3:].split("/") if p]
    if len(parts) < 3:
        return 0
    repo, name = "/".join(parts[:2]), "/".join(parts[2:])
    try:
        listed = dict(files(repo))
        total = int(listed.get(name, 0))
        if total and draft == "auto":
            head = draft_for(repo)
            total += int(listed.get(head.rsplit("/", 2)[-1], 0)) if head else 0
        return total
    except Exception:  # noqa: BLE001 - the Hub is somebody else's machine
        return 0

