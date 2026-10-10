"""Poolhouse: pool every device's compute. One pool, one board per project, agents and people on every device.

    import poolhouse as ph
    ph.pool     the devices you pooled: status, members, join, add_device, remove, policy, sync, capacity
    ph.board    a project's board, shared by every device: send, inbox, dm, thread, wait, notes, claims, links
    ph.leases   this device's lease table: take and give back CPU slots, memory, the GPU, model slots
    ph.test     run a project's tests on other devices of the pool
    ph.models   model files in the Hugging Face hub cache: pull, path;  ph.serve: up, down, status
    ph.client   talk to a served model: Client, Request, is_healthy;  ph.hub: discover, fetch
    ph.Error    what all of these raise (NotRunning, Denied, Conflict); ph.__version__
Stable within a minor version (docs/api.md); everything else is internal. Importing loads nothing else, and
no code inside poolhouse imports through ``ph``: this is the one facade.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

from poolhouse.api import NAMESPACES as _NAMESPACES
from poolhouse.errors import Conflict, Denied, Error, NotRunning

if TYPE_CHECKING:  # what a type checker sees; the namespaces load on first use (`__getattr__`)
    from poolhouse import board, client, hub, serve
    from poolhouse.api import leases, models, pool, remote_tests as test

    __version__: str

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Conflict", "Denied", "Error", "NotRunning", "__version__", "board", "client", "hub", "leases", "models",
           "pool", "serve", "test"]


def __getattr__(name: str) -> Any:
    """Load a namespace (or the version) the first time it is used."""
    if name in _NAMESPACES:
        return importlib.import_module(_NAMESPACES[name])
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("poolhouse")
        except PackageNotFoundError:
            return "0+unknown"
    raise AttributeError(f"module 'poolhouse' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
