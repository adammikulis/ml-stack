"""The public side of serving, `ph.serve.up/down/status` (docs/api.md): a model server held under a broker lease.

`up` asks this device's broker for a server of a model and holds the lease in a detached process until `down`, the
idle timeout, or the holder's death; two `up`s of the same shape share one server. Nothing starts a server
without the lease: the broker admits it against the memory the device lets a model use, queues it when memory
is short, and picks the port.
"""

from __future__ import annotations

from dataclasses import dataclass

from poolhouse import hub
from poolhouse.serve import holding, ops
from poolhouse.serve.backend import ServerSpec
from poolhouse.serve.holding import Hold

__all__ = ["Served", "down", "status", "up"]


@dataclass(frozen=True, slots=True)
class Served:
    """A leased model server."""

    lease: str
    model: str
    state: str
    """``ready``, ``queued`` (memory is short; it starts when it frees) or ``starting``."""
    base_url: str
    """The OpenAI-compatible root, such as ``http://127.0.0.1:8080``; empty until ready."""
    port: int
    context: int
    parallel: int
    adopted: bool = False
    """True when an earlier `up` of the same shape already held the server."""


def _served(h: Hold) -> Served:
    return Served(h.id, h.model, h.status, h.base_url, h.port, h.context, h.parallel, h.adopted)


def up(model: str, *, context: int = 0, parallel: int = 0, wait: bool = True,
       reason: str = "") -> Served:
    """Lease a server of ``model`` (a path, a file name found in the Hub cache, or ``hf:owner/repo/file.gguf``).

    ``context`` is tokens across all slots and ``parallel`` the slots (the server defaults when 0). With ``wait``
    this returns when the server is ready; without, it returns at once with state ``queued`` when memory is
    short. ``reason`` is one line saying why, kept in the lease's record. Raises `poolhouse.Error` when the
    request is refused, saying what would fit.
    """
    found = str(hub.located(model) or model)
    asked = ServerSpec(model=found)
    spec_in = ServerSpec(model=found, context=context or asked.context, parallel=parallel or asked.parallel)
    manager = ops.manager_for()
    spec = ops.resolve_spec(spec_in, manager=manager).spec
    weight = holding.check_fits(spec, ask=model)
    held = holding.up(spec, manager=manager, say=lambda _line: None,
                      terms=holding.Terms(wait_s=600.0, wait=wait, weight=weight, reason=reason))
    return _served(held)


def down(target: str) -> list[str]:
    """Release the lease named by ``target``: a lease id, a port, or part of a model name. The server stops when
    no other lease uses it. Returns the ids released (none when nothing matched)."""
    found = [h for h in holding.holds() if h.id == target] or holding.target(target)
    return [released.id for released, _said in holding.down(found, manager=ops.manager_for())]


def status() -> list[Served]:
    """The leases held through `up` on this device whose holder is running."""
    return [_served(h) for h in holding.holds()]
