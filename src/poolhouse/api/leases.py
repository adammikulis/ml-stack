"""``ph.leases``: the one lease table of this device's node, where compute is taken and given back.

A lease holds typed resources (CPU slots, memory, the GPU, a served-model slot, a named claim such as a branch
or a port) for a holder. Exclusive ones have one holder at a time; CPU slots and memory are counted against what
the device offers. A request that cannot be granted yet queues, shorter work first. Leases of other devices are
decided by their own nodes. See docs/api.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from poolhouse import errors
from poolhouse.api import _node

__all__ = ["LeaseRecord", "acquire", "claim", "cpu_slots", "gpu", "memory_mb", "model_slot", "release", "renew", "view", "wait"]

Resource = dict[str, object]
"""One resource of a lease, made by `cpu_slots`, `memory_mb`, `gpu`, `model_slot` or `claim`."""


@dataclass(frozen=True, slots=True)
class LeaseRecord:
    """One lease of the table: its state is ``held``, ``queued`` or ``ended``."""

    id: str
    state: str
    holder: str
    resources: tuple[Resource, ...]
    background: bool
    held_since_ms: int
    expires_ms: int
    queue_position: int
    wait_estimate_s: int


def cpu_slots(count: int) -> Resource:
    """``count`` CPU job slots of this device."""
    return {"type": "cpu_slots", "count": count}


def memory_mb(mb: int) -> Resource:
    """``mb`` megabytes of this device's memory budget."""
    return {"type": "memory_mb", "mb": mb}


def gpu() -> Resource:
    """This device's GPU, one holder at a time."""
    return {"type": "gpu"}


def model_slot(model: str, *, context: int = 0, parallel: int = 0, draft: str = "") -> Resource:
    """A served model in one shape; one holder per device and model."""
    return {"type": "model_slot", "model": model, "context": context, "parallel": parallel, "draft": draft}


def claim(kind: str, name: str) -> Resource:
    """A named claim: ``kind`` is worktree, branch, area, port, server or install."""
    return {"type": "claim", "kind": kind, "name": name}


def _lease(raw: dict) -> LeaseRecord:
    return LeaseRecord(str(raw["id"]), str(raw["state"]), str(raw["holder"]), tuple(raw.get("resources", [])),
                 raw.get("class") == "background", int(raw.get("granted_ms") or 0), int(raw.get("expires_ms") or 0),
                 int(raw.get("position") or 0), int(raw.get("wait_estimate_s") or 0))


def view() -> list[LeaseRecord]:
    """Every lease of this device's table, held and queued; other boards' holders show as ``other-board``."""
    return [_lease(x) for x in _node.session().call("lease_list")["leases"]]


def acquire(resources: Sequence[Resource], *, ttl_s: int = 0, estimate_s: int = 0, background: bool = False,
            wait: bool = True) -> LeaseRecord:
    """Ask for ``resources``. With ``wait`` the lease may come back ``queued`` (follow it with `wait`); without,
    a busy resource raises `Conflict` naming nothing but the fact. Renew before ``ttl_s`` runs out, or it lapses."""
    params: dict[str, object] = {"resources": list(resources), "wait": wait,
                                 "class": "background" if background else "interactive"}
    params.update({k: v for k, v in (("ttl_s", ttl_s), ("estimate_s", estimate_s)) if v})
    got = _node.session().call("lease_acquire", **params)
    if got.get("state") == "busy":
        raise errors.Conflict("those resources are held by another session; try later or acquire with wait=True")
    return _lease(got)


def wait(lease_id: str, timeout_s: float = 60.0) -> LeaseRecord:
    """Block up to ``timeout_s`` (at most 600) for a queued lease to be granted; its state then."""
    return _lease(_node.session().call("lease_wait", id=lease_id, timeout_ms=int(timeout_s * 1000)))


def renew(lease_id: str, ttl_s: int = 0) -> LeaseRecord:
    """Extend a lease this session holds by ``ttl_s`` (the node's default when 0)."""
    return _lease(_node.session().call("lease_renew", id=lease_id, ttl_s=ttl_s))


def release(lease_id: str) -> bool:
    """Give a lease back, or cancel it while it queues; True when it existed."""
    return bool(_node.session().call("lease_release", id=lease_id)["released"])
