"""``ph.pool``: the pool of devices this device belongs to; who is in it, how it is joined, and what it can compute.

A pool is the set of devices that trust one another. Every device starts as a pool of one. Another device
JOINS it: under the `open` policy automatically when both are on the same network, under `secure` with a short
code that `add_device` makes. A member is removed with `remove`. See docs/api.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from poolhouse import errors, hub, node_join, node_launch
from poolhouse.api import _node

__all__ = ["Capacity", "Member", "Pool", "add_device", "capacity", "join", "listen", "members", "policy", "remove",
           "set_policy", "status", "sync"]

POLICIES = ("open", "secure")
DEFAULT_PORT = 7447


@dataclass(frozen=True, slots=True)
class Member:
    """One device of the pool, as this device's node knows it."""

    name: str
    fingerprint: str
    """The device's identity: SHA-256 of its public key, 64 hex digits."""
    active: bool
    """False for a device that was removed."""
    this_device: bool
    connected: bool
    address: str
    last_seen_ms: int
    last_sync_ms: int
    """When the boards of the two devices last agreed (milliseconds since the epoch); 0 when never."""
    last_error: str = ""


@dataclass(frozen=True, slots=True)
class Pool:
    """The pool of this device: its id, its project, how devices join it and who is in it."""

    id: str
    project: str
    policy: str
    fingerprint: str
    listening: str
    """The address this device's node listens on for other devices; empty while its network is off (a pool of one)."""
    joining_open: bool
    """True while a code made by `add_device` can still be used."""
    members: tuple[Member, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Capacity:
    """What this device can give the pool: cores and slots, memory, accelerators, and what is leased now."""

    device: str
    cpu_cores: int
    cpu_slots: int
    cpu_slots_leased: int
    memory_bytes: int
    memory_available_bytes: int
    accelerators: tuple[str, ...]
    accelerator_memory_bytes: int
    unified_memory: bool
    leases_held: int
    leases_queued: int


def _member(raw: dict) -> Member:
    name = raw.get("name") or str(raw["fingerprint"])[:12]
    return Member(str(name), str(raw["fingerprint"]), raw.get("status") == "active", bool(raw.get("self")),
                  bool(raw.get("connected")), str(raw.get("addr") or ""), int(raw.get("last_seen_ms") or 0),
                  int(raw.get("last_sync_ms") or 0), str(raw.get("last_error") or ""))


def status() -> Pool:
    """The pool this device belongs to, with every member (removed ones too); starts this device's node if needed."""
    raw = _node.call("pool_status")
    return Pool(str(raw["pool"]), str(raw.get("project") or ""), str(raw["policy"]), str(raw["fingerprint"]),
                str(raw.get("listen") or ""), bool(raw.get("pairing_open")),
                tuple(_member(m) for m in raw.get("members", [])))


def members(*, include_removed: bool = False) -> list[Member]:
    """The devices of the pool, this one included; the removed ones only with ``include_removed``."""
    return [m for m in status().members if m.active or include_removed]


def policy() -> str:
    """How devices join this pool: ``open`` (automatic on the same network) or ``secure`` (with a code)."""
    return status().policy


def set_policy(new: str) -> str:
    """Set how devices join: ``open`` or ``secure``. Returns the policy now in force."""
    if new not in POLICIES:
        raise errors.Error(f"the join policy is one of {', '.join(POLICIES)}, not {new!r}")
    return str(_node.call("set_join_policy", {"policy": new}, token=_node.session().token)["policy"])


def listen(join_policy: str = "open", *, agreed: bool = False) -> Pool:
    """Turn this device's network on so other devices can join, and set how: the node listens on TCP 7447 and
    sends a signed beacon on UDP 7448 on the local network.

    This changes what the machine exposes, so it is a person's decision: ``agreed=True`` says one made it
    (the `poolhouse` first run asks; a script passes it on the person's order). Without it, `Error`.
    """
    if not agreed:
        raise errors.Error("turning the network on is a person's decision: pass agreed=True once they said yes")
    node_join.join(node_launch.default_state(), policy=join_policy)
    return status()


def add_device(ttl_s: int = 300) -> dict[str, object]:
    """Make a short-lived code for another device to join this pool with (policy ``secure``).

    Returns what the node answers: the code and how long it is valid. Show the code to the person at the other
    device; it is typed there (or passed to `join`) within ``ttl_s`` seconds.
    """
    return dict(_node.call("pair_accept", {"ttl_s": ttl_s}, token=_node.session().token))


def join(host: str, code: str, *, port: int = DEFAULT_PORT) -> dict[str, object]:
    """Join the pool of the device at ``host`` using the ``code`` its `add_device` made; this device becomes a member."""
    if not code:
        raise errors.Error("give the code the other device showed")
    return dict(_node.call("pair_start", {"host": host, "port": port, "passphrase": code}, token=_node.session().token))


def remove(device: str) -> Member:
    """Remove a member by name or fingerprint: it can no longer sync boards or take work. Not this device."""
    found = [m for m in members() if device in (m.name, m.fingerprint)]
    if len(found) != 1:
        names = ", ".join(m.name for m in members()) or "none"
        raise errors.Error(f"{'no' if not found else 'two'} member is called {device!r} (members: {names})")
    _node.call("member_revoke", {"fingerprint": found[0].fingerprint}, token=_node.session().token)
    return next(m for m in members(include_removed=True) if m.fingerprint == found[0].fingerprint)


def sync() -> None:
    """Ask the node to exchange board entries with every connected member now, rather than at its next round."""
    _node.call("sync_now", token=_node.session().token)


def capacity() -> Capacity:
    """What this device gives the pool: cores, memory, accelerators and the lease table's slots, held and queued.

    Only this device is measured here; the capacity of the other members is planned (docs/api.md, "planned").
    """
    node_name = status()
    seat = _node.session()
    table = seat.call("lease_list")
    cfg = table.get("config", {})
    held = [x for x in table.get("leases", []) if x.get("state") == "held"]
    slots = sum(r.get("count", 0) for x in held for r in x.get("resources", []) if r.get("type") == "cpu_slots")
    memory = hub.machine_memory()
    return Capacity(
        _this_name(node_name), os.cpu_count() or 1, int(cfg.get("cpu_slots", 0)), int(slots), memory.total_ram,
        memory.available_ram, tuple(g.name for g in memory.gpus), memory.vram_total or memory.gpu_limit_bytes,
        memory.unified, len(held), sum(1 for x in table.get("leases", []) if x.get("state") == "queued"))


def _this_name(pool: Pool) -> str:
    return next((m.name for m in pool.members if m.this_device), pool.fingerprint[:12])
