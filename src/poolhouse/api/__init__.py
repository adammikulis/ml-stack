"""The public API of poolhouse, declared once (docs/api.md).

`import poolhouse as ph` is the facade; this package holds the wrappers that have no package of their own
(`pool`, `leases`, `models`, `remote_tests`) and `SURFACE`, the exact list of public names. `ph.board`,
`ph.serve`, `ph.client` and `ph.hub` are the packages of those names, so the public names of each are listed
here and pinned by tests/test_api.py. Nothing in poolhouse imports through `ph`; this is the one facade.
"""

from __future__ import annotations

NAMESPACES: dict[str, str] = {
    "pool": "poolhouse.api.pool",
    "board": "poolhouse.board",
    "leases": "poolhouse.api.leases",
    "test": "poolhouse.api.remote_tests",
    "models": "poolhouse.api.models",
    "serve": "poolhouse.serve",
    "client": "poolhouse.client",
    "hub": "poolhouse.hub",
}
"""Each name of ``ph`` that is a namespace, and the module that answers for it (imported on first use)."""

ERRORS = ("Error", "NotRunning", "Denied", "Conflict")
"""The exceptions the public API raises, available as ``ph.Error`` and so on."""

SURFACE: dict[str, tuple[str, ...]] = {
    "pool": ("status", "members", "policy", "set_policy", "listen", "add_device", "join", "remove", "sync", "capacity",
             "Pool", "Member", "Capacity"),
    "board": ("connect", "register", "Board", "Message", "Note", "Claim", "Agent"),
    "leases": ("view", "acquire", "wait", "renew", "release", "cpu_slots", "memory_mb", "gpu", "model_slot", "claim", "LeaseRecord"),
    "test": ("devices", "run", "Device", "Result"),
    "models": ("pull", "path", "cache_dir"),
    "serve": ("up", "down", "status", "Served"),
    "client": ("Client", "Request", "is_healthy", "ServerError", "ServerUnreachable"),
    "hub": ("discover", "ModelInfo", "fetch", "located", "hub_cache"),
}
"""The public names of each namespace. Only these are promised; everything else reachable is private."""

PLANNED = ("pool.capacities", "jobs.submit", "jobs.run", "board.tasks", "trust.standing", "pool.flags")
"""Named in docs/api.md as planned: they do not exist yet and importing them fails."""
