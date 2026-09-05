"""The other machines on this LAN, and how to run work on them."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_WHERE = {
    "BENCH_KIND": "bench",
    "calibrate": "bench",
    "measure": "bench",
    "ChatError": "chat",
    "Target": "chat",
    "targets": "chat",
    "Conversation": "conversations",
    "Conversations": "conversations",
    "Message": "conversations",
    "DaemonError": "daemon",
    "Job": "daemon",
    "JobRunner": "daemon",
    "REPORT_GROUP": "daemon",
    "device_report": "daemon",
    "load_or_create_token": "daemon",
    "make_handler": "daemon",
    "registered_reports": "daemon",
    "resolve_report": "daemon",
    "safe_relpath": "daemon",
    "serve_forever": "daemon",
    "stdlib_device_report": "daemon",
    "Advertiser": "discovery",
    "Beacon": "discovery",
    "DEFAULT_CLUSTER": "discovery",
    "DiscoveryError": "discovery",
    "MIN_PASSPHRASE": "discovery",
    "check_passphrase": "discovery",
    "cluster_group": "discovery",
    "create_cluster_key": "discovery",
    "derive_token": "discovery",
    "discover": "discovery",
    "group_path": "discovery",
    "in_cluster": "discovery",
    "join_cluster": "discovery",
    "key_from_passphrase": "discovery",
    "key_path": "discovery",
    "load_cluster_key": "discovery",
    "Downloads": "models",
    "Getting": "models",
    "Model": "models",
    "ModelError": "models",
    "Models": "models",
    "Candidate": "pool",
    "Requires": "pool",
    "Score": "pool",
    "candidates": "pool",
    "choose": "pool",
    "eligible": "pool",
    "soonest": "pool",
    "Rates": "rates",
    "Peer": "remote",
    "PeerError": "remote",
    "sha256_file": "remote",
    "Endpoint": "serving",
    "Served": "serving",
    "Serving": "serving",
    "discover_serving": "serving",
    "Placement": "work",
    "Unit": "work",
    "run": "work",
}

__all__ = sorted(_WHERE)


def __getattr__(name: str) -> Any:
    """The named export, loading the submodule that defines it."""
    where = _WHERE.get(name)
    if where is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{where}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
