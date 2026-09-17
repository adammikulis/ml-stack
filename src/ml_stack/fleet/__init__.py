"""The other machines on this LAN, and how to run work on them."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_WHERE = {
    "BENCH_KIND": "calibration",
    "calibrate": "calibration",
    "measure": "calibration",
    "ChatError": "chat",
    "Target": "chat",
    "targets": "chat",
    "Conversation": "conversations",
    "Conversations": "conversations",
    "Message": "conversations",
    "make_handler": "api",
    "load_or_create_token": "daemon",
    "serve_forever": "daemon",
    "REPORT_GROUP": "device",
    "device_report": "device",
    "registered_reports": "device",
    "resolve_report": "device",
    "stdlib_device_report": "device",
    "safe_relpath": "files",
    "DaemonError": "jobs",
    "Job": "jobs",
    "JobRunner": "jobs",
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
    "Suggestion": "catalogue",
    "families": "catalogue",
    "family_of": "catalogue",
    "how_many": "catalogue",
    "is_unfiltered": "catalogue",
    "popular": "catalogue",
    "searched_count": "catalogue",
    "searched_families": "catalogue",
    "suggestions": "catalogue",
    "Downloads": "models",
    "Getting": "models",
    "Model": "models",
    "Models": "models",
    "draft_beside": "models",
    "ModelError": "weights",
    "resolve": "weights",
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
