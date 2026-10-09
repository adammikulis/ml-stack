"""The node's pool of devices, for a machine that joined it without a cluster key.

The node (``poolhouse-node run --lan``) takes the project of the repository it runs in, listens for an open pool of that
project, joins the one it hears and makes one when it hears none (docs/sentinel-open-join.md, 3.7). Nothing here asks for
a key file: this reads who is in that pool from the running node.
"""

from __future__ import annotations

import json
from pathlib import Path

from poolhouse import node_health
from poolhouse.home import state as state_root

LAN = "--lan"


def network_args(*, project: str = "", project_dir: str = "") -> list[str]:
    """The arguments that make a node reach its pool on the LAN: ``ensure_node(state, extra=network_args())``.

    The project is that of the repository the node is started in; ``project`` names one instead and ``project_dir``
    takes the repository from another directory.
    """
    return [LAN, *(["--project", project] if project else []), *(["--project-dir", project_dir] if project_dir else [])]


def status(state: Path | None = None) -> dict | None:
    """The pool the node of ``state`` belongs to (id, project, policy, members), or None when no node answers."""
    try:
        return node_health.call(state or state_root("node"), "pool_status")
    except (OSError, ValueError):
        return None


def rows(pool: dict) -> list[dict]:
    """One row per active member of a pool status: name, fingerprint, address, whether it is this machine or connected."""
    return [{"name": m.get("name") or m["fingerprint"][:12], "fingerprint": m["fingerprint"], "addr": m.get("addr") or "",
             "state": "this machine" if m.get("self") else "connected" if m.get("connected") else "known"}
            for m in pool.get("members", []) if m.get("status") == "active"]


def render(pool: dict, *, as_json: bool) -> str:
    """The text `poolhouse-peers ls` prints for a pool."""
    if as_json:
        return json.dumps({"pool": pool.get("pool"), "project": pool.get("project"), "policy": pool.get("policy"),
                           "members": rows(pool)}, indent=2)
    lines = [f"pool {pool.get('pool')}  project {pool.get('project') or '-'}  join policy {pool.get('policy')}",
             f"{'NAME':<20} {'STATE':<13} {'ADDRESS':<24} FINGERPRINT"]
    lines += [f"{r['name']:<20} {r['state']:<13} {r['addr']:<24} {r['fingerprint'][:16]}" for r in rows(pool)]
    return "\n".join(lines)
