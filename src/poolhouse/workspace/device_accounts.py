"""Read-only access to person-enrolled device account membership."""
from poolhouse.graph.store import GraphStore, StoreNeedsUpgrade
from poolhouse.workspace.chain import held


def account_in(graph, identity):
    edges = graph.edges("device-member")
    target = next((edge["target"] for edge in edges if edge["source"] == f"agent:{identity}"), "")
    node = next((node for node in graph.nodes("device-account") if node["id"] == target), None)
    if node is None:
        return None
    return {**node["attrs"], "members": sorted(edge["source"].removeprefix("agent:")
                                             for edge in edges if edge["target"] == target)}


def account_for(ws, identity):
    """Absent persisted person bindings remain absent; the store is opened read-only, so a lookup writes nothing."""
    database = ws.base / "device-accounts.db"
    if not database.exists():
        return None
    with held(ws.base / "device-accounts.lock"):
        try:
            graph = GraphStore(database, read_only=True)
        except StoreNeedsUpgrade:
            graph = GraphStore(database)
        with graph:
            return account_in(graph, identity)
