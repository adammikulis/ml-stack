"""Read-only access to person-enrolled device account membership."""
from ml_stack.graph.store import GraphStore


def account_in(graph, identity):
    edges = graph.edges("device-member")
    target = next((edge["target"] for edge in edges if edge["source"] == f"agent:{identity}"), "")
    node = next((node for node in graph.nodes("device-account") if node["id"] == target), None)
    if node is None:
        return None
    return {**node["attrs"], "members": sorted(edge["source"].removeprefix("agent:")
                                             for edge in edges if edge["target"] == target)}


def account_for(ws, identity):
    """Absent persisted person bindings remain absent."""
    with GraphStore(ws.base / "device-accounts.db") as graph:
        return account_in(graph, identity)
