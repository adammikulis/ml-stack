"""Opening a page, pressing what is on it, and reading what came back.

`ops` walks the fleet interface and the graph page; `cli` is ``poolhouse-walk``.
"""

from poolhouse.walk.ops import FLEET, GRAPH, PAGES, Stop, Walk, WalkFailed, fleet, graph, walk

__all__ = ["FLEET", "GRAPH", "PAGES", "Stop", "Walk", "WalkFailed", "fleet", "graph",
           "walk"]
