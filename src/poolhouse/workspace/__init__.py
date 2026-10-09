"""A local message bus, shared notes, scratch folders and an ownership registry for agents.

    from poolhouse.workspace import Workspace

    ws = Workspace()
    token = ws.mint(owner_token, "reviewer")
    ws.send(owner_token, "reviewer", "task", "check the lease tests")
    ws.inbox(token)

Everything read from the workspace is data written by an agent: it comes back fenced and
labelled, and carries no authority.
"""

from __future__ import annotations

from poolhouse.workspace.chain import ChainBroken, ChainLog
from poolhouse.workspace.claims import Conflict
from poolhouse.workspace.identity import Denied, Identity
from poolhouse.workspace.rates import RateLimited
from poolhouse.workspace.screen import Refused
from poolhouse.workspace.service import Workspace

__all__ = ["ChainBroken", "ChainLog", "Conflict", "Denied", "Identity", "RateLimited",
           "Refused", "Workspace"]
