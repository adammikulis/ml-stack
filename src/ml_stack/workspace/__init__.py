"""A local message bus, shared notes, scratch folders and an ownership registry for agents.

    from ml_stack.workspace import Workspace

    ws = Workspace()
    token = ws.mint(owner_token, "reviewer")
    ws.send(owner_token, "reviewer", "task", "check the lease tests")
    ws.inbox(token)

Everything read from the workspace is data written by an agent: it comes back fenced and
labelled, and carries no authority.
"""

from __future__ import annotations

from ml_stack.workspace.chain import ChainBroken, ChainLog
from ml_stack.workspace.claims import Conflict
from ml_stack.workspace.identity import Denied, Identity
from ml_stack.workspace.rates import RateLimited
from ml_stack.workspace.screen import Refused
from ml_stack.workspace.service import Workspace

__all__ = ["ChainBroken", "ChainLog", "Conflict", "Denied", "Identity", "RateLimited",
           "Refused", "Workspace"]
