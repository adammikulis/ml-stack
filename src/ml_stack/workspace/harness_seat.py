"""A coding harness's workspace agent identity and its lifecycle."""
from __future__ import annotations

from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path

from ml_stack.client.health import reported_models
from ml_stack.workspace import onboard, tokens
from ml_stack.workspace.identity import AGENT, Denied, Identity
from ml_stack.workspace.modelid import VERIFIED
from ml_stack.workspace.service import Workspace


@dataclass(slots=True)
class Seat:
    """How a session appears on the workspace: ``name`` is the identity, ``minted`` says whether
    the launcher created it (and so revokes it), ``parent`` the agent it acts for otherwise."""

    name: str
    parent: str = ""
    minted: bool = False
    base: Path | None = None
    issuer: Identity | None = None

    def flags(self) -> list[str]:
        """The workspace command flags this session's messages carry."""
        return ["--agent", self.parent, "--label", self.name] if self.parent else ["--agent", self.name]

    def record_model(self, alias: str, harness: str, server: str = "") -> bool:
        """Record the served alias and harness as the agent's verified model
        after checking the served alias for this private session; False on refusal."""
        if not self.minted:
            return False
        try:
            ws = Workspace(self.base)
            if server:
                who = ws.auth(tokens.load(ws.base, self.name))
                if who.id != self.name or who.role != AGENT or alias not in reported_models(server):
                    return False
                ws._record_model(self.name, alias, harness, VERIFIED)
            else:
                ws.set_model(self.name, alias, harness, verified=True)
        except (Denied, ValueError, OSError, HTTPException):
            return False
        return True

    def revoke(self) -> bool:
        """Stop the token working and delete its file; False when nothing was minted."""
        if not self.minted or self.base is None:
            return False
        ws = Workspace(self.base)
        try:
            authority = self.issuer or onboard.SETUP
            ws.registry.revoke(authority, self.name)
            ws.audit("revoke", authority.id, agent=self.name)
        except (ValueError, Denied):
            return False
        finally:
            (tokens.directory(ws.base) / self.name.replace("/", "~")).unlink(missing_ok=True)
        return True

