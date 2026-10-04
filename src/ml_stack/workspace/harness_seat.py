"""A coding harness's workspace agent identity and its lifecycle."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml_stack.workspace import onboard, tokens
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.service import Workspace


@dataclass(slots=True)
class Seat:
    """How a session appears on the workspace: ``name`` is the identity, ``minted`` says whether
    the launcher created it (and so revokes it), ``parent`` the agent it acts for otherwise."""

    name: str
    parent: str = ""
    minted: bool = False
    base: Path | None = None

    def flags(self) -> list[str]:
        """The workspace command flags this session's messages carry."""
        return ["--agent", self.parent, "--label", self.name] if self.parent else ["--agent", self.name]

    def record_model(self, alias: str, harness: str) -> bool:
        """Record the served alias and harness as the agent's verified model
        (``Workspace.set_model``, person-only); False for an unminted seat or a refusal."""
        if not self.minted:
            return False
        try:
            Workspace(self.base).set_model(self.name, alias, harness, verified=True)
        except (Denied, ValueError, PermissionError):
            return False
        return True

    def revoke(self) -> bool:
        """Stop the token working and delete its file; False when nothing was minted."""
        if not self.minted or self.base is None:
            return False
        ws = Workspace(self.base)
        try:
            ws.registry.revoke(onboard.SETUP, self.name)
            ws.audit("revoke", onboard.SETUP.id, agent=self.name)
        except (ValueError, Denied):
            return False
        finally:
            (tokens.directory(ws.base) / self.name.replace("/", "~")).unlink(missing_ok=True)
        return True

