"""A coding harness's workspace agent identity and its lifecycle."""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from types import SimpleNamespace

from ml_stack.client.health import reported_models
from ml_stack.workspace import harness_remote, onboard, profilehook, tokens, worktree_lifecycle
from ml_stack.workspace.identity import AGENT, Denied, Identity
from ml_stack.workspace.remote import RemoteWorkspace
from ml_stack.workspace.service import Workspace

PROFILE_TIMEOUT = 2.0

@dataclass(slots=True)
class Seat:
    """How a session appears on the workspace: ``name`` is the identity, ``minted`` says whether
    the launcher created it (and so revokes it), ``parent`` the agent it acts for otherwise."""

    name: str
    parent: str = ""
    minted: bool = False
    base: Path | None = None
    issuer: Identity | None = None
    managed_inbox: bool = False
    persistent: bool = False
    record_claim: Callable[[str, str], object] | None = None
    remote: RemoteWorkspace | None = None
    lifecycle_base: Path | None = None

    def flags(self) -> list[str]:
        """The workspace command flags this session's messages carry."""
        return ["--agent", self.parent, "--label", self.name] if self.parent else ["--agent", self.name]

    def record_model(self, alias: str, harness: str, server: str = "") -> bool:
        """Record a connected agent's model as claimed; False on an inactive seat or refusal."""
        if not self.minted and not self.persistent:
            return False
        try:
            if self.record_claim is not None:
                if server and alias not in reported_models(server):
                    return False
                self.record_claim(Path(alias).name, harness)
                return True
            if self.remote is not None:
                token = self.remote.token(agent=self.name)
                who = self.remote.call("whoami", token)
                if who.get("id") != self.name or who.get("role") != AGENT:
                    return False
                if server and alias not in reported_models(server):
                    return False
                self.remote.call("claim_model", token, Path(alias).name, harness)
                return True
            ws = Workspace(self.base)
            if server:
                who = ws.auth(tokens.load(ws.base, self.name))
                if who.id != self.name or who.role != AGENT or alias not in reported_models(server):
                    return False
                ws.claim_model(tokens.load(ws.base, self.name), Path(alias).name, harness)
            else:
                ws.claim_model(tokens.load(ws.base, self.name), alias, harness)
        except (Denied, ValueError, OSError, HTTPException):
            return False
        return True

    def record_execution(self, args, harness: str, served: tuple, root: Path) -> bool:
        """Start a bounded best-effort observation through the authenticated seat."""
        if not self.minted and not self.persistent:
            return False
        snapshot = SimpleNamespace(**{key: getattr(args, key, None) for key in
                                      ('model', 'ctx', 'effort', 'max_output_tokens', 'max_turns')})
        completed = threading.Event()
        result = []

        def observe():
            try:
                document = profilehook.launched(snapshot, harness, served)
                result.append(self._record_execution(document, root))
            except (OSError, ValueError, TypeError, RuntimeError):
                result.append(False)
            finally:
                completed.set()

        threading.Thread(target=observe, name='execution-profile-observer', daemon=True).start()
        return completed.wait(PROFILE_TIMEOUT) and bool(result and result[0])

    def _record_execution(self, document, root):
        try:
            if self.remote is not None:
                token = self.remote.token(agent=self.name)
                if self.remote.call('whoami', token).get('id') != self.name:
                    return False
                self.remote.call('record_execution_profile', token, document)
            elif self.base is not None and self.record_claim is None:
                ws = Workspace(self.base)
                token = tokens.load(ws.base, self.name)
                if ws.auth(token).id != self.name:
                    return False
                ws.record_execution_profile(token, document)
            else:
                profilehook.send(document, self.name, root, self.base)
        except (Denied, ValueError, OSError, HTTPException, RuntimeError, TypeError):
            return False
        return True

    def pending_worktrees(self) -> list[dict]:
        """Return this session identity's unfinished coding scopes."""
        return worktree_lifecycle.pending(self.base, self.parent or self.name) if self.base else []

    def require_clean(self) -> None:
        """Refuse a successful session exit with unfinished attributed checkouts."""
        if self.base:
            try:
                worktree_lifecycle.require_clean(self.base, self.parent or self.name,
                                                self.name if self.parent else '')
            except Denied as error:
                raise ValueError(str(error)) from error

    def revoke(self) -> bool:
        """Stop the token working and delete its file; False when nothing was minted."""
        if not self.minted or self.base is None:
            return False
        if self.remote is not None:
            try:
                snapshot = harness_remote.revocation_snapshot(self.remote, self.name)
                self.remote.self_revoke(self.name)
                harness_remote.release_revoked(self.remote, self.name, snapshot)
                return True
            except (ValueError, Denied, OSError):
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
