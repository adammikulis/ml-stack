"""A GraphStore held in memory and kept on disk as one sealed file, under the memory vault's
AES-256-GCM seal and a key of its own in the OS keystore."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import board_names, files, home, lock
from ml_stack.graph.store import GraphStore
from ml_stack.memory import vault

__all__ = ["PROFILE", "SCHEMA_VERSION", "SealedGraph", "Tampered"]

SCHEMA_VERSION = 1
PROFILE = "reputation"


class Tampered(RuntimeError):
    """The file failed authentication and no earlier copy held."""


class SealedGraph:
    """The per-user reputation graph. ``path`` is the sealed file (tests give one)."""

    def __init__(self, path: Path | None = None, *, user: str | None = None,
                 keys: vault.Keys | None = None) -> None:
        self.user = user if user is not None else vault.identity()
        uid = hashlib.sha256(self.user.encode()).hexdigest()[:12]
        self.path = Path(path) if path else home.state("reputation", f"u-{uid}", "graph.enc")
        self.prev = self.path.with_name(self.path.name + ".prev")
        self.owner = f"{self.user}|{PROFILE}"
        self.keys = keys or vault.default_keys(self.user, PROFILE, str(self.path.parent.resolve()),
                                                       (self.path, self.prev), purpose="reputation")
        self.status, self.why = "fresh", ""
        self._g: GraphStore | None = None
        self._stamp: tuple[str, str] | None = None
        self._salt = os.urandom(vault.SALT)

    def close(self) -> None:
        """Release the in-memory graph."""
        if self._g is not None:
            self._g.close()
        self._g, self._stamp = None, None

    @staticmethod
    def _bytes(path: Path) -> bytes | None:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def _decrypt(self, blob: bytes) -> dict[str, Any] | None:
        try:
            mode, salt = vault.header(blob)
            if mode != self.keys.mode:
                raise vault.KeyUnavailable(f"this store is kept under a {mode}, not a {self.keys.mode}")
            found = self.keys.keys(salt)
            if not found:
                raise vault.KeyUnavailable("the OS keystore holds no key for this store")
            plain, _ = vault.open_blob(blob, found, owner=self.owner)
            snap = json.loads(plain)
        except (vault.BadSeal, ValueError):
            return None
        if not isinstance(snap, dict) or snap.get("schema_version") != SCHEMA_VERSION:
            return None
        self._salt = salt
        return snap

    def graph(self) -> GraphStore | None:
        """The graph for what is on disk now; None when the file is locked or tampered with."""
        main, prev = self._bytes(self.path), self._bytes(self.prev)
        stamp = (hashlib.sha256(main or b"").hexdigest(), hashlib.sha256(prev or b"").hexdigest())
        if stamp == self._stamp and self._g is not None:
            return self._g
        self.close()
        try:
            snap: dict[str, Any] | None = {}
            self.status, self.why = "fresh", ""
            if main is not None or prev is not None:
                snap, self.status = (self._decrypt(main) if main else None), "ok"
                if snap is None and prev is not None:
                    snap, self.status = self._decrypt(prev), "recovered"
        except vault.KeyUnavailable as exc:
            self.status, self.why = "locked", str(exc)
            return None
        if snap is None:
            self.status, self.why = "tampered", f"{self.path} failed its integrity check"
            return None
        g = GraphStore(":memory:")
        g.write({"nodes": snap.get("nodes", []), "edges": snap.get("edges", [])})
        g.put_doc("rep", snap.get("rep", {}))
        if not board_names.marker(self.path).exists():
            board_names.rewrite_graph(g)
        self._g, self._stamp = g, stamp
        return g

    def edit(self, change: Callable[[GraphStore], Any], *, keep_previous: bool = True) -> Any:
        """Run ``change`` on the graph under the file lock and seal the result."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with lock.only_one(self.path.parent / "store.lock", timeout=10, announce=lambda _: None):
            g = self.graph()
            if g is None:
                if self.status == "locked":
                    raise vault.KeyUnavailable(self.why)
                raise Tampered(f"{self.why}; restore an authenticated backup before changing this store")
            out = change(g)
            self._save(g, keep_previous=keep_previous)
            if not keep_previous:
                self.prev.unlink(missing_ok=True)
                self._stamp = None
            return out

    def _save(self, g: GraphStore, *, keep_previous: bool = True) -> None:
        self.path.parent.chmod(0o700)
        key = self.keys.keys(self._salt, create=True)[0]
        snap = {"schema_version": SCHEMA_VERSION, "nodes": g.nodes(), "edges": g.edges(),
                "rep": g.get_doc("rep", {})}
        plain = json.dumps(snap, sort_keys=True, separators=(",", ":")).encode()
        blob = vault.seal_blob(plain, key, mode=self.keys.mode, salt=self._salt, owner=self.owner)
        with files.writing(self.path) as tmp:
            tmp.write_bytes(blob)
            if self.status == "ok" and keep_previous:
                files.promote(self.path, self.prev)
        main, prev = self._bytes(self.path), self._bytes(self.prev)
        self._stamp = (hashlib.sha256(main or b"").hexdigest(), hashlib.sha256(prev or b"").hexdigest())
        self.status = "ok"
        board_names.marker(self.path).write_text(str(board_names.VERSION))

    def delete(self) -> None:
        """Remove the sealed file and its previous copy."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with lock.only_one(self.path.parent / "store.lock", timeout=10, announce=lambda _: None):
            for each in (self.path, self.prev, board_names.marker(self.path)):
                each.unlink(missing_ok=True)
        self.close()
