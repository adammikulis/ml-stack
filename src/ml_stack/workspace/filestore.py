"""The files posted to the workspace: encrypted content kept once by hash, and a graph of
which agent posted what where, sealed as one encrypted snapshot (the memory store's pattern)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import files, keystore, lock
from ml_stack.graph.search import lexical, rrf_scored
from ml_stack.graph.store import GraphStore

try:
    from cryptography.exceptions import InvalidTag
except ImportError:
    InvalidTag = ValueError  # type: ignore[assignment,misc]

__all__ = ["PURPOSE", "SCHEMA_VERSION", "FileStore", "Unavailable", "handle_of"]

PURPOSE = "workspace-files"
SCHEMA_VERSION = 1
HANDLE = 12
_MAGIC = b"MLF1"
_AAD = b"ml-stack/workspace-files/v1|"
FAILURES = (OSError, RuntimeError, ValueError, TypeError, KeyError, InvalidTag,
            keystore.KeystoreError)
DB_BYTES = 1 << 28
SEARCH_POOL = 500


class Unavailable(RuntimeError):
    """The files store cannot be read or written, so nothing is posted or read through it."""


def handle_of(data: bytes) -> tuple[str, str]:
    """``(sha256 hex, handle)`` for ``data``: the handle is the first 12 hex characters."""
    digest = hashlib.sha256(data).hexdigest()
    return digest, digest[:HANDLE]


def _migrate(snap: dict[str, Any]) -> dict[str, Any]:
    version = snap.get("schema_version")
    if version != SCHEMA_VERSION:
        raise Unavailable(f"the files graph is schema {version}; this build reads {SCHEMA_VERSION}")
    return snap


class FileStore:
    """Content blobs under ``files/blobs`` and the graph in ``files/graph.enc``, both
    AES-256-GCM under the ``workspace-files`` subkey. Nothing opens the keystore until a file
    that exists has to be read or one is written; a key that cannot be had is `Unavailable`."""

    def __init__(self, base: Path, *, key: Callable[[], bytes] | None = None) -> None:
        self.directory = base / "files"
        self.blobs = self.directory / "blobs"
        self.path = self.directory / "graph.enc"
        self.key_from = key or self._subkey
        self._key: bytes | None = None

    def _subkey(self) -> bytes:
        return keystore.default().salted_subkey(PURPOSE, self.directory)

    def _cipher(self) -> Any:
        try:
            if self._key is None:
                self._key = self.key_from()
            return keystore.aead(self._key)
        except FAILURES as exc:
            raise Unavailable(f"the files key cannot be had ({type(exc).__name__}); nothing is "
                              f"read or written") from exc

    def _seal(self, plain: bytes, aad: str) -> bytes:
        nonce = os.urandom(12)
        return _MAGIC + nonce + self._cipher().encrypt(nonce, plain, _AAD + aad.encode())

    def _open(self, blob: bytes, aad: str) -> bytes:
        if not blob.startswith(_MAGIC):
            raise Unavailable("not a workspace files blob")
        try:
            return bytes(self._cipher().decrypt(blob[4:16], blob[16:], _AAD + aad.encode()))
        except InvalidTag as exc:
            raise Unavailable("a workspace file failed its integrity check") from exc

    def _private(self) -> None:
        for d in (self.directory, self.blobs):
            d.mkdir(parents=True, exist_ok=True, mode=0o700)
            d.chmod(0o700)

    # -- content ---------------------------------------------------------------------
    def _blob(self, digest: str) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("a content hash is 64 hex characters")
        return self.blobs / f"{digest}.enc"

    def put(self, data: bytes) -> str:
        """Store ``data`` once under its hash (sealed); returns the hash."""
        digest, _ = handle_of(data)
        self._cipher()
        self._private()
        path = self._blob(digest)
        if not path.exists():
            blob = self._seal(data, digest)
            with files.writing(path) as tmp:
                tmp.write_bytes(blob)
                tmp.chmod(0o600)
        return digest

    def get(self, digest: str) -> bytes:
        """The content stored under ``digest``; `Unavailable` when absent, shredded or sealed
        under another key."""
        try:
            blob = self._blob(digest).read_bytes()
        except FileNotFoundError as exc:
            raise Unavailable("the content is not stored") from exc
        except OSError as exc:
            raise Unavailable(f"the content cannot be read ({type(exc).__name__})") from exc
        data = self._open(blob, digest)
        if hashlib.sha256(data).hexdigest() != digest:
            raise Unavailable("a workspace file does not match its hash")
        return data

    def shred(self, digest: str) -> bool:
        """Overwrite and remove the content under ``digest``; whether there was any."""
        path = self._blob(digest)
        try:
            size = path.stat().st_size
            with path.open("r+b") as handle:
                handle.write(b"\0" * size)
                handle.flush()
                os.fsync(handle.fileno())
        except FileNotFoundError:
            return False
        path.unlink()
        return True

    # -- the graph -------------------------------------------------------------------
    def _snapshot(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "nodes": [], "edges": []}
        try:
            blob = self.path.read_bytes()
        except OSError as exc:
            raise Unavailable(f"the files graph cannot be read ({type(exc).__name__})") from exc
        snap = json.loads(self._open(blob, "graph"))
        if not isinstance(snap, dict):
            raise Unavailable("the files graph is not a snapshot")
        return _migrate(snap)

    @staticmethod
    def _build(snap: dict[str, Any]) -> GraphStore:
        g = GraphStore(":memory:", max_db_size=DB_BYTES)
        g.write({"nodes": snap["nodes"], "edges": snap["edges"]})
        return g

    @contextlib.contextmanager
    def reading(self) -> Any:
        """A read-only in-memory graph of the snapshot; an empty one when nothing is stored."""
        if not self.path.exists():
            g = self._build({"nodes": [], "edges": []})
        else:
            g = self._build(self._snapshot())
        try:
            yield g
        finally:
            g.close()

    def edit(self, change: Callable[[GraphStore], Any]) -> Any:
        """Apply ``change`` to the graph under the store lock and seal the result."""
        self._cipher()
        self._private()
        with lock.only_one(self.directory / "store.lock", timeout=10, announce=lambda _m: None):
            g = self._build(self._snapshot())
            try:
                out = change(g)
                snap = {"schema_version": SCHEMA_VERSION, "nodes": g.nodes(), "edges": g.edges()}
                plain = json.dumps(snap, sort_keys=True, separators=(",", ":")).encode()
                blob = self._seal(plain, "graph")
            finally:
                g.close()
            with files.writing(self.path) as tmp:
                tmp.write_bytes(blob)
                tmp.chmod(0o600)
            return out

    # -- questions the graph answers -------------------------------------------------
    def node(self, ident: str) -> dict[str, Any] | None:
        """The file node ``ident`` (``file:<handle>``) with its attributes."""
        with self.reading() as g:
            found = [n for n in g.nodes("file") if n["id"] == ident]
        return found[0] if found else None

    def nodes(self) -> list[dict[str, Any]]:
        """Every file node, ordered by id."""
        with self.reading() as g:
            return sorted(g.nodes("file"), key=lambda n: n["id"])

    def linked(self, ident: str, rel: str) -> list[str]:
        """The labels of the nodes ``ident`` points at by ``rel``, sorted."""
        with self.reading() as g:
            label = {n["id"]: n["label"] for n in g.nodes()}
            return sorted(label[e["target"]] for e in g.edges(rel)
                          if e["source"] == ident and e["target"] in label)

    def where(self, *, by: str = "", project: str = "", board: str = "",
              derived_from: str = "") -> set[str]:
        """File node ids matching every given condition: posted by ``by``, in ``project`` (key
        or name), in ``board`` (a board name, a DM as ``dm:A|B``), derived from a file."""
        with self.reading() as g:
            nodes = {n["id"]: n for n in g.nodes()}
            edges = g.edges()
        want = {"posted_by": f"agent:{by}" if by else "", "in_board": f"board:{board}" if board else "",
                "derived_from": f"file:{derived_from}" if derived_from else ""}
        found = {i for i, n in nodes.items() if n["kind"] == "file"}
        for rel, target in want.items():
            if target:
                found &= {e["source"] for e in edges if e["rel"] == rel and e["target"] == target}
        if project:
            low = project.casefold()
            hits = {i for i, n in nodes.items() if n["kind"] == "project"
                    and low in (str(n["attrs"].get("key", "")).casefold(), n["label"].casefold())}
            found &= {e["source"] for e in edges if e["rel"] == "in_project" and e["target"] in hits}
        return found

    def search(self, query: str, within: set[str]) -> list[tuple[str, float]]:
        """``(node id, score)`` for files in ``within`` that match the words of ``query`` in
        their name, note or indexed text, best first, ties by id."""
        words = [w for w in " ".join(query.split()).casefold().split(" ") if w][:6]
        if not words:
            return []
        with self.reading() as g:
            graph = {"nodes": [n for n in g.nodes("file") if n["id"] in within], "edges": []}
        rankings = [lexical(graph, " ".join(words), limit=SEARCH_POOL)]
        rankings += [lexical(graph, w, limit=SEARCH_POOL) for w in words]
        return rrf_scored(*rankings, limit=SEARCH_POOL)
