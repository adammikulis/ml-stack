"""The facts as a graph in ml-stack's ``GraphStore``, held in memory and kept on disk only as
one encrypted, authenticated file per user."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack import files, home, lock
from ml_stack.graph.search import hybrid
from ml_stack.graph.store import GraphStore
from ml_stack.memory import vault
from ml_stack.memory.entities import Entity, parse
from ml_stack.memory.facts import (
    BUILD_BOUND,
    KINDS,
    MAX_FACTS,
    SOURCES,
    STALE_DAYS,
    Fact,
    Refused,
    check,
)
from ml_stack.memory.migrate import import_v1
from ml_stack.memory.project import Project
from ml_stack.sentinel.human import protect
from ml_stack.serve.binary import managed_current

__all__ = ["FACT_LINKS", "MAX_FACTS", "MAX_LINKS", "RELATIONS", "SCHEMA_VERSION", "Setup", "Store", "Tampered", "View",
           "current_scope", "stale_reason"]

SCHEMA_VERSION = 2
MAX_LINKS = 8
CLOSE = 0.65
"""Similarity (cosine 0.3) a hit found only by meaning must reach."""
DAY = 86400.0
RELATIONS = ("about", "learned_on", "supersedes", "contradicts", "related")
FACT_LINKS = ("supersedes", "contradicts", "related")
Vector = Sequence[float]


class Tampered(RuntimeError):
    """The store file failed authentication and no earlier copy held."""


def current_scope() -> dict[str, str]:
    """The machine and the managed llama.cpp build in use now."""
    link = managed_current()
    build = link.resolve().name if link.exists() else ""
    return {"machine": platform.node(), "build": build}


def stale_reason(fact: Fact, now: float, scope: dict[str, str]) -> str:
    """Why ``fact`` should be re-checked before it is relied on, or an empty string."""
    if fact.kind not in BUILD_BOUND:
        return ""
    then, here = fact.scope.get("build", ""), scope.get("build", "")
    if then and here and then != here:
        return f"learned on build {then}; build {here} is in use"
    days = (now - fact.last_confirmed) / DAY
    if days > STALE_DAYS[fact.kind]:
        return f"last confirmed {days:.0f} days ago"
    return ""


def node_id(ident: str) -> str:
    return f"fact:{ident}"


@dataclass(slots=True)
class View:
    """Every fact read once, with the entities each hangs on and the links between facts."""

    facts: list[Fact] = field(default_factory=list)
    entity_name: dict[str, str] = field(default_factory=dict)
    members: dict[str, list[str]] = field(default_factory=dict)
    ties: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    def get(self, ident: str) -> Fact | None:
        return next((f for f in self.facts if f.id == ident), None)

    def around(self, fact: Fact, limit: int = 2) -> list[Fact]:
        """Current facts that share an entity with ``fact`` or are linked to it, most shared first."""
        shared: dict[str, int] = {}
        for row in self.entity_ids(fact.id):
            for other in self.members.get(row, ()):
                if other != fact.id:
                    shared[other] = shared.get(other, 0) + 1
        for _, other in self.ties.get(fact.id, ()):
            shared[other] = shared.get(other, 0) + 1
        live = {f.id: f for f in self.facts if f.state == "current"}
        order = sorted((i for i in shared if i in live), key=lambda i: (-shared[i], -live[i].last_confirmed, i))
        return [live[i] for i in order[:limit]]

    def entity_ids(self, ident: str) -> list[str]:
        return [e for e, ids in self.members.items() if ident in ids]


@dataclass(slots=True)
class Setup:
    """Whose store it is and how it is keyed. ``user`` defaults to this process's OS account and
    ``profile`` (or ``$ML_STACK_MEMORY_PROFILE``) names one of several memories for it; neither
    is ever taken from model or request text. ``keys`` defaults to the OS keystore, ``legacy``
    is a version 1 file to import, ``embed`` turns a fact into the vector stored with it and
    ``project`` makes the store that project's (a separate sealed file under the same key)."""

    user: str | None = None
    profile: str | None = None
    keys: vault.Keys | None = None
    legacy: Path | None = None
    embed: Callable[[str], Sequence[float]] | None = None
    project: Project | None = None


@dataclass(slots=True)
class Draft:
    """A fact that passed its checks and is about to be written."""

    text: str
    kind: str
    source: str
    ents: list[Entity]
    now: float
    scope: dict[str, str]
    vector: Vector | None = None


class Store:
    """The facts of one user, or of one project of that user, encrypted at rest. ``path`` is the
    file (tests give one; the default is under the state directory, per user, profile and
    project). ``realm`` is ``user`` or ``project``."""

    def __init__(self, path: Path | None = None, *, clock: Callable[[], float] = time.time,
                 scope: Callable[[], dict[str, str]] = current_scope, setup: Setup | None = None) -> None:
        setup = setup or Setup()
        self.embed = setup.embed
        self.user = setup.user if setup.user is not None else vault.identity()
        self.profile = vault.valid_profile(setup.profile or os.environ.get(vault.PROFILE_ENV) or "default")
        uid = hashlib.sha256(self.user.encode()).hexdigest()[:12]
        self.project = setup.project
        self.realm = "project" if self.project else "user"
        base = home.state("memory", f"u-{uid}", self.profile)
        if path:
            self.path = Path(path)
        elif self.project:
            self.path = base / "projects" / self.project.key / "graph.enc"
        else:
            self.path = base / "graph.enc"
        self.prev = self.path.with_name(self.path.name + ".prev")
        self.clock, self.scope = clock, scope
        self.owner = f"{self.user}|{self.profile}" + (f"|project:{self.project.key}" if self.project else "")
        keydir = base if path is None else self.path.parent
        self.keys = setup.keys or vault.default_keys(self.user, self.profile, str(keydir.resolve()))
        self._g: GraphStore | None = None
        self._stamp: tuple[str, str] | None = None
        self._status, self._why, self._salt = "fresh", "", os.urandom(vault.SALT)
        self._view: View | None = None
        self.saved_project: dict[str, str] = {}
        self._next_key: bytes | None = None
        protect(base.parent if path is None else self.path.parent)
        old = setup.legacy or (home.state("memory", "facts.json") if path is None and not self.project else None)
        if old is not None and old.exists() and not self.path.exists():
            with contextlib.suppress(vault.KeyUnavailable):
                import_v1(self, old)

    # -- loading --------------------------------------------------------------------
    def close(self) -> None:
        """Release the in-memory graph."""
        if self._g is not None:
            self._g.close()
        self._g, self._stamp, self._view = None, None, None

    def _decrypt(self, blob: bytes) -> dict[str, Any]:
        mode, salt = vault.header(blob)
        if mode != self.keys.mode:
            raise vault.KeyUnavailable(f"this store is kept under a {mode}, not a {self.keys.mode}")
        found = self.keys.keys(salt)
        if not found:
            raise vault.KeyUnavailable("the OS keystore holds no key for this store")
        plain, _ = vault.open_blob(blob, found, owner=self.owner)
        snap = json.loads(plain)
        if not isinstance(snap, dict) or snap.get("schema_version") != SCHEMA_VERSION:
            raise vault.BadSeal("unknown schema")
        self._salt = salt
        self.saved_project = dict(snap.get("project") or {})
        return snap

    def _try(self, blob: bytes | None) -> dict[str, Any] | None:
        if blob is None:
            return None
        try:
            return self._decrypt(blob)
        except (vault.BadSeal, ValueError):
            return None

    @staticmethod
    def _bytes(path: Path) -> bytes | None:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def _sync(self) -> GraphStore | None:
        """The graph for what is on disk now, rebuilt only when the file changed."""
        main, prev = self._bytes(self.path), self._bytes(self.prev)
        stamp = (hashlib.sha256(main or b"").hexdigest(), hashlib.sha256(prev or b"").hexdigest())
        if stamp == self._stamp and self._g is not None:
            return self._g
        self.close()
        try:
            if main is None and prev is None:
                snap, status = {}, "fresh"
            else:
                snap, status = self._try(main), "ok"
                if snap is None:
                    snap, status = self._try(prev), "recovered"
        except vault.KeyUnavailable as exc:
            self._status, self._why = "locked", str(exc)
            return None
        if snap is None:
            self._status, self._why = "tampered", f"{self.path} failed its integrity check"
            return None
        self._status, self._why = status, ""
        self._g, self._stamp = self._build(snap), stamp
        return self._g

    @staticmethod
    def _build(snap: dict[str, Any]) -> GraphStore:
        g = GraphStore(":memory:")
        g.write({"nodes": snap.get("nodes", []), "edges": snap.get("edges", [])})
        g.put_doc("memory", {"next": int(snap.get("next", 1))})
        for ident, vector in (snap.get("embeddings") or {}).items():
            try:
                g.set_embedding(ident, vector)
            except RuntimeError:
                break
        return g

    @property
    def status(self) -> str:
        """``fresh``, ``ok``, ``recovered`` (the previous copy was used), ``tampered`` or
        ``locked`` (the key is not available)."""
        self._sync()
        return self._status

    @property
    def why(self) -> str:
        """What went wrong when ``status`` is ``tampered`` or ``locked``."""
        self._sync()
        return self._why

    def problems(self) -> list[tuple[str, str]]:
        """``(scope, status)`` when this store was not read (tampered or locked), else empty."""
        return [(self.realm, self.status)] if self.status in ("tampered", "locked") else []

    def view(self) -> View:
        """Every fact and link, empty when the store is tampered with or locked."""
        g = self._sync()
        if g is None:
            return View()
        if self._view is None:
            self._view = self._read(g)
        return self._view

    @staticmethod
    def _read(g: GraphStore) -> View:
        nodes = {n["id"]: n for n in g.nodes()}
        view = View()
        about: dict[str, list[str]] = {}
        ties: dict[str, list[tuple[str, str]]] = {}
        for edge in g.edges():
            a, b, rel = edge["source"], edge["target"], edge["rel"]
            if rel in ("about", "learned_on") and b in nodes:
                about.setdefault(a, []).append(b)
                view.members.setdefault(b, []).append(nodes[a]["attrs"]["ident"])
            elif rel in FACT_LINKS and a in nodes and b in nodes:
                x, y = nodes[a]["attrs"]["ident"], nodes[b]["attrs"]["ident"]
                ties.setdefault(x, []).append((rel, y))
                ties.setdefault(y, []).append((f"{rel}:by", x))
        view.ties = ties
        for ident, n in nodes.items():
            if n["kind"] == "fact":
                view.facts.append(Store._fact(n, [nodes[e] for e in about.get(ident, []) if e in nodes], ties))
            else:
                view.entity_name[ident] = n["label"]
        view.facts.sort(key=lambda f: f.id)
        return view

    @staticmethod
    def _fact(node: dict[str, Any], ents: list[dict[str, Any]], ties: dict[str, list[tuple[str, str]]]) -> Fact:
        a = node["attrs"]
        links = [(f"superseded by {o}" if r == "supersedes:by" else
                  f"{r.replace(':by', '')}{' (by ' + o + ')' if r.endswith(':by') else ' ' + o}")
                 for r, o in sorted(ties.get(a["ident"], ()))]
        return Fact.from_json({"id": a["ident"], "text": node["label"], "kind": a["fkind"],
                               "source": a["source"], "created": a["created"],
                               "last_confirmed": a["last_confirmed"],
                               "confirm_count": a["confirm_count"], "scope": a.get("scope", {}),
                               "entities": sorted(f"{e['kind']}:{e['label']}" for e in ents),
                               "state": a.get("state", "current"), "links": links})

    # -- reading --------------------------------------------------------------------
    def facts(self) -> list[Fact]:
        """Every fact, oldest first; empty when the store is tampered with or locked."""
        return list(self.view().facts)

    def get(self, ident: str) -> Fact | None:
        return self.view().get(ident)

    def stale(self, fact: Fact) -> str:
        """Why ``fact`` needs re-checking now, or an empty string."""
        return stale_reason(fact, self.clock(), self.scope())

    def hits(self, query: str, vector: Vector | None = None, limit: int = 12) -> list[str]:
        """Fact and entity ids for ``query`` by the graph's hybrid search (words, characters,
        meaning), best first."""
        g = self._sync()
        if g is None or not query.strip():
            return []
        graph = {"nodes": g.nodes(), "edges": g.edges()}
        found = hybrid(graph, query, store=g, vector=vector, limit=limit, rich=True)
        near = {r["id"]: r["similarity"] for r in g.similar(vector, limit=limit * 4)} if vector else {}
        return [h["id"] for h in found
                if h["matched"] != ["meaning"] or near.get(h["id"], 0.0) >= CLOSE]

    def chain(self, a: str, b: str) -> list[str]:
        """Fact ids and entity labels along the shortest chain between two facts."""
        g = self._sync()
        if g is None:
            return []
        label = {n["id"]: n["label"] for n in g.nodes()}
        return [label[i] if not i.startswith("fact:") else i[5:] for i in g.shortest_path(node_id(a), node_id(b))]

    def stats(self) -> dict[str, Any]:
        view, facts = self.view(), self.view().facts
        return {"scope": self.realm, "project": self.project.name if self.project else "",
                "status": self.status, "path": str(self.path), "user": self.user.split(":")[-1],
                "profile": self.profile, "facts": len(facts), "limit": MAX_FACTS,
                "by_kind": {k: sum(f.kind == k for f in facts) for k in KINDS},
                "entities": len(view.entity_name), "superseded": sum(f.state != "current" for f in facts),
                "stale": sum(bool(self.stale(f)) for f in facts), "encrypted": True}

    def export(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "facts": [f.to_json() for f in self.facts()]}

    # -- writing --------------------------------------------------------------------
    def _snapshot(self, g: GraphStore) -> dict[str, Any]:
        nodes = g.nodes()
        known = {n["id"] for n in nodes}
        snap = {"schema_version": SCHEMA_VERSION, "nodes": nodes, "edges": g.edges(),
                "next": int(g.get_doc("memory", {}).get("next", 1)),
                "embeddings": {i: v for i, v in g.embeddings().items() if i in known}}
        if self.project:
            snap["project"] = self._meta()
        return snap

    def _meta(self) -> dict[str, str]:
        if self.project is None:
            return {}
        return {"ident": self.project.ident, "name": self.project.name, "root": str(self.project.root)}

    def _save(self, snap: dict[str, Any]) -> None:
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        key = self._next_key or self.keys.keys(self._salt, create=True)[0]
        plain = json.dumps(snap, sort_keys=True, separators=(",", ":")).encode()
        blob = vault.seal_blob(plain, key, mode=self.keys.mode, salt=self._salt, owner=self.owner)
        with files.writing(self.path) as tmp:
            tmp.write_bytes(blob)
            if self._status == "ok":
                files.promote(self.path, self.prev)

    def _edit(self, change: Callable[[GraphStore], Any]) -> Any:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with lock.only_one(self.path.parent / "store.lock", timeout=10, announce=lambda _: None):
            g = self._sync()
            if g is None:
                if self._status == "locked":
                    raise vault.KeyUnavailable(self._why)
                raise Tampered(f"{self._why}; `ml-stack-memory forget --all` starts a new store")
            try:
                out = change(g)
                self._save(self._snapshot(g))
            finally:
                self.close()
            return out

    def _put_fact(self, g: GraphStore, fact: Fact, ents: Iterable[Entity]) -> None:
        g.upsert_node({"id": node_id(fact.id), "kind": "fact", "label": fact.text,
                       "mentions": fact.confirm_count,
                       "attrs": {"ident": fact.id, "fkind": fact.kind, "source": fact.source,
                                 "created": fact.created, "last_confirmed": fact.last_confirmed,
                                 "confirm_count": fact.confirm_count, "scope": fact.scope,
                                 "state": fact.state}})
        for e in ents:
            g.upsert_node({"id": e.id, "kind": e.kind, "label": e.name})
            g.upsert_edge({"source": node_id(fact.id), "target": e.id,
                           "rel": "learned_on" if e.kind == "build" else "about"})

    def add(self, text: str, kind: str = "note", source: str = "user-said", *,
            entities: Iterable[Any] = (), person: bool = False) -> Fact:
        """Store a fact, or confirm the one with the same text. ``person`` is true when the
        person typed ``text`` themselves; ``entities`` are ``kind:name`` strings. A new fact that
        conflicts with a current one about the same entities and topic supersedes it (or, for a
        note, contradicts it). Raises ``Refused``, ``Tampered`` or ``KeyUnavailable``."""
        if kind not in KINDS:
            raise Refused(f"kind is one of {', '.join(KINDS)}")
        if source not in SOURCES:
            raise Refused(f"source is one of {', '.join(SOURCES)}")
        text = check(text, person=person)
        ents = parse([entities] if isinstance(entities, (str, Mapping)) else list(entities))
        scope = dict(self.scope())
        if kind in BUILD_BOUND and scope.get("build"):
            ents = [*ents, *(e for e in parse([f"build:{scope['build']}"]) if e not in ents)]
        draft = Draft(text, kind, source, ents, self.clock(), scope, self._vector(text))
        return self._edit(lambda g: self._add(g, draft))

    def _vector(self, text: str) -> Vector | None:
        if self.embed is None:
            return None
        try:
            return list(self.embed(text))
        except (OSError, ValueError, RuntimeError):
            return None

    def _add(self, g: GraphStore, d: Draft) -> Fact:
        view = self._read(g)
        for old in view.facts:
            if old.text.casefold() == d.text.casefold() and old.kind == d.kind:
                return self._reconfirm(g, old, d.ents, d.now, d.scope)
        if len(view.facts) >= MAX_FACTS:
            raise Refused(f"the store holds {MAX_FACTS} facts; forget some first "
                          "(ml-stack-memory list, forget ID)")
        number = int(g.get_doc("memory", {}).get("next", 1))
        g.put_doc("memory", {"next": number + 1})
        fact = Fact(f"m{number:04d}", d.text, d.kind, d.source, d.now, d.now, 1, d.scope,
                    sorted(str(e) for e in d.ents))
        self._put_fact(g, fact, d.ents)
        self._conflicts(g, view, fact, d.ents)
        if d.vector is not None:
            with contextlib.suppress(RuntimeError):
                g.set_embedding(node_id(fact.id), d.vector)
        return fact

    def _reconfirm(self, g: GraphStore, old: Fact, ents: list[Entity], now: float, scope: dict[str, str]) -> Fact:
        old.last_confirmed, old.confirm_count = now, old.confirm_count + 1
        old.scope = {**old.scope, **scope}
        old.entities = sorted({*old.entities, *(str(e) for e in ents)})
        self._put_fact(g, old, ents)
        return old

    def _conflicts(self, g: GraphStore, view: View, fact: Fact, ents: list[Entity]) -> None:
        """Link ``fact`` to the current facts it replaces or disagrees with."""
        about = {e.id for e in ents if e.kind != "build"}
        if not any(i.startswith("entity:topic:") for i in about):
            return
        rival = []
        for old in view.facts:
            if old.state != "current" or old.kind != fact.kind:
                continue
            theirs = {i for i, ids in view.members.items() if old.id in ids and not i.startswith("entity:build:")}
            if theirs == about:
                rival.append(old)
        rel = "contradicts" if fact.kind == "note" else "supersedes"
        for old in rival[:MAX_LINKS]:
            g.upsert_edge({"source": node_id(fact.id), "target": node_id(old.id), "rel": rel})
            if rel == "supersedes":
                g.set_attribute(node_id(old.id), "state", "superseded")

    def confirm(self, ident: str) -> Fact:
        """Mark a fact as checked now, under the build in use now."""
        now, scope = self.clock(), dict(self.scope())

        def change(g: GraphStore) -> Fact:
            view = self._read(g)
            old = view.get(ident)
            if old is None:
                raise KeyError(ident)
            build = [e for e in parse([f"build:{scope['build']}"]) if old.kind in BUILD_BOUND] if scope.get("build") else []
            return self._reconfirm(g, old, build, now, scope)

        return self._edit(change)

    def edit(self, ident: str, text: str) -> Fact:
        """Replace the text of a fact the person typed, keeping its links."""
        text = check(text, person=True)

        def change(g: GraphStore) -> Fact:
            old = self._read(g).get(ident)
            if old is None:
                raise KeyError(ident)
            old.text, old.last_confirmed = text, self.clock()
            self._put_fact(g, old, [])
            return old

        return self._edit(change)

    def link(self, a: str, rel: str, b: str, *, remove: bool = False) -> bool:
        """Join (or unjoin) two facts by ``supersedes``, ``contradicts`` or ``related``."""
        if rel not in FACT_LINKS or a == b:
            raise Refused(f"a link is one of {', '.join(FACT_LINKS)} between two different facts")

        def change(g: GraphStore) -> bool:
            if not (g.has(node_id(a)) and g.has(node_id(b))):
                raise KeyError(a if not g.has(node_id(a)) else b)
            if remove:
                return g.remove_edge(node_id(a), rel, node_id(b))
            g.upsert_edge({"source": node_id(a), "target": node_id(b), "rel": rel})
            if rel == "supersedes":
                g.set_attribute(node_id(b), "state", "superseded")
            return True

        return self._edit(change)

    def forget(self, ident: str) -> bool:
        """Delete one fact and any entity only it hung on; true when it existed."""
        def change(g: GraphStore) -> bool:
            if not g.has(node_id(ident)):
                return False
            g.drop([node_id(ident)], force=True)
            lonely = [n["id"] for n in g.nodes() if n["kind"] != "fact" and not g.neighbours(n["id"])]
            g.drop(lonely, force=True)
            return True

        return self._edit(change)

    def adopt(self, old: Store) -> int:
        """Copy every fact of ``old`` (the same user's store of a project that has moved) into
        this empty project store, sealed as this store's; returns how many facts moved."""
        if not (self.project and old.project and old.owner.split("|project:")[0] == self.owner.split("|project:")[0]):
            raise Refused("only another project store of the same user and profile can be adopted")
        graph = old._sync()
        if graph is None:
            raise Tampered(f"{old._why}; nothing was copied")
        if self.facts():
            raise Refused("this project already has memory; forget it first")
        snap = old._snapshot(graph)
        snap["project"] = self._meta()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with lock.only_one(self.path.parent / "store.lock", timeout=10, announce=lambda _: None):
            self._save(snap)
        self.close()
        return len(old.facts())

    def forget_all(self) -> int:
        """Delete every fact and start again, which also clears a tampered store; returns how
        many facts were held."""
        held = len(self.facts())
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with lock.only_one(self.path.parent / "store.lock", timeout=10, announce=lambda _: None):
            for each in (self.path, self.prev):
                each.unlink(missing_ok=True)
        self.close()
        return held

    def rekey(self, *, key: bytes | None = None, settle: bool = True) -> bytes:
        """Re-encrypt the store and its previous copy under a new key (``key`` when given, which
        must already be the current key), then drop the old key unless ``settle`` is false.
        Returns the key now in use."""
        def change(g: GraphStore) -> None:
            if key is not None:
                self._next_key = key
                return
            if self.keys.mode == "passphrase":
                self._salt = os.urandom(vault.SALT)
            self._next_key = self.keys.rotate(self._salt)

        used = b""
        try:
            self._edit(change)
            used = self._next_key or b""
            if self.path.exists():
                shutil.copyfile(self.path, self.prev)
                self.prev.chmod(0o600)
        finally:
            self._next_key = None
        if settle:
            self.keys.settle()
        return used
