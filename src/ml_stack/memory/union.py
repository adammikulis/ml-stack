"""The user store and the project store together: a read-only merged view for recall, and the
two opened as a pair."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ml_stack.memory.facts import Fact
from ml_stack.memory.project import SCOPES, Project, detect
from ml_stack.memory.store import Setup, Store, View

__all__ = ["Memory", "Merged", "words"]

TAGS = {"user": "u", "project": "p"}
Vector = Sequence[float]


def words(scope: str, project: Project | None = None) -> str:
    """The scope in plain words, as the person sees it."""
    if scope == "project":
        return f"this project ({project.name})" if project else "this project"
    return "you (all projects)"


class Merged:
    """A read-only view over the stores of one session. Fact ids are prefixed ``u.`` and ``p.``
    so the two number lines do not meet; entities with the same id are one node, so a user
    fact and a project fact about the same model sit side by side. Nothing here is written to
    disk and nothing in a store changes."""

    realm = ""

    def __init__(self, stores: dict[str, Store]) -> None:
        self.stores = stores
        self._seen: tuple[int, ...] = ()
        self._view = View()

    @property
    def status(self) -> str:
        found = {s.status for s in self.stores.values()}
        return next((x for x in ("tampered", "locked") if x in found), "ok")

    def problems(self) -> list[tuple[str, str]]:
        return [p for s in self.stores.values() for p in s.problems()]

    def view(self) -> View:
        """Both stores' facts and entities in one view."""
        views = {scope: store.view() for scope, store in self.stores.items()}
        stamp = tuple(id(v) for v in views.values())
        if stamp == self._seen:
            return self._view
        out = View()
        for scope, v in views.items():
            tag = TAGS[scope]
            out.facts += [replace(f, id=f"{tag}.{f.id}", realm=scope) for f in v.facts]
            out.entity_name.update(v.entity_name)
            for ent, ids in v.members.items():
                out.members.setdefault(ent, []).extend(f"{tag}.{i}" for i in ids)
            for ident, ties in v.ties.items():
                out.ties[f"{tag}.{ident}"] = [(rel, f"{tag}.{o}") for rel, o in ties]
        out.facts.sort(key=lambda f: f.id)
        self._seen, self._view = stamp, out
        return out

    def facts(self) -> list[Fact]:
        return list(self.view().facts)

    def stale(self, fact: Fact) -> str:
        return self.stores[fact.realm].stale(fact)

    def hits(self, query: str, vector: Vector | None = None, limit: int = 12) -> list[str]:
        """Fact and entity ids from both stores, taking turns, best first."""
        per = []
        for scope, store in self.stores.items():
            tag = TAGS[scope]
            per.append([f"fact:{tag}.{h[5:]}" if h.startswith("fact:") else h
                        for h in store.hits(query, vector, limit)])
        mixed = [h for row in itertools.zip_longest(*per) for h in row if h]
        return list(dict.fromkeys(mixed))[:limit]

    def where(self, fact: Fact) -> str:
        """``fact``'s scope in words."""
        return words(fact.realm, self.stores[fact.realm].project)


@dataclass(slots=True)
class Memory:
    """The user's store and, when a project is known, that project's store."""

    user: Store
    project: Store | None = None

    @classmethod
    def open(cls, *, root: Path | None = None, explicit: Path | None = None,
             setup: Setup | None = None, user_path: Path | None = None,
             project_path: Path | None = None) -> Memory:
        """The stores for the project at ``explicit``, or else the one ``root`` (default: the
        working directory) is in. ``setup`` is shared, so both use the same user, profile and keys."""
        base = setup or Setup()
        found = detect(root, explicit=explicit)
        user = Store(user_path, setup=replace(base, project=None))
        mine = Store(project_path, setup=replace(base, project=found)) if found else None
        return cls(user, mine)

    @property
    def stores(self) -> dict[str, Store]:
        return {"user": self.user, **({"project": self.project} if self.project else {})}

    def store(self, scope: str) -> Store:
        """The store for ``scope``; ``ValueError`` when it is unknown or no project is open."""
        if scope not in SCOPES:
            raise ValueError("scope is user or project")
        if scope == "project" and self.project is None:
            raise ValueError("no project is open here; only scope user is available")
        return self.stores[scope]

    def merged(self) -> Merged:
        return Merged(self.stores)

    def on_disk(self) -> list[Store]:
        """Every project store of this user and profile that has a file, this project's included."""
        out = []
        for each in sorted((self.user.path.parent / "projects").glob("*/graph.enc")):
            same = self.project is not None and self.project.project is not None \
                and self.project.project.key == each.parent.name
            out.append(self.project if same and self.project else Store(
                setup=Setup(user=self.user.user, profile=self.user.profile, keys=self.user.keys,
                            project=Project(Path(), "", "", each.parent.name))))
        return out

    def rekey(self) -> None:
        """Re-encrypt the user store and every project store, each under the key of its own new salt."""
        for each in [self.user, *self.on_disk()]:
            each.rekey()

    def close(self) -> None:
        for each in self.stores.values():
            each.close()
