"""JSON record stores: what ships with the package, what this machine added over it.

`Records` is a list of records keyed one per subject -- the measured fits, the measured
profiles. `Document` is a single record in one object -- this machine's limits, its
settings, its observed rates.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Generic, TypeVar

from ml_stack.files import write_json

__all__ = ["Document", "Records"]

R = TypeVar("R")

DATA = Path(__file__).resolve().parent / "data"
"""Where the records that ship with ml-stack live."""

def _installed(path: Path) -> bool:
    return "site-packages" in path.parts or "dist-packages" in path.parts


class Records(Generic[R]):
    """Records of one kind: the file that ships with ml-stack, with this machine's own
    laid over it, keyed so a local record replaces the shipped one it supersedes."""

    def __init__(self, name: str, *, env: str,
                 build: Callable[[Mapping[str, Any]], R],
                 unbuild: Callable[[R], dict[str, Any]],
                 key: Callable[[R], Any],
                 order: Callable[[R], Any],
                 named: str = "model") -> None:
        self.name = name
        self.env = env
        self.build = build
        self.unbuild = unbuild
        self.key = key
        self.order = order
        self.named = named

    def package_path(self) -> Path:
        """The file that ships with ml-stack."""
        return DATA / self.name

    def local_path(self) -> Path:
        """This machine's own file. The environment variable moves it."""
        named = os.environ.get(self.env)
        return Path(named).expanduser() if named else Path.home() / ".ml-stack" / self.name

    def writable_path(self) -> Path:
        """Where a new record goes: the shipped file in a checkout somebody can write to,
        this machine's own file otherwise."""
        shipped = self.package_path()
        if _installed(shipped):
            return self.local_path()
        try:
            shipped.parent.mkdir(parents=True, exist_ok=True)
            if os.access(shipped.parent, os.W_OK):
                return shipped
        except OSError:
            pass
        return self.local_path()

    def read(self, path: Path) -> list[R]:
        """Every record one file holds. A file that is absent, unreadable or not a list of
        objects contributes nothing."""
        try:
            parsed = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        out: list[R] = []
        for row in parsed:
            if not isinstance(row, Mapping) or not row.get(self.named):
                continue
            try:
                out.append(self.build(row))
            except (TypeError, ValueError):
                continue
        return out

    def all(self, *, package: Path | None = None, local: Path | None = None) -> list[R]:
        """The shipped records with this machine's own over them, sorted."""
        merged: dict[Any, R] = {}
        for one in self.read(package or self.package_path()) + self.read(
                local or self.local_path()):
            merged[self.key(one)] = one
        return sorted(merged.values(), key=self.order)

    def write(self, records: Iterable[R], path: Path) -> Path:
        """Write ``records`` to ``path``, sorted, and return where they went."""
        kept = sorted(records, key=self.order)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([self.unbuild(one) for one in kept], indent=2) + "\n",
                        encoding="utf-8")
        return path

    def add(self, record: R, *, path: Path | None = None,
            merge: Callable[[R, R | None], R] | None = None) -> Path:
        """Write one record in, replacing the one it supersedes. Returns where it went.

        ``merge`` is handed the new record and the one it replaces, if any.
        """
        where = path or self.writable_path()
        held = self.read(where)
        older = next((one for one in held if self.key(one) == self.key(record)), None)
        kept = [one for one in held if self.key(one) != self.key(record)]
        kept.append(merge(record, older) if merge else record)
        return self.write(kept, where)


class Document(Generic[R]):
    """One record kept as a JSON object, at a path an environment variable may move."""

    def __init__(self, *, default: Callable[[], Path] | None = None, env: str = "",
                 build: Callable[[Mapping[str, Any]], R],
                 unbuild: Callable[[R], Any],
                 empty: Callable[[], R]) -> None:
        self.default = default
        self.env = env
        self.build = build
        self.unbuild = unbuild
        self.empty = empty

    def path(self, path: Path | str | None = None) -> Path:
        """Where this record is kept: what was asked for, else the environment variable,
        else the default."""
        if path is not None:
            return Path(path).expanduser()
        named = os.environ.get(self.env) if self.env else None
        if named:
            return Path(named).expanduser()
        if self.default is None:
            raise ValueError("this record has no default path; name one")
        return self.default()

    def read(self, path: Path | str | None = None) -> R:
        """What is on disk, or an empty record. A file that will not parse is empty."""
        try:
            held = json.loads(self.path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self.empty()
        if not isinstance(held, Mapping):
            return self.empty()
        try:
            return self.build(held)
        except (TypeError, ValueError):
            return self.empty()

    def write(self, record: R, path: Path | str | None = None, *,
              atomic: bool = False) -> Path:
        """Write ``record`` down and return where it went."""
        where = self.path(path)
        if atomic:
            write_json(where, self.unbuild(record))
            return where
        where.parent.mkdir(parents=True, exist_ok=True)
        where.write_text(json.dumps(self.unbuild(record), indent=2, sort_keys=True),
                         encoding="utf-8")
        return where
