"""One embedded ladybug database and its connection: Cypher in, rows out, the word and vector
indexes a reader searches through, and a census of what the store holds.

Nothing here knows a schema. `GraphStore` puts its nodes and edges on top of this; an app
with a schema of its own subclasses `CypherStore` the same way.

    with CypherStore(path) as db:
        db.query("MATCH (n:Person {id:$id}) RETURN n.name AS name", {"id": "p1"})
        db.index_words("Person", "person_words", ["name"])
        db.search_words("Person", "person_words", "ada", returns="node.id AS id")
"""

from __future__ import annotations

import hashlib
import os
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from ml_stack.graph.snapshots import DISAGREEING

__all__ = ["CypherStore", "GraphStoreUnavailable", "census", "literal", "store_memory"]


class GraphStoreUnavailable(RuntimeError):
    """Ladybug is not installed, or a store cannot be read. `pip install ml-stack[store]`."""


def store_memory() -> int:
    """The buffer pool a store opens with, in bytes, from ``$MLSTACK_STORE_MEMORY``.

    Zero leaves the engine to size it, which it does as a share of this machine's memory.
    """
    try:
        return max(0, int(os.environ.get("MLSTACK_STORE_MEMORY", "") or 0))
    except ValueError:
        return 0


_EXTENSION_OF = {"FTS": "fts", "HNSW": "vector"}


def _quoted(name: str) -> str:
    """A table or index name as a Cypher identifier."""
    return "`" + str(name).replace("`", "``") + "`"


def _text(value: str) -> str:
    """A name as a Cypher string literal, for the procedures that take names as strings."""
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def literal(value: Any) -> str:
    """A value as Cypher source, for the one place the engine takes no parameter.

    A recursive pattern's predicate -- ``-[r:R* SHORTEST 1..4 (e, n | WHERE ...)]-`` -- refuses
    every parameter, so what it compares against is written into the statement.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return "[" + ", ".join(literal(one) for one in value) + "]"
    raise TypeError(f"{type(value).__name__} has no Cypher literal; pass it as a parameter")


class CypherStore:
    """A ladybug database on disk and the one connection this handle asks through."""

    def __init__(self, path: str | Path, *, read_only: bool = False,
                 buffer_pool_size: int | None = None, max_db_size: int | None = None) -> None:
        try:
            import ladybug as lb
        except ImportError as exc:  # pragma: no cover - depends on what is installed
            raise GraphStoreUnavailable(str(exc)) from exc
        self.path = Path(path).expanduser()
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        options: dict[str, Any] = {
            "read_only": read_only,
            "buffer_pool_size": store_memory() if buffer_pool_size is None
            else max(0, int(buffer_pool_size))}
        if max_db_size:
            options["max_db_size"] = int(max_db_size)
        self._db = lb.Database(str(self.path), **options)
        self._conn = lb.Connection(self._db)
        self.read_only = read_only
        self._in_tx = False
        self._loaded: set[str] = set()
        if not read_only:
            self._load_indexed()

    def _load_indexed(self) -> None:
        """Load the extension behind every index the store holds, which a write to its table needs."""
        for row in self.query("CALL SHOW_INDEXES() RETURN index_type"):
            kind = str(row["index_type"]).upper()
            if kind in _EXTENSION_OF:
                self.load(_EXTENSION_OF[kind])

    # -- lifetime

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the connection, then the database. Closing twice is closing once."""
        for handle in (getattr(self, "_conn", None), getattr(self, "_db", None)):
            with suppress(Exception):  # closing twice is not worth an error
                handle.close()  # type: ignore[union-attr]

    # -- asking

    def query(self, cypher: str, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Any Cypher, as a list of rows keyed by what the query returned."""
        result = self._run(cypher, params)
        if isinstance(result, list):  # a multi-statement query returns one result each
            result = result[-1]
        names = result.get_column_names()
        return [dict(zip(names, row)) for row in result.get_all()]

    def _run(self, cypher: str, params: Mapping[str, Any] | None) -> Any:
        """Execute one statement, prepared afresh whenever it carries values."""
        if not params:
            return self._conn.execute(cypher)
        # ladybug's execute() prepares with the values bound and re-runs that plan for the same
        # text; the re-run segfaults (QUERY_FTS_INDEX, an edge MERGE after its node table changed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            statement = self._conn.prepare(cypher)
        if not statement.is_success():
            raise RuntimeError(statement.get_error_message())
        return self._conn.execute(statement, dict(params))

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Everything inside lands together, or none of it does."""
        if self._in_tx:
            yield
            return
        self._conn.execute("BEGIN TRANSACTION")
        self._in_tx = True
        try:
            yield
        except BaseException:
            with suppress(RuntimeError):  # a failed statement already rolled it back
                self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")
        finally:
            self._in_tx = False

    @property
    def in_transaction(self) -> bool:
        return self._in_tx

    # -- what the schema holds

    def tables(self) -> dict[str, str]:
        """Every table, by name, with its kind: ``NODE`` or ``REL``."""
        return {str(r["name"]): str(r["type"])
                for r in self.query("CALL SHOW_TABLES() RETURN name, type")}

    def has_table(self, name: str) -> bool:
        """Whether a table of this name is declared. Asking a table that is not is an error."""
        return str(name) in self.tables()

    def has_index(self, table: str, name: str) -> bool:
        """Whether an index of this name is built on the table."""
        return any(r["table_name"] == table and r["index_name"] == name
                   for r in self.query("CALL SHOW_INDEXES() RETURN table_name, index_name"))

    # -- the indexes a reader searches through

    def load(self, extension: str) -> None:
        """Install and load an extension, once per handle. A read-only handle may load one.

        A table carrying an index refuses writes from a database that has not loaded the
        extension owning it, so a writer loads every extension it may meet.
        """
        if extension in self._loaded:
            return
        self._conn.execute(f"INSTALL {extension}")
        self._conn.execute(f"LOAD EXTENSION {extension}")
        self._loaded.add(extension)

    def index_words(self, table: str, name: str, columns: Sequence[str]) -> bool:
        """Build a word index over some string columns. False on a read-only handle.

        A reader cannot build one, so whoever writes the table builds it.
        """
        self.load("fts")
        if self.read_only or self._in_tx:
            return False
        if not self.has_index(table, name):
            listed = ", ".join(_text(c) for c in columns)
            self._conn.execute(f"CALL CREATE_FTS_INDEX({_text(table)}, {_text(name)}, [{listed}])")
        return True

    def index_vectors(self, table: str, name: str, column: str, *, metric: str = "cosine") -> bool:
        """Build a nearest-neighbour index over a vector column. False on a read-only handle."""
        self.load("vector")
        if self.read_only or self._in_tx:
            return False
        if not self.has_index(table, name):
            self._conn.execute(f"CALL CREATE_VECTOR_INDEX({_text(table)}, {_text(name)}, "
                               f"{_text(column)}, metric := {_text(metric)})")
        return True

    def drop_index(self, table: str, name: str) -> bool:
        """Take an index off a table, which must happen before the table can be dropped."""
        found = [r for r in self.query("CALL SHOW_INDEXES() RETURN table_name, index_name, "
                                       "index_type") if r["table_name"] == table
                 and r["index_name"] == name]
        if not found:
            return False
        kind = "FTS" if str(found[0]["index_type"]).upper() == "FTS" else "VECTOR"
        self.load("fts" if kind == "FTS" else "vector")
        self._conn.execute(f"CALL DROP_{kind}_INDEX({_text(table)}, {_text(name)})")
        return True

    def search_words(self, table: str, index: str, text: str, *, returns: str,
                     limit: int = 10) -> list[dict[str, Any]]:
        """Rows matching some words, best first, each with a ``score``. Empty without an index.

        ``returns`` is the RETURN list over ``node``, e.g. ``"node.id AS id"``.
        """
        self.load("fts")
        if not self.has_index(table, index):
            return []
        return self.query(
            f"CALL QUERY_FTS_INDEX({_text(table)}, {_text(index)}, $q, TOP := $k) "
            f"RETURN {returns}, score AS score ORDER BY score DESC",
            {"q": str(text), "k": int(limit)})

    def nearest(self, table: str, index: str, vector: Sequence[float], *, returns: str,
                limit: int = 10) -> list[dict[str, Any]]:
        """Rows nearest a vector, nearest first, each with a ``distance``. Empty without an index."""
        self.load("vector")
        if not self.has_index(table, index):
            return []
        return self.query(
            f"CALL QUERY_VECTOR_INDEX({_text(table)}, {_text(index)}, $v, $k) "
            f"RETURN {returns}, distance AS distance ORDER BY distance",
            {"v": [float(x) for x in vector], "k": int(limit)})

    # -- what is in here

    def census(self) -> dict[str, int]:
        """Rows per table, and per relationship table how many read differently backwards.

        A relationship's properties are stored once for each direction, so it is read walking
        forward and walking backward, paired by its id, and counted when the two disagree.
        """
        counts: dict[str, int] = {}
        for name, kind in sorted(self.tables().items()):
            if kind == "NODE":
                counts[name] = int(self.query(f"MATCH (n:{_quoted(name)}) RETURN count(n) AS c")[0]["c"])
            elif kind == "REL":
                counts[name], counts[name + DISAGREEING] = self._walked_both_ways(name)
        return counts

    def _walked_both_ways(self, rel: str) -> tuple[int, int]:
        props = [str(r["name"]) for r in self.query(f"CALL TABLE_INFO({_text(rel)}) RETURN name")]
        values = "".join(f", r.{_quoted(p)}" for p in props)
        ahead = dict(self._rows_by_id(
            f"MATCH (a)-[r:{_quoted(rel)}]->(b) RETURN id(r), id(a), id(b){values}"))
        behind = dict(self._rows_by_id(
            f"MATCH (b)<-[r:{_quoted(rel)}]-(a) RETURN id(r), id(a), id(b){values}"))
        return len(ahead), sum(1 for key in ahead.keys() | behind.keys()
                               if ahead.get(key) != behind.get(key))

    def _rows_by_id(self, cypher: str) -> Iterator[tuple[str, bytes]]:
        result = self._conn.execute(cypher)
        while result.has_next():
            rel, *rest = result.get_next()
            yield repr(rel), hashlib.blake2b(repr(rest).encode(), digest_size=16).digest()


def census(path: str | Path) -> dict[str, int]:
    """Open a store read-only on a fresh handle and take its census.

    Only a fresh open sees what reached the disk, so this is what a snapshot is verified by.
    """
    with CypherStore(path, read_only=True) as store:
        return store.census()
