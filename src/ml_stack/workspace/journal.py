"""The directory of signed append-only journals: this device's own and verbatim copies of the others."""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from ml_stack.files import read_json, write_json
from ml_stack.workspace import journal_merge as rules
from ml_stack.workspace.chain import ChainLog, held

__all__ = ["Journals", "Signing", "Table"]

VERSION = 1


class Signing(Protocol):
    """What a journal needs of a device key."""

    def sign_bytes(self, data: bytes) -> bytes: ...

    @property
    def public(self) -> bytes: ...


class Table:
    """A small map kept in one versioned JSON file; callers serialise writers."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def all(self) -> dict[str, Any]:
        """Every entry."""
        doc = read_json(self.path, {})
        entries = doc.get("entries") if isinstance(doc, dict) else None
        return dict(entries) if isinstance(entries, dict) else {}

    def get(self, key: str, default: Any = None) -> Any:
        """The entry for ``key``."""
        return self.all().get(key, default)

    def put(self, key: str, value: Any) -> None:
        """Set the entry for ``key``."""
        entries = self.all()
        entries[key] = value
        write_json(self.path, {"version": VERSION, "entries": entries})


class Journals:
    """Journals under ``directory``, one chained file per origin. Only the file named for this
    device's origin is appended to; the others are copies that passed `journal_merge.accept`.
    The key signs only when `seal` runs."""

    def __init__(self, directory: Path, signer: Callable[[], Signing],
                 clock: Callable[[], float] = time.time, pool: str = "") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.signer, self.clock, self.pool = signer, clock, pool
        self.lock = self.directory / "journals.lock"
        self._origin = ""
        self._logs: dict[str, ChainLog] = {}
        self.keys, self.refused, self.clocks = (Table(self.directory / f"{n}.json") for n in ("keys", "damaged", "clock"))

    # -- identity and bookkeeping --
    @property
    def origin(self) -> str:
        """This device's origin id, created on first use."""
        if not self._origin:
            with held(self.lock):
                path = self.directory / "origin.json"
                doc = read_json(path, {})
                if not isinstance(doc, dict) or not doc.get("origin"):
                    doc = {"version": VERSION, "origin": uuid4().hex}
                    write_json(path, doc)
                self._origin = str(doc["origin"])
        return self._origin

    def damaged(self) -> dict[str, str]:
        """The origins whose copies were refused, with the reason."""
        return self.refused.all()

    def log(self, origin: str) -> ChainLog:
        """The chained file of ``origin``'s journal."""
        if origin not in self._logs:
            self._logs[origin] = ChainLog(self.directory / f"{origin}.jsonl", self.clock)
        return self._logs[origin]

    def rows(self, origin: str) -> list[dict[str, Any]]:
        """The verified rows held for ``origin``."""
        return self.log(origin).rows()

    def origins(self) -> list[str]:
        """Every origin with a journal held here."""
        return sorted(p.stem for p in self.directory.glob("*.jsonl")) if self.directory.exists() else []

    def vector(self) -> dict[str, dict[str, Any]]:
        """For each held origin, the sequence number and hash of its last row."""
        out: dict[str, dict[str, Any]] = {}
        for origin in self.origins():
            rows = self.rows(origin)
            if rows:
                out[origin] = {"seq": rows[-1]["seq"], "hash": rows[-1]["hash"]}
        return out

    def trusted(self, origin: str) -> list[dict[str, Any]]:
        """The rows of ``origin`` through its last head, the part a peer may be given."""
        rows = self.rows(origin)
        last = max((r["seq"] for r in rows if r["kind"] == rules.HEAD), default=0)
        return rows[:last]

    # -- own journal --
    def append(self, kind: str, actor: str, body: dict[str, Any], idem: str = "") -> dict[str, Any]:
        """Add a row to this device's journal; a repeated ``idem`` of the same actor returns the
        row already written."""
        origin = self.origin
        log = self.log(origin)
        with held(self.lock):
            rows = log.rows()
            if idem:
                for row in rows:
                    if row["actor"] == actor and row["idem"] == idem:
                        return row
            return self._add(log, rows, {"kind": kind, "actor": actor, "idem": idem, "body": body})

    def _add(self, log: ChainLog, rows: list[dict[str, Any]], entry: dict[str, Any]) -> dict[str, Any]:
        last = tuple(rows[-1]["hlc"][:2]) if rows else (0, 0)
        seen = self.clocks.get("seen", [0, 0])
        wall, counter = rules.tick((int(last[0]), int(last[1])), int(self.clock() * 1000), (int(seen[0]), int(seen[1])))
        return log.append({"origin": self.origin, "hlc": [wall, counter, self.origin], **entry})

    def seal(self) -> dict[str, Any] | None:
        """Sign the last row of this device's journal with a head row; None when it is already
        covered."""
        origin = self.origin
        log = self.log(origin)
        with held(self.lock):
            rows = log.rows()
            if not rows or rows[-1]["kind"] == rules.HEAD:
                return None
            tip = rows[-1]
            key = self.signer()
            signature = key.sign_bytes(rules.head_message(self.pool, origin, tip["seq"], tip["hash"]))
            return self._add(log, rows, {"kind": rules.HEAD, "actor": "", "idem": "", "body": {
                "head": tip["seq"], "hash": tip["hash"], "public": base64.b64encode(key.public).decode(),
                "sig": base64.b64encode(signature).decode()}})

    # -- copies of other journals --
    def ingest(self, origin: str, incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Store the rows of another device's journal that verify; returns the rows stored.
        A forged or forked copy marks the origin damaged and raises `Damaged`."""
        if origin == self.origin:
            return []
        if origin in self.damaged():
            raise rules.Damaged(f"journal {origin} was refused: {self.damaged()[origin]}")
        log = self.log(origin)
        with held(self.lock):
            have = log.rows()
            pin = self.keys.get(origin)
            try:
                took = rules.accept(origin, have, incoming, base64.b64decode(pin) if pin else None, self.pool)
            except rules.Damaged as bad:
                self.refused.put(origin, str(bad))
                raise
            if not took.rows:
                return []
            log.extend(took.rows)
            if not pin:
                self.keys.put(origin, base64.b64encode(took.public).decode())
            self._observe(took.rows)
            return took.rows

    def _observe(self, rows: list[dict[str, Any]]) -> None:
        now_ms = int(self.clock() * 1000)
        walls = [(int(r["hlc"][0]), int(r["hlc"][1])) for r in rows if not rules.held_back(r, now_ms)]
        seen = self.clocks.get("seen", [0, 0])
        top = max([(int(seen[0]), int(seen[1])), *walls])
        self.clocks.put("seen", list(top))
