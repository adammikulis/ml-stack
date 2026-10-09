"""Pure rules for per-device journals: hybrid clock order, chain and head checks, merge."""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from poolhouse.fleet.onboard.manifest import Signer
from poolhouse.workspace.chain import GENESIS, _digest

__all__ = [
    "HEAD",
    "SKEW_MAX_MS",
    "Accepted",
    "Damaged",
    "Gap",
    "Quota",
    "accept",
    "head_message",
    "held_back",
    "merge",
    "row_id",
    "tick",
    "total_order",
    "valid_origin",
]

HEAD = "head"
SKEW_MAX_MS = 300_000
ORIGIN = re.compile(r"[0-9a-f]{32}")


def valid_origin(origin: object) -> bool:
    """Whether ``origin`` is a journal origin id: 32 lower-case hexadecimal characters."""
    return type(origin) is str and ORIGIN.fullmatch(origin) is not None


class Damaged(ValueError):
    """A journal copy that is forged, forked or broken."""


class Quota(ValueError):
    """A journal that would exceed what a device keeps."""


class Gap(ValueError):
    """Rows that do not start where the held copy ends."""


@dataclass(frozen=True, slots=True)
class Accepted:
    """The rows to store for an origin and the public key its heads verified under."""

    rows: list[dict[str, Any]]
    public: bytes


def row_id(row: dict[str, Any]) -> str:
    """The immutable id of a row: its origin and sequence number."""
    return f"{row['origin']}:{row['seq']}"


def total_order(row: dict[str, Any]) -> tuple[int, int, str, int]:
    """The key rows are merged by: wall clock, counter, origin, sequence number."""
    wall, counter, origin = row["hlc"]
    return int(wall), int(counter), str(origin), int(row["seq"])


def tick(previous: tuple[int, int], now_ms: int, seen: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    """The next hybrid logical clock value after ``previous``, the wall clock and ``seen``."""
    wall = max(now_ms, previous[0], seen[0])
    counter = max(previous[1] if previous[0] == wall else -1, seen[1] if seen[0] == wall else -1) + 1
    return wall, counter


def held_back(row: dict[str, Any], now_ms: int) -> bool:
    """Whether the row's wall clock is further ahead of ``now_ms`` than the allowed skew."""
    return int(row["hlc"][0]) > now_ms + SKEW_MAX_MS


def head_message(pool: str, origin: str, seq: int, digest: str) -> bytes:
    """The bytes a journal head signs."""
    return "|".join(("poolhouse-journal-head-v1", pool, origin, str(seq), digest)).encode()


def merge(journals: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Every journal's rows in one total order, heads dropped and a repeated idempotency key
    of one actor on one origin kept only at its first row."""
    rows = sorted((r for journal in journals.values() for r in journal if r["kind"] != HEAD), key=total_order)
    seen: set[tuple[str, str, str]] = set()
    kept: list[dict[str, Any]] = []
    for row in rows:
        key = (row["origin"], row["actor"], row["idem"])
        if row["idem"] and key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def _shape(row: Any, origin: str) -> None:
    if not isinstance(row, dict) or row.get("origin") != origin or row.get("v") != 1:
        raise Damaged("row does not belong to this journal")
    hlc = row.get("hlc")
    if (not isinstance(hlc, list) or len(hlc) != 3 or hlc[2] != origin
            or type(hlc[0]) is not int or type(hlc[1]) is not int):
        raise Damaged("row carries no clock value")
    if not isinstance(row.get("kind"), str) or not isinstance(row.get("actor"), str) \
            or not isinstance(row.get("idem"), str) or not isinstance(row.get("body"), dict):
        raise Damaged("row is missing a field")


def _chain(rows: list[dict[str, Any]], origin: str, tip: dict[str, Any] | None) -> None:
    prev, seq, wall = (tip["hash"], tip["seq"], tip["hlc"][0]) if tip else (GENESIS, 0, 0)
    for row in rows:
        _shape(row, origin)
        if row.get("seq") != seq + 1 or row.get("prev") != prev:
            raise Damaged(f"row {row.get('seq')} does not follow row {seq}")
        if row.get("hash") != _digest(prev, row):
            raise Damaged(f"row {seq + 1} hash does not match its contents")
        if row["hlc"][0] < wall:
            raise Damaged(f"row {seq + 1} moves the wall clock backwards")
        prev, seq, wall = row["hash"], row["seq"], row["hlc"][0]


def _heads(new: list[dict[str, Any]], tops: dict[int, str], pinned: bytes | None, pool: str) -> tuple[bytes, int]:
    public, last = pinned, 0
    for row in new:
        if row["kind"] != HEAD:
            continue
        body = row["body"]
        try:
            key = base64.b64decode(body["public"], validate=True)
            signature = base64.b64decode(body["sig"], validate=True)
            target, digest = int(body["head"]), str(body["hash"])
        except (KeyError, ValueError, TypeError):
            raise Damaged(f"head at row {row['seq']} is malformed") from None
        if public is not None and key != public:
            raise Damaged(f"head at row {row['seq']} is signed by a different key")
        if target != row["seq"] - 1 or tops.get(target) != digest:
            raise Damaged(f"head at row {row['seq']} does not sign the row before it")
        if not Signer.check_bytes(key, head_message(pool, row["origin"], target, digest), signature):
            raise Damaged(f"head at row {row['seq']} has a bad signature")
        public, last = key, row["seq"]
    return public or b"", last


def accept(origin: str, held: list[dict[str, Any]], incoming: list[dict[str, Any]],
           pinned: bytes | None = None, pool: str = "") -> Accepted:
    """The rows of ``incoming`` to append to the ``held`` copy of ``origin``'s journal: those
    past what is held, through the last row a verified head covers. Raises `Damaged` for a
    fork, a broken chain or a forged head and `Gap` when the rows start after the held end."""
    if not valid_origin(origin):
        raise Damaged("origin is not a journal id")
    if not incoming:
        return Accepted([], pinned or b"")
    if not all(isinstance(r, dict) for r in incoming):
        raise Damaged("a journal is a list of rows")
    first = incoming[0].get("seq")
    if type(first) is not int or first < 1:
        raise Damaged("rows carry no sequence number")
    if first > len(held) + 1:
        raise Gap(f"rows start at {first} but {len(held)} are held")
    overlap = held[first - 1:first - 1 + len(incoming)]
    for mine, theirs in zip(overlap, incoming, strict=False):
        if mine.get("hash") != theirs.get("hash"):
            raise Damaged(f"row {mine['seq']} differs from the held copy")
    new = incoming[len(overlap):]
    if not new:
        return Accepted([], pinned or b"")
    _chain(new, origin, held[-1] if held else None)
    tops = {r["seq"]: r["hash"] for r in held}
    tops.update({r["seq"]: r["hash"] for r in new})
    public, last = _heads(new, tops, pinned, pool)
    return Accepted([r for r in new if r["seq"] <= last], public)
