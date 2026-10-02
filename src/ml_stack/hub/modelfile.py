"""The one reader of a GGUF header and a safetensors header.

Every count and length in a header comes from the file. Each is checked against what the rest
of the file can hold before anything is allocated or looped over, and a header is also held to
a nesting depth, a total number of array items and a time budget, so a hostile or corrupt file
is refused with `NotAModelFile` and never hangs a listing.
"""

from __future__ import annotations

import json
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

__all__ = ["ARRAY", "GGUF_MAGIC", "SCALARS", "Limits", "NotAModelFile", "Scan", "TensorEntry",
           "safetensors_header", "scan_gguf"]

GGUF_MAGIC = b"GGUF"
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
           10: "Q", 11: "q", 12: "d"}
"""The fixed-width value kinds of GGUF and the struct code that reads each."""
STRING, ARRAY = 8, 9
_PAIR_BYTES = 12
_TENSOR_BYTES = 24
_STRING_BYTES = 8
_ARRAY_BYTES = 12
MOST_SAFETENSORS_HEADER = 100 << 20


class NotAModelFile(ValueError):
    """The file is not a model file, or its header claims more than the file holds."""


@dataclass(frozen=True, slots=True)
class Limits:
    """What a header may ask for: pairs, one string's bytes, array items over the whole header,
    array nesting, dimensions per tensor, and seconds. ``keep_array``, when set, is the longest
    array of numbers whose value is built; any other array is read past."""

    pairs: int = 1 << 20
    text: int = 1 << 26
    items: int = 1 << 23
    depth: int = 4
    dims: int = 8
    seconds: float = 10.0
    keep_array: int | None = None


DEFAULT = Limits()


@dataclass(frozen=True, slots=True)
class TensorEntry:
    """One row of the tensor table: its name, shape, ggml type number and data offset."""

    name: str
    shape: tuple[int, ...]
    kind: int
    offset: int


@dataclass(slots=True)
class Scan:
    """What `scan_gguf` read. ``values`` holds the pairs asked for; ``stopped`` is
    ``(key, kind, item_kind, count)`` for the pair a ``stop`` ended the scan at, where
    ``item_kind`` and ``count`` are those of an array and -1 and 0 otherwise."""

    version: int = 0
    tensor_count: int = 0
    pair_count: int = 0
    values: dict[str, object] = field(default_factory=dict)
    tensors: list[TensorEntry] = field(default_factory=list)
    stopped: tuple[str, int, int, int] | None = None


class _Reader:
    def __init__(self, f: BinaryIO, size: int, name: str, limits: Limits) -> None:
        self.f, self.size, self.name, self.limits = f, size, name, limits
        self.deadline = time.monotonic() + limits.seconds
        self.items = limits.items

    def left(self) -> int:
        return self.size - self.f.tell()

    def refuse(self, why: str) -> NotAModelFile:
        return NotAModelFile(f"{self.name}: {why}; not a model file")

    def tick(self) -> None:
        if time.monotonic() > self.deadline:
            raise self.refuse(f"its header took more than {self.limits.seconds:g}s to read")

    def room(self, claimed: int, each: int, what: str, most: int | None = None) -> int:
        """``claimed`` when the rest of the file could hold that many of ``what``, at ``each``
        bytes at least, and it is no more than ``most``."""
        if claimed * each > self.left() or claimed > (most if most is not None else claimed):
            raise self.refuse(f"a {what} claims {claimed} entries of at least {each} bytes, "
                              "more than the file holds")
        return claimed

    def take(self, n: int) -> bytes:
        if n < 0 or n > self.left():
            raise self.refuse(f"a field claims {n} bytes, more than the file holds")
        got = self.f.read(n)
        if len(got) != n:
            raise self.refuse("the header is cut short")
        return got

    def skip(self, n: int) -> None:
        if n < 0 or n > self.left():
            raise self.refuse(f"a field claims {n} bytes, more than the file holds")
        self.f.seek(n, 1)

    def unpack(self, code: str) -> Any:
        return struct.unpack("<" + code, self.take(struct.calcsize("<" + code)))[0]

    def length(self) -> int:
        n = self.unpack("Q")
        if n > min(self.limits.text, self.left()):
            raise self.refuse(f"a string claims {n} bytes, more than the file holds")
        return n

    def text(self) -> str:
        return self.take(self.length()).decode("utf-8", "replace")

    def spend(self, count: int) -> None:
        self.items -= count
        if self.items < 0:
            raise self.refuse(f"its arrays hold more than {self.limits.items} items")

    def array_head(self, depth: int) -> tuple[int, int]:
        if depth >= self.limits.depth:
            raise self.refuse(f"arrays nest deeper than {self.limits.depth}")
        item = self.unpack("I")
        if item not in SCALARS and item not in (STRING, ARRAY):
            raise self.refuse(f"an array of unknown kind {item}")
        count = self.unpack("Q")
        each = (struct.calcsize("<" + SCALARS[item]) if item in SCALARS
                else _STRING_BYTES if item == STRING else _ARRAY_BYTES)
        self.room(count, each, "array")
        self.spend(count)
        return item, count

    def value(self, kind: int, depth: int = 0, keep: bool = True) -> object:
        """The value of ``kind``; with ``keep`` false it is read past and ``None`` is returned."""
        self.tick()
        if kind == STRING:
            if keep:
                return self.text()
            self.skip(self.length())
            return None
        if kind == ARRAY:
            item, count = self.array_head(depth + 1)
            longest = self.limits.keep_array
            keep = keep and (longest is None or (item in SCALARS and count <= longest))
            if item in SCALARS:
                width = struct.calcsize("<" + SCALARS[item])
                if not keep:
                    self.skip(count * width)
                    return None
                return list(struct.unpack(f"<{count}{SCALARS[item]}", self.take(count * width)))
            out = []
            for _ in range(count):
                got = self.value(item, depth + 1, keep)
                if keep:
                    out.append(got)
            return out if keep else None
        if kind not in SCALARS:
            raise self.refuse(f"a value of unknown kind {kind}")
        if keep:
            return self.unpack(SCALARS[kind])
        self.skip(struct.calcsize("<" + SCALARS[kind]))
        return None


def scan_gguf(path: Path | str, *, want: Callable[[str], bool] | None = None,
              stop: Callable[[str], bool] | None = None, tensors: bool = False,
              limits: Limits = DEFAULT) -> Scan:
    """The header of a GGUF file, checked against its size.

    ``want(key)`` says which pairs to keep (all by default; the others are read past without
    being built). ``stop(key)`` ends the scan before the value of the first key it accepts.
    With ``tensors`` the tensor table that follows the pairs is read too. Raises
    `NotAModelFile` for anything that is not a GGUF or claims more than the file holds, and
    `OSError` when it cannot be opened.
    """
    path = Path(path).expanduser()
    out = Scan()
    with path.open("rb") as f:
        size = path.stat().st_size
        rd = _Reader(f, size, str(path), limits)
        if f.read(4) != GGUF_MAGIC:
            raise NotAModelFile(f"{path}: not a GGUF file (no GGUF magic at the start)")
        out.version = rd.unpack("I")
        out.tensor_count = rd.unpack("Q")
        out.pair_count = rd.unpack("Q")
        if out.pair_count > limits.pairs:
            raise rd.refuse(f"it claims {out.pair_count} metadata pairs")
        rd.room(out.pair_count, _PAIR_BYTES, "metadata pair list")
        for _ in range(out.pair_count):
            key = rd.text()
            kind = rd.unpack("I")
            if stop is not None and stop(key):
                if kind == ARRAY:
                    item, count = rd.unpack("I"), rd.unpack("Q")
                    out.stopped = (key, kind, item, count)
                else:
                    out.stopped = (key, kind, -1, 0)
                return out
            keep = want is None or want(key)
            got = rd.value(kind, 0, keep)
            if keep:
                out.values[key] = got
        if tensors:
            out.tensors = _tensor_table(rd, out.tensor_count)
    return out


def _tensor_table(rd: _Reader, count: int) -> list[TensorEntry]:
    rd.room(count, _TENSOR_BYTES, "tensor list", rd.limits.pairs)
    found: list[TensorEntry] = []
    for _ in range(count):
        rd.tick()
        name = rd.text()
        dims = rd.unpack("I")
        if dims > rd.limits.dims:
            raise rd.refuse(f"a tensor claims {dims} dimensions")
        shape = struct.unpack(f"<{dims}Q", rd.take(8 * dims)) if dims else ()
        kind = rd.unpack("I")
        found.append(TensorEntry(name, tuple(shape), kind, rd.unpack("Q")))
    return found


def safetensors_header(path: Path | str) -> dict[str, object]:
    """The JSON header of a safetensors file, checked against the file's size: the length it
    declares must fit in the file and under 100 MiB, it must be a JSON object, and each tensor's
    ``data_offsets`` must be two ordered whole numbers."""
    path = Path(path)
    with path.open("rb") as f:
        size = path.stat().st_size
        head = f.read(8)
        declared = int.from_bytes(head, "little")
        if len(head) < 8 or declared > min(MOST_SAFETENSORS_HEADER, size - 8):
            raise NotAModelFile(f"{path}: a header of {declared} bytes; not a safetensors file")
        try:
            header = json.loads(f.read(declared))
        except (ValueError, RecursionError) as exc:
            raise NotAModelFile(f"{path}: the header is not JSON; not a safetensors file") from exc
    if not isinstance(header, dict):
        raise NotAModelFile(f"{path}: the header is not an object; not a safetensors file")
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if not (isinstance(offsets, list) and len(offsets) == 2
                and all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in offsets)
                and offsets[0] <= offsets[1]):
            raise NotAModelFile(f"{path}: tensor {name!r} has no valid data_offsets; "
                                "not a safetensors file")
    return header
