"""What a model's file is made of, read from its GGUF header alone: every tensor's type,
shape and size, without reading a byte of tensor data or starting a server."""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ml_stack.hub import pretty_name
from ml_stack.serve.preflight import _GGUF_MAGIC, _SCALAR_FMT, shard_names
from ml_stack.units import human_bytes

__all__ = ["Tensor", "render_tensors", "shard_paths", "table_bytes", "tensors_of",
           "totals_by_role", "totals_by_type"]

# ggml's own block quantisation, as (name, values per block, bytes per block). Straight out
# of `ggml.c`'s type_traits table; the arithmetic below is ggml_nbytes' arithmetic, which is
# how a per-tensor size can be had from a header without reading a byte of tensor data.
_GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24), 8: ("q8_0", 32, 34), 9: ("q8_1", 32, 40),
    10: ("q2_K", 256, 84), 11: ("q3_K", 256, 110), 12: ("q4_K", 256, 144),
    13: ("q5_K", 256, 176), 14: ("q6_K", 256, 210), 15: ("q8_K", 256, 292),
    16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74), 18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 50), 20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136), 24: ("i8", 1, 1), 25: ("i16", 1, 2),
    26: ("i32", 1, 4), 27: ("i64", 1, 8), 28: ("f64", 1, 8), 29: ("iq1_m", 256, 56),
    30: ("bf16", 1, 2), 34: ("tq1_0", 256, 54), 35: ("tq2_0", 256, 66),
    39: ("mxfp4", 32, 17),
}

# A gathered table, not a matmul: only the rows whose keys occur are ever touched, so with
# mmap the rest of it never becomes resident. `per_layer_token_embd` is Flash-Next's n-gram
# table -- 26.8G of a 103.7G file -- and the other two names are what the same idea is
# called elsewhere.
_TABLE_NAMES = ("per_layer_token_embd", "ngram", "engram")


@dataclass(frozen=True, slots=True)
class Tensor:
    """One tensor as the GGUF header describes it: what it is called, what it is stored
    as, how big it is, and how many bytes of the file it takes."""

    name: str
    type: str
    shape: tuple[int, ...]
    bytes: int

    @property
    def elements(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def role(self) -> str:
        """``table``, ``experts``, ``attention``, ``embedding`` or ``other``.

        The point of the grouping is the first one: a table is paged a row at a time and an
        expert is not, so two files of the same size can hold very different amounts of
        resident memory and only the names say which.
        """
        low = self.name.lower()
        if any(word in low for word in _TABLE_NAMES):
            return "table"
        if "exps" in low or ".experts" in low:
            return "experts"
        if "attn" in low or "attention" in low:
            return "attention"
        # `output.weight` is the lm_head; `output_norm.weight` is a norm and is not
        # one, so the dot is load-bearing rather than tidiness.
        if "token_embd" in low or low.startswith("output."):
            return "embedding"
        return "other"


def shard_paths(model: str | Path) -> list[Path]:
    """Every shard of a model that is on this machine, from its first file's name."""
    first = Path(model).expanduser()
    found = [first.parent / name for name in shard_names(first.name)]
    return [one for one in found if one.is_file()] or [first]


def _tensors_in(path: Path) -> list[Tensor]:
    """Every tensor one GGUF file's header names, without reading any tensor data.

    `preflight.read_gguf_header` deliberately stops before the tensor table -- nothing a
    preflight needs is in it, and walking it is the expensive half. This walks the same
    metadata block, with the same value-kind table, only to get past it to the table that
    answers "what is this file made *of*". Still a header read: the offsets are read, the
    bytes they point at never are.
    """
    out: list[Tensor] = []
    with Path(path).expanduser().open("rb") as f:
        if f.read(4) != _GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file (no GGUF magic at the start)")
        struct.unpack("<I", f.read(4))                        # version -- unused here
        (tensor_count,) = struct.unpack("<Q", f.read(8))
        (kv_count,) = struct.unpack("<Q", f.read(8))

        def text() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def skip(kind: int) -> None:
            if kind == 8:
                text()
                return
            if kind == 9:
                (item_kind,) = struct.unpack("<I", f.read(4))
                (count,) = struct.unpack("<Q", f.read(8))
                for _ in range(count):
                    skip(item_kind)
                return
            f.read(struct.calcsize(_SCALAR_FMT[kind]))

        for _ in range(kv_count):
            text()
            (kind,) = struct.unpack("<I", f.read(4))
            skip(kind)

        for _ in range(tensor_count):
            name = text()
            (dims,) = struct.unpack("<I", f.read(4))
            shape = struct.unpack(f"<{dims}Q", f.read(8 * dims)) if dims else ()
            (kind,) = struct.unpack("<I", f.read(4))
            struct.unpack("<Q", f.read(8))                     # offset -- unused here
            named, block, per_block = _GGML_TYPES.get(kind, (f"type{kind}", 1, 0))
            count = 1
            for dim in shape:
                count *= dim
            out.append(Tensor(name=name, type=named, shape=tuple(shape),
                              bytes=count // block * per_block if block else 0))
    return out


def tensors_of(model: str | Path) -> list[Tensor]:
    """Every tensor in a model, over all of its shards, largest first."""
    found: list[Tensor] = []
    for shard in shard_paths(model):
        found += _tensors_in(shard)
    return sorted(found, key=lambda t: (-t.bytes, t.name))


def totals_by_type(found: Iterable[Tensor]) -> list[tuple[str, int, int]]:
    """``(type, how many tensors, how many bytes)``, largest first.

    The listing that answers "what is the 15.7G of": a `UD-Q4_K_XL` is not one type, and
    the types a backend declines are exactly the ones that are not the majority.
    """
    counts: dict[str, list[int]] = {}
    for one in found:
        row = counts.setdefault(one.type, [0, 0])
        row[0] += 1
        row[1] += one.bytes
    return sorted(((name, n, size) for name, (n, size) in counts.items()),
                  key=lambda row: -row[2])


def totals_by_role(found: Iterable[Tensor]) -> list[tuple[str, int, int]]:
    """The same bytes grouped by what the tensor is *for*, largest first -- table,
    experts, attention, embedding, other. The one grouping that predicts residency."""
    counts: dict[str, list[int]] = {}
    for one in found:
        row = counts.setdefault(one.role, [0, 0])
        row[0] += 1
        row[1] += one.bytes
    return sorted(((name, n, size) for name, (n, size) in counts.items()),
                  key=lambda row: -row[2])


def table_bytes(model: str | Path) -> int:
    """Bytes of gathered lookup table in a model's file, or 0 when it has none or cannot
    be read. Never raises: a record is worth writing without this number."""
    try:
        return sum(one.bytes for one in tensors_of(model) if one.role == "table")
    except (OSError, ValueError, struct.error):
        return 0


def _dims(shape: Iterable[int]) -> str:
    return "(" + ", ".join(f"{int(d):,}" for d in shape) + ")"


def render_tensors(model: str | Path, *, top: int = 12) -> str:
    """What a model file is made of, from its header alone -- the answer to "why is it
    smaller in memory than on disk" that needs no server and no GPU.

    Three parts: the largest tensors with their type and shape (a lookup table marked, so
    the one tensor that is 26% of Flash-Next is not just another row), the totals per
    tensor type, and the totals per role. Sums every shard.
    """
    found = tensors_of(model)
    if not found:
        return f"{Path(model).name}: no tensors in the header"
    shards = shard_paths(model)
    total = sum(one.bytes for one in found)

    lines = [f"{pretty_name(Path(model).name)}"
             + (f"  ({len(shards)} shards)" if len(shards) > 1 else ""),
             f"  {len(found):,} tensors, {human_bytes(total)} of weights in the header"]
    table = sum(one.bytes for one in found if one.role == "table")
    if table:
        lines.append(f"  {human_bytes(table)} of that is a gathered lookup table: paged a row "
                     "at a time, so most of it never becomes resident")
    lines += ["", f"  the {min(top, len(found))} largest tensors", ""]
    for one in found[:top]:
        lines.append(f"  {human_bytes(one.bytes):>8}  {one.type:<8} {one.name:<40} "
                     f"{_dims(one.shape)}"
                     + ("   <- gathered table" if one.role == "table" else ""))
    lines += ["", "  by type", ""]
    for name, count, size in totals_by_type(found):
        lines.append(f"  {human_bytes(size):>8}  {name:<8} {count:>5} tensors")
    lines += ["", "  by what it is for", ""]
    for name, count, size in totals_by_role(found):
        lines.append(f"  {human_bytes(size):>8}  {name:<10} {count:>5} tensors")
    return "\n".join(lines)
