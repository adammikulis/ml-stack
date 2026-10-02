"""What a GGUF file says about itself, read from the front of the file and nothing more."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ml_stack.hub.modelfile import ARRAY, Limits, NotAModelFile, scan_gguf

FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4",
}
"""``general.file_type`` values, as llama.cpp's ``llama_ftype`` numbers them."""

_SIZE_LABEL = re.compile(r"(\d+(?:\.\d+)?)\s*([KMBT])", re.IGNORECASE)
_UNITS = {"K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}


@dataclass(frozen=True, slots=True)
class Header:
    """The facts a listing wants from a GGUF header."""

    architecture: str = ""
    name: str = ""
    size_label: str = ""
    parameters: int = 0
    quantization: str = ""
    context_length: int = 0
    layers: int = 0

    def as_dict(self) -> dict[str, object]:
        return {f: getattr(self, f) for f in self.__slots__}

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> Header:
        return cls(**{f: row[f] for f in cls.__slots__ if f in row})


def parameters_of(label: str) -> int:
    """The parameter count a size label such as ``8B``, ``270M`` or ``30B-A3B`` names, or 0."""
    found = _SIZE_LABEL.search(label or "")
    if not found:
        return 0
    return int(float(found.group(1)) * _UNITS[found.group(2).upper()])


MOST_ARRAY = 4096
"""Longest array of numbers `meta` returns; others are skipped."""
VOCAB = "vocab_size"
"""The key `meta` adds: how many tokens the tokenizer lists."""


def meta(path: Path | str) -> dict[str, object] | None:
    """Every key the header holds before its tokenizer, or ``None`` when it is not a GGUF
    or its header is not one a model file has.

    Scalars, strings and short numeric arrays are read; the tokenizer's arrays are the only
    large part of a header, so reading stops at the first tokenizer key, counting the tokens
    there as ``vocab_size``.
    """
    try:
        got = scan_gguf(path, stop=lambda key: key.startswith("tokenizer."),
                       limits=Limits(keep_array=MOST_ARRAY))
    except (OSError, NotAModelFile):
        return None
    found = {key: value for key, value in got.values.items() if value is not None}
    if got.stopped and got.stopped[0] == "tokenizer.ggml.tokens" and got.stopped[1] == ARRAY:
        found[VOCAB] = got.stopped[3]
    return found


def summary(found: dict[str, object]) -> Header:
    """The listing facts out of a header read by `meta`."""
    arch = str(found.get("general.architecture", ""))
    label = str(found.get("general.size_label", ""))
    count_key = found.get("general.parameter_count")
    kind_id = found.get("general.file_type")
    return Header(
        architecture=arch,
        name=str(found.get("general.name", "")),
        size_label=label,
        parameters=int(count_key) if isinstance(count_key, int) else parameters_of(label),
        quantization=FILE_TYPES.get(kind_id, "") if isinstance(kind_id, int) else "",
        context_length=int(found.get(f"{arch}.context_length", 0) or 0),
        layers=int(found.get(f"{arch}.block_count", 0) or 0),
    )


def read(path: Path | str) -> Header | None:
    """The header summary of a GGUF file, or ``None`` when it is not one."""
    found = meta(path)
    return summary(found) if found is not None else None
