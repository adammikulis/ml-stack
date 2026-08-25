"""Rewrite a GGUF's tokenizer metadata in place."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

ADD_SPACE_PREFIX = "tokenizer.ggml.add_space_prefix"

_HEADER_PREFIX = "GGUF."


class VocabPatchError(RuntimeError):
    """The GGUF could not be rewritten."""


def read_metadata(path: Path | str) -> dict[str, Any]:
    """Every metadata key in a GGUF, for inspection and for tests."""
    try:
        from gguf import GGUFReader
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise VocabPatchError("the `gguf` package is required to read GGUF metadata") from exc

    reader = GGUFReader(str(path))
    out: dict[str, Any] = {}
    for name, field in reader.fields.items():
        if name.startswith(_HEADER_PREFIX):
            continue
        try:
            out[name] = field.contents()
        except Exception:  # a field type this gguf version cannot render
            out[name] = None
    return out


def set_metadata(
    src: Path | str,
    dst: Path | str,
    values: dict[str, Any],
) -> Path:
    """Copy ``src`` to ``dst``, overriding the metadata keys in ``values``."""
    try:
        import numpy as np
        from gguf import GGUFReader, GGUFValueType, GGUFWriter
    except ImportError as exc:  # pragma: no cover
        raise VocabPatchError("the `gguf` and `numpy` packages are required") from exc

    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise VocabPatchError(f"no GGUF at {src}")

    reader = GGUFReader(str(src))
    arch_field = reader.fields.get("general.architecture")
    if arch_field is None:
        raise VocabPatchError(f"{src} has no general.architecture; is it a GGUF?")
    architecture = str(arch_field.contents())

    tmp = dst.with_suffix(dst.suffix + ".part")
    writer = GGUFWriter(str(tmp), architecture)

    skipped: list[str] = []
    copied = 0
    for name, field in reader.fields.items():
        if name == "general.architecture" or name in values:
            continue
        if name.startswith(_HEADER_PREFIX):
            continue
        value_type = field.types[0] if field.types else None
        try:
            if value_type == GGUFValueType.ARRAY:
                subtype = field.types[1]
                items = [field.contents(i) for i in range(len(field.data))]
                if subtype == GGUFValueType.STRING:
                    writer.add_array(name, [str(v) for v in items])
                else:
                    writer.add_array(name, items)
            else:
                writer.add_key_value(name, field.contents(), value_type)
            copied += 1
        except Exception as exc:
            skipped.append(f"{name} ({type(exc).__name__})")

    for key, value in values.items():
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, str):
            writer.add_string(key, value)
        else:
            raise VocabPatchError(f"cannot write metadata {key}={value!r} of type {type(value)}")

    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, np.array(tensor.data), raw_dtype=tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    if skipped:
        raise VocabPatchError(
            f"{len(skipped)} metadata key(s) could not be copied and would be lost: "
            + ", ".join(skipped)
            + ". Refusing to write a GGUF that is missing metadata the source had."
        )

    shutil.move(str(tmp), str(dst))
    return dst


def fix_space_prefix(
    src: Path | str,
    dst: Path | str | None = None,
    *,
    add_space_prefix: bool = False,
) -> Path:
    """Write ``tokenizer.ggml.add_space_prefix`` into a GGUF that lacks it."""
    src = Path(src)
    target = Path(dst) if dst is not None else src

    staging = target.with_suffix(target.suffix + ".fixed")
    set_metadata(src, staging, {ADD_SPACE_PREFIX: add_space_prefix})
    shutil.move(str(staging), str(target))
    return target
