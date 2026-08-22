"""Rewrite a GGUF's tokenizer metadata in place.

``convert_hf_to_gguf.py``'s generic Llama path never emits
``tokenizer.ggml.add_space_prefix`` -- only the Gemma3 path calls it. Without the key,
llama.cpp **defaults it to true** and inserts a space after every special token, so
``"<user> go"`` tokenizes with an extra ``▁`` that the Python tokenizer never produces.
Training data and inference then disagree on every single sequence, and nothing reports
it: the model simply learns slightly worse and serves slightly worse, forever.

Written for any metadata key rather than just this one, because the underlying class of
bug -- a converter path that omits a key whose default is wrong for your tokenizer -- is
not specific to this field.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

ADD_SPACE_PREFIX = "tokenizer.ggml.add_space_prefix"

#: Reader fields that describe the *container* rather than the model. ``GGUFReader``
#: surfaces these alongside real metadata, but the writer emits its own header -- copying
#: them through produces a file with duplicate header keys that the reader then refuses
#: to open.
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
    """Copy ``src`` to ``dst``, overriding the metadata keys in ``values``.

    Every other key and every tensor is copied through untouched. Rewriting rather than
    patching in place because GGUF metadata is length-prefixed at the head of the file --
    changing a value shifts every tensor offset after it.
    """
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
            # Recorded and returned rather than printed: a key this gguf version cannot
            # round-trip is a real loss, and a caller comparing two exports needs to know
            # which ones went missing.
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
    """Write ``tokenizer.ggml.add_space_prefix`` into a GGUF that lacks it.

    ``dst=None`` rewrites ``src`` through a temporary file.

    Default ``False``, which is the correct value for a plain SentencePiece tokenizer --
    and the whole point is that llama.cpp's default when the key is *absent* is ``True``.
    """
    src = Path(src)
    target = Path(dst) if dst is not None else src

    staging = target.with_suffix(target.suffix + ".fixed")
    set_metadata(src, staging, {ADD_SPACE_PREFIX: add_space_prefix})
    shutil.move(str(staging), str(target))
    return target
