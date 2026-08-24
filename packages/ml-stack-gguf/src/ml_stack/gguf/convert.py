"""HF safetensors -> GGUF -> quantised GGUF."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ml_stack.gguf.tools import require_converter, require_quantize

logger = logging.getLogger(__name__)

DEFAULT_OUTTYPE = "f16"
DEFAULT_QUANT = "Q8_0"


class ConversionError(RuntimeError):
    """The conversion or quantisation failed."""


@dataclass(frozen=True, slots=True)
class ConversionResult:
    path: Path
    outtype: str
    sha256: str
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1e6


def _run(argv: list[str], *, what: str) -> None:
    logger.info("%s: %s", what, " ".join(argv))
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-30:]
        raise ConversionError(f"{what} failed (exit {result.returncode}):\n" + "\n".join(tail))


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _describe(path: Path, outtype: str, *, write_sidecar: bool) -> ConversionResult:
    digest = _digest(path)
    result = ConversionResult(path, outtype, digest, path.stat().st_size)
    if write_sidecar:
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{digest}  {path.name}\n", encoding="utf-8"
        )
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(
                {"file": path.name, "outtype": outtype,
                 "sha256": digest, "size_bytes": result.size_bytes},
                indent=2,
            ),
            encoding="utf-8",
        )
    return result


def convert(
    model_dir: Path | str,
    outfile: Path | str,
    *,
    outtype: str = DEFAULT_OUTTYPE,
    converter: Path | str | None = None,
    python: str | None = None,
    write_sidecar: bool = True,
) -> ConversionResult:
    """Run ``convert_hf_to_gguf.py`` over a HF-layout directory."""
    model_dir, outfile = Path(model_dir), Path(outfile)
    if not model_dir.is_dir():
        raise ConversionError(f"no model directory at {model_dir}")

    script = require_converter(converter)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [python or sys.executable, str(script), str(model_dir),
         "--outfile", str(outfile), "--outtype", outtype],
        what="convert_hf_to_gguf",
    )

    if not outfile.is_file():
        raise ConversionError(f"{script.name} reported success but wrote no {outfile}")
    return _describe(outfile, outtype, write_sidecar=write_sidecar)


def quantize(
    src: Path | str,
    dst: Path | str,
    quant: str = DEFAULT_QUANT,
    *,
    binary: Path | str | None = None,
    write_sidecar: bool = True,
) -> ConversionResult:
    """Run ``llama-quantize``."""
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise ConversionError(f"no GGUF at {src}")

    tool = require_quantize(binary)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run([str(tool), str(src), str(dst), quant], what="llama-quantize")

    if not dst.is_file():
        raise ConversionError(f"llama-quantize reported success but wrote no {dst}")
    return _describe(dst, quant, write_sidecar=write_sidecar)


def export(
    model_dir: Path | str,
    out_dir: Path | str,
    *,
    name: str = "model",
    quant: str = DEFAULT_QUANT,
    intermediate: str = "f32",
    fix_space_prefix: bool | None = False,
    keep_intermediate: bool = False,
) -> ConversionResult:
    """The whole path: convert, patch the tokenizer metadata, quantise."""
    from ml_stack.gguf.vocab import fix_space_prefix as _patch

    out_dir = Path(out_dir)
    raw = out_dir / f"{name}-{intermediate}.gguf"
    final = out_dir / f"{name}-{quant}.gguf"

    convert(model_dir, raw, outtype=intermediate, write_sidecar=False)

    if fix_space_prefix is not None:
        _patch(raw, add_space_prefix=fix_space_prefix)

    result = quantize(raw, final, quant)

    if not keep_intermediate:
        raw.unlink(missing_ok=True)
    return result
