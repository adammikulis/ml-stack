"""GGUF conversion, quantisation and tokenizer-metadata repair."""

from __future__ import annotations

from ml_stack.gguf.convert import (
    ConversionError,
    ConversionResult,
    convert,
    export,
    quantize,
)
from ml_stack.gguf.tools import (
    LLAMA_CPP_SRC,
    SOURCE_DIRS,
    ToolNotFound,
    ensure_converter,
    find_converter,
    find_quantize,
    require_converter,
    require_quantize,
)
from ml_stack.gguf.vocab import (
    ADD_SPACE_PREFIX,
    VocabPatchError,
    fix_space_prefix,
    read_metadata,
    set_metadata,
)

__all__ = [
    "ADD_SPACE_PREFIX",
    "LLAMA_CPP_SRC",
    "SOURCE_DIRS",
    "ConversionError",
    "ConversionResult",
    "ToolNotFound",
    "VocabPatchError",
    "convert",
    "ensure_converter",
    "export",
    "find_converter",
    "find_quantize",
    "fix_space_prefix",
    "quantize",
    "read_metadata",
    "require_converter",
    "require_quantize",
    "set_metadata",
    "Check", "FidelityReport", "verify_metadata", "verify_tokenizer_fidelity",
]

from ml_stack.gguf.verify import (  # noqa: E402
    Check, FidelityReport, verify_metadata, verify_tokenizer_fidelity,
)
