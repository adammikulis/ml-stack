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
    ToolNotFound,
    ensure_converter,
    find_converter,
    find_quantize,
    llama_cpp_src,
    require_converter,
    require_quantize,
    source_dirs,
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
    "llama_cpp_src",
    "quantize",
    "read_metadata",
    "require_converter",
    "require_quantize",
    "source_dirs",
    "set_metadata",
    "Check", "FidelityReport", "verify_metadata", "verify_tokenizer_fidelity",
]

from ml_stack.gguf.verify import (
    Check, FidelityReport, verify_metadata, verify_tokenizer_fidelity,
)
