"""A model's chat template, with the guards that refuse a Claude Code conversation lifted."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

__all__ = ["LATE_SYSTEM", "forgiving", "needs_forgiving", "template_of",
           "trained_context", "written_beside"]


#: The guard Qwen's template raises on when a system message is not the first one.
LATE_SYSTEM = "System message must be at the beginning"

_RAISE = re.compile(
    r"\{\{-?\s*raise_exception\(\s*'" + re.escape(LATE_SYSTEM) + r"\.?'\s*\)\s*-?\}\}")


def template_of(model: str | Path) -> str:
    """The chat template a GGUF carries, or "" when it names none."""
    from ml_stack.serve.preflight import read_gguf_header

    try:
        return str(read_gguf_header(Path(model)).get("tokenizer.chat_template") or "")
    except Exception:  # noqa: BLE001 - a header that will not read names no template
        return ""


def needs_forgiving(template: str) -> bool:
    """Whether this template refuses a system message that is not the first."""
    return bool(_RAISE.search(template or ""))


def forgiving(template: str) -> str:
    """``template`` with the late-system guard rendering the message instead of raising.

    Claude Code sends system-role messages after the conversation has started, which the
    Qwen family's template refuses outright. Rendering them where they stand is what
    llama.cpp's own generic template and Ollama's pass-through both do.
    """
    return _RAISE.sub(
        "{{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}", template or "")


def written_beside(model: str | Path, *, where: str | Path | None = None) -> Path | None:
    """A file holding this model's forgiving template, or None when it needs none."""
    template = template_of(model)
    if not needs_forgiving(template):
        return None
    root = Path(where) if where is not None else Path(tempfile.mkdtemp(prefix="ml-stack-tpl-"))
    root.mkdir(parents=True, exist_ok=True)
    out = root / (Path(model).stem + ".jinja")
    out.write_text(forgiving(template), encoding="utf-8")
    return out


def trained_context(model: str | Path) -> int:
    """The context length a GGUF says it was trained for, or 0 when it says none."""
    try:
        from ml_stack.serve.layout import layout

        return int(getattr(layout(Path(model)), "context_length", 0) or 0)
    except Exception:  # noqa: BLE001 - a header that will not read names no length
        return 0
