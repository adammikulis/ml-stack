"""The text of a PDF that a reader can see: invisible render mode, white or tiny text, text off
the page and text in a layer that is switched off are left out."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ml_stack.net.untrusted import clean_text

__all__ = ["MIN_POINTS", "visible_text"]

MIN_POINTS = 2.0
"""Text smaller than this many points is not read."""


def _white(color: Any) -> bool:
    return isinstance(color, (tuple, list)) and len(color) >= 3 and all(c >= 0.97 for c in color[:3])


def _hidden_spans(page: Any, off_layers: set[str]) -> Counter[str]:
    bad: Counter[str] = Counter()
    rect = page.rect
    for span in page.get_texttrace():
        text = "".join(chr(c[0]) for c in span.get("chars", ())).strip()
        if not text:
            continue
        box = span.get("bbox") or (0, 0, 0, 0)
        outside = box[2] < rect.x0 or box[0] > rect.x1 or box[3] < rect.y0 or box[1] > rect.y1
        if (span.get("type") == 3 or _white(span.get("color")) or span.get("size", 12) < MIN_POINTS
                or outside or span.get("layer", "") in off_layers
                or span.get("opacity", 1.0) <= 0.02):
            bad[text] += 1
    return bad


def _page_text(page: Any, off_layers: set[str]) -> tuple[str, int]:
    bad = _hidden_spans(page, off_layers)
    lines: list[str] = []
    dropped = 0
    for block in page.get_text("dict").get("blocks", ()):
        for line in block.get("lines", ()):
            kept = []
            for span in line.get("spans", ()):
                piece = str(span.get("text", ""))
                if bad.get(piece.strip(), 0) > 0 and piece.strip():
                    bad[piece.strip()] -= 1
                    dropped += 1
                    continue
                kept.append(piece)
            if kept:
                lines.append("".join(kept))
    return "\n".join(lines) + "\n", dropped


def visible_text(path: str | Path, *, limit: int | None = None) -> tuple[str, str, int, int]:
    """``(title, text, page_count, removed)`` of a PDF with only what a reader can see, and
    invisible Unicode characters stripped. ``removed`` counts hidden spans and characters."""
    try:
        import pymupdf
    except ImportError as exc:
        raise ImportError("reading a PDF needs pymupdf: pip install 'ml-stack[pdf]'") from exc
    removed = 0
    with pymupdf.open(str(Path(path).expanduser())) as doc:
        off = {str(v.get("name", "")) for v in doc.get_ocgs().values() if not v.get("on", True)}
        out: list[str] = []
        size = 0
        for page in doc:
            text, dropped = _page_text(page, off)
            removed += dropped
            out.append(text)
            size += len(text)
            if limit is not None and size >= limit:
                break
        title, gone = clean_text(" ".join(str((doc.metadata or {}).get("title") or "").split()))
        body, more = clean_text("".join(out))
        return title.strip(), body[:limit], doc.page_count, removed + gone + more
