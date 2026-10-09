"""The text of a PDF that a reader can see. Invisible render mode, white or tiny text and
transparent text are left out here; text off the page and in a layer that is switched off is
left out by the extraction itself."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ml_stack.net.pdfread import Limits, engine, load
from ml_stack.net.untrusted import clean_text

__all__ = ["MIN_POINTS", "visible_text"]

MIN_POINTS = 2.0
"""Text smaller than this many points is not read."""


def _white(color: Any) -> bool:
    return isinstance(color, (tuple, list)) and len(color) >= 3 and all(c >= 0.97 for c in color[:3])


def _hidden_spans(page: Any) -> Counter[str]:
    """The text of spans drawn invisibly, white, under `MIN_POINTS` or transparent."""
    bad: Counter[str] = Counter()
    for span in page.get_texttrace():
        text = "".join(chr(c[0]) for c in span.get("chars", ())).strip()
        if text and (span.get("type") == 3 or _white(span.get("color"))
                     or span.get("size", 12) < MIN_POINTS or span.get("opacity", 1.0) <= 0.02):
            bad[text] += 1
    return bad


def _page_text(page: Any) -> tuple[str, int]:
    bad = _hidden_spans(page)
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


PAGE_LIMITS = Limits(max_bytes=32 * 1024 * 1024, max_pages=1500, timeout_s=90.0,
                     memory_bytes=2 * 1024 ** 3)
"""What reading a downloaded PDF may use. Past any of it the file is refused (`PdfRefused`)."""


def _page_text_mined(page: Any) -> tuple[str, int]:
    """The visible text of one page of the default engine, and how many hidden spans it dropped."""
    lines: list[str] = []
    dropped = 0
    for block in page.get_text("dict").get("blocks", ()):
        for line in block.get("lines", ()):
            kept = []
            for span in line.get("spans", ()):
                if span.get("hidden") and str(span.get("text", "")).strip():
                    dropped += 1
                    continue
                kept.append(str(span.get("text", "")))
            text = "".join(kept).rstrip("\n")
            if text.strip():
                lines.append(text)
    return "\n".join(lines) + "\n", dropped


def visible_text(path: str | Path, *, limit: int | None = None) -> tuple[str, str, int, int]:
    """``(title, text, page_count, removed)`` of a PDF with only what a reader can see, and
    invisible Unicode characters stripped. ``removed`` counts hidden spans and characters.

    The default engine (pdfminer.six, MIT) runs in a bounded child process and refuses a file
    that crosses a bound with `PdfRefused`; ``ML_STACK_PDF_ENGINE=pymupdf`` reads with MuPDF
    (AGPL-3.0, opt-in extra ``pdf-agpl``) instead.
    """
    if engine() == "pymupdf":
        return _visible_text_mupdf(path, limit=limit)
    where = Path(path).expanduser()
    removed = 0
    with load(where, limits=PAGE_LIMITS, text_limit=limit) as doc:
        out: list[str] = []
        size = 0
        for page in doc:
            text, dropped = _page_text_mined(page)
            removed += dropped
            out.append(text)
            size += len(text)
            if limit is not None and size >= limit:
                break
        title, gone = clean_text(" ".join(str(doc.metadata.get("title") or "").split()))
        body, more = clean_text("".join(out))
        return title.strip(), body[:limit], doc.page_count, removed + gone + more


def _visible_text_mupdf(path: str | Path, *, limit: int | None = None) -> tuple[str, str, int, int]:
    try:
        import pymupdf
    except ImportError as exc:
        raise ImportError("ML_STACK_PDF_ENGINE=pymupdf needs MuPDF (AGPL-3.0): "
                          "pip install 'ml-stack[pdf-agpl]'") from exc
    removed = 0
    with pymupdf.open(str(Path(path).expanduser())) as doc:
        out: list[str] = []
        size = 0
        for page in doc:
            text, dropped = _page_text(page)
            removed += dropped
            out.append(text)
            size += len(text)
            if limit is not None and size >= limit:
                break
        title, gone = clean_text(" ".join(str((doc.metadata or {}).get("title") or "").split()))
        body, more = clean_text("".join(out))
        return title.strip(), body[:limit], doc.page_count, removed + gone + more
