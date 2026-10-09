"""Turning pages of a PDF nobody vouches for into PNG, with permissively licensed code and hard bounds.

PDFium (Apache-2.0 / BSD-3-Clause, through pypdfium2, extra ``pdf-render``) runs in a child
interpreter started by path under :class:`RenderLimits`, like the text reader
(:mod:`poolhouse.net.pdfread`); every bound crossed (file size, pages in the file, pages
rendered, the points of a page side, pixels, bytes of PNG, resident memory, CPU, wall time) or
an unreadable or encrypted file raises :class:`~poolhouse.net.pdfread.PdfRefused`.
"""

from __future__ import annotations

import base64
import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from poolhouse.net.pdfread import PdfRefused, run_child

__all__ = ["RenderLimits", "Rendered", "available", "render_pages"]

CHILD = Path(__file__).with_name("pdfrenderchild.py")
"""Run by path (``-P``), so it depends on no copy of poolhouse; it imports pypdfium2 and Pillow."""
NEEDS = ("rendering a page needs PDFium (pypdfium2, Apache-2.0/BSD-3-Clause): "
         "pip install 'poolhouse[pdf-render]'")


@dataclass(frozen=True)
class RenderLimits:
    """The bounds one rendering runs under. Every one is hard: crossing it refuses the file."""

    max_bytes: int = 64 * 1024 * 1024
    max_pages: int = 3000
    max_render_pages: int = 8
    max_page_pt: float = 5000.0
    """The longest side of a page, in points (72 per inch): A0 is 3370."""
    max_pixels: int = 16_000_000
    max_image_pixels: int = 150_000_000
    max_out: int = 32 * 1024 * 1024
    timeout_s: float = 60.0
    memory_bytes: int = 2 * 1024 ** 3


@dataclass(frozen=True)
class Rendered:
    """One rendered page: ``png`` bytes of ``width`` x ``height`` pixels; ``blank`` when every
    pixel is white or nearly so."""

    page: int
    png: bytes
    width: int
    height: int
    blank: bool


def available() -> bool:
    """Whether PDFium can be imported here (without importing it)."""
    return importlib.util.find_spec("pypdfium2") is not None


def render_pages(path: str | Path, pages: Sequence[int], *, dpi: int = 150, crop: bool = False,  # noqa: PLR0913 - the facts of one rendering
                 max_width: int = 1568, pad_pt: float = 14.0,
                 limits: RenderLimits | None = None) -> list[Rendered]:
    """The 1-based ``pages`` of the PDF at ``path`` as PNG in a bounded child.

    The scale is ``dpi`` unless that would pass ``max_width`` pixels (uncropped) or
    ``limits.max_pixels`` for the page. ``crop`` cuts each page to the box around everything
    that is not white, grown by ``pad_pt`` points. Raises :class:`PdfRefused` on any bound and
    ``RuntimeError`` naming the extra when PDFium is not installed.
    """
    if not available():
        raise RuntimeError(NEEDS)
    limits = limits or RenderLimits()
    where = Path(path).expanduser()
    try:
        size = where.stat().st_size
    except OSError as exc:
        raise PdfRefused(f"cannot open {where.name}: {exc.strerror or exc}") from exc
    if size > limits.max_bytes:
        raise PdfRefused(f"the file is {size} bytes, over {limits.max_bytes}")
    request = {"path": str(where), "pages": [int(p) for p in pages], "dpi": int(dpi),
               "crop": bool(crop), "max_width": int(max_width), "pad_pt": float(pad_pt),
               "limits": limits.__dict__}
    answer = run_child(CHILD, request, timeout_s=limits.timeout_s,
                       max_out=int(limits.max_out * 4 / 3) + 4096)
    return [Rendered(int(p["page"]), base64.b64decode(p["png"]), int(p["w"]), int(p["h"]),
                     bool(p["blank"])) for p in answer["pages"]]
