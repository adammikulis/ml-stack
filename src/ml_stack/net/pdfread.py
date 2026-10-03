"""Reading a PDF nobody vouches for, with MIT-licensed code (pdfminer.six) and hard bounds.

:func:`load` runs ``ml_stack.net.pdfchild`` in a child interpreter under :class:`Limits` and takes
plain JSON back; every bound crossed (file size, page count, a stream inflating, characters per
page, total text, time, memory) or unreadable or encrypted file raises :class:`PdfRefused`.
Text a reader cannot see (invisible, white, tiny, transparent, off the page, in a layer that is
off) comes back marked ``hidden``. The result mimics the part of MuPDF's API that
:mod:`ml_stack.sources.pdf` uses. MuPDF (AGPL-3.0) is chosen only by ``ML_STACK_PDF_ENGINE=pymupdf``.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["ENGINE_ENV", "Limits", "Pdf", "PdfRefused", "engine", "load"]

ENGINE_ENV = "ML_STACK_PDF_ENGINE"
ENGINES = ("pdfminer", "pymupdf")


CHILD = Path(__file__).with_name("pdfchild.py")
"""The reader that runs in the child. It imports only the standard library and pdfminer, so it is
run by path (``-P``: nothing from its folder joins the module path) and does not depend on which
copy of ml_stack the child interpreter would find."""

class PdfRefused(ValueError):
    """The PDF was not read, and why: too big, too many pages, a bomb, too slow, encrypted, bad."""


@dataclass(frozen=True)
class Limits:
    """The bounds one read runs under. Every one is hard: crossing it refuses the file."""

    max_bytes: int = 64 * 1024 * 1024
    max_pages: int = 3000
    max_stream: int = 48 * 1024 * 1024
    max_chars: int = 24_000_000
    max_out: int = 96 * 1024 * 1024
    timeout_s: float = 120.0
    memory_bytes: int = 3 * 1024 ** 3
    max_images: int = 400
    max_page_chars: int = 8000
    """Layout analysis grows with the square of the characters scattered over a page (6,000
    single characters at random positions cost ten seconds, 20,000 over a minute), and a real
    page holds three to six thousand, so a page with more than this is refused before it is laid out."""


def engine() -> str:
    """The engine to read with: ``pdfminer`` unless the person asked for ``pymupdf`` by name.

    MuPDF is AGPL-3.0, so it is never picked because it happens to be installed: only the
    environment variable ``ML_STACK_PDF_ENGINE=pymupdf`` selects it, and it still needs the
    ``pdf-agpl`` extra.
    """
    asked = os.environ.get(ENGINE_ENV, "").strip().lower() or "pdfminer"
    if asked not in ENGINES:
        raise ValueError(f"{ENGINE_ENV}={asked!r}: one of {', '.join(ENGINES)}")
    return asked


# -- the parent: a child process, a path, limits, JSON back ----------------------------------


def load(path: str | Path, *, limits: Limits | None = None, images: bool = False,
         max_width: int = 768, text_limit: int | None = None) -> Pdf:
    """The PDF at ``path`` read in a bounded child; raises :class:`PdfRefused` on any bound.

    ``text_limit`` stops the layout work after about that many characters (the page count is
    still taken from the whole page tree). ``images`` also renders each embedded picture to PNG
    no wider than ``max_width``.
    """
    limits = limits or Limits()
    where = Path(path).expanduser()
    try:
        size = where.stat().st_size
    except OSError as exc:
        raise PdfRefused(f"cannot open {where.name}: {exc.strerror or exc}") from exc
    if size > limits.max_bytes:
        raise PdfRefused(f"the file is {size} bytes, over {limits.max_bytes}")
    request = {"path": str(where), "images": images, "max_width": max_width,
               "text_limit": text_limit, "limits": limits.__dict__}
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TMPDIR", "TEMP")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryFile() as out:
        try:
            done = subprocess.run(
                [sys.executable, "-P", str(CHILD)], input=json.dumps(request).encode(),
                stdout=out, stderr=subprocess.PIPE, timeout=limits.timeout_s, check=False, env=env,
                cwd=tempfile.gettempdir())
        except subprocess.TimeoutExpired as exc:
            raise PdfRefused(f"reading took more than {limits.timeout_s:g} s") from exc
        end = out.seek(0, 2)
        if end > limits.max_out:
            raise PdfRefused(f"the reading is {end} bytes, over {limits.max_out}")
        out.seek(0)
        raw = out.read()
    try:
        answer = json.loads(raw)
    except ValueError:
        tail = done.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no output"]
        raise PdfRefused(f"the reader stopped (exit {done.returncode}): {tail[0][:200]}") from None
    if "error" in answer:
        raise PdfRefused(str(answer["error"]))
    return Pdf(answer)


class _Rect:
    def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def width(self) -> float:
        return self.x1 - self.x0


class _Page:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.rect = _Rect(0.0, 0.0, float(data["width"]), float(data["height"]))

    def get_text(self, kind: str = "text") -> str:
        blocks = self._data["blocks"]
        if kind == "dict":
            return {"width": self.rect.width, "height": self.rect.height,  # type: ignore[return-value]
                    "blocks": blocks}
        lines = ["".join(s["text"] for s in line["spans"] if not s.get("hidden"))
                 for block in blocks for line in block["lines"]]
        return "\n".join(lines) + "\n"

    def images(self) -> list[dict[str, Any]]:
        return list(self._data["images"])

    def get_images(self, full: bool = False) -> list[tuple[int]]:
        return [(i,) for i in range(len(self._data["images"]))]

    def get_image_rects(self, order: int) -> list[_Rect]:
        return [_Rect(*self._data["images"][order]["bbox"])]


class Pdf:
    """A read PDF, shaped like the part of MuPDF's document that the callers use."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._pages = [_Page(p) for p in data["pages"]]
        self.page_count = int(data["page_count"])
        self.metadata = {str(k).lower(): str(v) for k, v in (data.get("metadata") or {}).items()}
        self._toc = [list(e) for e in data.get("toc") or ()]

    def __getitem__(self, index: int) -> _Page:
        # pages past the text limit are counted but were not laid out
        return self._pages[index] if index < len(self._pages) else _Page(
            {"width": 0, "height": 0, "blocks": [], "images": []})

    def __iter__(self) -> Any:
        return iter(self._pages)

    def get_toc(self) -> list[list[Any]]:
        return self._toc

    def close(self) -> None:
        self._pages = []

    def __enter__(self) -> Pdf:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def image_png(self, page: int, order: int) -> tuple[bytes, int, int]:
        """``(png, width, height)`` of the ``order``-th picture of 0-based ``page``; empty if none."""
        images = self[page].images()
        if not 0 <= order < len(images) or not images[order].get("png"):
            return b"", 0, 0
        one = images[order]
        return base64.b64decode(one["png"]), int(one["w"]), int(one["h"])
