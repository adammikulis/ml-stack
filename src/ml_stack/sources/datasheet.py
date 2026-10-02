"""A component datasheet PDF read for what an electronics workflow needs: the text, the pin
tables as rows, the pages that draw the package outline and land pattern, and those pages as
PNG crops a vision model can read."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.sources.pdf import _pymupdf

PIN_PAGE = re.compile(r"pin\s+(description|function|configuration|definition|assignment)s?|"
                      r"terminal\s+(function|description)s?|pinout|ball\s+(description|assignment)",
                      re.IGNORECASE)
NAME_HEAD = re.compile(r"name|signal|symbol|mnemonic|function after reset", re.IGNORECASE)
TYPE_HEAD = re.compile(r"\btype\b|i/o|\bio\b|direction|dir\.", re.IGNORECASE)
DESC_HEAD = re.compile(r"descr|function|remark|comment", re.IGNORECASE)
NUMBER_HEAD = re.compile(r"\bno\b|number|\bnum\b|\bpin\b|\bball\b|\bpad\b|#", re.IGNORECASE)
PIN_ID = re.compile(r"^[A-Z]{0,2}\d{1,3}([,/\s-]+[A-Z]{0,2}\d{1,3})*$|^[A-Z]{1,2}\d{1,2}$")

SECTIONS: dict[str, tuple[tuple[str, int], ...]] = {
    "land_pattern": (("land pattern", 6), ("recommended footprint", 6), ("board layout", 3),
                     ("solder pad", 3), ("pcb footprint", 5), ("stencil", 2),
                     ("recommended pcb", 4), ("example board layout", 4)),
    "package_outline": (("package outline", 6), ("package dimensions", 6),
                        ("package information", 4), ("package drawing", 6),
                        ("mechanical data", 4), ("mechanical drawing", 5),
                        ("outline drawing", 5), ("package mechanical", 5), ("dimensions in mm", 2),
                        ("top view", 1), ("side view", 1), ("bottom view", 1)),
}
MIN_PART = 5
"""The fewest characters of a part number worth matching."""
STRONG = 3
"""The weight a section phrase needs to make a page a candidate on its own."""
DRAWING_STROKES = 40
CROP_PAD_PT = 14.0
MAX_PX = 1568


@dataclass
class Pin:
    """One row of a pin table: the pin number(s) and name, its type and description, the page
    it was on, and any other column keyed by its header."""

    number: str
    name: str
    type: str = ""
    description: str = ""
    page: int = 0
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class Outline:
    """A page that probably draws a package outline or land pattern."""

    page: int
    kind: str
    score: int
    heading: str


def _clean(cell: Any) -> str:
    return " ".join(str(cell or "").split())


def text(path: str | Path, *, limit: int | None = None) -> tuple[str, str, int]:
    """``(title, text, page_count)`` of a PDF: the text of its pages in order, cut at ``limit``
    characters when given. The title is the metadata title or ""."""
    pymupdf = _pymupdf()
    with pymupdf.open(str(Path(path).expanduser())) as doc:
        out: list[str] = []
        size = 0
        for page in doc:
            out.append(page.get_text())
            size += len(out[-1])
            if limit is not None and size >= limit:
                break
        return _clean((doc.metadata or {}).get("title")), "".join(out)[:limit], doc.page_count


def _squash(value: str) -> str:
    return re.sub(r"[\s\u2010-\u2015\-_]+", "", value).lower()


def part_match(path: str | Path, part: str) -> tuple[float, bool]:
    """How much of ``part`` the PDF names: ``(share, on_first_page)``.

    The share is the longest leading part of the number found in the text, over its length,
    so a datasheet for ``STM32U575xx`` matches ``STM32U575CIT6`` at 9/13 and a datasheet for
    something else matches 0. Under five characters nothing counts. Case, spacing and dashes
    are ignored.
    """
    pymupdf = _pymupdf()
    wanted = _squash(part)
    with pymupdf.open(str(Path(path).expanduser())) as doc:
        pages = [_squash(page.get_text()) for page in doc]
    for size in range(len(wanted), MIN_PART - 1, -1):
        head = wanted[:size]
        if any(head in body for body in pages):
            return size / len(wanted), head in pages[0]
    return 0.0, False


def mentions(path: str | Path, needle: str) -> bool:
    """Whether the PDF's text contains ``needle``, ignoring case, spacing and dashes."""
    return part_match(path, needle)[0] == 1.0


def _header_rows(rows: list[list[str]]) -> int:
    """How many leading rows are header: one, or two when the second is column labels."""
    if len(rows) < 2:
        return 1
    second = rows[1]
    body = PIN_ID.match(second[0].strip()) if second and second[0].strip() else None
    return 1 if body else 2 if second[0] == "" or NUMBER_HEAD.search(second[0]) else 1


def _columns(rows: list[list[str]], count: int) -> dict[str, Any] | None:
    """Which column is the pin number, name, type and description, or None for no pin table."""
    top = rows[:count]
    heads: list[str] = []
    for col in range(len(rows[0])):
        carried = ""
        parts = []
        for row in top:
            cell = row[col] if col < len(row) else ""
            if cell:
                carried = cell
            parts.append(cell or (carried if row is top[0] else ""))
        heads.append(" ".join(p for p in parts if p))
    found: dict[str, Any] = {}
    for col, head in enumerate(heads):
        for role, pattern in (("name", NAME_HEAD), ("type", TYPE_HEAD), ("description", DESC_HEAD),
                              ("number", NUMBER_HEAD)):
            if pattern.search(head):
                found.setdefault(role, col)
                break
    found["heads"] = heads
    return found if "number" in found and "name" in found else None


def _rows(rows: list[list[str]], cols: dict[str, Any], page: int, skip: int) -> list[Pin]:
    pins: list[Pin] = []
    for row in rows[skip:]:
        def cell(role: str, row: list[str] = row) -> str:
            col = cols.get(role)
            return row[col] if isinstance(col, int) and col < len(row) else ""
        number, name = cell("number"), cell("name")
        if not number and pins:
            pins[-1].description = _clean(f"{pins[-1].description} {cell('description')}")
            continue
        if not number and not name:
            continue
        used = {c for c in (cols.get(r) for r in ("number", "name", "type", "description"))
                if isinstance(c, int)}
        extra = {cols["heads"][i]: v for i, v in enumerate(row)
                 if i not in used and v and i < len(cols["heads"])}
        pins.append(Pin(number=number, name=name, type=cell("type"),
                        description=cell("description"), page=page, extra=extra))
    return pins


def pin_tables(path: str | Path, *, pages: int | None = None) -> list[Pin]:
    """Every row of the datasheet's pin tables ('Pin number | Name | Type | Description').

    A table is a PyMuPDF table whose header, one row or two, has a pin-number column and a
    name column; a table that continues on the next page without repeating the header is
    followed. Only pages whose text mentions a pin description are searched. ``pages`` limits
    how many pages from the start are read.
    """
    pymupdf = _pymupdf()
    pins: list[Pin] = []
    with pymupdf.open(str(Path(path).expanduser())) as doc:
        last = doc.page_count if pages is None else min(pages, doc.page_count)
        layout: dict[str, Any] | None = None
        for index in range(last):
            page = doc[index]
            if not PIN_PAGE.search(page.get_text()) and layout is None:
                continue
            found = False
            for table in page.find_tables().tables:
                rows = [[_clean(c) for c in row] for row in table.extract()]
                if len(rows) < 2:
                    continue
                skip = _header_rows(rows)
                cols = _columns(rows, skip)
                if cols is not None:
                    layout, found = cols, True
                    pins += _rows(rows, cols, index + 1, skip)
                elif layout is not None and len(rows[0]) == len(layout["heads"]) \
                        and PIN_ID.match(rows[0][layout["number"]] or "x"):
                    found = True
                    pins += _rows(rows, layout, index + 1, 0)
            if not found:
                layout = None
    return pins


def outline_pages(path: str | Path) -> list[Outline]:
    """Pages that probably draw the package outline or the recommended land pattern, best
    first. A page scores for the section phrases in its text and for how much vector drawing
    it carries; ``page`` is 1-based."""
    pymupdf = _pymupdf()
    found: list[Outline] = []
    with pymupdf.open(str(Path(path).expanduser())) as doc:
        for index in range(doc.page_count):
            page = doc[index]
            body = page.get_text().lower()
            drawn = len(page.get_drawings()) >= DRAWING_STROKES
            for kind, phrases in SECTIONS.items():
                hits = [weight for phrase, weight in phrases if phrase in body]
                if drawn and any(weight >= STRONG for weight in hits):
                    found.append(Outline(index + 1, kind, sum(hits) + 3,
                                         _heading(page, phrases)))
    return sorted(found, key=lambda o: (-o.score, o.page))


def _heading(page: Any, phrases: tuple[tuple[str, int], ...]) -> str:
    """The first line of the page that holds one of the section phrases."""
    for line in page.get_text().splitlines():
        if any(phrase in line.lower() for phrase, _ in phrases):
            return _clean(line)[:120]
    return ""


def _drawing_box(page: Any, pymupdf: Any) -> Any:
    """The box around the page's vector drawing, without the full-page frame, or the page."""
    area = page.rect.width * page.rect.height
    boxes = [d["rect"] for d in page.get_drawings()
             if d["rect"].width * d["rect"].height < 0.8 * area and not d["rect"].is_empty]
    if not boxes:
        return page.rect
    union = pymupdf.Rect(boxes[0])
    for box in boxes[1:]:
        union |= box
    pad = pymupdf.Rect(union.x0 - CROP_PAD_PT, union.y0 - CROP_PAD_PT,
                       union.x1 + CROP_PAD_PT, union.y1 + CROP_PAD_PT)
    return pad & page.rect


def render(path: str | Path, page: int, *, crop: bool = True, dpi: int = 150) -> bytes:
    """PNG bytes of a 1-based page, cropped to its drawing when ``crop``, no wider than
    ``MAX_PX`` pixels (the resolution drops before the width goes over)."""
    pymupdf = _pymupdf()
    with pymupdf.open(str(Path(path).expanduser())) as doc:
        held = doc[page - 1]
        box = _drawing_box(held, pymupdf) if crop else held.rect
        scale = min(dpi / 72.0, MAX_PX / max(box.width, 1.0))
        return bytes(held.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=box
                                     ).tobytes("png"))
