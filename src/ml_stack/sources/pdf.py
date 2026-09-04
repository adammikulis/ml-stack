"""A textbook PDF read into the units an extractor can take.

A book is not a document to a model. A thousand-page PDF has no prompt that fits it, and
the obvious cut -- a page, or a fixed number of characters -- cuts through the middle of a
definition, so the sentence that says what a thing *is* arrives without the name it defines.
The unit that survives being read alone is a **section**: it is what the book itself decided
was one idea, it names itself, and it carries its own figures.

So :func:`read` gives back a :class:`Document` of :class:`Chapter`s of :class:`Section`s.
A publisher's PDF carries a table of contents (``doc.get_toc()``) and that is believed when
it is there; a book printed to PDF by a browser has none, and then the headings are found by
the way they are set -- a section heading is numbered ``N.M`` and set larger than the body,
a chapter opens with ``CHAPTER N`` above the largest line on its page. Which of the two was
used is on the document as ``how``, because "the sections look wrong" is answered by knowing
which.

The text of a section is cleaned the way reading it aloud would clean it: a word broken
across a line is put back together, the running head and foot and the page number are
dropped (they are found by *repetition* -- a line in the top or bottom margin that appears
on a quarter of the pages is furniture, whatever it says), and the two things worth keeping
that are not prose -- a figure's caption and a bolded key term -- are kept and labelled, so
the extractor can tell a caption from a sentence.

:func:`units` is the last cut: a section longer than about 2,500 tokens is split on
paragraph boundaries, never mid-paragraph, and each part says which part it is.

Nothing here asks a model anything. It is a reader, so a test builds a two-page PDF and
reads it back in a second.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Chapter", "Document", "Figure", "IMAGE_WIDTH", "MAX_TOKENS", "Section", "Unit",
           "is_openstax", "read", "units"]

IMAGE_WIDTH = 768
"""How wide a figure is handed to a model, at most. A textbook plate is 2,000 pixels across
and a vision tower sees a few hundred; the rest is bytes in the prompt and nothing else."""

MAX_TOKENS = 2500
"""Where a long section is cut. Big enough that a definition and its example stay together,
small enough that a book's longest section still leaves room for the answer."""

LICENCE_PAGES = 12
"""How far into a book :func:`is_openstax` looks for the licence page."""

MARGIN = 0.07
"""The share of a page's height, top and bottom, that counts as its margin."""

FURNITURE_SHARE = 0.2
"""A margin line repeated on this share of the pages is a running head or foot."""

_SECTION = re.compile(r"^(\d{1,2})\.(\d{1,2})\s+(\S.*)$")
_CHAPTER = re.compile(r"^chapter\s+(\d{1,3})\b\s*(.*)$", re.IGNORECASE)
_CAPTION = re.compile(r"^(FIGURE|TABLE)\s+(\d{1,2}\.\d{1,3})\b[\s.]*(.*)$", re.DOTALL)
_NUMBER = re.compile(r"^[ivxlcdm\d.,()\s-]+$", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
_BOLD = re.compile(r"bold|black|heavy|semibold", re.IGNORECASE)
_SPACE = re.compile(r"[ \t]+")


def _pymupdf() -> Any:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - depends on what is installed
        raise ImportError(
            "reading a PDF needs pymupdf: pip install 'ml-stack[pdf]'"
        ) from exc
    return pymupdf


# -- what comes back ----------------------------------------------------------------------


@dataclass(frozen=True)
class Figure:
    """One picture, its caption, and the page it was on.

    ``png`` is empty unless :func:`read` was asked for images: rendering every plate of a
    1,500-page book costs minutes and most runs never look at one.
    """

    id: str
    page: int                      # 1-based, as a reader would say it
    label: str = ""                # "FIGURE 2.9", when the caption gives one
    caption: str = ""
    png: bytes = b""
    width: int = 0
    height: int = 0

    @property
    def shown(self) -> bool:
        """Whether there is an image to hand a model, as opposed to only a caption."""
        return bool(self.png)


@dataclass
class Section:
    """One numbered section of a chapter: its text, its pages, its figures, its key terms."""

    number: str                    # "2.1"; "" for an unnumbered section ("Key Terms")
    title: str
    chapter: str = ""
    chapter_title: str = ""
    first_page: int = 0
    last_page: int = 0
    text: str = ""
    key_terms: list[str] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.number or f"{self.chapter}:{_slug(self.title)}"

    @property
    def pages(self) -> tuple[int, int]:
        return (self.first_page, self.last_page)

    @property
    def heading(self) -> str:
        return f"{self.number} {self.title}".strip()


@dataclass
class Chapter:
    """One chapter and the sections under it, in the order the book prints them."""

    number: str
    title: str
    first_page: int = 0
    last_page: int = 0
    sections: list[Section] = field(default_factory=list)


@dataclass
class Document:
    """A book, read. ``how`` is ``"toc"`` or ``"headings"`` -- which reading found these."""

    path: str
    title: str
    page_count: int = 0
    openstax: bool = False
    how: str = "toc"
    chapters: list[Chapter] = field(default_factory=list)

    @property
    def sections(self) -> list[Section]:
        return [s for c in self.chapters for s in c.sections]

    @property
    def slug(self) -> str:
        return _slug(self.title)


@dataclass
class Unit:
    """What one extraction is asked about: a section, or one part of a long one.

    Everything a node's provenance needs is here, so nothing downstream has to reach back
    into the document to say where a concept came from.
    """

    source: str                    # the document's slug
    book_title: str
    chapter: str
    chapter_title: str
    section: str
    section_title: str
    first_page: int
    last_page: int
    text: str
    part: int = 1                  # 1-based; 1 of 1 for a section that fits
    parts: int = 1
    key_terms: list[str] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    # Which "2.1" this is. A book that prints its section headings again in a chapter
    # review has two of them, and a unit id that collided would make the second overwrite
    # the first in the store and in the progress file -- silently, and only in the books
    # that do it.
    seen: int = 1

    @property
    def id(self) -> str:
        """A name for this unit that is the same on every run over the same source."""
        stem = f"{self.source}:{self.chapter or '0'}:{self.section or _slug(self.section_title)}"
        if self.seen > 1:
            stem += f"~{self.seen}"
        return stem if self.parts == 1 else f"{stem}#{self.part}"

    @property
    def where(self) -> dict[str, Any]:
        """The provenance every node and edge read out of this unit carries."""
        return {"source": self.source, "chapter": self.chapter, "section": self.section or "",
                "page": self.first_page, "pages": [self.first_page, self.last_page],
                "unit": self.id}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).casefold()).strip("-") or "untitled"


# -- reading a page into lines ------------------------------------------------------------


@dataclass(frozen=True)
class _Line:
    text: str
    size: float
    font: str
    top: float
    bottom: float
    page: int
    block: int


def _lines(page: Any, number: int) -> list[_Line]:
    """Every text line on a page, in reading order, with how it is set."""
    out: list[_Line] = []
    for index, block in enumerate(page.get_text("dict").get("blocks") or ()):
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or ():
            spans = line.get("spans") or ()
            text = _SPACE.sub(" ", "".join(s.get("text", "") for s in spans)).strip()
            if not text:
                continue
            biggest = max(spans, key=lambda s: float(s.get("size") or 0.0))
            box = line.get("bbox") or (0.0, 0.0, 0.0, 0.0)
            out.append(_Line(text=text, size=round(float(biggest.get("size") or 0.0), 1),
                             font=str(biggest.get("font") or ""), top=float(box[1]),
                             bottom=float(box[3]), page=number, block=index))
    return out


def _bold_terms(page: Any, body: float) -> list[str]:
    """The key terms a page bolds inside its body text.

    A textbook marks the word it is defining by setting it bold at body size, which is the
    one piece of typography that carries meaning rather than layout. A whole bold *sentence*
    is emphasis, not a term, so anything long or ending in a full stop is left alone.
    """
    found: list[str] = []
    for block in page.get_text("dict").get("blocks") or ():
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or ():
            for span in line.get("spans") or ():
                text = _SPACE.sub(" ", str(span.get("text") or "")).strip(" ,;:")
                size = round(float(span.get("size") or 0.0), 1)
                if not text or abs(size - body) > 0.6 or not _BOLD.search(str(span.get("font"))):
                    continue
                if len(text.split()) > 6 or text.endswith(".") or text.isupper():
                    continue
                if len(text) < 3 or _NUMBER.match(text):
                    continue
                if text not in found:
                    found.append(text)
    return found


def _furniture(pages: Sequence[Sequence[_Line]], heights: Sequence[float]) -> set[str]:
    """The running heads, feet and page numbers, found by repetition rather than by wording.

    A page number, a chapter's name across the top of every left-hand page and the
    publisher's line along the bottom all differ from book to book and none of them is prose.
    What they share is that they sit in a margin and say almost the same thing on a quarter
    of the pages, so that -- with the digits blanked, since the number is what changes -- is
    what identifies them.
    """
    seen: dict[str, int] = {}
    for lines, height in zip(pages, heights):
        top, bottom = height * MARGIN, height * (1 - MARGIN)
        here = {_DIGITS.sub("#", line.text.casefold())
                for line in lines if line.bottom <= top or line.top >= bottom}
        for text in here:
            seen[text] = seen.get(text, 0) + 1
    floor = max(3, int(len(pages) * FURNITURE_SHARE))
    return {text for text, count in seen.items() if count >= floor}


def _body_size(pages: Sequence[Sequence[_Line]]) -> float:
    """The size the book sets its prose in: the commonest line size, by how much text it sets."""
    weight: dict[float, int] = {}
    for lines in pages:
        for line in lines:
            weight[line.size] = weight.get(line.size, 0) + len(line.text)
    if not weight:
        return 0.0
    return max(sorted(weight), key=lambda size: weight[size])


# -- cleaning ------------------------------------------------------------------------------


def _join(lines: Sequence[str]) -> str:
    """Lines of one paragraph as one string, a word broken across a line put back together."""
    out = ""
    for line in lines:
        piece = line.strip()
        if not piece:
            continue
        if out.endswith("-") and not out.endswith(("--", " -")) and piece[:1].islower():
            out = out[:-1] + piece
        elif out:
            out += " " + piece
        else:
            out = piece
    return out


def _paragraphs(lines: Iterable[_Line], furniture: set[str]) -> list[tuple[str, str]]:
    """``(kind, text)`` per paragraph: ``"para"``, or ``"caption"`` for a figure's caption.

    A block of the PDF is a paragraph; the lines inside it are joined, and a block whose
    first line begins ``FIGURE 2.9`` is labelled rather than dropped, because the sentence
    under a picture is often the only place the book says what the picture shows.
    """
    out: list[tuple[str, str]] = []
    block: list[str] = []
    at: tuple[int, int] | None = None

    def flush() -> None:
        if not block:
            return
        text = _join(block)
        block.clear()
        if not text:
            return
        out.append(("caption" if _CAPTION.match(text) else "para", text))

    for line in lines:
        if _DIGITS.sub("#", line.text.casefold()) in furniture:
            continue
        here = (line.page, line.block)
        if at is not None and here != at:
            flush()
        at = here
        block.append(line.text)
    flush()
    return out


# -- figures --------------------------------------------------------------------------------


def _render(pymupdf: Any, doc: Any, xref: int, max_width: int) -> tuple[bytes, int, int]:
    """One embedded image as PNG bytes, no wider than ``max_width``. Empty on anything odd."""
    try:
        pix = pymupdf.Pixmap(doc, xref)
        if pix.n - pix.alpha >= 4 or pix.alpha:          # CMYK or transparency: to plain RGB
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        if pix.width > max_width:
            height = max(1, round(pix.height * max_width / pix.width))
            pix = pymupdf.Pixmap(pix, max_width, height)  # scaled copy
        return pix.tobytes("png"), pix.width, pix.height
    except Exception:  # noqa: BLE001 - a plate that will not render is a plate, not a failure
        return b"", 0, 0


def _figures(pymupdf: Any, doc: Any, page: Any, number: int, captions: Sequence[tuple[float, str]],
             *, images: bool, max_width: int) -> list[Figure]:
    """The pictures on one page, each with the caption nearest below it.

    A textbook puts the caption under the plate, so "nearest below" is the rule; a page whose
    only caption sits above every image still pairs them, because a caption with no picture
    is worse than a picture with the wrong one is unlikely.
    """
    out: list[Figure] = []
    for order, entry in enumerate(page.get_images(full=True)):
        xref = int(entry[0])
        try:
            boxes = page.get_image_rects(xref)
        except Exception:  # noqa: BLE001 - an image the page cannot place is still an image
            boxes = []
        bottom = max((float(b.y1) for b in boxes), default=0.0)
        below = [(y, text) for y, text in captions if y >= bottom]
        picked = min(below or list(captions), key=lambda c: abs(c[0] - bottom), default=None)
        caption = picked[1] if picked else ""
        found = _CAPTION.match(caption)
        png, width, height = (_render(pymupdf, doc, xref, max_width) if images else (b"", 0, 0))
        out.append(Figure(id=f"p{number}-{order + 1}", page=number,
                          label=f"{found.group(1).title()} {found.group(2)}" if found else "",
                          caption=caption, png=png, width=width, height=height))
    return out


# -- the table of contents, and the headings when there is none -----------------------------


def _from_toc(toc: Sequence[Sequence[Any]], page_count: int) -> list[tuple[str, str, str, int]]:
    """``(kind, number, title, page)`` from a publisher's outline; kind is chapter or section."""
    found: list[tuple[str, str, str, int]] = []
    for entry in toc:
        title = _SPACE.sub(" ", str(entry[1] or "")).strip()
        page = int(entry[2] or 0)
        if not title or not 1 <= page <= page_count:
            continue
        chapter = _CHAPTER.match(title)
        section = _SECTION.match(title)
        if chapter and not section:
            found.append(("chapter", chapter.group(1), chapter.group(2).strip() or title, page))
        elif section:
            found.append(("section", f"{section.group(1)}.{section.group(2)}",
                          section.group(3).strip(), page))
        else:
            found.append(("section", "", title, page))
    return found


def _from_headings(pages: Sequence[Sequence[_Line]], body: float,
                   furniture: set[str]) -> list[tuple[str, str, str, int]]:
    """The same list, read off the way the book is set, for a PDF with no outline.

    A section heading is a numbered line set clearly larger than the prose; a chapter opens
    with a line saying ``CHAPTER N`` and takes its title from the largest line on that page.
    Both tests are about *size relative to the body*, never an absolute point size, because
    every book sets its body differently.
    """
    floor = body * 1.2 if body else 0.0
    found: list[tuple[str, str, str, int]] = []
    for lines in pages:
        if not lines:
            continue
        number = lines[0].page
        biggest = max(line.size for line in lines)
        for line in lines:
            if _DIGITS.sub("#", line.text.casefold()) in furniture:
                continue
            chapter = _CHAPTER.match(line.text)
            if chapter and line.size >= floor and len(line.text.split()) <= 4:
                title = next((one.text for one in lines
                              if one.size == biggest and not _CHAPTER.match(one.text)), "")
                found.append(("chapter", chapter.group(1),
                              title or chapter.group(2).strip(), number))
                continue
            section = _SECTION.match(line.text)
            if section and line.size >= floor and line.size > body:
                found.append(("section", f"{section.group(1)}.{section.group(2)}",
                              section.group(3).strip(), number))
    return found


def _title_of(doc: Any, path: Path) -> str:
    """What the book is called: its metadata title when that is a title, else its filename.

    A book printed to PDF by a browser carries the URL it was printed from as its title, so
    anything with a slash in it is not one.
    """
    named = _SPACE.sub(" ", str((doc.metadata or {}).get("title") or "")).strip()
    if named and "/" not in named and not named.lower().startswith("http"):
        return named
    stem = path.stem.replace("_", " ").replace("-", " ")
    return _SPACE.sub(" ", re.sub(r"\bWEB\b", "", stem, flags=re.IGNORECASE)).strip() or path.stem


def is_openstax(doc: Any) -> bool:
    """Whether this is an OpenStax book, read off its licence page rather than its filename.

    The front matter of one names the publisher beside the Creative Commons licence it is
    released under; a file renamed by whoever downloaded it says nothing at all. It matters
    because an OpenStax book is numbered ``N.M`` throughout, which is what the heading
    reading relies on when a PDF carries no outline.
    """
    metadata = " ".join(str(v) for v in (doc.metadata or {}).values()).casefold()
    if "openstax" in metadata:
        return True
    for number in range(min(LICENCE_PAGES, doc.page_count)):
        text = doc[number].get_text().casefold()
        if "openstax" in text and "creative commons" in text:
            return True
    return False


# -- reading a book ---------------------------------------------------------------------------


def read(path: str | Path, *, images: bool = False, max_width: int = IMAGE_WIDTH,
         chapter: str | int | None = None) -> Document:
    """A textbook PDF as chapters of sections, cleaned, with its figures.

    ``images`` renders each figure to PNG bytes no wider than ``max_width``; without it a
    figure carries only its caption, which is what a run with no vision projector can use
    anyway. ``chapter`` reads one chapter and leaves the rest of the book unread -- the
    smoke run, and the only way to try a book without paying for all of it.
    """
    pymupdf = _pymupdf()
    where = Path(path).expanduser()
    doc = pymupdf.open(str(where))
    try:
        return _read(pymupdf, doc, where, images=images, max_width=max_width, chapter=chapter)
    finally:
        doc.close()


def _read(pymupdf: Any, doc: Any, where: Path, *, images: bool, max_width: int,
          chapter: str | int | None) -> Document:
    title = _title_of(doc, where)
    out = Document(path=str(where), title=title, page_count=doc.page_count,
                   openstax=is_openstax(doc), how="toc")

    pages = [_lines(doc[n], n + 1) for n in range(doc.page_count)]
    heights = [float(doc[n].rect.height) for n in range(doc.page_count)]
    furniture = _furniture(pages, heights)
    body = _body_size(pages)

    marks = _from_toc(doc.get_toc() or (), doc.page_count)
    if not any(kind == "section" for kind, *_ in marks):
        out.how = "headings"
        marks = _from_headings(pages, body, furniture)
    marks = _tidied(marks)
    if not marks:
        return out

    wanted = str(chapter) if chapter is not None else ""
    for number, name, first, last, sections in _chapters(marks, doc.page_count):
        if wanted and number != wanted:
            continue
        held = Chapter(number=number, title=name, first_page=first, last_page=last)
        for s_number, s_title, s_first, s_last in sections:
            held.sections.append(_section(pymupdf, doc, pages, furniture, body,
                                          number=s_number, title=s_title, chapter=number,
                                          chapter_title=name, first=s_first, last=s_last,
                                          images=images, max_width=max_width))
        out.chapters.append(held)
    return out


def _tidied(marks: Sequence[tuple[str, str, str, int]]) -> list[tuple[str, str, str, int]]:
    """The marks in page order, with a heading repeated on the same page counted once.

    A running head that says the section's own name, and a heading a font heuristic found
    twice, both arrive as a duplicate; the first of them is the real one because the page
    is read from the top.
    """
    seen: set[tuple[str, str, int]] = set()
    out: list[tuple[str, str, str, int]] = []
    for order, (kind, number, title, page) in enumerate(marks):
        key = (kind, number or _slug(title), page)
        if key in seen:
            continue
        seen.add(key)
        out.append((kind, number, title, page))
    return sorted(out, key=lambda m: (m[3], 0 if m[0] == "chapter" else 1))


def _chapters(marks: Sequence[tuple[str, str, str, int]], page_count: int
              ) -> Iterator[tuple[str, str, int, int, list[tuple[str, str, int, int]]]]:
    """Group the marks into chapters, each with its sections' page ranges worked out.

    A section runs to the page before the next heading of any kind, and the last one runs to
    the end of its chapter; a chapter runs to the page before the next chapter. Sections
    before the first chapter mark (a preface) are a chapter with no number, kept rather than
    dropped, because the front matter of a textbook states what the book is about.
    """
    marks = _numbered(_one_per_chapter(marks))
    chapters = [i for i, m in enumerate(marks) if m[0] == "chapter"]
    starts = [0, *chapters] if not chapters or chapters[0] != 0 else chapters
    for at, start in enumerate(starts):
        head = marks[start]
        opening = head[0] == "chapter"
        number, title = (head[1], head[2]) if opening else ("", "Front matter")
        first = head[3]
        after = starts[at + 1] if at + 1 < len(starts) else len(marks)
        last = marks[after][3] - 1 if after < len(marks) else page_count
        inside = _once([m for m in marks[start + (1 if opening else 0):after]
                        if m[0] == "section"])
        if not inside and not opening:
            continue
        last = max(first, last)
        sections: list[tuple[str, str, int, int]] = []
        for index, (_, s_number, s_title, s_page) in enumerate(inside):
            # the first section of a chapter starts where the chapter does, so the opening
            # page -- the outline, the photograph and its caption -- is read rather than
            # falling into the gap between the chapter mark and the first section mark
            begins = first if index == 0 else s_page
            ends = inside[index + 1][3] - 1 if index + 1 < len(inside) else last
            sections.append((s_number, s_title, begins, max(begins, ends)))
        if not sections and opening:
            sections = [("", title, first, last)]
        yield number, title, first, last, sections


def _one_per_chapter(marks: Sequence[tuple[str, str, str, int]]
                     ) -> list[tuple[str, str, str, int]]:
    """One mark per chapter number: the place the chapter actually starts.

    A book's table of contents names every chapter, in a line set large enough to look like
    a chapter opening, so a heading reading finds "Chapter 2" on page 7 as well as on page
    53 and cuts the book at both. The one that is followed by that chapter's sections is the
    real one; a chapter with no sections anywhere keeps its first mark.
    """
    order = [i for i, m in enumerate(marks) if m[0] == "chapter"]
    if not order:
        return list(marks)
    best: dict[str, int] = {}
    for place, index in enumerate(order):
        number = marks[index][1]
        after = order[place + 1] if place + 1 < len(order) else len(marks)
        has = any(m[0] == "section" for m in marks[index + 1:after])
        if number not in best or (has and not _has_sections(marks, order, best[number])):
            best[number] = index
    kept = set(best.values())
    return [m for i, m in enumerate(marks) if m[0] != "chapter" or i in kept]


def _has_sections(marks: Sequence[tuple[str, str, str, int]], order: Sequence[int],
                  index: int) -> bool:
    place = order.index(index)
    after = order[place + 1] if place + 1 < len(order) else len(marks)
    return any(m[0] == "section" for m in marks[index + 1:after])


def _numbered(marks: Sequence[tuple[str, str, str, int]]) -> list[tuple[str, str, str, int]]:
    """Chapter marks invented from the section numbers, for a book that prints no chapter line.

    A section called ``2.6`` says which chapter it is in whatever the page it opens on looks
    like, so a book whose chapter openings are pictures -- or which was printed to PDF
    without them -- still comes back as chapters rather than as one heap of sections. A book
    that *does* say ``CHAPTER 2`` is left exactly as it was: the book's own word wins.
    """
    if any(kind == "chapter" for kind, *_ in marks):
        return list(marks)
    out: list[tuple[str, str, str, int]] = []
    at = ""
    for mark in marks:
        number = mark[1].partition(".")[0]
        if mark[0] == "section" and number and number != at:
            at = number
            out.append(("chapter", number, f"Chapter {number}", mark[3]))
        out.append(mark)
    return out


def _once(sections: Sequence[tuple[str, str, str, int]]) -> list[tuple[str, str, str, int]]:
    """One entry per section of a chapter: the first place it is printed.

    A textbook repeats its section headings in the chapter review at the back, so the
    headings reading finds ``1.1`` twice and would cut the chapter into two sections of the
    same name -- and, downstream, into two units with the same id. The first is the section;
    the second is the review, and the review goes to the section it reviews.
    """
    out: list[tuple[str, str, str, int]] = []
    seen: set[str] = set()
    for mark in sections:
        key = mark[1] or _slug(mark[2])
        if key in seen:
            continue
        seen.add(key)
        out.append(mark)
    return out


def _section(pymupdf: Any, doc: Any, pages: Sequence[Sequence[_Line]], furniture: set[str],
             body: float, *, number: str, title: str, chapter: str, chapter_title: str,
             first: int, last: int, images: bool, max_width: int) -> Section:
    """One section's cleaned text, key terms and figures, off the pages it covers."""
    lines = [line for page in pages[first - 1:last] for line in page]
    blocks = _paragraphs(lines, furniture)
    text = "\n\n".join(
        (piece if kind == "para" else _labelled(piece)) for kind, piece in blocks if piece)

    terms: list[str] = []
    figures: list[Figure] = []
    for number_of_page in range(first, last + 1):
        page = doc[number_of_page - 1]
        for term in _bold_terms(page, body):
            if term not in terms:
                terms.append(term)
        captions = [(line.top, joined) for line, joined in _captions(pages[number_of_page - 1],
                                                                    furniture)]
        figures += _figures(pymupdf, doc, page, number_of_page, captions,
                            images=images, max_width=max_width)

    return Section(number=number, title=title, chapter=chapter, chapter_title=chapter_title,
                   first_page=first, last_page=last, text=text.strip(), key_terms=terms,
                   figures=figures)


def _labelled(caption: str) -> str:
    """A caption marked as one, so a model reading the section can tell it from the prose."""
    found = _CAPTION.match(caption)
    if not found:
        return caption
    return f"[{found.group(1).title()} {found.group(2)}] {found.group(3).strip()}".rstrip()


def _captions(lines: Sequence[_Line], furniture: set[str]) -> list[tuple[_Line, str]]:
    """The caption blocks on one page, each with the whole block joined."""
    out: list[tuple[_Line, str]] = []
    at: tuple[int, int] | None = None
    held: list[_Line] = []

    def flush() -> None:
        if held and _CAPTION.match(held[0].text):
            out.append((held[0], _join([one.text for one in held])))
        held.clear()

    for line in lines:
        if _DIGITS.sub("#", line.text.casefold()) in furniture:
            continue
        here = (line.page, line.block)
        if at is not None and here != at:
            flush()
        at = here
        held.append(line)
    flush()
    return out


# -- the last cut ------------------------------------------------------------------------------


_QUESTION = re.compile(r"^\s*\d{1,3}\.\s+\S")
_OPTION = re.compile(r"^\s*[a-eA-E]\.\s+\S")


def is_question_bank(text: str, *, options: int = 12, share: float = 0.4) -> bool:
    """Whether a stretch of text is a chapter-end question bank rather than prose.

    A textbook ends a chapter with numbered questions and lettered answers -- an AP book's
    section 5.4 had one part with 66 lettered options. Put through the extractor those run
    to the ceiling: every option is a claim the model tries to turn into concepts and
    relations, and it never reaches the end. A part is a question bank when it carries at
    least ``options`` lettered answers and questions and answers together are ``share`` of
    its lines; nothing in a graph is lost by skipping it, since the questions restate what
    the chapter's prose already said.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    answers = sum(1 for ln in lines if _OPTION.match(ln))
    asked = sum(1 for ln in lines if _QUESTION.match(ln))
    return answers >= options and (answers + asked) / len(lines) >= share


def question_banks(document: Document, *, max_tokens: int = MAX_TOKENS) -> int:
    """How many parts `units` leaves out of ``document`` as question banks."""
    return sum(1 for unit in units(document, max_tokens=max_tokens, keep_questions=True)
               if is_question_bank(unit.text))


def units(document: Document, *, max_tokens: int = MAX_TOKENS,
          keep_questions: bool = False) -> list[Unit]:
    """The document as units to extract from: a section, or one part of a long one.

    A section is split on paragraph boundaries only. A single paragraph over the ceiling is
    its own part rather than being cut in half: a cut paragraph is where a definition loses
    its subject, which is the failure this whole module exists to avoid. A part that is a
    question bank (`is_question_bank`) is left out unless ``keep_questions``.
    """
    from ml_stack.client.tokens import estimate_tokens

    out: list[Unit] = []
    for section in document.sections:
        pieces = [p for p in section.text.split("\n\n") if p.strip()]
        parts: list[list[str]] = []
        held: list[str] = []
        cost = 0
        for piece in pieces:
            size = estimate_tokens(piece)
            if held and cost + size > max_tokens:
                parts.append(held)
                held, cost = [], 0
            held.append(piece)
            cost += size
        if held or not parts:
            parts.append(held)
        for index, part in enumerate(parts, start=1):
            out.append(Unit(
                source=document.slug, book_title=document.title,
                chapter=section.chapter, chapter_title=section.chapter_title,
                section=section.number, section_title=section.title,
                first_page=section.first_page, last_page=section.last_page,
                text="\n\n".join(part).strip(), part=index, parts=len(parts),
                key_terms=list(section.key_terms) if index == 1 else [],
                figures=list(section.figures) if index == 1 else []))
    if not keep_questions:
        out = [unit for unit in out if not is_question_bank(unit.text)]
    return _unique(out)


def _unique(found: list[Unit]) -> list[Unit]:
    """Every unit with an id of its own, whatever the book did with its headings.

    The id is what the progress file, the store's documents and every node's provenance are
    keyed on, so two units sharing one would make the second overwrite the first everywhere
    at once, and only in the books that repeat a heading. Cheap to guarantee here; expensive
    to notice later.
    """
    seen: dict[str, int] = {}
    for unit in found:
        count = seen.get(unit.id, 0) + 1
        seen[unit.id] = count
        if count > 1:
            unit.seen = count
    return found
