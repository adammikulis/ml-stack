"""The document shape every reader in `ml_stack.sources` returns, and the last cut of it.

`Document`, `Chapter`, `Section`, `Figure` and `Unit` are what a reader -- `pdf.read`,
`html.read`, `html.read_xml` -- builds; `units()` is the split every reader shares: a
section over `max_tokens` is cut on paragraph boundaries, never mid-paragraph, into `Unit`s
a model reads one at a time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MAX_TOKENS",
    "Chapter",
    "Document",
    "Figure",
    "Section",
    "Unit",
    "is_question_bank",
    "question_banks",
    "units",
]

MAX_TOKENS = 2500
"""Where a long section is cut. Big enough that a definition and its example stay together,
small enough that a source's longest section still leaves room for the answer."""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).casefold()).strip("-") or "untitled"


@dataclass(frozen=True)
class Figure:
    """One picture, its caption, and the page it was on.

    ``png`` is empty unless a reader was asked for images.
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
    """One chapter and the sections under it, in the order the source prints them."""

    number: str
    title: str
    first_page: int = 0
    last_page: int = 0
    sections: list[Section] = field(default_factory=list)


@dataclass
class Document:
    """A source, read. ``how`` says which reading found the sections."""

    path: str
    title: str
    page_count: int = 0
    openstax: bool = False
    how: str = "toc"
    url: str = ""                  # where a web source was read from
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
    url: str = ""                  # where a web source was read from
    # Which "2.1" this is. A source that prints its section headings again in a review has
    # two of them, and a unit id that collided would make the second overwrite the first in
    # the store and in the progress file -- silently, and only in the sources that do it.
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
        out = {"source": self.source, "chapter": self.chapter, "section": self.section or "",
              "page": self.first_page, "pages": [self.first_page, self.last_page],
              "unit": self.id}
        if self.url:
            out["url"] = self.url
        return out


_QUESTION = re.compile(r"^\s*\d{1,3}\.\s+\S")
_OPTION = re.compile(r"^\s*[a-eA-E]\.\s+\S")


def is_question_bank(text: str, *, options: int = 12, share: float = 0.4) -> bool:
    """Whether a stretch of text is a chapter-end question bank rather than prose.

    A part is a question bank when it carries at least ``options`` lettered answers and
    questions and answers together are ``share`` of its lines.
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
    its own part rather than being cut in half. A part that is a question bank
    (`is_question_bank`) is left out unless ``keep_questions``.
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
                figures=list(section.figures) if index == 1 else [], url=document.url))
    if not keep_questions:
        out = [unit for unit in out if not is_question_bank(unit.text)]
    return _unique(out)


def _unique(found: list[Unit]) -> list[Unit]:
    """Every unit with an id of its own, whatever the source did with its headings.

    The id is what the progress file, the store's documents and every node's provenance are
    keyed on, so two units sharing one would make the second overwrite the first everywhere
    at once.
    """
    seen: dict[str, int] = {}
    for unit in found:
        count = seen.get(unit.id, 0) + 1
        seen[unit.id] = count
        if count > 1:
            unit.seen = count
    return found
