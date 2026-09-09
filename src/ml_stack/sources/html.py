"""HTML and XML sources read into the units an extractor can take.

`read` finds a page's `<h1>`-`<h3>` headings, tests each against a `SectionRule` and reads
`(number, title)` out of the ones that match; the text between two matched headings is its
section, cleaned with `ml_stack.markup.extract`. `read_xml` walks an XML tree instead of
guessing from headings: each `section_tag` element is a section, numbered by its
`id_attr`. Both give back the `Document` of `Chapter`s of `Section`s that
`ml_stack.sources.pdf` does, so `ml_stack.sources.units.units` splits either one the same
way.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from html import unescape
from pathlib import Path

from ml_stack.home import expand
from ml_stack.sources.units import Chapter, Document, Section

__all__ = ["DEFAULT_RULE", "SectionRule", "read", "read_xml"]

_HEADING = re.compile(r"<h[1-3][^>]*>(.*?)</h[1-3]\s*>", re.IGNORECASE | re.DOTALL)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class SectionRule:
    """A pattern over a heading's text, and ``(number, title)`` out of what it matches."""

    pattern: re.Pattern[str]
    parse: Callable[[re.Match[str]], tuple[str, str]]


def _plain(match: re.Match[str]) -> tuple[str, str]:
    return "", match.group(1).strip()


DEFAULT_RULE = SectionRule(pattern=re.compile(r"^\s*(\S.*)$"), parse=_plain)
"""Every non-blank heading is its own unnumbered section, titled by its text."""


def _text(fragment: str) -> str:
    """A heading's markup, tags stripped and entities unescaped."""
    return " ".join(unescape(_TAG.sub(" ", fragment)).split())


def _paragraphed(text: str) -> str:
    """``extract``'s lines as blank-line-separated paragraphs, whichever backend joined them."""
    return "\n\n".join(line.strip() for line in text.splitlines() if line.strip())


def _located(source: str | Path) -> tuple[Path | None, str]:
    """The file ``source`` names, if it names one; else ``source`` itself as markup."""
    if isinstance(source, Path):
        return source, str(source)
    candidate = expand(str(source))
    if candidate.is_file():
        return candidate, str(candidate)
    return None, str(source)


def _markup(source: str | Path) -> tuple[str, str]:
    """``(markup, path)`` -- path is empty when ``source`` was markup, not a file."""
    path, text = _located(source)
    return (path.read_text(encoding="utf-8"), text) if path else (text, "")


def _xml_bytes(source: str | Path) -> tuple[bytes, str]:
    """``(bytes, path)`` -- bytes so `xml.etree` reads an encoding declaration itself."""
    path, text = _located(source)
    return (path.read_bytes(), text) if path else (text.encode("utf-8"), "")


def read(source: str | Path, *, url: str = "",
        sections: SectionRule | None = None) -> Document:
    """A page's headings and text as a `Document` of one `Chapter`.

    ``sections`` picks which headings start a section and reads its number and title out of
    one; the default takes every heading. ``source`` is a path to a file or a string of
    markup. Body text goes through `ml_stack.markup.extract`.
    """
    from ml_stack.markup import extract

    rule = sections or DEFAULT_RULE
    markup, path = _markup(source)
    title_match = _TITLE.search(markup)
    title = _text(title_match.group(1)) if title_match else ""
    doc = Document(path=path or url, title=title or url or "untitled", how="headings",
                   url=url)
    chapter = Chapter(number="", title=doc.title)
    matched = [(m, rule.pattern.match(_text(m.group(1)))) for m in _HEADING.finditer(markup)]
    matched = [(m, found) for m, found in matched if found]
    for index, (match, found) in enumerate(matched):
        number, section_title = rule.parse(found)
        start = match.end()
        end = matched[index + 1][0].start() if index + 1 < len(matched) else len(markup)
        _, text = extract(f"<div>{markup[start:end]}</div>", url)
        chapter.sections.append(Section(number=number, title=section_title,
                                        chapter_title=doc.title, text=_paragraphed(text)))
    if chapter.sections:
        doc.chapters = [chapter]
    return doc


def _local(tag: str) -> str:
    """An XML tag's name, its namespace dropped."""
    return tag.rpartition("}")[2].casefold()


def _xml_title(root: ET.Element) -> str:
    for element in root.iter():
        if _local(element.tag) == "title" and (element.text or "").strip():
            return " ".join(element.text.split())
    return ""


def _xml_heading(element: ET.Element) -> tuple[str, str]:
    """``(the heading child's tag, its text)``, or ``("", "")`` when there is none."""
    for child in element:
        tag = _local(child.tag)
        if tag in ("title", "heading", "head") and (child.text or "").strip():
            return tag, " ".join(child.text.split())
    return "", ""


def _xml_text(element: ET.Element, *, skip: str) -> str:
    """One paragraph per direct child other than ``skip``, joined blank-line apart."""
    paragraphs = []
    for child in element:
        if _local(child.tag) == skip:
            continue
        text = " ".join(" ".join(child.itertext()).split())
        if text:
            paragraphs.append(text)
    if paragraphs:
        return "\n\n".join(paragraphs)
    return " ".join(" ".join(element.itertext()).split())


def read_xml(source: str | Path, *, url: str = "", section_tag: str = "section",
            id_attr: str = "id") -> Document:
    """An XML document's ``section_tag`` elements as a `Document` of one `Chapter`.

    Each element numbers its section from ``id_attr`` and titles it from a ``title`` or
    ``heading`` child, falling back to the id. ``source`` is a path to a file or a string of
    markup.
    """
    raw, path = _xml_bytes(source)
    root = ET.fromstring(raw)
    title = _xml_title(root)
    doc = Document(path=path or url, title=title or url or "untitled", how="markup", url=url)
    chapter = Chapter(number="", title=doc.title)
    for element in root.iter(section_tag):
        number = element.get(id_attr, "")
        heading_tag, heading_text = _xml_heading(element)
        text = _xml_text(element, skip=heading_tag)
        chapter.sections.append(Section(number=number, title=heading_text or number or
                                        "untitled", chapter_title=doc.title, text=text))
    if chapter.sections:
        doc.chapters = [chapter]
    return doc
