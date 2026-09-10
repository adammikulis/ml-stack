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
from typing import Any

from ml_stack.home import expand
from ml_stack.sources.units import Chapter, Document, Section

__all__ = ["DEFAULT_RULE", "Marks", "SectionRule", "read", "read_xml", "rule_for"]

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


def _grouped(match: re.Match[str]) -> tuple[str, str]:
    """``(number, title)`` from a pattern's first two groups; one group is the title."""
    found = match.groups()
    if len(found) >= 2:
        return (found[0] or "").strip(), (found[1] or "").strip()
    return "", ((found[0] if found else match.group(0)) or "").strip()


def rule_for(pattern: str) -> SectionRule | None:
    """A `SectionRule` from a pattern whose first group is a section's number and whose
    second is its title; None for an empty pattern."""
    return SectionRule(pattern=re.compile(pattern), parse=_grouped) if pattern else None


@dataclass(frozen=True)
class Marks:
    """How one source marks a section, for whichever reader takes it.

    In XML: ``section_tag`` is the element, and a section's number comes from the
    ``number_tag`` child -- where a statute puts it -- else the ``id_attr`` attribute; its
    title comes from the ``title_tag`` child. In HTML: ``pattern`` is what a heading must
    match, its first group the number and its second the title. Every tag and attribute
    name is matched without its namespace and without regard to case.
    """

    section_tag: str = "section"
    id_attr: str = "id"
    number_tag: str = ""
    title_tag: str = ""
    pattern: str = ""

    def rule(self) -> SectionRule | None:
        """`pattern` as a `SectionRule`, or None when there is none."""
        return rule_for(self.pattern)

    @classmethod
    def from_args(cls, args: Any) -> Marks:
        """The marks a command line asked for: ``--section-tag``, ``--id-attr``,
        ``--number-tag``, ``--title-tag`` and ``--section-pattern``."""
        def said(name: str, fallback: str = "") -> str:
            return str(getattr(args, name, "") or "") or fallback

        return cls(section_tag=said("section_tag", "section"),
                   id_attr=said("id_attr", "id"), number_tag=said("number_tag"),
                   title_tag=said("title_tag"), pattern=said("section_pattern"))


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


def _attribute(element: ET.Element, name: str) -> str:
    """That attribute of the element, matched without its namespace or its case."""
    wanted = _local(name)
    for key, value in element.attrib.items():
        if _local(key) == wanted:
            return value
    return ""


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


def _child_text(element: ET.Element, name: str) -> str:
    """That direct child's text, matched without its namespace or its case."""
    wanted = _local(name)
    for child in element:
        if _local(child.tag) == wanted:
            return " ".join(" ".join(child.itertext()).split())
    return ""


def _without(element: ET.Element, *, skip: tuple[str, ...]) -> str:
    """One paragraph per direct child whose tag is not in ``skip``."""
    unwanted = {_local(name) for name in skip if name}
    paragraphs = [" ".join(" ".join(child.itertext()).split()) for child in element
                  if _local(child.tag) not in unwanted]
    return "\n\n".join(p for p in paragraphs if p)


def read_xml(source: str | Path, *, url: str = "", marks: Marks | None = None) -> Document:
    """An XML document's sections as a `Document` of one `Chapter`.

    ``marks`` says which element a section is and where its number and title are -- see
    `Marks`; without one, every ``section`` element numbered by its ``id``. A tag named for
    the number or the title is kept out of the section's text, and a section with no title
    of its own falls back to a ``title`` or ``heading`` child and then to its number.
    ``source`` is a path to a file or a string of markup.
    """
    marks = marks or Marks()
    section_tag, id_attr = marks.section_tag, marks.id_attr
    number_tag, title_tag = marks.number_tag, marks.title_tag
    raw, path = _xml_bytes(source)
    root = ET.fromstring(raw)
    title = _xml_title(root)
    doc = Document(path=path or url, title=title or url or "untitled", how="markup", url=url)
    chapter = Chapter(number="", title=doc.title)
    wanted = _local(section_tag)
    for element in root.iter():
        if _local(element.tag) != wanted:
            continue
        number = (_child_text(element, number_tag) if number_tag else "") \
            or _attribute(element, id_attr)
        if title_tag:
            heading_tag, heading_text = title_tag, _child_text(element, title_tag)
        else:
            heading_tag, heading_text = _xml_heading(element)
        text = (_without(element, skip=(heading_tag, number_tag))
                if number_tag or title_tag else _xml_text(element, skip=heading_tag))
        chapter.sections.append(Section(number=number, title=heading_text or number or
                                        "untitled", chapter_title=doc.title, text=text))
    if chapter.sections:
        doc.chapters = [chapter]
    return doc
