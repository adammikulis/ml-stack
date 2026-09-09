"""HTML and XML read into the units an extractor can take, and the reader chosen by
suffix or URL scheme.

Every fixture here is markup the test writes inline: an invented statute, an invented
handbook, nothing fetched and nothing read off this machine.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ml_stack import ingest
from ml_stack.sources import html, pdf, units
from ml_stack.web import Refused

XML = """<?xml version="1.0" encoding="UTF-8"?>
<statute>
  <title>Velthorne Lattice Act</title>
  <section id="s-4">
    <heading>Definitions</heading>
    <para>In this Act, glimmer node means a node that stores charge.</para>
    <para>Vault means a housing for one or more glimmer nodes.</para>
  </section>
  <section id="s-5">
    <heading>Application</heading>
    <para>This Act applies to every vault operated in the province.</para>
  </section>
</statute>
"""

PAGE = """
<html><head><title>Glimmer Handbook</title></head>
<body>
<h2>2.1 Glimmer Nodes</h2>
<p>A glimmer node is the smallest part of a lattice that can hold a charge.</p>
<p>Every node sits inside a vault.</p>
<h2>2.2 Vault Currents</h2>
<p>Vault currents run between quickened nodes.</p>
</body></html>
"""

NUMBERED = html.SectionRule(
    pattern=re.compile(r"^(\d+)\.(\d+)\s+(\S.*)$"),
    parse=lambda m: (f"{m.group(1)}.{m.group(2)}", m.group(3)))


def test_read_xml_sections_keep_their_markup_ids():
    document = html.read_xml(XML, section_tag="section", id_attr="id")
    assert [s.number for s in document.sections] == ["s-4", "s-5"]
    assert document.sections[0].title == "Definitions"
    assert "glimmer node means" in document.sections[0].text
    assert document.sections[1].title == "Application"


def test_a_caller_supplied_section_rule_drives_read():
    default = html.read(PAGE)
    assert [s.title for s in default.sections] == ["2.1 Glimmer Nodes", "2.2 Vault Currents"]
    assert default.sections[0].number == ""

    numbered = html.read(PAGE, sections=NUMBERED)
    assert [s.number for s in numbered.sections] == ["2.1", "2.2"]
    assert [s.title for s in numbered.sections] == ["Glimmer Nodes", "Vault Currents"]


def test_units_respect_the_token_cap_and_split_on_paragraphs():
    document = html.read(PAGE, sections=NUMBERED)
    whole = units.units(document)
    assert len(whole) == 2

    split = units.units(document, max_tokens=1)
    first = [u for u in split if u.section == "2.1"]
    assert len(first) == 2
    assert first[0].text == "A glimmer node is the smallest part of a lattice that can hold a charge."
    assert first[1].text == "Every node sits inside a vault."
    assert first[0].parts == 2 and first[1].part == 2


def test_unit_where_carries_the_url():
    document = html.read(PAGE, url="https://example.test/handbook", sections=NUMBERED)
    unit = units.units(document)[0]
    assert unit.where["url"] == "https://example.test/handbook"
    assert unit.where["unit"] == unit.id

    pdf_unit = pdf.Unit(source="s", book_title="b", chapter="", chapter_title="",
                        section="1", section_title="t", first_page=1, last_page=1, text="x")
    assert "url" not in pdf_unit.where


def test_reader_for_picks_the_reader_by_suffix(tmp_path, monkeypatch):
    from ml_stack.sources import html as html_module
    from ml_stack.sources import pdf as pdf_module

    seen = []
    monkeypatch.setattr(html_module, "read", lambda where, **kw: seen.append(("html", where)))
    monkeypatch.setattr(html_module, "read_xml",
                        lambda where, **kw: seen.append(("xml", where)))
    monkeypatch.setattr(pdf_module, "read", lambda where, **kw: seen.append(("pdf", where)))

    htm = tmp_path / "a.htm"
    htm.write_text(PAGE)
    xml = tmp_path / "a.xml"
    xml.write_text(XML)
    doc = tmp_path / "a.pdf"
    doc.write_text("not a real pdf, never opened")

    ingest.reader_for(str(htm))()
    ingest.reader_for(str(xml))()
    ingest.reader_for(str(doc))()

    assert seen == [("html", str(htm)), ("xml", str(xml)), ("pdf", str(doc))]


def test_reader_for_fetches_a_url_and_dispatches_on_what_came_down(tmp_path, monkeypatch):
    import ml_stack.http as http
    import ml_stack.media.download as download

    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(http, "check", lambda url: url)

    fetched = []

    def stand_in_fetch(url: str, target: Path, **_: object) -> Path:
        fetched.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(PAGE)
        return target

    monkeypatch.setattr(download, "fetch", stand_in_fetch)

    document = ingest.reader_for("https://example.test/codes.html", chapter=None)()
    assert fetched == ["https://example.test/codes.html"]
    assert document.url == "https://example.test/codes.html"
    assert [s.title for s in document.sections] == ["2.1 Glimmer Nodes", "2.2 Vault Currents"]


def test_reader_for_refuses_a_url_this_machine_should_not_fetch():
    with pytest.raises(Refused):
        ingest.reader_for("http://127.0.0.1/secret")()
