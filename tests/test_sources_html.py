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
    document = html.read_xml(XML)
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


NAMESPACED = """<?xml version="1.0" encoding="UTF-8"?>
<Statute xmlns="urn:example:law">
  <Title>An Invented Act</Title>
  <Section Id="4.1">
    <Title>What a licensee shall do</Title>
    <Text>A licensee shall keep the records this section names.</Text>
  </Section>
  <Section Id="4.2">
    <Title>What a licensee shall report</Title>
    <Text>A licensee shall report a change to the regulator before making it.</Text>
  </Section>
</Statute>
"""


def test_read_xml_matches_a_tag_whatever_its_case_or_namespace():
    document = html.read_xml(NAMESPACED, marks=html.Marks(id_attr="ID"))
    assert [s.number for s in document.sections] == ["4.1", "4.2"]
    assert document.sections[0].title == "What a licensee shall do"


def test_reader_for_takes_the_spelling_a_source_uses(tmp_path):
    xml = tmp_path / "act.xml"
    xml.write_text(NAMESPACED)
    plain = ingest.reader_for(str(xml))()
    assert [s.number for s in plain.sections] == ["4.1", "4.2"]

    named = ingest.reader_for(str(xml), marks=html.Marks(section_tag="Title"))()
    assert len(named.sections) == 3


def test_marks_read_the_spelling_off_a_command_line():
    import argparse

    args = argparse.Namespace(section_tag="Section", id_attr="", number_tag="Label",
                              title_tag="MarginalNote",
                              section_pattern=r"^(\d+\.\d+)\s+(.*)$")
    marks = html.Marks.from_args(args)
    assert (marks.section_tag, marks.number_tag, marks.title_tag) \
        == ("Section", "Label", "MarginalNote")
    assert marks.id_attr == "id"
    assert marks.rule() is not None


def test_marks_leave_each_reader_its_own_default():
    import argparse

    marks = html.Marks.from_args(argparse.Namespace())
    assert (marks.section_tag, marks.id_attr) == ("section", "id")
    assert marks.rule() is None


def test_rule_for_reads_a_number_and_a_title_out_of_a_heading():
    rule = html.rule_for(r"^REGDOC\s+(\S+)\s+(.*)$")
    assert rule.parse(rule.pattern.match("REGDOC 7.3 Fitness for service")) \
        == ("7.3", "Fitness for service")
    assert html.rule_for("") is None


def test_an_html_section_rule_reaches_the_reader(tmp_path):
    page = tmp_path / "reg.html"
    page.write_text("<html><body><h2>REGDOC 7.3 Fitness for service</h2><p>Words.</p>"
                    "<h2>Not a clause</h2><p>More words.</p></body></html>")
    marks = html.Marks(pattern=r"^REGDOC\s+(\S+)\s+(.*)$")
    document = ingest.reader_for(str(page), marks=marks)()
    assert [(s.number, s.title) for s in document.sections] == [("7.3", "Fitness for service")]


STATUTE = """<?xml version="1.0" encoding="UTF-8"?>
<Statute>
  <Title>An Invented Act</Title>
  <Section lims:id="99001" xmlns:lims="urn:example:lims">
    <MarginalNote>Short title</MarginalNote>
    <Label>1</Label>
    <Text>This Act may be cited as the Invented Act.</Text>
  </Section>
  <Section lims:id="99002" xmlns:lims="urn:example:lims">
    <MarginalNote>Records to be kept</MarginalNote>
    <Label>2</Label>
    <Text>A licensee shall keep the records this section names.</Text>
  </Section>
</Statute>
"""


def test_read_xml_numbers_a_section_from_a_child_element():
    document = html.read_xml(STATUTE, marks=html.Marks(
        section_tag="Section", number_tag="Label", title_tag="MarginalNote"))
    assert [(s.number, s.title) for s in document.sections] == [
        ("1", "Short title"), ("2", "Records to be kept")]


def test_the_number_and_the_title_are_not_repeated_in_the_section_text():
    document = html.read_xml(STATUTE, marks=html.Marks(
        section_tag="Section", number_tag="Label", title_tag="MarginalNote"))
    text = document.sections[1].text
    assert text == "A licensee shall keep the records this section names."


def test_a_number_child_wins_over_the_id_attribute():
    document = html.read_xml(STATUTE, marks=html.Marks(
        section_tag="Section", number_tag="Label"))
    assert [s.number for s in document.sections] == ["1", "2"]

    without = html.read_xml(STATUTE, marks=html.Marks(section_tag="Section"))
    assert [s.number for s in without.sections] == ["99001", "99002"]


def test_a_source_is_re_read_with_the_marks_it_was_read_with(tmp_path):
    """`sources_for` re-reads a document; other marks would mint other unit ids."""
    from dataclasses import asdict

    from ml_stack.ingest.judge import sources_for
    from ml_stack.ingest.progress import Progress
    from ml_stack.sources import units as source_units

    statute = tmp_path / "act.xml"
    statute.write_text(STATUTE)
    marks = html.Marks(section_tag="Section", number_tag="Label", title_tag="MarginalNote")
    document = ingest.reader_for(str(statute), marks=marks)()
    wanted = source_units.units(document)

    out = tmp_path / "store.ladybug"
    record = Progress(Progress.beside(out))
    record.source(document.slug, title=document.title, path=str(statute),
                  sections=len(wanted), marks=asdict(marks))
    record.save()

    text_of = sources_for(out)
    assert text_of(wanted[0].id) == wanted[0].text


def test_a_source_recorded_without_marks_is_still_re_read(tmp_path):
    from ml_stack.ingest.judge import sources_for
    from ml_stack.ingest.progress import Progress
    from ml_stack.sources import units as source_units

    page = tmp_path / "plain.xml"
    page.write_text(XML)
    document = ingest.reader_for(str(page))()
    wanted = source_units.units(document)

    out = tmp_path / "store.ladybug"
    record = Progress(Progress.beside(out))
    record.source(document.slug, title=document.title, path=str(page), sections=len(wanted))
    record.save()

    assert sources_for(out)(wanted[0].id) == wanted[0].text
