"""A textbook read out of a PDF: chapters, sections, figures, and the furniture dropped.

Every PDF here is written by the test, two to four pages of an invented subject, so nothing
depends on a book being on this machine and no real text is copied anywhere. Two are built:
one carrying an outline the way a publisher's PDF does, and one with none at all -- a book
printed to PDF by a browser -- which is the case the heading heuristics exist for.
"""

from __future__ import annotations

import pytest

pymupdf = pytest.importorskip("pymupdf")

from ml_stack.sources import pdf  # noqa: E402

BODY = 9.0
SECTION = 13.0
CHAPTER = 15.0
TITLE = 22.0
FOOT = 7.5

# An invented discipline. Nothing here is a real subject, a real book or a real person.
LICENCE = ("Published by Velthorne Open Texts. This work is released under a "
           "Creative Commons Attribution licence. openstax")
INTRO = ("Glimmer nodes are the smallest part of a lattice that can hold a charge. Every\n"
         "node sits inside a vault and each vault holds many nodes.")
HYPHENATED = ("A node that has been charged is said to be quick-\n"
              "ened, and a quickened node passes its charge to the vault around it.")
CAPTION = ("FIGURE 1.1 A glimmer node in cross-section, with the vault wall drawn\n"
           "around it.")
SECOND = ("Vault currents run between quickened nodes. A current consumes charge and\n"
          "produces heat, which the lattice sheds through its outer wall.")


def _plate(colour: tuple[float, float, float] = (0.2, 0.4, 0.8)) -> bytes:
    """A small solid PNG standing in for a textbook plate."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 40))
    pix.set_rect(pix.irect, tuple(int(c * 255) for c in colour))
    return pix.tobytes("png")


def _page(doc, *, footer: str = "") -> object:
    page = doc.new_page(width=430, height=620)
    if footer:
        page.insert_text((40, 600), footer, fontsize=FOOT, fontname="hebo")
        page.insert_text((380, 600), str(doc.page_count), fontsize=FOOT, fontname="hebo")
    return page


def a_textbook(path, *, outline: bool = True, licence: bool = True) -> str:
    """Two chapters, four pages, one figure with a caption, and a footer on every page."""
    doc = pymupdf.open()

    front = _page(doc)
    front.insert_text((40, 80), "Lattice Studies", fontsize=TITLE)
    if licence:
        front.insert_textbox(pymupdf.Rect(40, 120, 390, 260), LICENCE, fontsize=BODY)

    one = _page(doc, footer="1 • The Glimmer Cascade")
    one.insert_text((40, 80), "CHAPTER 1", fontsize=CHAPTER)
    one.insert_text((40, 120), "The Glimmer Cascade", fontsize=TITLE)
    one.insert_text((40, 180), INTRO, fontsize=BODY)

    two = _page(doc, footer="1 • The Glimmer Cascade")
    two.insert_text((40, 80), "1.1 Glimmer Nodes", fontsize=SECTION, fontname="hebo")
    two.insert_text((40, 120), HYPHENATED, fontsize=BODY)
    two.insert_text((40, 180), "glimmer node", fontsize=BODY, fontname="hebo")
    two.insert_image(pymupdf.Rect(40, 220, 220, 340), stream=_plate())
    two.insert_text((40, 380), CAPTION, fontsize=FOOT, fontname="hebo")

    three = _page(doc, footer="2 • Vault Currents")
    three.insert_text((40, 80), "CHAPTER 2", fontsize=CHAPTER)
    three.insert_text((40, 120), "Vault Currents", fontsize=TITLE)
    three.insert_text((40, 180), "2.1 Currents in Practice", fontsize=SECTION, fontname="hebo")
    three.insert_text((40, 230), SECOND, fontsize=BODY)

    if outline:
        doc.set_toc([[1, "Chapter 1 The Glimmer Cascade", 2],
                     [2, "1.1 Glimmer Nodes", 3],
                     [1, "Chapter 2 Vault Currents", 4],
                     [2, "2.1 Currents in Practice", 4]])
    where = str(path)
    doc.save(where)
    doc.close()
    return where


# -- the outline a publisher ships ------------------------------------------------------------


def test_an_outline_gives_the_chapters_and_their_sections(tmp_path):
    document = pdf.read(a_textbook(tmp_path / "lattice.pdf"))

    assert document.how == "toc"
    assert [(c.number, c.title) for c in document.chapters] == [
        ("1", "The Glimmer Cascade"), ("2", "Vault Currents")]
    assert [(s.chapter, s.number, s.title) for s in document.sections] == [
        ("1", "1.1", "Glimmer Nodes"), ("2", "2.1", "Currents in Practice")]


def test_without_an_outline_the_headings_are_read_off_the_way_they_are_set(tmp_path):
    """The case that matters: a browser prints a textbook to PDF and drops the outline."""
    document = pdf.read(a_textbook(tmp_path / "plain.pdf", outline=False))

    assert document.how == "headings"
    assert [(c.number, c.title) for c in document.chapters] == [
        ("1", "The Glimmer Cascade"), ("2", "Vault Currents")]
    assert [(s.chapter, s.number) for s in document.sections] == [("1", "1.1"), ("2", "2.1")]


def test_a_section_knows_which_pages_it_covers(tmp_path):
    document = pdf.read(a_textbook(tmp_path / "lattice.pdf"))
    first, second = document.sections
    assert first.pages == (2, 3), "a chapter's first section starts where the chapter does"
    assert second.pages == (4, 4)


def test_reading_one_chapter_leaves_the_rest_of_the_book_unread(tmp_path):
    document = pdf.read(a_textbook(tmp_path / "lattice.pdf"), chapter=2)
    assert [c.number for c in document.chapters] == ["2"]
    assert [s.number for s in document.sections] == ["2.1"]


def test_the_licence_page_says_it_is_an_openstax_book_and_a_book_without_one_does_not(tmp_path):
    assert pdf.read(a_textbook(tmp_path / "with.pdf")).openstax
    assert not pdf.read(a_textbook(tmp_path / "without.pdf", licence=False)).openstax


# -- cleaning ------------------------------------------------------------------------------------


def test_a_word_broken_across_a_line_is_put_back_together(tmp_path):
    """"quick-\\nened" is one word. Left alone it becomes two, and neither is in the book."""
    section = pdf.read(a_textbook(tmp_path / "lattice.pdf")).sections[0]
    assert "quickened" in section.text
    assert "quick- ened" not in section.text and "quick-\n" not in section.text


def test_the_running_foot_and_the_page_number_are_dropped(tmp_path):
    document = pdf.read(a_textbook(tmp_path / "lattice.pdf"))
    everything = "\n".join(s.text for s in document.sections)
    assert "Vault Currents" in everything, "the chapter's own words are kept"
    assert "1 • The Glimmer Cascade" not in everything
    assert "2 • Vault Currents" not in everything


def test_a_figure_caption_is_kept_and_labelled_rather_than_dropped(tmp_path):
    """The sentence under a picture is often the only place a book says what it shows."""
    section = pdf.read(a_textbook(tmp_path / "lattice.pdf")).sections[0]
    assert "[Figure 1.1]" in section.text
    assert "A glimmer node in cross-section" in section.text


def test_a_bolded_key_term_is_kept_and_a_bold_sentence_is_not(tmp_path):
    section = pdf.read(a_textbook(tmp_path / "lattice.pdf")).sections[0]
    assert "glimmer node" in section.key_terms
    assert not any(len(term.split()) > 6 for term in section.key_terms)


# -- figures -------------------------------------------------------------------------------------


def test_a_figure_carries_the_caption_nearest_below_it(tmp_path):
    section = pdf.read(a_textbook(tmp_path / "lattice.pdf")).sections[0]
    assert len(section.figures) == 1
    figure = section.figures[0]
    assert figure.page == 3 and figure.label == "Figure 1.1"
    assert "glimmer node in cross-section" in figure.caption


def test_a_figure_is_rendered_only_when_the_images_are_asked_for(tmp_path):
    where = a_textbook(tmp_path / "lattice.pdf")
    assert not pdf.read(where).sections[0].figures[0].shown

    figure = pdf.read(where, images=True).sections[0].figures[0]
    assert figure.shown and figure.png.startswith(b"\x89PNG")
    assert 0 < figure.width <= pdf.IMAGE_WIDTH


def test_a_figure_is_never_wider_than_it_was_asked_for(tmp_path):
    figure = pdf.read(a_textbook(tmp_path / "lattice.pdf"), images=True,
                      max_width=16).sections[0].figures[0]
    assert figure.width <= 16 and figure.height > 0


# -- units ---------------------------------------------------------------------------------------


def test_a_short_section_is_one_unit_and_says_where_it_came_from(tmp_path):
    document = pdf.read(a_textbook(tmp_path / "lattice.pdf"))
    units = pdf.units(document)

    assert [u.parts for u in units] == [1, 1]
    first = units[0]
    assert first.id == "lattice:1:1.1"
    assert first.where == {"book": "lattice", "chapter": "1", "section": "1.1",
                           "page": 2, "pages": [2, 3], "unit": "lattice:1:1.1"}


def test_a_long_section_is_split_on_paragraph_boundaries_and_never_inside_one(tmp_path):
    document = pdf.read(a_textbook(tmp_path / "lattice.pdf"))
    paragraphs = [p for p in document.sections[0].text.split("\n\n") if p.strip()]
    assert len(paragraphs) > 1, "the fixture needs more than one paragraph to be split"

    units = pdf.units(document, max_tokens=1)
    parts = [u for u in units if u.section == "1.1"]
    assert len(parts) == len(paragraphs)
    assert [u.part for u in parts] == list(range(1, len(parts) + 1))
    assert all(u.parts == len(parts) for u in parts)
    for unit, paragraph in zip(parts, paragraphs):
        assert unit.text == paragraph
    assert len({u.id for u in units}) == len(units)


def test_the_figures_and_terms_ride_on_the_first_part_only(tmp_path):
    """Sending every plate with every part of a section is the same picture four times."""
    document = pdf.read(a_textbook(tmp_path / "lattice.pdf"), images=True)
    parts = [u for u in pdf.units(document, max_tokens=1) if u.section == "1.1"]
    assert parts[0].figures and parts[0].key_terms
    assert all(not u.figures and not u.key_terms for u in parts[1:])


@pytest.mark.parametrize(("stamped", "expected"), [
    ("Lattice Studies 2e", "Lattice Studies 2e"),
    ("https:/example.invalid/media/Lattice_Studies-WEB", "lattice"),
    ("", "lattice"),
])
def test_the_title_is_the_book_s_own_unless_the_metadata_holds_a_url(tmp_path, stamped,
                                                                    expected):
    """A book printed to PDF by a browser carries the address it was printed from."""
    where = a_textbook(tmp_path / "lattice.pdf")
    doc = pymupdf.open(where)
    doc.set_metadata({"title": stamped})
    doc.saveIncr()
    doc.close()
    assert pdf.read(where).title == expected


def test_reading_the_same_book_twice_gives_the_same_units(tmp_path):
    where = a_textbook(tmp_path / "lattice.pdf")
    first = [(u.id, u.text) for u in pdf.units(pdf.read(where))]
    second = [(u.id, u.text) for u in pdf.units(pdf.read(where))]
    assert first == second
