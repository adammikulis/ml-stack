"""Pin tables and package drawings out of a datasheet PDF.

The PDF is written by the test: an invented regulator, "ACME1234", with a pin table whose
header spans two rows and continues onto a second page, a package outline and a land pattern
drawn as vector strokes, and a page that only talks about a land pattern.
"""

from __future__ import annotations

import pytest

from ml_stack.sources import datasheet

COLS = (50, 90, 50, 250)
ROW_H = 22


@pytest.fixture
def pymupdf():
    return pytest.importorskip("pymupdf", reason="ml-stack[pdf]")


def grid(page, top, rows):
    """Ruled table: every cell boxed, text at the cell's corner."""
    x = [40.0]
    for w in COLS:
        x.append(x[-1] + w)
    bottom = top + ROW_H * len(rows)
    for i in range(len(rows) + 1):
        page.draw_line((x[0], top + i * ROW_H), (x[-1], top + i * ROW_H))
    for xv in x:
        page.draw_line((xv, top), (xv, bottom))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell:
                page.insert_text((x[c] + 4, top + r * ROW_H + 14), cell, fontsize=8)


def strokes(page, left, top, n=60):
    """A plausible package drawing: a body square, pads, and dimension ticks."""
    page.draw_rect((left, top, left + 120, top + 120))
    for i in range(n // 2):
        page.draw_line((left - 10, top + 4 * i), (left - 2, top + 4 * i))
        page.draw_line((left + 122, top + 4 * i), (left + 130, top + 4 * i))
    page.insert_text((left, top + 150), "3.00 mm x 3.00 mm", fontsize=8)


@pytest.fixture
def sheet(pymupdf, tmp_path):
    doc = pymupdf.open()
    one = doc.new_page(width=500, height=700)
    one.insert_text((40, 60), "ACME1234 3-A Step-Down Converter", fontsize=16)
    one.insert_text((40, 90), "Part numbers ACME1234 and ACME1234A.", fontsize=9)
    two = doc.new_page(width=500, height=700)
    two.insert_text((40, 50), "Pin Functions", fontsize=14)
    grid(two, 70, [
        ["PIN", "", "I/O", "DESCRIPTION"],
        ["NO.", "NAME", "", ""],
        ["1,2,3", "SW", "O", "Switch node"],
        ["4", "PG", "O", "Power good output"],
        ["5", "FB", "I", "Feedback input"],
        ["", "", "", "continued text"],
        ["6", "AGND", "", "Analog ground"],
    ])
    three = doc.new_page(width=500, height=700)
    grid(three, 70, [
        ["7", "VIN", "I", "Supply input"],
        ["8", "EN", "I", "Enable"],
    ])
    four = doc.new_page(width=500, height=700)
    four.insert_text((40, 50), "Package Outline", fontsize=14)
    strokes(four, 200, 200)
    five = doc.new_page(width=500, height=700)
    five.insert_text((40, 50), "Recommended Land Pattern", fontsize=14)
    strokes(five, 200, 200)
    six = doc.new_page(width=500, height=700)
    six.insert_text((40, 50), "See the land pattern and package outline in section 4.", fontsize=9)
    path = tmp_path / "acme1234.pdf"
    doc.save(str(path))
    return path


def test_the_two_row_header_pin_table_becomes_rows(sheet):
    pins = [p for p in datasheet.pin_tables(sheet) if p.page == 2]
    assert [(p.number, p.name, p.type) for p in pins] == [
        ("1,2,3", "SW", "O"), ("4", "PG", "O"), ("5", "FB", "I"), ("6", "AGND", "")]
    assert pins[1].description == "Power good output"


def test_a_wrapped_description_joins_the_row_above(sheet):
    pins = {p.name: p for p in datasheet.pin_tables(sheet)}
    assert pins["FB"].description == "Feedback input continued text"


def test_a_table_continued_on_the_next_page_without_a_header_is_followed(sheet):
    pins = datasheet.pin_tables(sheet)
    assert [p.name for p in pins] == ["SW", "PG", "FB", "AGND", "VIN", "EN"]
    assert [p.page for p in pins][-2:] == [3, 3]


def test_a_table_that_is_not_a_pin_table_gives_nothing(pymupdf, tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=700)
    page.insert_text((40, 50), "Pin Functions are described elsewhere", fontsize=10)
    grid(page, 70, [["PART NUMBER", "OUTPUT", "", ""], ["X1", "1.8 V", "", ""]])
    path = tmp_path / "other.pdf"
    doc.save(str(path))
    assert datasheet.pin_tables(path) == []


def test_extra_columns_are_kept_by_header(pymupdf, tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=700)
    page.insert_text((40, 50), "Pin description", fontsize=10)
    grid(page, 70, [["Pin number", "Name", "Type", "Description"], ["A1", "PA0", "I/O", "GPIO"]])
    path = tmp_path / "gpio.pdf"
    doc.save(str(path))
    [pin] = datasheet.pin_tables(path)
    assert (pin.number, pin.name, pin.type, pin.description) == ("A1", "PA0", "I/O", "GPIO")


def test_drawn_package_pages_are_found_and_ranked(sheet):
    found = datasheet.outline_pages(sheet)
    assert {(o.page, o.kind) for o in found} == {(4, "package_outline"), (5, "land_pattern")}
    assert found[0].score >= found[-1].score
    assert {o.page: o.heading for o in found} == {4: "Package Outline",
                                                  5: "Recommended Land Pattern"}


def test_a_page_that_only_mentions_a_land_pattern_is_not_one(sheet):
    assert 6 not in {o.page for o in datasheet.outline_pages(sheet)}


def test_a_crop_is_a_png_smaller_than_the_page(sheet):
    full = datasheet.render(sheet, 4, crop=False)
    crop = datasheet.render(sheet, 4)
    assert crop[:8] == b"\x89PNG\r\n\x1a\n" == full[:8]

    def size(png):
        return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    assert size(crop)[0] < size(full)[0] and size(crop)[1] < size(full)[1]
    assert max(size(full)) <= datasheet.MAX_PX


def test_text_and_mentions(sheet):
    _, body, pages = datasheet.text(sheet, limit=200)
    assert pages == 6 and "ACME1234" in body and len(body) <= 200
    assert datasheet.mentions(sheet, "acme-1234a")
    assert not datasheet.mentions(sheet, "ZZZ999")


def test_a_part_number_matches_by_how_much_of_it_the_text_names(sheet):
    assert datasheet.part_match(sheet, "ACME1234A") == (1.0, True)
    share, first = datasheet.part_match(sheet, "ACME1234XYZ")
    assert 0.6 < share < 1.0 and first
    assert datasheet.part_match(sheet, "ZZZ999") == (0.0, False)
    assert datasheet.part_match(sheet, "ACME")[0] == 0.0
