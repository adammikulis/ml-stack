"""Page rendering and outline-page detection without MuPDF (AGPL-3.0).

Pages are rendered by PDFium (pypdfium2: Apache-2.0 / BSD-3-Clause) in a bounded child
(`poolhouse.net.pdfrender`). Every PDF is written by the test from bytes (`poolhouse.redteam.minipdf`
and raw objects); hostile ones are built the same way, and each bound has a test that fails when
the bound is removed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import zlib
from io import BytesIO
from pathlib import Path

import pytest
from test_pdf_engine import one_page, raw_pdf

from poolhouse.net import pdfrender
from poolhouse.net.pdfread import PdfRefused
from poolhouse.net.pdfrender import RenderLimits, render_pages
from poolhouse.redteam.minipdf import Doc
from poolhouse.sources import datasheet

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def default_engine(monkeypatch):
    monkeypatch.delenv("POOLHOUSE_PDF_ENGINE", raising=False)


@pytest.fixture
def pdfium():
    return pytest.importorskip("pypdfium2", reason="pip install 'poolhouse[pdf-render]'")


def png_size(png: bytes) -> tuple[int, int]:
    assert png[:8] == PNG
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def dark_pixels(png: bytes) -> int:
    from PIL import Image
    gray = Image.open(BytesIO(png)).convert("L")
    return sum(1 for v in gray.tobytes() if v < 100)


def a_drawn_page(path) -> str:
    doc = Doc()
    page = doc.page(430, 620)
    page.text(100, 200, "PACKAGE OUTLINE", size=24, bold=True)
    page.ops.append(b"2 w 100 120 200 100 re S\n")
    doc.page(430, 620)                                           # a page with nothing on it
    Path(path).write_bytes(doc.to_bytes())
    return str(path)


def render(tmp_path, data: bytes, pages=(1,), **limits):
    path = tmp_path / "hostile.pdf"
    path.write_bytes(data)
    return render_pages(path, list(pages), limits=RenderLimits(**limits))


# -- rendering --------------------------------------------------------------------------------


def test_a_page_renders_at_its_size_with_ink_on_it_and_a_blank_page_is_flagged(pdfium, tmp_path):
    drawn, empty = render_pages(a_drawn_page(tmp_path / "drawn.pdf"), [1, 2])
    assert (drawn.width, drawn.height) == (896, 1292) == png_size(drawn.png)  # 430 x 620 pt at 150 dpi
    assert dark_pixels(drawn.png) > 2000 and not drawn.blank
    assert empty.blank and dark_pixels(empty.png) == 0


def test_a_crop_is_the_box_around_what_is_drawn_and_a_blank_page_is_not_cropped_to_nothing(pdfium, tmp_path):
    where = a_drawn_page(tmp_path / "drawn.pdf")
    (full,), (crop,), (blank,) = (render_pages(where, [1]), render_pages(where, [1], crop=True),
                                  render_pages(where, [2], crop=True))
    assert crop.width < full.width and crop.height < full.height and not crop.blank
    assert crop.width >= 200 * 150 / 72, "the rectangle (200 pt wide) is inside the crop"
    assert blank.blank and blank.width > 0 and blank.height > 0


def test_the_width_and_the_pixel_bound_lower_the_resolution_instead_of_failing(pdfium, tmp_path):
    where = a_drawn_page(tmp_path / "drawn.pdf")
    (narrow,) = render_pages(where, [1], max_width=300)
    assert narrow.width <= 300 and narrow.height > narrow.width
    (cropped,) = render_pages(where, [1], crop=True, max_width=300)
    assert cropped.width == 300 and not cropped.blank, "a crop wider than max_width is scaled down"
    (small,) = render_pages(where, [1], limits=RenderLimits(max_pixels=20_000))
    assert small.width * small.height <= 20_000 * 1.05


def test_the_default_engine_renders_a_datasheet_page_with_pdfium_and_crops_it(pdfium, tmp_path):
    where = a_drawn_page(tmp_path / "drawn.pdf")
    code = ("import sys\nfrom poolhouse.sources import datasheet\n"
            f"full, crop = datasheet.render({where!r}, 1, crop=False), datasheet.render({where!r}, 1)\n"
            "assert full[:8] == crop[:8] == b'\\x89PNG\\r\\n\\x1a\\n'\n"
            "w = lambda p: int.from_bytes(p[16:20], 'big')\n"
            "assert w(crop) < w(full) <= datasheet.MAX_PX, (w(crop), w(full))\n"
            "assert 'pymupdf' not in sys.modules and 'fitz' not in sys.modules\nprint('ok')")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                          env=env)
    assert done.stdout.strip() == "ok", done.stderr[-600:]


def test_rendering_names_the_extra_when_pdfium_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(pdfrender, "available", lambda: False)
    with pytest.raises(RuntimeError, match="pdf-render"):
        render_pages(a_drawn_page(tmp_path / "d.pdf"), [1])


def test_the_render_child_runs_by_path_with_dash_P_in_a_scrubbed_environment(
        pdfium, tmp_path, monkeypatch):
    """The real child runs; the call is only watched, to see how it was started."""
    seen = {}
    real = subprocess.run

    def watch(argv, **kw):
        seen.update(argv=argv, env=kw["env"], cwd=kw["cwd"])
        return real(argv, **kw)

    monkeypatch.setenv("POOLHOUSE_TEST_SECRET", "hunter2")
    monkeypatch.setattr(subprocess, "run", watch)
    (page,) = render_pages(a_drawn_page(tmp_path / "d.pdf"), [1])
    assert page.width > 0
    assert seen["argv"][1:] == ["-P", str(pdfrender.CHILD)] and "-m" not in seen["argv"]
    assert "POOLHOUSE_TEST_SECRET" not in seen["env"] and set(seen["env"]) <= {
        "PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "PYTHONDONTWRITEBYTECODE"}
    assert seen["cwd"] != str(tmp_path)


# -- hostile files: bounded, and refused with a reason ------------------------------------------


def one_page_with_image(dim: int, media: bytes = b"300 300") -> bytes:
    data = zlib.compress(b"\x00" * min(dim * dim // 8, 1 << 26), 9)
    image = (b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray "
             b"/BitsPerComponent 1 /Filter /FlateDecode /Length %d >>\nstream\n" % (dim, dim, len(data))
             + data + b"\nendstream")
    draw = b"q 300 0 0 300 0 0 cm /Im1 Do Q"
    return raw_pdf({
        1: b"<< /Type /Catalog /Pages 2 0 R >>", 2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " + media + b"] /Contents 4 0 R "
           b"/Resources << /XObject << /Im1 5 0 R >> >> >>",
        4: b"<< /Length %d >>\nstream\n" % len(draw) + draw + b"\nendstream", 5: image})


def test_a_page_larger_than_any_paper_is_refused_before_a_pixel_is_drawn(pdfium, tmp_path):
    for side in (b"14400", b"1000000000"):
        with pytest.raises(PdfRefused, match="points, over 5000"):
            render(tmp_path, one_page_with_image(8, side + b" " + side))
    with pytest.raises(PdfRefused, match="points, over 100"):
        render(tmp_path, one_page_with_image(8), max_page_pt=100.0)


def test_a_picture_that_declares_more_pixels_than_the_bound_is_refused_unread(pdfium, tmp_path):
    started = time.monotonic()
    with pytest.raises(PdfRefused, match="declares more than 150000000 pixels"):
        render(tmp_path, one_page_with_image(100_000))      # 10^10 pixels in a 300 KB file
    assert time.monotonic() - started < 20
    with pytest.raises(PdfRefused, match="declares more than 1000 pixels"):
        render(tmp_path, one_page_with_image(64), max_image_pixels=1000)


def test_memory_past_the_bound_ends_the_render_even_where_the_os_would_not(pdfium, tmp_path):
    with pytest.raises(PdfRefused, match="more memory than the limit"):
        render(tmp_path, one_page_with_image(64), memory_bytes=1 << 20)


def test_a_picture_that_takes_too_long_to_draw_is_killed(pdfium, tmp_path):
    big = one_page_with_image(100_000)
    started = time.monotonic()
    with pytest.raises(PdfRefused, match="took more than 2 s"):
        render(tmp_path, big, timeout_s=2.0, max_image_pixels=10 ** 12)
    assert time.monotonic() - started < 30


def test_a_file_with_more_pages_than_the_bound_or_a_billion_claimed_is_refused(pdfium, tmp_path):
    doc = Doc()
    for n in range(12):
        doc.page().text(40, 80, f"page {n}")
    with pytest.raises(PdfRefused, match="12 pages, over 5"):
        render(tmp_path, doc.to_bytes(), max_pages=5)
    claim = raw_pdf({1: b"<< /Type /Catalog /Pages 2 0 R >>",
                     2: b"<< /Type /Pages /Kids [] /Count 1000000000 >>"})
    started = time.monotonic()
    with pytest.raises(PdfRefused):
        render(tmp_path, claim)
    assert time.monotonic() - started < 20


def test_more_pages_asked_for_than_the_bound_or_not_in_the_file_are_refused(pdfium, tmp_path):
    data = a_drawn_page(tmp_path / "d.pdf")
    with pytest.raises(PdfRefused, match="3 pages asked for, the bound is 2"):
        render_pages(data, [1, 2, 1], limits=RenderLimits(max_render_pages=2))
    with pytest.raises(PdfRefused, match="page 3 is not in a file of 2 pages"):
        render_pages(data, [3])
    with pytest.raises(PdfRefused, match="page 0 is not"):
        render_pages(data, [0])


def test_png_bytes_past_the_bound_are_refused(pdfium, tmp_path):
    with pytest.raises(PdfRefused, match="pictures are over 100 bytes"):
        render_pages(a_drawn_page(tmp_path / "d.pdf"), [1], limits=RenderLimits(max_out=100))


def test_a_file_over_the_byte_bound_is_refused_before_it_is_opened_by_the_renderer(pdfium, tmp_path):
    with pytest.raises(PdfRefused, match="over 100"):
        render(tmp_path, one_page(b"BT ET"), max_bytes=100)


def test_files_that_are_not_pdfs_or_are_cut_short_or_need_a_password_are_refused(pdfium, tmp_path):
    with pytest.raises(PdfRefused, match="not a readable PDF"):
        render(tmp_path, b"MZ\x90\x00 this is not a pdf at all" * 20)
    data = Path(a_drawn_page(tmp_path / "d.pdf")).read_bytes()
    try:
        got = render(tmp_path, data[: len(data) // 3])
    except PdfRefused as refusal:
        assert str(refusal)
    else:
        assert got[0].width > 0
    locked = raw_pdf({
        1: b"<< /Type /Catalog /Pages 2 0 R >>", 2: b"<< /Type /Pages /Kids [] /Count 0 >>",
        3: b"<< /Filter /Standard /V 1 /R 2 /P -4 /O (" + b"o" * 32 + b") /U (" + b"u" * 32 + b") >>"},
        extra_trailer=b"/Encrypt 3 0 R /ID [<00112233445566778899aabbccddeeff> "
                      b"<00112233445566778899aabbccddeeff>] ")
    with pytest.raises(PdfRefused, match=r"encrypted|not a readable"):
        render(tmp_path, locked)


# -- the pages that draw a package, found without MuPDF ----------------------------------------


def a_datasheet(path) -> str:
    doc = Doc()
    doc.page().text(40, 60, "ACME1234 3-A Step-Down Converter", size=16)
    outline = doc.page()
    outline.text(40, 50, "Package Outline", size=14)
    outline.text(40, 80, "Dimensions in mm", size=9)
    outline.ops.append(b"1 0 0 RG 100 300 120 120 re S\n" + b"".join(
        b"%d 300 m %d 310 l S\n" % (60 + 4 * i, 60 + 4 * i) for i in range(30)) + b"".join(
        b"90 %d m 98 %d l S\n" % (300 + 4 * i, 300 + 4 * i) for i in range(30)))
    talk = doc.page()
    talk.text(40, 50, "Recommended Land Pattern", size=14)
    talk.text(40, 80, "See the application note for land pattern advice.", size=9)
    plain = doc.page()
    plain.text(40, 50, "Package Outline", size=14)               # the words, but no drawing
    Path(path).write_bytes(doc.to_bytes())
    return str(path)


def test_outline_pages_are_found_with_the_default_engine_from_text_and_strokes(tmp_path):
    found = datasheet.outline_pages(a_datasheet(tmp_path / "ds.pdf"))
    assert [(o.page, o.kind, o.heading) for o in found] == [(2, "package_outline", "Package Outline")]
