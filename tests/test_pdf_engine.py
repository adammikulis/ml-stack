"""The PDF reader is permissively licensed by default, bounded, and fails closed.

Every PDF here is written by the test from plain bytes (`ml_stack.redteam.minipdf`), so none of
these tests needs MuPDF (AGPL-3.0). Hostile inputs are built the same way: a stream that inflates
to hundreds of megabytes, a page tree that claims a billion pages or contains itself, a file cut
in half, an object nested a hundred thousand deep, an encryption dictionary nobody can open.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import sys
import time
import tomllib
import zlib
from importlib import metadata
from pathlib import Path

import pytest

from ml_stack.net import pdfread
from ml_stack.net.pdfread import Limits, PdfRefused
from ml_stack.net.pdftext import visible_text
from ml_stack.redteam.minipdf import Doc
from ml_stack.sources import datasheet, pdf

ROOT = Path(__file__).resolve().parents[1]
BODY, SECTION, CHAPTER, TITLE, FOOT = 9.0, 13.0, 15.0, 22.0, 7.5


@pytest.fixture(autouse=True)
def default_engine(monkeypatch):
    monkeypatch.delenv("ML_STACK_PDF_ENGINE", raising=False)


def a_textbook(path, *, outline=True) -> str:
    doc = Doc()
    doc.title = "Lattice Studies"
    front = doc.page()
    front.text(40, 80, "Lattice Studies", size=TITLE)
    one = doc.page()
    one.text(40, 80, "CHAPTER 1", size=CHAPTER)
    one.text(40, 120, "The Glimmer Cascade", size=TITLE)
    one.text(40, 180, "Glimmer nodes are the smallest part of a lattice that can hold a charge.",
             size=BODY)
    two = doc.page()
    two.text(40, 80, "1.1 Glimmer Nodes", size=SECTION, bold=True)
    two.text(40, 120, "A node that has been charged is said to be quick-\nened, and it passes "
             "its charge on.", size=BODY)
    two.text(40, 180, "glimmer node", size=BODY, bold=True)
    two.image(40, 220, 180, 100)
    two.text(40, 380, "FIGURE 1.1 A glimmer node in cross-section.", size=FOOT, bold=True)
    three = doc.page()
    three.text(40, 80, "CHAPTER 2", size=CHAPTER)
    three.text(40, 120, "Vault Currents", size=TITLE)
    three.text(40, 180, "2.1 Currents in Practice", size=SECTION, bold=True)
    three.text(40, 230, "Vault currents run between quickened nodes.", size=BODY)
    for n, page in enumerate(doc.pages[1:], 1):
        page.text(40, 600, f"{n} The Glimmer Cascade", size=FOOT, bold=True)
        page.text(380, 600, str(n), size=FOOT, bold=True)
    if outline:
        doc.outline = [(1, "Chapter 1 The Glimmer Cascade", 2), (2, "1.1 Glimmer Nodes", 3),
                       (1, "Chapter 2 Vault Currents", 4), (2, "2.1 Currents in Practice", 4)]
    where = Path(path)
    where.write_bytes(doc.to_bytes())
    return str(where)


# -- the default engine, the opt-in engine, and which one is chosen ---------------------------


def test_the_default_engine_is_the_permissive_one_and_a_bad_name_is_refused(monkeypatch):
    assert pdfread.engine() == "pdfminer"
    monkeypatch.setenv("ML_STACK_PDF_ENGINE", "pymupdf")
    assert pdfread.engine() == "pymupdf"
    monkeypatch.setenv("ML_STACK_PDF_ENGINE", "poppler")
    with pytest.raises(ValueError, match="one of pdfminer, pymupdf"):
        pdfread.engine()


def test_mupdf_is_never_imported_by_a_default_read(tmp_path):
    """Run in a fresh interpreter, so a copy installed beside ml-stack proves nothing by being there."""
    where = a_textbook(tmp_path / "lattice.pdf")
    code = ("import sys\nfrom ml_stack.sources import pdf\nfrom ml_stack.net.pdftext import visible_text\n"
            f"d = pdf.read({where!r})\nvisible_text({where!r})\n"
            "assert len(d.sections) == 2, d.sections\n"
            "assert 'pymupdf' not in sys.modules and 'fitz' not in sys.modules, 'MuPDF was imported'\n"
            "print('ok')")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}   # this checkout, not an installed copy
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert done.stdout.strip() == "ok", done.stderr[-600:]


def test_the_mupdf_only_features_refuse_by_default_even_when_mupdf_is_installed():
    with pytest.raises(RuntimeError, match="pdf-agpl"):
        pdf._pymupdf()


def test_the_mupdf_engine_reads_the_same_book_when_asked_for_by_name(tmp_path, monkeypatch):
    pytest.importorskip("pymupdf", reason="MuPDF is AGPL and opt-in: pip install 'ml-stack[pdf-agpl]'")
    where = a_textbook(tmp_path / "lattice.pdf")
    default = pdf.read(where)
    monkeypatch.setenv("ML_STACK_PDF_ENGINE", "pymupdf")
    chosen = pdf.read(where)
    assert [(s.number, s.title) for s in chosen.sections] == [(s.number, s.title) for s in default.sections]


# -- reading with the default engine --------------------------------------------------------


def test_a_book_is_read_into_chapters_sections_terms_and_a_figure(tmp_path):
    document = pdf.read(a_textbook(tmp_path / "lattice.pdf"), images=True)
    assert document.how == "toc" and document.title == "Lattice Studies"
    assert [(s.chapter, s.number, s.title) for s in document.sections] == [
        ("1", "1.1", "Glimmer Nodes"), ("2", "2.1", "Currents in Practice")]
    first = document.sections[0]
    assert "quickened, and it passes its charge on." in first.text, "a broken word is mended"
    assert first.key_terms == ["glimmer node"]
    assert "1 The Glimmer Cascade" not in first.text, "the running foot is dropped"
    (figure,) = first.figures
    assert figure.label == "Figure 1.1" and figure.png[:8] == b"\x89PNG\r\n\x1a\n"
    assert (figure.width, figure.height) == (60, 40)


def test_without_an_outline_the_headings_are_read_off_the_way_they_are_set(tmp_path):
    document = pdf.read(a_textbook(tmp_path / "plain.pdf", outline=False))
    assert document.how == "headings"
    assert [(c.number, c.title) for c in document.chapters] == [
        ("1", "The Glimmer Cascade"), ("2", "Vault Currents")]


def hidden_kinds() -> bytes:
    doc = Doc()
    doc.title = "Spec sheet"
    doc.layers = {"secret": False}
    p = doc.page()
    p.text(72, 72, "VISIBLE heading", size=12)
    p.text(72, 100, "WHITE ATTACK", color=(1, 1, 1))
    p.text(72, 130, "TINY ATTACK", size=0.5)
    p.text(72, 160, "INVISIBLE ATTACK", render=3)
    p.text(2000, 2000, "OFFPAGE ATTACK")
    p.text(72, 200, "LAYER ATTACK", layer="secret")
    p.text(72, 230, "CLEAR ATTACK", alpha=0.0)
    p.text(72, 260, "zero width text")
    return doc.to_bytes()


def test_only_the_text_a_reader_can_see_is_returned(tmp_path):
    path = tmp_path / "t.pdf"
    path.write_bytes(hidden_kinds())
    title, text, pages, removed = visible_text(path)
    assert "VISIBLE heading" in text and "ATTACK" not in text
    assert title == "Spec sheet" and pages == 1 and removed >= 6


def test_datasheet_text_and_part_matching_need_no_mupdf(tmp_path):
    doc = Doc()
    doc.page().text(50, 80, "ACME1234 buck converter, 3 A", size=11)
    path = tmp_path / "ds.pdf"
    path.write_bytes(doc.to_bytes())
    assert "buck converter" in datasheet.text(path)[1]
    assert datasheet.part_match(path, "ACME1234") == (1.0, True)
    assert datasheet.part_match(path, "ZETA9999")[0] == 0.0


# -- hostile files: bounded, and refused with a reason ------------------------------------------


def raw_pdf(objects: dict[int, bytes], root: int = 1, extra_trailer: bytes = b"") -> bytes:
    out = bytearray(b"%PDF-1.7\n")
    offsets = {}
    for n, body in objects.items():
        offsets[n] = len(out)
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    start = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (max(objects) + 1)
    for n in range(1, max(objects) + 1):
        out += b"%010d 00000 n \n" % offsets.get(n, 0)
    out += b"trailer\n<< /Size %d /Root %d 0 R %s>>\nstartxref\n%d\n%%%%EOF\n" % (
        max(objects) + 1, root, extra_trailer, start)
    return bytes(out)


def one_page(content: bytes, *, filter_: bytes = b"") -> bytes:
    return raw_pdf({
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R "
           b"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>",
        4: b"<< /Length %d %s>>\nstream\n" % (len(content), filter_) + content + b"\nendstream"})


def read(tmp_path, data: bytes, **limits):
    path = tmp_path / "hostile.pdf"
    path.write_bytes(data)
    return pdfread.load(path, limits=Limits(**limits))


def test_a_decompression_bomb_is_refused_at_the_stream_bound(tmp_path):
    packed = zlib.compress(b"BT /F1 12 Tf (x) Tj ET\n" + b" " * (400 * 1024 * 1024), 9)
    assert len(packed) < 1_000_000, "the bomb itself is small"
    data = one_page(packed, filter_=b"/Filter /FlateDecode ")
    with pytest.raises(PdfRefused, match="inflates to more than"):
        read(tmp_path, data)
    with pytest.raises(PdfRefused, match="inflates to more than 1048576"):
        read(tmp_path, one_page(zlib.compress(b" " * (8 << 20)), filter_=b"/Filter /FlateDecode "),
             max_stream=1 << 20)


def test_a_page_tree_that_claims_a_billion_pages_is_refused_without_walking_it(tmp_path):
    data = raw_pdf({1: b"<< /Type /Catalog /Pages 2 0 R >>",
                    2: b"<< /Type /Pages /Kids [] /Count 1000000000 >>"})
    with pytest.raises(PdfRefused, match="claims 1000000000 pages"):
        read(tmp_path, data)


def test_more_real_pages_than_the_bound_is_refused(tmp_path):
    doc = Doc()
    for n in range(12):
        doc.page().text(40, 80, f"page {n}")
    with pytest.raises(PdfRefused, match="more than 5 pages"):
        read(tmp_path, doc.to_bytes().replace(b"/Count 12", b""), max_pages=5)  # no count to trust


def test_a_page_tree_that_contains_itself_ends(tmp_path):
    data = raw_pdf({1: b"<< /Type /Catalog /Pages 2 0 R >>",
                    2: b"<< /Type /Pages /Kids [2 0 R 3 0 R] /Count 2 >>",
                    3: b"<< /Type /Pages /Kids [2 0 R] /Count 1 >>"})
    try:
        loaded = read(tmp_path, data, timeout_s=20.0)
    except PdfRefused:
        return  # refused is fine; hanging or crashing is not
    assert loaded.page_count == 0


def test_a_file_cut_in_half_is_refused_or_read_as_far_as_it_goes(tmp_path):
    doc = Doc()
    doc.page().text(40, 80, "first page survives")
    data = doc.to_bytes()
    try:
        loaded = read(tmp_path, data[: len(data) // 2], timeout_s=20.0)
    except PdfRefused as refusal:
        assert str(refusal)
    else:
        assert loaded.page_count <= 1


def test_something_that_is_not_a_pdf_is_refused(tmp_path):
    with pytest.raises(PdfRefused, match="not a readable PDF"):
        read(tmp_path, b"MZ\x90\x00 this is not a pdf at all" * 20)


def test_an_object_nested_a_hundred_thousand_deep_is_refused_not_crashed(tmp_path):
    deep = b"[" * 100_000 + b"]" * 100_000
    data = one_page(b"BT /F1 12 Tf (x) Tj ET").replace(
        b"/MediaBox", b"/Junk " + deep + b" /MediaBox")
    try:
        loaded = read(tmp_path, data, timeout_s=4.0)
    except PdfRefused as refusal:
        assert str(refusal)
    else:
        assert loaded.page_count <= 1


def test_an_encrypted_pdf_that_needs_a_password_is_refused(tmp_path):
    data = raw_pdf({
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [] /Count 0 >>",
        3: b"<< /Filter /Standard /V 1 /R 2 /P -4 /O (" + b"o" * 32 + b") /U (" + b"u" * 32 + b") >>"},
        extra_trailer=b"/Encrypt 3 0 R /ID [<00112233445566778899aabbccddeeff> "
                      b"<00112233445566778899aabbccddeeff>] ")
    with pytest.raises(PdfRefused, match="encrypted"):
        read(tmp_path, data)


def test_a_file_over_the_byte_bound_is_refused_before_it_is_opened(tmp_path):
    with pytest.raises(PdfRefused, match="over 100"):
        read(tmp_path, one_page(b"BT ET"), max_bytes=100)


def scattered(count: int) -> bytes:
    """One page of ``count`` single characters at pseudo-random places: the layout step's worst case."""
    rng = random.Random(1)
    ops = b"BT /F1 6 Tf\n" + b"".join(
        b"1 0 0 1 %d %d Tm (a) Tj\n" % (rng.randint(0, 290), rng.randint(0, 290))
        for _ in range(count)) + b"ET"
    return one_page(ops)


def test_a_page_with_more_characters_than_a_page_holds_is_refused_before_it_is_laid_out(tmp_path):
    started = time.monotonic()
    with pytest.raises(PdfRefused, match="a page holds more than 8000 characters"):
        read(tmp_path, scattered(9000))
    assert time.monotonic() - started < 20


def test_work_that_takes_too_long_is_killed(tmp_path):
    """20,000 scattered characters lay out in over a minute; the page bound is lifted so the
    time bound is what stops it."""
    started = time.monotonic()
    with pytest.raises(PdfRefused, match="took more than 4 s"):
        read(tmp_path, scattered(20000), timeout_s=4.0, max_page_chars=1_000_000)
    assert time.monotonic() - started < 30


# -- the licence of what `pip install ml-stack[all]` and `[redteam]` bring ------------------------

AGPL = re.compile(r"\b(AGPL|Affero|GPL-?[23]|GNU General Public)", re.IGNORECASE)
LGPL = re.compile(r"\bLGPL|Lesser", re.IGNORECASE)
ALLOWED_COPYLEFT: dict[str, str] = {}
"""Packages allowed to carry a (L)GPL-family licence in `all` or `redteam`, each with the reason.
Empty: nothing in these extras is AGPL or GPL."""


def closure(extra: str, table: dict[str, list[str]]) -> set[str]:
    names: set[str] = set()
    for requirement in table[extra]:
        found = re.match(r"ml-stack\[([^\]]+)\]", requirement)
        if found:
            for inner in found.group(1).split(","):
                names |= closure(inner.strip(), table)
        else:
            names.add(re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0].lower().replace("_", "-"))
    return names


def test_all_and_redteam_pull_in_no_agpl_package():
    table = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert "pymupdf" in table["pdf-agpl"][0], "the AGPL engine has an extra whose name says so"
    for extra in ("all", "redteam", "pdf"):
        names = closure(extra, table)
        assert not names & {"pymupdf", "fitz", "pymupdf4llm"}, f"{extra} pulls MuPDF: {names}"
    assert "pdfminer.six" in closure("pdf", table)
    assert closure("pdf-render", table) == {"pypdfium2", "pillow"} <= closure("pdf", table)
    assert "pypdfium2" in closure("all", table) and "pypdfium2" in closure("redteam", table)


def test_pdfium_and_the_licences_of_the_binary_it_ships_are_permissive():
    """Page rendering is in `all` and `redteam`, so what pypdfium2 carries must be BSD/Apache.
    Its wheel bundles the PDFium binary and, per library inside it, that library's licence."""
    try:
        dist = metadata.distribution("pypdfium2")
    except metadata.PackageNotFoundError:
        pytest.skip("pip install 'ml-stack[pdf-render]'")
    claim = dist.metadata.get("License-Expression") or dist.metadata.get("License") or ""
    assert "BSD-3-Clause" in claim and "Apache-2.0" in claim and not AGPL.search(claim), claim
    files = {str(f): f for f in dist.files or ()}
    licences = {name.rsplit("/", 1)[-1]: f for name, f in files.items() if "licenses/" in name}
    assert {"pdfium.txt", "Apache-2.0.txt", "BSD-3-Clause.txt"} <= set(licences)
    assert "Redistribution and use in source and binary forms" in dist.locate_file(licences["pdfium.txt"]).read_text()
    for name, f in licences.items():
        text = dist.locate_file(f).read_text(errors="replace")
        for banned in ("GNU AFFERO", "GNU LESSER", "GNU LIBRARY"):
            assert banned not in text.upper(), f"{name} carries {banned}"


def test_the_installed_packages_of_those_extras_carry_no_copyleft_licence():
    table = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    checked = 0
    for name in sorted(closure("all", table) | closure("redteam", table)):
        try:
            meta = metadata.metadata(name)
        except metadata.PackageNotFoundError:
            continue
        legacy = meta.get("License") or ""                                  # an identifier, or (matplotlib, ...) the whole licence text
        text = " ".join([meta.get("License-Expression") or "", legacy if len(legacy) <= 80 else "",
                         *[c for c in (meta.get_all("Classifier") or []) if c.startswith("License")]])
        checked += 1
        if AGPL.search(text) and not LGPL.search(text) and name not in ALLOWED_COPYLEFT:
            pytest.fail(f"{name} is licensed {text!r}: not for `all` or `redteam`")
    assert checked, "none of the extras' packages is installed, so nothing was compared"
