"""Finding a datasheet: ranking search results, then downloading and checking candidates from a
local server, with the search engine a stand-in."""

from __future__ import annotations

import pytest

from ml_stack import web
from ml_stack.datasheet import NotFound, find, rank, tools
from ml_stack.scrape.polite import Polite
from tests.web_site import allow_all, serving


def rows(*pairs):
    return [{"title": t, "url": u, "snippet": ""} for t, u in pairs]


def test_a_manufacturers_pdf_outranks_a_distributor_and_an_aggregator():
    ranked = rank(rows(
        ("QX2200 Datasheet", "https://www.alldatasheet.com/datasheet-pdf/qx2200.html"),
        ("QX2200 product page", "https://www.digikey.com/en/products/detail/qx2200"),
        ("QX2200 Datasheet", "https://www.quenlow.com/resource/qx2200.pdf"),
    ), "QX2200", "Quenlow")
    assert [c.url.split("/")[2] for c in ranked] == [
        "www.quenlow.com", "www.digikey.com", "www.alldatasheet.com"]
    assert "manufacturer domain" in ranked[0].why and "pdf link" in ranked[0].why
    assert "aggregator" in ranked[-1].why


def test_a_known_maker_name_maps_to_its_domains():
    [first, second] = rank(rows(("a", "https://www.ti.com/lit/ds/x.pdf"),
                                ("b", "https://example.org/x.pdf")), "X1234", "Texas Instruments")
    assert first.url.startswith("https://www.ti.com")
    assert first.score > second.score


def test_the_part_number_in_the_address_counts_whatever_its_punctuation():
    [hit] = rank(rows(("t", "https://x.example/PX-17A.pdf")), "px 17a")
    assert "part number in title or address" in hit.why


@pytest.fixture
def lan():
    yield from serving()


def polite():
    return Polite(guard=allow_all, robots=False, min_interval_s=0.0, backoff_s=0.0, tries=1)


def make_pdf(text):
    from ml_stack.redteam.minipdf import Doc

    doc = Doc()
    doc.page().text(50, 80, text, size=11)
    return doc.to_bytes()


def engine_for(lan, *paths):
    def engine(query, limit, page=1):
        return [] if page > 1 else [
            {"title": f"{p} datasheet", "url": f"{lan.base}{p}", "snippet": ""} for p in paths]
    return engine


def test_the_first_candidate_that_is_a_pdf_naming_the_part_wins(lan, tmp_path):
    lan.file("/wall.pdf", b"<html>are you a robot</html>", "text/html")
    lan.file("/other.pdf", make_pdf("ZZ9999 something else entirely"), "application/pdf")
    lan.file("/good.pdf", make_pdf("QX2200 low-noise regulator"), "application/pdf")
    got = find("QX2200", "Quenlow", engine=engine_for(lan, "/wall.pdf", "/other.pdf", "/good.pdf"),
               polite=polite(), dest=tmp_path)
    assert got.url.endswith("/good.pdf") and got.pages == 1 and got.matched == 1.0
    assert len(got.sha256) == 64 and 0.5 < got.confidence <= 1.0
    assert len(got.tried) == 2
    assert any("does not name QX2200" in t for t in got.tried)
    assert any("is not one of application/pdf" in t for t in got.tried)


def test_a_landing_page_is_searched_for_a_pdf_link(lan, tmp_path):
    lan.page("/product", '<a href="/manual.pdf">Manual</a><a href="/ds/qx2200.pdf">Datasheet</a>')
    lan.file("/ds/qx2200.pdf", make_pdf("QX2200 data"), "application/pdf")
    lan.file("/manual.pdf", make_pdf("QX2200 manual"), "application/pdf")
    got = find("QX2200", engine=engine_for(lan, "/product"), polite=polite(), dest=tmp_path)
    assert got.url.endswith("/ds/qx2200.pdf") and got.source.endswith("/product")


def test_a_family_datasheet_matches_a_longer_part_number_with_less_confidence(lan, tmp_path):
    lan.file("/fam.pdf", make_pdf("QX2200xx family"), "application/pdf")
    exact = find("QX2200", engine=engine_for(lan, "/fam.pdf"), polite=polite(), dest=tmp_path)
    longer = find("QX2200CIT6", engine=engine_for(lan, "/fam.pdf"), polite=polite(),
                  dest=tmp_path / "b")
    assert 0.6 <= longer.matched < 1.0
    assert longer.confidence < exact.confidence


def test_nothing_that_names_the_part_raises_with_what_was_tried(lan, tmp_path):
    lan.file("/a.pdf", make_pdf("ZZ9999"), "application/pdf")
    with pytest.raises(NotFound, match="tried"):
        find("QX2200", engine=engine_for(lan, "/a.pdf"), polite=polite(), dest=tmp_path)


def test_a_search_that_will_not_answer_is_not_found():
    def down(query, limit, page=1):
        raise web.SearchUnavailable("rate limited")
    with pytest.raises(NotFound, match="search unavailable"):
        find("QX2200", engine=down)


@pytest.fixture
def scratch(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "downloads_dir", lambda: tmp_path)
    monkeypatch.setattr("ml_stack.datasheet.tooling.downloads_dir", lambda: tmp_path)
    return tmp_path


def test_the_find_tool_returns_the_record_or_a_reason(lan, scratch, monkeypatch):
    lan.file("/good.pdf", make_pdf("QX2200 regulator"), "application/pdf")
    monkeypatch.setattr("ml_stack.datasheet.finder.politeness", polite)
    finding, _, _ = (c for _, c in tools(engine=engine_for(lan, "/good.pdf")))
    got = finding({"part": "QX2200"})
    assert got["path"].startswith(str(scratch)) and got["pages"] == 1
    assert "pass a part number" in finding({})["none"]
    assert "no datasheet" in finding({"part": "NOPE9999"})["none"]


def test_the_reading_tools_refuse_a_path_outside_the_download_directory(scratch, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "x.pdf"
    outside.write_bytes(make_pdf("QX2200"))
    _, pins, outline = (c for _, c in tools())
    assert "pass the path" in pins({"path": str(outside)})["none"]
    assert "pass the path" in outline({"path": "/etc/hosts"})["none"]
    assert "pass the path" in pins({"path": str(scratch / ".." / "x.pdf")})["none"]


def test_the_outline_tool_is_offered_only_with_vision():
    assert [s["function"]["name"] for s, _ in tools(vision=False)] == [
        "datasheet_find", "datasheet_pins"]
    assert len(tools()) == 3
