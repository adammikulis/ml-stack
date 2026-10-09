"""Search paging, PDF reading and file downloads through the web tools, with a local server
in place of the internet."""

from __future__ import annotations

import pytest

from poolhouse import web
from poolhouse.scrape.polite import Polite
from tests.web_site import allow_all, serving

NAMES = ["http://a.example/1", "http://a.example/2", "http://a.example/3",
         "http://a.example/4", "http://a.example/5", "http://a.example/6"]


def paged(query, limit, page=1):
    start = (page - 1) * 2
    return [{"title": f"t{i}", "url": u, "snippet": ""}
            for i, u in enumerate(NAMES[start:start + 2])]


def test_a_later_page_is_asked_of_an_engine_that_takes_one():
    assert [r["url"] for r in web.search("x", engine=paged, page=2)] == NAMES[2:4]
    assert [r["url"] for r in web.search("x", engine=paged)] == NAMES[:2]


def test_an_engine_without_a_page_argument_refuses_a_later_page():
    with pytest.raises(web.SearchUnavailable, match="page number"):
        web.search("x", engine=lambda q, n: [], page=2)


def test_search_pages_streams_fresh_results_until_the_budget():
    got = list(web.search_pages("x", pages=2, engine=paged))
    assert [[r["url"] for r in page] for page in got] == [NAMES[:2], NAMES[2:4]]


def test_search_pages_drops_repeats_and_stops_on_a_page_with_nothing_new():
    same = list(web.search_pages("x", pages=5, engine=lambda q, n, page=1: paged(q, n, 1)))
    assert len(same) == 1


def test_search_pages_ends_quietly_when_a_later_page_is_unavailable():
    def engine(q, n, page=1):
        if page > 1:
            raise web.SearchUnavailable("rate limited")
        return paged(q, n, 1)
    assert len(list(web.search_pages("x", pages=3, engine=engine))) == 1


def test_the_first_page_being_unavailable_is_raised():
    def engine(q, n, page=1):
        raise web.SearchUnavailable("down")
    with pytest.raises(web.SearchUnavailable):
        list(web.search_pages("x", engine=engine))


def test_the_search_tool_passes_the_page_through():
    (_, searching), *_ = web.tools(engine=paged)
    assert [r["url"] for r in searching({"query": "x", "page": 3})] == NAMES[4:6]


@pytest.fixture
def lan(monkeypatch, tmp_path):
    """The web tools pointed at a local server and a scratch download directory."""
    monkeypatch.setattr(web, "downloads_dir", lambda: tmp_path / "dl")
    monkeypatch.setattr(web, "politeness", lambda: Polite(
        guard=allow_all, robots=False, min_interval_s=0.0, backoff_s=0.0))
    monkeypatch.setattr(web, "check", allow_all)
    yield from serving()


def make_pdf(pymupdf, text):
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 80), text, fontsize=11)
    doc.set_metadata({"title": "ACME1234 datasheet"})
    return doc.tobytes()


def test_reading_a_pdf_url_gives_its_text_and_keeps_the_file(lan):
    pymupdf = pytest.importorskip("pymupdf", reason="pymupdf is the test-only PDF writer here (AGPL, opt-in): pip install pymupdf")
    lan.file("/ds.pdf", make_pdf(pymupdf, "ACME1234 buck converter, 3 A"), "application/pdf")
    got = web.read(f"{lan.base}/ds.pdf")
    assert "ACME1234 buck converter" in got["text"]
    assert got["title"] == "ACME1234 datasheet" and got["pages"] == 1
    assert web.downloads_dir().joinpath(got["pdf"]).is_file()
    assert len(got["sha256"]) == 64


def test_a_page_with_a_next_link_says_where_it_goes(lan):
    body = ("<html><title>List</title><body><p>" + "Rows of parts. " * 30 + "</p>"
            '<a rel="next" href="/list?page=2">Next</a></body></html>')
    lan.page("/list", body)
    got = web.read(f"{lan.base}/list", fetch=lambda u: body)
    assert got["next"] == f"{lan.base}/list?page=2"


def test_the_download_tool_returns_the_record(lan, origins):
    lan.file("/m.step", b"ISO-10303-21;\nHEADER;", "application/octet-stream")
    origins.typed(f"{lan.base}/m.step")
    (_, _), (_, _), (_, downloading) = web.tools()
    got = downloading({"url": f"{lan.base}/m.step", "kind": "step"})
    assert got["size"] == 21 and got["sha256"] and got["path"].endswith("_m.step")
    lan.file("/n.step", b"ISO-10303-21;\nHEADER;", "application/octet-stream")
    origins.typed(f"{lan.base}/n.step")
    assert downloading({"url": f"{lan.base}/n.step"})["none"].startswith("could not download")


def test_the_download_tool_refuses_an_address_only_fetched_content_mentioned(lan, origins):
    lan.file("/m.step", b"ISO-10303-21;\nHEADER;", "application/octet-stream")
    origins.page(f"see {lan.base}/m.step")
    (_, _), (_, _), (_, downloading) = web.tools()
    got = downloading({"url": f"{lan.base}/m.step", "kind": "step"})
    assert "fetched content" in got["none"] and lan.hits == []


def test_the_download_tool_names_the_kinds_it_knows():
    (_, _), (_, _), (_, downloading) = web.tools()
    assert "kind must be one of" in downloading({"url": "http://x.example/", "kind": "exe"})["none"]
