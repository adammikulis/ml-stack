"""Downloading with provenance, against a real local server."""

from __future__ import annotations

import hashlib
import json

import pytest

from ml_stack.http import Refused
from ml_stack.scrape.download import ACCEPT_CAD, DownloadError, Wanted, as_dict, download, kind_of
from ml_stack.scrape.polite import Disallowed, Polite
from tests.web_site import allow_all, serving

PDF = b"%PDF-1.7\n" + b"x" * 5000
STEP = b"ISO-10303-21;\nHEADER;\nENDSEC;\n"


@pytest.fixture
def site():
    yield from serving()


def polite(**kw) -> Polite:
    return Polite(**{"guard": allow_all, "min_interval_s": 0.0, "backoff_s": 0.0,
                     "robots": False, **kw})


def test_a_pdf_lands_with_its_record(site, tmp_path):
    site.file("/ds/part.pdf", PDF, "application/pdf")
    got = download(f"{site.base}/ds/part.pdf", tmp_path, polite=polite())
    assert got.path.read_bytes() == PDF
    assert got.path.name.endswith("_part.pdf")
    assert got.sha256 == hashlib.sha256(PDF).hexdigest()
    assert got.content_type == "application/pdf" and got.size == len(PDF)
    record = json.loads(got.record.read_text())
    assert record["url"] == f"{site.base}/ds/part.pdf"
    assert record["sha256"] == got.sha256 and record["version"] == 1
    assert record["robots"] == "robots.txt not consulted"
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]


def test_a_second_request_for_the_same_url_is_free(site, tmp_path):
    site.file("/a.pdf", PDF, "application/pdf")
    first = download(f"{site.base}/a.pdf", tmp_path, polite=polite())
    again = download(f"{site.base}/a.pdf", tmp_path, polite=polite())
    assert again.cached and not first.cached and again.path == first.path
    assert site.hits == ["/a.pdf"]
    download(f"{site.base}/a.pdf", tmp_path, wanted=Wanted(refresh=True), polite=polite())
    assert site.hits == ["/a.pdf", "/a.pdf"]


def test_a_cache_entry_whose_file_changed_is_fetched_again(site, tmp_path):
    site.file("/a.pdf", PDF, "application/pdf")
    first = download(f"{site.base}/a.pdf", tmp_path, polite=polite())
    first.path.write_bytes(b"%PDF-tampered")
    assert not download(f"{site.base}/a.pdf", tmp_path, polite=polite()).cached


def test_redirects_are_followed_and_the_final_address_recorded(site, tmp_path):
    site.routes["/old"] = (302, {"Location": "/files/new.pdf"}, b"")
    site.file("/files/new.pdf", PDF, "application/pdf")
    got = download(f"{site.base}/old", tmp_path, polite=polite())
    assert got.url.endswith("/old") and got.final_url.endswith("/files/new.pdf")
    assert got.path.name.endswith("_new.pdf")


def test_a_redirect_to_a_private_host_is_refused(site, tmp_path):
    site.routes["/bounce"] = (302, {"Location": "http://10.0.0.5/secret.pdf"}, b"")

    def only_the_site(url):
        if url.startswith(site.base):
            return url
        raise Refused(f"{url} is not public")
    with pytest.raises(Refused):
        download(f"{site.base}/bounce", tmp_path, polite=polite(guard=only_the_site))
    assert list(tmp_path.iterdir()) == []


def test_a_bot_wall_served_as_the_pdf_is_rejected(site, tmp_path):
    site.file("/x.pdf", b"<!DOCTYPE html><title>Access denied</title>", "text/html")
    with pytest.raises(DownloadError, match="is not one of application/pdf"):
        download(f"{site.base}/x.pdf", tmp_path, polite=polite())
    assert list(tmp_path.iterdir()) == []


def test_a_wrong_magic_under_the_right_type_is_rejected(site, tmp_path):
    site.file("/x.pdf", b"not a pdf at all", "application/pdf")
    with pytest.raises(DownloadError):
        download(f"{site.base}/x.pdf", tmp_path, polite=polite())


def test_a_pdf_served_as_octet_stream_is_accepted(site, tmp_path):
    site.file("/blob", PDF, "application/octet-stream",
              **{"Content-Disposition": 'attachment; filename="TPS 6213.pdf"'})
    got = download(f"{site.base}/blob", tmp_path, polite=polite())
    assert got.path.name.endswith("_TPS_6213.pdf")


def test_a_declared_length_over_the_limit_is_refused_before_the_body(site, tmp_path):
    site.file("/big.pdf", PDF, "application/pdf")
    with pytest.raises(DownloadError, match="over the 1000"):
        download(f"{site.base}/big.pdf", tmp_path, wanted=Wanted(max_bytes=1000), polite=polite())
    assert list(tmp_path.iterdir()) == []


def test_step_and_zip_are_accepted_when_asked_for(site, tmp_path):
    site.file("/m.step", STEP, "application/octet-stream")
    site.file("/m.zip", b"PK\x03\x04rest", "application/zip")
    assert download(f"{site.base}/m.step", tmp_path, wanted=Wanted(accept=ACCEPT_CAD), polite=polite()).size
    assert download(f"{site.base}/m.zip", tmp_path, wanted=Wanted(accept=ACCEPT_CAD), polite=polite()).size
    with pytest.raises(DownloadError):
        download(f"{site.base}/m.step", tmp_path, wanted=Wanted(refresh=True), polite=polite())


@pytest.mark.parametrize(("served", "first", "accept", "kind"), [
    ("application/pdf", b"%PDF-1.4", ("application/pdf",), "pdf"),
    ("application/pdf; charset=binary", b"%PDF-1.4", ("application/pdf",), "pdf"),
    ("text/html", b"%PDF-1.4", ("application/pdf",), ""),
    ("application/octet-stream", b"  ISO-10303-21;", ("model/step",), "step"),
    ("application/octet-stream", b"ISO-10303-21;", ("application/pdf",), ""),
    ("text/plain", b"(footprint \"R_0603\"", ("text/x-kicad",), "kicad"),
    ("image/png", b"\x89PNG\r\n", ("image/png",), "png"),
])
def test_kind_of(served, first, accept, kind):
    assert kind_of(served, first, accept) == kind


def test_robots_forbidding_a_download_stops_it(site, tmp_path):
    site.file("/robots.txt", b"User-agent: *\nDisallow: /pdfs\n", "text/plain")
    site.file("/pdfs/a.pdf", PDF, "application/pdf")
    with pytest.raises(Disallowed):
        download(f"{site.base}/pdfs/a.pdf", tmp_path, polite=polite(robots=True))
    assert "/pdfs/a.pdf" not in site.hits


def test_the_record_says_robots_allowed_it(site, tmp_path):
    site.file("/robots.txt", b"User-agent: *\nDisallow: /nope\n", "text/plain")
    site.file("/a.pdf", PDF, "application/pdf")
    got = download(f"{site.base}/a.pdf", tmp_path, wanted=Wanted(licence="CC-BY-4.0"),
                   polite=polite(robots=True))
    assert got.robots == "robots.txt allows" and got.licence == "CC-BY-4.0"
    assert as_dict(got)["path"] == str(got.path)
