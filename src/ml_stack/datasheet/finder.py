"""Search for a part's datasheet, rank the candidates, download the best, and check the PDF
names the part."""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.http import Refused, ServerError
from ml_stack.scrape.crawl import links
from ml_stack.scrape.download import DownloadError, download
from ml_stack.scrape.polite import Polite
from ml_stack.sources import datasheet
from ml_stack.web import Engine, SearchUnavailable, downloads_dir, politeness, search_pages

MANUFACTURERS: dict[str, tuple[str, ...]] = {
    "texas instruments": ("ti.com",), "ti": ("ti.com",),
    "stmicroelectronics": ("st.com",), "st": ("st.com",), "st microelectronics": ("st.com",),
    "microchip": ("microchip.com",), "atmel": ("microchip.com",),
    "nxp": ("nxp.com",), "analog devices": ("analog.com",), "adi": ("analog.com",),
    "maxim": ("analog.com", "maximintegrated.com"), "linear technology": ("analog.com",),
    "infineon": ("infineon.com",), "on semiconductor": ("onsemi.com",),
    "onsemi": ("onsemi.com",), "renesas": ("renesas.com",), "espressif": ("espressif.com",),
    "rohm": ("rohm.com",), "murata": ("murata.com",), "nordic semiconductor": ("nordicsemi.com",),
    "diodes incorporated": ("diodes.com",), "vishay": ("vishay.com",),
    "silicon labs": ("silabs.com",), "winbond": ("winbond.com",), "bosch": ("bosch-sensortec.com",),
    "tdk": ("tdk.com", "invensense.tdk.com"), "knowles": ("knowles.com",),
}
DISTRIBUTORS = ("digikey.", "mouser.", "lcsc.com", "farnell.", "rs-online.", "arrow.com",
                "newark.com", "element14.", "jlcpcb.com", "tme.eu")
AGGREGATORS = ("alldatasheet.", "datasheet4u.", "datasheetq.", "datasheetspdf.", "datasheetcafe.",
               "datasheetarchive.", "pdf1.alldatasheet.", "manualslib.")
TRIES = 4
PAGES = 2
MOST_CANDIDATES = 12
MATCH = 0.6
"""The share of a part number a PDF must name to count as its datasheet."""


class NotFound(LookupError):
    """No candidate downloaded as a PDF that names the part."""


@dataclass(frozen=True)
class Candidate:
    """A search result and why it ranks where it does."""

    url: str
    title: str
    score: int
    why: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Found:
    """A datasheet on disk: where, the PDF's address, the search result or page it came from
    (``source``), how sure, and what was tried first."""

    path: str
    url: str
    sha256: str
    pages: int
    confidence: float
    final_url: str = ""
    source: str = ""
    matched: float = 0.0
    tried: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _host(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def _domains(manufacturer: str) -> tuple[str, ...]:
    name = " ".join(manufacturer.lower().split())
    if name in MANUFACTURERS:
        return MANUFACTURERS[name]
    squashed = re.sub(r"[^a-z0-9]", "", name)
    return (squashed,) if len(squashed) >= 3 else ()


def rank(rows: list[dict[str, Any]], part: str, manufacturer: str = "") -> list[Candidate]:
    """Search results best first: a manufacturer's own domain, a known distributor, a PDF link,
    'datasheet' and the part number in the title or address; aggregator sites last."""
    wanted = re.sub(r"[^a-z0-9]", "", part.lower())
    domains = _domains(manufacturer)
    out: list[Candidate] = []
    for row in rows:
        url, title = str(row.get("url") or ""), str(row.get("title") or "")
        host, path = _host(url), urllib.parse.urlsplit(url).path.lower()
        hay = re.sub(r"[^a-z0-9]", "", (url + title).lower())
        score, why = 0, []
        for flag, points, label in (
                (bool(domains) and any(d in host for d in domains), 40, "manufacturer domain"),
                (any(d in host for d in DISTRIBUTORS), 15, "distributor"),
                (any(d in host for d in AGGREGATORS), -20, "aggregator"),
                (path.endswith(".pdf"), 30, "pdf link"),
                ("datasheet" in (url + title).lower(), 15, "datasheet in title or address"),
                (bool(wanted) and wanted in hay, 20, "part number in title or address")):
            if flag:
                score += points
                why.append(label)
        out.append(Candidate(url=url, title=title, score=score, why=why))
    return sorted(out, key=lambda c: -c.score)


def _pdf_links(html: str, url: str, part: str) -> list[str]:
    """The PDF links on a page, those that say 'datasheet' or hold the part number first."""
    wanted = re.sub(r"[^a-z0-9]", "", part.lower())
    found = [link for link in links(html, url)[0] if urllib.parse.urlsplit(link.href).path.lower()
             .endswith(".pdf")]

    def key(link: Any) -> int:
        text = re.sub(r"[^a-z0-9]", "", (link.href + link.text).lower())
        return -(("datasheet" in text) + (wanted in text))
    return [link.href for link in sorted(found, key=key)][:3]


def _page_html(polite: Polite, url: str) -> str:
    with polite.fetch(url, accept="text/html,*/*;q=0.5") as reply:
        return str(reply.read(2 * 1024 * 1024).decode("utf-8", "replace"))


def _confidence(candidate: Candidate, matched: float, first_page: bool) -> float:
    manufacturer = "manufacturer domain" in candidate.why
    trusted = manufacturer or "distributor" in candidate.why
    value = 0.45 * matched + 0.15 * first_page + 0.25 * manufacturer + 0.1 * trusted \
        + 0.05 * ("pdf link" in candidate.why)
    return round(min(value, 1.0), 2)


def _try(url: str, candidate: Candidate, part: str, dest: Path, polite: Polite
         ) -> Found | str:
    """The datasheet at ``url`` if it downloads as a PDF naming the part, else why not."""
    try:
        got = download(url, dest, polite=polite)
        matched, first = datasheet.part_match(got.path, part)
    except (DownloadError, Refused, ServerError, OSError) as exc:
        return f"{url}: {exc}"
    if matched < MATCH:
        return f"{url}: the PDF does not name {part}"
    _, _, pages = datasheet.text(got.path, limit=0)
    return Found(path=str(got.path), url=url, source=candidate.url, final_url=got.final_url,
                 sha256=got.sha256, pages=pages, matched=round(matched, 2),
                 confidence=_confidence(candidate, matched, first))



def find(part: str, manufacturer: str = "", *, engine: Engine | None = None,
         polite: Polite | None = None, dest: str | Path | None = None) -> Found:
    """Download the best datasheet PDF for ``part`` and check it names the part.

    Searches (two pages of results), ranks the candidates, and tries them best first, up to
    ``TRIES`` downloads: a PDF link is downloaded, a page is searched for PDF links to
    try. A PDF counts when at least ``MATCH`` of the part number is in its text. Raises
    ``NotFound``, listing what was tried, when nothing does.
    """
    polite = polite or politeness()
    dest = Path(dest) if dest else downloads_dir()
    rows: list[dict[str, Any]] = []
    try:
        for batch in search_pages(f"{part} {manufacturer} datasheet pdf", pages=PAGES,
                                  engine=engine):
            rows += batch
    except SearchUnavailable as exc:
        raise NotFound(f"search unavailable: {exc}") from exc
    tried: list[str] = []
    for candidate in rank(rows, part, manufacturer)[:MOST_CANDIDATES]:
        if len(tried) >= TRIES:
            break
        urls = [candidate.url]
        if not urllib.parse.urlsplit(candidate.url).path.lower().endswith(".pdf"):
            try:
                urls = _pdf_links(_page_html(polite, candidate.url), candidate.url, part)
            except (Refused, ServerError, OSError) as exc:
                tried.append(f"{candidate.url}: {exc}")
                continue
        for url in urls:
            done = _try(url, candidate, part, dest, polite)
            if isinstance(done, Found):
                return Found(**{**asdict(done), "tried": tried})
            tried.append(done)
    raise NotFound(f"no datasheet naming {part!r} found; tried {tried or 'nothing'}")
