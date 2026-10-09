# Web research: search, read, page, download

Measured 2026-10-02 on one macOS machine, Python 3.13, `poolhouse[web,pdf,scrape]`. The live
check is `POOLHOUSE_NET=1 pytest tests/test_web_live.py`; it finds a regulator's datasheet from its
part number, reads its pin table and finds its package pages.

## What there is

| Need | Call | Extra |
| --- | --- | --- |
| Search the web | `poolhouse.web.search(query, limit=8, page=1)` | `web` |
| Several result pages, no repeats | `poolhouse.web.search_pages(query, pages=3)` | `web` |
| A page as text (a PDF URL is read as a PDF; `next` is the following page) | `poolhouse.web.read(url)` | `web`, `pdf`, `scrape` for JS-only pages |
| A listing across pages | `poolhouse.scrape.crawl.pages(start, items, walk)` / `crawl(...)` | none |
| A listing behind "load more" | `poolhouse.scrape.crawl.more(page, items=...)` | `scrape` |
| A file with a provenance record | `poolhouse.scrape.download.download(url, dest, Wanted(...))` | none |
| A part's datasheet | `poolhouse.datasheet.find(part, manufacturer)` | `web`, `pdf` |
| Pin table as rows | `poolhouse.sources.datasheet.pin_tables(pdf)` | `pdf-agpl` (MuPDF) |
| Package outline and land-pattern pages, as PNG crops | `outline_pages(pdf)`, `render(pdf, page)` | `pdf` (pdfminer.six + PDFium) |

**PDF engine and licence.** The `pdf` extra reads PDFs with pdfminer.six (MIT) and Pillow, in a child
process with hard bounds (`poolhouse.net.pdfread`): a file over 32 MiB, over 1,500 pages, a stream that
inflates past 48 MiB, text past 24 M characters or a read past 90 s is refused with a clear
`PdfRefused`, as is an encrypted or unparseable file. Text that a reader cannot see (invisible render
mode, white, under 2 pt, fully transparent, off the page, in a layer that is switched off) is dropped
and counted. Pages are rendered to PNG by PDFium (pypdfium2, Apache-2.0/BSD-3-Clause; the `pdf-render` part of
`pdf`) in a second child started by file path with `-P` in a scrubbed environment
(`poolhouse.net.pdfrender`): at most 8 pages per call, a page side of at most 5,000 pt, 16 M pixels per
page, 150 M declared picture pixels, 32 MiB of PNG, 2 GiB resident memory (a watchdog, because macOS
does not enforce an address-space limit) and 60 s are the bounds; past any the file is refused with
`PdfRefused`. A crop is the box around everything on the page that is not white. Outline-page
detection reads the visible text and the count of stroked paths that pdfminer lays out. MuPDF is
AGPL-3.0, so it is not in `pdf`, `all` or `redteam`: `pip install 'poolhouse[pdf-agpl]'` and
`POOLHOUSE_PDF_ENGINE=pymupdf` select it, and only the datasheet pin tables need it (with it,
`render` also crops to the vector drawing alone). An installed copy is never used unless that
variable names it.

Model tools, as `(schema, callable)` pairs for `converse` (and loadable by name as
`python:poolhouse.web:tools` and `python:poolhouse.datasheet:tools`): `web_search` (with `page`),
`web_read`, `web_look` (vision), `web_download`, `datasheet_find`, `datasheet_pins`,
`datasheet_outline` (vision; returns cropped drawings as `_images`).

## Paging

`scrape.crawl.next_link(html, url)` finds the next page from, in order: `<link rel=next>` or
`rel=next` on an anchor; an anchor reading Next, Older or an arrow; a numbered anchor with the
smallest `page=`/`offset=` above this page's on the same listing; and the URL's own `page=N`
plus one when the page links to no other page number. `pages()` walks that under a `Walk`
budget (`max_pages`, `max_items`), drops rows already seen (`Walk.seen` can be kept across
runs), never fetches a URL twice, and stops on a page with no new rows or a 404 after the first
page. `more()` clicks a "load more" button in a Playwright page until it disappears or the page
stops growing.

Search paging: `ddgs` and SearXNG take `page`; an engine function that does not refuses a page
beyond the first.

## Manners

`scrape.polite.Polite` is what every fetch here goes through: robots.txt is read once per host
and respected (a missing file allows everything, a 4xx allows, a 5xx or unreachable host
forbids, `Crawl-delay` is honoured); at least `min_interval_s` (1 s) passes between requests to
one host; one request per host at a time; 429 and 5xx are retried with backoff. Set
`POOLHOUSE_ROBOTS=off` to stop consulting robots.txt and `POOLHOUSE_MIN_INTERVAL` to change the gap.
Every redirect target is checked against `http.check`, so a public page cannot send a read to a
private address.

## Downloads

`download()` accepts only the kinds asked for (`pdf`, `zip`, `step`, `kicad`, `png`, `jpeg`):
the content type must be that kind's or generic, and the first bytes must match (`%PDF-`, `PK`,
`ISO-10303`, `(footprint`). A bot-wall page served at a `.pdf` address is refused. The body is
size-limited (64 MB), written through a temporary file, and a `<name>.json` beside it records
the URL, final URL, sha256, content type, size, time, robots note and licence note. The same
URL again returns the file on disk. The licence note says it was not checked; pass the real one.

## What works and what does not

Worked on the test machine:

- Search for "STM32U575 datasheet pdf": eight results in 3 s, the manufacturer's PDF first.
- A manufacturer product page (ti.com) read as text without a browser; a distributor page
  (lcsc.com) read as a table.
- The TI TPS62130 datasheet PDF fetched in 0.4 s, 43 pages; 12 pin rows read from its two-row
  header table; the land-pattern and package-outline pages found and cropped.

Failed, and why:

- **st.com** answered nothing to urllib, curl or headless Chromium from this network (timeouts,
  HTTP/2 protocol errors). Its datasheets, including STM32U575, were not reachable from here.
  `find()` reports what it tried and falls back to other candidates.
- **digikey.com** serves a bot check ("verifying you are not a bot") to a plain fetch and to
  the headless browser. Nothing here solves that, by design.
- **mouser.com** `pdfDocs` links returned an HTML page at the PDF address; `download()` refuses
  it by its first bytes.
- A PDF read as an HTML page (before this module's PDF branch) returned binary garbage after
  122 s; `web_read` now routes any `.pdf` URL to the download and PDF reader.
- Pages that need a login, a captcha, or a paid account are out of scope.

## Ethics and limits

- Robots.txt is respected by default and the record says how it was treated.
- Requests are spaced and serial per host; do not raise concurrency against one site.
- No login, captcha, paywall or bot-check is bypassed. A site that blocks automated clients
  is not worked around; fetch the file by hand and put it in the download directory.
- Datasheets, models and footprints are the vendor's. The record carries the source URL so a
  licence can be checked; this library does not decide whether a file may be redistributed.
- The user agent says what it is (`Mozilla/5.0 (compatible; poolhouse)`).
