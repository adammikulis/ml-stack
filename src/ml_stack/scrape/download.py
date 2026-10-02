"""Downloading a file with a record of where it came from.

``download`` follows redirects (each hop vetted), checks the content type and the first
bytes against what the caller accepts, refuses a file over the size limit, writes through a
temporary file, and leaves a ``.json`` beside the result holding the source, the final
address, the hash and the time. Asking again for the same URL returns the file already on
disk.
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml_stack import files
from ml_stack.scrape.polite import Polite

MOST_BYTES = 64 * 1024 * 1024
CHUNK = 64 * 1024
RECORD_VERSION = 1
LICENCE = "not checked: confirm the source's terms before redistributing"
GENERIC = frozenset({"application/octet-stream", "binary/octet-stream", ""})


def _starts(*heads: bytes) -> Callable[[bytes], bool]:
    return lambda first: first.lstrip().startswith(heads)


KINDS: dict[str, tuple[frozenset[str], Callable[[bytes], bool]]] = {
    "pdf": (frozenset({"application/pdf", "application/x-pdf"}), lambda b: b"%PDF-" in b[:1024]),
    "zip": (frozenset({"application/zip", "application/x-zip-compressed"}),
            _starts(b"PK\x03\x04", b"PK\x05\x06")),
    "step": (frozenset({"model/step", "application/step", "application/x-step",
                        "model/step+zip"}), _starts(b"ISO-10303")),
    "kicad": (frozenset({"text/x-kicad"}),
              _starts(b"(footprint", b"(module", b"(kicad_symbol_lib")),
    "png": (frozenset({"image/png"}), _starts(b"\x89PNG")),
    "jpeg": (frozenset({"image/jpeg"}), _starts(b"\xff\xd8\xff")),
}
"""A kind of file: the content types it is served as, and whether its first bytes fit."""

TEXTUAL = frozenset({"kicad"})
"""Kinds that servers send as ``text/*``."""

ACCEPT_PDF = ("application/pdf",)
ACCEPT_CAD = ("model/step", "application/zip", "text/x-kicad")


class DownloadError(RuntimeError):
    """The server's answer is not a file the caller accepts, or is too large."""


@dataclass(frozen=True)
class Download:
    """A file on disk and the record kept beside it."""

    path: Path
    url: str
    final_url: str
    sha256: str
    content_type: str
    size: int
    fetched_at: str
    robots: str
    licence: str
    cached: bool = False

    @property
    def record(self) -> Path:
        return self.path.with_name(self.path.name + ".json")


def kind_of(content_type: str, first: bytes, accept: tuple[str, ...]) -> str:
    """The accepted kind these headers and first bytes show, or "" for none.

    A kind matches when the first bytes fit and the content type is one of the kind's, or
    is generic (``application/octet-stream``).
    """
    served = content_type.split(";")[0].strip().lower()
    for kind, (mimes, fits) in KINDS.items():
        wanted = mimes & {a.lower() for a in accept}
        if not wanted:
            continue
        plain = kind in TEXTUAL and served.startswith("text/")
        if fits(first) and (served in mimes or served in GENERIC or plain):
            return kind
    return ""


def _name(url: str, disposition: str) -> str:
    named = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', disposition or "", re.IGNORECASE)
    raw = urllib.parse.unquote(named.group(1)) if named else \
        urllib.parse.unquote(Path(urllib.parse.urlsplit(url).path).name)
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or "download"
    return clean[:80]


def _slot(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


def _cached(dest: Path, url: str) -> Download | None:
    for record in sorted(dest.glob(f"{_slot(url)}_*.json")):
        held = files.read_json(record, {})
        target = record.with_name(record.name.removesuffix(".json"))
        if held.get("url") == url and target.is_file() \
                and files.sha256_file(target) == held.get("sha256"):
            return Download(
                path=target, url=url, final_url=held.get("final_url", url),
                sha256=held["sha256"], content_type=held.get("content_type", ""),
                size=int(held.get("size", 0)), fetched_at=held.get("fetched_at", ""),
                robots=held.get("robots", ""), licence=held.get("licence", LICENCE), cached=True)
    return None


@dataclass(frozen=True)
class Wanted:
    """What a download may be: ``accept`` lists content types (``ACCEPT_PDF``, ``ACCEPT_CAD``,
    or any type in ``KINDS``), ``max_bytes`` the size limit, ``licence`` the note recorded
    beside the file; ``refresh`` fetches again even when the URL is already on disk."""

    accept: tuple[str, ...] = ACCEPT_PDF
    max_bytes: int = MOST_BYTES
    licence: str = LICENCE
    refresh: bool = False


def download(url: str, dest_dir: str | Path, wanted: Wanted | None = None,
             polite: Polite | None = None) -> Download:
    """Fetch ``url`` into ``dest_dir`` and return where it landed.

    Raises ``DownloadError`` for a wrong type, bytes that do not look like the type, or a
    body over the size limit; ``Disallowed`` when robots.txt forbids it; ``Refused`` for a
    private host at any redirect hop. A file already fetched from ``url`` is returned with
    ``cached=True``.
    """
    wanted = wanted or Wanted()
    accept, max_bytes = wanted.accept, wanted.max_bytes
    dest = Path(dest_dir).expanduser()
    if not wanted.refresh and (held := _cached(dest, url)) is not None:
        return held
    polite = polite or Polite()
    with polite.fetch(url, accept=", ".join(accept) + ", */*;q=0.1") as reply:
        served = str(reply.headers.get("Content-Type", ""))
        size = int(reply.headers.get("Content-Length") or 0)
        if size > max_bytes:
            raise DownloadError(f"{url}: {size} bytes is over the {max_bytes} limit")
        final = str(reply.geturl())
        name = _name(final, str(reply.headers.get("Content-Disposition", "")))
        target = dest / f"{_slot(url)}_{name}"
        digest, total = hashlib.sha256(), 0
        with files.writing(target) as tmp, tmp.open("wb") as out:
            first = reply.read(CHUNK)
            if not kind_of(served, first, accept):
                raise DownloadError(
                    f"{url}: {served or 'no content type'} starting {first[:12]!r} is not "
                    f"one of {', '.join(accept)}")
            block = first
            while block:
                total += len(block)
                if total > max_bytes:
                    raise DownloadError(f"{url}: over the {max_bytes} byte limit")
                digest.update(block)
                out.write(block)
                block = reply.read(CHUNK)
    got = Download(path=target, url=url, final_url=final, sha256=digest.hexdigest(),
                   content_type=served.split(";")[0].strip().lower(), size=total,
                   fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
                   robots=polite.note(final), licence=wanted.licence)
    saved = {k: v for k, v in asdict(got).items() if k not in ("path", "cached")}
    files.write_json(got.record, files.versioned(saved, RECORD_VERSION))
    return got


def as_dict(got: Download) -> dict[str, Any]:
    """A ``Download`` as plain values, the path as a string."""
    return {**asdict(got), "path": str(got.path)}
