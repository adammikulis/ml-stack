"""Image bytes: sniff the format, move it in and out of data URLs, synthesise a probe.

Standard library only. Resizing and per-adapter constraint trimming need Pillow and live
in ``mainspring.vision`` (host tier); everything here runs on an embedded target.
"""

from __future__ import annotations

import base64
import binascii
import struct
import zlib

MIME_BY_KIND = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "webp": "image/webp",
    "heic": "image/heic",
}


class ImageError(ValueError):
    """The bytes are not an image this package recognises."""


def kind(raw: bytes) -> str | None:
    """Identify an image by its magic bytes. ``None`` if unrecognised.

    Deliberately not trusting a Content-Type header or a file extension: uploads arrive
    mislabelled routinely, and HEIC in particular is commonly announced as JPEG.
    """
    if len(raw) < 12:
        return None
    if raw[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if raw[:2] == b"BM":
        return "bmp"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    if raw[4:8] == b"ftyp" and raw[8:12] in (b"heic", b"heix", b"hevc", b"mif1", b"heim"):
        return "heic"
    return None


def mime(raw: bytes) -> str:
    """The MIME type for these bytes. Raises if unrecognised."""
    found = kind(raw)
    if found is None:
        raise ImageError("unrecognised image format (first bytes: %r)" % raw[:12])
    return MIME_BY_KIND[found]


def to_data_url(raw: bytes, *, mime_type: str | None = None) -> str:
    """``bytes -> "data:image/png;base64,..."``."""
    return f"data:{mime_type or mime(raw)};base64,{base64.b64encode(raw).decode('ascii')}"


def from_data_url(url: str) -> tuple[bytes, str]:
    """``"data:image/png;base64,..." -> (bytes, mime_type)``.

    Raises on malformed base64 rather than returning empty bytes. A silently-empty image
    reaches the model as "no image" and comes back with a confidently hallucinated
    description -- which is the failure this whole module is shaped to prevent.
    """
    if not url.startswith("data:"):
        raise ImageError("not a data URL")
    try:
        header, payload = url[5:].split(",", 1)
    except ValueError as exc:
        raise ImageError("data URL has no comma separator") from exc

    mime_type = header.split(";", 1)[0] or "application/octet-stream"
    if ";base64" not in header:
        raise ImageError("only base64 data URLs are supported")
    try:
        return base64.b64decode(payload, validate=True), mime_type
    except (binascii.Error, ValueError) as exc:
        raise ImageError(f"invalid base64 in data URL: {exc}") from exc


def _png_chunk(tag: bytes, body: bytes) -> bytes:
    return (
        struct.pack(">I", len(body))
        + tag
        + body
        + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    )


def probe_png(colors: list[tuple[int, int, int]], size: int = 256) -> bytes:
    """Build a PNG of vertical colour bands, by hand.

    This image exists to test whether a vision model can actually see. If building it
    needed Pillow, then on a machine without Pillow the self-test would not run -- and a
    gate that cannot run **fails open**, which is the one outcome it exists to prevent.
    zlib and struct are always there.
    """
    if not colors:
        raise ImageError("probe_png needs at least one colour")

    band = max(1, size // len(colors))
    rows = bytearray()
    for _ in range(size):
        rows.append(0)  # PNG per-row filter byte: 0 = None
        for x in range(size):
            r, g, b = colors[min(x // band, len(colors) - 1)]
            rows += bytes((r, g, b))

    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(bytes(rows), 6)),
            _png_chunk(b"IEND", b""),
        )
    )
