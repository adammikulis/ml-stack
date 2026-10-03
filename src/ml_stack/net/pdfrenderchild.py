"""The child process of :func:`ml_stack.net.pdfrender.render_pages`: PDFium (pypdfium2) under bounds.

Reads ``{"path", "pages", "dpi", "crop", "max_width", "limits"}`` as JSON on stdin and writes the
rendered pages as base64 PNG in JSON on stdout. A bound crossed is ``{"error": ...}``; anything
unforeseen ends the process with a traceback, which the parent reports as a refusal too, so a
failure never reads as a blank page. Bounds, all hard: bytes of the file, pages in the file,
pages rendered, points of a page side, pixels of a page, bytes of PNG in all, resident memory
(a watchdog thread, because macOS does not enforce an address-space limit), CPU seconds, and the
parent's wall-clock kill. It imports only the standard library, pypdfium2 and Pillow.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

SIZES = (re.compile(rb"/Width\s+(\d+)[^>]{0,300}?/Height\s+(\d+)"),
         re.compile(rb"/Height\s+(\d+)[^>]{0,300}?/Width\s+(\d+)"))

class Refusal(Exception):
    """A bound was crossed or the file cannot be rendered; the message is the reason."""


def peak_bytes() -> int:
    """The resident set high-water mark of this process, in bytes (0 where unknown)."""
    try:
        import resource
    except ImportError:
        return 0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def watchdog(limit: int) -> None:
    """End the process with a refusal the moment its resident memory passes ``limit``."""
    def watch() -> None:
        while True:
            if peak_bytes() > limit:
                os.write(1, json.dumps(
                    {"error": "rendering needs more memory than the limit allows"}).encode())
                os._exit(3)
            time.sleep(0.02)
    threading.Thread(target=watch, daemon=True).start()


def limit_resources(lim: dict[str, Any]) -> None:
    """Cap the address space and CPU time of this process where the OS enforces them."""
    try:
        import resource
    except ImportError:
        return  # Windows: the wall-clock kill and the watchdog apply
    for which, value in ((resource.RLIMIT_AS, lim["memory_bytes"] * 4),
                         (resource.RLIMIT_CPU, int(lim["timeout_s"]) + 5)):
        with contextlib.suppress(ValueError, OSError):  # not every OS enforces every limit
            resource.setrlimit(which, (int(value), int(value)))


def declared_image_pixels(data: bytes) -> int:
    """The largest width x height any dictionary written in the clear declares. A picture
    inside a compressed object stream is not seen here; the memory bound catches it."""
    return max((int(w) * int(h) for pattern in SIZES for w, h in pattern.findall(data)[:100000]),
               default=0)


def content_box(picture: Any, pad: int) -> tuple[int, int, int, int]:
    """The box around everything that is not white, grown by ``pad``; the whole picture if blank."""
    from PIL import Image, ImageChops

    flat = picture.convert("RGB")
    diff = ImageChops.difference(flat, Image.new("RGB", flat.size, (255, 255, 255)))
    box = diff.getbbox()
    if box is None:
        return (0, 0, *flat.size)
    return (max(0, box[0] - pad), max(0, box[1] - pad),
            min(flat.width, box[2] + pad), min(flat.height, box[3] + pad))


def render(request: dict[str, Any]) -> dict[str, Any]:
    import pypdfium2 as pdfium
    from PIL import Image

    lim = request["limits"]
    data = Path(request["path"]).read_bytes()
    if len(data) > lim["max_bytes"]:
        raise Refusal(f"the file is {len(data)} bytes, over {lim['max_bytes']}")
    if declared_image_pixels(data) > lim["max_image_pixels"]:
        raise Refusal(f"a picture in the file declares more than {lim['max_image_pixels']} pixels")
    wanted = [int(p) for p in request["pages"]]
    if not wanted or len(wanted) > lim["max_render_pages"]:
        raise Refusal(f"{len(wanted)} pages asked for, the bound is {lim['max_render_pages']}")
    Image.MAX_IMAGE_PIXELS = int(lim["max_pixels"]) * 2
    try:
        doc = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        text = str(exc).lower()
        raise Refusal("the PDF is encrypted and needs a password" if "password" in text
                      else "not a readable PDF") from exc
    try:
        if len(doc) > lim["max_pages"]:
            raise Refusal(f"the file has {len(doc)} pages, over {lim['max_pages']}")
        out: list[dict[str, Any]] = []
        total = 0
        for number in wanted:
            if not 1 <= number <= len(doc):
                raise Refusal(f"page {number} is not in a file of {len(doc)} pages")
            page = doc[number - 1]
            width, height = page.get_size()
            if not (math.isfinite(width) and math.isfinite(height) and width >= 1 and height >= 1):
                raise Refusal(f"page {number} has no usable size")
            if max(width, height) > lim["max_page_pt"]:
                raise Refusal(f"page {number} is {width:.0f} x {height:.0f} points, "
                              f"over {lim['max_page_pt']:g}")
            scale = min(request["dpi"] / 72.0, math.sqrt(lim["max_pixels"] / (width * height)))
            if not request["crop"]:
                scale = min(scale, request["max_width"] / width)
            picture = page.render(scale=scale, may_draw_forms=False).to_pil().convert("RGB")
            if request["crop"]:
                picture = picture.crop(content_box(picture, round(request["pad_pt"] * scale)))
            if picture.width > request["max_width"]:
                picture = picture.resize((request["max_width"], max(
                    1, round(picture.height * request["max_width"] / picture.width))))
            buf = io.BytesIO()
            picture.save(buf, "PNG")
            total += buf.tell()
            if total > lim["max_out"]:
                raise Refusal(f"the pictures are over {lim['max_out']} bytes")
            out.append({"page": number, "w": picture.width, "h": picture.height,
                        "blank": picture.convert("L").getextrema()[0] >= 250,
                        "png": base64.b64encode(buf.getvalue()).decode()})
        return {"pages": out}
    finally:
        doc.close()


def serve() -> None:
    """Read the request from stdin and answer on stdout."""
    request = json.loads(sys.stdin.buffer.read())
    limit_resources(request["limits"])
    watchdog(int(request["limits"]["memory_bytes"]))
    try:
        answer = render(request)
    except Refusal as exc:
        answer = {"error": str(exc)}
    except MemoryError:
        answer = {"error": "rendering needs more memory than the limit allows"}
    except RecursionError:
        answer = {"error": "the PDF nests deeper than the renderer allows"}
    sys.stdout.write(json.dumps(answer))


if __name__ == "__main__":
    serve()
