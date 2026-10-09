"""The child process of :func:`poolhouse.net.pdfread.load`: pdfminer.six under bounds.

Reads ``{"path", "limits", ...}`` as JSON on stdin and writes the page records, outline and
metadata as JSON on stdout. A bound crossed is ``{"error": ...}``; anything this module did not
foresee ends the process with a traceback, which the parent reports as a refusal too, so an
unexpected failure never reads as an empty document.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import re
import shutil
import struct
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfminer.pdftypes as pdftypes
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTContainer,
    LTCurve,
    LTImage,
    LTTextContainer,
    LTTextLine,
)
from pdfminer.pdfdocument import PDFDocument, PDFEncryptionError, PDFPasswordIncorrect
from pdfminer.pdfexceptions import PDFException
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import PDFObjRef, resolve1
from pdfminer.psexceptions import PSException
from pdfminer.psparser import PSLiteral
from pdfminer.utils import decode_text

_FONT_PREFIX = re.compile(r"^[A-Z]{6}\+")
MIN_POINTS = 2.0
HIDDEN_RENDER = (3, 7)
MALFORMED = (ArithmeticError, AssertionError, AttributeError, LookupError, PDFException,
             PSException, RuntimeError, TypeError, ValueError, struct.error, zlib.error)
"""What pdfminer raises on a file that is not well formed. Anything else is a bug, and ends the process."""


class Refusal(Exception):
    """A bound was crossed or the file cannot be read; the message is the reason."""


def _bound_zlib(max_stream: int) -> Any:
    """A ``zlib`` for pdfminer's stream decoding that stops at ``max_stream`` inflated bytes."""

    class Out:
        def __init__(self) -> None:
            self._d = zlib.decompressobj()
            self._total = 0

        def decompress(self, data: bytes) -> bytes:
            out = self._d.decompress(data, max_stream + 1 - self._total)
            self._total += len(out)
            if self._total > max_stream or self._d.unconsumed_tail:
                raise Refusal(f"a stream inflates to more than {max_stream} bytes")
            return out

    class Bounded:
        error = zlib.error

        @staticmethod
        def decompress(data: bytes) -> bytes:
            return Out().decompress(data)

        @staticmethod
        def decompressobj() -> Out:
            return Out()

    return Bounded


def white(color: Any) -> bool:
    """Whether a fill colour (gray, RGB or CMYK components) is white."""
    values = (float(color),) if isinstance(color, (int, float)) else tuple(color or ())
    if not all(isinstance(v, (int, float)) for v in values):
        return False
    return ((len(values) in (1, 3) and all(v >= 0.97 for v in values))
            or (len(values) == 4 and all(v <= 0.03 for v in values)))


def layers_off(catalog: dict[str, Any]) -> set[int]:
    """Object ids of the optional-content layers the file switches off by default."""
    props = resolve1(catalog.get("OCProperties")) or {}
    config = resolve1(props.get("D")) or {}
    off = {r.objid for r in resolve1(config.get("OFF")) or () if isinstance(r, PDFObjRef)}
    if getattr(resolve1(config.get("BaseState")), "name", "ON") == "OFF":
        on = {r.objid for r in resolve1(config.get("ON")) or () if isinstance(r, PDFObjRef)}
        off |= {r.objid for r in resolve1(props.get("OCGs")) or ()
                if isinstance(r, PDFObjRef) and r.objid not in on}
    return off


class Interpreter(PDFPageInterpreter):
    """pdfminer's interpreter that also tracks transparency and switched-off layers."""

    off: set[int] = frozenset()  # type: ignore[assignment]

    def reset(self) -> None:
        self.alpha = 1.0
        self.alphas: list[float] = []
        self.marked: list[bool] = []

    def hidden_layer(self, ref: Any) -> bool:
        if isinstance(ref, PDFObjRef) and ref.objid in self.off:
            return True
        node = resolve1(ref)
        if isinstance(node, dict) and getattr(resolve1(node.get("Type")), "name", "") == "OCMD":
            members = resolve1(node.get("OCGs"))
            members = members if isinstance(members, list) else [node.get("OCGs")]
            refs = [m for m in members if isinstance(m, PDFObjRef)]
            return bool(refs) and all(m.objid in self.off for m in refs)
        return False

    def do_q(self) -> None:
        self.alphas.append(self.alpha)
        super().do_q()

    def do_Q(self) -> None:
        if self.alphas:
            self.alpha = self.alphas.pop()
        super().do_Q()

    def do_gs(self, name: Any) -> None:
        states = resolve1(self.resources.get("ExtGState")) or {}
        state = resolve1(states.get(getattr(name, "name", name))) or {}
        if "ca" in state:
            self.alpha = float(resolve1(state["ca"]))
        super().do_gs(name)

    def do_BMC(self, tag: Any) -> None:
        self.marked.append(False)
        super().do_BMC(tag)

    def do_BDC(self, tag: Any, props: Any) -> None:
        """A layer reference that cannot be read counts as switched off: fail closed."""
        hides = False
        if getattr(tag, "name", "") == "OC":
            try:
                table = resolve1(self.resources.get("Properties")) or {}
                hides = self.hidden_layer(table.get(props.name) if isinstance(props, PSLiteral) else props)
            except MALFORMED:
                hides = True
        self.marked.append(hides)
        super().do_BDC(tag, props)

    def do_EMC(self) -> None:
        if self.marked:
            self.marked.pop()
        super().do_EMC()


class Device(PDFPageAggregator):
    """Collects characters, marks the hidden ones, and counts them against the page bound."""

    itp: Interpreter
    max_page_chars = 8000
    seen = 0

    def begin_page(self, page: Any, ctm: Any) -> None:
        self.seen = 0
        super().begin_page(page, ctm)

    def render_char(self, matrix: Any, font: Any, fontsize: Any, scaling: Any,  # noqa: PLR0913 - pdfminer's signature
                    rise: Any, cid: Any, ncs: Any, graphicstate: Any) -> float:
        self.seen += 1
        if self.seen > self.max_page_chars:
            raise Refusal(f"a page holds more than {self.max_page_chars} characters")
        advance = super().render_char(matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate)
        self.cur_item._objs[-1].pcbe_hidden = (
            getattr(self.itp.textstate, "render", 0) in HIDDEN_RENDER
            or white(getattr(graphicstate, "ncolor", None))
            or self.itp.alpha <= 0.02 or any(self.itp.marked))
        return advance


@dataclass(frozen=True)
class Box:
    """A page's origin and size, to turn pdfminer's bottom-up coordinates into top-down ones."""

    x0: float
    y0: float
    width: float
    height: float

    def rect(self, item: Any) -> list[float]:
        return [item.x0 - self.x0, self.height - (item.y1 - self.y0),
                item.x1 - self.x0, self.height - (item.y0 - self.y0)]

    def outside(self, item: Any) -> bool:
        return (item.x1 < self.x0 or item.x0 > self.x0 + self.width
                or item.y1 < self.y0 or item.y0 > self.y0 + self.height)


class Reader:
    """One read of one file."""

    def __init__(self, request: dict[str, Any]) -> None:
        self.request = request
        self.lim = request["limits"]
        self.chars = 0
        self.images = 0
        self.images_dir = tempfile.mkdtemp(prefix="pdf-images-")
        self.deadline = time.monotonic() + float(self.lim["timeout_s"]) - 1.0

    # -- the file -------------------------------------------------------------------------

    def open(self) -> PDFDocument:
        try:
            doc = PDFDocument(PDFParser(io.BytesIO(Path(self.request["path"]).read_bytes())))
        except (PDFPasswordIncorrect, PDFEncryptionError) as exc:
            raise Refusal("the PDF is encrypted and needs a password") from exc
        except MALFORMED as exc:
            raise Refusal(f"not a readable PDF: {type(exc).__name__}") from exc
        node = resolve1(doc.catalog.get("Pages"))
        claimed = resolve1(node.get("Count")) if isinstance(node, dict) else 0
        if isinstance(claimed, int) and claimed > self.lim["max_pages"]:
            raise Refusal(f"the page tree claims {claimed} pages, over {self.lim['max_pages']}")
        return doc

    def run(self) -> dict[str, Any]:
        try:
            pdftypes.zlib = _bound_zlib(int(self.lim["max_stream"]))  # type: ignore[attr-defined]
            doc = self.open()
            pages, ids = self.pages(doc)
            return {"pages": pages, "page_count": len(pages), "metadata": self.info(doc),
                    "toc": self.outline(doc, ids)}
        except Refusal as exc:
            return {"error": str(exc)}
        except (PDFPasswordIncorrect, PDFEncryptionError):
            return {"error": "the PDF is encrypted and needs a password"}
        except MemoryError:
            return {"error": "reading needs more memory than the limit allows"}
        except RecursionError:
            return {"error": "the PDF nests deeper than the reader allows"}
        except MALFORMED as exc:
            return {"error": f"not a readable PDF: {type(exc).__name__}"}
        finally:
            shutil.rmtree(self.images_dir, ignore_errors=True)

    # -- pages ----------------------------------------------------------------------------

    def pages(self, doc: PDFDocument) -> tuple[list[dict[str, Any]], dict[int, int]]:
        manager = PDFResourceManager()
        device = Device(manager, laparams=LAParams(all_texts=True))
        device.max_page_chars = int(self.lim["max_page_chars"])
        itp = Interpreter(manager, device)
        itp.off = layers_off(doc.catalog)
        device.itp = itp
        out: list[dict[str, Any]] = []
        ids: dict[int, int] = {}
        limit = self.request.get("text_limit")
        for number, page in enumerate(PDFPage.create_pages(doc)):
            if number >= self.lim["max_pages"]:
                raise Refusal(f"the page tree has more than {self.lim['max_pages']} pages")
            if time.monotonic() > self.deadline:
                raise Refusal(f"reading took more than {self.lim['timeout_s']:g} s")
            ids[page.pageid] = number
            x0, y0, x1, y1 = (float(v) for v in page.mediabox)
            box = Box(x0, y0, abs(x1 - x0), abs(y1 - y0))
            record: dict[str, Any] = {"width": box.width, "height": box.height, "blocks": [], "images": [],
                                      "strokes": 0}
            out.append(record)
            if limit is None or self.chars < limit:  # past the limit pages are counted, not laid out
                itp.reset()
                itp.process_page(page)
                self.collect(device.get_result(), record, box)
        return out, ids

    def collect(self, node: Any, record: dict[str, Any], box: Box) -> None:
        for item in node:
            if isinstance(item, LTTextContainer):
                lines = [self.line(line, box) for line in item if isinstance(line, LTTextLine)]
                lines = [line for line in lines if line["spans"]]
                if lines:
                    record["blocks"].append({"type": 0, "bbox": box.rect(item), "lines": lines})
            elif isinstance(item, LTImage):
                self.images += 1
                if self.images <= self.lim["max_images"]:
                    png, w, h = self.png(item) if self.request["images"] else ("", 0, 0)
                    record["images"].append({"bbox": box.rect(item), "png": png, "w": w, "h": h})
            elif isinstance(item, LTCurve):  # a stroked or filled path: a line, a rectangle, a curve
                record["strokes"] += 1
            elif isinstance(item, LTContainer):
                self.collect(item, record, box)

    def line(self, line: Any, box: Box) -> dict[str, Any]:
        spans: list[dict[str, Any]] = []
        last: tuple[str, float, bool] | None = None
        for ch in line:
            if isinstance(ch, LTAnno) and spans:
                spans[-1]["text"] += ch.get_text()
            if not isinstance(ch, LTChar):
                continue
            size = round(float(ch.size), 2)
            hidden = bool(getattr(ch, "pcbe_hidden", False)) or size < MIN_POINTS or box.outside(ch)
            font = _FONT_PREFIX.sub("", str(ch.fontname))
            if (font, size, hidden) != last or not spans:
                spans.append({"text": "", "size": size, "font": font, "hidden": hidden})
                last = (font, size, hidden)
            spans[-1]["text"] += ch.get_text()
        if spans:
            spans[-1]["text"] = spans[-1]["text"].rstrip("\n")
        text = "".join(s["text"] for s in spans)
        self.chars += len(text)
        if self.chars > int(self.lim["max_chars"]):
            raise Refusal(f"more than {self.lim['max_chars']} characters of text")
        return {"bbox": box.rect(line), "spans": spans if text.strip() else []}

    def png(self, item: Any) -> tuple[str, int, int]:
        """The picture as base64 PNG no wider than the request says; empty when it will not render."""
        from pdfminer.image import ImageWriter
        from PIL import Image

        width = int(self.request["max_width"])
        try:
            full = Path(self.images_dir) / ImageWriter(self.images_dir).export_image(item)
            with Image.open(full) as opened:
                picture = opened.convert("RGB")
            full.unlink()
            if picture.width > width:
                picture = picture.resize((width, max(1, round(picture.height * width / picture.width))))
            buf = io.BytesIO()
            picture.save(buf, "PNG")
            return base64.b64encode(buf.getvalue()).decode(), picture.width, picture.height
        except (OSError, Image.DecompressionBombError, *MALFORMED):
            return "", 0, 0

    # -- metadata and outline -------------------------------------------------------------

    @staticmethod
    def info(doc: PDFDocument) -> dict[str, str]:
        out: dict[str, str] = {}
        for entry in doc.info or ():
            for key, value in entry.items():
                value = resolve1(value)
                if isinstance(value, bytes):
                    out[str(key)] = decode_text(value)
                elif isinstance(value, str):
                    out[str(key)] = value
        return out

    @staticmethod
    def outline(doc: PDFDocument, ids: dict[int, int]) -> list[list[Any]]:
        """``[level, title, 1-based page]`` per outline entry; none when the file has no outline."""
        out: list[list[Any]] = []
        try:
            entries = list(doc.get_outlines())[:20000]
        except MALFORMED:
            return out
        for level, title, dest, action, _ in entries:
            target = dest
            if target is None and isinstance(action, dict):
                target = resolve1(action.get("D"))
            with contextlib.suppress(*MALFORMED):
                if isinstance(target, (bytes, str, PSLiteral)):
                    target = doc.get_dest(target.name if isinstance(target, PSLiteral) else target)
            target = resolve1(target)
            target = resolve1(target.get("D")) if isinstance(target, dict) else target
            first = target[0] if isinstance(target, list) and target else None
            at = ids.get(first.objid, -1) + 1 if isinstance(first, PDFObjRef) else 0
            out.append([int(level), str(title or ""), at])
        return out


def limit_resources(lim: dict[str, Any]) -> None:
    """Cap the address space and CPU time of this process where the OS enforces them."""
    try:
        import resource
    except ImportError:
        return  # Windows: the wall-clock timeout and the stream bound apply
    for which, value in ((resource.RLIMIT_AS, lim["memory_bytes"]),
                         (resource.RLIMIT_CPU, int(lim["timeout_s"]) + 5)):
        with contextlib.suppress(ValueError, OSError):  # not every OS enforces every limit
            resource.setrlimit(which, (int(value), int(value)))


def serve() -> None:
    """Read the request from stdin and answer on stdout."""
    request = json.loads(sys.stdin.buffer.read())
    limit_resources(request["limits"])
    sys.stdout.write(json.dumps(Reader(request).run()))


if __name__ == "__main__":
    serve()
