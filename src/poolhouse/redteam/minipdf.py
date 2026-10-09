"""A PDF written by hand, from plain bytes, so that fixtures need no PDF library at all.

The tests and the red-team pages need PDFs with text of a given size, font, colour and render
mode, a picture, an outline, a layer that is switched off. Writing those with MuPDF would make
the AGPL library a build-time dependency of every fixture; they are a few hundred lines of
text-based PDF instead, and being plain bytes they are deterministic.

``Doc`` holds pages; each page holds operations added by ``text``, ``image``. ``to_bytes``
writes it. Fonts are the PDF base-14 Helvetica pair, so nothing is embedded.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Doc", "Page"]


def _esc(text: str) -> bytes:
    raw = text.encode("latin-1", "replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


@dataclass
class Page:
    width: float = 430.0
    height: float = 620.0
    ops: list[bytes] = field(default_factory=list)
    images: list[tuple[int, int, bytes]] = field(default_factory=list)  # (w, h, rgb)
    image_ops: list[bytes] = field(default_factory=list)
    alphas: list[float] = field(default_factory=list)

    def text(self, x: float, y: float, text: str, *, size: float = 12.0, bold: bool = False,  # noqa: PLR0913 - the facts of one run of text
             color: tuple[float, float, float] = (0.0, 0.0, 0.0), **hide: Any) -> None:
        """Text with its baseline at ``(x, y)`` measured from the top; ``\\n`` starts a line.

        ``hide`` takes ``render`` (the text render mode, 3 is invisible), ``alpha`` (fill opacity)
        and ``layer`` (the name of an optional-content layer the text belongs to).
        """
        render, alpha, layer = hide.pop("render", 0), hide.pop("alpha", None), hide.pop("layer", None)
        if hide:
            raise TypeError(f"unexpected options {sorted(hide)}")
        font = "F2" if bold else "F1"
        state = b""
        if alpha is not None:
            self.alphas.append(alpha)
            state = f"/GS{len(self.alphas)} gs ".encode()
        lines = text.split("\n")
        body = b"".join(
            (b"T* " if i else b"") + b"(" + _esc(line) + b") Tj " for i, line in enumerate(lines))
        op = (f"q {state.decode()}{color[0]} {color[1]} {color[2]} rg BT /{font} {size} Tf "
              f"{size * 1.2} TL {render} Tr {x} {self.height - y} Td ").encode() + body + b"ET Q\n"
        if layer:
            op = f"/OC /{layer} BDC\n".encode() + op + b"EMC\n"
        self.ops.append(op)

    def image(self, x: float, y: float, width: float, height: float,  # noqa: PLR0913 - a picture's facts
              rgb: tuple[int, int, int] = (50, 100, 200), pixels: tuple[int, int] = (60, 40)) -> None:
        """A solid picture ``width`` x ``height`` points with its top-left at ``(x, y)``."""
        w, h = pixels
        self.images.append((w, h, bytes(rgb) * (w * h)))
        n = len(self.images)
        self.image_ops.append(
            f"q {width} 0 0 {height} {x} {self.height - y - height} cm /Im{n} Do Q\n".encode())


class Doc:
    def __init__(self) -> None:
        self.pages: list[Page] = []
        self.title = ""
        self.outline: list[tuple[int, str, int]] = []   # (level, title, 1-based page)
        self.layers: dict[str, bool] = {}               # name -> on

    def page(self, width: float = 430.0, height: float = 620.0) -> Page:
        self.pages.append(Page(width, height))
        return self.pages[-1]

    def to_bytes(self) -> bytes:
        objs: dict[int, bytes] = {}
        nxt = [1]

        def new(body: bytes = b"") -> int:
            n = nxt[0]
            nxt[0] += 1
            objs[n] = body
            return n

        catalog = new()
        tree = new()
        f1 = new(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        f2 = new(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        layer_ids = {name: new(f"<< /Type /OCG /Name ({name}) >>".encode()) for name in self.layers}
        page_ids: list[int] = []
        for page in self.pages:
            ims = {}
            for i, (w, h, rgb) in enumerate(page.images, 1):
                data = zlib.compress(rgb)
                ims[i] = new(f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
                             f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
                             f"/Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream")
            content = b"".join(page.ops) + b"".join(page.image_ops)
            stream = new(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
            gs = " ".join(f"/GS{i} << /ca {a} >>" for i, a in enumerate(page.alphas, 1))
            xo = " ".join(f"/Im{i} {n} 0 R" for i, n in ims.items())
            props = " ".join(f"/{name} {n} 0 R" for name, n in layer_ids.items())
            pid = new(f"<< /Type /Page /Parent {tree} 0 R /MediaBox [0 0 {page.width} {page.height}] "
                      f"/Contents {stream} 0 R /Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> "
                      f"/ExtGState << {gs} >> /XObject << {xo} >> /Properties << {props} >> >> >>"
                      .encode())
            page_ids.append(pid)
        objs[tree] = (f"<< /Type /Pages /Count {len(page_ids)} /Kids ["
                      + " ".join(f"{p} 0 R" for p in page_ids) + "] >>").encode()
        extra = ""
        if self.outline:
            top = new()
            items = [new() for _ in self.outline]
            for i, (_level, title, at) in enumerate(self.outline):
                nxt_ref = f" /Next {items[i + 1]} 0 R" if i + 1 < len(items) else ""
                prev_ref = f" /Prev {items[i - 1]} 0 R" if i else ""
                objs[items[i]] = (f"<< /Title ({title}) /Parent {top} 0 R{nxt_ref}{prev_ref} "
                                  f"/Dest [{page_ids[at - 1]} 0 R /Fit] >>").encode()
            objs[top] = f"<< /Type /Outlines /First {items[0]} 0 R /Last {items[-1]} 0 R /Count {len(items)} >>".encode()
            extra += f" /Outlines {top} 0 R"
        if self.layers:
            off = " ".join(f"{layer_ids[n]} 0 R" for n, on in self.layers.items() if not on)
            allr = " ".join(f"{i} 0 R" for i in layer_ids.values())
            extra += f" /OCProperties << /OCGs [{allr}] /D << /OFF [{off}] >> >>"
        objs[catalog] = f"<< /Type /Catalog /Pages {tree} 0 R{extra} >>".encode()
        trailer_info = ""
        if self.title:
            trailer_info = f" /Info {new(b'<< /Title (' + _esc(self.title) + b') >>')} 0 R"

        out = bytearray(b"%PDF-1.7\n")
        offsets: dict[int, int] = {}
        for n in sorted(objs):
            offsets[n] = len(out)
            out += f"{n} 0 obj\n".encode() + objs[n] + b"\nendobj\n"
        start = len(out)
        out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
        for n in sorted(objs):
            out += f"{offsets[n]:010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {len(objs) + 1} /Root {catalog} 0 R{trailer_info} >>\n"
                f"startxref\n{start}\n%%EOF\n").encode()
        return bytes(out)
