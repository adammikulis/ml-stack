"""Showing quarantined text without letting it do anything: escaped, no markup, no scripts."""

from __future__ import annotations

import html
import unicodedata

__all__ = ["render_html", "render_text"]

_SHOWN = {"\n", "\t"}
_CSP = "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"


def render_text(text: str, *, limit: int = 20_000) -> str:
    """``text`` with every control, format and private-use character written as ``<U+XXXX>``,
    cut to ``limit`` characters."""
    out = []
    for char in text[:limit]:
        category = unicodedata.category(char)
        if char in _SHOWN or not (category[0] == "C" or category in ("Zl", "Zp")):
            out.append(char)
        else:
            out.append(f"<U+{ord(char):04X}>")
    if len(text) > limit:
        out.append(f"\n[... {len(text) - limit} more characters not shown]")
    return "".join(out)


def render_html(title: str, text: str, *, limit: int = 20_000) -> str:
    """A static page showing ``text`` in a preformatted block. It carries a policy that
    allows no script, image, frame, form or network request."""
    body = html.escape(render_text(text, limit=limit), quote=True)
    return ("<!doctype html><meta charset=utf-8>"
            f"<meta http-equiv=\"Content-Security-Policy\" content=\"{_CSP}\">"
            f"<title>{html.escape(title)}</title>"
            "<style>pre{white-space:pre-wrap;word-break:break-word;font:14px monospace}"
            ".note{font:13px sans-serif;color:#555}</style>"
            "<p class=note>Held by sentinel. This is inert text: nothing in it runs or loads.</p>"
            f"<pre>{body}</pre>")
