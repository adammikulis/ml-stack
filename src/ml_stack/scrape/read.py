"""Reading rows out of a page, including the ones the page has thrown away.

A modern list is virtualised: a row scrolled far enough out of view is removed from the
document, not merely hidden. So scrolling to the top and reading once returns the oldest
screenful and nothing else — which looks exactly like a short conversation, and is the bug
this module exists to not have. Rows are collected after every step instead, and the walk
stops when several steps in a row turn up nothing new.

The page is anything with ``evaluate(js, arg)``; Playwright's is one, and so is a fake.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from ml_stack.scrape.presets import Site

QUIET_ROUNDS = 3
MAX_ROUNDS = 400


class Page(Protocol):
    def evaluate(self, expression: str, arg: Any = None) -> Any: ...


def _reader_js(site: Site) -> str:
    """The script that reads one screenful, built from what the site looks like."""
    return """
    (spec) => {
      const clean = (s) => String(s ?? '').replace(/\\s+/g, ' ').trim();
      const pick = (el, sel) => (sel ? el.querySelector(sel) : null);
      const rows = Array.from(document.querySelectorAll(spec.rows));
      if (rows.length) {
        return rows.map((el) => {
          let key = '';
          for (const attr of spec.keyAttrs) { key = key || el.getAttribute(attr) || ''; }
          const out = {
            key,
            author: clean((pick(el, spec.author) || {}).innerText),
            text: clean((pick(el, spec.body) || el).innerText).slice(0, spec.maxChars),
          };
          for (const [name, sel] of Object.entries(spec.extra || {})) {
            const found = pick(el, sel);
            if (found) out[name] = clean(found.innerText);
          }
          return out;
        });
      }
      // nothing matched: hand back the pane whole, so a person can see what moved
      const pane = document.querySelector(spec.fallbackPane);
      const text = clean(pane ? pane.innerText : '');
      return text ? [{ key: '', author: '', text: text.slice(0, spec.maxChars * 2),
                       degraded: true }] : [];
    }
    """


def _spec(site: Site) -> dict[str, Any]:
    return {"rows": site.rows, "author": site.author, "body": site.body,
            "keyAttrs": list(site.key_attrs), "maxChars": site.max_chars,
            "fallbackPane": site.fallback_pane, "extra": dict(site.extra)}


def read_once(page: Page, site: Site) -> list[dict[str, Any]]:
    """What is in the document right now."""
    rows = page.evaluate(_reader_js(site), _spec(site)) or []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append({**row, "key": site.key_of(str(row.get("key") or ""))})
    return out


def _scroll_js(site: Site) -> str:
    direction = "-" if site.scroll_up else ""
    return f"""
    (sel) => {{
      const pane = document.querySelector(sel);
      if (!pane) return false;
      const before = pane.scrollTop;
      pane.scrollTop = Math.max(0, pane.scrollTop {direction}
                                   {'-' if site.scroll_up else '+'} pane.clientHeight * 0.8);
      return pane.scrollTop !== before;
    }}
    """


def scroll(page: Page, site: Site) -> bool:
    """One screenful further back. False when the pane would not move."""
    return bool(page.evaluate(_scroll_js(site), site.pane))


def read_all(page: Page, site: Site, *, rounds: int = MAX_ROUNDS,
             quiet_rounds: int = QUIET_ROUNDS, wait: Any = None) -> list[dict[str, Any]]:
    """Everything in the list, gathered a screenful at a time.

    Rows are keyed, so a row seen twice is kept once, and the first sighting wins — it was
    read when it was fully rendered rather than half scrolled away.
    """
    seen: dict[str, dict[str, Any]] = {}
    quiet = 0
    for _ in range(max(1, rounds)):
        before = len(seen)
        for row in read_once(page, site):
            key = row.get("key") or ""
            if key and key not in seen:
                seen[key] = row
            elif not key and not seen:
                seen[f"#{len(seen)}"] = row       # a page with one unkeyed row is still a page
        quiet = quiet + 1 if len(seen) == before else 0
        if quiet >= quiet_rounds:
            break
        if not scroll(page, site):
            break
        if wait is not None:
            wait()
    return sorted(seen.values(), key=lambda r: str(r.get("key") or ""))
