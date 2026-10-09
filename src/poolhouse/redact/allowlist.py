"""``hook allow``: put a phrase on the allow-list instead of editing the file by hand."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TextIO

CONTRACT = "name-shapes.json"


def allow(fixtures: str, phrases: list[str], out: TextIO, rules: object | None = None) -> int:
    """Add ``phrases`` to the allow-list at ``fixtures``, once each, under a dated heading.
    Refuses an empty list and says so. With ``rules`` (that is, ``--why`` or ``NAMES_WHY``)
    it first says, for each phrase, which rule already covers it or which rule came nearest
    -- because a fixture covers one phrase and a rule covers every phrase of that shape."""
    phrases = [p.strip() for p in phrases if p and p.strip()]
    if not phrases:
        print("allow what? e.g.: allow \"Windows Defender Firewall\" \"x1 - x0\"", file=out)
        return 2
    if rules is not None:
        for phrase in phrases:
            fired = rules.stood_down(phrase) or rules.in_context(phrase)
            if fired:
                print(f"why {phrase!r}: already stood down by {fired}; no fixture needed",
                      file=out)
                continue
            print(f"why {phrase!r}: no rule stood it down. The nearest:", file=out)
            for miss in rules.near_misses(phrase):
                print(f"           {miss}", file=out)
            print(f"           a line in {CONTRACT} covers every phrase of that shape; "
                  "a fixture covers only this one.", file=out)
    path = Path(fixtures)
    have = {ln.strip().casefold() for ln in path.read_text(encoding="utf-8").splitlines()} \
        if path.exists() else set()
    new = [p for p in phrases if p.casefold() not in have]
    if not new:
        print(f"already allowed in {fixtures}: {', '.join(phrases)}", file=out)
        return 0
    stamp = time.strftime("%Y-%m-%d")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    text += f"\n# allowed with `hook allow` on {stamp}: not people\n" + "".join(f"{p}\n" for p in new)
    path.write_text(text, encoding="utf-8")
    print(f"allowed in {fixtures}: {', '.join(new)}", file=out)
    return 0
