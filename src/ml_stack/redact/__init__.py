"""Replace known names in text with stable placeholders."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

__all__ = ["Redactor", "tag"]


def tag(name: str, prefix: str = "person") -> str:
    """A stable stand-in for one name. The same name always gives the same tag."""
    return f"{prefix}#" + hashlib.sha256(name.strip().casefold().encode()).hexdigest()[:4]


class Redactor:
    """Swaps a given set of names out of any text, longest name first, case-insensitively."""

    def __init__(self, names: Iterable[str], *, prefix: str = "person") -> None:
        self.prefix = prefix
        ordered = sorted({n for n in names if n and n.strip()}, key=len, reverse=True)
        self.pattern = re.compile(
            "|".join(rf"(?<![\w.]){re.escape(n)}(?![\w])" for n in ordered),
            re.I) if ordered else None

    def __call__(self, text: Any) -> str:
        body = str(text)
        if self.pattern is None:
            return body
        return self.pattern.sub(lambda m: tag(m.group(0), self.prefix), body)
