"""A browser context that cannot reach this machine or its network: every request a page makes
is checked, and downloads are off."""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

from poolhouse.httpguard import Limits, Refused, resolve, split

__all__ = ["INLINE", "allowed", "install"]

logger = logging.getLogger(__name__)

INLINE = frozenset({"data", "blob", "about"})
"""Schemes that carry their own content and reach nowhere."""


def allowed(url: str, limits: Limits | None = None) -> bool:
    """Whether a page may request ``url``: an inline scheme, or http(s) to a public address."""
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme in INLINE:
        return True
    try:
        _parts, host, port = split(url)
        resolve(host, port, limits or Limits())
    except Refused:
        return False
    return True


def install(context: Any, limits: Limits | None = None) -> list[str]:
    """Route every request of ``context`` through `allowed`; returns the list that refused
    URLs are appended to. Websocket and non-http schemes are refused."""
    refused: list[str] = []

    def check(route: Any) -> None:
        url = route.request.url
        if allowed(url, limits):
            route.continue_()
            return
        refused.append(url)
        logger.warning("the browser was stopped from reaching %s", urllib.parse.urlsplit(url).netloc)
        route.abort("blockedbyclient")

    context.route("**/*", check)
    return refused
