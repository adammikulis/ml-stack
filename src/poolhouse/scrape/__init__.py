"""Reading a site you are signed in to, without rewriting the same scraper every time."""

from __future__ import annotations

from poolhouse.scrape.browser import (
                                     BrowserUnavailable,
                                     Window,
                                     browser,
                                     pace,
                                     sign_in,
                                     signed_in,
                                     within_hours,
)
from poolhouse.scrape.presets import DISCORD, PRESETS, SLACK, WEBSITE, Site, preset
from poolhouse.scrape.read import Page, read_all, read_once, scroll
from poolhouse.scrape.seen import Seen, digest

__all__ = [
                                     "DISCORD",
                                     "PRESETS",
                                     "SLACK",
                                     "WEBSITE",
                                     "BrowserUnavailable",
                                     "Page",
                                     "Seen",
                                     "Site",
                                     "Window",
                                     "browser",
                                     "digest",
                                     "pace",
                                     "preset",
                                     "read_all",
                                     "read_once",
                                     "scroll",
                                     "sign_in",
                                     "signed_in",
                                     "within_hours",
]
