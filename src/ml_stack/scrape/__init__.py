"""Reading a site you are signed in to, without rewriting the same scraper every time."""

from __future__ import annotations

from ml_stack.scrape.browser import (BrowserUnavailable, Window, browser, pace, sign_in,
                                     signed_in, within_hours)
from ml_stack.scrape.presets import DISCORD, PRESETS, SLACK, WEBSITE, Site, preset
from ml_stack.scrape.read import Page, read_all, read_once, scroll
from ml_stack.scrape.seen import Seen

__all__ = ["BrowserUnavailable", "DISCORD", "PRESETS", "Page", "SLACK", "Seen", "Site",
           "WEBSITE", "Window", "browser", "pace", "preset", "read_all", "read_once", "scroll",
           "sign_in", "signed_in", "within_hours"]
