"""Normal working hours: inside them long test suites are capped to half the broker budget."""
from __future__ import annotations

import datetime as dt
import re

DEFAULT_NORMAL_HOURS = "08:00-21:00"
WEEKDAYS = 5
WINDOW = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$")


def parse_window(setting: str | None) -> tuple[dt.time, dt.time] | None:
    """The (start, end) of DEV_TEST_NORMAL_HOURS ("HH:MM-HH:MM", weekdays), None for "off"; ValueError when malformed."""
    text = (setting if setting else DEFAULT_NORMAL_HOURS).strip().lower()
    if text == "off":
        return None
    match = WINDOW.match(text)
    if not match:
        raise ValueError(f"DEV_TEST_NORMAL_HOURS must be HH:MM-HH:MM or off, not {setting!r}")
    a, b, c, d = map(int, match.groups())
    try:
        start, end = dt.time(a, b), dt.time(c, d)
    except ValueError as error:
        raise ValueError(f"DEV_TEST_NORMAL_HOURS has an impossible time: {setting!r}") from error
    if start >= end:
        raise ValueError("DEV_TEST_NORMAL_HOURS must start before it ends")
    return start, end


def cap_applies(now: dt.datetime, window: tuple[dt.time, dt.time] | None) -> bool:
    """Whether the long-run cap holds at ``now``: inside normal hours on a weekday, or always when the setting is off."""
    if window is None:
        return True
    return now.weekday() < WEEKDAYS and window[0] <= now.time() < window[1]


def capped_now(environment: dict, now: dt.datetime) -> bool:
    """cap_applies for the environment's setting; a malformed setting keeps the cap on."""
    try:
        return cap_applies(now, parse_window(environment.get("DEV_TEST_NORMAL_HOURS")))
    except ValueError:
        return True
