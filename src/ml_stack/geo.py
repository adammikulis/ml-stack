"""Place name -> lat/lon through Nominatim, cached on disk, one request per second.

People write where they live in prose — "Raleigh", "MD", "sf" — and a map needs a point.
Nominatim answers free text, but ranks a county above the city that shares its name and
reads a two-letter code as whichever country owns it (MD is Moldova). :func:`expand` turns
the shorthand into what was meant, :func:`best` picks the answer that is actually called
what was asked, and :func:`geocode_all` keeps the whole thing in a JSON cache so a place is
asked about once, at the one request per second Nominatim's usage policy allows.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ml_stack.files import read_json, write_json

__all__ = ["CACHE_VERSION", "SHORTHAND", "URL", "USER_AGENT", "best", "expand", "geocode_all",
           "lookup"]

URL = "https://nominatim.openstreetmap.org/search"

# Nominatim's usage policy asks that a client say who it is; a project passes its own,
# with a way to reach whoever runs it, rather than shipping under this one
USER_AGENT = "ml-stack/geo (https://github.com/adammikulis/ml-stack)"

# bump when best() changes its mind about what a query means, so cached answers are re-asked
CACHE_VERSION = 4

# people write where they live the short way, and a two-letter code means something else
# entirely to a world gazetteer: MD is Moldova, SF is Santa Fe. Keys are casefolded.
SHORTHAND: dict[str, str] = {
    "al": "Alabama", "ak": "Alaska", "az": "Arizona", "ar": "Arkansas", "ca": "California",
    "co": "Colorado", "ct": "Connecticut", "de": "Delaware", "fl": "Florida", "ga": "Georgia",
    "hi": "Hawaii", "ia": "Iowa", "id": "Idaho", "il": "Illinois", "in": "Indiana",
    "ks": "Kansas", "ky": "Kentucky", "ma": "Massachusetts", "md": "Maryland",
    "me": "Maine", "mi": "Michigan", "mn": "Minnesota", "mo": "Missouri", "ms": "Mississippi",
    "mt": "Montana", "nc": "North Carolina", "nd": "North Dakota", "ne": "Nebraska",
    "nh": "New Hampshire", "nj": "New Jersey", "nm": "New Mexico", "nv": "Nevada",
    "ny": "New York", "oh": "Ohio", "ok": "Oklahoma", "or": "Oregon", "pa": "Pennsylvania",
    "ri": "Rhode Island", "sc": "South Carolina", "sd": "South Dakota", "tn": "Tennessee",
    "tx": "Texas", "ut": "Utah", "va": "Virginia", "vt": "Vermont", "wa": "Washington",
    "wi": "Wisconsin", "wv": "West Virginia", "wy": "Wyoming",
    "uk": "United Kingdom", "usa": "United States", "us": "United States",
    "uae": "United Arab Emirates",
    # "la" is Louisiana to the post office and Los Angeles to everyone who writes it
    "sf": "San Francisco", "nyc": "New York City", "la": "Los Angeles", "dc": "Washington, D.C.",
}


def expand(place: str, shorthand: Mapping[str, str] | None = None) -> str:
    """"MD" -> "Maryland". Left alone when it is not shorthand.

    ``shorthand`` adds to, or overrides, :data:`SHORTHAND` — what one community writes that
    the world does not.
    """
    table = {**SHORTHAND, **shorthand} if shorthand else SHORTHAND
    return table.get((place or "").strip().casefold(), place)


def best(rows: list[dict[str, Any]], query: str = "") -> dict[str, Any] | None:
    """The result that is actually called what was asked for, largest first.

    Nominatim will answer "Raleigh" with Raleigh County before the city, and "Ontario" with the
    town in California before the province. A candidate whose own name equals what was typed is
    the one that was meant; among those, the lowest ``place_rank`` is the widest area, and its
    point is already the middle of it.
    """
    if not rows:
        return None
    asked = (query or "").split(",")[0].strip().casefold()
    exact = [r for r in rows if asked and (r.get("name") or "").strip().casefold() == asked]
    pool = exact or rows
    admin = [r for r in pool if r.get("category") == "boundary" and r.get("type") == "administrative"]
    if exact:
        # several places share the name; Nominatim's importance is how well known each one is,
        # which is what someone writing it without qualification meant
        return max(admin or pool,
                   key=lambda r: (float(r.get("importance") or 0), -int(r.get("place_rank") or 30)))
    return min(admin or pool,
               key=lambda r: (int(r.get("place_rank") or 30), -float(r.get("importance") or 0)))


def lookup(place: str, *, user_agent: str = USER_AGENT, url: str = URL, timeout: float = 15.0,
           shorthand: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """One place, asked of Nominatim now: ``{"lat", "lon", "display", "type"}`` or None.

    Uncached and unthrottled — :func:`geocode_all` is the one to call for a list.
    """
    asked = expand(place, shorthand)
    q = urllib.parse.urlencode({"q": asked, "format": "jsonv2", "limit": 10})
    req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        rows = json.loads(r.read().decode())
    top = best(rows, asked)
    if top is None:
        return None
    return {"lat": float(top["lat"]), "lon": float(top["lon"]),
            "display": top.get("display_name", place), "type": top.get("type", "")}


def geocode_all(places: list[str], cache_path: Path, *, user_agent: str = USER_AGENT,
                url: str = URL, sleep: float = 1.1, log: Callable[[str], None] = print,
                shorthand: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Every place in ``places``, from the cache at ``cache_path`` or Nominatim.

    The cache is a JSON mapping of place -> :func:`lookup` result (None when nothing was
    found, so it is not asked again), written after each answer so a run cut short keeps
    what it got. A cache from an older :data:`CACHE_VERSION` is discarded whole. ``sleep``
    is the pause between requests; Nominatim allows one per second.
    """
    cache: dict[str, Any] = read_json(cache_path, {})
    if cache.get("_v") != CACHE_VERSION:
        cache = {"_v": CACHE_VERSION}
    pending = [p for p in places if p and p not in cache]
    for i, place in enumerate(pending):
        try:
            cache[place] = lookup(place, user_agent=user_agent, url=url, shorthand=shorthand)
        except Exception as exc:  # noqa: BLE001
            log(f"geocode failed for {place!r}: {exc}")
            continue
        log(f"geocoded {place!r} -> {cache[place]}")
        write_json(cache_path, cache)
        if i < len(pending) - 1:
            time.sleep(sleep)
    return cache
