"""``ml-stack fleet peers``: the paired devices a model download asks before the internet.

``peers`` lists them (with the rate last measured for each), ``add`` stores one by hand (its share
address, the certificate its TLS is pinned to, the owner's manifest key and the request key from
pairing; pairing itself fills the book, so this is only for a device paired some other way),
``remove`` drops one, ``off`` / ``on`` switch peer-first downloads for this machine, and
``limit [NAME] [--rate 20MiB/s] [--streams N] [--metered on|off]`` caps what one device (or every
device) is asked for: bytes a second, requests at once, or not asked at all because its link
costs per byte. One pull skips them with ``--no-peers``,
a shell with ``ML_STACK_NO_PEERS=1``. See docs/model-discovery.md.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any
from urllib.parse import urlsplit

from ml_stack.hub.peerbook import PeerBook
from ml_stack.log import say, warn
from ml_stack.units import human_bytes

from .lan import NotLocal, require_local_url

__all__ = ["add_peers", "cmd_peers"]


def add_peers(sub: Any, common: Any) -> None:
    p = common(sub.add_parser("peers", help="the paired devices model downloads ask first"))
    p.add_argument("action", nargs="?", default="list",
                   choices=("list", "add", "remove", "on", "off", "limit"))
    p.add_argument("name", nargs="?", default="")
    p.add_argument("--url", default="", help="https://host:port of the device's 'share' (a "
                                             "LAN, VPN or overlay address, never a public one)")
    p.add_argument("--certificate", default="", help="the device's certificate (its beacon)")
    p.add_argument("--signing-key", default="", help="the owner's manifest key, base64")
    p.add_argument("--device-secret", default="", help="this device's request key from pairing")
    p.add_argument("--rate", default="", help="limit: the most bytes a second to ask of the device "
                                              "(20MiB/s, 512KiB/s; 'off' removes the limit)")
    p.add_argument("--streams", type=int, default=None, help="limit: requests in flight at once to "
                                                          "the device (0 removes the limit)")
    p.add_argument("--metered", choices=("on", "off"), default="", help="limit: the link to the "
                   "device costs per byte, so it is not asked (the Hub is)")


UNITS = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
RATE = re.compile(r"(\d+(?:\.\d+)?)\s*([kmg]?)(?:i?b)?(?:/s|ps)?", re.IGNORECASE)


def parse_rate(text: str) -> float | None:
    """Bytes a second in ``text`` (``20MiB/s``, ``512k``, ``1.5 MB/s``; k, m and g are 1024-based),
    0.0 for ``off`` / ``0``; None when it is not a rate."""
    if text.strip().lower() in ("off", "none", "0"):
        return 0.0
    hit = RATE.fullmatch(text.strip())
    return float(hit[1]) * UNITS[hit[2].lower()] if hit else None


def _limit(args: argparse.Namespace, book: PeerBook) -> int:
    changes: dict[str, Any] = {}
    if args.rate:
        rate = parse_rate(args.rate)
        if rate is None:
            warn(f"error: {args.rate!r} is not a rate (try 20MiB/s)")
            return 2
        changes["limit_bps"] = int(rate) or None
    if args.streams is not None:
        changes["streams"] = args.streams or None
    if args.metered:
        changes["metered"] = True if args.metered == "on" else None
    if not changes:
        warn("error: limit needs --rate, --streams or --metered")
        return 2
    if args.name and args.name not in {r["name"] for r in book.rows()}:
        warn(f"error: no such peer: {args.name}")
        return 1
    changed = book.configure(args.name or None, **changes)
    _emit(args, {"changed": changed, "settings": changes},
          f"limits set on {', '.join(changed) or 'nobody'}")
    return 0


def _emit(args: argparse.Namespace, document: dict[str, Any], text: str) -> None:
    say(json.dumps(document, indent=1) if args.json else text)


def cmd_peers(args: argparse.Namespace) -> int:
    book = PeerBook(f"{args.state}/peers.json" if args.state else None)
    if args.action in ("on", "off"):
        book.set_enabled(args.action == "on")
        _emit(args, {"enabled": book.enabled}, f"peer-first downloads are {args.action}")
        return 0
    if args.action == "limit":
        return _limit(args, book)
    if args.action == "remove":
        gone = book.remove(args.name)
        _emit(args, {"removed": gone}, "removed" if gone else "no such peer")
        return 0 if gone else 1
    if args.action == "add":
        scheme = urlsplit(args.url).scheme
        if not (args.name and args.url and args.signing_key) or scheme not in ("https", "http"):
            warn("error: add NAME --url https://host:port --certificate ... --signing-key ...")
            return 2
        try:
            require_local_url(args.url)
        except NotLocal as exc:
            warn(f"error: {exc}")
            return 2
        book.add({"name": args.name, "url": args.url, "certificate": args.certificate,
                  "signing_key": args.signing_key, "device_secret": args.device_secret,
                  "min_serial": 0})
    rows = [{k: v for k, v in r.items() if k not in ("device_secret", "certificate")}
            for r in book.rows()]

    def line(r: dict[str, Any]) -> str:
        more = [f"{human_bytes(r['rate'])}/s measured"] if r.get("rate") else []
        more += [f"limit {human_bytes(r['limit_bps'])}/s"] if r.get("limit_bps") else []
        more += [f"{r['streams']} streams"] if r.get("streams") else []
        more += ["metered, not asked"] if r.get("metered") else []
        return f"{r['name']} ({r['url']}" + ("; " + ", ".join(more) if more else "") + ")"
    _emit(args, {"enabled": book.enabled, "peers": rows},
          f"peer-first downloads {'on' if book.enabled else 'off'}; "
          + (", ".join(line(r) for r in rows) or "no peers"))
    return 0
