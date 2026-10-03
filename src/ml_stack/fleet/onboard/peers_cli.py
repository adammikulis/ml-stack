"""``ml-stack fleet peers``: the paired devices a model download asks before the internet.

``peers`` lists them, ``add`` stores one (its share address, the certificate its TLS is pinned
to, the owner's manifest key and the request key from pairing), ``remove`` drops one, ``off`` /
``on`` switch peer-first downloads for this machine. One pull skips them with ``--no-peers``,
a shell with ``ML_STACK_NO_PEERS=1``. See docs/model-discovery.md.
"""

from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.parse import urlsplit

from ml_stack.log import say, warn

from .lan import NotLocal, require_local_url
from .peerfirst import PeerBook

__all__ = ["add_peers", "cmd_peers"]


def add_peers(sub: Any, common: Any) -> None:
    p = common(sub.add_parser("peers", help="the paired devices model downloads ask first"))
    p.add_argument("action", nargs="?", default="list",
                   choices=("list", "add", "remove", "on", "off"))
    p.add_argument("name", nargs="?", default="")
    p.add_argument("--url", default="", help="https://host:port of the device's 'share' (a "
                                             "LAN, VPN or overlay address, never a public one)")
    p.add_argument("--certificate", default="", help="the device's certificate (its beacon)")
    p.add_argument("--signing-key", default="", help="the owner's manifest key, base64")
    p.add_argument("--device-secret", default="", help="this device's request key from pairing")


def _emit(args: argparse.Namespace, document: dict[str, Any], text: str) -> None:
    say(json.dumps(document, indent=1) if args.json else text)


def cmd_peers(args: argparse.Namespace) -> int:
    book = PeerBook(f"{args.state}/peers.json" if args.state else None)
    if args.action in ("on", "off"):
        book.set_enabled(args.action == "on")
        _emit(args, {"enabled": book.enabled}, f"peer-first downloads are {args.action}")
        return 0
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
    _emit(args, {"enabled": book.enabled, "peers": rows},
          f"peer-first downloads {'on' if book.enabled else 'off'}; "
          + (", ".join(f"{r['name']} ({r['url']})" for r in rows) or "no peers"))
    return 0
