"""``ml-stack fleet share``: serve files to the devices that may have them.

Each file has a sharing level (`sharing.py`). Programs are ``open``; a model is ``owner`` unless
said otherwise, because a model whose licence status is not known is treated as restricted.
For an ``owner`` file the first transfer asks the person at the terminal to confirm they
accepted its licence (typing the licence id back), and writes down who, when, which licence and
its URL. Without a terminal nothing is recorded and the file stays withheld.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Any

from ml_stack import home, macauth
from ml_stack.log import say, warn
from ml_stack.units import parse_duration

from ..discovery import memberships
from ..tls import TlsUnavailable, identity
from .human import HumanRequired, mint
from .manifest import SHARING_LEVELS, Entry, Signer, verify
from .requests import Devices
from .sharing import OPEN, OWNER, Licences
from .signing import KeyStoreError, SigningKeys
from .signing_cli import confirm_signing
from .transfer import Share, ShareServer, mac_gate

__all__ = ["add_share", "cmd_share"]

SUFFIX_KIND = {".whl": "wheel", ".gz": "sdist", ".zip": "sdist", ".gguf": "model",
               ".safetensors": "model"}


def add_share(sub: Any, common: Any) -> None:
    p = common(sub.add_parser("share", help="serve files, with a signed manifest, to paired "
                                            "devices of this cluster"))
    p.add_argument("--dir", required=True, help="the files to share")
    p.add_argument("--for", dest="span", default="10m")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--host", default="", help="the address to listen on (default: every "
                                              "interface)")
    p.add_argument("--sharing", action="append", default=[], metavar="NAME=LEVEL",
                   help="open, owner or never for a file (default: programs open, models owner)")
    p.add_argument("--licence", action="append", default=[], metavar="NAME=ID,URL",
                   help="the licence a model comes under")
    p.add_argument("--source", action="append", default=[], metavar="NAME=URL",
                   help="where a model comes from, for a device that must fetch it itself")


def _pairs(items: list[str]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in items if "=" in item)


def _entries(share_dir: Path, args: argparse.Namespace) -> list[Entry]:
    level, licence, source = _pairs(args.sharing), _pairs(args.licence), _pairs(args.source)
    out = []
    for f in sorted(share_dir.iterdir()):
        if not f.is_file():
            continue
        kind = SUFFIX_KIND.get(f.suffix, "other")
        chosen = level.get(f.name, OWNER if kind == "model" else OPEN)
        lid, _, url = licence.get(f.name, "").partition(",")
        out.append(Signer.entry_for(f, kind=kind, sharing=chosen, licence=lid,
                                    licence_url=url, source=source.get(f.name, "")))
    return out


def _confirm_licences(entries: list[Entry], licences: Licences) -> None:
    """Ask once, at a terminal, for each owner-level file whose licence is not on record."""
    for entry in entries:
        if entry.sharing != OWNER or not entry.licence or licences.accepted(entry):
            continue
        try:
            warn(f"{entry.name} is under {entry.licence} ({entry.licence_url or 'no URL'}).")
            licences.record(mint("accept-licence", entry.licence), entry)
        except HumanRequired as exc:
            warn(f"{entry.name} stays withheld: {exc}")


def _emit(args: argparse.Namespace, document: dict[str, Any], text: str) -> None:
    say(json.dumps(document, indent=1, default=str) if args.json else text)


def cmd_share(args: argparse.Namespace) -> int:
    directory = Path(args.state) if args.state else home.state("onboard")
    share_dir, held = Path(args.dir), memberships()
    bad = [v for v in _pairs(args.sharing).values() if v not in SHARING_LEVELS]
    if not held or not share_dir.is_dir() or bad:
        _emit(args, {"error": "needs a cluster, --dir and sharing levels of "
                              f"{'/'.join(SHARING_LEVELS)}"}, "error: needs a cluster and --dir")
        return 2
    try:
        ident = identity(directory / "tls", "share")
        keys = SigningKeys(directory)
        entries = _entries(share_dir, args)
        raw = keys.sign(entries, serial=int(time.time()), confirm=confirm_signing)
    except (TlsUnavailable, KeyStoreError) as exc:
        _emit(args, {"error": str(exc)}, f"error: {exc}")
        return 2
    licences = Licences(directory / "licences.json")
    _confirm_licences(entries, licences)
    devices = Devices(directory / "devices.json")
    gate = mac_gate(macauth.derive(held[0].key), devices.all)
    span = parse_duration(args.span) or 600.0
    with ShareServer(Share(share_dir, raw, verify(raw, keys.public), licences),
                     authenticate=gate, ident=ident,
                     address=(args.host or "0.0.0.0", args.port)) as server:  # noqa: S104
        _emit(args, {"sharing": True, "port": server.port, "certificate": ident.fingerprint,
                     "files": [{"name": e.name, "size": e.size, "sharing": e.sharing,
                                "licence_recorded": bool(licences.accepted(e))}
                               for e in entries]},
              f"sharing {len(entries)} files on port {server.port}")
        with contextlib.suppress(KeyboardInterrupt):
            time.sleep(span)
    return 0
