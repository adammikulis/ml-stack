"""``ml-stack fleet nearby | pair | listen | requests | accept | decline | revoke | bootstrap``.

Every command takes ``--json`` and prints one JSON document; nothing is scraped from text.
State lives under ``--state`` (default ``<state root>/onboard``): the received requests, the
paired devices, this machine's pairing certificate, the cluster signing key.

The words: the machine that *listens* is the one that already has ml-stack and a cluster, and
whose owner accepts; the machine that *pairs* is the one asking to join.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.log import say, warn
from ml_stack.units import parse_duration

from ..discovery import Membership, memberships
from ..discovery import _write_memberships as write_memberships
from ..tls import TlsUnavailable, identity
from . import nearby as near
from .notify import compose, pick
from .pairing import DEFAULT_PORT, Grant, PairError, PairingClient, PairingServer
from .requests import Refused, Request, Requests, State, short

__all__ = ["add_commands", "adopt", "run"]

COMMANDS = ("nearby", "pair", "listen", "requests", "accept", "decline", "revoke", "bootstrap")


def state_dir(args: argparse.Namespace) -> Path:
    return Path(args.state) if getattr(args, "state", "") else home.state("onboard")


def _emit(args: argparse.Namespace, document: dict[str, Any], text: str) -> None:
    say(json.dumps(document, indent=1, default=str) if args.json else text)


def add_commands(sub: Any) -> None:
    """Add the onboarding commands to a subparsers object."""
    def common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--json", action="store_true")
        p.add_argument("--state", default="", help="where onboarding keeps its files "
                       "(default: <state root>/onboard)")
        return p

    p = common(sub.add_parser("nearby", help="machines on this network that are open to "
                                             "pairing"))
    p.add_argument("--timeout", type=float, default=3.0)
    p.add_argument("--port", type=int, default=near.DEFAULT_PORT)

    p = common(sub.add_parser("listen", help="open pairing on this machine for a while: "
                                             "announce, tell the owner of each request"))
    p.add_argument("--for", dest="span", default="10m", help="how long to stay open")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--announce-port", type=int, default=near.DEFAULT_PORT)
    p.add_argument("--host", default="", help="the address to listen on (default: every "
                                              "interface)")
    p.add_argument("--no-cluster", action="store_true",
                   help="pair the device without giving it this machine's cluster key")

    p = common(sub.add_parser("pair", help="ask a machine to let this one join"))
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--code", default="", help="the code the owner reads out (prompted for "
                                              "when omitted)")
    p.add_argument("--wait", default="5m", help="how long to wait for the owner to accept")
    p.add_argument("--name", default="")

    common(sub.add_parser("requests", help="join requests waiting for an answer"))
    for verb, text in (("accept", "say yes to a request and show its code"),
                       ("decline", "say no to a request")):
        p = common(sub.add_parser(verb, help=text))
        p.add_argument("request", help="the request id, or the first letters of it")

    p = common(sub.add_parser("revoke", help="stop trusting a paired device"))
    p.add_argument("device", help="its name or the first digits of its fingerprint")

    p = common(sub.add_parser("bootstrap", help="offer ml-stack to a machine that has none"))
    p.add_argument("--share", default="", help="a directory holding the wheel or archive to "
                                               "offer")
    p.add_argument("--valid", default="10m")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--ssh", default="", help="designed, not built; see docs/onboarding.md")


def adopt(grant: Grant, directory: Path, *, cluster_path: Path | str | None = None) -> bool:
    """Make a grant this machine's: join its cluster if it carries one, and remember the
    certificate and signing key of the machine that took it in. True if a cluster was joined."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    joined = False
    if grant.key:
        rows = [m for m in memberships(cluster_path) if m.group != grant.group]
        rows.insert(0, Membership(group=grant.group, key=grant.key.encode(), salt=grant.salt))
        write_memberships(rows, cluster_path)
        joined = True
    from ml_stack.files import write_json
    write_json(directory / "trust.json", {"schema_version": 1, "certificate": grant.certificate,
                                          "signing_key": grant.signing_key})
    return joined


def _identity(directory: Path) -> Any:
    try:
        return identity(directory / "tls", socket.gethostname()[:40])
    except TlsUnavailable as exc:
        raise SystemExit(f"cannot make a certificate: {exc}") from None


def _requests(directory: Path) -> Requests:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return Requests(directory / "requests.json")


def cmd_nearby(args: argparse.Namespace) -> int:
    transport = near.UdpTransport(port=args.port, group=near.DEFAULT_GROUP)
    try:
        found = near.Browser(transport).listen(args.timeout)
    finally:
        transport.close()
    rows = [n.public() for n in found]
    lines = [f"{r['name']}  {r['address']}:{r['port']}  certificate {r['fingerprint_short']}"
             for r in rows] or ["no machine is open to pairing"]
    _emit(args, {"nearby": rows}, "\n".join(lines))
    return 0 if rows else 1


def cmd_listen(args: argparse.Namespace) -> int:
    directory = state_dir(args)
    ident = _identity(directory)
    requests = _requests(directory)
    notifier = pick()
    held = memberships()
    signer_pub = ""
    if (directory / "signing.pub").exists():
        signer_pub = (directory / "signing.pub").read_text().strip()

    def grant(_request: Request) -> Grant:
        if args.no_cluster or not held:
            return Grant(certificate=ident.beacon, signing_key=signer_pub)
        m = held[0]
        return Grant(group=m.group, key=m.key.decode(), salt=m.salt, certificate=ident.beacon,
                     signing_key=signer_pub)

    def tell(request: Request) -> None:
        title, body = compose(request)
        notifier.notify(title, body)

    host = args.host or "0.0.0.0"  # noqa: S104  the owner opened pairing on purpose
    span = parse_duration(args.span) or 600.0
    transport = near.UdpTransport(port=args.announce_port, group=near.DEFAULT_GROUP)
    with PairingServer(requests, ident, grant=grant, notify=tell, host=host,
                       port=args.port) as server:
        announcer = near.Announcer(
            transport, name=socket.gethostname()[:40], hostname=socket.gethostname()[:60],
            model=f"{platform.system()} {platform.machine()}", port=server.port,
            fingerprint=ident.fingerprint).start()
        _emit(args, {"listening": True, "port": server.port, "for_s": span,
                     "fingerprint": ident.fingerprint, "notifier": notifier.name,
                     "gives_cluster_key": bool(held) and not args.no_cluster},
              f"pairing is open on port {server.port} for {int(span)} s "
              f"(certificate {short(ident.fingerprint)}); answer requests with "
              "'ml-stack fleet accept ID'")
        try:
            time.sleep(span)
        except KeyboardInterrupt:
            pass
        finally:
            announcer.stop()
            transport.close()
    return 0


def cmd_requests(args: argparse.Namespace) -> int:
    rows = [r.public() for r in _requests(state_dir(args)).pending()]
    lines = [f"{r['id'][:8]}  {r['state']:8}  {r['name']} ({r['hostname']}) {r['address']}  "
             f"certificate {r['fingerprint_short']}" for r in rows] or ["no requests waiting"]
    _emit(args, {"requests": rows}, "\n".join(lines))
    return 0


def _answer(args: argparse.Namespace, accept: bool) -> int:
    requests = _requests(state_dir(args))
    try:
        found = requests.find(args.request)
        done = requests.accept(found.id) if accept else requests.decline(found.id)
    except (KeyError, ValueError, Refused) as exc:
        reason = exc.reason if isinstance(exc, Refused) else str(exc) or "no such request"
        _emit(args, {"error": reason}, f"error: {reason}")
        return 2
    document = done.public()
    if accept:
        document["code"] = done.code
        document["code_valid_s"] = requests.limits.code_ttl_s
    text = (f"accepted {done.name} ({done.hostname}) at {done.address}, certificate "
            f"{short(done.fingerprint)}\nread this code to the person at that machine: "
            f"{done.code[:3]} {done.code[3:]}  (good for {int(requests.limits.code_ttl_s)} s)"
            if accept else f"declined {done.name}")
    _emit(args, document, text)
    return 0


def cmd_accept(args: argparse.Namespace) -> int:
    return _answer(args, True)


def cmd_decline(args: argparse.Namespace) -> int:
    return _answer(args, False)


def cmd_revoke(args: argparse.Namespace) -> int:
    requests = _requests(state_dir(args))
    try:
        gone = requests.devices.revoke(args.device)
    except (KeyError, ValueError) as exc:
        _emit(args, {"error": str(exc)}, f"error: {exc}")
        return 2
    rotate = gone.shared_cluster_key
    _emit(args, {"revoked": gone.fingerprint, "name": gone.name,
                 "cluster_key_rotation_needed": rotate},
          f"revoked {gone.name} ({short(gone.fingerprint)}): it cannot ask to join again."
          + (" It holds the cluster key, so the cluster key must be changed on every machine "
             "to lock it out (not automated yet; see docs/onboarding.md)." if rotate else ""))
    return 0


def cmd_pair(args: argparse.Namespace) -> int:
    directory = state_dir(args)
    ident = _identity(directory)
    client = PairingClient(args.host, args.port, fingerprint=ident.fingerprint)
    try:
        request_id = client.ask(name=args.name or socket.gethostname()[:40],
                                hostname=socket.gethostname()[:60],
                                model=f"{platform.system()} {platform.machine()}")
        warn(f"asked {args.host}; certificate {short(client.server_fingerprint)} -- check "
             "that this matches what the owner sees. Waiting for them to accept.")
        state = client.wait(parse_duration(args.wait) or 300.0)
        if state != State.ACCEPTED.value:
            _emit(args, {"state": state}, f"the request ended as {state}")
            return 1
        code = args.code or (getpass.getpass("code shown on the other machine: ")
                             if sys.stdin.isatty() else sys.stdin.readline().strip())
        grant = client.finish(code)
    except PairError as exc:
        _emit(args, {"error": str(exc), "tries_left": exc.tries_left}, f"error: {exc}")
        return 2
    joined = adopt(grant, directory)
    _emit(args, {"paired": True, "request": request_id, "joined_cluster": joined,
                 "server_fingerprint": client.server_fingerprint},
          "paired" + (f"; this machine is now in cluster '{grant.group}'" if joined else ""))
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    if args.ssh:
        _emit(args, {"error": "ssh push is designed, not built"},
              "error: ssh push is designed, not built (docs/onboarding.md, 'SSH push')")
        return 2
    from .bootstrap import BootstrapServer
    from .manifest import Signer, load_signer, verify

    directory = state_dir(args)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    share = Path(args.share) if args.share else None
    if share is None or not share.is_dir():
        _emit(args, {"error": "--share DIR with the wheel or archive to offer"},
              "error: name a directory holding the wheel or archive with --share")
        return 2
    keyfile = directory / "signing.key"
    if keyfile.exists():
        signer = load_signer(keyfile)
    else:
        signer = Signer.generate()
        signer.save(keyfile)
        (directory / "signing.pub").write_text(base64.b64encode(signer.public).decode())
    entries = [signer.entry(f, kind="wheel" if f.suffix == ".whl" else "sdist")
               for f in sorted(share.iterdir())
               if f.is_file() and f.suffix in (".whl", ".gz", ".zip")]
    raw = signer.sign(entries, serial=int(time.time()))
    manifest = verify(raw, signer.public)
    ident = _identity(directory)
    valid = parse_duration(args.valid) or 600.0
    with BootstrapServer(share, raw, manifest, ident, host=args.host, port=args.port,
                         valid_s=valid, owner=socket.gethostname()) as server:
        offer = server.offer
        _emit(args, {"url": offer.url, "expires_in_s": valid, "certificate": offer.fingerprint,
                     "signing_key": offer.key_id, "manifest_sha256": offer.manifest_sha256,
                     "files": [e.name for e in entries], "command": offer.command()},
              f"open this on the new machine: {offer.url}\ncertificate {short(offer.fingerprint)}"
              f", good for {int(valid)} s")
        try:
            time.sleep(valid)
        except KeyboardInterrupt:
            pass
    return 0


def run(args: argparse.Namespace) -> int:
    fn = {"nearby": cmd_nearby, "pair": cmd_pair, "listen": cmd_listen,
          "requests": cmd_requests, "accept": cmd_accept, "decline": cmd_decline,
          "revoke": cmd_revoke, "bootstrap": cmd_bootstrap}[args.cmd]
    return fn(args)

