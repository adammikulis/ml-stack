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
import contextlib
import getpass
import json
import platform
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ml_stack import home, macauth
from ml_stack.files import read_json, write_json
from ml_stack.log import say, warn
from ml_stack.units import parse_duration

from ..discovery import Membership, _write_memberships as write_memberships, memberships, primary_ip
from ..tailnet import detect
from ..tls import TlsUnavailable, identity, pinned_context
from . import nearby as near
from .bootstrap import BootstrapServer, Terms
from .devices_cli import add_devices, cmd_devices
from .manifest import ManifestError, RotationAnnounced, Signer, key_fingerprint, verify
from .notify import compose, compose_code, pick
from .pairing import (
    DEFAULT_PORT,
    DEFAULT_SHARE_PORT,
    Grant,
    Hooks,
    Offer,
    PairError,
    PairingClient,
    PairingServer,
)
from .peerlearn import forget, learn_from_grant, learn_from_offer, new_secret
from .peers_cli import add_peers, cmd_peers
from .requests import Devices, Refused, Request, Requests, State, short
from .routes import beyond_the_lan, resolve
from .share_cli import add_share, cmd_share
from .signing import KeyStoreError, SigningKeys
from .signing_cli import add_signing, cmd_signing, confirm_signing
from .ssh_cli import add_ssh, cmd_ssh
from .transfer import (
    Downloader,
    NotShareable,
    PeerSource,
    Share,
    TransferError,
    fetch_manifest,
)
from .trusted import remember

__all__ = ["add_commands", "adopt", "run"]

COMMANDS = ("nearby", "pair", "listen", "requests", "accept", "decline", "revoke", "bootstrap",
            "share", "fetch", "signing", "devices", "peers")


def state_dir(args: argparse.Namespace) -> Path:
    return Path(args.state) if getattr(args, "state", "") else home.state("onboard")


def _emit(args: argparse.Namespace, document: dict[str, Any], text: str) -> None:
    say(json.dumps(document, indent=1, default=str) if args.json else text)


def _share_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--share-port", type=int, default=DEFAULT_SHARE_PORT,
                   help="the port 'fleet share' runs on here: the other machine records it as "
                        "this one's share address")
    p.add_argument("--no-share", action="store_true",
                   help="this machine shares nothing: the other machine records no share address")


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
    p.add_argument("--bind", default="", help="listen on this address only, and do not join "
                                              "the multicast group (for testing)")

    p = common(sub.add_parser("listen", help="open pairing on this machine for a while: "
                                             "announce, tell the owner of each request"))
    p.add_argument("--for", dest="span", default="10m", help="how long to stay open")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--announce-port", type=int, default=near.DEFAULT_PORT)
    p.add_argument("--announce-to", action="append", default=[], metavar="HOST:PORT",
                   help="announce to these addresses instead of the multicast group and the "
                        "broadcast address (repeatable)")
    p.add_argument("--no-announce", action="store_true",
                   help="do not announce; a machine must be told this one's address")
    p.add_argument("--host", default="", help="the address to listen on (default: every "
                                              "interface)")
    p.add_argument("--no-cluster", action="store_true",
                   help="pair the device without giving it this machine's cluster key")
    _share_args(p)

    p = common(sub.add_parser("pair", help="ask a machine to let this one join"))
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--code", default="", help="the code the owner reads out (prompted for "
                                              "when omitted)")
    p.add_argument("--wait", default="5m", help="how long to wait for the owner to accept")
    p.add_argument("--name", default="")
    _share_args(p)
    p.add_argument("--mine", action="store_true",
                   help="the machine being asked is yours: it may be given your gated models")

    common(sub.add_parser("requests", help="join requests waiting for an answer"))
    for verb, text in (("accept", "say yes to a request and show its code"),
                       ("decline", "say no to a request")):
        p = common(sub.add_parser(verb, help=text))
        p.add_argument("request", help="the request id, or the first letters of it")
        if verb == "accept":
            who = p.add_mutually_exclusive_group(required=True)
            who.add_argument("--mine", dest="mine", action="store_true",
                             help="this device is yours: it may be given your gated models")
            who.add_argument("--other", dest="mine", action="store_false",
                             help="this device is another person's")

    p = common(sub.add_parser("revoke", help="stop trusting a paired device"))
    p.add_argument("device", help="its name or the first digits of its fingerprint")

    p = common(sub.add_parser("bootstrap", help="offer ml-stack to a machine that has none"))
    p.add_argument("--share", default="", help="a directory holding the wheel or archive to "
                                               "offer")
    p.add_argument("--valid", default="10m")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--ssh", default="", metavar="[USER@]HOST",
                   help="install over the system's ssh instead of serving an address: the "
                        "owner's keys, a host key typed in full, a fixed audited script")
    add_ssh(p)
    p.add_argument("--advertise", default="", help="the address to put in the URL (default: "
                                                   "this machine's LAN address when --host is "
                                                   "a wildcard)")

    add_share(sub, common)
    add_devices(sub, common)
    add_peers(sub, common)
    add_signing(sub, common)

    p = common(sub.add_parser("fetch", help="fetch files from a machine this one paired with"))
    p.add_argument("names", nargs="+")
    p.add_argument("--from", dest="source", required=True, metavar="HOST:PORT",
                   help="a machine running 'share'")
    p.add_argument("--into", default="", help="where to stage them (default: "
                                              "<state>/staging)")


def adopt(grant: Grant, directory: Path, *, cluster_path: Path | str | None = None) -> bool:
    """Make a grant this machine's: join its cluster if it carries one, and remember the
    certificate and signing key of the machine that took it in. True if a cluster was joined."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    joined = False
    if grant.key:
        rows = [m for m in memberships(cluster_path) if m.group != grant.group]
        rows.insert(0, Membership(group=grant.group, key=grant.key.encode(), selection="manual"))
        write_memberships(rows, cluster_path)
        joined = True
    write_json(directory / "trust.json", {"schema_version": 1, "certificate": grant.certificate,
                                          "signing_key": grant.signing_key,
                                          "device_secret": grant.device_secret})
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
    transport = near.UdpTransport(bind=args.bind, port=args.port,
                                  group=None if args.bind else near.DEFAULT_GROUP)
    try:
        found = near.Browser(transport).listen(args.timeout)
    finally:
        transport.close()
    devices = Devices(state_dir(args) / "devices.json")
    known = {d.fingerprint: d for d in devices.all()}
    rows = [{**n.public(), "paired": None if n.fingerprint not in known else (
        "yours" if known[n.fingerprint].mine else "another person's")} for n in found]
    tailnet = detect() if known else None
    if tailnet is not None and tailnet.up:      # a beacon never crosses a tailnet
        for far in beyond_the_lan(devices, {n.fingerprint for n in found}, tailnet, DEFAULT_PORT):
            rows.append({"name": far.name, "address": far.address, "port": DEFAULT_PORT,
                         "fingerprint": far.fingerprint, "fingerprint_short": short(far.fingerprint),
                         "route": far.route.value,
                         "paired": "yours" if known[far.fingerprint].mine else "another person's"})
    lines = [f"{r['name']}  {r['address']}:{r['port']}  certificate {r['fingerprint_short']}"
             + (f"  paired, {r['paired']}" if r["paired"] else "") for r in rows] \
        or ["no machine is open to pairing"]
    _emit(args, {"nearby": rows}, "\n".join(lines))
    return 0 if rows else 1


def cmd_listen(args: argparse.Namespace) -> int:
    directory = state_dir(args)
    ident = _identity(directory)
    requests = _requests(directory)
    notifier = pick()
    held = memberships()
    signer_pub = base64.b64encode(SigningKeys(directory).public).decode()

    share_port = 0 if args.no_share else args.share_port
    me = socket.gethostname()[:40]

    def grant(_request: Request) -> Grant:
        secret = new_secret()
        if args.no_cluster or not held:
            return Grant(certificate=ident.beacon, signing_key=signer_pub, device_secret=secret,
                         share_port=share_port, name=me)
        m = held[0]
        return Grant(group=m.group, key=m.key.decode(), certificate=ident.beacon,
                     signing_key=signer_pub, device_secret=secret, share_port=share_port, name=me)

    def learned(request: Request, offer: Offer) -> None:
        learn_from_offer(directory, request, offer)

    def tell(request: Request) -> None:
        """Ask the owner with real buttons, off the request's own thread; the answer goes
        through the same state machine the command line uses."""
        title, body = compose(request)

        def work() -> None:
            outcome = notifier.ask(title, body)
            try:
                if outcome in ("mine", "other"):
                    accepted = requests.accept(request.id, mine=outcome == "mine")
                    notifier.show_code(*compose_code(accepted))
                elif outcome == "decline":
                    requests.decline(request.id)
            except Refused:
                pass            # answered elsewhere, or it ran out, while the dialog was up
        threading.Thread(target=work, daemon=True, name="onboard-ask").start()

    host = args.host or "0.0.0.0"  # noqa: S104  the owner opened pairing on purpose
    span = parse_duration(args.span) or 600.0
    where = [(h, int(p)) for h, _, p in (a.rpartition(":") for a in args.announce_to)]
    transport = near.UdpTransport(port=0 if where else args.announce_port,
                                  destinations=where or None,
                                  group=None if where else near.DEFAULT_GROUP,
                                  bind="127.0.0.1" if where else "")
    with PairingServer(requests, ident, Hooks(grant, tell, learned), address=(host, args.port)) as server:
        announcer = near.Announcer(transport, near.Presence(
            socket.gethostname()[:40], socket.gethostname()[:60],
            f"{platform.system()} {platform.machine()}", server.port, ident.fingerprint))
        if not args.no_announce:
            announcer.start()
        _emit(args, {"listening": True, "port": server.port, "for_s": span,
                     "fingerprint": ident.fingerprint, "notifier": notifier.name,
                     "gives_cluster_key": bool(held) and not args.no_cluster},
              f"pairing is open on port {server.port} for {int(span)} s "
              f"(certificate {short(ident.fingerprint)}); answer requests with "
              "'ml-stack fleet accept ID'")
        try:
            with contextlib.suppress(KeyboardInterrupt):
                time.sleep(span)
        finally:
            announcer.stop()
            transport.close()
    return 0


def cmd_requests(args: argparse.Namespace) -> int:
    rows = [r.public() for r in _requests(state_dir(args)).pending()]
    lines = [f"{r['id'][:8]}  {r['state']:8}  {r['name']} ({r['hostname']}) {r['address']}  "
             f"certificate {r['fingerprint_short']}"
             + (f"  {'yours' if r['mine'] else 'another person' + chr(39) + 's'}"
                if r["mine"] is not None else "") for r in rows] or ["no requests waiting"]
    _emit(args, {"requests": rows}, "\n".join(lines))
    return 0


def _answer(args: argparse.Namespace, accept: bool) -> int:
    requests = _requests(state_dir(args))
    try:
        found = requests.find(args.request)
        done = requests.accept(found.id, mine=args.mine) if accept \
            else requests.decline(found.id)
    except (KeyError, ValueError, Refused) as exc:
        reason = exc.reason if isinstance(exc, Refused) else str(exc) or "no such request"
        _emit(args, {"error": reason}, f"error: {reason}")
        return 2
    document = done.public()
    if accept:
        document["code"] = done.code
        document["code_valid_s"] = requests.limits.code_ttl_s
        document["signing_key"] = SigningKeys(state_dir(args)).key_id
        document["mine"] = done.mine
    text = (f"accepted {done.name} ({done.hostname}) at {done.address} as "
            f"{'yours' if done.mine else 'another person' + chr(39) + 's'}, certificate "
            f"{short(done.fingerprint)}\nread this code to the person at that machine: "
            f"{done.code[:3]} {done.code[3:]}  (good for {int(requests.limits.code_ttl_s)} s)\n"
            f"signing key {document.get('signing_key', '')[:16]}: the new machine shows the "
            "same digits when it has paired"
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
    dropped = forget(state_dir(args), gone)
    _emit(args, {"revoked": gone.fingerprint, "name": gone.name,
                 "cluster_key_rotation_needed": rotate, "peers_removed": dropped},
          f"revoked {gone.name} ({short(gone.fingerprint)}): it cannot ask to join again."
          + (" It holds the cluster key, so the cluster key must be changed on every machine "
             "to lock it out (not automated yet; see docs/onboarding.md)." if rotate else ""))
    return 0


def _offer(directory: Path, ident: Any, args: argparse.Namespace, secret: str) -> Offer:
    """How the machine that accepts can reach this one's share. Without a share port, or with
    no signing key to be had here, the offer says so and no row is made for it over there."""
    if args.no_share:
        return Offer()
    try:
        key = base64.b64encode(SigningKeys(directory).public).decode()
    except (KeyStoreError, OSError) as exc:
        warn(f"not offering a share address: {exc}")
        return Offer()
    return Offer(args.share_port, ident.beacon, key, secret)


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
        mine_secret = new_secret()
        grant = client.finish(code, _offer(directory, ident, args, mine_secret))
    except PairError as exc:
        _emit(args, {"error": str(exc), "tries_left": exc.tries_left}, f"error: {exc}")
        return 2
    joined = adopt(grant, directory)
    learn_from_grant(directory, grant, host=args.host, server_fingerprint=client.server_fingerprint,
                     my_secret=mine_secret, mine=args.mine)
    key_id = key_fingerprint(base64.b64decode(grant.signing_key)) if grant.signing_key else ""
    _emit(args, {"paired": True, "request": request_id, "joined_cluster": joined,
                 "server_fingerprint": client.server_fingerprint, "signing_key": key_id},
          "paired" + (f"; this machine is now in cluster '{grant.group}'" if joined else "")
          + f"\nsigning key {key_id[:16]}: it should match the other machine's screen")
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    if args.ssh:
        return cmd_ssh(args)
    directory = state_dir(args)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    share = Path(args.share) if args.share else None
    if share is None or not share.is_dir():
        _emit(args, {"error": "--share DIR with the wheel or archive to offer"},
              "error: name a directory holding the wheel or archive with --share")
        return 2
    keys = SigningKeys(directory)
    entries = [Signer.entry_for(f, kind="wheel" if f.suffix == ".whl" else "sdist")
               for f in sorted(share.iterdir())
               if f.is_file() and f.suffix in (".whl", ".gz", ".zip")]
    try:
        raw = keys.sign(entries, serial=int(time.time()), confirm=confirm_signing)
    except KeyStoreError as exc:
        _emit(args, {"error": str(exc)}, f"error: {exc}")
        return 2
    manifest = verify(raw, keys.public)
    ident = _identity(directory)
    valid = parse_duration(args.valid) or 600.0
    advertise = args.advertise or (primary_ip() if args.host in ("", "0.0.0.0")  # noqa: S104
                                   else args.host)
    with BootstrapServer(Share(share, raw, manifest), ident,
                         terms=Terms(valid, socket.gethostname(), advertise=advertise),
                         address=(args.host, args.port)) as server:
        offer = server.offer
        _emit(args, {"url": offer.url, "expires_in_s": valid, "certificate": offer.fingerprint,
                     "signing_key": offer.key_id, "manifest_sha256": offer.manifest_sha256,
                     "files": [e.name for e in entries], "command": offer.command()},
              f"open this on the new machine: {offer.url}\ncertificate {short(offer.fingerprint)}"
              f", good for {int(valid)} s")
        with contextlib.suppress(KeyboardInterrupt):
            time.sleep(valid)
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    directory = state_dir(args)
    trust = read_json(directory / "trust.json", {})
    held = memberships()
    host, _, port = args.source.rpartition(":")
    if not (held and trust.get("certificate") and trust.get("signing_key") and host
            and port.isdigit()):
        _emit(args, {"error": "pair with a machine first, and name HOST:PORT"},
              "error: pair with a machine first, and name HOST:PORT")
        return 2
    try:
        host = resolve(host, int(port), Devices(directory / "devices.json"), detect)
    except OSError as exc:
        _emit(args, {"error": str(exc)}, f"error: {exc}")
        return 2
    secret = macauth.derive(base64.urlsafe_b64decode(trust["device_secret"])) \
        if trust.get("device_secret") else macauth.derive(held[0].key)
    peer = PeerSource(f"https://{host}:{port}", secret,
                      pinned_context(trust["certificate"]), name=host)
    try:
        raw = fetch_manifest(peer)
        manifest = verify(raw, base64.b64decode(trust["signing_key"]),
                          min_serial=int(trust.get("serial", 0)),
                          revoked_keys=trust.get("revoked", []))
        remember(raw, base64.b64decode(trust["signing_key"]))
        staged = []
        for name in args.names:
            got = Downloader(manifest, [peer], Path(args.into) if args.into
                             else directory / "staging").download(name)
            staged.append(str(got))
    except RotationAnnounced as exc:
        write_json(directory / "trust.json",
                   {**trust, "pending_key": base64.b64encode(exc.new_public).decode()})
        _emit(args, {"error": str(exc), "pending_key": key_fingerprint(exc.new_public)},
              f"error: {exc}")
        return 2
    except (ManifestError, TransferError, OSError) as exc:
        reason = exc.source if isinstance(exc, NotShareable) and exc.source else str(exc)
        _emit(args, {"error": str(exc), "source": getattr(exc, "source", "")},
              f"error: {exc}" + (f" ({reason})" if reason != str(exc) else ""))
        return 2
    write_json(directory / "trust.json", {
        **trust, "serial": manifest.serial,
        "revoked": sorted({*trust.get("revoked", []), *manifest.revoked_keys})})
    _emit(args, {"staged": staged, "serial": manifest.serial},
          "staged (not installed): " + ", ".join(staged))
    return 0


def run(args: argparse.Namespace) -> int:
    fn = {"nearby": cmd_nearby, "pair": cmd_pair, "listen": cmd_listen,
          "requests": cmd_requests, "accept": cmd_accept, "decline": cmd_decline,
          "revoke": cmd_revoke, "bootstrap": cmd_bootstrap, "share": cmd_share,
          "fetch": cmd_fetch, "signing": cmd_signing, "devices": cmd_devices,
          "peers": cmd_peers}[args.cmd]
    return fn(args)

