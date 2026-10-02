"""A machine with nothing installed asks for ml-stack; nobody pushes it.

The owner runs ``ml-stack fleet bootstrap`` on a machine that has ml-stack. It serves, for a
few minutes, a page at an unguessable address on the LAN. The owner opens it on the new
machine (typed, or scanned from a QR code drawn from `Offer.url`), reads what would be
installed, and runs the one command the page shows. The address carries the offer's
certificate fingerprint and the command pins that certificate; the installer is fixed text
that checks the manifest digest printed on the page and each file's size and SHA-256 before it
hands anything to pip. Offers carry program files only, never models and never a cluster key:
joining a cluster is the pairing step, with a code, once ml-stack is installed.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from ml_stack import macauth
from ml_stack.fleet import tls
from ml_stack.safenames import Unsafe, safe_filename, safe_join

from .events import BUS, Bus
from .transfer import Share
from .web import Call, Listener, Reply

__all__ = ["INSTALLER", "BootstrapServer", "Offer", "Terms", "spki_pin"]

PROGRAM_KINDS = ("wheel", "sdist")

INSTALLER = r'''#!/usr/bin/env python3
"""Installs ml-stack from one machine on this network, checking everything it takes.

Read it before you run it. It makes no change except `pip install` of a file whose size
and SHA-256 it has checked, and only after you agree (or --yes). --dry-run stops before pip.
"""
import argparse, base64, hashlib, http.client, json, ssl, subprocess, sys, tempfile
from pathlib import Path

HOST, PORT, TOKEN = @HOST@, @PORT@, @TOKEN@
CERT_DER = base64.b64decode(@CERT@)
MANIFEST_SHA256 = @MANIFEST_SHA256@
KEY_ID = @KEY_ID@


def fetch(path):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.verify_flags = (ctx.verify_flags | ssl.VERIFY_X509_PARTIAL_CHAIN) & ~ssl.VERIFY_X509_STRICT
    ctx.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(CERT_DER))
    conn = http.client.HTTPSConnection(HOST, PORT, context=ctx, timeout=30)
    conn.request("GET", "/b/" + TOKEN + "/" + path)
    response = conn.getresponse()
    data = response.read(1 << 31)
    if response.status != 200:
        sys.exit("the offer answered %d for %s" % (response.status, path))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    raw = fetch("manifest")
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        sys.exit("the manifest is not the one the page named; nothing was installed")
    outer = json.loads(raw)
    body = outer["manifest"]
    if body.get("key_id") != KEY_ID:
        sys.exit("the manifest names a different signing key than the page; stopping")
    work = Path(tempfile.mkdtemp(prefix="ml-stack-bootstrap-"))
    wheels = []
    for entry in body["entries"]:
        if entry["kind"] not in ("wheel", "sdist"):
            continue
        name = entry["name"]
        if "/" in name or "\\" in name or name.startswith("."):
            sys.exit("a file name in the manifest is not a plain name; stopping")
        data = fetch("files/" + name)
        if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            sys.exit("%s does not match the manifest; nothing was installed" % name)
        (work / name).write_bytes(data)
        wheels.append(work / name)
        print("verified", name, entry["size"], "bytes", entry["sha256"][:16])
    if not wheels:
        sys.exit("the offer lists nothing to install")
    if args.dry_run:
        print("dry run: files verified in", work)
        return
    if not args.yes and input("Install %s with pip? [y/N] " % ", ".join(w.name for w in wheels)).lower() != "y":
        sys.exit("not installed")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *map(str, wheels)])


main()
'''


@dataclass(frozen=True, slots=True)
class Offer:
    token: str
    expires: float
    host: str
    port: int
    fingerprint: str
    key_id: str
    manifest_sha256: str
    spki_pin: str = ""
    """base64 SHA-256 of the certificate's public key, the form ``curl --pinnedpubkey`` takes."""

    @property
    def url(self) -> str:
        return f"https://{self.host}:{self.port}/b/{self.token}/#fp={self.fingerprint}"

    def command(self) -> str:
        """What the page shows to run: fetch the installer over the pinned certificate, read
        it, then run it."""
        return (f"curl --fail -k --pinnedpubkey 'sha256//{self.spki_pin}' "
                f"-o ml-stack-install.py https://{self.host}:{self.port}/b/{self.token}/install.py"
                " && python3 ml-stack-install.py")


def spki_pin(der: bytes) -> str:
    """The pin ``curl --pinnedpubkey sha256//...`` wants for a certificate."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    key = x509.load_der_x509_certificate(der).public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.b64encode(hashlib.sha256(key).digest()).decode()


@dataclass(slots=True)
class Terms:
    """How long an offer lasts, whose it is, and the clock that says."""

    valid_s: float = 600.0
    owner: str = ""
    clock: Callable[[], float] = time.time
    advertise: str = ""
    """The address to put in the URL and the installer, when it is not the one listened on."""


class BootstrapServer:
    """Serves one offer. Uniform 404 for a wrong token, a lockout on repeated wrong ones."""

    def __init__(self, share: Share, ident: tls.Identity, *, terms: Terms | None = None,
                 address: tuple[str, int] = ("127.0.0.1", 0), bus: Bus = BUS) -> None:
        self.share, self.ident, self.bus = share, ident, bus
        self.terms = terms or Terms()
        self.token = secrets.token_urlsafe(16)
        self.expires = self.terms.clock() + self.terms.valid_s
        self.lockout = macauth.Lockout(failures=8, window_s=60.0, lock_s=300.0)
        self.listener = Listener(self.dispatch, address, tls.server_context(ident))
        self.host = self.terms.advertise or address[0]
        self.offer = self._offer()

    def _offer(self) -> Offer:
        return Offer(self.token, self.expires, self.host, self.listener.port,
                     self.ident.fingerprint, self.share.manifest.key_id,
                     hashlib.sha256(self.share.manifest_raw).hexdigest(), spki_pin(self.ident.der))

    @property
    def port(self) -> int:
        return self.listener.port

    def start(self) -> BootstrapServer:
        self.offer = self._offer()
        self.listener.start()
        self.bus.emit("onboard.bootstrap.offered", "notice", "", host=self.host,
                      port=self.port, expires=self.expires)
        return self

    def stop(self) -> None:
        self.listener.stop()

    def __enter__(self) -> BootstrapServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def installer(self) -> str:
        values = {"@HOST@": json.dumps(self.host), "@PORT@": str(self.port),
                  "@TOKEN@": json.dumps(self.token),
                  "@CERT@": json.dumps(base64.b64encode(self.ident.der).decode()),
                  "@MANIFEST_SHA256@": json.dumps(self.offer.manifest_sha256),
                  "@KEY_ID@": json.dumps(self.share.manifest.key_id)}
        text = INSTALLER
        for key, value in values.items():
            text = text.replace(key, value)
        return text

    def page(self) -> str:
        rows = "".join(
            f"<li><code>{html.escape(e.name)}</code> {e.size:,} bytes, sha256 "
            f"<code>{e.sha256}</code></li>"
            for e in self.share.manifest.entries if e.kind in PROGRAM_KINDS)
        who = html.escape(self.terms.owner) or "a machine on your network"
        left = max(0, int(self.expires - self.terms.clock()))
        return (
            "<!doctype html><meta charset=utf-8><title>Install ml-stack</title>"
            "<h1>Install ml-stack on this computer?</h1>"
            f"<p>{who} is offering the files below. Nothing has been installed. Nothing will be "
            "until you run the command at the end.</p>"
            f"<ul>{rows}</ul>"
            f"<p>Manifest sha256: <code>{self.offer.manifest_sha256}</code><br>"
            f"Signing key: <code>{self.share.manifest.key_id}</code><br>"
            f"This offer's certificate: <code>{self.offer.fingerprint}</code><br>"
            f"Expires in {left} s.</p>"
            "<p>The installer is a short Python script. Download it, read it, then run it:</p>"
            f"<pre>{html.escape(self.offer.command())}</pre>")

    def dispatch(self, call: Call) -> Reply:
        if self.lockout.locked(call.client):
            return Reply(429, b"too many wrong addresses", content_type="text/plain")
        parts = call.path.split("?")[0].split("/")
        ok = (call.method == "GET" and len(parts) >= 3 and parts[1] == "b"
              and secrets.compare_digest(parts[2].encode(), self.token.encode())
              and self.terms.clock() <= self.expires)
        if not ok:
            self.lockout.failed(call.client)
            return Reply(404, b"not found", content_type="text/plain")
        rest = "/".join(parts[3:])
        if rest in ("", "index.html"):
            return Reply(200, self.page().encode(), content_type="text/html; charset=utf-8")
        if rest == "manifest":
            return Reply(200, self.share.manifest_raw, content_type="application/json")
        if rest == "install.py":
            return Reply(200, self.installer().encode(), content_type="text/x-python; charset=utf-8")
        if rest.startswith("files/"):
            return self.file(rest[6:])
        return Reply(404, b"not found", content_type="text/plain")

    def file(self, name: str) -> Reply:
        try:
            entry = self.share.manifest.entry(safe_filename(name))
            path = safe_join(self.share.root, entry.name)
        except (KeyError, Unsafe):
            return Reply(404, b"not found", content_type="text/plain")
        if entry.kind not in PROGRAM_KINDS or not entry.shareable or not path.is_file():
            self.bus.emit("onboard.bootstrap.refused", "notice", "", file=entry.name)
            return Reply(403, b"this offer carries program files only", content_type="text/plain")
        return Reply(200, path.read_bytes())
