"""A machine with nothing installed asks for ml-stack; nobody pushes it.

The owner runs ``ml-stack fleet bootstrap`` on a machine that has ml-stack. It serves, for a
few minutes, a page on the LAN at an unguessable address. The owner opens that address on the
new machine (typing it, or scanning the QR code a front end draws from `Offer.url`), reads what
will be installed, and chooses to run the one command the page shows. Nothing runs on the new
machine until a person there does something.

What the new machine checks, and what it relies on:

* the address carries the SHA-256 fingerprint of the offer's certificate, and the page's
  command pins it: the installer talks TLS to that one certificate and nothing else;
* the installer script is fixed text (`INSTALLER`), the same for every offer apart from three
  substituted values (address, certificate, manifest digest), short enough to read;
* it fetches the manifest and each file it lists, checks the manifest's SHA-256 against the
  digest printed on the page, and each file's size and SHA-256 against the manifest, and only
  then hands the wheel to pip. The Ed25519 signature is checked later, by ml-stack itself,
  once it is installed and holds the pinned key (pairing delivers it); the page's pinned
  manifest digest and certificate are what authenticate this first step.

Offers serve program files only (wheel, source archive), never models and never a cluster
key: joining a cluster is the pairing step, with a code, after ml-stack is installed.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import html
import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from ml_stack import macauth
from ml_stack.fleet import tls
from ml_stack.fleet.framing import Limited, LimitedServer
from ml_stack.safenames import Unsafe, safe_filename, safe_join

from .events import BUS, Bus
from .manifest import Manifest

__all__ = ["BootstrapServer", "INSTALLER", "Offer"]

PROGRAM_KINDS = ("wheel", "sdist")
MOST_DOWNLOADS = 8

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


class BootstrapServer:
    """Serves one offer. Uniform 404 for a wrong token, a lockout on repeated wrong ones."""

    def __init__(self, root: Path, manifest_raw: bytes, manifest: Manifest, ident: tls.Identity,
                 *, host: str = "127.0.0.1", port: int = 0, valid_s: float = 600.0,
                 owner: str = "", bus: Bus = BUS,
                 clock: Callable[[], float] = time.time) -> None:
        self.root, self.manifest_raw, self.manifest = Path(root), manifest_raw, manifest
        self.owner, self.bus, self.clock = owner, bus, clock
        self.token = secrets.token_urlsafe(16)
        self.expires = clock() + valid_s
        self.downloads = 0
        self.lockout = macauth.Lockout(failures=8, window_s=60.0, lock_s=300.0)
        self.ident = ident
        self.httpd = LimitedServer((host, port), _handler(self), tls=tls.server_context(ident))
        self._thread: threading.Thread | None = None
        self.offer = Offer(self.token, self.expires, host, self.port, ident.fingerprint,
                           manifest.key_id, hashlib.sha256(manifest_raw).hexdigest())

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def start(self) -> BootstrapServer:
        self.offer = Offer(self.token, self.expires, self.offer.host, self.port,
                           self.ident.fingerprint, self.manifest.key_id,
                           hashlib.sha256(self.manifest_raw).hexdigest(),
                           spki_pin(self.ident.der))
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True,
                                        name="onboard-bootstrap")
        self._thread.start()
        self.bus.emit("onboard.bootstrap.offered", "notice", "", host=self.offer.host,
                      port=self.port, expires=self.expires)
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self) -> BootstrapServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def installer(self) -> str:
        values = {"@HOST@": json.dumps(self.offer.host), "@PORT@": str(self.port),
                  "@TOKEN@": json.dumps(self.token),
                  "@CERT@": json.dumps(base64.b64encode(self.ident.der).decode()),
                  "@MANIFEST_SHA256@": json.dumps(self.offer.manifest_sha256),
                  "@KEY_ID@": json.dumps(self.manifest.key_id)}
        text = INSTALLER
        for key, value in values.items():
            text = text.replace(key, value)
        return text

    def page(self) -> str:
        rows = "".join(
            f"<li><code>{html.escape(e.name)}</code> {e.size:,} bytes, sha256 "
            f"<code>{e.sha256}</code></li>"
            for e in self.manifest.entries if e.kind in PROGRAM_KINDS)
        who = html.escape(self.owner) or "a machine on your network"
        return (
            "<!doctype html><meta charset=utf-8><title>Install ml-stack</title>"
            "<h1>Install ml-stack on this computer?</h1>"
            f"<p>{who} is offering the files below. Nothing has been installed. Nothing will be "
            "until you run the command at the end.</p>"
            f"<ul>{rows}</ul>"
            f"<p>Manifest sha256: <code>{self.offer.manifest_sha256}</code><br>"
            f"Signing key: <code>{self.manifest.key_id}</code><br>"
            f"This offer's certificate: <code>{self.offer.fingerprint}</code><br>"
            f"Expires in {max(0, int(self.expires - self.clock()))} s.</p>"
            "<p>The installer is a short Python script. Download it, read it, then run it:</p>"
            f"<pre>{html.escape(self.offer.command())}</pre>")


def _handler(server: BootstrapServer) -> type[BaseHTTPRequestHandler]:
    class Handler(Limited, BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: object) -> None:
            return

        def _say(self, status: int, body: bytes, kind: str = "text/plain") -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            who = self.client_address[0]
            if server.lockout.locked(who):
                self._say(429, b"too many wrong addresses")
                return
            parts = self.path.split("?")[0].split("/")
            ok = (len(parts) >= 3 and parts[1] == "b"
                  and secrets.compare_digest(parts[2].encode(), server.token.encode())
                  and server.clock() <= server.expires)
            if not ok:
                server.lockout.failed(who)
                self._say(404, b"not found")
                return
            rest = "/".join(parts[3:])
            if rest in ("", "index.html"):
                self._say(200, server.page().encode(), "text/html; charset=utf-8")
            elif rest == "manifest":
                server.downloads += 1
                self._say(200, server.manifest_raw, "application/json")
            elif rest == "install.py":
                self._say(200, server.installer().encode(), "text/x-python; charset=utf-8")
            elif rest.startswith("files/"):
                self._file(rest[6:])
            else:
                self._say(404, b"not found")

        def _file(self, name: str) -> None:
            try:
                entry = server.manifest.entry(safe_filename(name))
                path = safe_join(server.root, entry.name)
            except (KeyError, Unsafe):
                self._say(404, b"not found")
                return
            if entry.kind not in PROGRAM_KINDS or not entry.shareable or not path.is_file():
                server.bus.emit("onboard.bootstrap.refused", "notice", "", file=entry.name)
                self._say(403, b"this offer carries program files only")
                return
            with contextlib.suppress(OSError):
                self._say(200, path.read_bytes(), "application/octet-stream")

    return Handler
