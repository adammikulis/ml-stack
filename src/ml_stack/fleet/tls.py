"""Transport security for fleet traffic: a self-signed certificate per daemon, pinned by the
machines that talk to it.

A device makes its own certificate (`identity`) and is that certificate: it is pinned at pairing
and listed in the cluster's membership (`fleet.membership`), it is what the device presents as a
client and what its daemon serves. A client trusts the one certificate it was told to and no
other (`pinned_context`): no certificate authority and no first-use trust. A server asks every
client for its certificate and completes the handshake only with one that is a member
(`member_context`); a member that is revoked afterwards is refused at its next request.

Nothing turns this off: a link off this machine is TLS 1.3 or it is not made.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import shutil
import socket
import ssl
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ml_stack import home
from ml_stack.files import write_json
from ml_stack.windows_private import problem as windows_problem, restrict

__all__ = ["Identity", "Members", "TlsUnavailable", "device_directory", "identity", "local",
           "member_context", "pinned_context", "present", "server_context"]

VALID_DAYS = 3650
"""A device certificate is an identity pinned by the people it was paired with, not a session
credential: it lasts ten years and a replacement is a new pairing."""
RENEW_DAYS = 30
INSTALL = "pip install 'ml-stack[fleet-tls]' (the cryptography package), or put openssl on PATH"


class TlsUnavailable(RuntimeError):
    """No way to make a certificate here."""


@dataclass(frozen=True, slots=True)
class Identity:
    """A daemon's certificate and key on disk, and the certificate as peers receive it."""

    certfile: Path
    keyfile: Path
    der: bytes
    not_after: float

    @property
    def beacon(self) -> str:
        """The certificate for a beacon: base64 of its DER."""
        return base64.b64encode(self.der).decode()

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.der).hexdigest()


def _with_cryptography(name: str, now: float, days: float) -> tuple[bytes, bytes]:
    import datetime as dt

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    who = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name[:60] or "ml-stack")])
    start = dt.datetime.fromtimestamp(now - 3600, dt.UTC)
    cert = (x509.CertificateBuilder().subject_name(who).issuer_name(who)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(start).not_valid_after(start + dt.timedelta(days=days, hours=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, content_commitment=False,
                key_encipherment=False, data_encipherment=False, key_agreement=False,
                crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
                           critical=False)
            .sign(key, hashes.SHA256()))
    return (cert.public_bytes(serialization.Encoding.PEM),
            key.private_bytes(serialization.Encoding.PEM,
                              serialization.PrivateFormat.PKCS8,
                              serialization.NoEncryption()))


def _with_openssl(name: str, days: float) -> tuple[bytes, bytes]:
    exe = shutil.which("openssl")
    if exe is None:
        raise TlsUnavailable(f"cannot make a certificate: {INSTALL}")
    with tempfile.TemporaryDirectory() as tmp:
        cert, key = Path(tmp, "c.pem"), Path(tmp, "k.pem")
        done = subprocess.run(
            [exe, "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
             "-nodes", "-keyout", str(key), "-out", str(cert), "-days", str(max(1, int(days))),
             "-subj", f"/CN={''.join(c for c in name if c.isalnum() or c in '-.') or 'ml-stack'}",
             "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0",
             "-addext", "keyUsage=critical,digitalSignature,keyCertSign"],
            capture_output=True, text=True, timeout=60, check=False)
        if done.returncode != 0:
            raise TlsUnavailable(f"openssl could not make a certificate: {done.stderr[-300:]}")
        return cert.read_bytes(), key.read_bytes()


def _make(name: str, now: float, days: float) -> tuple[bytes, bytes]:
    try:
        return _with_cryptography(name, now, days)
    except ImportError:
        return _with_openssl(name, days)


def _plain(path: Path) -> None:
    for candidate in (path, *path.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (os.name == "nt" and
                info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise TlsUnavailable(f"TLS path is a symlink or Windows reparse point: {candidate}")


def _private(path: Path, mode: int) -> None:
    _plain(path)
    if os.name == "nt":
        restrict(path)
        if reason := windows_problem(path):
            raise TlsUnavailable(f"TLS path {path} {reason}")
    else:
        path.chmod(mode)


def identity(root: Path, name: str, *, now: float | None = None,
             days: float = VALID_DAYS, renew_days: float = RENEW_DAYS) -> Identity:
    """Return this daemon's certificate and account-private key, renewing near expiry."""
    now = time.time() if now is None else now
    _plain(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _private(root, 0o700)
    certfile, keyfile, meta = root / "cert.pem", root / "key.pem", root / "cert.json"
    for path in (certfile, keyfile, meta):
        _plain(path)
    if keyfile.exists():
        reason = (windows_problem(keyfile) if os.name == "nt" else
                  "permits another account" if keyfile.stat().st_mode & 0o077 else "")
        if reason:
            raise TlsUnavailable(f"TLS private key {keyfile} {reason}")
    with contextlib.suppress(OSError, ValueError, KeyError):
        held = json.loads(meta.read_text())
        if float(held["not_after"]) - now > renew_days * 86400 and keyfile.is_file():
            pem = certfile.read_text()
            return Identity(certfile, keyfile, ssl.PEM_cert_to_DER_cert(pem),
                            float(held["not_after"]))
    cert_pem, key_pem = _make(name, now, days)
    der = ssl.PEM_cert_to_DER_cert(cert_pem.decode())
    for path, data in ((keyfile, key_pem), (certfile, cert_pem)):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        _private(path, 0o600)
    not_after = now + days * 86400
    write_json(meta, {"not_after": not_after, "fingerprint": hashlib.sha256(der).hexdigest()})
    return Identity(certfile, keyfile, der, not_after)


def server_context(ident: Identity) -> ssl.SSLContext:
    """The context a listener answers TLS clients with, asking none for a certificate: TLS 1.3
    or nothing. Pairing and installing use it, since their clients are not members yet."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(str(ident.certfile), str(ident.keyfile))
    return ctx


def _partial(ctx: ssl.SSLContext) -> None:
    """Trust a self-signed certificate that is itself listed as the anchor."""
    ctx.verify_flags = (ctx.verify_flags | ssl.VERIFY_X509_PARTIAL_CHAIN) & ~ssl.VERIFY_X509_STRICT


class Members(Protocol):
    """What `member_context` reads from a membership record."""

    def stamp(self) -> Hashable: ...

    def certs_pem(self) -> str: ...


def member_context(ident: Identity, members: Members) -> Callable[[], ssl.SSLContext]:
    """A daemon's server context: TLS 1.3, and a client that presents a certificate must present
    a member's. The context is made again whenever the record changes, so a device that is
    revoked fails its next handshake; a client with no certificate still completes the
    handshake (a phone, a browser) and is held to its own credential by the request check."""
    built: list[Any] = [None, None]
    guard = threading.Lock()

    def current() -> ssl.SSLContext:
        stamp = members.stamp()
        with guard:
            if built[1] is None or built[0] != stamp:
                ctx = server_context(ident)
                ctx.verify_mode = ssl.CERT_OPTIONAL
                _partial(ctx)
                if pem := members.certs_pem():
                    ctx.load_verify_locations(cadata=pem)
                built[:] = [stamp, ctx]
            return built[1]

    return current


_PRESENTED: Identity | None = None


def present(ident: Identity | None) -> None:
    """Say which identity this process shows as a client from now on (a daemon says its own)."""
    global _PRESENTED
    _PRESENTED = ident


def device_directory() -> Path:
    """Where this machine's own identity lives."""
    return home.state("onboard", "tls")


def local() -> Identity:
    """This machine's identity: the one it presents as a client, made on first use."""
    return _PRESENTED or identity(device_directory(), socket.gethostname())


def pinned_context(cert: str, *, who: Identity | None = None) -> ssl.SSLContext:
    """A client context that trusts the one certificate ``cert`` (base64 DER) and nothing else:
    another certificate, or this one past its dates, fails the handshake. It presents ``who``
    (this machine's identity unless said) to a server that asks who is calling."""
    pem = ssl.DER_cert_to_PEM_cert(base64.b64decode(cert))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    _partial(ctx)
    ctx.load_verify_locations(cadata=pem)
    with contextlib.suppress(TlsUnavailable, OSError):
        caller = who or local()
        ctx.load_cert_chain(str(caller.certfile), str(caller.keyfile))
    return ctx
