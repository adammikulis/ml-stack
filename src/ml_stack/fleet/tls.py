"""Transport security for fleet traffic: a self-signed certificate per daemon, pinned by the
machines that talk to it.

A daemon makes its own certificate (`identity`) and puts it in its beacon, which is signed with
the cluster key, so a machine that holds the key learns the certificate from a source it can
trust and a stranger cannot forge. A client then trusts that one certificate and no other
(`pinned_context`): no certificate authority and no first-use trust. A certificate that has
expired, or been replaced since the beacon, is refused by the handshake.

`ML_STACK_FLEET_TLS=off` is the one switch that turns this off, for signed-only traffic.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ml_stack.files import write_json

__all__ = ["ENV", "Identity", "TlsUnavailable", "disabled", "identity", "pinned_context",
           "server_context"]

ENV = "ML_STACK_FLEET_TLS"
VALID_DAYS = 90
RENEW_DAYS = 30
INSTALL = "pip install 'ml-stack[fleet-tls]' (the cryptography package), or put openssl on PATH"


class TlsUnavailable(RuntimeError):
    """No way to make a certificate here."""


def disabled() -> bool:
    """Whether the operator has named signed-only traffic with ``ML_STACK_FLEET_TLS=off``."""
    return os.environ.get(ENV, "").strip().lower() == "off"


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


def identity(root: Path, name: str, *, now: float | None = None,
             days: float = VALID_DAYS, renew_days: float = RENEW_DAYS) -> Identity:
    """This daemon's certificate under ``root``, made on first use and made again when fewer
    than ``renew_days`` of it are left. The key is mode 0600 in a directory that is 0700."""
    now = time.time() if now is None else now
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    certfile, keyfile, meta = root / "cert.pem", root / "key.pem", root / "cert.json"
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
        path.chmod(0o600)
    not_after = now + days * 86400
    write_json(meta, {"not_after": not_after, "fingerprint": hashlib.sha256(der).hexdigest()})
    return Identity(certfile, keyfile, der, not_after)


def server_context(ident: Identity) -> ssl.SSLContext:
    """The context a daemon answers TLS clients with."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(ident.certfile), str(ident.keyfile))
    return ctx


def pinned_context(cert: str) -> ssl.SSLContext:
    """A client context that trusts the one certificate ``cert`` (a beacon's base64 DER) and
    nothing else: another certificate, or this one past its dates, fails the handshake."""
    pem = ssl.DER_cert_to_PEM_cert(base64.b64decode(cert))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.verify_flags = (ctx.verify_flags | ssl.VERIFY_X509_PARTIAL_CHAIN) \
        & ~ssl.VERIFY_X509_STRICT
    ctx.load_verify_locations(cadata=pem)
    return ctx
