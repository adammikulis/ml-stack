"""The local signing certificate of Poolhouse.app, and signing the bundle with it.

There is no Apple developer account, so the certificate is self-signed, made once in the person's login keychain and trusted
for code signing in the user domain only. A signature from the same certificate keeps its designated requirement across
rebuilds, which is what lets macOS keep a privacy permission (Local Network) for the app. When the certificate cannot be made
(the keychain refuses, or the person is not at the machine to approve it) the bundle is signed ad hoc with an identifier-only
requirement instead, and `sign` says so.
"""

from __future__ import annotations

import re
import secrets
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from ml_stack.home import account_roots
from ml_stack.log import warn

CERT_NAME = "Poolhouse Local Signing"
OPENSSL = "/usr/bin/openssl"  # the system LibreSSL: its .p12 files are the kind `security import` reads
DAYS = 3650
TOOL_TIMEOUT = 120.0


class SigningError(OSError):
    """The certificate could not be made or trusted, or the bundle could not be signed."""


def keychain() -> Path:
    """The account's login keychain (the account's own home, whatever HOME says)."""
    return account_roots()[0].parent / "Library" / "Keychains" / "login.keychain-db"


def parse_sha1(output: str) -> str:
    """The SHA-1 fingerprint (lower-case hex) in `security find-certificate -Z` output, or ''."""
    found = re.search(r"SHA-1 hash:\s*([0-9A-Fa-f]{40})", output)
    return found.group(1).lower() if found else ""


def certificate_request(common_name: str = CERT_NAME) -> list[str]:
    """The openssl arguments that make the self-signed code-signing certificate (key and certificate files are added by the caller)."""
    return ["req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", str(DAYS), "-subj", f"/CN={common_name}",
            "-addext", "basicConstraints=critical,CA:false", "-addext", "keyUsage=critical,digitalSignature",
            "-addext", "extendedKeyUsage=critical,codeSigning"]


def sign_command(bundle: Path, identifier: str, requirement: str, identity: str = CERT_NAME) -> list[str]:
    """The codesign command line: ``identity`` is the certificate's name, or "-" for an ad-hoc signature."""
    return ["codesign", "--force", "--sign", identity, "--identifier", identifier, "--timestamp=none", f"-r={requirement}", str(bundle)]


def _run(argv: list[str]) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, timeout=TOOL_TIMEOUT, check=False)
    if done.returncode:
        raise SigningError(f"{Path(argv[0]).name} {argv[1] if len(argv) > 1 else ''} failed: {(done.stderr or done.stdout).strip()[-400:]}")
    return done.stdout + done.stderr


def certificate_sha1() -> str:
    """The fingerprint of the certificate in the login keychain, or '' when there is none."""
    done = subprocess.run(["security", "find-certificate", "-c", CERT_NAME, "-Z", str(keychain())], capture_output=True, text=True,
                          timeout=TOOL_TIMEOUT, check=False)
    return parse_sha1(done.stdout) if done.returncode == 0 else ""


def create_certificate() -> str:
    """Make the certificate and its key, put both in the login keychain (codesign allowed to use the key), trust it for code
    signing in the user domain; returns the fingerprint. The key and certificate files exist only inside a private temporary directory.
    """
    password = secrets.token_hex(16)  # protects the throwaway .p12 for the seconds it exists
    with tempfile.TemporaryDirectory(prefix="poolhouse-cert") as scratch:
        key, cert, bundle = (Path(scratch) / name for name in ("key.pem", "cert.pem", "id.p12"))
        _run([OPENSSL, *certificate_request(), "-keyout", str(key), "-out", str(cert)])
        _run([OPENSSL, "pkcs12", "-export", "-inkey", str(key), "-in", str(cert), "-out", str(bundle), "-passout", f"pass:{password}"])
        _run(["security", "import", str(bundle), "-k", str(keychain()), "-P", password, "-T", "/usr/bin/codesign"])
        _run(["security", "add-trusted-cert", "-p", "codeSign", "-k", str(keychain()), str(cert)])
    return certificate_sha1()


def ensure_certificate() -> str:
    """The certificate's fingerprint, making the certificate first when the keychain has none (idempotent)."""
    return certificate_sha1() or create_certificate()


def sign(bundle: Path, identifier: str, requirement_for: Callable[[str], str], ad_hoc: str) -> str:
    """Sign ``bundle``: with the certificate when it can be had ("certificate"), else ad hoc with ``ad_hoc`` ("ad-hoc")."""
    try:
        fingerprint = ensure_certificate()
        if not fingerprint:
            raise SigningError("the certificate is not in the login keychain after making it")
        _run(sign_command(bundle, identifier, requirement_for(fingerprint)))
    except (SigningError, OSError, subprocess.SubprocessError) as exc:
        warn(f"node app: signing with {CERT_NAME!r} failed ({exc}); signing ad hoc, so the permission may be asked again after a rebuild")
        _run(sign_command(bundle, identifier, ad_hoc, identity="-"))
        return "ad-hoc"
    return "certificate"
