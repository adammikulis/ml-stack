"""HTTPS against a certificate authority Python does not ship."""

from __future__ import annotations

import datetime
import http.server
import ipaddress
import ssl
import threading

import pytest

from ml_stack.client.health import is_healthy
from ml_stack.http import Server, trust


def _self_signed(directory) -> tuple[str, str]:
    """A certificate and key for 127.0.0.1, written into ``directory``."""
    x509 = pytest.importorskip("cryptography.x509", reason="ml-stack[privacy]")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path = directory / "cert.pem"
    key_path = directory / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    return str(cert_path), str(key_path)


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - the name http.server dispatches on
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def log_message(self, *args: object) -> None:
        pass


def test_a_server_signing_its_own_certificate_is_reachable_once_it_is_trusted(tmp_path):
    cert, key = _self_signed(tmp_path)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    httpd = Server(("127.0.0.1", 0), _Handler)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"https://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert not is_healthy(url), "an unknown authority must not verify"
        trust(cert)
        assert is_healthy(url)
    finally:
        trust(None)
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
