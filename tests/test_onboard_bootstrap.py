"""The offer for a machine with nothing installed: a page, a pinned certificate, a fixed
installer that checks every byte, and an address that runs out."""

import http.client
import os
import shutil
import ssl
import subprocess
import sys
from dataclasses import replace

import pytest
from onboard_support import Clock, Recorder, identity

from ml_stack.fleet.onboard import (
    bootstrap as bs,
    manifest as mf,
)
from ml_stack.fleet.onboard.transfer import Share

WHEEL = b"PK-this-is-a-wheel" * 5000


@pytest.fixture(autouse=True)
def needs_cryptography():
    pytest.importorskip("cryptography")


@pytest.fixture
def offer(tmp_path):
    share = tmp_path / "share"
    share.mkdir()
    (share / "ml_stack-0.2-py3-none-any.whl").write_bytes(WHEEL)
    (share / "tiny-model.gguf").write_bytes(b"GGUF" * 100)
    signer = mf.Signer.generate()
    entries = [signer.entry(share / "ml_stack-0.2-py3-none-any.whl", kind="wheel"),
               signer.entry(share / "tiny-model.gguf", kind="model")]
    raw = signer.sign(entries, serial=1)
    rec, clock = Recorder(), Clock()
    server = bs.BootstrapServer(Share(share, raw, mf.verify(raw, signer.public)),
                                identity(tmp_path, "ctl"),
                                terms=bs.Terms(owner="den-mac <b>", clock=clock),
                                bus=rec.bus).start()
    yield type("Offer", (), {"server": server, "rec": rec, "clock": clock, "share": share,
                             "tmp": tmp_path})
    server.stop()


def get(offer, path, token=None):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    conn = http.client.HTTPSConnection("127.0.0.1", offer.server.port, context=ctx, timeout=10)
    conn.request("GET", f"/b/{token or offer.server.token}/{path}")
    r = conn.getresponse()
    return r.status, r.read()


def run_installer(offer, *flags):
    script = offer.tmp / "install.py"
    script.write_text(offer.server.installer())
    return subprocess.run([sys.executable, str(script), *flags], capture_output=True, text=True,
                          timeout=60, check=False,
                          env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})


def test_the_page_says_what_will_happen_and_escapes_the_owner_name(offer):
    status, body = get(offer, "")
    page = body.decode()
    assert status == 200 and "Nothing has been installed" in page
    assert "ml_stack-0.2-py3-none-any.whl" in page and "tiny-model" not in page   # programs only
    assert "&lt;b&gt;" in page and "<b>" not in page
    assert offer.server.offer.fingerprint in page and offer.server.offer.manifest_sha256 in page


def test_the_installer_verifies_and_stops_before_pip_on_a_dry_run(offer):
    done = run_installer(offer, "--dry-run")
    assert done.returncode == 0, done.stderr
    assert "verified ml_stack-0.2-py3-none-any.whl" in done.stdout


def test_a_tampered_file_makes_the_installer_install_nothing(offer):
    (offer.share / "ml_stack-0.2-py3-none-any.whl").write_bytes(WHEEL[:-1] + b"X")
    done = run_installer(offer, "--dry-run")
    assert done.returncode != 0 and "does not match the manifest" in done.stderr


def test_a_swapped_manifest_is_caught_by_the_digest_the_page_pinned(offer):
    offer.server.share = replace(offer.server.share, manifest_raw=offer.server.share.manifest_raw.replace(b"serial", b"seria1"))
    done = run_installer(offer, "--dry-run")
    assert done.returncode != 0 and "not the one the page named" in done.stderr


def test_the_installer_talks_only_to_the_offers_own_certificate(offer, tmp_path):
    script = offer.server.installer()
    # point it at an impostor: same address, other certificate
    other = bs.BootstrapServer(offer.server.share, identity(tmp_path, "impostor"),
                               bus=offer.rec.bus).start()
    try:
        forged = script.replace(f", {offer.server.port},", f", {other.port},")
        assert forged != script
        path = tmp_path / "forged.py"
        path.write_text(forged)
        done = subprocess.run([sys.executable, str(path), "--dry-run"], capture_output=True,
                              text=True, timeout=60, check=False)
        assert done.returncode != 0 and "verified" not in done.stdout
        assert "CERTIFICATE_VERIFY_FAILED" in done.stderr
    finally:
        other.stop()


def test_a_wrong_token_is_a_404_and_repeats_are_locked_out(offer):
    statuses = [get(offer, "", token="x" * 22)[0] for _ in range(10)]
    assert statuses[0] == 404 and statuses[-1] == 429
    assert get(offer, "")[0] == 429                  # even the right address, from the locked-out one


def test_the_offer_runs_out(offer):
    assert get(offer, "manifest")[0] == 200
    offer.clock.advance(601)
    assert get(offer, "manifest")[0] == 404


def test_models_and_unknown_files_are_never_served(offer):
    assert get(offer, "files/tiny-model.gguf")[0] == 403
    assert get(offer, "files/..%2F..%2Fetc%2Fpasswd")[0] == 404
    assert get(offer, "files/nothing.whl")[0] == 404
    assert offer.rec.of("onboard.bootstrap.refused")


@pytest.mark.skipif(shutil.which("curl") is None, reason="needs curl")
def test_the_curl_command_pins_the_offers_key_and_refuses_another(offer, tmp_path):
    target = f"https://127.0.0.1:{offer.server.port}/b/{offer.server.token}/install.py"
    good = subprocess.run(["curl", "-sS", "--fail", "-k", "--pinnedpubkey",
                           f"sha256//{offer.server.offer.spki_pin}", "-o", str(tmp_path / "i.py"),
                           target], capture_output=True, text=True, timeout=30, check=False)
    if "pinnedpubkey" in good.stderr and "not supported" in good.stderr.lower():
        pytest.skip("this curl cannot pin")
    assert good.returncode == 0 and (tmp_path / "i.py").read_text() == offer.server.installer()
    other = bs.spki_pin(identity(tmp_path, "someone-else").der)
    bad = subprocess.run(["curl", "-sS", "--fail", "-k", "--pinnedpubkey", f"sha256//{other}",
                          "-o", str(tmp_path / "j.py"), target], capture_output=True, text=True,
                         timeout=30, check=False)
    assert bad.returncode != 0 and not (tmp_path / "j.py").exists()
    assert offer.server.offer.command().count("--pinnedpubkey") == 1


def test_the_installer_is_short_enough_to_read(offer):
    lines = offer.server.installer().splitlines()
    assert len(lines) < 90
    assert "curl" not in offer.server.installer() and "| sh" not in offer.server.installer()
