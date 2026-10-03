"""A model ml-stack pulled is pinned at pull time (never first-use trusted), and a signed
manifest this machine accepted is checked at load. Real files, real sealed sentinel store under
the test's state root, real Ed25519 signatures, a real local HTTP site / fake Hub."""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import time
from pathlib import Path

import pytest

from ml_stack import home, http, hub, net, sentinel
from ml_stack.fleet.onboard.manifest import Entry, ManifestError, Signer
from ml_stack.fleet.onboard.trusted import TrustedLists
from ml_stack.httpguard import Limits
from ml_stack.net.download import accept
from ml_stack.net.scan import Outcome, ScanPolicy, ScanResult
from ml_stack.sentinel import State
from ml_stack.serve import guarded
from ml_stack.serve.guarded import SentinelRefused
from ml_stack.testing.fakehub import fake_hub
from tests.net_site import Site, gguf_bytes

SHA = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731


class Clean:
    name = "clean"

    def available(self):
        return True

    def scan(self, path):
        return ScanResult(self.name, Outcome.CLEAN, "")


@pytest.fixture
def site():
    with Site() as s:
        yield s


@pytest.fixture
def pipe(tmp_path, site):
    return net.Pipeline(
        policy=net.Policy(allowed=["127.0.0.1"], path=tmp_path / "approvals.jsonl"),
        limits=Limits(allow_hosts=frozenset({"127.0.0.1"}), timeout=2.0, deadline_s=8.0),
        scanners=[Clean()], scan_policy=ScanPolicy())


def where(name: str) -> Path:
    """A model path under the state root, where sentinel may move a file aside."""
    folder = home.home() / "models"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def pins():
    return sentinel.default().manifest.pins()


def load(path: Path) -> str:
    return guarded.verify(path, state_file=home.home() / "servers.json", stop=lambda _p: None)


def pulled(site, pipe, name="m.gguf", sha=True):
    body = gguf_bytes(extra=4096)
    site.add(f"/{name}", body, headers={"Content-Type": "application/octet-stream"})
    final = home.home() / "models" / name
    net.download(site.base + f"/{name}", final, net.Want(sha256=SHA(body) if sha else ""), pipe)
    return final, body


# -- pin at pull -------------------------------------------------------------------------
def test_a_pulled_model_is_pinned_at_pull_with_its_source(tmp_path, site, pipe):
    final, body = pulled(site, pipe)
    pin = pins()[str(final)]
    assert (pin.sha256, pin.bytes, pin.kind, pin.source) == (SHA(body), len(body), "model", "pull")
    assert pin.origin == site.base + "/m.gguf" and pin.digest_from == "expected"
    assert pin.pinned_at > 0
    assert [e.evidence["origin"] for e in sentinel.default().bus.recent(kind="model.pinned_at_pull")]


def test_a_pulled_model_is_not_first_use_trusted_at_start(tmp_path, site, pipe):
    final, _ = pulled(site, pipe)
    assert load(final) == ""
    assert pins()[str(final)].source == "pull"           # untouched, not re-pinned as first-use
    assert not sentinel.default().bus.recent(kind="model.pinned_first_use")


def test_tamper_after_pull_is_refused_at_start_and_moved_aside(tmp_path, site, pipe):
    final, body = pulled(site, pipe)
    final.write_bytes(body[:-1] + b"X")                    # same size, other bytes
    with pytest.raises(SentinelRefused, match="does not match its pin"):
        load(final)
    assert not final.exists()
    assert sentinel.default().store.state_of("model", str(final)) == State.QUARANTINED


def test_a_pull_without_a_published_digest_pins_the_bytes_it_got_and_says_so(tmp_path, site, pipe):
    final, body = pulled(site, pipe, sha=False)
    pin = pins()[str(final)]
    assert pin.sha256 == SHA(body) and pin.digest_from == "computed" and pin.source == "pull"


def test_a_model_ml_stack_did_not_pull_is_pinned_on_first_use_and_logged_distinctly(tmp_path):
    path = where("other.gguf")
    path.write_bytes(gguf_bytes(extra=512))
    load(path)
    pin = pins()[str(path)]
    assert pin.source == "first-use" and pin.digest_from == "computed" and pin.origin == ""
    assert sentinel.default().bus.recent(kind="model.pinned_first_use")
    assert not sentinel.default().bus.recent(kind="model.pinned_at_pull")


def test_a_download_that_fails_its_hash_records_no_pin(tmp_path, site, pipe):
    body = gguf_bytes(extra=4096)
    site.add("/bad.gguf", body, headers={"Content-Type": "application/octet-stream"})
    with pytest.raises(net.ChecksumMismatch):
        net.download(site.base + "/bad.gguf", tmp_path / "bad.gguf", net.Want(sha256=SHA(b"no")), pipe)
    assert pins() == {} and not (tmp_path / "bad.gguf").exists()


def test_a_download_that_fails_the_format_check_records_no_pin(tmp_path, site, pipe):
    junk = b"this is not a gguf" * 40
    site.add("/fake.gguf", junk)
    with pytest.raises(net.Blocked):
        net.download(site.base + "/fake.gguf", tmp_path / "fake.gguf", net.Want(sha256=SHA(junk)), pipe)
    assert pins() == {}


def test_a_file_from_a_paired_device_is_pinned_with_its_peer(tmp_path, pipe):
    body = gguf_bytes(extra=2048)
    staged = tmp_path / "stage" / "p.gguf"
    staged.parent.mkdir()
    staged.write_bytes(body)
    final = tmp_path / "models" / "p.gguf"
    accept(staged, final, net.Want(sha256=SHA(body)), "peer:desk", pipe)
    pin = pins()[str(final)]
    assert pin.origin == "peer:desk" and pin.source == "pull" and pin.sha256 == SHA(body)


def test_hub_pull_pins_every_shard(monkeypatch):
    shard = lambda seed: (b"GGUF" + struct.pack("<IQQ", 3, 0, 0) + bytes([seed]) * 4096)  # noqa: E731
    repos = {"maker/big-GGUF": {"Q4/big-00001-of-00002.gguf": shard(1),
                                "Q4/big-00002-of-00002.gguf": shard(2)}}
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", "/nonexistent-hf-home")
    monkeypatch.setattr(http, "check", lambda url: url)
    with fake_hub(repos) as hub_:
        hub_.point(monkeypatch)
        first = hub.pull("hf:maker/big-GGUF/Q4/big-00001-of-00002.gguf", peers=False)
    held = pins()
    for name, body in repos["maker/big-GGUF"].items():
        pin = held[str(first.parent.parent / name)]
        assert pin.source == "pull" and pin.sha256 == SHA(body) and pin.digest_from == "expected"


# -- signed manifests at load --------------------------------------------------------------
def pin_owner_key(signer: Signer) -> bytes:
    """What pairing does: the owner's key lands in the onboarding state (trust.json)."""
    folder = home.state("onboard")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "trust.json").write_text(json.dumps(
        {"signing_key": base64.b64encode(signer.public).decode()}), encoding="utf-8")
    return signer.public


def listing(signer: Signer, serial: int, *entries: Entry) -> bytes:
    return signer.sign(entries, serial=serial)


def entry_for(path: Path, body: bytes) -> Entry:
    return Entry(path.name, len(body), SHA(body), 65536, (SHA(body),), kind="model")


def test_a_model_the_signed_manifest_lists_is_verified_by_its_serial(tmp_path):
    signer = Signer.generate()
    key = pin_owner_key(signer)
    path = where("listed.gguf")
    body = gguf_bytes(extra=1024)
    path.write_bytes(body)
    TrustedLists().ingest(listing(signer, 7, entry_for(path, body)), key)
    assert load(path) == "verified by manifest serial 7"
    assert load(path) == "verified by manifest serial 7"        # also from the pin on the next start
    assert sentinel.default().bus.recent(kind="model.manifest_verified")


def test_a_model_that_differs_from_the_manifest_is_refused_and_quarantined_not_pinned(tmp_path):
    signer = Signer.generate()
    key = pin_owner_key(signer)
    path = where("listed.gguf")
    good = gguf_bytes(extra=1024)
    path.write_bytes(good + b"evil")                     # what is on disk is not what was signed
    TrustedLists().ingest(listing(signer, 3, entry_for(path, good)), key)
    with pytest.raises(SentinelRefused, match="signed manifest"):
        load(path)
    assert not path.exists()
    assert sentinel.default().store.state_of("model", str(path)) == State.QUARANTINED
    assert str(path) not in pins()


def test_a_pinned_model_whose_pin_disagrees_with_the_manifest_is_refused(tmp_path, site, pipe):
    """A file pulled without a trusted digest is pinned as it arrived; the manifest still wins."""
    final, _ = pulled(site, pipe, sha=False)
    signer = Signer.generate()
    key = pin_owner_key(signer)
    other = gguf_bytes(extra=9000)
    TrustedLists().ingest(listing(signer, 1, entry_for(final, other)), key)
    with pytest.raises(SentinelRefused, match="signed manifest"):
        load(final)
    assert sentinel.default().store.state_of("model", str(final)) == State.QUARANTINED


def test_a_replayed_older_manifest_is_refused_and_does_not_replace_the_newer(tmp_path):
    signer = Signer.generate()
    key = pin_owner_key(signer)
    path = where("listed.gguf")
    new, old = gguf_bytes(extra=100), gguf_bytes(extra=200)
    path.write_bytes(new)
    lists = TrustedLists()
    lists.ingest(listing(signer, 9, entry_for(path, new)), key)
    with pytest.raises(ManifestError, match="older"):
        lists.ingest(listing(signer, 8, entry_for(path, old)), key)
    assert lists.high_water(signer.key_id) == 9
    assert load(path) == "verified by manifest serial 9"        # the old list did not take over
    lists.ingest(listing(signer, 9, entry_for(path, new)), key)  # the same serial again is fine


def test_an_unsigned_or_foreign_manifest_is_ignored(tmp_path):
    signer, stranger = Signer.generate(), Signer.generate()
    key = pin_owner_key(signer)
    path = where("listed.gguf")
    body = gguf_bytes(extra=300)
    path.write_bytes(body)
    lists = TrustedLists()
    plain = json.dumps({"manifest": {"serial": 5, "entries": []}}).encode()
    with pytest.raises(ManifestError):
        lists.ingest(plain, key)
    forged = listing(stranger, 99, entry_for(path, b"something else"))
    with pytest.raises(ManifestError, match="not by the pinned key"):
        lists.ingest(forged, key)
    assert lists.lookup(path.name) == [] and lists.high_water(signer.key_id) == -1
    assert load(path) == ""                              # no manifest applies: first-use pin
    assert pins()[str(path)].source == "first-use"


def test_a_stored_list_whose_key_is_no_longer_pinned_stops_counting(tmp_path):
    signer = Signer.generate()
    key = pin_owner_key(signer)
    path = where("listed.gguf")
    body = gguf_bytes(extra=300)
    path.write_bytes(body + b"!")
    TrustedLists().ingest(listing(signer, 2, entry_for(path, body)), key)
    (home.state("onboard") / "trust.json").unlink()      # the person un-paired
    assert load(path) == ""


def test_a_list_signed_by_a_key_that_was_revoked_since_stops_counting(tmp_path):
    signer = Signer.generate()
    key = pin_owner_key(signer)
    path = where("listed.gguf")
    body = gguf_bytes(extra=300)
    path.write_bytes(body + b"!")
    TrustedLists().ingest(listing(signer, 2, entry_for(path, body)), key)
    trust = home.state("onboard") / "trust.json"
    doc = json.loads(trust.read_text(encoding="utf-8"))
    trust.write_text(json.dumps({**doc, "revoked": [signer.key_id]}), encoding="utf-8")
    assert load(path) == ""
    assert pins()[str(path)].source == "first-use"


def test_a_signed_list_expires_for_fetching_but_still_checks_held_bytes(tmp_path):
    signer = Signer.generate()
    key = pin_owner_key(signer)
    path = where("listed.gguf")
    body = gguf_bytes(extra=300)
    path.write_bytes(body)
    stale = signer.sign([entry_for(path, body)], serial=4, valid_s=1, now=time.time() - 10)
    with pytest.raises(ManifestError, match="expired"):
        TrustedLists().ingest(stale, key)
    fresh = signer.sign([entry_for(path, body)], serial=4, valid_s=1)
    TrustedLists().ingest(fresh, key)
    time.sleep(1.2)                                       # the list lapses; the bytes it vouched for stay checked
    assert load(path) == "verified by manifest serial 4"
