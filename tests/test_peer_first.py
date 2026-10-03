"""Model downloads ask the owner's paired devices before the Hub (docs/model-discovery.md).

Real processes and files: each peer is `peer_serve.py` (the real share server over pinned TLS,
with its own home and sentinel), the Hub is the local stand-in, the file is a small real GGUF.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from onboard_support import Recorder
from test_hub_pull import blob

from ml_stack import hub, macauth, sentinel
from ml_stack.fleet import tls
from ml_stack.fleet.onboard import lan, manifest as mf, peerfirst
from ml_stack.fleet.onboard.human import mint
from ml_stack.fleet.onboard.peers_cli import cmd_peers
from ml_stack.fleet.onboard.requests import Device, Devices
from ml_stack.fleet.onboard.sharing import Licences
from ml_stack.hub import peers as hub_peers, transfer as pulling
from ml_stack.sentinel.store import Holding
from ml_stack.testing.fakehub import fake_hub

MIB = 1 << 20
CHUNK = 256 * 1024
REF = "hf:maker/thing-GGUF/thing-Q4_K_M.gguf"
NAME = "thing-Q4_K_M.gguf"
GOOD = blob(3 * MIB, 1)
SHA = hashlib.sha256(GOOD).hexdigest()
CLUSTER = "k" * 32
HELPER = Path(__file__).parent / "peer_serve.py"


@pytest.fixture(autouse=True)
def needs_cryptography():
    pytest.importorskip("cryptography")


@pytest.fixture
def stand_in(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("ML_STACK_NO_PEERS", raising=False)
    monkeypatch.setenv("HF_HOME", "/nonexistent-hf-home")
    with fake_hub({"maker/thing-GGUF": {NAME: GOOD}}) as hub_:
        hub_.point(monkeypatch)
        yield hub_


def file_requests(hub_) -> int:
    """Requests the Hub stand-in saw for the file's bytes (its listing is not one)."""
    return sum(1 for _m, p, _r, _a in [*hub_.hub_seen, *hub_.cdn_seen] if "/resolve/" in p
               or p.startswith("/cdn/"))


class Fleet:
    """Peers started as processes, each with its own folder, home, certificate and request log."""

    def __init__(self, tmp: Path, monkeypatch) -> None:
        self.tmp, self.monkeypatch, self.procs = tmp, monkeypatch, []
        self.signer = mf.Signer.generate()
        self.book = peerfirst.PeerBook()
        self.rows: list[dict] = []

    def peer(self, name: str, data: bytes | None = GOOD, *, entry: mf.Entry | None = None,
             secret: str = CLUSTER, env=None, **terms) -> dict:
        root = self.tmp / name
        root.mkdir()
        (root / "state").mkdir()
        if data is not None:
            (root / "files").mkdir()
            (root / "files" / NAME).write_bytes(data)
        signer, licence = self.signer, terms.get("licence", "")
        entries = []
        if entry is not None:
            entries = [entry]
        elif data is not None:
            entries = [signer.entry(root / "files" / NAME, kind="model", chunk_size=CHUNK,
                                    licence_url="https://licence.example/x" if licence else "",
                                    **{"sharing": "open", **terms})]
        raw = signer.sign(entries, serial=7)
        (root / "manifest").write_bytes(raw)
        tls_dir = root / "tls"
        ident = tls.identity(tls_dir, "peer")
        log = root / "requests.log"
        log.write_text("")
        home = root / "home"
        argv = [sys.executable, str(HELPER), str(root / "files"), str(root / "manifest"),
                base64.b64encode(signer.public).decode(), macauth.derive(CLUSTER.encode()),
                str(tls_dir), str(root / "state")]
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "PEER_LOG": str(log),
                 "ML_STACK_HOME": str(home), **(env or {})})
        self.procs.append(proc)
        port = int(proc.stdout.readline())
        row = {"name": name, "url": f"https://127.0.0.1:{port}", "certificate": ident.beacon,
               "signing_key": base64.b64encode(self.signer.public).decode(),
               "cluster_key": secret_key(secret), "min_serial": 0}
        self.rows.append(row)
        self.book.add(row)
        return {"root": root, "log": log, "home": home, "row": row, "port": port,
                "entry": entries[0] if entries else None}

    def requests(self, peer: dict) -> list[str]:
        return peer["log"].read_text().splitlines()

    def close(self) -> None:
        for p in self.procs:
            with contextlib.suppress(OSError):
                p.stdin.close()
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.wait(timeout=20)
            p.kill()


def secret_key(secret: str) -> str:
    return CLUSTER if secret == CLUSTER else "wrong-key-" + "w" * 22


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    f = Fleet(tmp_path, monkeypatch)
    yield f
    f.close()


@pytest.fixture
def events():
    """Sentinel's events during the test (the process-wide sentinel of this test's state)."""
    seen = []
    stop = sentinel.default().bus.subscribe(seen.append)
    yield seen
    stop()


def staging_left(tmp_path) -> list[Path]:
    return list((tmp_path / "machine-state" / "net" / "staging" / "peers").glob("*/*"))


# (a) --------------------------------------------------------------------------------------
def test_a_paired_peer_that_has_the_file_serves_it_and_the_hub_sees_no_file_request(
        stand_in, fleet, tmp_path):
    peer = fleet.peer("kitchen")
    seen = []
    got = hub.pull(REF, tmp_path / "out", seen.append)
    assert got.read_bytes() == GOOD
    assert hashlib.sha256(got.read_bytes()).hexdigest() == SHA
    assert file_requests(stand_in) == 0 and not stand_in.downloads
    assert len(fleet.requests(peer)) == 3 * MIB // CHUNK
    assert seen[-1].phase == "done" and seen[-1].done_bytes == 3 * MIB
    assert {p.phase for p in seen} >= {"downloading", "verifying", "done"}
    assert not staging_left(tmp_path)


def test_the_file_two_peers_share_comes_from_both(stand_in, fleet, tmp_path):
    a, b = fleet.peer("a"), fleet.peer("b")
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert fleet.requests(a) and fleet.requests(b)
    assert file_requests(stand_in) == 0


# (b) --------------------------------------------------------------------------------------
def test_a_corrupted_copy_is_rejected_the_hub_is_used_and_sentinel_hears_of_it(
        stand_in, fleet, tmp_path, events):
    bad = bytearray(GOOD)
    bad[CHUNK * 4 + 9] ^= 0xFF
    honest = fleet.signer.entry_for(_write(tmp_path / "ref" / NAME, GOOD), kind="model",
                                    sharing="open", chunk_size=CHUNK)
    peer = fleet.peer("rotten", bytes(bad), entry=honest)
    got = hub.pull(REF, tmp_path / "out")
    assert got.read_bytes() == GOOD
    assert file_requests(stand_in) > 0
    assert any(e.kind == "onboard.peer.bad_copy" for e in events)
    assert any(e.kind == "onboard.transfer.bad_chunk" for e in events)
    assert not staging_left(tmp_path)                      # the partial bytes were discarded
    assert fleet.requests(peer)


def test_a_peer_that_sends_the_pinned_digest_with_other_bytes_is_caught_by_the_whole_file_hash(
        stand_in, fleet, tmp_path, events):
    bad = bytearray(GOOD)
    bad[100] ^= 0xFF
    # a signed entry claiming the Hub's digest, with chunk digests of the poisoned bytes
    wrong = fleet.signer.entry_for(_write(tmp_path / "ref" / NAME, bytes(bad)), kind="model",
                                   sharing="open", chunk_size=CHUNK)
    forged = mf.Entry(NAME, wrong.size, SHA, CHUNK, wrong.chunks, "model", "open")
    fleet.peer("liar", bytes(bad), entry=forged)
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert file_requests(stand_in) > 0
    assert any(e.kind == "onboard.transfer.bad_file" for e in events)


# (c) --------------------------------------------------------------------------------------
def test_a_never_file_is_not_asked_of_a_peer(stand_in, fleet, tmp_path):
    peer = fleet.peer("strict", sharing="never")
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert fleet.requests(peer) == [] and file_requests(stand_in) > 0


def test_an_owner_file_is_withheld_from_a_device_that_is_not_marked_the_owners(
        stand_in, fleet, tmp_path):
    peer = fleet.peer("gated", sharing="owner", licence="llama")
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert file_requests(stand_in) > 0 and fleet.requests(peer)        # asked, refused


def test_an_owner_file_goes_to_the_owners_device_once_the_licence_is_on_record(
        stand_in, fleet, tmp_path):
    peer = fleet.peer("mine", sharing="owner", licence="llama")
    device_secret = base64.urlsafe_b64encode(b"d" * 32).decode()
    me = Device("f" * 64, "laptop", "laptop.local", "127.0.0.1", 1.0, mine=True,
                secret=device_secret)
    Devices(peer["root"] / "state" / "devices.json")._write([me])
    peer["row"].pop("cluster_key")
    fleet.book.add({**peer["row"], "device_secret": device_secret})
    # no acceptance yet: the file stays withheld and the Hub is used
    assert hub.pull(REF, tmp_path / "first").read_bytes() == GOOD
    assert file_requests(stand_in) > 0
    stand_in.hub_seen.clear()
    stand_in.cdn_seen.clear()
    # the owner accepts the licence on the serving device (once, by a person: here a typed grant)
    Licences(peer["root"] / "state" / "licences.json").record(
        mint("accept-licence", "llama", typed=lambda _p: "llama", terminal=(True, True), env={}),
        peer["entry"], who="owner")
    got = hub.pull(REF, tmp_path / "second")
    assert got.read_bytes() == GOOD and file_requests(stand_in) == 0


def test_a_device_that_is_not_paired_is_refused_and_the_hub_is_used(stand_in, fleet, tmp_path):
    peer = fleet.peer("stranger", secret="not-the-cluster")
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert fleet.requests(peer) == [] and file_requests(stand_in) > 0


# (d) --------------------------------------------------------------------------------------
def test_a_cut_connection_falls_back_and_the_verified_chunks_resume_next_time(
        stand_in, fleet, tmp_path):
    cut = fleet.peer("flaky", env={"PEER_CUT_AFTER": "5"})
    got = hub.pull(REF, tmp_path / "out")
    assert got.read_bytes() == GOOD and file_requests(stand_in) > 0
    assert 5 <= len(fleet.requests(cut)) < 3 * MIB // CHUNK
    held = staging_left(tmp_path)
    assert any(p.name.endswith(".part.json") for p in held)           # kept for the next try
    got.unlink()
    fleet.close()
    fleet.rows.clear()
    healthy = fleet.peer("healthy")
    stand_in.hub_seen.clear()
    stand_in.cdn_seen.clear()
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert file_requests(stand_in) == 0
    assert len(fleet.requests(healthy)) < 3 * MIB // CHUNK            # only what was missing


# (e) --------------------------------------------------------------------------------------
def test_a_cancel_stops_the_peer_transfer_and_the_next_pull_continues_it(
        stand_in, fleet, tmp_path):
    peer = fleet.peer("kitchen")
    token = pulling.CancelToken()

    def stop(progress):
        if progress.done_bytes >= MIB:
            token.cancel()

    with pytest.raises(pulling.Cancelled):
        hub.pull(REF, tmp_path / "out", stop, token)
    assert file_requests(stand_in) == 0 and not (tmp_path / "out" / NAME).exists()
    first = len(fleet.requests(peer))
    assert 0 < first < 3 * MIB // CHUNK and staging_left(tmp_path)
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert file_requests(stand_in) == 0
    assert len(fleet.requests(peer)) - first < 3 * MIB // CHUNK - 1


# (f) --------------------------------------------------------------------------------------
def test_an_unreachable_peer_costs_nothing_but_a_note(stand_in, fleet, tmp_path):
    with socket.socket() as s:                                        # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    fleet.book.add({"name": "gone", "url": f"https://127.0.0.1:{port}", "min_serial": 0,
                    "signing_key": base64.b64encode(fleet.signer.public).decode(),
                    "certificate": tls.identity(tmp_path / "tlsgone", "peer").beacon,
                    "cluster_key": CLUSTER})
    started = time.monotonic()
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert time.monotonic() - started < 10 and file_requests(stand_in) > 0


def test_a_peer_that_stalls_is_given_up_on_within_a_bounded_time(
        stand_in, fleet, tmp_path, monkeypatch):
    monkeypatch.setattr(peerfirst, "ASK_TIMEOUT", 1.0)
    fleet.peer("stalled", env={"PEER_STALL": "1"})
    started = time.monotonic()
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert time.monotonic() - started < 15 and file_requests(stand_in) > 0


def test_a_peer_whose_certificate_is_not_the_pinned_one_is_not_trusted(
        stand_in, fleet, tmp_path):
    peer = fleet.peer("impostor")
    other = tls.identity(tmp_path / "other-tls", "peer").beacon
    fleet.book.add({**peer["row"], "certificate": other})
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert fleet.requests(peer) == [] and file_requests(stand_in) > 0


# (g) --------------------------------------------------------------------------------------
def test_a_digest_the_peer_states_cannot_replace_the_hubs(stand_in, fleet, tmp_path):
    other = blob(3 * MIB, 99)                    # a different file of the same name and size
    peer = fleet.peer("confident", other)        # its signed manifest honestly lists `other`
    assert peer["entry"].sha256 != SHA
    got = hub.pull(REF, tmp_path / "out")
    assert got.read_bytes() == GOOD              # the Hub's bytes, not the peer's
    assert fleet.requests(peer) == [] and file_requests(stand_in) > 0


def test_without_a_digest_in_the_listing_a_signed_manifest_supplies_one(fleet, tmp_path):
    peer = fleet.peer("kitchen")
    session = peerfirst.PeerSession(fleet.book, bus=Recorder().bus)
    final = tmp_path / "out" / NAME
    final.parent.mkdir()
    ok = session.fetch(hub_peers.Wanted("maker/thing-GGUF", NAME, 3 * MIB, ""), final,
                       cancelled=lambda: False, progress=lambda _n: None, phase=lambda _n: None)
    assert ok and final.read_bytes() == GOOD and fleet.requests(peer)


def test_a_manifest_older_than_one_already_seen_is_not_used(stand_in, fleet, tmp_path):
    peer = fleet.peer("replayer")
    fleet.book.add({**peer["row"], "min_serial": 8})              # it serves serial 7
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert fleet.requests(peer) == [] and file_requests(stand_in) > 0


# (h) --------------------------------------------------------------------------------------
def test_no_peers_skips_them(stand_in, fleet, tmp_path):
    peer = fleet.peer("kitchen")
    assert hub.pull(REF, tmp_path / "out", peers=False).read_bytes() == GOOD
    assert fleet.requests(peer) == [] and file_requests(stand_in) > 0


def test_the_environment_variable_skips_them(stand_in, fleet, tmp_path, monkeypatch):
    peer = fleet.peer("kitchen")
    monkeypatch.setenv(hub_peers.ENV, "1")
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert fleet.requests(peer) == [] and file_requests(stand_in) > 0


def test_the_machine_setting_skips_them(stand_in, fleet, tmp_path):
    peer = fleet.peer("kitchen")
    fleet.book.set_enabled(False)
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert fleet.requests(peer) == []


def test_the_fetch_command_has_the_flag(stand_in, fleet, tmp_path):
    from ml_stack.hub import cli

    peer = fleet.peer("kitchen")
    assert cli.main(["fetch", "--no-peers", REF]) == 0
    assert fleet.requests(peer) == [] and file_requests(stand_in) > 0


def test_the_peers_command_turns_it_off_and_refuses_a_public_address(tmp_path, capsys):
    import argparse

    def run(action, **more):
        base = {"action": action, "name": "", "url": "", "certificate": "", "signing_key": "",
                "device_secret": "", "state": str(tmp_path / "s"), "json": True}
        return cmd_peers(argparse.Namespace(**{**base, **more}))

    assert run("off") == 0 and not peerfirst.PeerBook(tmp_path / "s" / "peers.json").enabled
    assert run("add", name="evil", url="https://8.8.8.8:8443", signing_key="AAAA") == 2
    assert peerfirst.PeerBook(tmp_path / "s" / "peers.json").rows() == []


# (i) --------------------------------------------------------------------------------------
def test_a_copy_sentinel_holds_in_quarantine_is_not_served(
        stand_in, fleet, tmp_path, monkeypatch):
    peer = fleet.peer("kitchen")
    here = os.environ["ML_STACK_HOME"]
    monkeypatch.setenv("ML_STACK_HOME", str(peer["home"]))        # the serving device's sentinel
    held = _write(tmp_path / "held" / NAME, b"the copy")
    sentinel.default().store.quarantine(("artifact", f"download:{SHA}"), "flagged later",
                                        {"sha256": SHA}, Holding(path=held), actor="test")
    monkeypatch.setenv("ML_STACK_HOME", here)
    assert hub.pull(REF, tmp_path / "out").read_bytes() == GOOD
    assert file_requests(stand_in) > 0               # asked, refused, so the Hub was needed


# (j) --------------------------------------------------------------------------------------
def test_a_public_address_is_never_contacted(fleet, tmp_path):
    session = peerfirst.PeerSession(fleet.book)
    row = {"name": "far", "url": "https://8.8.8.8:8443", "signing_key": "AAAA", "min_serial": 0,
           "cluster_key": CLUSTER, "certificate": tls.identity(tmp_path / "t", "peer").beacon}
    started = time.monotonic()
    assert session._ask(row) is None
    assert time.monotonic() - started < 1.0
    assert any("public internet" in n for n in session.notes)


def test_the_address_policy_lets_the_private_ranges_through_and_stops_the_public_one():
    for fine in ("127.0.0.1", "192.168.1.20", "10.1.2.3", "100.101.102.103", "169.254.1.1"):
        lan.require_local(fine)
    with pytest.raises(lan.NotLocal):
        lan.require_local("1.1.1.1")


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
