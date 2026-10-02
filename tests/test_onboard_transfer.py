"""Chunked, resumable downloads from peers that run as their own processes on loopback."""

import base64
import contextlib
import http.client
import os
import subprocess
import sys
from pathlib import Path

import pytest
from onboard_support import Recorder

from ml_stack import macauth
from ml_stack.fleet import tls
from ml_stack.fleet.onboard import (
    manifest as mf,
    transfer,
)

CHUNK = 65536
SIZE = CHUNK * 5 + 123              # six chunks, the last one short
SECRET = macauth.derive(b"k" * 32)
HELPER = Path(__file__).parent / "onboard_serve.py"
PAYLOAD = os.urandom(SIZE)


@pytest.fixture(autouse=True)
def needs_cryptography():
    pytest.importorskip("cryptography")


def downloader(manifest, peers, stage, *, reserve=0, on_staged=None, **cfg):
    return transfer.Downloader(manifest, peers, stage, on_staged=on_staged,
                               settings=transfer.Settings(reserve=reserve, **cfg))


@pytest.fixture
def signer():
    return mf.Signer.generate()


class Peers:
    """Peers started as processes; each has its own directory of files."""

    def __init__(self, tmp_path, signer):
        self.tmp, self.signer, self.procs = tmp_path, signer, []

    def make(self, name, *, files, entries, tls_dir=None):
        root = self.tmp / name
        root.mkdir()
        for fname, data in files.items():
            (root / fname).write_bytes(data)
        raw = self.signer.sign(entries, serial=1)
        manifest_file = self.tmp / f"{name}.manifest"
        manifest_file.write_bytes(raw)
        argv = [sys.executable, str(HELPER), str(root), str(manifest_file),
                base64.b64encode(self.signer.public).decode(), SECRET]
        if tls_dir:
            argv.append(str(tls_dir))
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        self.procs.append(proc)
        port = int(proc.stdout.readline())
        return transfer.PeerSource(f"http://127.0.0.1:{port}", SECRET, name=name), port, raw

    def close(self):
        for p in self.procs:
            with contextlib.suppress(OSError):
                p.stdin.close()
            p.wait(timeout=20)


@pytest.fixture
def peers(tmp_path, signer):
    p = Peers(tmp_path, signer)
    yield p
    p.close()


@pytest.fixture
def entry(tmp_path, signer):
    src = tmp_path / "source" / "ml_stack-0.2-py3-none-any.whl"
    src.parent.mkdir()
    src.write_bytes(PAYLOAD)
    return signer.entry(src, kind="wheel", chunk_size=CHUNK)


def manifest_of(signer, *entries):
    return mf.verify(signer.sign(entries, serial=1), signer.public)


def poisoned(entry, chunk=2):
    bad = bytearray(PAYLOAD)
    bad[chunk * CHUNK + 5] ^= 0xFF
    return bytes(bad)


def test_a_file_comes_from_two_peers_chunk_by_chunk_and_matches(tmp_path, signer, peers, entry):
    a, *_ = peers.make("a", files={entry.name: PAYLOAD}, entries=[entry])
    b, *_ = peers.make("b", files={entry.name: PAYLOAD}, entries=[entry])
    rec = Recorder()
    got = downloader(manifest_of(signer, entry), [a, b], tmp_path / "stage",
                              reserve=0, bus=rec.bus).download(entry.name)
    assert got.read_bytes() == PAYLOAD
    assert not list((tmp_path / "stage").glob("*.part*"))
    assert rec.kinds() == ["onboard.transfer.staged"]


def test_chunks_really_are_pulled_from_both_peers(tmp_path, signer, peers, entry):
    a, *_ = peers.make("a", files={entry.name: PAYLOAD}, entries=[entry])
    b, *_ = peers.make("b", files={entry.name: PAYLOAD}, entries=[entry])
    d = downloader(manifest_of(signer, entry), [a, b], tmp_path / "stage", reserve=0)
    d.download(entry.name)
    assert set(d.fetched_from) == {"a", "b"} and sum(d.fetched_from.values()) == SIZE


def test_a_peer_that_serves_a_poisoned_chunk_is_dropped_and_the_file_is_still_right(
        tmp_path, signer, peers, entry):
    bad, *_ = peers.make("bad", files={entry.name: poisoned(entry)}, entries=[entry])
    good, *_ = peers.make("good", files={entry.name: PAYLOAD}, entries=[entry])
    rec = Recorder()
    d = downloader(manifest_of(signer, entry), [bad, good], tmp_path / "stage",
                            reserve=0, strikes=1, bus=rec.bus, workers=1)
    assert d.download(entry.name).read_bytes() == PAYLOAD
    assert rec.of("onboard.transfer.bad_chunk")[0].severity == "critical"
    assert bad.banned and not good.banned


def test_when_every_peer_lies_nothing_is_staged(tmp_path, signer, peers, entry):
    bad, *_ = peers.make("bad", files={entry.name: poisoned(entry)}, entries=[entry])
    stage = tmp_path / "stage"
    with pytest.raises(transfer.TransferError, match="no peer could supply chunk 2"):
        downloader(manifest_of(signer, entry), [bad], stage, reserve=0).download(
            entry.name)
    assert not (stage / entry.name).exists()


def test_a_download_resumes_and_distrusts_a_damaged_partial_chunk(tmp_path, signer, peers,
                                                                  entry):
    a, *_ = peers.make("a", files={entry.name: PAYLOAD}, entries=[entry])
    stage = tmp_path / "stage"
    stage.mkdir()
    part = stage / f"{entry.name}.part"
    data = bytearray(SIZE)
    data[: 3 * CHUNK] = PAYLOAD[: 3 * CHUNK]                       # chunks 0-2 arrived earlier
    data[CHUNK + 7] ^= 0xFF                                        # ...and chunk 1 rotted
    part.write_bytes(bytes(data))
    (stage / f"{entry.name}.part.json").write_text(
        f'{{"schema_version": 1, "sha256": "{entry.sha256}", "done": [0, 1, 2]}}')
    d = downloader(manifest_of(signer, entry), [a], stage, reserve=0)
    assert d.download(entry.name).read_bytes() == PAYLOAD
    assert sum(d.fetched_from.values()) == SIZE - 2 * CHUNK       # chunks 0 and 2 were kept


def test_a_partial_file_for_a_different_manifest_is_discarded(tmp_path, signer, peers, entry):
    a, *_ = peers.make("a", files={entry.name: PAYLOAD}, entries=[entry])
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / f"{entry.name}.part").write_bytes(b"\0" * SIZE)
    (stage / f"{entry.name}.part.json").write_text(
        f'{{"schema_version": 1, "sha256": "{"9" * 64}", "done": [0, 1]}}')
    d = downloader(manifest_of(signer, entry), [a], stage, reserve=0)
    assert d.download(entry.name).read_bytes() == PAYLOAD
    assert sum(d.fetched_from.values()) == SIZE


def test_not_enough_disk_refuses_before_asking_a_peer(tmp_path, signer, peers, entry):
    a, *_ = peers.make("a", files={entry.name: PAYLOAD}, entries=[entry])
    d = downloader(manifest_of(signer, entry), [a], tmp_path / "stage",
                            reserve=1000, free=lambda p: SIZE + 999)
    with pytest.raises(transfer.TransferError, match="free"):
        d.download(entry.name)
    assert d.fetched_from == {}


def test_a_file_that_may_not_be_shared_is_neither_asked_for_nor_served(tmp_path, signer, peers):
    gated = mf.Entry("gated-model.gguf", SIZE, "1" * 64, CHUNK, ("2" * 64,) * 6, "model",
                      shareable=False, licence="gated", source="hf://org/gated")
    a, port, _ = peers.make("a", files={"gated-model.gguf": PAYLOAD}, entries=[gated])
    m = manifest_of(signer, gated)
    d = downloader(m, [a], tmp_path / "stage", reserve=0)
    with pytest.raises(transfer.NotShareable) as why:
        d.download("gated-model.gguf")
    assert why.value.source == "hf://org/gated"
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)    # and the server itself
    url = f"http://127.0.0.1:{port}{transfer.API}/files/gated-model.gguf"
    conn.request("GET", f"{transfer.API}/files/gated-model.gguf",
                 headers={"Range": "bytes=0-9", **macauth.sign(SECRET, "GET", url, None)})
    assert conn.getresponse().status == 403


def test_the_server_wants_a_signed_request_and_a_sane_range(tmp_path, signer, peers, entry):
    _, port, raw = peers.make("a", files={entry.name: PAYLOAD}, entries=[entry])

    def get(path, headers=None, signed=True):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        url = f"http://127.0.0.1:{port}{path}"
        sig = macauth.sign(SECRET, "GET", url, None) if signed else {}
        conn.request("GET", path, headers={**(headers or {}), **sig})
        r = conn.getresponse()
        return r.status, r.read()

    path = f"{transfer.API}/files/{entry.name}"
    assert get(path, {"Range": "bytes=0-9"}, signed=False)[0] == 401
    assert get(f"{transfer.API}/manifest") == (200, raw)
    status, body = get(path, {"Range": "bytes=10-19"})
    assert status == 206 and body == PAYLOAD[10:20]
    assert get(path, {"Range": "bytes=-5"})[0] == 400                     # no suffix ranges
    assert get(path, {"Range": "bytes=abc"})[0] == 400
    assert get(path, {"Range": f"bytes={SIZE + 10}-"})[0] == 416
    assert get(path, {"Range": f"bytes=0-{CHUNK * 4}"})[0] == 416          # not the whole file
    assert get(f"{transfer.API}/files/..%2Fsecret")[0] == 404
    assert get(f"{transfer.API}/files/other.whl")[0] == 404


def test_a_file_whose_size_changed_on_the_peer_is_not_served(tmp_path, signer, peers, entry):
    a, *_ = peers.make("a", files={entry.name: PAYLOAD + b"x"}, entries=[entry])
    d = downloader(manifest_of(signer, entry), [a], tmp_path / "stage", reserve=0)
    with pytest.raises(transfer.TransferError):
        d.download(entry.name)


def test_over_pinned_tls_and_a_peer_with_a_different_certificate_is_not_trusted(
        tmp_path, signer, peers, entry):
    ident = tls.identity(tmp_path / "tlsdir", "peer")
    p, port, _ = peers.make("p", files={entry.name: PAYLOAD}, entries=[entry],
                            tls_dir=tmp_path / "tlsdir")
    p.base_url = f"https://127.0.0.1:{port}"
    p.context = tls.pinned_context(ident.beacon)
    assert downloader(manifest_of(signer, entry), [p], tmp_path / "s1",
                               reserve=0).download(entry.name).read_bytes() == PAYLOAD
    stranger = tls.identity(tmp_path / "other", "stranger")
    p2 = transfer.PeerSource(p.base_url, SECRET, tls.pinned_context(stranger.beacon), "p2")
    with pytest.raises(transfer.TransferError):
        downloader(manifest_of(signer, entry), [p2], tmp_path / "s2",
                            reserve=0).download(entry.name)


def test_the_staged_file_is_handed_to_the_scan_hook_and_not_installed(tmp_path, signer, peers,
                                                                      entry):
    a, *_ = peers.make("a", files={entry.name: PAYLOAD}, entries=[entry])
    seen = []
    downloader(manifest_of(signer, entry), [a], tmp_path / "stage", reserve=0,
                        on_staged=lambda path, e: seen.append((path.name, e.kind))
                        ).download(entry.name)
    assert seen == [(entry.name, "wheel")]


def test_names_that_climb_out_are_refused_before_any_request(tmp_path, signer, entry):
    d = downloader(manifest_of(signer, entry), [transfer.PeerSource("http://x")],
                            tmp_path / "stage")
    from ml_stack.safenames import Unsafe
    with pytest.raises(Unsafe):
        d.download("../../etc/passwd")


def test_a_signed_entry_whose_whole_file_digest_disagrees_with_its_chunks_stages_nothing(
        tmp_path, signer, peers, entry):
    """Every chunk is right and the entry's own whole-file digest is wrong: the signer made a
    mistake or lied, and the finished file is not accepted."""
    odd = mf.Entry(entry.name, entry.size, "0" * 64, entry.chunk_size, entry.chunks, "wheel")
    a, *_ = peers.make("a", files={entry.name: PAYLOAD}, entries=[odd])
    stage = tmp_path / "stage"
    with pytest.raises(transfer.TransferError, match="does not hash"):
        downloader(manifest_of(signer, odd), [a], stage).download(entry.name)
    assert not (stage / entry.name).exists() and not list(stage.glob("*.part*"))
