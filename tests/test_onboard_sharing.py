"""Who may have which file: open, owner and never, with a real server, real signed requests and
a real devices ledger."""

import base64
import os

import pytest
from onboard_support import Clock, Recorder, info, requests

from ml_stack import macauth
from ml_stack.fleet.onboard import manifest as mf, sharing, transfer
from ml_stack.fleet.onboard.human import HumanRequired, mint
from ml_stack.fleet.onboard.requests import Requests
from ml_stack.fleet.onboard.sharing import Licences

CHUNK = 65536
PAYLOAD = os.urandom(CHUNK * 2 + 17)
CLUSTER = macauth.derive(b"c" * 32)
URL = "https://example.org/licence"


@pytest.fixture(autouse=True)
def needs_cryptography():
    pytest.importorskip("cryptography")


def pair(tmp_path, rec, fp, *, mine):
    """A paired device, through the real request flow; returns its file-request secret."""
    rq = requests(tmp_path, rec, Clock())
    secret = base64.urlsafe_b64encode(os.urandom(32)).decode()
    r = rq.submit(info(fp, name=f"dev-{fp[:2]}"), f"10.0.0.{int(fp[:2], 16) % 200 + 1}")
    rq.accept(r.id, mine=mine)
    rq.paired(r.id, shared_cluster_key=True, secret=secret)
    return macauth.derive(base64.urlsafe_b64decode(secret))


class World:
    def __init__(self, tmp_path):
        self.tmp, self.rec = tmp_path, Recorder()
        self.signer = mf.Signer.generate()
        self.share_dir = tmp_path / "share"
        self.share_dir.mkdir()
        (self.share_dir / "prog.whl").write_bytes(PAYLOAD)
        (self.share_dir / "gated.gguf").write_bytes(PAYLOAD)
        self.licences = Licences(tmp_path / "licences.json")

    def serve(self, **terms):
        entries = [self.signer.entry(self.share_dir / "prog.whl", kind="wheel"),
                   self.signer.entry(self.share_dir / "gated.gguf", kind="model", chunk_size=CHUNK,
                                     **({"sharing": "owner", "licence": "llama",
                                         "licence_url": URL} | terms))]
        raw = self.signer.sign(entries, serial=1)
        self.manifest = mf.verify(raw, self.signer.public)
        devices = Requests(self.tmp / "requests.json", bus=self.rec.bus).devices
        self.server = transfer.ShareServer(
            transfer.Share(self.share_dir, raw, self.manifest, self.licences),
            authenticate=transfer.mac_gate(CLUSTER, devices.all), ident=None,
            bus=self.rec.bus).start()
        return self

    def fetch(self, secret, name="gated.gguf"):
        peer = transfer.PeerSource(f"http://127.0.0.1:{self.server.port}", secret, name="owner")
        return transfer.Downloader(self.manifest, [peer], self.tmp / "stage",
                                   settings=transfer.Settings(reserve=0)).download(name)

    def accept_licence(self):
        entry = self.manifest.entry("gated.gguf")
        self.licences.record(mint("accept-licence", "llama", typed=lambda _p: "llama",
                                  terminal=(True, True), env={}), entry, who="owner")


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    yield w
    getattr(w, "server", None) and w.server.stop()


def test_a_device_the_owner_marked_theirs_gets_a_gated_file_once_the_licence_is_on_record(world):
    mine = pair(world.tmp, world.rec, "11" * 32, mine=True)
    world.serve()
    world.accept_licence()
    assert world.fetch(mine).read_bytes() == PAYLOAD


def test_another_persons_device_is_refused_a_gated_file_even_in_the_same_fleet(world):
    other = pair(world.tmp, world.rec, "22" * 32, mine=False)
    world.serve()
    world.accept_licence()
    with pytest.raises(transfer.Withheld, match="not marked theirs"):
        world.fetch(other)
    assert world.fetch(other, "prog.whl").read_bytes() == PAYLOAD      # programs are open
    withheld = world.rec.of("onboard.transfer.withheld")
    assert withheld and withheld[0].evidence["level"] == "owner"


def test_a_request_signed_only_with_the_cluster_key_has_no_device_and_is_refused(world):
    world.serve()
    world.accept_licence()
    with pytest.raises(transfer.Withheld):
        world.fetch(CLUSTER)
    assert world.fetch(CLUSTER, "prog.whl").read_bytes() == PAYLOAD


def test_without_the_licence_on_record_even_the_owners_device_is_refused(world):
    mine = pair(world.tmp, world.rec, "11" * 32, mine=True)
    world.serve()
    with pytest.raises(transfer.Withheld, match=r"acceptance .* not on record"):
        world.fetch(mine)


def test_a_record_for_another_licence_or_url_does_not_count(world):
    mine = pair(world.tmp, world.rec, "11" * 32, mine=True)
    world.serve()
    other = mf.Entry("x", 1, "0" * 64, CHUNK, ("0" * 64,), "model", "owner", "llama",
                     "https://elsewhere/licence")
    world.licences.record(mint("accept-licence", "llama", typed=lambda _p: "llama",
                               terminal=(True, True), env={}), other, who="owner")
    with pytest.raises(transfer.Withheld):
        world.fetch(mine)


def test_a_file_whose_licence_forbids_copies_goes_to_nobody(world):
    mine = pair(world.tmp, world.rec, "11" * 32, mine=True)
    world.serve(sharing="never")
    world.accept_licence()
    with pytest.raises(transfer.NotShareable):
        world.fetch(mine)
    import http.client
    url = f"http://127.0.0.1:{world.server.port}{transfer.API}/files/gated.gguf"
    conn = http.client.HTTPConnection("127.0.0.1", world.server.port, timeout=5)
    conn.request("GET", f"{transfer.API}/files/gated.gguf",
                 headers={"Range": "bytes=0-9", **macauth.sign(mine, "GET", url, None)})
    assert conn.getresponse().status == 403


def test_a_revoked_device_is_not_served_at_all(world):
    mine = pair(world.tmp, world.rec, "11" * 32, mine=True)
    world.serve()
    world.accept_licence()
    Requests(world.tmp / "requests.json", bus=world.rec.bus).devices.revoke("1111")
    with pytest.raises(transfer.TransferError):
        world.fetch(mine, "prog.whl")


@pytest.mark.parametrize("gated,forbids,level", [
    (None, None, "owner"), (True, None, "owner"), (None, False, "owner"),
    (False, None, "owner"), (False, False, "open"), (True, False, "owner"),
    (False, True, "never"), (None, True, "never"), (True, True, "never")])
def test_unknown_licence_status_is_treated_as_restricted(gated, forbids, level):
    assert sharing.classify(gated=gated, forbids_copies=forbids) == level


def test_recording_a_licence_needs_a_person(world):
    entry = mf.Entry("m", 1, "0" * 64, CHUNK, ("0" * 64,), "model", "owner", "llama", URL)
    with pytest.raises(HumanRequired):
        world.licences.record(mint("accept-licence", "llama", typed=lambda _p: "llama",
                                   terminal=(False, False), env={}), entry)
    with pytest.raises(HumanRequired):
        world.licences.record(mint("export", "llama", typed=lambda _p: "llama",
                                   terminal=(True, True), env={}), entry)
    assert world.licences.accepted(entry) is None


def test_the_devices_ledger_remembers_whose_device_it_is(tmp_path):
    rec = Recorder()
    pair(tmp_path, rec, "11" * 32, mine=True)
    pair(tmp_path, rec, "22" * 32, mine=False)
    ledger = Requests(tmp_path / "requests.json", bus=rec.bus).devices.all()
    assert {d.fingerprint[:2]: d.mine for d in ledger} == {"11": True, "22": False}
    assert (tmp_path / "devices.json").stat().st_mode & 0o077 == 0
