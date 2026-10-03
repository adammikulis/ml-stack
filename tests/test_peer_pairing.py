"""Pairing fills the peer book, and the model store is served to the devices that may have it.

Real processes and files: the owner and the new machine are `python -m ml_stack.fleet.join`
processes with their own homes (as in test_onboard_cli), pairing is the real flow with the code
read out, ``share --models`` is a process serving a real folder of GGUF files over pinned TLS, and
the downloading side is the real `PeerSession`. Nothing is added with ``peers add``.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import socket
from pathlib import Path

import pytest
from onboard_support import Clock, Recorder, identity as make_identity, requests as make_requests
from test_hub_pull import blob
from test_onboard_cli import env_for, fleet as run_fleet, read_document, spawn, stop

from ml_stack import hub, sentinel
from ml_stack.fleet import discovery, tls
from ml_stack.fleet.onboard import modelstore, pake, peerfirst, peerlearn
from ml_stack.fleet.onboard.human import mint
from ml_stack.fleet.onboard.manifest import DEFAULT_CHUNK, Entry
from ml_stack.fleet.onboard.pairing import (
    API,
    Grant,
    Hooks,
    Offer,
    PairingClient,
    PairingServer,
    context_for,
)
from ml_stack.fleet.onboard.requests import Devices, Request
from ml_stack.fleet.onboard.sharing import Access, Licences
from ml_stack.fleet.onboard.transfer import Share, confined, serve_file
from ml_stack.hub import origins, peers as hub_peers
from ml_stack.hub.peerbook import PeerBook
from ml_stack.hub.places import Place
from ml_stack.safenames import Unsafe
from ml_stack.sentinel.store import Holding

KIB = 1024
FP = "a" * 64


@pytest.fixture(autouse=True)
def needs_crypto():
    pytest.importorskip("spake2")
    pytest.importorskip("cryptography")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# -- the offer is authenticated by the pairing -------------------------------------------------
@pytest.fixture
def world(tmp_path):
    rec = Recorder()
    acceptor, joiner = make_identity(tmp_path, "acceptor"), make_identity(tmp_path, "joiner")
    rq = make_requests(tmp_path, rec, Clock())
    learned = []
    server = PairingServer(
        rq, acceptor, Hooks(lambda r: Grant(certificate=acceptor.beacon, share_port=9),
                            learned=lambda r, o: learned.append((r, o))),
        address=("127.0.0.1", 0), bus=rec.bus).start()
    yield type("World", (), {"rq": rq, "server": server, "acceptor": acceptor,
                             "joiner": joiner, "learned": learned})
    server.stop()


def the_offer(joiner, port=8773) -> Offer:
    return Offer(port, joiner.beacon, base64.b64encode(b"k" * 32).decode(), "secret-for-them")


def joiner_client(world) -> PairingClient:
    return PairingClient("127.0.0.1", world.server.port, fingerprint=world.joiner.fingerprint)


def asked(world):
    c = joiner_client(world)
    rid = c.ask(name="kitchen-pi", hostname="kitchen.local", model="Linux")
    return c, rid, world.rq.accept(rid, mine=True).code


def test_the_accepting_side_hears_the_offer_the_asker_sealed_under_the_exchange(world):
    c, _rid, code = asked(world)
    grant = c.finish(code, the_offer(world.joiner))
    assert grant.share_port == 9
    [(request, offer)] = world.learned
    assert request.name == "kitchen-pi" and offer == the_offer(world.joiner)


def test_an_offer_whose_tag_is_not_the_exchanges_is_refused_and_nothing_is_learned(world):
    c, rid, code = asked(world)
    session = pake.start_initiator(code, context=context_for(rid, c.nonce),
                                   mine=world.joiner.fingerprint, theirs=world.acceptor.fingerprint)
    _s, body = c._call("POST", f"{API}/{rid}/exchange", {"message": session.message})
    session.receive(body["message"])
    raw = the_offer(world.joiner).encode()
    forged = base64.b64encode(raw).decode()
    status, answer = c._call("POST", f"{API}/{rid}/confirm", {
        "confirmation": session.confirmation(), "offer": forged, "offer_tag": session.seal(raw + b"x")})
    assert status == 403 and "tag" in answer["error"]
    assert world.learned == []


def test_a_tag_made_for_the_grant_does_not_authenticate_an_offer(world):
    c, rid, code = asked(world)
    session = pake.start_initiator(code, context=context_for(rid, c.nonce),
                                   mine=world.joiner.fingerprint, theirs=world.acceptor.fingerprint)
    _s, body = c._call("POST", f"{API}/{rid}/exchange", {"message": session.message})
    session.receive(body["message"])
    offer = the_offer(world.joiner).encode()
    status, _ = c._call("POST", f"{API}/{rid}/confirm", {
        "confirmation": session.confirmation(), "offer": base64.b64encode(offer).decode(),
        "offer_tag": session.seal(offer.removeprefix(b"offer:"))})
    assert status == 403 and world.learned == []


def request_from(joiner, **more) -> Request:
    base = {"id": "1" * 32, "name": "kitchen-pi", "hostname": "k", "address": "127.0.0.1",
            "model": "m", "fingerprint": joiner.fingerprint, "nonce": "n", "created": 1.0}
    return Request(**{**base, **more})


def test_an_offer_for_some_other_certificate_than_the_one_the_exchange_bound_is_not_stored(tmp_path):
    real, other = make_identity(tmp_path, "real"), make_identity(tmp_path, "other")
    request = request_from(real)
    good = peerlearn.learn_from_offer(tmp_path, request, the_offer(real))
    assert good and good["certificate"] == real.beacon and good["fingerprint"] == real.fingerprint
    assert peerlearn.learn_from_offer(tmp_path / "b", request, the_offer(other)) is None
    assert PeerBook(tmp_path / "b" / "peers.json").rows() == []


def test_an_offer_with_no_share_port_or_a_bad_key_gives_no_row(tmp_path):
    real = make_identity(tmp_path, "real")
    request = request_from(real)
    assert peerlearn.learn_from_offer(tmp_path, request, Offer()) is None
    bad_key = Offer(8773, real.beacon, "not a key", "s")
    assert peerlearn.learn_from_offer(tmp_path, request, bad_key) is None
    assert PeerBook(tmp_path / "peers.json").rows() == []


def test_a_grant_whose_certificate_is_not_the_one_presented_adds_no_row(tmp_path):
    real, other = make_identity(tmp_path, "real"), make_identity(tmp_path, "other")
    grant = Grant(certificate=other.beacon, signing_key=base64.b64encode(b"k" * 32).decode(),
                  device_secret="s", share_port=8773, name="owner")
    assert peerlearn.learn_from_grant(tmp_path, grant, host="127.0.0.1",
                                      server_fingerprint=real.fingerprint, my_secret="m") is None
    assert PeerBook(tmp_path / "peers.json").rows() == [] and Devices(
        tmp_path / "devices.json").all() == []


def test_two_devices_with_one_name_do_not_overwrite_each_other(tmp_path):
    a, b = make_identity(tmp_path, "a"), make_identity(tmp_path, "b")
    first = peerlearn.learn_from_offer(tmp_path, request_from(a), the_offer(a))
    second = peerlearn.learn_from_offer(tmp_path, request_from(b), the_offer(b))
    assert first["name"] == "kitchen-pi" and second["name"].startswith("kitchen-pi-")
    assert len(PeerBook(tmp_path / "peers.json").rows()) == 2


# -- two homes, the real flow ------------------------------------------------------------------
def start_owner(tmp_path, *listen_args):
    env = env_for(tmp_path / "owner")
    key_path = tmp_path / "owner" / "cluster.key"
    discovery.join("a-long-enough-passphrase", path=key_path,
                   salting=discovery.Salting(salt=b"s" * 16))
    state = tmp_path / "owner" / "state"
    port = free_port()
    proc = spawn(env, "listen", "--json", "--state", str(state), "--host", "127.0.0.1",
                 "--port", "0", "--no-announce", "--for", "180s", "--share-port", str(port),
                 *listen_args)
    started = json.loads(read_document(proc))
    return proc, type("Owner", (), {"env": env, "state": state, "port": started["port"],
                                    "share_port": port, "key_path": key_path,
                                    "home": tmp_path / "owner"})


def pair_new_box(owner, tmp_path, *pair_args, accept="--mine"):
    env = env_for(tmp_path / "newbox")
    state = tmp_path / "newbox" / "state"
    share_port = free_port()
    pair = spawn(env, "pair", "--host", "127.0.0.1", "--port", str(owner.port), "--json",
                 "--state", str(state), "--wait", "60s", "--name", "new-box",
                 "--share-port", str(share_port), *pair_args)
    try:
        seen = []
        for _ in range(100):
            seen = json.loads(run_fleet(owner.env, "requests", "--json", "--state",
                                        str(owner.state)).stdout)["requests"]
            if seen:
                break
        code = json.loads(run_fleet(owner.env, "accept", seen[0]["id"][:8], accept, "--json",
                                    "--state", str(owner.state)).stdout)["code"]
        pair.stdin.write(code + "\n")
        pair.stdin.flush()
        _, err = pair.communicate(timeout=60)
        assert pair.returncode == 0, err
    finally:
        pair.kill()
    return type("Box", (), {"env": env, "state": state, "share_port": share_port,
                            "home": tmp_path / "newbox"})


@pytest.fixture
def paired(tmp_path):
    proc, owner = start_owner(tmp_path)
    box = pair_new_box(owner, tmp_path)
    yield owner, box
    stop(proc)


def row_of(state: Path) -> dict:
    [row] = PeerBook(state / "peers.json").rows()
    return row


def signing_public(state: Path) -> str:
    return json.loads((state / "signing.json").read_text())["public"]


def test_pairing_alone_fills_the_peer_book_of_both_devices(paired):
    owner, box = paired
    on_owner, on_box = row_of(owner.state), row_of(box.state)
    new_ident = tls.identity(box.state / "tls", "x")
    owner_ident = tls.identity(owner.state / "tls", "x")
    assert on_owner["name"] == "new-box" and on_owner["source"] == "pairing"
    assert on_owner["url"] == f"https://127.0.0.1:{box.share_port}"
    assert on_owner["certificate"] == new_ident.beacon
    assert on_owner["fingerprint"] == new_ident.fingerprint
    assert on_owner["signing_key"] == signing_public(box.state) and on_owner["device_secret"]
    assert on_box["url"] == f"https://127.0.0.1:{owner.share_port}"
    assert on_box["certificate"] == owner_ident.beacon
    assert on_box["fingerprint"] == owner_ident.fingerprint
    assert on_box["signing_key"] == signing_public(owner.state)
    owner_devices = json.loads((owner.state / "devices.json").read_text())["devices"]
    assert on_box["device_secret"] == owner_devices[0]["secret"]     # what the owner expects from it
    box_devices = json.loads((box.state / "devices.json").read_text())["devices"]
    assert on_owner["device_secret"] == box_devices[0]["secret"]      # and the other way round
    assert oct((owner.state / "peers.json").stat().st_mode & 0o777) == "0o600"


def test_revoking_a_device_removes_it_from_the_peer_book(paired):
    owner, box = paired
    assert row_of(owner.state)["name"] == "new-box"
    done = run_fleet(owner.env, "revoke", "new-box", "--json", "--state", str(owner.state))
    assert json.loads(done.stdout)["peers_removed"] == ["new-box"]
    assert PeerBook(owner.state / "peers.json").rows() == []
    assert len(PeerBook(box.state / "peers.json").rows()) == 1       # the other side is its own call


def test_a_device_that_shares_nothing_has_no_share_address_recorded(tmp_path):
    proc, owner = start_owner(tmp_path)
    try:
        box = pair_new_box(owner, tmp_path, "--no-share")
        assert PeerBook(owner.state / "peers.json").rows() == []        # it shares nothing
        assert row_of(box.state)["url"].endswith(f":{owner.share_port}")
    finally:
        stop(proc)


def test_an_owner_that_shares_nothing_leaves_no_row_on_the_new_machine(tmp_path):
    proc, owner = start_owner(tmp_path, "--no-share")
    try:
        box = pair_new_box(owner, tmp_path)
        assert PeerBook(box.state / "peers.json").rows() == []
        assert row_of(owner.state)["name"] == "new-box"
        assert len(json.loads((box.state / "devices.json").read_text())["devices"]) == 1
    finally:
        stop(proc)


# -- the model store ---------------------------------------------------------------------------
class Models:
    """A folder of model files with one of each kind of thing a store has to get right."""

    def __init__(self, tmp: Path) -> None:
        self.root = tmp / "models"
        self.outside = tmp / "outside"
        self.outside.mkdir()
        # sizes differ: discovery treats two files of one size and header as one model's copies
        self.data = {"open-Q4.gguf": blob(300 * KIB, 1), "gated-Q4.gguf": blob(310 * KIB, 2),
                     "unaccepted-Q4.gguf": blob(320 * KIB, 3),
                     "deep/er/nested-Q8.gguf": blob(200 * KIB, 4), "held-Q4.gguf": blob(210 * KIB, 5)}
        for name, data in self.data.items():
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_bytes(data)
        self.secret = blob(330 * KIB, 9)
        (self.outside / "secret.gguf").write_bytes(self.secret)
        (self.root / "escape.gguf").symlink_to(self.outside / "secret.gguf")

    def wanted(self, name: str) -> hub_peers.Wanted:
        data = self.data[name]
        return hub_peers.Wanted("maker/thing-GGUF", name.rsplit("/", 1)[-1], len(data),
                                hashlib.sha256(data).hexdigest())


def quarantine(home: Path, data: bytes, monkeypatch, tmp: Path) -> None:
    here = os.environ["ML_STACK_HOME"]
    monkeypatch.setenv("ML_STACK_HOME", str(home))
    digest = hashlib.sha256(data).hexdigest()
    held = tmp / "held-copy" / "x.gguf"
    held.parent.mkdir(parents=True, exist_ok=True)
    held.write_bytes(b"the copy")
    sentinel.default().store.quarantine(("artifact", f"download:{digest}"), "flagged",
                                        {"sha256": digest}, Holding(path=held), actor="test")
    monkeypatch.setenv("ML_STACK_HOME", here)


def accept_licence(owner, name: str, who: str = "adam") -> None:
    entry = Entry(name.rsplit("/", 1)[-1], 1, "0" * 64, 65536, ("0" * 64,), licence=name.rsplit("/", 1)[-1])
    Licences(owner.state / "licences.json").record(
        mint("accept-licence", entry.licence, typed=lambda _p: entry.licence,
             terminal=(True, True), env={}), entry, who=who)


@pytest.fixture
def store(tmp_path, paired, monkeypatch):
    owner, box = paired
    models = Models(tmp_path)
    accept_licence(owner, "gated-Q4.gguf")
    quarantine(owner.home, models.data["held-Q4.gguf"], monkeypatch, tmp_path)
    sharing = spawn(owner.env, "share", "--models", "--models-dir", str(models.root),
                    "--sharing", "open-Q4.gguf=open", "--sharing", "nested-Q8.gguf=open",
                    "--host", "127.0.0.1", "--port", str(owner.share_port), "--json",
                    "--state", str(owner.state), "--for", "180s")
    started = json.loads(read_document(sharing))
    session = peerfirst.PeerSession(PeerBook(box.state / "peers.json"), staging=tmp_path / "stage",
                                    origins_path=box.state / "peer-downloads.jsonl")

    def get(name: str):
        final = tmp_path / "got" / name.rsplit("/", 1)[-1]
        final.parent.mkdir(exist_ok=True)
        ok = session.fetch(models.wanted(name), final, cancelled=lambda: False,
                           progress=lambda _n: None, phase=lambda _p: None)
        return ok, final

    yield type("Store", (), {"owner": owner, "box": box, "models": models, "doc": started,
                             "session": session, "get": staticmethod(get)})
    stop(sharing)


def test_the_model_store_lists_what_discovery_finds_and_leaves_out_what_it_must(store):
    listed = {f["name"]: f for f in store.doc["files"]}
    assert set(listed) == {"open-Q4.gguf", "gated-Q4.gguf", "unaccepted-Q4.gguf", "nested-Q8.gguf"}
    assert listed["open-Q4.gguf"]["sharing"] == "open"
    assert listed["gated-Q4.gguf"]["sharing"] == "owner" and listed["gated-Q4.gguf"]["licence_recorded"]
    assert not listed["unaccepted-Q4.gguf"]["licence_recorded"]
    skipped = " ".join(store.doc["skipped"])
    assert "held-Q4.gguf" in skipped and "quarantine" in skipped


def test_an_open_model_in_a_subfolder_comes_from_the_store_by_name_and_digest(store):
    ok, final = store.get("open-Q4.gguf")
    assert ok and final.read_bytes() == store.models.data["open-Q4.gguf"]
    ok, final = store.get("deep/er/nested-Q8.gguf")
    assert ok and final.read_bytes() == store.models.data["deep/er/nested-Q8.gguf"]


def test_the_manifest_is_signed_and_its_serial_rises(tmp_path):
    first = modelstore.next_serial(tmp_path / "s.json", now=lambda: 1_000.0)
    second = modelstore.next_serial(tmp_path / "s.json", now=lambda: 1_000.0)
    earlier_clock = modelstore.next_serial(tmp_path / "s.json", now=lambda: 5.0)
    assert first == 1_000 and second == 1_001 and earlier_clock == 1_002


def test_a_file_a_symlink_points_out_of_the_roots_is_never_listed_or_served(store, tmp_path):
    models = store.models
    ok, _final = store.get("open-Q4.gguf")           # the store works...
    assert ok
    session = store.session
    leak = hub_peers.Wanted("x/y", "escape.gguf", len(models.secret),
                            hashlib.sha256(models.secret).hexdigest())
    assert session.fetch(leak, tmp_path / "leak.gguf", cancelled=lambda: False,
                         progress=lambda _n: None, phase=lambda _p: None) is False
    assert not (tmp_path / "leak.gguf").exists()


def test_a_gated_model_goes_to_the_owners_own_device_once_the_licence_is_on_record(store):
    ok, final = store.get("gated-Q4.gguf")
    assert ok and final.read_bytes() == store.models.data["gated-Q4.gguf"]
    line = origins.latest("gated-Q4.gguf", len(final.read_bytes()), store.box.state / "peer-downloads.jsonl")
    assert line["relies_on"]["accepted_by"] == "adam" and line["relies_on"]["licence"] == "gated-Q4.gguf"
    assert line["sha256"] == hashlib.sha256(final.read_bytes()).hexdigest() and line["peer"]


def test_a_gated_model_whose_licence_the_owner_never_accepted_is_withheld(store):
    ok, final = store.get("unaccepted-Q4.gguf")
    assert not ok and not final.exists()


def test_a_gated_model_is_withheld_from_a_device_that_is_not_marked_the_owners(store):
    path = store.owner.state / "devices.json"
    doc = json.loads(path.read_text())
    for d in doc["devices"]:
        d["mine"] = False
    path.write_text(json.dumps(doc))
    ok, _final = store.get("gated-Q4.gguf")
    assert not ok
    ok, final = store.get("open-Q4.gguf")           # an open model still goes to any paired device
    assert ok and final.exists()


def test_ml_stack_models_list_shows_where_a_peer_model_came_from_and_whose_acceptance(
        store, tmp_path, monkeypatch, capsys):
    ok, final = store.get("gated-Q4.gguf")
    assert ok
    monkeypatch.setenv("ML_STACK_HOME", str(store.box.home))
    here = final.parent
    shown = Place("ml-stack", "flat", here, True, 4)
    monkeypatch.setattr(hub, "standard", lambda roots=None: [shown])
    origins.record({"file": final.name, "size": final.stat().st_size, "peer": "owner-box",
                    "relies_on": {"device": "owner-box", "accepted_by": "adam"}})
    from ml_stack.serve import models_cli
    assert models_cli.run(["list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    [row] = [r for r in rows if r["path"].endswith("gated-Q4.gguf")]
    assert row["from_peer"]["peer"] == "owner-box"
    assert models_cli.run(["list"]) == 0
    assert "peer owner-box, licence accepted by adam on owner-box" in capsys.readouterr().out


# -- store-root confinement, quarantine and levels at the serving function -----------------------
def share_of(models: Models, names: list[str], *, withhold=None, licences=None, level="open"):
    entries = [Entry(n.rsplit("/", 1)[-1], len(models.data[n]), hashlib.sha256(models.data[n]).hexdigest(),
                     DEFAULT_CHUNK, (hashlib.sha256(models.data[n]).hexdigest(),), kind="model",
                     sharing=level) for n in names]
    paths = {e.name: models.root / n for e, n in zip(entries, names, strict=True)}
    from ml_stack.fleet.onboard.manifest import Manifest
    manifest = Manifest(1, 0.0, 0.0, "", tuple(entries))
    return Share(models.root, b"", manifest, licences, withhold, paths, (models.root,)), entries


class Bus:
    def emit(self, *_a, **_k):
        return None


MINE = Access("f" * 64, True)


def serve(share, entry, access=MINE):
    return serve_file(share, entry.name, "bytes=0-9", Bus(), access)


def test_serving_follows_only_paths_that_stay_inside_the_model_roots(tmp_path):
    models = Models(tmp_path)
    share, [entry] = share_of(models, ["open-Q4.gguf"])
    assert serve(share, entry).status == 206
    # swapped for a link out of the roots after the manifest was made, same size: refused
    target = models.outside / "same-size.gguf"
    target.write_bytes(blob(len(models.data["open-Q4.gguf"]), 77))
    (models.root / "open-Q4.gguf").unlink()
    (models.root / "open-Q4.gguf").symlink_to(target)
    refused = serve(share, entry)
    assert refused.status == 404 and b"here" in refused.body
    # a link that stays inside is fine (a Hugging Face cache is links into blobs/)
    (models.root / "blobs").mkdir()
    (models.root / "blobs" / "b").write_bytes(models.data["open-Q4.gguf"])
    (models.root / "open-Q4.gguf").unlink()
    (models.root / "open-Q4.gguf").symlink_to(models.root / "blobs" / "b")
    assert serve(share, entry).status == 206


def test_a_name_with_dots_or_a_path_is_no_way_out_of_the_roots(tmp_path):
    models = Models(tmp_path)
    share, _entries = share_of(models, ["open-Q4.gguf"])
    for name in ("../outside/secret.gguf", "..%2foutside%2fsecret.gguf", "/etc/hosts",
                 "deep/er/nested-Q8.gguf", "escape.gguf", "secret.gguf"):
        assert serve_file(share, name, "bytes=0-9", Bus(), Access("f" * 64, True)).status == 404
    with pytest.raises(Unsafe):
        confined(models.root / ".." / "outside" / "secret.gguf", (models.root,))
    with pytest.raises(Unsafe):
        confined(models.root / "escape.gguf", (models.root,))
    assert confined(models.root / "open-Q4.gguf", (models.root,)).name == "open-Q4.gguf"


def test_a_file_that_is_not_listed_cannot_be_asked_for_by_giving_it_an_entry_name(tmp_path):
    models = Models(tmp_path)
    share, [entry] = share_of(models, ["open-Q4.gguf"])
    other = Entry("escape.gguf", entry.size, entry.sha256, entry.chunk_size, entry.chunks)
    assert serve_file(share, other.name, "bytes=0-9", Bus(), Access("f" * 64, True)).status == 404


def test_a_copy_sentinel_holds_is_not_served_even_after_the_manifest_was_made(
        tmp_path, monkeypatch):
    models = Models(tmp_path)
    share, [entry] = share_of(models, ["open-Q4.gguf"], withhold=peerfirst.quarantine_veto)
    assert serve(share, entry).status == 206
    quarantine(tmp_path / "home", models.data["open-Q4.gguf"], monkeypatch, tmp_path)
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
    held = serve(share, entry)
    assert held.status == 403 and b"quarantine" in held.body


def test_building_the_store_skips_a_held_copy_a_link_out_and_a_second_file_of_one_name(
        tmp_path, monkeypatch):
    models = Models(tmp_path)
    twin = models.root / "other" / "open-Q4.gguf"
    twin.parent.mkdir()
    twin.write_bytes(blob(10 * KIB, 5))
    quarantine(tmp_path / "home", models.data["held-Q4.gguf"], monkeypatch, tmp_path)
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
    found, roots = modelstore.candidates([models.root])
    found.append(modelstore.Candidate("escape.gguf", models.root / "escape.gguf",
                                      len(models.secret), ""))          # whatever found it
    built = modelstore.build(found, roots, digests=modelstore.Digests(None),
                             veto=peerfirst.quarantine_veto)
    names = [e.name for e in built.entries]
    assert "held-Q4.gguf" not in names and "escape.gguf" not in names
    assert names.count("open-Q4.gguf") == 1
    why = " ".join(built.skipped)
    assert "quarantine" in why and "outside" in why and "already listed" in why
    assert set(built.paths) == set(names)
    assert all(e.sharing == "owner" and e.licence for e in built.entries)     # the default level


def test_a_gated_file_is_served_to_the_owners_devices_only_with_the_acceptance_on_record(tmp_path):
    models = Models(tmp_path)
    licences = Licences(tmp_path / "licences.json")
    share, [entry] = share_of(models, ["gated-Q4.gguf"], licences=licences, level="owner")
    entry = Entry(**{**{f: getattr(entry, f) for f in entry.__slots__}, "licence": "llama"})
    share = Share(share.root, b"", share.manifest.__class__(1, 0.0, 0.0, "", (entry,)), licences,
                  None, share.paths, share.roots)
    mine, other = Access("f" * 64, True), Access("e" * 64, False)
    assert serve(share, entry, mine).status == 403                   # no acceptance yet
    licences.record(mint("accept-licence", "llama", typed=lambda _p: "llama",
                         terminal=(True, True), env={}), entry, who="adam")
    assert serve(share, entry, other).status == 403                  # another person's device
    assert serve(share, entry, Access()).status == 403                # the cluster key alone
    ok = serve(share, entry, mine)
    assert ok.status == 206 and ok.headers["X-Licence-Accepted-By"] == "adam"


def test_the_share_command_needs_a_cluster_and_something_to_share(tmp_path):
    env = env_for(tmp_path / "lonely")
    refused = run_fleet(env, "share", "--json", "--state", str(tmp_path / "s"))
    assert refused.returncode == 2
    with contextlib.suppress(OSError):
        assert "needs a cluster" in json.loads(refused.stdout)["error"]
