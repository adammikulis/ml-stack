"""``ml-stack-fleet plan``: which model each peer serves, and how many seats, for N users.

Profiles, memory records and peers are invented; the command is driven through `join.main`
against daemons on loopback, and ``--apply`` against the real daemon handler over a
llama-server that is a script. Every name is invented.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from ml_stack.fleet import join as joining
from ml_stack.fleet.daemon import JobRunner, load_or_create_token, make_handler
from ml_stack.fleet.discovery import Advertiser, Beacon, join as join_cluster, load_cluster_key
from ml_stack.fleet.join import main
from ml_stack.fleet.models import Models
from ml_stack.fleet.plan import Room, fit_for, place, ranked, room_of, table
from ml_stack.fleet.remote import Peer, PeerError
from ml_stack.fleet.serving import Hosting, NoRoom, Serving
from ml_stack.serve.fit import Fit
from ml_stack.serve.profile import Profile
from test_fleet_join import WORDS, FakeDaemon, _free_tcp, _free_udp
from test_fleet_serving import FAKE_SERVER

G = 1 << 30
M = 1 << 20

BIG = "quince-70b-Q4_K_M.gguf"
MID = "larch-9b-Q4_K_M.gguf"
SMALL = "quince-2b-Q4_K_M.gguf"
SHINY = "thornfell-1b-Q4_K_M.gguf"

PROFILES = [
    Profile(model=MID, questions=100, right=0.5, seconds_per_question=3.0),
    Profile(model=BIG, questions=100, right=0.8, seconds_per_question=20.0),
    Profile(model=SMALL, questions=100, right=0.3, seconds_per_question=1.0),
    Profile(model=SHINY, questions=5, right=0.99, seconds_per_question=0.5),
]

# loaded = weights_gpu + compute; a seat at 16384 tokens costs per_token * 16384 + per_seq
FITS = [
    Fit(model=BIG, weights_gpu=40 * G, compute=1 * G, per_token=1 * M, per_seq=0),
    Fit(model="larch-9b-IQ4_XS.gguf", weights_gpu=8 * G, compute=1 * G, per_token=256 * 1024,
        per_seq=0),
    Fit(model=SMALL, weights_gpu=2 * G, compute=G // 2, per_token=64 * 1024, per_seq=0),
    Fit(model=SHINY, weights_gpu=1 * G, compute=G // 4, per_token=64 * 1024, per_seq=0),
]

ROOMY = Room("roomy", 96 * G)
SMALLER = Room("small", 24 * G)
TINY = Room("tiny", 8 * G)


class TestRanking:
    def test_best_f1_first_then_faster_and_thin_records_last(self):
        assert [p.model for p in ranked(PROFILES)] == [BIG, MID, SMALL, SHINY]

    def test_a_fit_is_found_by_name_then_by_family(self):
        assert fit_for(BIG, FITS).model == BIG
        assert fit_for(MID, FITS).model == "larch-9b-IQ4_XS.gguf"
        assert fit_for("nothing-3b.gguf", FITS) is None

    def test_among_several_the_measured_cache_type_wins(self):
        fits = [Fit(model=BIG, cache_type="q8_0", per_token=1),
                Fit(model=BIG, cache_type="f16", per_token=2)]
        assert fit_for(BIG, fits).per_token == 2
        assert fit_for(BIG, fits, cache_type="q8_0").per_token == 1

    def test_a_room_is_read_off_a_beacon_a_row_and_a_health_body(self):
        beacon = Beacon(name="roomy", port=8770, host="10.0.0.2",
                        device={"room_bytes": 96 * G,
                                "serving": [{"port": 1, "models": [BIG]}]})
        assert room_of(beacon) == Room("roomy", 96 * G, "http://10.0.0.2:8770", (BIG,))
        assert room_of(joining.describe(beacon)).room == 96 * G
        assert room_of({"name": "small", "room_bytes": 24 * G}) == Room("small", 24 * G)


class TestPlacing:
    def test_three_users_all_get_the_best_model_on_the_roomy_peer(self):
        got = place(3, 16384, [SMALLER, ROOMY], PROFILES, FITS)
        assert [(r.peer, r.model, r.seats) for r in got.rows] == [("roomy", BIG, 3)]
        assert got.unplaced == 0 and got.seated == 3
        assert got.rows[0].used == 41 * G + 3 * 16 * G and got.rows[0].room == 96 * G

    def test_more_users_than_the_best_model_seats_go_to_a_smaller_model(self):
        got = place(5, 16384, [SMALLER, ROOMY], PROFILES, FITS)
        assert [(r.peer, r.model, r.seats) for r in got.rows] == [
            ("roomy", BIG, 3), ("small", MID, 2)]
        assert got.unplaced == 0
        assert ("small", BIG, "room 24.0G < 41.0G loaded") in got.why

    def test_a_tiny_peer_gets_the_small_model(self):
        got = place(7, 16384, [TINY, SMALLER, ROOMY], PROFILES, FITS)
        assert [(r.peer, r.model, r.seats) for r in got.rows] == [
            ("roomy", BIG, 3), ("small", MID, 3), ("tiny", SMALL, 1)]
        assert got.unplaced == 0

    def test_nobody_is_dropped_silently(self):
        said: list[str] = []
        got = place(10, 16384, [SMALLER, ROOMY], PROFILES, FITS, log=said.append)
        assert got.seated == 6 and got.unplaced == 4
        assert any("4 user(s) without a seat" in line for line in said)
        text = table(got)
        assert "4 user(s) without a seat" in text
        assert "small: quince-70b-Q4_K_M.gguf: room 24.0G < 41.0G loaded" in text

    def test_a_shorter_context_seats_everyone_on_one_peer(self):
        long = place(5, 16384, [SMALLER, ROOMY], PROFILES, FITS)
        short = place(5, 4096, [SMALLER, ROOMY], PROFILES, FITS)
        assert [r.peer for r in long.rows] == ["roomy", "small"]
        assert [(r.peer, r.seats) for r in short.rows] == [("roomy", 5)]
        assert short.rows[0].context == 4096

    def test_a_peer_with_no_room_figure_is_named(self):
        got = place(1, 16384, [Room("blank", 0)], PROFILES, FITS)
        assert got.rows == [] and got.unplaced == 1
        assert ("blank", BIG, "room unknown") in got.why

    def test_a_model_with_no_memory_record_is_named(self):
        got = place(1, 16384, [ROOMY], [Profile(model="nothing-3b.gguf", questions=50,
                                                right=0.9)], FITS)
        assert got.rows == [] and ("*", "nothing-3b.gguf", "no memory measurement") in got.why

    def test_as_dict_carries_rows_and_reasons(self):
        got = place(5, 16384, [SMALLER, ROOMY], PROFILES, FITS)
        d = got.as_dict()
        assert d["seated"] == 5 and d["unplaced"] == 0
        assert d["rows"][0]["peer"] == "roomy" and d["rows"][0]["seats"] == 3
        assert {"peer": "small", "model": BIG, "reason": "room 24.0G < 41.0G loaded"} in d["why"]


# -- the command --------------------------------------------------------------------
@pytest.fixture
def udp() -> int:
    return _free_udp()


@pytest.fixture
def key(tmp_path) -> Path:
    path = tmp_path / "cluster.key"
    join_cluster(WORDS, group="home", path=path)
    return path


@pytest.fixture
def daemons():
    started: list = []
    yield started
    for d in started:
        d.close()


@pytest.fixture
def measured(monkeypatch):
    monkeypatch.setattr(joining, "_measurements", lambda: (list(PROFILES), list(FITS)))


def _device(room: int) -> dict:
    return {"gpu": "Pellard P40", "room_bytes": room, "serving": [], "models": []}


class TestCommand:
    def test_it_prints_the_split_as_json(self, key, udp, daemons, measured, monkeypatch,
                                         capsys):
        raw = load_cluster_key(key)
        roomy, small = _free_tcp(), _free_tcp()
        daemons.append(FakeDaemon(roomy, raw, udp, name="roomy", device=_device(96 * G)))
        daemons.append(FakeDaemon(small, raw, udp, name="small", device=_device(24 * G)))
        monkeypatch.setenv("ML_STACK_DISCOVERY_PORT", str(udp))
        code = main(["--cluster-key", str(key), "--port", str(roomy), "plan", "--users", "5",
                     "--context", "16384", "--json", "--timeout", "1"])
        got = json.loads(capsys.readouterr().out)
        assert code == 0
        assert [(r["peer"], r["model"], r["seats"]) for r in got["rows"]] == [
            ("roomy", BIG, 3), ("small", MID, 2)]
        assert got["unplaced"] == 0 and got["applied"] == []

    def test_it_prints_a_table_and_says_who_is_without_a_seat(
            self, key, udp, daemons, measured, monkeypatch, capsys):
        raw = load_cluster_key(key)
        small = _free_tcp()
        daemons.append(FakeDaemon(small, raw, udp, name="small", device=_device(24 * G)))
        monkeypatch.setenv("ML_STACK_DISCOVERY_PORT", str(udp))
        code = main(["--cluster-key", str(key), "--port", str(small), "plan", "--users", "4",
                     "--timeout", "1"])
        out = capsys.readouterr().out
        assert code == 1
        assert "PEER" in out and "small" in out and MID in out
        assert "3 of 4 user(s) seated at 16384 tokens each" in out
        assert "1 user(s) without a seat" in out
        assert f"small: {BIG}: room 24.0G < 41.0G loaded" in out

    def test_in_no_cluster_it_says_join(self, tmp_path, capsys):
        assert main(["--cluster-key", str(tmp_path / "none.key"), "plan", "--users", "1"]) == 1
        assert "ml-stack-fleet join" in capsys.readouterr().err


# -- POST /serve, and --apply -------------------------------------------------------------
@pytest.fixture
def fake_llama_server(tmp_path, monkeypatch):
    from ml_stack.serve import backend as backend_module

    monkeypatch.setattr(backend_module, "LOG_DIR", tmp_path / "logs")
    script = tmp_path / "server.py"
    script.write_text(FAKE_SERVER)
    binary = tmp_path / "llama-server"
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    binary.chmod(0o755)
    return binary


class ServingDaemon:
    """The real daemon handler on loopback with a model store of one file, a `Hosting`
    over a llama-server that is a script, and a beacon saying how much room it has."""

    def __init__(self, tmp_path: Path, binary: Path, *, name: str, room: int,
                 key: bytes | None = None, udp: int = 0, fits=None) -> None:
        from ml_stack.serve import LlamaServerBackend, ServerManager

        root = tmp_path / name
        (root / "files").mkdir(parents=True)
        (root / "models").mkdir()
        self.model = root / "models" / MID
        self.model.write_bytes(b"GGUF" + b"\x00" * (1 << 20))
        self.token = load_or_create_token(root, key)
        self.runner = JobRunner(root)
        self.serving = Serving(root / "serving.json")
        manager = ServerManager(LlamaServerBackend(binary=binary),
                                state_file=root / "servers.json")
        self.hosting = Hosting(root, self.serving, manager=manager,
                               fits=fits or (lambda: list(FITS)))
        self.models = Models([root / "models"], root / "store")
        self.port = _free_tcp()
        self.httpd = ThreadingHTTPServer(
            ("127.0.0.1", self.port),
            make_handler(self.runner, root / "files", self.token, name,
                         report=lambda: {"room_bytes": room}, serving=self.serving,
                         models=self.models, hosting=self.hosting))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.client = Peer(f"http://127.0.0.1:{self.port}", self.token)
        self.advertiser = None
        if key is not None:
            self.advertiser = Advertiser(
                Beacon(name=name, port=self.port, device=_device(room)), key, port=udp,
                interval_s=0.2).start()

    def close(self) -> None:
        if self.advertiser is not None:
            self.advertiser.stop()
        for port in list(self.hosting.leases):
            self.hosting.stop(port)
        self.runner.shutdown()
        self.httpd.shutdown()
        self.httpd.server_close()


class TestServeRoute:
    def test_it_serves_a_model_this_machine_holds_with_its_seats(
            self, tmp_path, fake_llama_server, daemons):
        d = ServingDaemon(tmp_path, fake_llama_server, name="small", room=24 * G)
        daemons.append(d)
        served = d.client._json("POST", "/serve", {"model": MID, "context": 16384,
                                                   "parallel": 2})
        assert served["models"] == [MID] and served["slots"] == 2
        assert [s.port for s in d.serving.live(force=True)] == [served["port"]]
        argv = json.loads((tmp_path / "argv.json").read_text())
        assert argv[argv.index("-np") + 1] == "2"
        assert argv[argv.index("-c") + 1] == "16384"
        health = d.client.health()
        assert health["serving"][0]["slots"] == 2

    def test_asked_again_it_answers_with_the_server_already_up(
            self, tmp_path, fake_llama_server, daemons):
        d = ServingDaemon(tmp_path, fake_llama_server, name="small", room=24 * G)
        daemons.append(d)
        first = d.client._json("POST", "/serve", {"model": MID, "context": 16384,
                                                  "parallel": 2})
        status, body, _ = d.client._request(
            "POST", "/serve", data=json.dumps({"model": MID, "parallel": 1}).encode(),
            headers={"Content-Type": "application/json"})
        assert status == 200 and json.loads(body)["port"] == first["port"]
        assert len(d.hosting.leases) == 1

    def test_a_model_it_does_not_hold_is_404(self, tmp_path, fake_llama_server, daemons):
        d = ServingDaemon(tmp_path, fake_llama_server, name="small", room=24 * G)
        daemons.append(d)
        with pytest.raises(PeerError, match="404"):
            d.client._json("POST", "/serve", {"model": BIG, "parallel": 1})

    def test_more_seats_than_the_room_holds_is_409(self, tmp_path, fake_llama_server,
                                                   daemons):
        d = ServingDaemon(tmp_path, fake_llama_server, name="small", room=24 * G)
        daemons.append(d)
        with pytest.raises(PeerError, match="409") as caught:
            d.client._json("POST", "/serve", {"model": MID, "context": 16384,
                                              "parallel": 8})
        assert "refused" in str(caught.value) and "does not fit" in str(caught.value)
        assert d.hosting.leases == {}

    def test_without_a_bearer_token_it_is_401(self, tmp_path, fake_llama_server, daemons):
        d = ServingDaemon(tmp_path, fake_llama_server, name="small", room=24 * G)
        daemons.append(d)
        with pytest.raises(PeerError, match="401"):
            Peer(d.client.base_url, "wrong")._json("POST", "/serve", {"model": MID})

    def test_hosting_refuses_before_leasing(self, tmp_path):
        hosting = Hosting(tmp_path, Serving(tmp_path / "serving.json"),
                          fits=lambda: list(FITS))
        with pytest.raises(NoRoom, match="24.0G"):
            hosting.start(tmp_path / MID, name=MID, context=16384, parallel=8, room=24 * G)


class TestApply:
    def test_apply_serves_the_plan_through_each_daemon(
            self, tmp_path, fake_llama_server, key, udp, daemons, measured, monkeypatch,
            capsys):
        raw = load_cluster_key(key)
        d = ServingDaemon(tmp_path, fake_llama_server, name="small", room=24 * G, key=raw,
                          udp=udp)
        daemons.append(d)
        monkeypatch.setenv("ML_STACK_DISCOVERY_PORT", str(udp))
        code = main(["--cluster-key", str(key), "--port", str(d.port), "plan", "--users",
                     "2", "--context", "16384", "--apply", "--timeout", "1"])
        out = capsys.readouterr().out
        assert code == 0, out
        assert f"small: {MID}: 2 seat(s) on port" in out
        live = d.serving.live(force=True)
        assert len(live) == 1 and live[0].models == [MID] and live[0].slots == 2
        assert f"small            {MID}:{live[0].port} (2 seat(s))" in out

    def test_apply_as_json_carries_each_answer(
            self, tmp_path, fake_llama_server, key, udp, daemons, measured, monkeypatch,
            capsys):
        raw = load_cluster_key(key)
        d = ServingDaemon(tmp_path, fake_llama_server, name="small", room=24 * G, key=raw,
                          udp=udp)
        daemons.append(d)
        monkeypatch.setenv("ML_STACK_DISCOVERY_PORT", str(udp))
        main(["--cluster-key", str(key), "--port", str(d.port), "plan", "--users", "2",
              "--apply", "--json", "--timeout", "1"])
        got = json.loads(capsys.readouterr().out)
        assert got["applied"][0]["status"] == 201
        assert got["applied"][0]["served"]["slots"] == 2
        assert got["applied"][0]["serving"][0]["models"] == [MID]

    def test_a_peer_that_refuses_is_reported_not_dropped(
            self, tmp_path, fake_llama_server, key, udp, daemons, monkeypatch, capsys):
        raw = load_cluster_key(key)
        d = ServingDaemon(tmp_path, fake_llama_server, name="small", room=24 * G, key=raw,
                          udp=udp, fits=lambda: [])
        daemons.append(d)
        # The planner believes a record the daemon does not hold; the daemon holds BIG nowhere.
        monkeypatch.setattr(joining, "_measurements", lambda: (
            [Profile(model=BIG, questions=100, right=0.9)],
            [Fit(model=BIG, weights_gpu=G, compute=0, per_token=1024)]))
        monkeypatch.setenv("ML_STACK_DISCOVERY_PORT", str(udp))
        code = main(["--cluster-key", str(key), "--port", str(d.port), "plan", "--users",
                     "1", "--apply", "--json", "--timeout", "1"])
        got = json.loads(capsys.readouterr().out)
        assert code == 0
        assert got["applied"][0]["status"] == 404
        assert "no model called" in got["applied"][0]["error"]
