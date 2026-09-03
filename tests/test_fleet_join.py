"""``ml-stack-fleet``: one command makes a machine a peer, and the page's Join runs it too.

The daemon it starts, the llama-server it fetches and the logon service it installs are the
three things faked here -- each behind a parameter of `join_machine` -- and everything else
is real: a real cluster key written under ``tmp_path``, a real ``/health`` on a real loopback
socket, a real beacon on a randomised UDP port, and the real daemon with the UI mounted for
the page's routes. Every name is invented.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from ml_stack.fleet import join as joining
from ml_stack.fleet.discovery import (
    Advertiser,
    Beacon,
    in_cluster,
    load_cluster_key,
    memberships,
)
from ml_stack.fleet.join import (
    JoinError,
    checks,
    describe,
    join_machine,
    leave_machine,
    main,
    peers,
    sweep_argv,
    table,
)
from ml_stack.setup import Finding

WORDS = "quince larch marlow"
DEVICE = {"gpu": "Pellard P40", "vram_total_gb": 24.0, "vram_free_gb": 20.5,
          "serving": [{"port": 8099, "models": ["quince-2b.gguf"], "slots": 1}],
          "models": [{"name": "quince-2b.gguf", "size": 1}, {"name": "larch-9b.gguf", "size": 2}],
          "commit": "abc1234"}


def _free_udp() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _free_tcp() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeDaemon:
    """Answers ``/health`` on loopback and beacons on the test's UDP port -- what a daemon
    `join` started would do, without the daemon."""

    def __init__(self, port: int, key: bytes, udp: int, name: str = "larch") -> None:
        self.name = name

        class H(BaseHTTPRequestHandler):
            def do_GET(self_) -> None:  # noqa: N802
                body = json.dumps({"ok": True, "name": name, "busy": False,
                                   "free": 1, "slots": 1, "queued": 0}).encode()
                self_.send_response(200 if self_.path == "/health" else 404)
                self_.send_header("Content-Type", "application/json")
                self_.send_header("Content-Length", str(len(body)))
                self_.end_headers()
                self_.wfile.write(body)

            def log_message(self_, *a: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.advertiser = Advertiser(Beacon(name=name, port=port, device=dict(DEVICE)),
                                     key, port=udp, interval_s=0.2).start()

    def close(self) -> None:
        self.advertiser.stop()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse=True)
def machine_looks_fine(monkeypatch):
    """The checks read sysctl and libllama; here they read a list."""
    monkeypatch.setattr(joining, "_machine_findings", lambda: [
        Finding(name="memory a model may use", good=True, said="96G of 128G (75%)"),
        Finding(name="models on this machine", good=True, said="2 file(s)")])
    monkeypatch.setattr(joining, "_server_here", lambda: "/opt/fake/bin/llama-server")


@pytest.fixture
def udp() -> int:
    return _free_udp()


@pytest.fixture
def key(tmp_path) -> Path:
    return tmp_path / "cluster.key"


@pytest.fixture
def daemons():
    started: list[FakeDaemon] = []
    yield started
    for d in started:
        d.close()


# -- the join --------------------------------------------------------------------------
class TestJoin:
    @pytest.mark.slow
    def test_it_checks_joins_starts_announces_and_lists(self, tmp_path, key, udp, daemons):
        tcp = _free_tcp()
        said: list[str] = []
        calls: list[tuple[int, Path, str]] = []

        def start(port: int, root: Path, name: str) -> int:
            calls.append((port, root, name))
            # The daemon `join` starts reads the key `join` wrote a moment before.
            daemons.append(FakeDaemon(port, load_cluster_key(key), udp, name="larch"))
            return 4242

        assert not in_cluster(key)
        joined = join_machine(name="larch", passphrase=WORDS, group="home", port=tcp,
                              root=tmp_path / "root", cluster_key_path=key, start=start,
                              discovery_port=udp, say=said.append)

        assert in_cluster(key) and memberships(key)[0].group == "home"
        assert calls == [(tcp, tmp_path / "root", "larch")]
        assert joined.started and joined.daemon_pid == 4242
        assert joined.name == "larch", "the name comes from the daemon's own /health"
        assert {c.name for c in joined.checks} >= {"memory a model may use", "llama-server"}
        assert all(c.good for c in joined.checks)

        names = [p["name"] for p in joined.peers]
        assert names == ["larch"], f"the fleet must see the machine that joined; saw {names}"
        me = joined.peers[0]
        assert me["is_self"]
        assert me["serving"] == ["quince-2b.gguf:8099"]
        assert me["room"] == "20.5/24.0 GB"
        assert me["commit"] == "abc1234"
        assert me["models"] == ["quince-2b.gguf", "larch-9b.gguf"]
        assert me["clusters"] == ["home"]
        text = "\n".join(said)
        assert "joined cluster 'home'" in text
        assert f"discovery port {udp}" in text
        assert "larch" in text and "quince-2b.gguf:8099" in text, "it prints what the fleet sees"

    def test_no_cluster_and_no_passphrase_is_a_refusal_not_a_daemon(self, tmp_path, key, udp):
        started = []
        with pytest.raises(JoinError) as left:
            join_machine(passphrase="", port=_free_tcp(), root=tmp_path, cluster_key_path=key,
                         start=lambda *a: started.append(a) or 1, discovery_port=udp,
                         say=lambda s: None)
        assert "passphrase" in str(left.value)
        assert started == [], "nothing may start on a machine that cannot announce"

    @pytest.mark.slow

    def test_a_running_daemon_is_reused_and_enrolled_through(self, tmp_path, key, udp, daemons):
        """Writing the key file under a running daemon leaves it announcing the old set;
        the cluster has to go in through the daemon, which re-reads them."""
        tcp = _free_tcp()
        from ml_stack.fleet.discovery import join as join_cluster

        # Already in a cluster, already running, before `join` is asked for a second one.
        join_cluster(WORDS, group="home", path=key)
        daemons.append(FakeDaemon(tcp, load_cluster_key(key), udp, name="quince"))
        enrolled: list[tuple[str, str]] = []
        started: list = []

        joined = join_machine(name="ignored", passphrase="other words here", group="lab",
                              port=tcp, root=tmp_path, cluster_key_path=key,
                              start=lambda *a: started.append(a) or 1,
                              enrol=lambda words, group: enrolled.append((words, group)),
                              discovery_port=udp, say=lambda s: None)
        assert started == [], "a daemon that answers is the daemon"
        assert enrolled == [("other words here", "lab")]
        assert not joined.started
        assert joined.name == "quince", "the running daemon's name wins over --name"
        assert [p["name"] for p in joined.peers] == ["quince"]

    @pytest.mark.slow

    def test_already_in_a_cluster_needs_no_passphrase(self, tmp_path, key, udp, daemons):
        from ml_stack.fleet.discovery import join as join_cluster

        join_cluster(WORDS, group="home", path=key)
        tcp = _free_tcp()
        daemons.append(FakeDaemon(tcp, load_cluster_key(key), udp))
        joined = join_machine(port=tcp, root=tmp_path, cluster_key_path=key,
                              discovery_port=udp, say=lambda s: None)
        assert joined.group == "home"

    @pytest.mark.slow

    def test_persist_installs_at_logon_and_says_when_it_could_not(self, tmp_path, key, udp,
                                                                 daemons):
        from ml_stack.fleet.autostart import Autostart

        tcp = _free_tcp()
        asked: list[str] = []

        def start(port: int, root: Path, name: str) -> int:
            daemons.append(FakeDaemon(port, load_cluster_key(key), udp))
            return 7

        def installs(mode: str, **kw) -> Autostart:
            asked.append(mode)
            return Autostart(mode, installed=True, path=tmp_path / "agent.plist")

        # the fake logon service never brings a daemon up, so the first wait runs its
        # deadline out on purpose; the FakeDaemon `start` puts up answers on the first poll,
        # so a short deadline measures the same fall-through in three seconds instead of twenty
        joined = join_machine(passphrase=WORDS, persist=True, port=tcp, root=tmp_path,
                              cluster_key_path=key, start=start, persist_with=installs,
                              discovery_port=udp, say=lambda s: None, wait_s=3.0)
        assert asked == ["login"] and joined.persisted and joined.persist_note == ""

        def refuses(mode: str, **kw) -> Autostart:
            return Autostart(mode, installed=False, command="sudo cp agent.plist /Library",
                             note="needs administrator rights")

        joined = join_machine(persist=True, port=tcp, root=tmp_path, cluster_key_path=key,
                              start=start, persist_with=refuses, discovery_port=udp,
                              say=lambda s: None, wait_s=3.0)
        assert not joined.persisted
        assert "sudo cp" in joined.persist_note and "administrator" in joined.persist_note

    def test_a_daemon_that_never_answers_is_an_error_naming_the_log(self, tmp_path, key, udp):
        with pytest.raises(JoinError) as left:
            join_machine(passphrase=WORDS, port=_free_tcp(), root=tmp_path / "r",
                         cluster_key_path=key, start=lambda *a: 99, wait_s=0.6,
                         discovery_port=udp, say=lambda s: None)
        assert "traind.log" in str(left.value)


# -- the checks -----------------------------------------------------------------------
class TestChecks:
    def test_a_missing_server_is_fetched_and_said(self, tmp_path, monkeypatch):
        monkeypatch.setattr(joining, "_server_here", lambda: "")
        got: list[Path] = []

        def ensure(root: Path) -> Path:
            got.append(root)
            return root / "vendor" / "llama-server"

        said: list[str] = []
        found = checks(tmp_path, ensure=ensure, say=said.append)
        assert got == [tmp_path]
        server = next(c for c in found if c.name == "llama-server")
        assert server.good and "downloaded now" in server.said
        assert any("getting a release build" in s for s in said)

    def test_a_server_that_cannot_be_fetched_is_a_finding_with_a_fix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(joining, "_server_here", lambda: "")

        def cannot(root: Path) -> Path:
            raise RuntimeError("no build for this machine")

        server = next(c for c in checks(tmp_path, ensure=cannot, say=lambda s: None)
                      if c.name == "llama-server")
        assert not server.good and server.fix == "ml-stack-serve build"

    def test_only_the_facts_serving_depends_on_are_kept(self, tmp_path):
        names = {c.name for c in checks(tmp_path, say=lambda s: None)}
        assert names == {"memory a model may use", "llama-server"}


# -- what the fleet sees ---------------------------------------------------------------
class TestStatus:
    def test_describe_reads_an_older_daemon_defensively(self):
        row = describe(Beacon(name="pi", port=8770, device={"cpus": 4, "ram_gb": 8.0}))
        assert row["room"] == "8.0 GB ram" and row["serving"] == []
        assert row["commit"] == "?" and row["lock"] == "" and row["models"] == []

    def test_describe_reads_what_the_bench_host_puts_on_the_beacon(self):
        """`fleet.bench.BenchHost.report`: room in bytes, the commit it runs, measuring."""
        row = describe(Beacon(name="studio", port=8770, device={
            "room_bytes": 96 * 2 ** 30, "bench_commit": "0ce5bc5", "measuring": True,
            "vram_total_gb": 128.0}))
        assert row["room"] == "96.0G", "the bench's room, not the card's"
        assert row["commit"] == "0ce5bc5" and row["lock"] == "measuring"
        assert "measuring" in table([row])

    def test_the_table_has_every_column_and_a_measuring_state(self):
        busy = describe(Beacon(name="larch", port=8770, host="10.0.0.2",
                               device={**DEVICE, "measuring": "sweep quince-2b"}))
        idle = describe(Beacon(name="pi", port=8770, host="10.0.0.3", device={"ram_gb": 8}))
        text = table([busy, idle])
        head, first, second = text.splitlines()
        for col in ("NAME", "URL", "ROOM", "STATE", "COMMIT", "SERVING"):
            assert col in head
        assert "measuring" in first and "abc1234" in first and "quince-2b.gguf:8099" in first
        assert "idle" in second and "8 GB ram" in second

    def test_no_peers_says_what_to_run(self):
        assert "ml-stack-fleet join" in table([])

    def test_a_tracking_peer_shows_its_commit_and_the_branch_it_follows(self):
        """A fleet half on one commit and half on another is what this column exists to
        make visible, so the age is when that peer last looked, not when it started."""
        import time as _time

        row = describe(Beacon(name="harrowgate", port=8770, host="10.0.0.4", device={
            **DEVICE, "bench_commit": "9f2c1ab", "commit_age_s": 3 * 3600,
            "version": "0.4.1", "tracking": "main",
            "update_checked_at": _time.time() - 240}))
        assert row["tracking"] == "main" and row["version"] == "0.4.1"

        head, line = table([row]).splitlines()
        assert "COMMIT" in head and "UPDATES" in head
        assert "9f2c1ab 3h" in line, line
        assert "main 4m" in line, line

    def test_a_peer_following_releases_says_so_and_one_following_nothing_says_off(self):
        releasing = describe(Beacon(name="quincewood", port=8770, device={
            "tracking": "releases", "update_checked_at": 1.0}))
        stuck = describe(Beacon(name="larchmere", port=8770, device={"tracking": "off"}))
        text = table([releasing, stuck])
        assert "releases" in text and "off" in text

    def test_a_peer_whose_last_update_failed_is_marked_rather_than_looking_fine(self):
        row = describe(Beacon(name="pellard", port=8770, device={
            "tracking": "main", "update_checked_at": 2.0,
            "update_error": "git pull --ff-only failed"}))
        assert "main !" in table([row])

    def test_a_branch_to_follow_is_written_where_the_daemon_will_read_it(self, tmp_path):
        """Written into the settings rather than passed as a flag: the logon service that
        brings the daemon back after a reboot carries no flags of ours."""
        from ml_stack.fleet.settings import Settings

        assert joining.remember_track(tmp_path, "main") == "main"
        assert Settings.load(tmp_path / "settings.json").track_branch == "main"

        assert joining.remember_track(tmp_path, "off") == ""
        assert Settings.load(tmp_path / "settings.json").track_branch == ""

    @pytest.mark.slow

    def test_peers_lists_one_machine_once_across_two_clusters(self, key, udp, daemons):
        from ml_stack.fleet.discovery import join as join_cluster

        join_cluster(WORDS, group="home", path=key)
        join_cluster("other words here", group="lab", path=key)
        tcp = _free_tcp()
        beacon = Beacon(name="larch", port=tcp, device=dict(DEVICE))
        both = [Advertiser(beacon, m.key, port=udp, interval_s=0.2).start()
                for m in memberships(key)]
        try:
            rows = peers(cluster_key_path=key, timeout_s=1.0, port=udp, self_name="larch")
        finally:
            for a in both:
                a.stop()
        assert [r["name"] for r in rows] == ["larch"]
        assert sorted(rows[0]["clusters"]) == ["home", "lab"]

    def test_status_command_prints_json_rows(self, key, udp, daemons, monkeypatch, capsys):
        from ml_stack.fleet.discovery import join as join_cluster

        join_cluster(WORDS, group="home", path=key)
        tcp = _free_tcp()
        daemons.append(FakeDaemon(tcp, load_cluster_key(key), udp, name="larch"))
        monkeypatch.setenv("ML_STACK_DISCOVERY_PORT", str(udp))
        code = main(["--cluster-key", str(key), "--port", str(tcp), "status", "--json",
                     "--timeout", "1"])
        rows = json.loads(capsys.readouterr().out)
        assert code == 0 and [r["name"] for r in rows] == ["larch"]
        assert rows[0]["is_self"], "the daemon on --port is this machine"

    def test_status_in_no_cluster_says_join(self, key, capsys):
        assert main(["--cluster-key", str(key), "status"]) == 1
        assert "ml-stack-fleet join" in capsys.readouterr().err

    def test_join_takes_the_passphrase_name_and_cluster_from_the_environment(
            self, key, tmp_path, monkeypatch):
        """An install script has no terminal to type a passphrase at, and prompting one
        that cannot answer hangs the install rather than failing it."""
        monkeypatch.setenv("ML_STACK_PASSPHRASE", WORDS)
        monkeypatch.setenv("ML_STACK_NAME", "harrowgate")
        monkeypatch.setenv("ML_STACK_CLUSTER", "attic")
        asked: dict = {}

        def fake_join(**kw):
            asked.update(kw)
            return joining.Joined(name=kw["name"], port=kw["port"],
                                  root=Path(kw["root"]), group=kw["group"])

        monkeypatch.setattr(joining, "join_machine", fake_join)
        code = main(["--cluster-key", str(key), "--root", str(tmp_path), "join",
                     "--track", "main"])

        assert code == 0
        assert asked["passphrase"] == WORDS
        assert asked["name"] == "harrowgate"
        assert asked["group"] == "attic"
        assert asked["track"] == "main"

    def test_a_flag_beats_the_environment(self, key, tmp_path, monkeypatch):
        monkeypatch.setenv("ML_STACK_PASSPHRASE", "the wrong words entirely")
        monkeypatch.setenv("ML_STACK_NAME", "harrowgate")
        asked: dict = {}
        monkeypatch.setattr(joining, "join_machine",
                            lambda **kw: (asked.update(kw),
                                          joining.Joined(name="x", port=1,
                                                         root=Path(kw["root"]),
                                                         group="g"))[1])
        main(["--cluster-key", str(key), "--root", str(tmp_path), "join",
              "--passphrase", WORDS, "--name", "larchmere"])

        assert asked["passphrase"] == WORDS and asked["name"] == "larchmere"


# -- leaving ---------------------------------------------------------------------------
class TestLeave:
    def test_leave_drops_the_cluster_the_service_and_the_daemon_it_started(self, tmp_path, key):
        from ml_stack.fleet.discovery import join as join_cluster

        join_cluster(WORDS, group="home", path=key)
        # A process standing in for the daemon `join` started, stopped by pid, never by name.
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        root = tmp_path / "root"
        root.mkdir()
        joining.started_file(root).write_text(json.dumps({"pid": child.pid}))
        removed: list[Path] = [tmp_path / "agent.plist"]

        out = leave_machine(root=root, cluster_key_path=key, say=lambda s: None,
                            unpersist=lambda: removed)
        try:
            assert out["left"] == ["home"] and not in_cluster(key)
            assert out["removed"] == [str(tmp_path / "agent.plist")]
            assert out["stopped"] == child.pid
            assert child.wait(timeout=5) != 0
            assert not joining.started_file(root).exists()
        finally:
            if child.poll() is None:
                child.kill()

    def test_leave_one_group_keeps_the_other(self, tmp_path, key):
        from ml_stack.fleet.discovery import join as join_cluster

        join_cluster(WORDS, group="home", path=key)
        join_cluster("other words here", group="lab", path=key)
        out = leave_machine(group="lab", root=tmp_path, cluster_key_path=key,
                            say=lambda s: None, unpersist=lambda: [])
        assert out["left"] == ["lab"] and [m.group for m in memberships(key)] == ["home"]


# -- a sweep over the fleet ------------------------------------------------------------
class TestSweepLine:
    def test_the_line_is_one_serve_per_model_and_nothing_this_machine_would_run(self):
        argv = sweep_argv(["quince-2b.gguf", "hf:pellard/larch-9B-GGUF/larch-9B-Q4_K_M.gguf"],
                          peers=["larch", "pi"], sample=8, label="tuesday")
        assert argv[:2] == ["sweep", "--fleet"]
        assert argv.count("--serve") == 2
        assert argv[argv.index("--peers") + 1] == "larch,pi"
        assert argv[argv.index("--sample") + 1] == "8"
        assert "--detach" not in argv, "the caller detaches, and hands the log back"

    def test_no_models_is_refused(self):
        with pytest.raises(ValueError):
            sweep_argv([])

    def test_the_line_parses_as_the_bench_would(self, tmp_path, monkeypatch):
        from ml_stack.graph import bench
        from ml_stack.graph.bench.run import _parser

        monkeypatch.setattr(bench, "HOME", tmp_path / "bench")
        args = _parser().parse_args(sweep_argv(["quince-2b.gguf"], peers=["larch"], sample=4))
        assert args.fleet and args.serve == ["quince-2b.gguf"] and args.peers == "larch"


# -- the page --------------------------------------------------------------------------
class TestThePage:
    """The Cluster view's Join button and its 'Run across fleet' form, against the real
    daemon with the UI mounted -- `test_fleet_ui.Serving` -- on a real socket."""

    @pytest.fixture
    def page(self, tmp_path, udp, monkeypatch, daemons):
        from test_fleet_ui import Serving

        from ml_stack.graph import bench

        monkeypatch.setattr(bench, "HOME", tmp_path / "bench")
        s = Serving(tmp_path, name="studio")
        s.ui.peer_port = s.port
        s.ui.discovery_port = udp
        s.ui.root = tmp_path / "traind"
        s.call("/ui/setup/join", method="POST", body={"passphrase": WORDS, "group": "home"})
        _, _, headers = s.call("/ui/session", method="POST", body={"passphrase": WORDS})
        cookie = headers["Set-Cookie"].split(";")[0]
        # Another machine in the same cluster, holding two models and serving one.
        daemons.append(FakeDaemon(_free_tcp(), load_cluster_key(s.keyfile), udp, name="larch"))
        try:
            yield s, cookie
        finally:
            s.close()

    def test_the_fleet_view_lists_peers_with_serving_room_and_the_models_held(self, page):
        s, cookie = page
        status, body, _ = s.call("/ui/fleet", cookie=cookie)
        assert status == 200
        larch = next(p for p in body["peers"] if p["name"] == "larch")
        assert larch["serving"] == ["quince-2b.gguf:8099"]
        assert larch["room"] == "20.5/24.0 GB" and larch["commit"] == "abc1234"
        assert body["models"] == ["larch-9b.gguf", "quince-2b.gguf"]
        assert body["bench"]["available"] and "nothing is measuring" in body["bench"]["text"]

    @pytest.mark.slow

    def test_the_join_button_runs_the_same_join(self, page):
        s, cookie = page
        status, body, _ = s.call("/ui/fleet/join", method="POST", body={}, cookie=cookie)
        assert status == 400, "an empty form must not start anything"

        status, body, _ = s.call("/ui/fleet/join", method="POST", cookie=cookie,
                                 body={"passphrase": "other words here", "group": "lab"})
        assert status == 200, body
        assert body["group"] == "lab" and not body["started"], "this daemon is the daemon"
        assert [m.group for m in memberships(s.keyfile)] == ["home", "lab"]
        assert any("already running as 'studio'" in line for line in body["said"])
        assert {c["name"] for c in body["checks"]} >= {"llama-server"}
        assert "larch" in [p["name"] for p in body["peers"]]

    def test_run_across_fleet_builds_the_line_and_detaches_it(self, page, tmp_path):
        from ml_stack.graph.bench.run import measuring_file

        s, cookie = page
        status, body, _ = s.call("/ui/bench/sweep", method="POST", body={}, cookie=cookie)
        assert status == 400

        asked = {"models": ["quince-2b.gguf"], "peers": ["larch"], "sample": 4,
                 "label": "tuesday", "dry_run": True}
        status, body, _ = s.call("/ui/bench/sweep", method="POST", body=asked, cookie=cookie)
        assert status == 200
        assert body["command"] == ("ml-stack-bench sweep --fleet --serve quince-2b.gguf "
                                   "--peers larch --sample 4 --label tuesday")

        detached: list[list[str]] = []

        def fake_detach(argv: list[str]) -> Path:
            detached.append(list(argv))
            log = tmp_path / "bench" / "logs" / "sweep-quince-20260902T101500.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("argv: sweep\nstarted: now\n")
            measuring_file().write_text(json.dumps({"pid": 2 ** 22 + 7, "argv": argv,
                                                    "log": str(log), "started": "now"}))
            return log

        s.ui.detach = fake_detach
        asked.pop("dry_run")
        status, body, _ = s.call("/ui/bench/sweep", method="POST", body=asked, cookie=cookie)
        assert status == 202, body
        assert detached == [body["argv"]] and body["pid"] == 2 ** 22 + 7
        assert body["log"].endswith("sweep-quince-20260902T101500.log")

        status, body, _ = s.call("/ui/bench/status", cookie=cookie)
        assert status == 200 and "sweep-quince-20260902T101500.log" in body["text"]
        status, body, _ = s.call("/ui/bench/history", cookie=cookie)
        assert status == 200 and [e["subcommand"] for e in body["history"]] == ["sweep"]

    def test_stop_needs_the_pid_the_page_was_shown(self, page):
        s, cookie = page
        status, body, _ = s.call("/ui/bench/stop", method="POST", body={}, cookie=cookie)
        assert status == 400
        status, body, _ = s.call("/ui/bench/stop", method="POST", body={"pid": 12},
                                 cookie=cookie)
        assert status == 200 and "nothing is measuring" in body["stopped"]

    def test_the_fleet_routes_need_a_session(self, page):
        s, _ = page
        for path in ("/ui/fleet", "/ui/bench/status", "/ui/bench/history"):
            status, _, _ = s.call(path)
            assert status == 401, path
