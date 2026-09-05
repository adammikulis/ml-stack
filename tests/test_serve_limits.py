"""What ml-stack may take on a machine, and stopping the servers nobody is using.

Every limit file is in ``tmp_path``; nothing reads or writes the machine's own.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler

import pytest

from ml_stack.serve import limits as caps
from ml_stack.serve import reclaim as reclaimer

from conftest import threaded_server


@pytest.fixture(autouse=True)
def own_file(tmp_path, monkeypatch):
    """Every test in here reads and writes its own limits file."""
    monkeypatch.setenv("MLSTACK_LIMITS_FILE", str(tmp_path / "limits.json"))
    return tmp_path / "limits.json"


# ------------------------------------------------------------------------- the limits


def test_a_machine_nobody_has_limited_behaves_as_it_did(own_file):
    assert caps.read() == caps.Limits()
    assert caps.read().room(100) == 100
    assert caps.read().refusal(running=9, seats=64) == ""
    assert caps.read().said() == []
    assert not own_file.exists(), "reading writes nothing"


def test_the_memory_limit_caps_what_the_machine_allows_and_stands_alone():
    small = caps.Limits(memory_bytes=40)
    assert small.room(100) == 40, "ours is the smaller"
    assert caps.Limits(memory_bytes=400).room(100) == 100, "the machine's is the smaller"
    assert small.room(0) == 40, "a machine that says nothing still has our limit"


def test_a_lease_over_the_server_or_seat_limit_is_refused_with_what_to_do():
    said = caps.Limits(servers=2).refusal(running=2)
    assert "2 server(s) at once" in said and "ml-stack-serve limits --servers" in said
    assert caps.Limits(servers=2).refusal(running=1) == ""
    said = caps.Limits(seats=1).refusal(seats=4)
    assert "asks for 4 seat(s)" in said and "--seats" in said
    assert caps.Limits(seats=4).refusal(seats=4) == ""


def test_limits_are_written_read_back_and_taken_off(own_file):
    caps.changed(memory_bytes=90 * 2**30, servers=2, idle_s=600.0)
    assert caps.read() == caps.Limits(memory_bytes=90 * 2**30, servers=2, idle_s=600.0)
    assert json.loads(own_file.read_text())["servers"] == 2
    caps.changed(servers=3)
    assert caps.read().servers == 3 and caps.read().idle_s == 600.0, "one field at a time"
    caps.clear()
    assert caps.read() == caps.Limits()


def test_a_file_that_will_not_parse_is_no_limits_rather_than_a_crash(own_file):
    own_file.write_text("{ this is not json")
    assert caps.read() == caps.Limits()
    own_file.write_text('["a list"]')
    assert caps.read() == caps.Limits()
    own_file.write_text('{"servers": 2, "something_else": 9}')
    assert caps.read().servers == 2, "a key from a newer version is ignored"


def test_a_limit_nobody_has_is_refused_by_name():
    with pytest.raises(ValueError, match="no such limit: bandwidth"):
        caps.changed(bandwidth=10)


def test_the_room_a_model_may_use_reads_the_limit_without_anybody_passing_it(monkeypatch):
    """`hub.room` is what every preflight, fit and lease asks. Mutation: have it return
    `machine_room` unchanged."""
    import ml_stack.hub as hub

    monkeypatch.setattr(hub, "machine_room", lambda: 100 * 2**30)
    assert hub.room() == 100 * 2**30
    caps.changed(memory_bytes=60 * 2**30)
    assert hub.room() == 60 * 2**30


# ------------------------------------------------------------------ the lease refuses

def test_a_lease_past_the_server_limit_never_starts_a_process(tmp_path, monkeypatch):
    """The refusal comes before the backend is asked to start anything."""
    from ml_stack.serve.backend import ServerSpec
    from ml_stack.serve.manager import ServerFailed, ServerManager

    state = tmp_path / "servers.json"
    state.write_text(json.dumps({
        "8100": {"port": 8100, "pid": 4242, "model": "one.gguf", "owner_pid": 4242},
    }))
    monkeypatch.setattr("ml_stack.serve.manager.pid_exists", lambda pid: True)
    started: list = []

    class Backend:
        name = "fake"

        def start(self, spec, **kw):
            started.append(spec)
            raise AssertionError("a refused lease must start nothing")

    caps.changed(servers=1)
    manager = ServerManager(backend=Backend(), state_file=state)
    with pytest.raises(ServerFailed, match="1 server\\(s\\) at once"):
        manager.lease(ServerSpec(model="two.gguf", port=8101), roam=False)
    assert started == []

    caps.changed(servers=0, seats=1)
    with pytest.raises(ServerFailed, match="asks for 4 seat"):
        manager.lease(ServerSpec(model="two.gguf", port=8101, parallel=4), roam=False)
    assert started == []


# --------------------------------------------------------------------- what is idle

class Slots(BaseHTTPRequestHandler):
    """A server that answers ``/slots`` with whatever ``processing`` is set to."""

    processing = False
    path_answers = True

    def do_GET(self):
        if not self.path_answers or not self.path.startswith("/slots"):
            self.send_error(404)
            return
        body = json.dumps([{"id": 0, "is_processing": self.processing}]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_busy_now_reads_the_servers_own_slots():
    class Busy(Slots):
        processing = True

    with threaded_server(Busy) as url:
        assert reclaimer.busy_now(url) is True
    with threaded_server(Slots) as url:
        assert reclaimer.busy_now(url) is False

    class Silent(Slots):
        path_answers = False

    with threaded_server(Silent) as url:
        assert reclaimer.busy_now(url) is None, "a server that does not say is not idle"
    assert reclaimer.busy_now("http://127.0.0.1:1") is None, "nor is one that is not there"


def watching_over(tmp_path, now, busy, *, trust=300.0):
    """An `Idleness` on its own file, over a clock and a busy flag the test moves."""
    return reclaimer.Idleness(clock=lambda: now[0], state=tmp_path / "idle.json",
                              trust=trust,
                              probe=lambda url: busy[int(url.rsplit(":", 1)[1])])


SERVERS = {8100: {"base_url": "http://127.0.0.1:8100"}}


def test_idleness_adds_up_across_looks_and_a_busy_server_starts_it_again(tmp_path):
    now, busy = [1000.0], {8100: False}
    watcher = watching_over(tmp_path, now, busy)

    assert watcher.look(SERVERS) == {8100: 0.0}, "the first look only starts the clock"
    now[0] += 60
    assert watcher.look(SERVERS) == {8100: 60.0}
    now[0] += 60
    assert watcher.look(SERVERS) == {8100: 120.0}, "the looks add up"
    busy[8100] = True
    now[0] += 60
    assert watcher.look(SERVERS) == {8100: 0.0}, "seen busy, so the clock starts again"
    busy[8100] = False
    now[0] += 30
    assert watcher.look(SERVERS) == {8100: 30.0}


def test_a_gap_in_the_looking_does_not_count_as_time_nobody_used_it(tmp_path):
    """A server not seen busy at nine and not seen busy at five was not therefore idle all
    day. Mutation: credit the whole gap."""
    now, busy = [1000.0], {8100: False}
    watcher = watching_over(tmp_path, now, busy, trust=120.0)
    watcher.look(SERVERS)
    now[0] += 60
    assert watcher.look(SERVERS) == {8100: 60.0}
    now[0] += 5000
    assert watcher.look(SERVERS) == {8100: 0.0}, "the silent afternoon is not evidence"
    now[0] += 60
    assert watcher.look(SERVERS) == {8100: 60.0}, "and the clock starts again"


def test_what_one_look_found_is_there_for_the_next_pass_to_build_on(tmp_path):
    """One look can only report zero, so a pass that started from nothing could never act."""
    now, busy = [1000.0], {8100: False}
    watching_over(tmp_path, now, busy).look(SERVERS)
    now[0] += 90
    assert watching_over(tmp_path, now, busy).look(SERVERS) == {8100: 90.0}
    assert json.loads((tmp_path / "idle.json").read_text())["8100"]["idle"] == 90.0


def test_a_watcher_with_no_file_keeps_its_readings_in_memory_alone(tmp_path):
    now, busy = [1000.0], {8100: False}
    watcher = reclaimer.Idleness(clock=lambda: now[0], state=None,
                                 probe=lambda url: busy[8100])
    watcher.look(SERVERS)
    now[0] += 60
    assert watcher.look(SERVERS) == {8100: 60.0}
    assert not list(tmp_path.glob("*.json"))


def test_a_state_file_that_will_not_parse_is_no_readings_rather_than_a_crash(tmp_path):
    (tmp_path / "idle.json").write_text("{ not json")
    now, busy = [1000.0], {8100: False}
    assert watching_over(tmp_path, now, busy).look(SERVERS) == {8100: 0.0}


def test_a_port_that_has_gone_is_forgotten(tmp_path):
    now, busy = [1000.0], {8100: False}
    watcher = watching_over(tmp_path, now, busy)
    watcher.look(SERVERS)
    now[0] += 60
    assert watcher.look({}) == {}
    now[0] += 999
    assert watcher.look(SERVERS) == {8100: 0.0}, "a port that went is a new port"


def test_a_server_that_does_not_answer_is_left_out_rather_than_called_idle(tmp_path):
    watcher = reclaimer.Idleness(clock=lambda: 0.0, probe=lambda url: None,
                                 state=tmp_path / "idle.json")
    assert watcher.look(SERVERS) == {}


# ----------------------------------------------------------------------- reclaiming

def watcher_over(idle_by_port):
    """An `Idleness` whose readings are exactly ``idle_by_port``."""
    watcher = reclaimer.Idleness(clock=lambda: 0.0, probe=lambda url: False, state=None)
    watcher.look = lambda servers: {p: idle_by_port[p] for p in servers if p in idle_by_port}
    return watcher


def test_only_the_servers_past_the_threshold_are_stopped_and_each_is_said():
    held = {8100: {"model": "/models/one.gguf", "pid": 11},
            8101: {"model": "/models/two.gguf", "pid": 12}}
    stopped, said = [], []

    def stop(port, entry):
        stopped.append((port, entry.get("pid")))
        return True

    ports = reclaimer.reclaim_idle(older_than=300, idleness=watcher_over({8100: 900.0, 8101: 30.0}),
                                   servers=lambda: held, stop=stop, say=said.append)
    assert ports == [8100] and stopped == [(8100, 11)]
    assert said == ["reclaimed port 8100 after 900s idle (one.gguf)"]


def test_nothing_is_reclaimed_without_an_idle_time():
    stopped = []
    assert reclaimer.reclaim_idle(older_than=0, servers=lambda: {8100: {}},
                                  stop=lambda p, e: stopped.append(p) or True) == []
    assert stopped == []


def test_a_stop_that_did_nothing_is_not_reported_as_reclaimed():
    ports = reclaimer.reclaim_idle(older_than=1, idleness=watcher_over({8100: 900.0}),
                                    servers=lambda: {8100: {}}, stop=lambda p, e: False)
    assert ports == []


def test_watching_reclaims_in_the_background_until_the_block_ends():
    held = {8100: {"model": "one.gguf", "pid": 11}}
    stopped = []
    with reclaimer.watching(older_than=1, every=0.05,
                            idleness=watcher_over({8100: 900.0}),
                            servers=lambda: dict(held),
                            stop=lambda p, e: stopped.append(p) or True) as done:
        for _ in range(100):
            if stopped:
                break
            time.sleep(0.02)
        assert stopped, "the watcher reclaimed while the block ran"
        assert not done.is_set()
    assert done.is_set(), "and stops when it ends"


# ------------------------------------------------------------------------ the commands

def test_the_limits_command_prints_what_is_set_and_sets_what_it_is_told(capsys, own_file):
    from ml_stack.serve import cli

    assert cli.main(["limits"]) == 0
    assert "nothing is limited here" in capsys.readouterr().out

    assert cli.main(["limits", "--memory", "90G", "--servers", "2", "--idle", "10m"]) == 0
    said = capsys.readouterr().out
    assert "90.0G" in said and "2 at once" in said and "600s" in said
    assert str(own_file) in said
    assert caps.read() == caps.Limits(memory_bytes=90 * 2**30, servers=2, idle_s=600.0)

    assert cli.main(["limits", "--clear"]) == 0
    assert "every limit is off" in capsys.readouterr().out
    assert caps.read() == caps.Limits()


def test_the_limits_command_refuses_what_it_cannot_read(capsys):
    from ml_stack.serve import cli

    assert cli.main(["limits", "--memory", "a lot"]) == 2
    assert "cannot read" in capsys.readouterr().err
    assert cli.main(["limits", "--idle", "soon"]) == 2
    assert "length of time" in capsys.readouterr().err


def test_the_reclaim_command_needs_an_idle_time_from_somewhere(capsys):
    from ml_stack.serve import cli

    assert cli.main(["reclaim"]) == 2
    assert "no idle time given" in capsys.readouterr().err


def test_the_reclaim_command_takes_its_default_from_the_limits(capsys, monkeypatch):
    from ml_stack.serve import cli

    caps.changed(idle_s=300.0)
    asked = {}

    def reclaim_idle(*, older_than, idleness=None, say=lambda _l: None, **kw):
        asked["older_than"] = older_than
        say("reclaimed port 8100 after 900s idle")
        return [8100]

    monkeypatch.setattr("ml_stack.serve.reclaim.reclaim_idle", reclaim_idle)
    monkeypatch.setattr("ml_stack.serve.reclaim.Idleness.look", lambda self, servers: {})
    assert cli.main(["reclaim", "--settle", "0"]) == 0
    assert asked["older_than"] == 300.0
    assert "reclaimed port 8100" in capsys.readouterr().out
