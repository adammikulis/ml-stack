"""The fleet side of ``ml-stack-bench sweep --fleet``: real daemons on loopback, a bench
that is a small script rather than a model, and stores in ``tmp_path``.

Two fake daemons are two real ``ThreadingHTTPServer``s running the real handler over a
real `JobRunner`; only three things are stood in for -- the memory a peer may use, the
commit it runs, and the launch of ``ml-stack-bench --detach``, which would load a model
and write under ``~/.ml-stack``. Nothing here reads that directory. Every name is invented.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from ml_stack.bench.peer_runs import gather, import_runs
from ml_stack.fleet.api import make_handler
from ml_stack.fleet.daemon import load_or_create_token
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.measuring import (
    BenchHost,
    Job,
    Local,
    Refused,
    ended_badly,
    here,
    installed_commit,
    jobs_from,
    same_commit,
)
from ml_stack.fleet.remote import Peer, PeerError
from ml_stack.fleet.sizing import estimate
from ml_stack.fleet.sweeps import (
    Handle,
    bench_export,
    dispatch,
    plan,
    submit_bench,
    wait,
)

G = 2**30
COMMIT = "ab12cd3"


# -- a bench that is a script --------------------------------------------------------
def scripted_launch(*, seconds: float = 0.4, says: str = "kept as bench:tried:20260902T101010",
                    fails: bool = False):
    """A `BenchHost.launch`: starts a child in its own session that writes ``says`` to a
    log under ``home/logs`` after ``seconds`` -- or ``error: ...`` and exit 1 -- and
    returns (pid, log) as `detach_bench` does. The child is reaped by a thread so its pid
    goes when it exits, the way a detached bench's does once launchd has it."""
    calls: list[list[str]] = []

    def launch(line, home: Path, python: Path):
        calls.append(list(line))
        launch.pythons.append(python)
        logs = home / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"sweep-{len(calls)}-{int(time.time() * 1000)}.log"
        line_said = "error: boom, nothing kept" if fails else says
        code = (f"import time; time.sleep({seconds}); print({line_said!r}, flush=True); "
                f"raise SystemExit({1 if fails else 0})")
        with log.open("ab") as out:
            out.write(f"argv: {' '.join(line)}\n".encode())
            proc = subprocess.Popen([sys.executable, "-c", code], stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        threading.Thread(target=proc.wait, daemon=True).start()
        return proc.pid, log

    launch.calls = calls
    launch.pythons = []
    return launch


@dataclass
class Box:
    """One fake daemon: its client, its host, and where it keeps things."""

    name: str
    peer: Peer
    host: BenchHost
    runner: JobRunner
    home: Path
    httpd: ThreadingHTTPServer

    @property
    def store(self) -> Path:
        return self.home / "runs.ladybug"


def _box(tmp_path: Path, name: str, *, room: int, launch=None, busy: bool = False) -> Box:
    root = tmp_path / name
    files = root / "files"
    files.mkdir(parents=True)
    home = root / "bench"
    token = load_or_create_token(root)
    runner = JobRunner(root, files)
    host = BenchHost(runner, home=home, room=lambda: room,
                     launch=launch or scripted_launch(), name=name)
    host.commit, host.poll_s, host.machine = COMMIT, 0.1, f"id-{name}"

    def report():
        return {"cpus": 8, **host.report()}

    httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                make_handler(runner, files, token, name, report, bench=host))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    peer = Peer(f"http://127.0.0.1:{httpd.server_address[1]}", token)
    box = Box(name=name, peer=peer, host=host, runner=runner, home=home, httpd=httpd)
    if busy:
        box.peer.submit([sys.executable, "-c", "import time; time.sleep(30)"], name="training")
        deadline = time.time() + 5
        while time.time() < deadline and not runner.status()["busy"]:
            time.sleep(0.05)
    return box


@pytest.fixture
def boxes(tmp_path):
    """Two daemons announcing room and idle: ``roomy`` may use 96G, ``small`` 24G."""
    made = [_box(tmp_path, "roomy", room=96 * G), _box(tmp_path, "small", room=24 * G)]
    try:
        yield made
    finally:
        for box in made:
            box.runner.shutdown()
            box.httpd.shutdown()
            box.httpd.server_close()


def _job(*models: str, needs: dict | None = None, commit: str = COMMIT, label: str = "sweep"):
    line = ["sweep", "--short"]
    for m in models:
        line += ["--serve", m]
    return Job(argv=tuple(line), models=models, commit=commit, kept_label=label,
               needs=needs or dict.fromkeys(models, 4 * G))


def _await(predicate, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# -- what a peer says about itself ---------------------------------------------------
def test_health_carries_room_commit_and_whether_it_is_measuring(boxes):
    roomy, small = boxes
    got = roomy.peer.health()
    assert got["room_bytes"] == 96 * G
    assert got["bench_commit"] == COMMIT
    assert got["measuring"] is False
    assert small.peer.health()["room_bytes"] == 24 * G


def test_the_installed_commit_names_this_checkout():
    """The pin: the sha of the checkout the package is imported from, dirty or not."""
    got = installed_commit()
    assert got, "this test runs from a checkout, so there is a sha"
    assert same_commit(got, got.split()[0])
    assert same_commit("ab12cd3 (dirty)", "ab12cd3")
    assert not same_commit("ab12cd3", "ff00ee1")
    assert not same_commit("", ""), "no answer is not the same answer"


# -- plan ----------------------------------------------------------------------------
def test_plan_puts_the_big_model_on_the_roomy_peer_and_reports_what_fits_nowhere(boxes):
    roomy, small = boxes
    needs = {"big-27B.gguf": 60 * G, "tiny-2B.gguf": 4 * G, "huge-120B.gguf": 200 * G}
    said: list[str] = []

    got = plan(list(needs), [roomy.peer, small.peer], needs=needs, log=said.append)

    assert got[roomy.peer] == ["big-27B.gguf"]
    assert got[small.peer] == ["tiny-2B.gguf"], "the second model spreads, not stacks"
    assert [m for m, _ in got.unplaced] == ["huge-120B.gguf"]
    why = got.unplaced[0][1]
    assert "roomy: room 96.0G < 200.0G" in why and "small: room 24.0G < 200.0G" in why
    text = "\n".join(said)
    assert "planning 3 model(s) over 2 peer(s)" in text
    assert "big-27B.gguf" in text and "-> roomy (room 96.0G)" in text
    assert "huge-120B.gguf" in text and "fits nowhere" in text


def test_plan_skips_a_busy_peer_and_says_so(tmp_path):
    busy = _box(tmp_path, "taken", room=96 * G, busy=True)
    idle = _box(tmp_path, "free", room=24 * G)
    try:
        said: list[str] = []
        got = plan(["m.gguf", "n.gguf"], [busy.peer, idle.peer], needs={"m.gguf": 8 * G,
                                                                      "n.gguf": 50 * G},
                   log=said.append)
        assert got[idle.peer] == ["m.gguf"]
        assert got[busy.peer] == []
        assert got.unplaced[0][0] == "n.gguf"
        assert "taken: busy" in got.unplaced[0][1]
    finally:
        for box in (busy, idle):
            box.runner.shutdown()
            box.httpd.shutdown()
            box.httpd.server_close()


def test_the_dispatcher_counts_as_a_peer(tmp_path, boxes):
    roomy, small = boxes
    me = here(name="desk", home=tmp_path / "desk-bench")
    me.host.room = lambda: 128 * G
    got = plan(["big.gguf"], [roomy.peer, small.peer, me], needs={"big.gguf": 100 * G},
               log=lambda _line: None)
    assert got[me] == ["big.gguf"]
    me.host.runner.shutdown()


def test_a_model_of_unknown_size_goes_to_the_roomiest_idle_peer(boxes):
    """Unknown is not enormous: the peer's own preflight sizes it before the load."""
    roomy, small = boxes
    said: list[str] = []
    got = plan(["mystery.gguf"], [roomy.peer, small.peer], needs={"mystery.gguf": 0},
               log=said.append)
    assert got[roomy.peer] == ["mystery.gguf"]
    assert "size unknown" in "\n".join(said)


def test_jobs_from_gives_each_peer_only_its_serves(boxes):
    roomy, small = boxes
    planned = {roomy.peer: ["big.gguf"], small.peer: ["tiny.gguf"]}
    base = Job(argv=("sweep", "--short", "--shortlist", "8"), models=(), commit=COMMIT,
               kept_label="tuesday", needs={"big.gguf": 60 * G, "tiny.gguf": 4 * G},
               files={"--graph": "{}"})
    jobs = jobs_from(planned, base, drafts={"big.gguf": "auto"})
    assert jobs[roomy.peer].argv == ("sweep", "--short", "--shortlist", "8",
                                     "--serve", "big.gguf", "--serve-draft", "auto")
    assert jobs[small.peer].argv == ("sweep", "--short", "--shortlist", "8",
                                     "--serve", "tiny.gguf")
    assert jobs[small.peer].needs == {"tiny.gguf": 4 * G}
    assert jobs[small.peer].name == "bench:tuesday"
    assert jobs[small.peer].files == jobs[roomy.peer].files == {"--graph": "{}"}


@pytest.mark.parametrize("flag", ["--kept", "--detach", "--no-queue", "--graph",
                                  "--questions"])
def test_a_job_may_not_carry_what_the_peer_owns(flag):
    with pytest.raises(ValueError, match=flag):
        Job(argv=("sweep", flag, "x"), models=("m",), commit=COMMIT)


# -- estimate ------------------------------------------------------------------------
def test_estimate_is_the_preflights_weights_and_kv_plus_the_head_and_the_allowance(tmp_path):
    from ml_stack.serve.preflight import RUNTIME_ALLOWANCE_BYTES
    from ml_stack.testing.fakes import FakePreflight

    weights = tmp_path / "invented-4B.gguf"
    weights.write_bytes(b"w" * 4096)
    head = tmp_path / "invented-4B.draft.gguf"
    head.write_bytes(b"h" * 512)
    fake = FakePreflight(weights_bytes=5 * G, kv_estimate_bytes=3 * G)

    got = estimate(str(weights), preflight=fake, binary="llama-server")

    assert got == 5 * G + 3 * G + 512 + RUNTIME_ALLOWANCE_BYTES
    assert str(fake.seen[0].draft) == str(head), "the head beside the weights is served"


def test_estimate_of_a_model_that_is_nowhere_is_unknown_not_enormous(tmp_path):
    assert estimate(str(tmp_path / "absent.gguf"), binary="llama-server") == 0


# -- refusals ------------------------------------------------------------------------
def test_a_daemon_refuses_a_mismatched_commit(boxes):
    roomy, _ = boxes
    with pytest.raises(Refused) as caught:
        submit_bench(roomy.peer, _job("m.gguf", commit="ff00ee1"))
    assert caught.value.kind == "commit"
    assert "roomy runs ml-stack ab12cd3, the dispatcher ff00ee1" in str(caught.value)
    # what the wire says: 409 with the reason and which refusal it was
    with pytest.raises(PeerError) as raw:
        roomy.peer._json("POST", "/bench", _job("m.gguf", commit="ff00ee1").public())
    assert "409" in str(raw.value) and '"refused": "commit"' in str(raw.value)


def test_a_dirty_tree_on_one_side_is_still_the_same_commit(boxes):
    roomy, _ = boxes
    got = submit_bench(roomy.peer, _job("m.gguf", commit=f"{COMMIT} (dirty)"))
    assert got["state"] == "preparing"


def test_a_daemon_refuses_while_its_measuring_lock_is_held(boxes):
    from ml_stack.lock import only_one

    roomy, _ = boxes
    with only_one(roomy.home / "measuring.lock"):
        assert roomy.peer.health()["measuring"] is True
        with pytest.raises(Refused) as caught:
            submit_bench(roomy.peer, _job("m.gguf"))
    assert caught.value.kind == "lock"
    assert "measuring.lock is held by pid" in str(caught.value)
    assert roomy.peer.health()["measuring"] is False


def test_a_daemon_refuses_a_second_job_while_its_own_is_running(boxes):
    roomy, _ = boxes
    first = submit_bench(roomy.peer, _job("m.gguf"))
    with pytest.raises(Refused) as caught:
        submit_bench(roomy.peer, _job("n.gguf"))
    assert caught.value.kind == "lock"
    assert first["id"] in str(caught.value)


def test_a_daemon_refuses_a_model_that_does_not_fit_its_room(boxes):
    _, small = boxes
    with pytest.raises(Refused) as caught:
        submit_bench(small.peer, _job("big.gguf", needs={"big.gguf": 60 * G}))
    assert caught.value.kind == "room"
    assert "big.gguf needs 60.0G and small may use 24.0G" in str(caught.value)
    assert small.host.launch.calls == [], "nothing was started"


def test_a_malformed_job_is_400_not_409(boxes):
    roomy, _ = boxes
    with pytest.raises(PeerError) as raw:
        roomy.peer._json("POST", "/bench", {"argv": "sweep --kept x", "commit": COMMIT})
    assert "400" in str(raw.value) and "--kept" in str(raw.value)


# -- dispatch and wait ---------------------------------------------------------------
def test_dispatch_and_wait_see_done_and_the_job_is_listed_like_any_other(boxes):
    roomy, small = boxes
    jobs = {roomy.peer: _job("big.gguf"), small.peer: _job("tiny.gguf")}
    said: list[str] = []

    handles = dispatch(jobs, log=said.append)
    assert [h.state for h in handles] == ["preparing", "preparing"], "accepted at once"
    assert all(h.id and h.log for h in handles)
    assert _await(lambda: roomy.peer.job(handles[0].id)["state"] == "running")
    assert roomy.peer.health()["measuring"] is True
    assert roomy.peer.health()["busy"] is True, "a measurement holds the GPU"
    listed = {j["id"]: j for j in roomy.peer.jobs()}
    assert listed[handles[0].id]["name"] == "bench:sweep"
    assert listed[handles[0].id]["argv"][:2] == ["ml-stack-bench", "sweep"]

    done = wait(handles, poll_s=0.1, timeout_s=20, log=said.append)

    assert [h.state for h in done] == ["done", "done"]
    assert roomy.peer.health()["measuring"] is False
    text = "\n".join(said)
    assert "roomy: bench:sweep preparing (job" in text and "roomy: bench:sweep running\n" in text
    assert "roomy: bench:sweep done" in text and "small: bench:sweep done" in text
    assert "kept as bench:tried" in text, "the log tail is printed when a job ends"
    assert "kept as bench:tried" in roomy.peer.log(handles[0].id)
    assert roomy.host.launch.calls == [["sweep", "--short", "--serve", "big.gguf"]]


def test_a_job_ships_its_graph_and_questions_and_the_peer_reads_its_own_copy(boxes):
    roomy, _ = boxes
    job = Job(argv=("sweep", "--short", "--serve", "big.gguf"), models=("big.gguf",),
              commit=COMMIT, files={"--graph": '{"nodes": []}', "--questions": "q?\n"})
    assert Job.from_request(json.loads(json.dumps(job.public()))) == job
    with pytest.raises(ValueError, match="--store"):
        Job(argv=("sweep",), models=(), commit=COMMIT, files={"--store": "x"})

    (handle,) = dispatch({roomy.peer: job}, log=lambda _l: None)

    assert _await(lambda: roomy.host.launch.calls)
    (line,) = roomy.host.launch.calls
    given = roomy.home / "given" / handle.id
    assert line == ["sweep", "--short", "--serve", "big.gguf",
                    "--graph", str(given / "graph.json"),
                    "--questions", str(given / "questions.jsonl")]
    assert (given / "graph.json").read_text() == '{"nodes": []}'
    assert (given / "questions.jsonl").read_text() == "q?\n"
    wait([handle], poll_s=0.1, timeout_s=20, log=lambda _l: None)
    assert _await(lambda: not given.exists()), "removed once the job ends"


def test_a_peer_nobody_answers_for_is_said_and_the_rest_go_on(boxes, tmp_path):
    roomy, _ = boxes
    gone = Peer(f"http://127.0.0.1:{_free_port()}", "no-token")
    said: list[str] = []

    placed = plan(["big.gguf"], [gone, roomy.peer], needs={"big.gguf": 4 * G},
                  log=said.append)
    assert placed[roomy.peer] == ["big.gguf"] and "did not answer" in "\n".join(said)

    handles = dispatch({gone: _job("tiny.gguf"), roomy.peer: _job("big.gguf")},
                       log=said.append)
    assert handles[0].state == "refused" and handles[0].why.startswith("unreachable: ")
    assert handles[1].state == "preparing"
    wait(handles, poll_s=0.1, timeout_s=20, log=said.append)
    handles[0].id = "never-there"
    got = gather(handles, into=tmp_path / "home.ladybug", log=said.append)
    assert "could not export" in "\n".join(said) and list(got) == ["id-roomy"]


def test_a_bundle_answers_the_commit_it_was_built_from(tmp_path, monkeypatch):
    from ml_stack.fleet import measuring

    monkeypatch.setattr(measuring, "__file__", str(tmp_path / "measuring.py"))
    (tmp_path / measuring.BUILT_FROM).write_text("ab12cd3 (dirty)\n")
    assert installed_commit() == "ab12cd3 (dirty)"
    assert same_commit(installed_commit(), COMMIT)


def test_a_bench_runs_on_this_interpreter_unless_the_app_is_frozen(boxes, monkeypatch):
    from ml_stack.fleet.measuring import bench_python

    roomy, _ = boxes
    (handle,) = dispatch({roomy.peer: _job("big.gguf")}, log=lambda _l: None)
    assert roomy.host.launch.pythons == [Path(sys.executable)]
    wait([handle], poll_s=0.1, timeout_s=20, log=lambda _l: None)

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(RuntimeError, match="What this machine can train with"):
        bench_python(None)
    said: list[str] = []
    (failed,) = wait(dispatch({roomy.peer: _job("big.gguf")}, log=said.append),
                     poll_s=0.1, timeout_s=20, log=said.append)
    assert failed.state == "failed"
    assert "error: roomy could not start ml-stack-bench" in "\n".join(said)
    assert "Measuring" in "\n".join(said)


def _wheel(where: Path, name: str, version: str, extras: tuple[str, ...],
           modules: tuple[str, ...] = ()) -> Path:
    """A wheel named ``name`` at ``version`` holding an empty file at each of ``modules``."""
    import base64
    import hashlib
    import zipfile

    stem = name.replace("-", "_")
    info = f"{stem}-{version}.dist-info"
    files = {f"{info}/METADATA": "Metadata-Version: 2.1\n" f"Name: {name}\nVersion: {version}\n"
             + "".join(f"Provides-Extra: {e}\n" for e in extras),
             f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
                              "Tag: py3-none-any\n",
             **dict.fromkeys(modules, "")}
    record = "".join(
        f"{path},sha256="
        f"{base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b'=').decode()}"
        f",{len(text.encode())}\n" for path, text in files.items()) + f"{info}/RECORD,,\n"
    made = where / f"{stem}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(made, "w") as out:
        for path, text in {**files, f"{info}/RECORD": record}.items():
            out.writestr(path, text)
    return made


def _slow_index(wheel: Path, *, delay_s: float):
    """A package index on loopback that serves ``wheel`` after ``delay_s``: (url, server)."""
    from http.server import BaseHTTPRequestHandler

    class Index(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.rstrip("/").endswith("/simple/ml-stack"):
                body = f'<a href="/files/{wheel.name}">{wheel.name}</a>'.encode()
                kind = "text/html"
            elif self.path == f"/files/{wheel.name}":
                time.sleep(delay_s)
                body, kind = wheel.read_bytes(), "application/octet-stream"
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Index)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/simple", server


@pytest.mark.slow
def test_a_frozen_peer_installs_the_bench_after_accepting_the_job(boxes, tmp_path, monkeypatch):
    """The install runs after ``POST /bench`` has answered, so an install slower than the
    dispatcher's peer timeout is ``preparing``, not an unreachable peer."""
    from ml_stack.fleet.environment import Environment

    roomy, _ = boxes
    environment = Environment(tmp_path / "managed")
    environment.create()
    # what `_measures` imports: the bench's store reader, and the store itself
    stand_in = _wheel(tmp_path, "ml-stack", "0.0.1", ("graph", "store", "serve", "hub"),
                      ("ml_stack/bench/__init__.py", "ml_stack/bench/peer_runs.py", "ladybug.py"))
    index, server = _slow_index(stand_in, delay_s=4.0)
    monkeypatch.setenv("PIP_INDEX_URL", index)
    monkeypatch.setenv("PIP_NO_CACHE_DIR", "1")
    monkeypatch.setenv("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    roomy.runner.environment = environment
    impatient = Peer(roomy.peer.base_url, roomy.peer.token, timeout=2.0)
    said: list[str] = []
    try:
        began = time.monotonic()
        (handle,) = dispatch({impatient: _job("big.gguf")}, log=said.append)
        assert handle.state == "preparing" and time.monotonic() - began < 2.0
        (done,) = wait([handle], poll_s=0.2, timeout_s=120, log=said.append)
    finally:
        server.shutdown()
    assert done.state == "done", "\n".join(said)
    assert time.monotonic() - began > 4.0, "the install was slower than the peer timeout"
    assert roomy.host.launch.pythons == [environment.python]
    assert environment.installed().get("ml-stack") == "0.0.1"
    assert "unreachable" not in "\n".join(said)
    assert [ln.split()[-1] for ln in said if "bench:sweep" in ln][-2:] == ["running", "done"]


def test_an_export_reads_the_store_through_peer_runs(tmp_path):
    from ml_stack.bench.peer_runs import exported

    store = tmp_path / "runs.ladybug"
    _kept(store, "kept-before", "2020-01-01T00:00:00")
    _kept(store, "kept-over-another", "2026-09-02T10:00:00", invented=False)
    _kept(store, "kept-after", "2026-09-02T11:00:00")
    since = "2026-09-02T00:00:00"
    flat = exported(store, since=since)
    assert [r["label"] for r in flat["runs"]] == ["kept-after"] and flat["skipped"] == 1
    assert "rows" not in flat["runs"][0]
    whole = exported(store, since=since, full=True, anyway=True)
    assert sorted(r["label"] for r in whole["runs"]) == ["kept-after", "kept-over-another"]
    assert all(r["rows"] for r in whole["runs"])
    said = subprocess.run([sys.executable, "-m", "ml_stack.bench.peer_runs"],
                          input=json.dumps({"store": str(store), "since": since}),
                          capture_output=True, text=True, check=True).stdout
    assert json.loads(said) == flat


@pytest.mark.slow
def test_a_real_bench_that_refuses_its_only_model_is_failed_with_the_refusal(tmp_path):
    from ml_stack.testing.fakes import fake_llama_binary

    runner = JobRunner(tmp_path / "traind", tmp_path / "files")
    host = BenchHost(runner, home=tmp_path / "bench", name="lone")
    host.poll_s = 0.2
    binary = fake_llama_binary(tmp_path)
    job = Job(argv=("sweep", "--serve", str(tmp_path / "absent.gguf"), "--plain-only",
                    "--no-smoke", "--store", "", "--binary", str(binary)),
              models=("absent.gguf",), commit=host.commit)
    try:
        (handle,) = dispatch({Local(host): job}, log=lambda _l: None)
        (done,) = wait([handle], poll_s=0.2, timeout_s=120, log=lambda _l: None)
        record = runner.jobs[handle.id]
        assert done.state == "failed", Path(record.log).read_text()
        assert "error: nothing measured: absent: preflight refused: FAIL  shards" in \
            Path(record.log).read_text()
    finally:
        runner.shutdown()


def test_a_bench_that_says_error_is_failed(tmp_path):
    box = _box(tmp_path, "shaky", room=96 * G, launch=scripted_launch(fails=True))
    try:
        said: list[str] = []
        handles = dispatch({box.peer: _job("m.gguf")}, log=said.append)
        done = wait(handles, poll_s=0.1, timeout_s=20, log=said.append)
        assert done[0].state == "failed"
        assert "error: boom" in "\n".join(said)
        assert box.peer.job(handles[0].id)["returncode"] == 1
    finally:
        box.runner.shutdown()
        box.httpd.shutdown()
        box.httpd.server_close()


def test_a_refusal_is_a_handle_not_an_exception(boxes):
    roomy, small = boxes
    said: list[str] = []
    handles = dispatch({small.peer: _job("big.gguf", needs={"big.gguf": 60 * G}),
                        roomy.peer: _job("big.gguf", needs={"big.gguf": 60 * G})},
                       log=said.append)
    assert handles[0].state == "refused" and handles[0].why.startswith("room:")
    assert handles[1].state == "preparing"
    assert "small: refused (room)" in "\n".join(said)
    wait(handles, poll_s=0.1, timeout_s=20, log=said.append)


def test_stopping_a_bench_job_terminates_the_detached_pid(tmp_path):
    box = _box(tmp_path, "slow", room=96 * G, launch=scripted_launch(seconds=30))
    try:
        handle = dispatch({box.peer: _job("m.gguf")}, log=lambda _l: None)[0]
        pid = box.peer.job(handle.id)["pid"]
        box.peer.stop(handle.id)
        assert box.peer.job(handle.id)["state"] == "stopped"
        from ml_stack.fleet.measuring import _alive

        assert _await(lambda: not _alive(pid), timeout=10), "the pid should be gone"
        time.sleep(0.3)
        assert box.peer.job(handle.id)["state"] == "stopped", "the watcher must not unsettle it"
    finally:
        box.runner.shutdown()
        box.httpd.shutdown()
        box.httpd.server_close()


def test_the_log_tail_names_what_failed():
    from ml_stack.fleet.measuring import FAILED_MARKS

    assert ended_badly(Path("/nonexistent/log")) == "no log was written"
    assert all(isinstance(m, str) for m in FAILED_MARKS)


# -- gather --------------------------------------------------------------------------
def _kept(store: Path, label: str, at: str, *, invented: bool = True, hits: int = 2) -> str:
    """A run in ``store`` as `save` would keep it: rows over the invented community."""
    from ml_stack.bench import invented_digest
    from ml_stack.graph.store import GraphStore

    rows = [{"label": label, "question": f"q{n}?", "expected": ["person:iris"],
             "shown": ["person:iris"] if n < hits else [], "seconds": 3.0, "calls": 2,
             "prompt_tokens": 900, "cached_tokens": 300, "processed_tokens": 600,
             "completion_tokens": 80, "answer_chars": 120, "error": "", "steps": ""}
            for n in range(4)]
    server = {"model": "invented-4B.gguf", "context": 32768, "slots": 1, "binary": "/b/llama",
              "graph": invented_digest() if invented else "someone-elses-graph"}
    key = f"bench:{label}:{at.replace('-', '').replace(':', '')}"
    with GraphStore(store) as held:
        held.put_doc(key, {"at": at, "label": label, "server": server, "rows": rows})
    return key


def _later(seconds: float) -> str:
    return time.strftime("%FT%T", time.localtime(time.time() + seconds))


def test_gather_imports_each_peers_runs_with_host_set_and_skips_duplicates(boxes, tmp_path):
    from ml_stack.bench import runs

    roomy, small = boxes
    into = tmp_path / "home.ladybug"
    handles = dispatch({roomy.peer: _job("big.gguf"), small.peer: _job("tiny.gguf")},
                       log=lambda _l: None)
    # kept while the jobs ran: one each, and one before the job began, which stays there
    _kept(roomy.store, "big-plain", _later(30))
    _kept(roomy.store, "big-old", "2020-01-01T00:00:00")
    _kept(small.store, "tiny-plain", _later(30))
    wait(handles, poll_s=0.1, timeout_s=20, log=lambda _l: None)
    said: list[str] = []

    got = gather(handles, into=into, log=said.append)

    assert sorted(got) == ["id-roomy", "id-small"]
    assert len(got["id-roomy"]) == 1 and len(got["id-small"]) == 1
    kept = {r["key"]: r for r in runs(into)}
    assert set(kept) == {*got["id-roomy"], *got["id-small"]}
    home = kept[got["id-roomy"][0]]
    assert home["label"] == "big-plain"
    assert home["server"]["host"] == "roomy"
    assert home["server"]["machine"] == "id-roomy"
    assert home["server"]["commit"] == COMMIT
    assert home["server"]["model"] == "invented-4B.gguf", "the peer's server record survives"
    assert len(home["rows"]) == 4, "the whole run comes home, rows and all"
    assert got["id-roomy"][0].endswith("@roomy")
    assert "roomy: imported 1 run(s)" in "\n".join(said)

    again = gather(handles, into=into, log=said.append)
    assert again == {"id-roomy": [], "id-small": []}
    assert len(runs(into)) == 2, "nothing imported twice, nothing overwritten"
    assert "1 already there" in "\n".join(said)


def test_gather_brings_home_a_run_over_another_graph_and_export_here_still_holds_it(
        boxes, tmp_path):
    from ml_stack.bench import runs
    from ml_stack.bench.score import _exportable

    roomy, _ = boxes
    into = tmp_path / "home.ladybug"
    handles = dispatch({roomy.peer: _job("big.gguf")}, log=lambda _l: None)
    _kept(roomy.store, "big-real", _later(30), invented=False)
    _kept(roomy.store, "big-plain", _later(31))
    wait(handles, poll_s=0.1, timeout_s=20, log=lambda _l: None)

    got = gather(handles, into=into, log=lambda _l: None)

    back = {r["label"]: r for r in runs(into)}
    assert len(got["id-roomy"]) == 2 and set(back) == {"big-real", "big-plain"}
    assert back["big-real"]["server"]["graph"] == "someone-elses-graph"
    exported, held = _exportable(list(back.values()))
    assert [r["label"] for r in exported] == ["big-plain"] and held == 1


def test_a_peer_that_kept_nothing_is_said_not_skipped_silently(boxes, tmp_path):
    roomy, _ = boxes
    handles = dispatch({roomy.peer: _job("big.gguf")}, log=lambda _l: None)
    wait(handles, poll_s=0.1, timeout_s=20, log=lambda _l: None)
    said: list[str] = []
    got = gather(handles, into=tmp_path / "home.ladybug", log=said.append)
    assert got == {"id-roomy": []}
    assert "roomy: kept no run since" in "\n".join(said)


def test_two_machines_of_one_name_gather_as_two_machines(tmp_path):
    from ml_stack.bench import runs
    from ml_stack.bench.score import machine_of

    made = [_box(tmp_path / where, "Mac", room=24 * G) for where in ("one", "two")]
    made[1].host.machine = "id-Mac-other"
    try:
        into = tmp_path / "home.ladybug"
        handles = dispatch({box.peer: _job("tiny.gguf") for box in made},
                           log=lambda _l: None)
        assert [h.host for h in handles] == ["Mac", "Mac"]
        assert [h.machine for h in handles] == ["id-Mac", "id-Mac-other"]
        at = _later(30)
        for box in made:
            _kept(box.store, "tiny-plain", at)
        wait(handles, poll_s=0.1, timeout_s=20, log=lambda _l: None)

        got = gather(handles, into=into, log=lambda _l: None)

        assert sorted(got) == ["id-Mac", "id-Mac-other"]
        assert all(len(keys) == 1 for keys in got.values()), got
        kept = runs(into)
        assert sorted(machine_of(r) for r in kept) == ["id-Mac", "id-Mac-other"]
        assert {r["server"]["host"] for r in kept} == {"Mac"}
    finally:
        for box in made:
            box.runner.shutdown()
            box.httpd.shutdown()
            box.httpd.server_close()


def test_the_export_route_answers_the_flat_shape_show_export_writes(boxes):
    roomy, _ = boxes
    _kept(roomy.store, "big-plain", "2026-09-02T10:00:00")
    flat = bench_export(roomy.peer, since="2026-09-02T00:00:00", full=False)
    assert flat["host"] == "roomy" and flat["commit"] == COMMIT
    (one,) = flat["runs"]
    assert one["label"] == "big-plain" and one["questions"] == 4
    assert one["f1"] == 0.5 and "rows" not in one
    nothing = bench_export(roomy.peer, since="2026-09-03T00:00:00", full=False)
    assert nothing["runs"] == []


def test_import_runs_by_hand_from_a_flat_export_file(tmp_path):
    """A peer with no daemon: `ml-stack-bench show --export` there, copy, import here."""
    from ml_stack.bench import derived, runs

    exported = tmp_path / "attic.json"
    exported.write_text(json.dumps([
        {"at": "2026-09-02T09:00:00", "label": "attic-plain", "questions": 20, "f1": 0.81,
         "recall": 0.9, "precision": 0.75, "lit_per_question": 2.4, "seconds": 300,
         "calls": 60, "read_tokens": 30000, "written_tokens": 4000, "draft_offered": 0,
         "draft_kept": 0, "speedup": None, "timed_out": 0, "context": 32768, "slots": 1,
         "cache_type": "", "reasoning_budget": None, "model": "invented-9B.gguf",
         "draft_model": "", "binary": "/b/llama", "load_s": 12.5, "resident_bytes": None,
         "kv_and_run_bytes": 3 * G, "mmapped": False, "sampling": {"temperature": 0.0},
         "finder": "vectors", "unread_named": 1, "concurrency": None},
    ]))
    into = tmp_path / "home.ladybug"

    keys = import_runs(exported, into, host="attic", commit="ab12cd3", log=lambda _l: None)

    (one,) = runs(into)
    assert keys == [one["key"]] and one["key"].startswith("bench:attic-plain:20260902T090000@attic")
    assert one["server"] == {"model": "invented-9B.gguf", "binary": "/b/llama",
                             "context": 32768, "slots": 1, "load_s": 12.5,
                             "kv_and_run_bytes": 3 * G, "sampling": {"temperature": 0.0},
                             "finder": "vectors", "host": "attic", "commit": "ab12cd3"}
    got = derived(one)
    assert got["right"] == 0.81 and got["seconds"] == 300 and got["paid_tokens"] == 34000
    assert got["questions"] == 20 and got["kv_bytes"] == 3 * G
    assert round(got["right_per_minute"], 3) == round(0.81 * 60 / 300, 3)
    assert one["totals"]["unread_named"] == 1

    assert import_runs(exported, into, host="attic", log=lambda _l: None) == []
    assert import_runs(exported, into, host="cellar", log=lambda _l: None) != [], \
        "the same run measured on another host is another run"


def test_import_runs_takes_the_json_text_and_the_answer_shape_too(tmp_path):
    from ml_stack.bench import runs

    into = tmp_path / "home.ladybug"
    text = json.dumps({"runs": [{"at": "2026-09-02T09:00:00", "label": "x", "questions": 2,
                                 "f1": 1.0, "seconds": 10}], "commit": "ff00ee1"})
    keys = import_runs(text, into, host="attic", log=lambda _l: None)
    assert len(keys) == 1
    assert runs(into)[0]["server"]["commit"] == "ff00ee1", "the answer's commit is taken"


# -- the dispatcher as a peer, end to end --------------------------------------------
def test_a_local_peer_goes_through_the_same_path(tmp_path):
    me = Local(name="desk", home=tmp_path / "desk")
    me.host.commit = COMMIT
    me.host.room = lambda: 64 * G
    me.host.launch = scripted_launch()
    me.host.poll_s = 0.1
    try:
        handles = dispatch({me: _job("m.gguf")}, log=lambda _l: None)
        assert handles[0].state == "preparing"
        assert me.health()["measuring"] is True and me.health()["busy"] is True
        done = wait(handles, poll_s=0.1, timeout_s=20, log=lambda _l: None)
        assert done[0].state == "done"
        _kept(me.host.home / "runs.ladybug", "m-plain", _later(30))
        got = gather(handles, into=tmp_path / "home.ladybug", log=lambda _l: None)
        assert len(got[me.host.machine]) == 1
    finally:
        me.host.runner.shutdown()


def test_a_handle_with_no_job_has_nothing_to_gather(tmp_path):
    refused = Handle(peer=None, job=_job("m.gguf"), state="refused", why="room: no")
    assert gather([refused], into=tmp_path / "home.ladybug", log=lambda _l: None) == {}


def test_a_detached_bench_is_told_the_home_whose_lock_the_daemon_watches(tmp_path, monkeypatch):
    """A daemon rooted away from ``~/.ml-stack`` watches ``<root.parent>/bench``; the bench it
    launches must record there too, or the daemon holds a lock nobody takes."""
    from ml_stack.fleet import measuring as fb

    seen = {}

    def run(argv, **kw):
        seen["argv"], seen["env"] = argv, kw.get("env") or {}
        home = Path(seen["env"]["MLSTACK_BENCH_HOME"])
        home.mkdir(parents=True, exist_ok=True)
        (home / "measuring.json").write_text(json.dumps({"pid": 4242, "log": str(home / "x.log")}))
        return subprocess.CompletedProcess(argv, 0, "log: " + str(home / "x.log"), "")

    monkeypatch.setattr(fb.subprocess, "run", run)
    pid, log = fb.detach_bench(["run", "m.gguf"], tmp_path / "bench", Path("/env/bin/python"))
    assert pid == 4242 and log == tmp_path / "bench" / "x.log"
    assert seen["argv"][:3] == ["/env/bin/python", "-m", "ml_stack.bench"]
    assert seen["env"]["MLSTACK_BENCH_HOME"] == str(tmp_path / "bench")
    assert "PATH" in seen["env"]                     # the rest of the environment came along


def test_the_bench_home_moves_with_the_environment(tmp_path):
    """Read in a fresh interpreter: reloading the module in this one would mint a second
    `RunNotKept` class and break every later `pytest.raises` on the first (CI, 2026-09-02)."""
    import os

    env = {**os.environ, "MLSTACK_BENCH_HOME": str(tmp_path / "elsewhere")}
    said = subprocess.run([sys.executable, "-c",
                           "from ml_stack.bench import keep; print(keep.home_dir())"],
                          capture_output=True, text=True, env=env, check=True).stdout.strip()
    assert Path(said) == tmp_path / "elsewhere"


# -- ops.fleet_planned / fleet_measure: the ml-stack-bench sweep --fleet side -----------
#
# `bench/ops.py` is the glue between `ml-stack-bench sweep --fleet` and the fleet
# functions above: it has to turn `--peers` names (or none) into the real peers `plan`
# needs, and the placement `plan` returns into the ``{peer: Job}`` mapping `dispatch`
# needs. Discovery itself (real UDP, a real cluster key) is `tests/test_fleet_discovery.py`'s
# job; here `ml_stack.fleet.join.peers` and `ml_stack.fleet.pausing.peer_clients` are stood
# in for, so what is real is exactly what was broken: `plan`, `jobs_from`, `dispatch`,
# `wait` and `gather`, over the same two daemons as the rest of this module.

def _discovery_stub(monkeypatch, clients):
    """`ml_stack.fleet.join.peers` and `ml_stack.fleet.pausing.peer_clients` answering with
    ``clients`` (``{name: Peer}``), so `ops._discovered` runs for real over them."""
    import ml_stack.fleet.join as join_module
    import ml_stack.fleet.pausing as pausing_module

    monkeypatch.setattr(join_module, "peers", lambda **kw: [{"name": n} for n in clients])
    monkeypatch.setattr(pausing_module, "peer_clients", lambda rows, **kw: dict(clients))


def test_fleet_planned_and_measure_spread_real_jobs_over_discovered_peers(boxes, tmp_path,
                                                                          monkeypatch):
    """The bug: `fleet_planned` handed `sweeps.plan` the ``--peers`` names as strings (or
    None), and `fleet_measure` handed `sweeps.dispatch` a list where it takes
    ``{peer: Job}``. Both are real peers and a real mapping here."""
    import ml_stack.fleet.sweeps as sweeps_module
    from ml_stack.bench import ops, runs

    roomy, small = boxes
    monkeypatch.setattr(ops, "_commit", lambda root=None: COMMIT)
    _discovery_stub(monkeypatch, {"roomy": roomy.peer, "small": small.peer})
    real_wait = sweeps_module.wait
    monkeypatch.setattr(sweeps_module, "wait",
                        lambda handles: real_wait(handles, poll_s=0.1, timeout_s=20))

    argv = ["sweep", "--fleet", "--plain-only", "--kept", "unused", "--serve", "big.gguf",
            "--serve", "tiny.gguf"]
    planned = ops.fleet_planned(argv, ["big.gguf", "tiny.gguf"])

    by_peer = {roomy.peer: "roomy", small.peer: "small"}
    named = {by_peer[peer]: job for peer, job in planned.jobs.items()}
    assert set(named) == {"roomy", "small"}, "one real Job per real peer, not a list"
    assert named["roomy"].argv == ("sweep", "--plain-only", "--serve", "big.gguf"), \
        "--kept never rides to a peer: it keeps its own runs in its own store"
    assert named["small"].argv == ("sweep", "--plain-only", "--serve", "tiny.gguf")
    said = "\n".join(planned.lines)
    assert "plan: 2 model(s) on commit ab12cd3 over roomy, small" in said
    assert "  big.gguf -> roomy" in said and "  tiny.gguf -> small" in said

    _kept(roomy.store, "big-plain", _later(30))
    _kept(small.store, "tiny-plain", _later(30))
    into = tmp_path / "home.ladybug"

    ops.fleet_measure(planned.jobs, into=into)

    assert sorted(r["label"] for r in runs(into)) == ["big-plain", "tiny-plain"]


def test_fleet_planned_refuses_before_dispatch_when_the_assigned_peer_is_on_another_commit(
        boxes, monkeypatch):
    from ml_stack.bench import ops
    from ml_stack.bench.ops import Refused as OpsRefused

    roomy, small = boxes
    monkeypatch.setattr(ops, "_commit", lambda root=None: "ffffff (dirty)")
    _discovery_stub(monkeypatch, {"roomy": roomy.peer, "small": small.peer})

    with pytest.raises(OpsRefused) as caught:
        ops.fleet_planned(["sweep", "--fleet", "--serve", "m.gguf"], ["m.gguf"])
    assert f"roomy is on commit {COMMIT}, this checkout is on ffffff (dirty)" in caught.value.error
    assert any(line.startswith("plan:") for line in caught.value.said)


def test_fleet_planned_refuses_a_named_peer_discovery_did_not_find(boxes, monkeypatch):
    from ml_stack.bench import ops
    from ml_stack.bench.ops import Refused as OpsRefused

    roomy, small = boxes
    monkeypatch.setattr(ops, "_commit", lambda root=None: COMMIT)
    _discovery_stub(monkeypatch, {"roomy": roomy.peer, "small": small.peer})

    with pytest.raises(OpsRefused, match="lantern"):
        ops.fleet_planned(["sweep", "--fleet", "--serve", "m.gguf"], ["m.gguf"],
                          peers=["lantern"])


def test_fleet_planned_refuses_a_store_on_this_machine_and_passes_an_empty_one(boxes,
                                                                                 monkeypatch):
    from ml_stack.bench import ops
    from ml_stack.bench.ops import Refused as OpsRefused

    roomy, _ = boxes
    monkeypatch.setattr(ops, "_commit", lambda root=None: COMMIT)
    _discovery_stub(monkeypatch, {"roomy": roomy.peer})

    with pytest.raises(OpsRefused, match=r"--store /here/graph\.ladybug is a graph store"):
        ops.fleet_planned(["sweep", "--fleet", "--store", "/here/graph.ladybug",
                           "--serve", "m.gguf"], ["m.gguf"])
    planned = ops.fleet_planned(["sweep", "--fleet", "--store=", "--serve", "m.gguf"],
                                ["m.gguf"])
    assert planned.jobs[roomy.peer].argv == ("sweep", "--store=", "--serve", "m.gguf")


def test_fleet_planned_refuses_when_discovery_finds_nobody(monkeypatch):
    from ml_stack.bench import ops
    from ml_stack.bench.ops import Refused as OpsRefused

    monkeypatch.setattr(ops, "_commit", lambda root=None: COMMIT)
    _discovery_stub(monkeypatch, {})

    with pytest.raises(OpsRefused, match="no peer answered discovery"):
        ops.fleet_planned(["sweep", "--fleet", "--serve", "m.gguf"], ["m.gguf"])


# -- the whole thing, with nothing stood in but the model -------------------------------

_OWN_GRAPH = {
    "nodes": [
        {"id": "topic:kiln", "kind": "topic", "label": "kiln", "mentions": 2, "attrs": {}},
        {"id": "person:wren", "kind": "person", "label": "Wren Tallis", "mentions": 1,
         "attrs": {}},
    ],
    "edges": [{"source": "person:wren", "target": "topic:kiln", "rel": "interested_in",
               "weight": 1}],
}
_OWN_QUESTIONS = [{"q": "who fires the kiln?", "expect": ["person:wren"]},
                  {"q": "what is Wren Tallis interested in?", "expect": ["topic:kiln"]}]

_FLEET_LLAMA_META = {
    "general.architecture": "llama",
    "llama.block_count": 2,
    "llama.attention.head_count_kv": 2,
    "llama.attention.key_length": 8,
}


def _free_port(kind: int = socket.SOCK_STREAM) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1" if kind == socket.SOCK_STREAM else "", 0))
        return s.getsockname()[1]


def _boot_daemon(tmp_path: Path, name: str, *, keyfile: Path, disco_port: int):
    """A real ``ml-stack-fleet`` daemon in a subprocess, once it answers /health: (proc,
    log, open log handle)."""
    from ml_stack.fleet.discovery import derive_token, load_cluster_key

    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    http_port = _free_port()
    log = tmp_path / f"{name}.out"
    fh = log.open("wb")
    env = {**os.environ, "ML_STACK_DISCOVERY_PORT": str(disco_port),
           "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
           "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "ml_stack.fleet.daemon", "--root", str(root / "traind"),
         "--bench-home", str(root / "bench"), "--host", "127.0.0.1",
         "--port", str(http_port), "--name", name, "--cluster-key", str(keyfile)],
        env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=root)
    driver = Peer(f"http://127.0.0.1:{http_port}", derive_token(load_cluster_key(keyfile)))
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            if driver.health().get("ok"):
                return proc, log, fh
        except PeerError:
            if proc.poll() is not None:
                pytest.fail(f"{name} traind died:\n{log.read_text(errors='replace')}")
        time.sleep(0.1)
    proc.kill()
    pytest.fail(f"{name} traind never answered /health:\n{log.read_text(errors='replace')}")


def _own_graph(where: Path) -> list[str]:
    """``--graph`` and ``--questions`` over `_OWN_GRAPH`, written under ``where``."""
    where.mkdir(parents=True, exist_ok=True)
    graph = where / "own.json"
    graph.write_text(json.dumps(_OWN_GRAPH))
    asked = where / "own.jsonl"
    asked.write_text("".join(json.dumps(q) + "\n" for q in _OWN_QUESTIONS))
    return ["--graph", str(graph), "--questions", str(asked)]


@pytest.mark.slow
def test_sweep_fleet_discovers_a_real_daemon_and_measures_on_it_for_real(tmp_path, monkeypatch):
    """Nothing here is mocked: a real ``ml-stack-fleet`` daemon booted as a subprocess, found
    by real UDP discovery, given a real HTTP job that runs the real ``ml-stack-bench`` over a
    graph that is not the shipped community, on a llama-server-shaped process instead of a
    GPU; the run it keeps comes home. The graph and questions are gone from the
    dispatcher's disk before the job is sent, so the peer reads only what the job carried."""
    import shutil

    from conftest import write_gguf

    import ml_stack.bench as bench
    import ml_stack.fleet.sweeps as sweeps_module
    from ml_stack.bench import runs
    from ml_stack.fleet.discovery import create_cluster_key
    from ml_stack.graph.cache import digest
    from ml_stack.testing.fakes import fake_llama_binary

    keyfile = tmp_path / "cluster.key"
    create_cluster_key(keyfile)
    disco_port = _free_port(socket.SOCK_DGRAM)
    booted = [_boot_daemon(tmp_path, name, keyfile=keyfile, disco_port=disco_port)
              for name in ("quill", "lantern")]
    try:
        monkeypatch.setenv("ML_STACK_CLUSTER_KEY", str(keyfile))
        monkeypatch.setenv("ML_STACK_DISCOVERY_PORT", str(disco_port))
        monkeypatch.setenv("MLSTACK_BENCH_HOME", str(tmp_path / "dispatcher-home"))

        gguf = write_gguf(tmp_path / "tiny.gguf", _FLEET_LLAMA_META)
        binary = fake_llama_binary(tmp_path)
        kept = tmp_path / "runs.ladybug"
        dispatcher = tmp_path / "dispatcher"
        real_dispatch = sweeps_module.dispatch

        def dispatch_without_the_files(jobs, **kw):
            shutil.rmtree(dispatcher)
            return real_dispatch(jobs, **kw)

        monkeypatch.setattr(sweeps_module, "dispatch", dispatch_without_the_files)
        argv = ["sweep", "--fleet", "--serve", str(gguf), "--binary", str(binary),
                *_own_graph(dispatcher), "--store", "", "--plain-only", "--no-smoke",
                "--no-profile", "--kept", str(kept), "--serve-port", str(_free_port())]
        code = bench._main(argv)
        assert code == 0, "\n".join(p[1].read_text(errors="replace") for p in booted)

        kept_runs = runs(kept)
        assert kept_runs, "the fleet measured nothing"
        assert kept_runs[0]["server"]["host"] in ("quill", "lantern")
        assert kept_runs[0]["server"]["graph"] == digest(_OWN_GRAPH) != bench.invented_digest()
        assert len(kept_runs[0]["rows"]) == len(_OWN_QUESTIONS), "the shipped questions"
        assert not dispatcher.exists()
        for name in ("quill", "lantern"):
            given = tmp_path / name / "bench" / "given"
            assert not given.exists() or not any(given.iterdir()), \
                f"{name} removes what a job shipped once the job ends"
    finally:
        for proc, _log, fh in booted:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            fh.close()
