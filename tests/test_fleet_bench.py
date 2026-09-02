"""The fleet side of ``ml-stack-bench sweep --fleet``: real daemons on loopback, a bench
that is a small script rather than a model, and stores in ``tmp_path``.

Two fake daemons are two real ``ThreadingHTTPServer``s running the real handler over a
real `JobRunner`; only three things are stood in for -- the memory a peer may use, the
commit it runs, and the launch of ``ml-stack-bench --detach``, which would load a model
and write under ``~/.ml-stack``. Nothing here reads that directory. Every name is invented.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from ml_stack.fleet.bench import (
    BenchHost,
    Handle,
    Job,
    Local,
    Refused,
    bench_export,
    dispatch,
    ended_badly,
    estimate,
    gather,
    here,
    import_runs,
    installed_commit,
    jobs_from,
    plan,
    same_commit,
    submit_bench,
    wait,
)
from ml_stack.fleet.daemon import JobRunner, load_or_create_token, make_handler
from ml_stack.fleet.remote import Peer, PeerError

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

    def launch(line, home: Path):
        calls.append(list(line))
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


def _box(tmp_path: Path, name: str, *, room: int, commit: str = COMMIT, launch=None,
         busy: bool = False) -> Box:
    root = tmp_path / name
    files = root / "files"
    files.mkdir(parents=True)
    home = root / "bench"
    token = load_or_create_token(root)
    runner = JobRunner(root, files)
    host = BenchHost(runner, home=home, commit=commit, room=lambda: room,
                     launch=launch or scripted_launch(), name=name, poll_s=0.1)

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
               needs=needs or {m: 4 * G for m in models})


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
    jobs = jobs_from(planned, ["sweep", "--short", "--shortlist", "8"], commit=COMMIT,
                     needs={"big.gguf": 60 * G, "tiny.gguf": 4 * G},
                     drafts={"big.gguf": "auto"}, kept_label="tuesday")
    assert jobs[roomy.peer].argv == ("sweep", "--short", "--shortlist", "8",
                                     "--serve", "big.gguf", "--serve-draft", "auto")
    assert jobs[small.peer].argv == ("sweep", "--short", "--shortlist", "8",
                                     "--serve", "tiny.gguf")
    assert jobs[small.peer].needs == {"tiny.gguf": 4 * G}
    assert jobs[small.peer].name == "bench:tuesday"


@pytest.mark.parametrize("flag", ["--kept", "--detach", "--no-queue"])
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
    assert got["state"] == "running"


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
    assert [h.state for h in handles] == ["running", "running"]
    assert all(h.id and h.log for h in handles)
    assert roomy.peer.health()["measuring"] is True
    assert roomy.peer.health()["busy"] is True, "a measurement holds the GPU"
    listed = {j["id"]: j for j in roomy.peer.jobs()}
    assert listed[handles[0].id]["name"] == "bench:sweep"
    assert listed[handles[0].id]["argv"][:2] == ["ml-stack-bench", "sweep"]

    done = wait(handles, poll_s=0.1, timeout_s=20, log=said.append)

    assert [h.state for h in done] == ["done", "done"]
    assert roomy.peer.health()["measuring"] is False
    text = "\n".join(said)
    assert "roomy: bench:sweep running (job" in text
    assert "roomy: bench:sweep done" in text and "small: bench:sweep done" in text
    assert "kept as bench:tried" in text, "the log tail is printed when a job ends"
    assert "kept as bench:tried" in roomy.peer.log(handles[0].id)
    assert roomy.host.launch.calls == [["sweep", "--short", "--serve", "big.gguf"]]


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
    assert handles[1].state == "running"
    assert "small: refused (room)" in "\n".join(said)
    wait(handles, poll_s=0.1, timeout_s=20, log=said.append)


def test_stopping_a_bench_job_terminates_the_detached_pid(tmp_path):
    box = _box(tmp_path, "slow", room=96 * G, launch=scripted_launch(seconds=30))
    try:
        handle = dispatch({box.peer: _job("m.gguf")}, log=lambda _l: None)[0]
        pid = box.peer.job(handle.id)["pid"]
        box.peer.stop(handle.id)
        assert box.peer.job(handle.id)["state"] == "stopped"
        from ml_stack.fleet.bench import _alive

        assert _await(lambda: not _alive(pid), timeout=10), "the pid should be gone"
        time.sleep(0.3)
        assert box.peer.job(handle.id)["state"] == "stopped", "the watcher must not unsettle it"
    finally:
        box.runner.shutdown()
        box.httpd.shutdown()
        box.httpd.server_close()


def test_the_log_tail_names_what_failed():
    from ml_stack.fleet.bench import FAILED_MARKS

    assert ended_badly(Path("/nonexistent/log")) == "no log was written"
    assert all(isinstance(m, str) for m in FAILED_MARKS)


# -- gather --------------------------------------------------------------------------
def _kept(store: Path, label: str, at: str, *, invented: bool = True, hits: int = 2) -> str:
    """A run in ``store`` as `save` would keep it: rows over the invented community."""
    from ml_stack.graph.bench import invented_digest
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
    from ml_stack.graph.bench import runs

    roomy, small = boxes
    into = tmp_path / "home.ladybug"
    handles = dispatch({roomy.peer: _job("big.gguf"), small.peer: _job("tiny.gguf")},
                       log=lambda _l: None)
    # kept while the jobs ran: one each, plus one before the job began and one over some
    # other graph, neither of which may come home
    _kept(roomy.store, "big-plain", _later(30))
    _kept(roomy.store, "big-old", "2020-01-01T00:00:00")
    _kept(roomy.store, "big-real", _later(31), invented=False)
    _kept(small.store, "tiny-plain", _later(30))
    wait(handles, poll_s=0.1, timeout_s=20, log=lambda _l: None)
    said: list[str] = []

    got = gather(handles, into=into, log=said.append)

    assert sorted(got) == ["roomy", "small"]
    assert len(got["roomy"]) == 1 and len(got["small"]) == 1
    kept = {r["key"]: r for r in runs(into)}
    assert set(kept) == {*got["roomy"], *got["small"]}
    home = kept[got["roomy"][0]]
    assert home["label"] == "big-plain"
    assert home["server"]["host"] == "roomy"
    assert home["server"]["commit"] == COMMIT
    assert home["server"]["model"] == "invented-4B.gguf", "the peer's server record survives"
    assert len(home["rows"]) == 4, "the whole run comes home, rows and all"
    assert got["roomy"][0].endswith("@roomy")
    assert "roomy: imported 1 run(s)" in "\n".join(said)

    again = gather(handles, into=into, log=said.append)
    assert again == {"roomy": [], "small": []}
    assert len(runs(into)) == 2, "nothing imported twice, nothing overwritten"
    assert "1 already there" in "\n".join(said)


def test_a_peer_that_kept_nothing_is_said_not_skipped_silently(boxes, tmp_path):
    roomy, _ = boxes
    handles = dispatch({roomy.peer: _job("big.gguf")}, log=lambda _l: None)
    wait(handles, poll_s=0.1, timeout_s=20, log=lambda _l: None)
    said: list[str] = []
    got = gather(handles, into=tmp_path / "home.ladybug", log=said.append)
    assert got == {"roomy": []}
    assert "roomy: kept no run since" in "\n".join(said)


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
    from ml_stack.graph.bench import derived, runs

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
    from ml_stack.graph.bench import runs

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
        assert handles[0].state == "running"
        assert me.health()["measuring"] is True and me.health()["busy"] is True
        done = wait(handles, poll_s=0.1, timeout_s=20, log=lambda _l: None)
        assert done[0].state == "done"
        _kept(me.host.home / "runs.ladybug", "m-plain", _later(30))
        got = gather(handles, into=tmp_path / "home.ladybug", log=lambda _l: None)
        assert len(got["desk"]) == 1
    finally:
        me.host.runner.shutdown()


def test_a_handle_with_no_job_has_nothing_to_gather(tmp_path):
    refused = Handle(peer=None, job=_job("m.gguf"), state="refused", why="room: no")
    assert gather([refused], into=tmp_path / "home.ladybug", log=lambda _l: None) == {}


def test_a_detached_bench_is_told_the_home_whose_lock_the_daemon_watches(tmp_path, monkeypatch):
    """A daemon rooted away from ``~/.ml-stack`` watches ``<root.parent>/bench``; the bench it
    launches must record there too, or the daemon holds a lock nobody takes."""
    from ml_stack.fleet import bench as fb

    seen = {}

    def run(argv, **kw):
        seen["env"] = kw.get("env") or {}
        home = Path(seen["env"]["MLSTACK_BENCH_HOME"])
        home.mkdir(parents=True, exist_ok=True)
        (home / "measuring.json").write_text(json.dumps({"pid": 4242, "log": str(home / "x.log")}))
        return subprocess.CompletedProcess(argv, 0, "log: " + str(home / "x.log"), "")

    monkeypatch.setattr(fb.subprocess, "run", run)
    pid, log = fb.detach_bench(["run", "m.gguf"], tmp_path / "bench")
    assert pid == 4242 and log == tmp_path / "bench" / "x.log"
    assert seen["env"]["MLSTACK_BENCH_HOME"] == str(tmp_path / "bench")
    assert "PATH" in seen["env"]                     # the rest of the environment came along


def test_the_bench_home_moves_with_the_environment(tmp_path):
    """Read in a fresh interpreter: reloading the module in this one would mint a second
    `RunNotKept` class and break every later `pytest.raises` on the first (CI, 2026-09-02)."""
    import os

    env = {**os.environ, "MLSTACK_BENCH_HOME": str(tmp_path / "elsewhere")}
    said = subprocess.run([sys.executable, "-c",
                           "from ml_stack.graph.bench import keep; print(keep.HOME)"],
                          capture_output=True, text=True, env=env, check=True).stdout.strip()
    assert Path(said) == tmp_path / "elsewhere"
