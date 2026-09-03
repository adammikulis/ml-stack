"""When the model server goes away mid-shelf, the run stops -- it does not write two hundred
failures in a minute, and none of them count against the units."""

from pathlib import Path

import pytest

from ml_stack import ingest, jobs
from tests.test_ingest import a_shelf


def test_an_unreachable_server_is_not_the_units_attempt(tmp_path):
    progress = ingest.Progress(ingest.Progress.beside(tmp_path / "shelf"))
    progress.book("velthorne", title="Velthorne", path="v.pdf", sections=1)
    fields = {"book": "velthorne", "chapter": "1", "section": "1.1", "title": "Vault Currents"}
    gone = ingest.Read(unit="velthorne:1:1.1#0", seconds=0.0,
                       error="ServerUnreachable: cannot reach http://127.0.0.1:1", **fields)
    progress.note("velthorne", gone)
    progress.note("velthorne", gone)
    progress.note("velthorne", gone)
    entry = progress.state["books"]["velthorne"]["done"][gone.unit]
    assert entry["attempts"] == 0 and entry["error"].startswith("ServerUnreachable")
    assert not progress.done("velthorne", gone.unit), "read again on the next --resume"
    assert progress.totals()["given_up"] == 0


def test_the_run_stops_when_its_server_is_gone(tmp_path, server, monkeypatch, capsys):
    book, instance, _ = a_shelf(tmp_path, server)
    store = tmp_path / "shelf.ladybug"
    seen = []

    def gone(client, unit, shape, **kw):
        seen.append(unit.id)
        return ingest.Read(unit=unit.id, book=unit.book, chapter=unit.chapter,
                           section=unit.section, title=unit.section_title,
                           error="ServerUnreachable: cannot reach it (Connection refused)")

    monkeypatch.setattr(ingest, "extract_unit", gone)
    monkeypatch.setattr(ingest, "_alive", lambda client: False)
    code = ingest.main([book, "--out", str(store), "--base-url", instance.base_url])
    out = capsys.readouterr().out
    assert len(seen) == 1, "one failed unit, then the run stopped"
    assert "went away" in out and "--resume reads on" in out
    assert code == 0
    progress = ingest.Progress(ingest.Progress.beside(store))
    entries = next(iter(progress.state["books"].values()))["done"]
    assert len(entries) == 1 and next(iter(entries.values()))["attempts"] == 0


def test_detach_records_the_run_as_this_machines_ingest_job(tmp_path, monkeypatch, capsys):
    """`detach` writes the pid, the argv and the log through `ml_stack.jobs`, and the same
    record is what `stop`, `wait` and `ml-stack-jobs status` read."""
    import subprocess
    import sys

    monkeypatch.setattr(ingest, "HOME", tmp_path)
    started = {}
    popen = subprocess.Popen

    def sleeping(command, **kw):
        started["command"] = list(command)
        started["child"] = held = popen([sys.executable, "-c", "import time; time.sleep(60)"])
        return held

    monkeypatch.setattr(subprocess, "Popen", sleeping)
    argv = ["book.pdf", "--out", str(tmp_path / "s"), "--detach"]
    try:
        assert ingest.main(argv) == 0
        assert started["command"][1:3] == ["-m", "ml_stack.ingest"], "the module, not a shell"
        assert "--detach" not in started["command"]
        held = jobs.held("ingest", home=tmp_path / "jobs")
        assert held["pid"] == started["child"].pid
        assert held["argv"] == [a for a in argv if a != "--detach"]
        assert held["log"].endswith(".log") and Path(held["log"]).is_file()
        assert jobs.alive("ingest", home=tmp_path / "jobs") == started["child"].pid
    finally:
        started["child"].kill()
        started["child"].wait()


def test_a_second_detach_is_refused_while_the_recorded_run_is_still_ending(tmp_path, monkeypatch, capsys):
    """A run beside one still folding its way out adopted its server and lost it."""
    import subprocess
    import sys

    monkeypatch.setattr(ingest, "HOME", tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        jobs.record("ingest", pid=child.pid, argv=[], home=tmp_path / "jobs")
        assert ingest.main(["book.pdf", "--out", str(tmp_path / "s"), "--detach"]) == 2
        assert "still running or still folding" in capsys.readouterr().err
    finally:
        child.kill()
        child.wait()


def test_stop_keeps_the_record_while_the_run_is_still_folding(tmp_path, capsys):
    import subprocess
    import sys

    # a child that ignores SIGTERM for a while: a fold on its way out
    child = subprocess.Popen([sys.executable, "-c",
                              "import signal, sys, time; "
                              "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                              "print('ready', flush=True); time.sleep(60)"], stdout=subprocess.PIPE)
    try:
        assert child.stdout.readline().strip() == b"ready", "the handler is in place"
        jobs.record("ingest", pid=child.pid, argv=[], home=tmp_path / "jobs")
        assert ingest.stop(home=tmp_path, wait=1.0) == 1
        out = capsys.readouterr().out
        assert "still being written" in out and "no new run starts beside it" in out
        assert (tmp_path / "jobs" / "ingest.json").exists(), "the record stays until it ends"
        assert ingest._recorded_alive(tmp_path) == child.pid
    finally:
        child.kill()
        child.wait()


def test_wait_blocks_until_the_recorded_run_has_ended(tmp_path, capsys):
    import subprocess
    import sys
    import threading

    assert ingest.wait(home=tmp_path) == 0
    assert "no detached ingest is running" in capsys.readouterr().out
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    jobs.record("ingest", pid=child.pid, argv=[], home=tmp_path / "jobs")
    done = []
    thread = threading.Thread(target=lambda: done.append(ingest.wait(home=tmp_path, every=1.0)))
    thread.start()
    thread.join(timeout=30)
    child.wait()
    assert done == [0] and "has ended" in capsys.readouterr().out


def test_a_judged_tidy_refuses_beside_a_live_run(tmp_path, monkeypatch, capsys):
    import subprocess
    import sys

    monkeypatch.setattr(ingest, "HOME", tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        jobs.record("ingest", pid=child.pid, argv=[], home=tmp_path / "jobs")
        assert ingest.main(["tidy", "--out", str(tmp_path / "s"), "--model", "x"]) == 2
        assert "wait` first" in capsys.readouterr().err
    finally:
        child.kill()
        child.wait()


def _leased(monkeypatch):
    """A fake `serve` that records whether its lease was released, on a bare `--model`."""
    import contextlib

    released = []

    class Up:
        base_url = "http://127.0.0.1:8099"

    @contextlib.contextmanager
    def fake_serve(model, manager=None, **lease):
        try:
            yield Up()
        finally:
            released.append(True)

    monkeypatch.setattr("ml_stack.serve.manager.serve", fake_serve)
    monkeypatch.setattr("ml_stack.serve.profile.profile_for", lambda m: None)
    monkeypatch.setattr(ingest, "_find_model", lambda m: "x.gguf")
    return released


def _terminated(*a, **k):
    """What a SIGTERM does mid-call: the handler in place is invoked, not the signal
    delivered, so a handler that is still the default fails the test rather than pytest."""
    import signal

    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler), "SIGTERM is not turned into Stopped here"
    handler(signal.SIGTERM, None)


def _gold(tmp_path):
    gold = tmp_path / "g.json"
    gold.write_text('{"passages": [{"passage_id": "p", "text": "Vault currents flow.", '
                    '"triples": []}]}')
    return ["--gold", str(gold), "--model", "x"]


def _ask(tmp_path, monkeypatch):
    (tmp_path / "s").mkdir()
    monkeypatch.setattr("ml_stack.ingest.cli.graph_of",
                        lambda out: {"nodes": [{"id": "concept:glimmer-node"}], "edges": []})
    return ["ask", "--out", str(tmp_path / "s"), "--model", "x", "what flows?"]


def _tidy(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "HOME", tmp_path)
    monkeypatch.setattr(ingest, "_judge", lambda *a, **k: None)
    return ["tidy", "--out", str(tmp_path / "s"), "--model", "x"]


@pytest.mark.parametrize("path", ["gold", "ask", "tidy"])
def test_a_sigterm_mid_run_releases_the_lease_on_every_path(tmp_path, monkeypatch, capsys,
                                                             path):
    """`tidy --model`, `ask` and `--gold` each hold a lease the way a read does; a stop
    reaching any of them takes the server down with it rather than leaving it up under
    nobody."""
    released = _leased(monkeypatch)
    if path == "gold":
        argv = _gold(tmp_path)
        monkeypatch.setattr("ml_stack.ingest.cli.gold_score", _terminated)
    elif path == "ask":
        argv = _ask(tmp_path, monkeypatch)
        monkeypatch.setattr(ingest, "ask", _terminated)
    else:
        argv = _tidy(tmp_path, monkeypatch)
        monkeypatch.setattr("ml_stack.graph.tidy.tidy", _terminated)
    assert ingest.main(argv) == 1
    assert released == [True], "the lease was not released"
    assert "stopped" in capsys.readouterr().out
