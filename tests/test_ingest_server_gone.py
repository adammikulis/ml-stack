"""When the model server goes away mid-shelf, the run stops -- it does not write two hundred
failures in a minute, and none of them count against the units."""

from ml_stack import ingest
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


def test_a_second_detach_is_refused_while_the_recorded_run_is_still_ending(tmp_path, monkeypatch, capsys):
    """A run beside one still folding its way out adopted its server and lost it."""
    import json
    import subprocess
    import sys

    monkeypatch.setattr(ingest, "HOME", tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (tmp_path / "ingesting.json").write_text(json.dumps({"pid": child.pid, "argv": []}))
        assert ingest.main(["book.pdf", "--out", str(tmp_path / "s"), "--detach"]) == 2
        assert "still running or still folding" in capsys.readouterr().err
    finally:
        child.kill()
        child.wait()


def test_stop_keeps_the_record_while_the_run_is_still_folding(tmp_path, capsys):
    import json
    import subprocess
    import sys

    # a child that ignores SIGTERM for a while: a fold on its way out
    child = subprocess.Popen([sys.executable, "-c",
                              "import signal, sys, time; "
                              "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                              "print('ready', flush=True); time.sleep(60)"], stdout=subprocess.PIPE)
    try:
        assert child.stdout.readline().strip() == b"ready", "the handler is in place"
        (tmp_path / "ingesting.json").write_text(json.dumps({"pid": child.pid, "argv": []}))
        assert ingest.stop(home=tmp_path, wait=1.0) == 1
        out = capsys.readouterr().out
        assert "still being written" in out and "no new run starts beside it" in out
        assert (tmp_path / "ingesting.json").exists(), "the record stays until it has ended"
        assert ingest._recorded_alive(tmp_path) == child.pid
    finally:
        child.kill()
        child.wait()
