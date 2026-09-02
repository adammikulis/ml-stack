"""`ml-stack-bench history`: the day, read out of the logs directory and joined to the store.

Everything here is invented and lives in ``tmp_path`` -- four fake logs, a fake
`measuring.json` pointing at this test's own pid, and a store with two runs, one inside a
log's window and one outside every window. Nothing reads ``~/.ml-stack``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import ml_stack.graph.bench_history as bh

# A day in the invented company's life: four measurements, in the order they were started.
DAY = "2026-09-01"
FINISHED = f"{DAY}T10:00:00"        # a sweep that ran two hours and kept a run
KILLED = f"{DAY}T12:00:00"          # a run stopped ten minutes in
CRASHED = f"{DAY}T13:00:00"         # a drafts run that raised two minutes in
RUNNING = f"{DAY}T14:00:00"         # a sweep still going
NOW = f"{DAY}T14:30:00"             # when the test looks


def _epoch(iso: str) -> float:
    return time.mktime(time.strptime(iso, "%Y-%m-%dT%H:%M:%S"))


def _stamp(iso: str) -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.strptime(iso, "%Y-%m-%dT%H:%M:%S"))


def _log(logs: Path, sub: str, name: str, started: str, text: str, *, ended: str) -> Path:
    path = logs / f"{sub}-{name}-{_stamp(started)}.log"
    path.write_text(text, encoding="utf-8")
    os.utime(path, (_epoch(ended), _epoch(ended)))
    return path


def _store(path: Path, at: list[tuple[str, str]]) -> None:
    from ml_stack.graph.store import GraphStore

    with GraphStore(path) as writer:
        for n, (when, label) in enumerate(at):
            writer.put_doc(f"bench:{label}:{_stamp(when)}", {
                "at": when, "label": label,
                "server": {"model": f"models/quill-{n}.gguf", "port": 8080 + n},
                "rows": [{"question": "who runs the workshop", "seconds": 3.0, "calls": 1}]})


@pytest.fixture
def day(tmp_path: Path) -> Path:
    """A bench home holding the four logs, `measuring.json` and the runs store."""
    pytest.importorskip("ladybug")
    home = tmp_path / "home"
    logs = home / "logs"
    logs.mkdir(parents=True)
    _log(logs, "sweep", "quill", FINISHED,
         "argv: sweep --serve models/quill.gguf --also terse\n"
         f"started: {FINISHED}\n"
         "commit: 0f1e2d3 (dirty)\n"
         "estimate: 2h 15m for 2 ways x 1 model\n"
         "    up in 12s\n"
         "   4.1s   2 calls  who runs the workshop\n"
         "  11.9s   3 calls  which team owns the glasshouse\n"
         "   2.2s   1 calls  what is the parish council for\n"
         "quill-terse  ...\n",
         ended=f"{DAY}T12:00:00")
    _log(logs, "run", "inkwell", KILLED,
         "   5.0s   1 calls  who runs the workshop\n"
         "[killed] pid 4242 asked to stop by ml-stack-bench stop\n",
         ended=f"{DAY}T12:10:00")
    _log(logs, "drafts", "lantern-head", CRASHED,
         "# argv: drafts --serve models/lantern.gguf --draft auto\n"
         "    up in 9s\n"
         "Traceback (most recent call last):\n"
         '  File "bench.py", line 1, in <module>\n'
         "    raise RuntimeError(...)\n"
         "RuntimeError: the head would not load under this build\n",
         ended=f"{DAY}T13:02:00")
    running = _log(logs, "sweep", "beacon", RUNNING,
                   "   7.5s   2 calls  who runs the workshop\n",
                   ended=f"{DAY}T14:20:00")
    (home / "measuring.json").write_text(json.dumps({
        "pid": os.getpid(), "argv": ["sweep", "--serve", "models/beacon.gguf"],
        "log": str(running), "started": RUNNING}), encoding="utf-8")
    _store(home / "runs.ladybug", [(f"{DAY}T11:30:00", "quill-terse"),
                                    ("2026-08-20T09:00:00", "old-quill")])
    return home


def test_history_reads_one_entry_per_log_oldest_first(day: Path):
    got = bh.history(day, now=_epoch(NOW))
    assert [e.subcommand for e in got] == ["sweep", "run", "drafts", "sweep"]
    assert [e.started for e in got] == [FINISHED, KILLED, CRASHED, RUNNING]

    done, killed, crashed, running = got
    assert done.name == "quill" and done.exit == "done"
    assert done.argv == ["sweep", "--serve", "models/quill.gguf", "--also", "terse"]
    assert done.commit == "0f1e2d3 (dirty)"
    assert done.ended == f"{DAY}T12:00:00" and done.seconds == 7200.0
    assert done.estimate_s == 2 * 3600 + 15 * 60
    assert done.questions == 3
    assert done.kept == ["quill-terse"] and done.model == "quill-0.gguf"

    assert killed.name == "inkwell" and killed.exit == "killed"
    assert killed.seconds == 600.0 and killed.kept == [] and killed.questions == 1
    assert killed.argv == [] and killed.commit == ""        # nothing recorded them

    assert crashed.name == "lantern-head"                  # a name may carry a dash
    assert crashed.exit == "crashed: RuntimeError: the head would not load under this build"
    assert crashed.argv == ["drafts", "--serve", "models/lantern.gguf", "--draft", "auto"]
    assert crashed.seconds == 120.0 and crashed.estimate_s is None

    assert running.name == "beacon" and running.exit == "running"
    assert running.argv == ["sweep", "--serve", "models/beacon.gguf"]   # from measuring.json
    assert running.ended == NOW and running.seconds == 1800.0        # to now, not the mtime


def test_a_gone_pid_is_a_finished_measurement_not_a_running_one(day: Path):
    held = json.loads((day / "measuring.json").read_text())
    held["pid"] = 2 ** 22 + 7                                      # nobody's
    (day / "measuring.json").write_text(json.dumps(held))
    last = bh.history(day, now=_epoch(NOW))[-1]
    assert last.exit == "done" and last.ended == f"{DAY}T14:20:00" and last.seconds == 1200.0


def test_old_runs_outside_every_window_are_kept_by_nobody(day: Path):
    labels = [label for e in bh.history(day, now=_epoch(NOW)) for label in e.kept]
    assert labels == ["quill-terse"]


def test_no_logs_directory_is_an_empty_history(tmp_path: Path):
    assert bh.history(tmp_path / "nowhere") == []


def test_table_has_the_header_a_row_per_log_and_the_totals(day: Path, monkeypatch, capsys):
    monkeypatch.setattr(bh, "_now", lambda: _epoch(NOW))
    assert bh.main(["--home", str(day)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["started", "sub", "model/label", "est", "actual", "exit", "kept"]
    assert len(lines) == 1 + 4 + 1

    rows = [line.split() for line in lines[1:5]]
    assert rows[0][:7] == [FINISHED, "sweep", "quill", "2h15m", "2h00m", "done", "quill-terse"]
    assert rows[1][:7] == [KILLED, "run", "inkwell", "-", "10m", "killed", "-"]
    assert rows[2][:5] == [CRASHED, "drafts", "lantern-head", "-", "2m"]
    assert "crashed: RuntimeError: the head would not load" in lines[3]
    assert rows[3][:7] == [RUNNING, "sweep", "beacon", "-", "30m", "running", "-"]

    # 2h + 10m + 2m + 30m = 2.7 GPU hours; 12 minutes kept nothing, the running half hour
    # is not counted against anybody yet.
    assert lines[-1] == ("4 runs, 2.7 GPU hours, 0.2 hours produced no kept run (wasted); "
                         "1 still running, not counted as wasted")


def test_since_narrows_to_today_a_span_or_a_date(day: Path, monkeypatch, capsys):
    monkeypatch.setattr(bh, "_now", lambda: _epoch(NOW))

    assert bh.main(["--home", str(day), "--since", "today"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("4 runs")

    assert bh.main(["--home", str(day), "--since", "3h"]) == 0
    said = capsys.readouterr().out.splitlines()
    assert [row.split()[1] for row in said[1:-1]] == ["run", "drafts", "sweep"]
    assert said[-1].startswith("3 runs, 0.7 GPU hours, 0.2 hours produced no kept run")

    assert bh.main(["--home", str(day), "--since", f"{DAY}T13:00"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("2 runs")

    assert bh.main(["--home", str(day), "--since", "2026-09-02"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("0 runs, 0.0 GPU hours")

    assert bh.main(["--home", str(day), "--since", "yesterday-ish"]) == 2
    assert "--since takes today, 24h, 7d or a date" in capsys.readouterr().err


def test_json_dumps_the_entries(day: Path, monkeypatch, capsys):
    monkeypatch.setattr(bh, "_now", lambda: _epoch(NOW))
    assert bh.main(["--home", str(day), "--json", "--since", "24h"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert [e["exit"][:7] for e in got] == ["done", "killed", "crashed", "running"]
    assert got[0]["kept"] == ["quill-terse"] and got[0]["estimate_s"] == 8100.0
    assert set(got[0]) >= {"log", "started", "subcommand", "argv", "commit", "ended",
                           "seconds", "exit", "estimate_s", "kept", "questions"}


def test_the_default_home_is_the_benchs_and_is_faked_here(tmp_path: Path, monkeypatch, capsys):
    import ml_stack.graph.bench as bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")             # never ~/.ml-stack
    assert bh.main([]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("0 runs")


def test_runs_as_a_module(day: Path):
    done = subprocess.run([sys.executable, "-m", "ml_stack.graph.bench_history",
                           "--home", str(day), "--json"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout)
    assert len(got) == 4 and got[-1]["exit"] == "running"      # this pid is alive


@pytest.mark.parametrize("text, seconds", [
    ("2h 15m", 8100.0), ("90s", 90.0), ("1.5h", 5400.0), ("~600", 600.0),
    ("about 40 min, 2 ways", 2400.0), ("no number here", None),
])
def test_an_estimate_line_is_read_in_any_of_the_shapes_it_is_printed_in(text, seconds):
    assert bh.parse_duration(text) == seconds
