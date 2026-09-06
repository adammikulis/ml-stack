"""What `ml-stack-bench status` says while a measurement is running.

The record is written when the measuring lock is taken and retired when it is released,
so a run started in the foreground is as visible as a detached one, and it says how it is
asking -- the sampling, the draft head, the cache, the thinking budget, the shape.
"""

from __future__ import annotations

import json
import os

import pytest

from tests.test_graph_bench import TINY, _Scripted


def _measurable(tmp_path, monkeypatch):
    """A bench home, a tiny graph and one question: the arguments of a `run` that measures."""
    pytest.importorskip("ladybug", reason="the store needs ml-stack[store]")
    import ml_stack.bench as bench

    monkeypatch.setenv("MLSTACK_BENCH_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(bench, "footprint", lambda url: {"base_url": url})
    monkeypatch.setattr(bench, "ask_from", lambda spec: _Scripted)
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps(TINY))
    asked = tmp_path / "q.jsonl"
    asked.write_text(json.dumps({"q": "who works on compilers?",
                                 "expect": ["topic:compiler"]}) + "\n")
    return ["--kept", str(tmp_path / "runs.ladybug"), "--graph", str(graph),
            "--questions", str(asked), "--client", "fake:client", "--no-smoke",
            "--no-selfcheck", "--no-prefetch"]


def test_a_run_in_the_foreground_is_measuring_and_says_how_it_asks(tmp_path, monkeypatch,
                                                                   capsys):
    """The gap this closes: a run started without `--detach` held the lock and the GPU
    while `status` said nothing was measuring, so the only way to learn its temperature
    was to read the source."""
    import ml_stack.bench as bench
    from ml_stack.bench import run as running

    line = _measurable(tmp_path, monkeypatch)
    seen: list[str] = []
    real_asking = bench.asking

    def watched(graph, **kw):
        seen.append(running.status(results=False))
        return real_asking(graph, **kw)

    monkeypatch.setattr(bench, "asking", watched)
    assert bench.main(["run", "in-the-foreground", *line]) == 0
    capsys.readouterr()

    said = seen[0]
    assert f"(pid {os.getpid()})" in said and "measuring for " in said
    assert "ml-stack-bench run in-the-foreground" in said
    assert "asking: temperature 0.0 (greedy), n_predict 16384" in said
    assert "log: none -- it prints to the terminal it was started in" in said
    assert "nothing is measuring" in running.status(results=False)


def test_a_run_asked_for_a_temperature_says_that_one(tmp_path, monkeypatch, capsys):
    import ml_stack.bench as bench
    from ml_stack.bench import run as running

    line = _measurable(tmp_path, monkeypatch)
    seen: list[str] = []
    real_asking = bench.asking

    def watched(graph, **kw):
        seen.append(running.status(results=False))
        return real_asking(graph, **kw)

    monkeypatch.setattr(bench, "asking", watched)
    assert bench.main(["run", "warm", *line, "--temperature", "0.7", "--top-p", "0.9"]) == 0
    capsys.readouterr()
    assert "asking: temperature 0.7, top_p 0.9, n_predict 16384" in seen[0]


def test_a_record_whose_process_has_gone_is_not_measuring(tmp_path, monkeypatch):
    monkeypatch.setenv("MLSTACK_BENCH_HOME", str(tmp_path / "home"))
    from ml_stack.bench import run as running

    running.remember(["sweep", "--serve", "models/beacon.gguf"], pid=os.getpid())
    assert running.measuring() is not None

    held = json.loads(running.measuring_file().read_text(encoding="utf-8"))
    held["pid"] = 2 ** 22 + 7                                          # nobody's
    running.measuring_file().write_text(json.dumps(held), encoding="utf-8")
    assert running.measuring() is None
    assert "nothing is measuring; the last one" in running.status(results=False)


def test_a_run_that_is_killed_leaves_no_live_record(tmp_path, monkeypatch, capsys):
    """SIGTERM becomes a `SystemExit`, so the lock's block runs its exit and the record it
    wrote is retired with it."""
    monkeypatch.setenv("MLSTACK_BENCH_HOME", str(tmp_path / "home"))
    from ml_stack.bench import run as running

    def killed(rest):
        assert running.measuring() is not None
        running._stop_on_sigterm(15, None)

    monkeypatch.setattr(running, "_main", killed)
    monkeypatch.setattr(running, "_estimated", lambda rest: 0)
    with pytest.raises(SystemExit):
        running.main(["run", "stopped", "--no-selfcheck", "--no-prefetch"])
    capsys.readouterr()
    assert running.measuring() is None
    assert json.loads(running.measuring_file().read_text(encoding="utf-8"))["ended"]


def test_the_record_says_the_shape_a_sweep_will_serve_in():
    from ml_stack.bench.run import _how_said, asking_said

    how = asking_said(["sweep", "--serve", "models/flash.gguf", "--serve-draft", "auto",
                       "--n-max", "4", "--serve-kv", "f16", "--reasoning-budget", "2048",
                       "--context", "131072", "--parallel", "4"])
    assert how["sampling"] == {"temperature": 0.0, "n_predict": 16384}
    assert how["head"] == "auto" and how["head_ahead"] == 4
    assert how["cache_type"] == "f16" and how["reasoning_budget"] == 2048
    assert how["context"] == 131072 and how["slots"] == 4
    assert _how_said(how)[1] == ("  shape: draft head auto; 4 ahead; f16 cache; "
                                "thinking budget 2048; 128k context across 4 seats")


def test_the_record_says_every_arm_a_drafts_run_will_serve():
    """`drafts` names its heads under --draft and a depth per arm, and the baseline arm is
    the empty head."""
    from ml_stack.bench.run import _how_said, asking_said

    how = asking_said(["drafts", "flash.gguf", "--draft", "", "--draft", "heads/mtp-a.gguf",
                       "--n-max", "2", "--n-max", "8", "--reasoning-budget", "0"])
    assert how["head"] == "none, mtp-a.gguf" and how["head_ahead"] == "2, 8"
    assert _how_said(how)[1] == ("  shape: draft heads none, mtp-a.gguf; 2, 8 ahead; "
                                "q8_0 cache; thinking budget 0; 32k context across 1 seat")


def test_a_sweep_told_to_drop_the_head_says_it_has_none():
    from ml_stack.bench.run import _how_said, asking_said

    how = asking_said(["sweep", "--serve", "models/flash.gguf", "--serve-draft", "auto",
                       "--no-draft"])
    assert how["head"] == ""
    assert "no draft head; q8_0 cache" in _how_said(how)[1]


def test_a_detached_run_keeps_the_log_its_parent_opened_for_it(tmp_path, monkeypatch):
    """The child writes its own record when it takes the lock; the log belongs to the
    parent that opened it."""
    monkeypatch.setenv("MLSTACK_BENCH_HOME", str(tmp_path / "home"))
    from ml_stack.bench import run as running

    running.remember(["sweep", "--serve", "models/beacon.gguf"], pid=os.getpid(),
                     log=str(tmp_path / "sweep.log"), started="2026-09-05T10:00:00")
    again = running.remember(["sweep", "--serve", "models/beacon.gguf"], pid=os.getpid())
    assert again["log"] == str(tmp_path / "sweep.log")
    assert again["started"] == "2026-09-05T10:00:00"
