"""An evening of measurements read out of a file and run one at a time.

Every fixture here is invented and lives in ``tmp_path``: no store outside it is opened, no
model is served, and the steps are run by a fake in place of `queue.run_step`, so nothing
here starts an `ml-stack-bench` or touches a GPU.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

import ml_stack.graph.bench as bench
from ml_stack.graph.bench import queue as q

from conftest import a_row

FX = ("hf:hearthstone/Bellwether-12B-GGUF/UD-Q4_K_XL/"
      "Bellwether-12B-UD-Q4_K_XL-00001-of-00002.gguf")


def a_queue(tmp_path, text: str) -> pathlib.Path:
    where = tmp_path / "evening.queue"
    where.write_text(text, encoding="utf-8")
    return where


# -- reading the file ------------------------------------------------------------------------

def test_a_queue_is_one_bench_line_per_step_with_comments_and_blanks_ignored(tmp_path):
    steps = q.read(a_queue(tmp_path, """
        # the evening's plan
        sweep --serve raincoat-2b.gguf --sample 10

        show --rank docs/model-ranking.md
    """))
    assert [s.argv for s in steps] == [
        ["sweep", "--serve", "raincoat-2b.gguf", "--sample", "10"],
        ["show", "--rank", "docs/model-ranking.md"]]
    assert [s.n for s in steps] == [1, 2]
    assert [s.kind for s in steps] == ["step", "step"]
    assert steps[0].line == 3 and steps[1].line == 5


def test_a_set_line_defines_a_variable_that_later_lines_expand(tmp_path):
    steps = q.read(a_queue(tmp_path, f"""
        set FX={FX}
        set SHAPE=--serve-kv q8_0 --context 65536 --parallel 2
        set BEST=--serve ${{FX}} ${{SHAPE}} --plain-only

        sweep ${{BEST}} --sample 10
    """))
    assert steps[0].argv == ["sweep", "--serve", FX, "--serve-kv", "q8_0", "--context",
                             "65536", "--parallel", "2", "--plain-only", "--sample", "10"]


def test_a_variable_nothing_sets_falls_back_to_the_environment_and_then_is_refused(
        tmp_path, monkeypatch):
    monkeypatch.setenv("BELLWETHER_HOME", "/opt/invented")
    steps = q.read(a_queue(tmp_path, "sweep --serve ${BELLWETHER_HOME}/model.gguf --smoke"))
    assert steps[0].argv[2] == "/opt/invented/model.gguf"

    with pytest.raises(q.QueueError) as why:
        q.read(a_queue(tmp_path, "# a plan\nsweep --serve ${NOTHING_SETS_THIS}/m.gguf"))
    # the line, and the name, before anything is measured: an unset ${FX} would otherwise
    # serve the default model for six hours and call it a result
    assert "line 2" in str(why.value) and "${NOTHING_SETS_THIS}" in str(why.value)


def test_a_smoke_line_pairs_with_the_then_line_under_it(tmp_path):
    steps = q.read(a_queue(tmp_path, """
        smoke: sweep --serve raincoat-2b.gguf --smoke
        then:  sweep --serve raincoat-2b.gguf --sample 10
        sweep --serve raincoat-2b.gguf --yes
    """))
    assert [s.kind for s in steps] == ["smoke", "then", "step"]
    assert steps[0].argv[-1] == "--smoke" and steps[1].argv[-2:] == ["--sample", "10"]


def test_a_then_with_no_smoke_and_a_smoke_with_no_then_are_both_refused(tmp_path):
    with pytest.raises(q.QueueError, match="line 1: `then:` with no `smoke:`"):
        q.read(a_queue(tmp_path, "then: sweep --serve raincoat-2b.gguf --sample 10"))
    with pytest.raises(q.QueueError, match="line 1: `smoke:` with no `then:`"):
        q.read(a_queue(tmp_path, "smoke: sweep --serve raincoat-2b.gguf --smoke\n"))
    with pytest.raises(q.QueueError, match="line 1"):
        q.read(a_queue(tmp_path, "smoke: sweep --serve a.gguf --smoke\n"
                                 "smoke: sweep --serve b.gguf --smoke\n"
                                 "then: sweep --serve b.gguf --sample 10\n"))


def test_a_flag_that_does_not_exist_is_refused_as_the_file_is_read(tmp_path, capsys):
    """The whole point of reading the file first: a typo on the last line of a nine-step
    evening used to be found after the eighth measurement."""
    where = a_queue(tmp_path, "sweep --serve raincoat-2b.gguf --sample 10\n"
                              "sweep --serve raincoat-2b.gguf --sampel 10\n")
    with pytest.raises(q.QueueError) as why:
        q.read(where)
    assert "line 2" in str(why.value) and "--sampel" in str(why.value)
    assert bench.main(["queue", str(where)]) == 2       # and nothing ran
    assert "--sampel" in capsys.readouterr().err


def test_the_label_a_step_keeps_under_is_read_off_the_line(tmp_path):
    assert q.label_of(["run", "with-shortlist", "--sample", "10"]) == "with-shortlist"
    # a sweep labels every way it asks with the model's stem -- fourteen characters of the
    # file name -- and the suffix, which is what `--resume` matches on
    assert q.label_of(["sweep", "--serve", FX, "--label-suffix=-v2"]) == "Bellwether-12B-v2"
    assert q.label_of(["show", "--rank", "docs/model-ranking.md"]) == ""


# -- running it -------------------------------------------------------------------------------

class FakeBench:
    """An `ml-stack-bench` that runs nothing: it writes down the line and returns a code."""

    def __init__(self, codes: dict[str, int] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.codes = codes or {}

    def __call__(self, argv):
        self.calls.append(list(argv))
        for word, code in self.codes.items():
            if word in argv:
                return code
        return 0


def test_every_step_runs_in_order_with_one_summary_line_each(tmp_path, capsys):
    where = a_queue(tmp_path, """
        set FX=raincoat-2b.gguf
        sweep --serve ${FX} --sample 10
        show --rank ranking.md
    """)
    fake = FakeBench()
    assert q.run_queue(where, runner=fake) == 0
    assert fake.calls == [["sweep", "--serve", "raincoat-2b.gguf", "--sample", "10"],
                          ["show", "--rank", "ranking.md"]]
    said = capsys.readouterr().out
    lines = [ln for ln in said.splitlines() if ln.startswith("=== ")]
    assert len(lines) == 3                                   # two steps and the last word
    assert " step 1/2: sweep --serve raincoat-2b.gguf --sample -- ok (0s)" in lines[0]
    assert " step 2/2: show --rank ranking.md -- ok (0s)" in lines[1]
    assert lines[0].split()[1].count(":") == 2               # HH:MM:SS
    assert "2 step(s), 2 ok, 0 failed, 0 skipped" in lines[2]


def test_a_failed_smoke_skips_its_then_and_says_so_but_the_queue_goes_on(tmp_path, capsys):
    """The `&&` of the shell scripts, kept: a shape that will not load is not measured on a
    hundred questions, and the rest of the evening still happens."""
    where = a_queue(tmp_path, """
        smoke: sweep --serve broken.gguf --smoke
        then:  sweep --serve broken.gguf --sample 10
        smoke: sweep --serve raincoat-2b.gguf --smoke
        then:  sweep --serve raincoat-2b.gguf --sample 10
    """)
    fake = FakeBench(codes={"broken.gguf": 1})
    assert q.run_queue(where, runner=fake) == 1              # a step failed
    assert [c[2] for c in fake.calls] == ["broken.gguf", "raincoat-2b.gguf",
                                          "raincoat-2b.gguf"]
    said = capsys.readouterr().out
    assert " step 1/4: sweep --serve broken.gguf --smoke -- failed (0s): exit 1" in said
    assert " step 2/4:" in said and "-- skipped (0s): its smoke (step 1) failed" in said
    assert "4 step(s), 2 ok, 1 failed, 1 skipped" in said


def test_a_step_that_fails_on_its_own_does_not_stop_the_queue(tmp_path, capsys):
    where = a_queue(tmp_path, "sweep --serve broken.gguf --sample 10\n"
                              "show --rank ranking.md\n")
    fake = FakeBench(codes={"broken.gguf": 3})
    assert q.run_queue(where, runner=fake) == 1
    assert len(fake.calls) == 2
    assert "-- failed (0s): exit 3" in capsys.readouterr().out


def test_the_go_ahead_and_the_ceiling_are_passed_to_every_step_that_takes_them(tmp_path):
    """`--yes` once at the top of the file instead of typed onto every long line -- and
    never onto `show`, which does not take one."""
    where = a_queue(tmp_path, "sweep --serve raincoat-2b.gguf\n"
                              "sweep --serve raincoat-2b.gguf --yes --ceiling 5\n"
                              "show --rank ranking.md\n")
    fake = FakeBench()
    assert q.run_queue(where, runner=fake, yes=True, ceiling=90.0) == 0
    assert fake.calls[0] == ["sweep", "--serve", "raincoat-2b.gguf", "--yes",
                             "--ceiling", "90.0"]
    assert fake.calls[1] == ["sweep", "--serve", "raincoat-2b.gguf", "--yes", "--ceiling", "5"]
    assert fake.calls[2] == ["show", "--rank", "ranking.md"]


def test_dry_run_prints_the_steps_expanded_and_runs_none_of_them(tmp_path, capsys):
    where = a_queue(tmp_path, f"""
        set FX={FX}
        smoke: sweep --serve ${{FX}} --label-suffix=-v2 --smoke
        then:  sweep --serve ${{FX}} --label-suffix=-v2 --sample 10
    """)
    fake = FakeBench()
    assert q.run_queue(where, runner=fake, dry_run=True, yes=True) == 0
    assert fake.calls == []
    said = capsys.readouterr().out
    assert "2 step(s), 1 smoke/then pair(s)" in said
    assert f"ml-stack-bench sweep --serve {FX} --label-suffix=-v2 --smoke" in said
    assert "[label Bellwether-12B-v2]" in said
    assert "--yes" in said                                   # what would be added, shown
    assert "nothing ran: --dry-run" in said
    assert not q.state_file().exists()                       # and nothing was written down


def test_resume_skips_a_step_whose_label_the_store_already_holds(tmp_path, capsys):
    """A queue stopped half-way does not measure the first half again -- and what counts as
    done is a run in the store since this queue started, not a line in a log."""
    from ml_stack.graph.bench import save

    where = a_queue(tmp_path, "sweep --serve raincoat-2b.gguf --sample 10\n"
                              "sweep --serve bellwether-12b.gguf --sample 10\n")
    fake = FakeBench()
    assert q.run_queue(where, runner=fake) == 0
    kept = bench.HOME / "runs.ladybug"
    save(kept, [a_row("who welds?", expected=["person:iris"], shown=["person:iris"],
                      calls=2, chars=90, label="raincoat-2b.gg-plain")])

    again = FakeBench()
    assert q.run_queue(where, runner=again, resume=True) == 0
    assert [c[2] for c in again.calls] == ["bellwether-12b.gguf"]
    said = capsys.readouterr().out
    assert "-- skipped (0s): raincoat-2b is already kept since" in said
    assert "resuming" in said


def test_status_says_which_step_is_running_and_what_is_left(tmp_path, monkeypatch):
    """`ml-stack-bench status` reads the same file the queue writes as it goes, so what is
    running is a fact about the machine and not a line somebody remembers typing."""
    where = a_queue(tmp_path, "sweep --serve raincoat-2b.gguf --sample 10\n"
                              "sweep --serve bellwether-12b.gguf --sample 10\n"
                              "show --rank ranking.md\n")
    from ml_stack.graph.bench import run as running

    seen = []

    def looking(argv):
        seen.append(running.status())
        return 0

    assert q.run_queue(where, runner=looking) == 0
    while_running = seen[1]
    assert f"queue: {where}" in while_running
    assert f"(pid {os.getpid()})" in while_running
    assert "step 2/3: sweep --serve bellwether-12b.gguf --sample" in while_running
    assert "done: 1 ok, 0 failed, 0 skipped" in while_running
    assert "left: 1 -- next show --rank ranking.md" in while_running

    ended = running.status()                                 # and afterwards, one line
    assert f"queue: {where} -- ended" in ended and "3 step(s), 3 ok" in ended

    held = json.loads(q.state_file().read_text())
    assert held["total"] == 3 and [s["state"] for s in held["steps"]] == ["ok"] * 3


def test_the_queue_runs_each_step_as_its_own_bench_so_each_takes_the_lock_itself(
        tmp_path, monkeypatch):
    """Not one process running four sweeps: a step is `python -m ml_stack.graph.bench ...`,
    which is what puts it through the self-check, the estimate, the smoke and the measuring
    lock that already exist. Two steps therefore never share the GPU."""
    import subprocess
    import sys

    started = {}

    class FakeChild:
        def wait(self):
            return 0

        def poll(self):
            return 0

    def fake_popen(command, **kw):
        started["command"], started["kw"] = command, kw
        return FakeChild()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    assert q.run_step(["sweep", "--serve", "raincoat-2b.gguf"]) == 0
    assert started["command"] == [sys.executable, "-m", "ml_stack.graph.bench",
                                  "sweep", "--serve", "raincoat-2b.gguf"]
    assert started["kw"]["env"]["PYTHONUNBUFFERED"] == "1"
    assert "stdout" not in started["kw"]                     # the queue's log, unredirected


def test_the_whole_queue_detaches_the_way_a_run_does(tmp_path, monkeypatch, capsys):
    """One log for the evening, owned by no terminal, named after the queue file."""
    import subprocess
    import sys

    monkeypatch.setattr(bench.platform, "system", lambda: "Darwin")
    where = a_queue(tmp_path, "sweep --serve raincoat-2b.gguf --sample 10\n")
    started = {}
    monkeypatch.setattr(subprocess, "Popen",
                        lambda command, **kw: started.update(command=command, kw=kw)
                        or type("C", (), {"pid": 4343})())
    argv = ["queue", str(where), "--detach", "--yes"]
    assert bench.main(argv) == 0

    assert started["command"][:3] == [sys.executable, "-m", "ml_stack.graph.bench"]
    assert started["command"][3:] == ["queue", str(where), "--yes"]
    assert started["kw"]["start_new_session"] is True
    log = pathlib.Path(started["kw"]["stdout"].name)
    assert log.parent == bench.HOME / "logs" and "evening.queue" in log.name

    held = json.loads((bench.HOME / "measuring.json").read_text())
    assert held["pid"] == 4343 and held["argv"] == ["queue", str(where), "--yes"]

    said = capsys.readouterr().out
    assert "ml-stack-bench status" in said and "ml-stack-bench stop" in said


def test_the_queue_is_dispatched_by_the_command_and_returns_what_the_steps_did(
        tmp_path, monkeypatch, capsys):
    where = a_queue(tmp_path, "sweep --serve broken.gguf --sample 10\n")
    monkeypatch.setattr(q, "run_step", FakeBench(codes={"broken.gguf": 1}))
    assert bench.main(["queue", str(where)]) == 1
    assert "-- failed (0s): exit 1" in capsys.readouterr().out


# -- the evening that was nine shell scripts ---------------------------------------------------

def test_the_shipped_example_reads_as_the_evening_it_replaces(monkeypatch):
    """`docs/examples/flash-next-restart.queue` is queue7.sh of 2026-09-02 as a file: two
    smoke/then pairs, the hundred-question run, then the ranking and the report."""
    monkeypatch.setenv("HOME", "/Users/invented")
    where = (pathlib.Path(__file__).resolve().parent.parent
             / "docs" / "examples" / "flash-next-restart.queue")
    steps = q.read(where)

    assert [s.kind for s in steps] == ["smoke", "then", "smoke", "then", "step", "step",
                                       "step"]
    assert [s.argv[0] for s in steps] == ["sweep"] * 5 + ["show", "report"]
    assert all("--serve-draft" in s.argv and "auto" in s.argv for s in steps[:5])
    assert steps[0].argv[-1] == "--smoke" and steps[1].argv[-2:] == ["--sample", "10"]
    assert steps[4].label == steps[3].label == "Qwen3.8-Flash--all"
    assert "--yes" not in steps[4].argv                      # the queue's --yes, given once
    assert "/Users/invented/.ml-stack/" in " ".join(steps[0].argv)   # ${HOME}, expanded
    assert steps[5].argv[1] == "--rank" and steps[6].argv[0] == "report"


def _bench_swapped_for(monkeypatch, program: str):
    """`run_step`'s `python -m ml_stack.graph.bench ...` replaced by `python -c program`,
    with every other argument to Popen kept -- so the real pipe and reader are exercised."""
    import subprocess
    import sys

    real = subprocess.Popen

    def swapped(command, **kw):
        assert command[:3] == [sys.executable, "-m", "ml_stack.graph.bench"]
        return real([sys.executable, "-c", program], **kw)

    monkeypatch.setattr(subprocess, "Popen", swapped)


def test_a_step_that_dies_at_once_has_its_last_stderr_lines_in_the_queue_log(
        monkeypatch, capsys):
    """A step that fails before it has measured anything -- an import that is not there, a
    flag the parser refuses -- says why beside its summary line, not only "exit 1"."""
    _bench_swapped_for(monkeypatch, "import sys; "
                       "sys.stderr.write('Traceback (most recent call last):\\n'); "
                       "sys.stderr.write('ModuleNotFoundError: No module named "
                       "ml_stack.nowhere\\n'); sys.exit(1)")
    assert q.run_step(["sweep", "--serve", "raincoat-2b.gguf"]) == 1
    out = capsys.readouterr().out
    assert "died in 0." in out and "s:" in out
    assert "ModuleNotFoundError: No module named ml_stack.nowhere" in out


def test_a_step_that_ran_for_a_while_before_failing_is_not_called_a_fast_death(
        monkeypatch, capsys):
    monkeypatch.setattr(q, "FAST_DEATH_S", 0.05)
    _bench_swapped_for(monkeypatch, "import sys, time; time.sleep(0.3); "
                       "sys.stderr.write('later\\n'); sys.exit(1)")
    assert q.run_step(["sweep", "--serve", "raincoat-2b.gguf"]) == 1
    assert "died in" not in capsys.readouterr().out


def test_a_steps_stderr_still_reaches_the_log_as_it_runs(monkeypatch, capsys):
    _bench_swapped_for(monkeypatch, "import sys; sys.stderr.write('waiting for "
                       "measuring.lock, held by pid 7\\n'); sys.exit(0)")
    assert q.run_step(["sweep", "--serve", "raincoat-2b.gguf"]) == 0
    got = capsys.readouterr()
    assert "waiting for measuring.lock, held by pid 7" in got.err
    assert "died in" not in got.out
