"""scripts/cutover against real temporary git repositories, a fake HOME and stub commands.

`--dry-run-in DIR` puts HOME, the stub `old build` and every recorded call under DIR; nothing here reads or
writes the real home directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "cutover"
COMMIT = "chore: the guards, hook settings and person-record code take the Poolhouse names"


def git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=env).stdout


class Machine:
    """A checkout cloned from a bare origin, a fake HOME and the stub commands, all under one temp directory."""

    def __init__(self, tmp: Path) -> None:
        self.dir = tmp / "dry"
        self.home = self.dir / "home"
        self.home.mkdir(parents=True)
        origin = tmp / "origin.git"
        git(tmp, "init", "-q", "--bare", "-b", "dev", str(origin))
        self.repo = tmp / "repo"
        git(tmp, "clone", "-q", str(origin), str(self.repo))
        git(self.repo, "checkout", "-q", "-b", "dev")
        settings = self.repo / ".claude" / "settings.json"
        settings.parent.mkdir()
        settings.write_text('{"env": "ML_STACK_WINDOW_POSITION"}\n')
        (self.repo / "scripts").mkdir()
        shutil.copy(SCRIPT, self.repo / "scripts" / "cutover")
        (self.repo / "docs").mkdir()
        git(self.repo, "add", ".claude/settings.json", "scripts/cutover")
        git(self.repo, "commit", "-qm", "feat: the old build")
        settings.write_text('{"env": "POOLHOUSE_WINDOW_POSITION"}\n')
        (self.repo / "docs" / "rename-protected.patch").write_text(git(self.repo, "diff"))
        git(self.repo, "checkout", "--", ".claude/settings.json")
        git(self.repo, "add", "docs/rename-protected.patch")
        git(self.repo, "commit", "-qm", "docs: the patch")
        git(self.repo, "push", "-q", "-u", "origin", "dev")
        (self.home / ".ml-stack").mkdir()
        (self.dir / "installed").parent.mkdir(exist_ok=True)
        (self.dir / "installed").write_text("ml-stack\n")

    def run(self, *args: str, platform: str = "macos", stdin: str = "") -> subprocess.CompletedProcess:
        env = {**os.environ, "HOME": str(self.home), "CUTOVER_PLATFORM": platform}
        return subprocess.run(["sh", str(self.repo / "scripts" / "cutover"), *args, "--dry-run-in", str(self.dir)],
                              capture_output=True, text=True, input=stdin, env=env, timeout=60)

    def calls(self) -> list[str]:
        path = self.dir / "calls.log"
        return path.read_text().splitlines() if path.exists() else []

    def called(self, prefix: str) -> list[str]:
        return [line for line in self.calls() if line.startswith(prefix)]

    def state(self) -> list[str]:
        path = self.repo / ".git" / "cutover.state"
        return path.read_text().split() if path.exists() else []


@pytest.fixture
def machine(tmp_path):
    return Machine(tmp_path)


def test_plan_lists_the_steps_and_what_is_running_and_changes_nothing(machine):
    (machine.dir / "procs").write_text("4242 /usr/bin/python -m ml_stack.node_launch supervise\n")
    head = git(machine.repo, "rev-parse", "HEAD")
    done = machine.run("--plan")
    assert done.returncode == 0, done.stderr
    for n in range(1, 8):
        assert f"step {n} [todo]" in done.stdout
    assert "4242" in done.stdout and f"HOME {machine.home}" in done.stdout
    assert machine.state() == [] and git(machine.repo, "rev-parse", "HEAD") == head
    assert machine.called("python3 -m pip install") == [] and machine.called("poolhouse migrate") == []


def test_without_yes_it_asks_once_and_a_no_changes_nothing(machine):
    done = machine.run(stdin="n\n")
    assert done.returncode == 1 and "nothing changed" in done.stdout
    assert machine.state() == [] and machine.called("python3 -m pip install") == []
    assert "READY" in machine.run(stdin="y\n").stdout
    assert machine.state() == ["1", "2", "3", "4", "5", "6", "7"]


def test_a_full_run_does_every_step_in_order_and_ends_ready(machine):
    done = machine.run("--yes")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "READY" in done.stdout and "rollback:" in done.stdout
    order = [line for line in machine.calls() if line.split()[0] in ("python3", "poolhouse", "land", "poolhouse-doctor")]
    heads = [" ".join(line.split()[:3]) for line in order]
    assert heads == ["python3 -c import", "python3 -m ml_stack.node_launch", "python3 -m pip", "python3 -m pip", "python3 -m pip", "python3 -m pip",
                     "poolhouse migrate plan", "poolhouse migrate run", "poolhouse migrate verify",
                     "poolhouse runtime ensure",
                     "poolhouse node build", "land up", "poolhouse-doctor"]
    assert git(machine.repo, "log", "-1", "--format=%s").strip() == COMMIT
    assert "POOLHOUSE_WINDOW_POSITION" in (machine.repo / ".claude" / "settings.json").read_text()
    assert (machine.home / ".poolhouse" / "migrate.log").exists() and not (machine.home / ".ml-stack").exists()
    assert machine.state() == ["1", "2", "3", "4", "5", "6", "7"]


def test_a_second_run_skips_every_step(machine):
    assert machine.run("--yes").returncode == 0
    before = [c for c in machine.calls() if not c.startswith("pgrep")]
    again = machine.run("--yes")
    assert again.returncode == 0 and again.stdout.count("done earlier, skipped") == 7
    assert [c for c in machine.calls() if not c.startswith("pgrep")] == before


def test_each_step_skips_itself_when_its_work_is_already_done(machine):
    git(machine.repo, "apply", "docs/rename-protected.patch")
    git(machine.repo, "commit", "-qam", "chore: patch by hand")
    (machine.home / ".ml-stack").rmdir()
    (machine.home / ".poolhouse").mkdir()
    (machine.home / ".poolhouse" / "migrate.log").write_text("done\n")
    (machine.dir / "installed").write_text("poolhouse\n")
    commits = git(machine.repo, "rev-list", "--count", "HEAD")
    done = machine.run("--yes")
    assert done.returncode == 0, done.stdout
    for said in ("already applied", "no stray", "poolhouse is installed", "already migrated"):
        assert said in done.stdout, said
    assert git(machine.repo, "rev-list", "--count", "HEAD") == commits
    assert machine.called("python3 -m pip install") == [] and machine.called("poolhouse migrate") == []


def test_a_patch_that_cannot_apply_stops_at_step_2_and_says_why(machine):
    (machine.repo / ".claude" / "settings.json").write_text('{"env": "something else"}\n')
    git(machine.repo, "commit", "-qam", "feat: diverged")
    done = machine.run("--yes")
    assert done.returncode == 1
    assert "STOPPED at step 2" in done.stdout and "does not apply" in done.stdout
    assert machine.state() == ["1"] and machine.called("python3 -m pip") == []


def test_a_live_process_of_the_old_name_stops_the_run_and_is_printed(machine):
    (machine.dir / "procs").write_text("4242 /usr/bin/python -m ml_stack.node_launch supervise\n")
    (machine.dir / "stubborn").write_text("")
    done = machine.run("--yes")
    assert done.returncode == 1
    assert "4242" in done.stdout and "STOPPED at step 3" in done.stdout and "scripts/cutover --resume" in done.stdout
    assert machine.state() == ["1", "2"]
    assert machine.called("python3 -m pip install") == [] and machine.called("poolhouse migrate") == []


def test_a_program_that_only_sits_in_a_folder_of_the_old_name_is_not_a_process_of_it(machine):
    (machine.dir / "procs").write_text(
        "9876 /x/ml-stack/.build-venv/bin/python /x/run-jedi-language-server.py\n"
        "9877 vim ml-stack-notes.md\n")
    done = machine.run("--yes")
    assert done.returncode == 0, done.stdout
    assert "9876" not in done.stdout and "STOPPED at step 3" not in done.stdout


def test_a_program_of_the_old_name_is_a_process_of_it(machine):
    (machine.dir / "procs").write_text("4244 /opt/bin/ml-stack-serve up\n")
    (machine.dir / "stubborn").write_text("")
    done = machine.run("--yes")
    assert done.returncode == 1 and "4244" in done.stdout and "STOPPED at step 3" in done.stdout


def test_a_process_that_stops_when_asked_lets_the_run_go_on(machine):
    (machine.dir / "procs").write_text("4242 /usr/bin/python -m ml_stack.node_launch supervise\n")
    (machine.dir / "launchd").write_text("123\t0\tcom.ml-stack.traind\n- 0 com.apple.other\n")
    done = machine.run("--yes")
    assert done.returncode == 0, done.stdout
    assert machine.called("python3 -m ml_stack.node_launch stop")
    assert machine.called("launchctl bootout") and "com.ml-stack.traind" in machine.called("launchctl bootout")[0]
    assert all("com.apple" not in line for line in machine.called("launchctl bootout"))
    assert machine.called("kill -TERM") == []


def test_a_runner_that_does_not_exit_is_sent_a_term_signal_and_then_reported(machine):
    (machine.dir / "procs").write_text("4243 /usr/bin/python scripts/land serve\n")
    done = machine.run("--yes")
    assert done.returncode == 1 and "4243" in done.stdout and "STOPPED at step 3" in done.stdout
    assert machine.called("kill -TERM 4243")


def test_a_poolhouse_directory_with_more_than_the_stubs_is_refused_and_kept(machine):
    stray = machine.home / ".poolhouse"
    (stray / "activity").mkdir(parents=True)
    (stray / "models").mkdir()
    done = machine.run("--yes")
    assert done.returncode == 1
    assert "STOPPED at step 4" in done.stdout and "models" in done.stdout
    assert (stray / "activity").is_dir() and (stray / "models").is_dir() and (machine.home / ".ml-stack").is_dir()
    assert machine.called("python3 -m pip install") == []


def test_a_poolhouse_directory_with_only_the_stubs_is_listed_and_removed(machine):
    stray = machine.home / ".poolhouse"
    (stray / "activity").mkdir(parents=True)
    (stray / "sentinel").mkdir()
    done = machine.run("--yes")
    assert done.returncode == 0, done.stdout
    assert "holds: activity sentinel" in done.stdout and "removed the stray" in done.stdout
    assert not (machine.home / ".poolhouse" / "activity").exists()


def test_resume_after_a_failure_at_step_6_repeats_only_the_rest(machine):
    (machine.dir / "migrate.rc").write_text("2")
    failed = machine.run("--yes")
    assert failed.returncode == 1
    assert "STOPPED at step 6" in failed.stdout and "exited 2" in failed.stdout
    assert "scripts/cutover --resume" in failed.stdout and "rollback:" in failed.stdout
    assert machine.state() == ["1", "2", "3", "4", "5"] and (machine.home / ".ml-stack").is_dir()
    (machine.dir / "migrate.rc").unlink()
    resumed = machine.run("--resume")
    assert resumed.returncode == 0 and "READY" in resumed.stdout
    assert resumed.stdout.count("done earlier, skipped") == 5
    assert len(machine.called("python3 -m pip install")) == 1
    assert machine.state() == ["1", "2", "3", "4", "5", "6", "7"]


def test_a_failing_verify_stops_at_step_6_with_the_problem_and_resume_repeats_it(machine):
    (machine.dir / "verify.rc").write_text("1")
    failed = machine.run("--yes")
    assert failed.returncode == 1
    assert "STOPPED at step 6" in failed.stdout and "dangling symlink" in failed.stdout
    assert "poolhouse migrate verify found" in failed.stdout and "scripts/cutover --resume" in failed.stdout
    assert machine.state() == ["1", "2", "3", "4", "5"] and machine.called("poolhouse runtime") == []
    (machine.dir / "verify.rc").unlink()
    assert machine.run("--resume").returncode == 0 and machine.state() == ["1", "2", "3", "4", "5", "6", "7"]


def test_a_failing_doctor_stops_at_step_7_and_names_the_resume_command(machine):
    (machine.dir / "doctor.rc").write_text("1")
    failed = machine.run("--yes")
    assert failed.returncode == 1 and "STOPPED at step 7" in failed.stdout and "READY" not in failed.stdout


def test_on_windows_it_runs_device_setup_and_touches_no_launchd_or_node_build(machine):
    done = machine.run("--yes", platform="windows")
    assert done.returncode == 0, done.stdout
    assert machine.called("poolhouse device-setup --yes")
    assert machine.called("poolhouse node build") == [] and machine.called("launchctl") == []


def test_it_only_ever_works_in_the_fake_home(machine, tmp_path):
    stand_in = tmp_path / "real-home"
    (stand_in / ".ml-stack").mkdir(parents=True)
    env = {**os.environ, "HOME": str(stand_in), "CUTOVER_PLATFORM": "macos"}
    done = subprocess.run(["sh", str(machine.repo / "scripts" / "cutover"), "--yes", "--dry-run-in", str(machine.dir)],
                          capture_output=True, text=True, env=env, timeout=60)
    assert done.returncode == 0, done.stdout
    assert f"HOME {machine.home}" in done.stdout and str(stand_in) not in done.stdout
    assert [p.name for p in stand_in.iterdir()] == [".ml-stack"]
    assert (machine.home / ".poolhouse" / "migrate.log").exists()
