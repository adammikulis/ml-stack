"""Which commits a runtime deploy accepts, the audit it leaves and the environment it ignores, against real git repositories and child processes."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ml_stack import keystore, runtime, runtime_deploy, runtime_store, runtime_trust
from ml_stack.activity import writer
from ml_stack.activity.schema import Entry

pty = importlib.import_module("pty") if sys.platform != "win32" else None
ROOT = Path(__file__).resolve().parent
SRC = str(ROOT.parent / "src")
GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"}


def git(repo, *words):
    return subprocess.run(["git", "-C", str(repo), *words], capture_output=True, text=True, check=True,
                          env={**os.environ, **GIT_ENV}).stdout.strip()


def commit(repo, name, text="x"):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", f"chore: {name}")
    return git(repo, "rev-parse", "HEAD")


class World:
    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.tmp = tmp_path
        self.user = tmp_path / "user"
        self.home = self.user / ".ml-stack"
        self.keyring = tmp_path / "keyring.json"
        for name in ("CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE", "ML_STACK_WORKSPACE_AGENT"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("ML_STACK_HOME", str(self.home))
        monkeypatch.setenv("ML_STACK_TEST_KEYRING", str(self.keyring))
        monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "onboard_support.FileKeyring")
        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "dev")
        commit(self.repo, "src/ml_stack/__init__.py", "")
        self.base = commit(self.repo, "packaging/runtime-floor", "2\n")
        self.tip = commit(self.repo, "later.txt")
        git(self.repo, "checkout", "-q", "-b", "evil", self.base)
        self.evil = commit(self.repo, "packaging/runtime-floor", "99\n")
        git(self.repo, "checkout", "-q", "dev")
        runtime_deploy.prepare_root()
        runtime_store.write_state({"checkout": str(self.repo)})

    def env(self, **extra) -> dict:
        env = {key: value for key, value in os.environ.items()
               if key not in ("CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE", "PYTEST_CURRENT_TEST", "ML_STACK_AUTHORITY_FLOOR")}
        return {**env, "PYTHONPATH": os.pathsep.join([SRC, str(ROOT)]), "ML_STACK_HOME": str(self.home), "HOME": str(self.user),
                "PYTHON_KEYRING_BACKEND": "onboard_support.FileKeyring",
                "ML_STACK_TEST_KEYRING": str(self.keyring), "PIP_NO_INDEX": "1", **extra}

    def cli(self, *argv: str, agent: bool = True, **extra) -> subprocess.CompletedProcess:
        env = self.env(**({"CLAUDECODE": "1"} if agent else {}), **extra)
        return subprocess.run([sys.executable, "-m", "ml_stack.runtime_cli", *argv], capture_output=True,
                              text=True, timeout=120, env=env, stdin=subprocess.DEVNULL)

    def audited(self) -> list[Entry]:
        with writer._LOCK:
            writer._LOGS.clear()
        return [e for e in writer.log().entries() if isinstance(e, Entry) and e.kind == "runtime.deploy"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    import keyring
    from onboard_support import FileKeyring
    made = World(tmp_path, monkeypatch)
    before = keyring.get_keyring()
    keyring.set_keyring(FileKeyring())
    # A person makes the master key once (`unlock`); an agent-marked child never may, so the
    # sealed activity writer in the child finds it here instead of refusing and leaving no record.
    keystore.default().provision()
    yield made
    keyring.set_keyring(before)


def test_an_agent_cannot_deploy_a_commit_that_is_not_on_the_development_branch(world):
    done = world.cli("ensure", "--ref", world.evil)
    assert done.returncode == 1 and "not on the development branch" in done.stderr
    assert not (runtime.directory() / world.evil).exists() and not (runtime.directory() / "selected.json").exists()


def test_a_refused_deploy_leaves_an_audit_record(world):
    world.cli("ensure", "--ref", world.evil, "--agent", "worker-1")
    [entry] = world.audited()
    assert (entry.subject, entry.outcome, entry.meta["command"]) == (world.evil[:12] if False else world.evil, "refused", "ensure")
    assert entry.meta["agent"] == "worker-1" and entry.meta["person"] is False
    assert "development branch" in entry.meta["detail"] and entry.actor == "agent:worker-1"


def test_an_agent_child_never_makes_the_master_key_for_its_audit(world):
    world.keyring.unlink()
    done = world.cli("ensure", "--ref", world.evil, "--agent", "worker-1")
    assert done.returncode == 1 and "not on the development branch" in done.stderr
    assert not world.keyring.exists(), "an agent-marked process made the encryption key itself"


def test_a_commit_before_the_tip_is_admitted_and_one_beside_it_is_not(world):
    runtime_trust.admit(world.repo, world.tip)
    runtime_trust.admit(world.repo, world.base)
    with pytest.raises(runtime_deploy.DeployError, match="development branch"):
        runtime_trust.admit(world.repo, world.evil)
    runtime_trust.admit(world.repo, world.evil, allow_unmerged=True)


def test_the_floor_comes_from_the_development_tip_never_from_the_deployed_commit(world):
    assert runtime_trust.floor(world.repo) == 2
    done = world.cli("status", "--json")
    assert done.returncode == 0 and json.loads(done.stdout)["floor"] == 2


def test_a_worktree_is_only_a_source_of_objects(world):
    other = world.tmp / "worktree"
    git(world.repo, "worktree", "add", "-q", "-b", "side", str(other), world.base)
    assert runtime_trust.main_worktree(other) == world.repo.resolve()
    done = world.cli("status", "--json", "--checkout", str(other))
    record = json.loads(done.stdout)
    assert done.returncode == 0 and record["wanted"] == world.tip and record["floor"] == 2


def test_an_agent_cannot_name_another_repository(world):
    clone = world.tmp / "clone"
    subprocess.run(["git", "clone", "-q", str(world.repo), str(clone)], check=True, capture_output=True)
    done = world.cli("ensure", "--checkout", str(clone))
    assert done.returncode == 1 and "recorded repository" in done.stderr
    assert not (runtime.directory() / world.tip).exists()


def test_an_agent_cannot_deploy_on_a_machine_with_no_recorded_repository(world):
    (runtime.directory() / runtime_store.STATE).unlink()
    done = world.cli("ensure", "--checkout", str(world.repo))
    assert done.returncode == 1 and "a person runs" in done.stderr


def test_background_ensure_of_the_tip_needs_no_agent_identity(world):
    runtime_deploy.prepare_root()
    lock = runtime.directory() / "deploy.lock"
    from ml_stack.lock import only_one
    with only_one(lock, note="busy"):
        done = world.cli("ensure", "--background")
    assert done.returncode == 0 and "already running" in done.stdout
    assert [e.outcome for e in world.audited()] == ["building"]


def test_unmerged_deploys_are_refused_to_an_agent_even_at_a_terminal(world):
    master, slave = pty.openpty()
    child = subprocess.Popen([sys.executable, "-m", "ml_stack.runtime_cli", "ensure", "--ref", world.evil, "--allow-unmerged"],
                             env=world.env(CLAUDECODE="1"), stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    assert child.wait(timeout=60) != 0
    os.close(master)
    assert not (runtime.directory() / world.evil).exists()


def test_an_agent_started_process_ignores_the_variables_that_move_the_state_root(world):
    other = world.tmp / "elsewhere"
    done = world.cli("status", "--json", ML_STACK_HOME=str(other))
    assert done.returncode == 0 and "ignoring ML_STACK_HOME" in done.stderr
    assert not other.exists() and json.loads(done.stdout)["wanted"] == world.tip


def test_a_person_process_keeps_the_variables(world):
    other = world.tmp / "elsewhere"
    done = world.cli("status", "--json", agent=False, ML_STACK_HOME=str(other))
    assert "ignoring" not in done.stderr
    assert done.returncode == 1 and "no source checkout recorded" in done.stderr


def test_a_person_at_a_terminal_may_deploy_an_unmerged_commit_by_typing_its_ref(world):
    import select
    master, slave = pty.openpty()
    child = subprocess.Popen([sys.executable, "-m", "ml_stack.runtime_cli", "ensure", "--ref", world.evil, "--allow-unmerged",
                              "--timeout", "30"], env=world.env(), stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    heard, typed = b"", False
    while child.poll() is None:
        if select.select([master], [], [], 0.2)[0]:
            try:
                heard += os.read(master, 4096)
            except OSError:
                break
        if not typed and b"type " in heard:
            os.write(master, (world.evil + "\n").encode())
            typed = True
    child.wait(timeout=120)
    os.close(master)
    assert typed and b"not on the development branch" not in heard
    [entry] = world.audited()
    assert entry.subject == world.evil and entry.outcome != "refused" and entry.meta["person"] is True


def test_a_checkout_path_with_shell_punctuation_reaches_git_as_one_argument(world):
    hostile = world.tmp / "w; touch pwn; $(touch pwn) `touch pwn`"
    git(world.repo, "worktree", "add", "-q", "-b", "odd", str(hostile), world.base)
    for argv in (("status", "--json"), ("ensure", "--ref", world.evil)):
        done = world.cli(*argv, "--checkout", str(hostile))
        assert "pwn" not in done.stdout.replace(str(hostile), "")
    assert not list(world.tmp.rglob("pwn")) and not (hostile / "pwn").exists()


@pytest.mark.parametrize("ref", ["--output=pwn", "-h", "dev; touch pwn", "$(touch pwn)", "`touch pwn`", "dev\ntouch pwn"])
def test_a_ref_that_is_an_option_or_a_command_names_no_commit(world, ref):
    done = world.cli("ensure", f"--ref={ref}")
    assert done.returncode == 1 and not list(world.tmp.rglob("pwn"))
    assert not (runtime.directory() / "selected.json").exists()


def test_a_forged_state_root_naming_a_hostile_clone_is_ignored_by_an_agent(world):
    clone = world.tmp / "clone"
    subprocess.run(["git", "clone", "-q", str(world.repo), str(clone)], check=True, capture_output=True)
    forged = world.tmp / "forged" / "runtimes"
    forged.mkdir(parents=True)
    (world.tmp / "forged").joinpath("marker").write_text("x")
    done = world.cli("ensure", "--checkout", str(clone), ML_STACK_HOME=str(world.tmp / "forged"))
    assert done.returncode == 1 and "ignoring ML_STACK_HOME" in done.stderr and "recorded repository" in done.stderr
    assert sorted(p.name for p in (world.tmp / "forged").iterdir()) == ["marker", "runtimes"]
