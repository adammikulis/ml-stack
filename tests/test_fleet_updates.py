"""Keeping a machine current without ever walking over what it is doing.

Two modes -- a published release, and the head of a branch -- and one gate in front of
both. Nothing here touches the network, git or pip: `track_once` takes the three seams
(``git``, ``pip``, ``restart``) and the fakes below answer them, so the shape of every
decision it makes is what is under test rather than anybody's real checkout. The model
cache tests build a Hub cache in ``tmp_path`` and are the only ones that touch a disk.

Every name is invented.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from ml_stack.fleet import autostart, updates


class FakeGit:
    """A git that answers from a script, and remembers what it was asked.

    ``answers`` maps the first word of a command to ``(returncode, output)``; a command
    with no answer succeeds silently, which is what a real git does for most of these.
    """

    def __init__(self, **answers: tuple[int, str]) -> None:
        self.answers = answers
        self.calls: list[list[str]] = []

    def __call__(self, args) -> tuple[int, str]:
        argv = [str(a) for a in args]
        self.calls.append(argv)
        return self.answers.get(argv[0], (0, ""))

    def ran(self, word: str) -> bool:
        return any(c[0] == word for c in self.calls)


OLD = "1111111111111111111111111111111111111111"
NEW = "2222222222222222222222222222222222222222"
REPO = "https://example.invalid/wrenfield/ml-stack"


def _git(**over: tuple[int, str]) -> FakeGit:
    """A checkout one commit behind the branch, fast-forwardable, nothing packaged moved."""
    base = {
        "ls-remote": (0, f"{NEW}\trefs/heads/main"),
        "rev-parse": (0, OLD),
        "fetch": (0, ""),
        "merge-base": (0, ""),
        "diff": (0, "src/ml_stack/fleet/join.py\nREADME.md\n"),
        "pull": (0, "Updating 1111111..2222222"),
    }
    base.update(over)
    return FakeGit(**base)


class TestFollowingABranch:
    def test_a_branch_that_has_not_moved_pulls_nothing(self, tmp_path):
        git = _git(**{"rev-parse": (0, NEW)})
        restarts = []
        got = updates.track_once(REPO, "main", tmp_path, git=git,
                                 restart=lambda: restarts.append(1))

        assert got.pulled is False and got.error == ""
        assert not git.ran("pull"), "it pulled a branch that had not moved"
        assert restarts == [], "it restarted for nothing"

    def test_a_fast_forward_is_pulled_and_restarted(self, tmp_path):
        # rev-parse answers OLD before the pull and NEW after it; a real git does the same.
        seen = iter([(0, OLD), (0, NEW)])
        git = _git()
        git.answers["rev-parse"] = (0, OLD)
        original = git.__call__

        def answering(args):
            argv = [str(a) for a in args]
            if argv[0] == "rev-parse":
                git.calls.append(argv)
                return next(seen)
            return original(args)

        restarts = []
        got = updates.track_once(REPO, "main", tmp_path, git=answering,
                                 restart=lambda: restarts.append(1) or "service")

        assert got.pulled and got.now == NEW and got.error == ""
        assert got.installed is False, "nothing packaged moved, so nothing was reinstalled"
        assert restarts == [1], "a machine on new code that never restarted runs the old"
        pull = next(c for c in git.calls if c[0] == "pull")
        assert "--ff-only" in pull and "merge" not in " ".join(pull)

    def test_a_diverged_checkout_is_reported_and_left_alone(self, tmp_path):
        """Somebody's work in progress is not a thing a daemon resets at 3am."""
        git = _git(**{"merge-base": (1, "")})
        restarts = []
        got = updates.track_once(REPO, "main", tmp_path, git=git,
                                 pip=lambda where: pytest.fail("it reinstalled"),
                                 restart=lambda: restarts.append(1))

        assert got.diverged and not got.pulled
        assert "left alone" in got.error and got.now == OLD
        assert not git.ran("pull"), "it pulled over a checkout that had diverged"
        assert restarts == []

    def test_a_pull_that_touches_packaging_reinstalls_first(self, tmp_path):
        git = _git(**{"diff": (0, "pyproject.toml\nsrc/ml_stack/fleet/join.py\n")})
        installed = []
        got = updates.track_once(REPO, "main", tmp_path, git=git,
                                 pip=lambda where: (installed.append(where), (0, "ok"))[1],
                                 restart=lambda: "service")

        assert got.pulled and got.installed
        assert installed == [Path(tmp_path)], "a dependency moved and pip never ran"

    def test_a_pull_that_touches_nothing_packaged_skips_pip(self, tmp_path):
        got = updates.track_once(REPO, "main", tmp_path, git=_git(),
                                 pip=lambda where: pytest.fail("pip ran for a code-only pull"),
                                 restart=lambda: "service")
        assert got.pulled and got.installed is False

    def test_a_failing_pull_leaves_the_daemon_on_the_code_it_has(self, tmp_path):
        git = _git(**{"pull": (1, "error: Your local changes would be overwritten")})
        restarts = []
        got = updates.track_once(REPO, "main", tmp_path, git=git,
                                 restart=lambda: restarts.append(1))

        assert not got.pulled and got.now == OLD
        assert "overwritten" in got.error
        assert restarts == [], "it restarted onto code it had failed to pull"

    def test_a_failing_install_does_not_restart_onto_a_broken_tree(self, tmp_path):
        git = _git(**{"diff": (0, "pyproject.toml\n")})
        restarts = []
        got = updates.track_once(REPO, "main", tmp_path, git=git,
                                 pip=lambda where: (1, "could not resolve ladybug"),
                                 restart=lambda: restarts.append(1))

        assert got.pulled and not got.installed
        assert "pip install -e ." in got.error and restarts == []

    def test_a_branch_that_does_not_exist_is_said_so(self, tmp_path):
        got = updates.track_once(REPO, "quince", tmp_path,
                                 git=FakeGit(**{"ls-remote": (0, "")}))
        assert "could not read quince" in got.error

    def test_a_directory_that_is_not_a_checkout_is_said_so(self, tmp_path):
        got = updates.track_once(REPO, "main", tmp_path,
                                 git=_git(**{"rev-parse": (128, "not a git repository")}))
        assert "not a git checkout" in got.error

    def test_the_loop_records_what_it_last_saw(self, tmp_path):
        thread = updates.track(REPO, "main", tmp_path, git=_git(**{"rev-parse": (0, NEW)}),
                               first_after_s=0.0, interval=0.01, rounds=1,
                               restart=lambda: "")
        thread.join(timeout=5)
        assert updates.LAST["tracking"] == "main"
        assert updates.LAST["checked_at"] > 0


class TestNothingIsWalkedOver:
    """An update that killed a measurement halfway would be worse than waiting a day."""

    def test_a_measuring_machine_is_left_alone(self):
        assert updates.in_the_way(measuring=lambda: True) == "a benchmark is measuring"
        assert not updates.quiet(measuring=lambda: True)()

    def test_a_loaded_model_is_left_alone(self):
        assert updates.in_the_way(leases=lambda: True) == "a model is loaded"

    def test_a_running_job_is_left_alone(self):
        assert updates.in_the_way(jobs=lambda: True) == "a job is running"

    def test_an_idle_machine_is_free(self):
        gate = updates.quiet(jobs=lambda: False, measuring=lambda: False,
                             leases=lambda: False)
        assert gate() is True

    def test_a_check_that_raises_counts_as_busy(self):
        """Not being able to tell is not a reason to go ahead."""
        def broken() -> bool:
            raise OSError("the bench lock file is on a disk that went away")

        assert updates.in_the_way(measuring=broken)
        assert not updates.quiet(measuring=broken)()

    def test_a_branch_is_not_pulled_over_a_measurement(self, tmp_path):
        git = _git()
        thread = updates.track(REPO, "main", tmp_path, git=git, idle=lambda: False,
                               first_after_s=0.0, interval=0.01, rounds=2,
                               restart=lambda: "")
        thread.join(timeout=5)
        assert git.calls == [], "it asked git about a machine that was busy"


class TestFollowingReleases:
    def test_a_newer_release_is_applied_and_restarted_once(self, monkeypatch, tmp_path):
        applied = []
        restarted = []
        monkeypatch.setattr(updates, "apply_if_newer",
                            lambda: (applied.append(1), {"installed": True,
                                                         "version": "9.9.9"})[1])
        thread = updates.watch(wanted=lambda: True, idle=lambda: True,
                               every_s=0.01, first_after_s=0.0,
                               restart=lambda: (restarted.append(1), "relaunched")[1])
        thread.join(timeout=5)

        assert applied == [1], "it applied the release more than once"
        assert restarted == [1], "the restart seam was not called exactly once"
        assert not thread.is_alive(), "the loop kept running after handing over"

    def test_a_busy_machine_is_not_updated(self, monkeypatch):
        tried = threading.Event()
        monkeypatch.setattr(updates, "apply_if_newer",
                            lambda: (tried.set(), {"installed": False})[1])
        thread = updates.watch(wanted=lambda: True, idle=lambda: False,
                               every_s=0.01, first_after_s=0.0, rounds=3,
                               restart=lambda: "")
        thread.join(timeout=5)
        assert not tried.is_set(), "it updated a machine that was working"

    def test_the_whole_install_is_replaced_not_only_the_running_binary(self, tmp_path):
        """The daemon and the CLI come out of one zip; replacing one leaves the other
        a version behind, which is a bug that costs an afternoon to find."""
        import zipfile

        here = tmp_path / "bin"
        here.mkdir()
        (here / "ml-stack-headless").write_text("old daemon")
        (here / "ml-stack").write_text("old cli")
        archive = tmp_path / "release.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("ml-stack-headless", "new daemon")
            zf.writestr("ml-stack", "new cli")

        updates.install(archive, app_path=here / "ml-stack-headless")

        assert (here / "ml-stack-headless").read_text() == "new daemon"
        assert (here / "ml-stack").read_text() == "new cli"

    def test_nothing_new_is_put_on_a_machine_that_had_not_got_it(self, tmp_path):
        import zipfile

        here = tmp_path / "bin"
        here.mkdir()
        (here / "ml-stack-headless").write_text("old")
        archive = tmp_path / "release.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("ml-stack-headless", "new")
            zf.writestr("ml-stack", "a window they never installed")

        updates.install(archive, app_path=here / "ml-stack-headless")
        assert not (here / "ml-stack").exists()


class TestWhatThisMachineSays:
    """`LAST` is one dict per process, written by whichever loop is running, so these
    give it a fresh one rather than reading whatever another test left in it."""

    @pytest.fixture(autouse=True)
    def _own_state(self, monkeypatch):
        monkeypatch.setattr(updates, "LAST", {"tracking": "off", "checked_at": 0.0,
                                              "error": "", "commit": ""})

    def test_the_state_names_the_mode_and_when_it_last_looked(self):
        updates.note(tracking="main", checked_at=1234.0, commit="abc1234", error="")
        said = updates.state()
        assert said["tracking"] == "main"
        assert said["update_checked_at"] == 1234.0
        assert said["commit"] == "abc1234"
        assert "version" in said

    def test_a_commit_nobody_can_date_is_zero_rather_than_a_guess(self, tmp_path):
        assert updates.commit_age_s("") == 0.0
        assert updates.commit_age_s("deadbee", checkout=tmp_path / "nowhere") == 0.0


class TestComingBackOnTheNewCode:
    def test_a_login_service_is_kicked_rather_than_re_execed(self, monkeypatch):
        monkeypatch.setattr(autostart.sys, "platform", "darwin")
        monkeypatch.setattr(autostart, "status", lambda: {"mode": "login", "paths": []})
        ran: list[list[str]] = []
        how = autostart.restart(run=lambda argv: (ran.append(argv), 0)[1],
                                reexec=lambda: pytest.fail("it re-execed under launchd"))
        assert how == "service"
        assert ran and ran[0][:2] == ["launchctl", "kickstart"]

    def test_a_daemon_started_by_hand_re_execs_itself(self, monkeypatch):
        monkeypatch.setattr(autostart, "status", lambda: {"mode": "manual", "paths": []})
        done = []
        how = autostart.restart(run=lambda argv: 1, reexec=lambda: done.append(1))
        assert how == "exec" and done == [1]

    def test_windows_re_execs_because_a_logon_task_would_not_fire_again(self, monkeypatch):
        monkeypatch.setattr(autostart.sys, "platform", "win32")
        monkeypatch.setattr(autostart, "status", lambda: {"mode": "login", "paths": []})
        done = []
        autostart.restart(run=lambda argv: 0, reexec=lambda: done.append(1))
        assert done == [1]

    def test_a_bundle_relaunches_itself_and_asks_autostart_for_nothing(self, monkeypatch):
        monkeypatch.setattr(updates, "relaunch", lambda: True)
        monkeypatch.setattr(autostart, "restart",
                            lambda **k: pytest.fail("it asked the service to restart too"))
        assert updates.restart_after_update() == "relaunched"
