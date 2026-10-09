"""Cheaper runtime builds: one build per burst of merges, uv-cached installs, log rotation and launcher-safe collection."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from poolhouse import runtime, runtime_cli, runtime_coalesce, runtime_deploy, runtime_store
from poolhouse.fleet import runtime_wheel
from poolhouse.net import uvinstall


class Tips:
    """A branch tip that moves on a scripted clock; the fake sleep advances the clock and the tip."""

    def __init__(self, moves):
        self.now, self.moves, self.seen = 0.0, dict(moves), "0" * 40
        self.asked = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.seen = next((c for t, c in sorted(self.moves.items(), reverse=True) if t <= self.now), self.seen)

    def retarget(self):
        self.asked.append(self.seen)
        return runtime_deploy.Plan(Path(), self.seen, None)


def commit(n):
    return f"{n:040x}"


def test_a_burst_of_merges_settles_to_one_plan_for_the_newest_tip_only():
    tips = Tips({0: commit(1), 20: commit(2), 50: commit(3)})
    tips.seen = commit(1)
    plan = runtime_coalesce.settle(tips.retarget, runtime_coalesce.Pace(100, 10, tips.clock, tips.sleep))
    assert plan.commit == commit(3)
    assert tips.now >= 150


def test_an_unmoving_tip_waits_the_stability_delay_and_no_delay_returns_at_once():
    tips = Tips({})
    tips.seen = commit(1)
    runtime_coalesce.settle(tips.retarget, runtime_coalesce.Pace(60, 15, tips.clock, tips.sleep))
    assert tips.now == 60
    tips.now = 0
    assert runtime_coalesce.settle(tips.retarget, runtime_coalesce.Pace(0, 15, tips.clock, tips.sleep)).commit == commit(1)
    assert tips.now == 0


def test_a_tip_that_moves_during_a_build_is_built_next_and_the_commits_between_are_skipped():
    tips = Tips({0: commit(1), 500: commit(2), 510: commit(3)})
    tips.seen = commit(1)
    built = []

    def run(plan):
        built.append(plan.commit)
        if len(built) == 1:
            tips.now = 600
            tips.seen = commit(3)
        return runtime_deploy.Outcome("switched", plan.commit)

    outcome = runtime_coalesce.coalesced(tips.retarget, run, runtime_coalesce.Pace(30, 10, tips.clock, tips.sleep))
    assert built == [commit(1), commit(3)] and outcome.commit == commit(3)


def test_a_failed_build_ends_the_rounds_instead_of_looping():
    tips = Tips({})
    tips.seen = commit(1)
    built = []

    def run(plan):
        built.append(plan.commit)
        tips.seen = commit(len(built) + 1)
        return runtime_deploy.Outcome("failed", plan.commit)

    runtime_coalesce.coalesced(tips.retarget, run, runtime_coalesce.Pace(0, 15, tips.clock, tips.sleep))
    assert len(built) == 1


def test_background_requests_settle_unless_now_is_given():
    args = argparse.Namespace(ref="HEAD", timeout=1.0, wait=1.0, checkout="", launchers="", agent="", label="",
                              force=False, force_build=False, now=False)
    assert "--settle" in runtime_cli._again(args)
    args.now = True
    assert "--settle" not in runtime_cli._again(args)


def test_the_ensure_log_is_emptied_in_place_past_its_cap(tmp_path):
    log = tmp_path / "ensure.log"
    log.write_text("x" * 100)
    assert not runtime_coalesce.rotate(log, 200)
    log.write_text("y" * 300)
    assert runtime_coalesce.rotate(log, 200)
    assert log.read_text() == ""
    assert not runtime_coalesce.rotate(log, 200)


def test_uv_installs_with_a_shared_cache_link_mode_and_no_config(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(uvinstall, "program", lambda path=None: "/bin/uv")
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append(argv) or subprocess.CompletedProcess(argv, 0, "", ""))
    done = uvinstall.install(tmp_path / "python", "poolhouse @ file:///tmp/w.whl", timeout=5, environment={"PATH": "/bin"})
    assert done is not None and done.returncode == 0
    argv = calls[0]
    assert argv[:3] == ["/bin/uv", "pip", "install"] and "--no-config" in argv
    assert argv[argv.index("--link-mode") + 1] in {"clone", "hardlink"}
    assert "--compile-bytecode" not in argv


@pytest.mark.parametrize("environment", [{"PATH": "/bin", "PIP_INDEX_URL": "https://example.org/simple"},
                                         {"PATH": "/bin", "UV_INDEX_URL": "https://example.org/simple"},
                                         {"PATH": "/bin", "HTTPS_PROXY": "http://proxy.example.org:3128"}])
def test_uv_defers_to_pip_when_the_environment_redirects_package_sources(tmp_path, monkeypatch, environment):
    monkeypatch.setattr(uvinstall, "program", lambda path=None: "/bin/uv")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("uv must not run"))
    assert uvinstall.install(tmp_path / "python", "poolhouse", timeout=5, environment=environment) is None


def test_the_runtime_install_falls_back_to_pip_without_uv_and_reports_a_uv_failure(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(runtime_wheel, "_run", lambda argv, timeout: ran.append(argv) or "")
    monkeypatch.setattr(uvinstall, "install", lambda *a, **k: None)
    runtime_wheel._install(tmp_path / "python", "spec", 5)
    assert ran == [[str(tmp_path / "python"), "-m", "pip", "install", "spec"]]
    monkeypatch.setattr(uvinstall, "install", lambda *a, **k: subprocess.CompletedProcess([], 2, "", "no such package"))
    with pytest.raises(ValueError, match="no such package"):
        runtime_wheel._install(tmp_path / "python", "spec", 5)
    assert len(ran) == 1


def tree(root, n, launcher_text=""):
    prefix = root / f"{n:040x}" / f"{n:032x}"
    prefix.mkdir(parents=True)
    runtime_store.mark_created(prefix, "agent", "ensure")
    runtime_store.mark_verified(runtime.Runtime(prefix, f"{n:040x}", "0.1.0", runtime.identity()))
    return prefix


def test_collection_never_removes_a_tree_a_launcher_names_nor_a_tree_without_the_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("POOLHOUSE_HOME", str(tmp_path / "state"))
    root = runtime.directory()
    old = [tree(root, n) for n in range(1, 6)]
    for n, prefix in enumerate(old):
        row = runtime_store.MARK
        (prefix / row).write_text((prefix / row).read_text().replace('"verified_at": ', f'"verified_at": {1000 + n}, "x": '))
    launchers = tmp_path / "bin"
    launchers.mkdir()
    (launchers / "poolhouse-demo").write_text(f"python = {str(old[0] / 'bin' / 'python')!r}\n")
    runtime_store.write_state({"launchers": str(launchers)}, root)
    foreign = root / f"{9:040x}" / f"{9:032x}"
    foreign.mkdir(parents=True)
    (foreign / "file").write_text("x")
    gone = set(runtime_store.collect({old[4]}, keep=1, root=root))
    assert gone == {old[1], old[2], old[3]}
    assert old[0].exists() and foreign.exists()
