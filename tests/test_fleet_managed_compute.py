"""Frozen fleet capability reports use the managed execution environment."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from ml_stack.fleet import device, managed_compute


@pytest.fixture
def environment(tmp_path):
    python = tmp_path / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    return SimpleNamespace(python=python, exists=True, _cache={})


def test_frozen_report_probes_the_job_interpreter_and_keeps_provenance(environment, monkeypatch):
    runtime = {"python": str(environment.python), "version": "0.2.1", "commit": "v0.2.1"}
    expected = {"backends": ["mlx", "torch"], "accelerator": True, "vendor": "apple",
                "compute_runtime": runtime}
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(expected))

    monkeypatch.setattr(managed_compute.subprocess, "run", run)
    monkeypatch.setattr(device.sys, "frozen", True, raising=False)
    monkeypatch.setattr(device, "stdlib_device_report", lambda: {"backends": [], "cpus": 16})
    monkeypatch.setattr(device, "registered_reports", lambda: [lambda: {"accelerator": False}])
    monkeypatch.setenv("EXAMPLE_API_KEY", "private-value")
    monkeypatch.setenv("PYTHONPATH", "/changing-checkout")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/frozen-libraries")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/installed-libraries")
    result = device.device_report(environment=environment)
    assert result == {"cpus": 16, **expected}
    assert device.device_report(environment=environment) == result
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[:3] == [str(environment.python), "-I", "-c"]
    assert kwargs["timeout"] == 20
    assert "EXAMPLE_API_KEY" not in kwargs["env"]
    assert "PYTHONPATH" not in kwargs["env"]
    assert kwargs["env"]["LD_LIBRARY_PATH"] == "/installed-libraries"


@pytest.mark.parametrize("failure", ["missing", "exit", "malformed", "timeout", "no-provenance"])
def test_managed_compute_does_not_advertise_unverified_capabilities(environment, monkeypatch, failure):
    def run(argv, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 20)
        if failure == "missing":
            raise FileNotFoundError(argv[0])
        return SimpleNamespace(returncode=1 if failure == "exit" else 0,
                               stdout="invalid" if failure == "malformed" else '{"accelerator": true}')

    monkeypatch.setattr(managed_compute.subprocess, "run", run)
    assert managed_compute.report(environment) == {}


def test_missing_environment_and_unfrozen_daemon_do_not_spawn_probe(environment, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected managed capability probe")

    monkeypatch.setattr(managed_compute.subprocess, "run", forbidden)
    environment.exists = False
    assert managed_compute.report(environment) == {}
    environment.exists = True
    monkeypatch.setattr(device.sys, "frozen", False, raising=False)
    monkeypatch.setattr(device, "stdlib_device_report", lambda: {"backends": ["torch"]})
    monkeypatch.setattr(device, "registered_reports", lambda: [])
    assert device.device_report(environment=environment) == {"backends": ["torch"]}


def test_stale_runtime_does_not_advertise_compute_or_start_jobs(environment, monkeypatch, tmp_path):
    from ml_stack.fleet.jobs import JobRunner

    def stale():
        raise OSError("Refresh Training essentials")

    environment.require_current_runtime = stale
    expected = managed_compute.report(environment)
    assert expected["backends"] == [] and expected["accelerator"] is False
    assert expected["compute_runtime"]["ready"] is False
    monkeypatch.setattr(JobRunner, "_spawn", lambda self, upto: None)
    runner = JobRunner(tmp_path, environment=environment)
    job = runner.submit("training", ["python", "-m", "ml_stack.train"], str(tmp_path))
    runner._run_one(job)
    assert job.state == "failed" and job.pid is None
    assert "Refresh Training essentials" in runner.log_path(job.id).read_text()


def test_non_python_job_runs_with_stale_managed_runtime(environment, monkeypatch, tmp_path):
    import sys

    from ml_stack.fleet.jobs import JobRunner

    def stale():
        pytest.fail("non-Python job must not require a managed runtime")

    environment.require_current_runtime = stale
    monkeypatch.setattr(JobRunner, "_spawn", lambda self, upto: None)
    runner = JobRunner(tmp_path, environment=environment)
    argv = ["cmd", "/c", "echo", "done"] if sys.platform == "win32" else ["/bin/echo", "done"]
    job = runner.submit("implementation-tool", argv, str(tmp_path))
    runner._run_one(job)
    assert job.state == "done" and job.returncode == 0
