"""Hostile pytest text reaches the container as literal arguments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.redteam
ROOT = Path(__file__).resolve().parents[1]


def test_container_spawn_keeps_arguments_literal_and_mounts_read_only(tmp_path):
    executables = tmp_path / "bin"
    executables.mkdir()
    record = tmp_path / "docker.jsonl"
    docker = executables / "docker"
    docker.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['DOCKER_RECORD'], 'a') as out:\n"
        " out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print('isolated-test-container') if 'run' in sys.argv else None\n"
    )
    docker.chmod(0o755)
    git = executables / "git"
    git.write_text(f"#!{sys.executable}\nimport pathlib, sys\npathlib.Path(sys.argv[-1]).mkdir(parents=True)\n")
    git.chmod(0o755)
    for name in ("tar",):
        executable = executables / name
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    broker = executables / "python3"
    broker.write_text(f"#!{sys.executable}\nimport json,os,subprocess,sys\n"
                     "with open(os.environ['BROKER_RECORD'], 'a') as out: out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                     "env={**os.environ,'DEV_TEST_WORKERS':'2','DEV_TEST_PYTEST_ENDPOINT':'host.docker.internal:12345','DEV_TEST_PYTEST_TOKEN':'test-token'}\n"
                     "sys.exit(subprocess.run(sys.argv[sys.argv.index('--')+1:],env=env).returncode)\n")
    broker.chmod(0o755)
    permits = tmp_path / "permits.jsonl"
    marker = tmp_path / "injected"
    hostile = f"$(touch {marker}); --privileged; --mount=type=bind,src=/,dst=/host"
    done = subprocess.run(
        [str(ROOT / "scripts/test-on-linux"), "-k", hostile],
        env={**os.environ, "PATH": f"{executables}:{os.environ['PATH']}",
             "DEV_TEST_WORKERS": "2", "DOCKER_RECORD": str(record),
             "BROKER_RECORD": str(permits), "DEV_TEST_SLOTS_DIR": str(tmp_path / "slots")},
        capture_output=True, text=True, timeout=15,
    )
    assert done.returncode == 0, done.stderr
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    run = next(call for call in calls if "run" in call)
    phases = [json.loads(line) for line in permits.read_text().splitlines()]
    assert phases[0][1:6] == ["run", "--want", "1", "--min", "1"]
    assert phases[1][1:7] == ["pytest", "--container", "--want", "0", "--min", "1"]
    executions = [call for call in calls if "exec" in call]
    assert executions[0][-2:] == ["-k", hostile]
    assert executions[1][-4:] == ["-k", hostile, "-n", "2"]
    assert "DEV_TEST_PYTEST_ENDPOINT" in executions[1]
    assert calls[-1][-3:] == ["rm", "-f", "isolated-test-container"]
    assert "--privileged" not in run and "--network=host" not in run
    mounts = [run[index + 1] for index, value in enumerate(run) if value == "-v"]
    assert len(mounts) == 2 and mounts[0].endswith("/repo:/src:ro")
    assert mounts[1].endswith(":/venv")
    assert not marker.exists()
    for call in calls:
        assert call[0] == "--config"
        assert not Path(call[1]).exists()
