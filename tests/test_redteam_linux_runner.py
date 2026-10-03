"""Hostile pytest text reaches the container as literal arguments."""

from __future__ import annotations

import json
import os
import subprocess
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
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        "with open(os.environ['DOCKER_RECORD'], 'a') as out:\n"
        " out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.stdin.read() if 'run' in sys.argv else None\n"
    )
    docker.chmod(0o755)
    git = executables / "git"
    git.write_text("#!/usr/bin/env python3\nimport pathlib, sys\npathlib.Path(sys.argv[-1]).mkdir(parents=True)\n")
    git.chmod(0o755)
    for name in ("tar",):
        executable = executables / name
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    marker = tmp_path / "injected"
    hostile = f"$(touch {marker}); --privileged; --mount=type=bind,src=/,dst=/host"
    done = subprocess.run(
        [str(ROOT / "scripts/test-on-linux"), "-k", hostile],
        env={**os.environ, "PATH": f"{executables}:{os.environ['PATH']}",
             "DEV_TEST_WORKERS": "2", "DOCKER_RECORD": str(record)},
        capture_output=True, text=True, timeout=15,
    )
    assert done.returncode == 0, done.stderr
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    run = next(call for call in calls if "run" in call)
    assert run[-5:] == ["--", "-k", hostile, "-n", "2"]
    assert "--privileged" not in run and "--network=host" not in run
    mounts = [run[index + 1] for index, value in enumerate(run) if value == "-v"]
    assert len(mounts) == 2 and mounts[0].endswith("/repo:/src:ro")
    assert mounts[1].endswith(":/venv")
    assert not marker.exists()
    for call in calls:
        assert call[0] == "--config"
        assert not Path(call[1]).exists()
