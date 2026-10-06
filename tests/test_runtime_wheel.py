"""Runtime wheel provenance and installation boundaries."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from ml_stack.fleet import runtime_wheel

COMMIT = "1234567890abcdef1234567890abcdef12345678"


def _wheel(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ml_stack/fleet/runtime.py", "VALUE = 42\n")
        archive.writestr("ml_stack-0.1.0.dist-info/RECORD", "")
    return path


def test_stamp_records_the_source_and_hashes_every_installed_file(tmp_path):
    wheel = _wheel(tmp_path / "ml_stack-0.1.0-py3-none-any.whl")
    runtime_wheel.stamp(wheel, COMMIT, tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read("ml_stack/fleet/built-from").decode().strip() == COMMIT
        assert archive.read("ml_stack/fleet/source-checkout").decode().strip() == str(tmp_path)
        rows = list(csv.reader(io.StringIO(archive.read("ml_stack-0.1.0.dist-info/RECORD").decode())))
        assert {row[0] for row in rows} == set(archive.namelist())
        for name, digest, size in rows[:-1]:
            data = archive.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            assert digest == f"sha256={expected}" and size == str(len(data))


def test_install_uses_committed_snapshot_and_replaces_same_version(tmp_path, monkeypatch):
    calls = []

    def run(argv, timeout):
        calls.append(argv)
        if argv[0] == "git" and "rev-parse" in argv:
            return COMMIT
        if argv[0] == "git" and "archive" in argv:
            with zipfile.ZipFile(argv[argv.index("--output") + 1], "w") as archive:
                archive.writestr("pyproject.toml", "[project]\nname='ml-stack'\n")
        if "wheel" in argv:
            wheels = Path(argv[argv.index("--wheel-dir") + 1])
            _wheel(wheels / "ml_stack-0.1.0-py3-none-any.whl")
        return "ok"

    monkeypatch.setattr(runtime_wheel, "_run", run)
    assert runtime_wheel.install_checkout(tmp_path, timeout=7) == (0, "ok")
    assert calls[1][-1] == COMMIT
    assert str(tmp_path) not in calls[2]
    assert calls[-2][4] == "--upgrade"
    assert calls[-1][4:6] == ["--force-reinstall", "--no-deps"]
    assert not any("-e" in argv for argv in calls)


def test_dependency_failure_does_not_replace_the_installed_distribution(tmp_path, monkeypatch):
    calls = []

    def run(argv, timeout):
        calls.append(argv)
        if "rev-parse" in argv:
            return COMMIT
        if "archive" in argv:
            with zipfile.ZipFile(argv[argv.index("--output") + 1], "w") as archive:
                archive.writestr("pyproject.toml", "")
        if "wheel" in argv:
            _wheel(Path(argv[argv.index("--wheel-dir") + 1]) / "ml_stack-0.1.0-py3-none-any.whl")
        if "--upgrade" in argv:
            raise ValueError("missing dependency")
        return ""

    monkeypatch.setattr(runtime_wheel, "_run", run)
    code, error = runtime_wheel.install_checkout(tmp_path, timeout=7)
    assert code == 1 and "missing dependency" in error
    assert not any("--force-reinstall" in argv for argv in calls)


@pytest.mark.slow
def test_built_wheel_imports_from_an_immutable_install(tmp_path):
    root = Path(__file__).resolve().parents[1]
    wheels = list((root / "dist").glob("ml_stack-*.whl"))
    assert wheels, "run python packaging/build.py before packaging tests"
    wheel = tmp_path / wheels[0].name
    shutil.copy2(wheels[0], wheel)
    runtime_wheel.stamp(wheel, COMMIT, root)
    target = tmp_path / "installed"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                    "--target", str(target), str(wheel)], check=True, capture_output=True, text=True)
    env = {**os.environ, "PYTHONPATH": str(target)}
    script = ("import json; from pathlib import Path; from importlib.metadata import version; "
              "from ml_stack.fleet import runtime_wheel as r; "
              "print(json.dumps([r.__file__, r.source_checkout().as_posix(), "
              "Path(r.__file__).with_name('built-from').read_text().strip(), version('ml-stack')]))")
    done = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                          check=True, capture_output=True, text=True)
    installed, source, commit, version = json.loads(done.stdout)
    assert Path(installed).is_relative_to(target)
    assert source == str(root) and commit == COMMIT
    assert version == wheels[0].name.split("-")[1]
