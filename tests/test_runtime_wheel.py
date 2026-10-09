"""Runtime wheel provenance and installation boundaries."""

from __future__ import annotations

import base64
import csv
import email
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from importlib.metadata import distribution
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.tags import parse_tag, sys_tags
from wheel_support import fresh_wheel

from ml_stack import runtime
from ml_stack.fleet import runtime_wheel
from ml_stack.net import packages

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


@pytest.mark.parametrize("frozen", [False, True])
def test_build_uses_the_committed_snapshot_of_the_named_commit(tmp_path, monkeypatch, frozen):
    host = tmp_path / "host" / "python"
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    calls = []

    def run(argv, timeout):
        calls.append(argv)
        if argv[0] == "git" and "archive" in argv:
            with zipfile.ZipFile(argv[argv.index("--output") + 1], "w") as archive:
                archive.writestr("pyproject.toml", "[project]\nname='ml-stack'\n")
        if "wheel" in argv:
            wheels = Path(argv[argv.index("--wheel-dir") + 1])
            _wheel(wheels / "ml_stack-0.1.0-py3-none-any.whl")
        return "ok"

    monkeypatch.setattr(runtime_wheel, "_run", run)
    chosen = runtime.Runtime(tmp_path / "new-prefix", COMMIT, "0.1.0", runtime.identity())
    monkeypatch.setattr(runtime_wheel, "prepare", lambda wheel, commit, timeout, **kw: chosen)
    stage = tmp_path / "stage"
    stage.mkdir()
    assert runtime_wheel.build(tmp_path, COMMIT, stage, timeout=7, target=runtime_wheel.Target(host=host)) == chosen
    assert calls[0][-1] == COMMIT
    assert calls[1][0] == str(host)
    assert str(tmp_path) not in calls[1]
    assert not any("install" in argv for argv in calls)
    assert not any("-e" in argv for argv in calls)
    with pytest.raises(ValueError, match="full commit"):
        runtime_wheel.build(tmp_path, "HEAD", stage, timeout=7)


def test_a_failed_prepare_leaves_no_prefix_behind(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
    wheel = _wheel(tmp_path / "ml_stack-0.1.0-py3-none-any.whl")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("ml_stack-0.1.0.dist-info/METADATA", "Name: ml-stack\nVersion: 0.1.0\n")
    runtime_wheel.stamp(wheel, COMMIT, tmp_path)
    monkeypatch.setattr(runtime_wheel, "_run", lambda argv, timeout: (_ for _ in ()).throw(ValueError("missing dependency")))
    with pytest.raises(ValueError, match="missing dependency"):
        runtime_wheel.prepare(wheel, COMMIT, timeout=7)
    assert not list((runtime.directory() / COMMIT).iterdir())


def _dependency_wheels(wheel, target):
    with zipfile.ZipFile(wheel) as archive:
        meta = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        requirements = email.message_from_bytes(archive.read(meta)).get_all("Requires-Dist", [])
    pending = [(value, "train") for value in requirements]
    seen = set()
    target.mkdir()
    while pending:
        value, extra = pending.pop()
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate({"extra": extra}):
            continue
        if requirement.name in seen:
            continue
        seen.add(requirement.name)
        installed = distribution(requirement.name)
        pending.extend((value, "") for value in installed.requires or [])
        manifest = email.message_from_string(installed.read_text("WHEEL"))
        tags = {tag for value in manifest.get_all("Tag") for tag in parse_tag(value)}
        tag = next(tag for tag in sys_tags() if tag in tags)
        path = target / f"{installed.metadata['Name'].replace('-', '_')}-{installed.version}-{tag}.whl"
        with zipfile.ZipFile(path, "w") as archive:
            for entry in installed.files:
                if ".." not in entry.parts and installed.locate_file(entry).is_file():
                    archive.write(installed.locate_file(entry), entry.as_posix())
    return target


@pytest.mark.slow
def test_built_wheel_imports_from_an_immutable_install(tmp_path):
    root = Path(__file__).resolve().parents[1]
    built = fresh_wheel(root)
    wheel = tmp_path / built.name
    shutil.copy2(built, wheel)
    runtime_wheel.stamp(wheel, COMMIT, root)
    target = tmp_path / "installed"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                    "--target", str(target), str(wheel)], check=True, capture_output=True, text=True)
    prefix = tmp_path / "runtime-prefix"
    cached = prefix / "ml-stack-wheels" / COMMIT / wheel.name
    cached.parent.mkdir(parents=True)
    shutil.copy2(wheel, cached)
    newer = tmp_path / "newer" / wheel.name
    newer.parent.mkdir()
    shutil.copy2(wheel, newer)
    runtime_wheel.stamp(newer, "f" * 40, root)
    next_cache = prefix / "ml-stack-wheels" / ("f" * 40) / newer.name
    next_cache.parent.mkdir(parents=True)
    shutil.copy2(newer, next_cache)
    env = {**os.environ, "PYTHONPATH": str(target), "PIP_NO_INDEX": "1",
           "PIP_FIND_LINKS": str(_dependency_wheels(wheel, tmp_path / "dependency-wheels"))}
    script = ("import json, os, subprocess, sys; from pathlib import Path; from importlib.metadata import version; "
              "from ml_stack.fleet import runtime_wheel as r; from ml_stack.fleet.environment import Environment; "
              f"sys.prefix = {str(prefix)!r}; managed = Environment(Path({str(tmp_path)!r}) / 'managed', python_version=f'{sys.version_info.major}.{sys.version_info.minor}'); "
              "result = managed.install(['core']); assert result['core']['ok'], result; "
              "clean = dict(os.environ); clean.pop('PYTHONPATH', None); "
              "loaded = subprocess.run([str(managed.python), '-c', "
              "'from ml_stack.fleet import runtime_wheel as r; print(r.current_wheel()); print(r.wheel_commit(r.current_wheel()))'], "
              "env=clean, capture_output=True, text=True, check=True); "
              "before = Path(r.__file__).with_name('built-from').read_text().strip(); old_wheels = str(managed.wheels()); "
              "Path(r.__file__).with_name('built-from').write_text('f' * 40); "
              "updated = managed.install(['core']); assert updated['core']['ok'], updated; "
              "reloaded = subprocess.run([str(managed.python), '-c', "
              "'from ml_stack.fleet import runtime_wheel as r; print(r.wheel_commit(r.current_wheel()))'], "
              "env=clean, capture_output=True, text=True, check=True); "
              "print(json.dumps([r.__file__, r.source_checkout().as_posix(), "
              "before, version('ml-stack'), old_wheels, loaded.stdout.splitlines(), reloaded.stdout.strip()]))")
    done = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                          check=False, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    installed, source, commit, version, wheels, managed_runtime, updated_commit = json.loads(done.stdout)
    assert Path(installed).is_relative_to(target)
    assert Path(source) == root and commit == COMMIT
    assert version == wheel.name.split("-")[1]
    assert Path(wheels) == cached.parent
    assert Path(managed_runtime[0]).is_relative_to(tmp_path / "managed" / "env")
    assert managed_runtime[1] == COMMIT
    assert updated_commit == "f" * 40


def test_current_wheel_requires_matching_immutable_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "prefix"))
    package = tmp_path / "installed" / "runtime_wheel.py"
    package.parent.mkdir()
    package.write_text("")
    package.with_name("built-from").write_text(COMMIT)
    monkeypatch.setattr(runtime_wheel, "__file__", str(package))
    with pytest.raises(OSError, match="unavailable"):
        runtime_wheel.current_wheel()
    wheel = _wheel(tmp_path / "ml_stack-0.1.0-py3-none-any.whl")
    runtime_wheel.stamp(wheel, COMMIT, tmp_path)
    cached = runtime_wheel.cache_wheel(wheel, COMMIT)
    assert runtime_wheel.current_wheel() == cached
    runtime_wheel.stamp(cached, "f" * 40, tmp_path)
    with pytest.raises(OSError, match="does not match"):
        runtime_wheel.current_wheel()


def test_managed_install_uses_direct_cached_wheel_and_refreshes_same_version(tmp_path, monkeypatch):
    from ml_stack.fleet.environment import Environment
    wheel = _wheel(tmp_path / "wheels" / "ml_stack-0.1.0-py3-none-any.whl")
    environment = Environment(tmp_path)
    monkeypatch.setattr(environment, "wheels", lambda: wheel.parent)
    monkeypatch.setattr(environment, "build_environment", lambda: {})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    monkeypatch.setattr(packages, "run", lambda python, args, **kwargs: run([str(python), "-m", "pip", *args], **kwargs))
    assert environment.pip(["install", "--upgrade", "ml-stack[train]"]).returncode == 0
    assert calls[0][-1] == f"ml-stack[train] @ {wheel.as_uri()}"
    assert calls[0][4:6] == ["--find-links", str(wheel.parent)]
    assert calls[1][-5:] == ["install", "--force-reinstall", "--no-deps", "--no-index", str(wheel)]


def test_third_party_installs_keep_matching_bundled_wheels(tmp_path, monkeypatch):
    from ml_stack.fleet.environment import Environment
    bundled = tmp_path / "bundled-wheels"
    bundled.mkdir()
    environment = Environment(tmp_path)
    monkeypatch.setattr(environment, "wheels", lambda: bundled)
    monkeypatch.setattr(environment, "build_environment", lambda: {})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    monkeypatch.setattr(packages, "run", lambda python, args, **kwargs: run([str(python), "-m", "pip", *args], **kwargs))
    assert environment.pip(["install", "--no-index", "torch"]).returncode == 0
    assert calls == [[str(environment.python), "-m", "pip", "install", "--find-links", str(bundled), "--no-index", "torch"]]


def test_frozen_build_uses_verified_external_host_python(tmp_path, monkeypatch):
    from ml_stack import runtime_deploy
    from ml_stack.fleet.environment import Environment
    host = tmp_path / "host" / "python"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(Environment, "host_python", lambda self: host)
    assert runtime_deploy.host_python() == host


def test_missing_host_python_refuses_before_selection(tmp_path, monkeypatch):
    from ml_stack.fleet.environment import Environment
    monkeypatch.setattr(Environment, "host_python", lambda self: None)
    monkeypatch.setattr(runtime, "publish", lambda value: pytest.fail("selected failed build"))
    from ml_stack import runtime_deploy
    with pytest.raises(OSError, match=r"Python 3\.13"):
        runtime_deploy.host_python()
