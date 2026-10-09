"""Isolated runtime installation, private selection and retained worker imports."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from poolhouse import jobs, runtime
from poolhouse.fleet import runtime_wheel

A = "a" * 40
B = "b" * 40


def _wheel(tmp_path, commit, value, runtime_module=True):
    root = Path(__file__).resolve().parents[1]
    wheel = tmp_path / commit / "poolhouse-0.1.0-py3-none-any.whl"
    wheel.parent.mkdir()
    contents = {f"poolhouse/{name}": (root / "src/poolhouse" / name).read_bytes()
                for name in ("runtime.py", "home.py", "files.py", "platform.py", "windows_private.py")}
    contents["poolhouse/__init__.py"] = b""
    if not runtime_module:
        del contents["poolhouse/runtime.py"]
    contents["poolhouse/lazy_probe.py"] = f"VALUE = {value!r}\n".encode()
    contents["poolhouse/runtime_probe.py"] = b"from poolhouse.lazy_probe import VALUE\nprint(VALUE, flush=True)\n"
    contents["poolhouse-0.1.0.dist-info/METADATA"] = b"Metadata-Version: 2.1\nName: poolhouse\nVersion: 0.1.0\n"
    contents["poolhouse-0.1.0.dist-info/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: regression\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    contents["poolhouse-0.1.0.dist-info/RECORD"] = b""
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    runtime_wheel.stamp(wheel, commit, root)
    return wheel


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("POOLHOUSE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    return tmp_path


@pytest.mark.slow
def test_normal_package_without_installer_helpers_is_verified(isolated):
    chosen = runtime_wheel.prepare(_wheel(isolated, A, "normal", runtime_module=False), A, timeout=60)
    assert runtime.verify(chosen) == chosen
    assert not list(chosen.prefix.rglob("poolhouse/runtime.py"))


@pytest.mark.slow
def test_retained_worker_lazy_imports_stay_old_after_new_runtime_selection(isolated, monkeypatch):
    old = runtime_wheel.prepare(_wheel(isolated, A, "old"), A, timeout=60)
    runtime.publish(old)
    marker = old.prefix / "pyvenv.cfg"
    baseline = marker.read_bytes()
    old_sources = {path: path.read_bytes() for path in old.prefix.rglob("*.py")}
    script = "import sys;print('ready',flush=True);sys.stdin.readline();from poolhouse.lazy_probe import VALUE;print(VALUE,flush=True)"
    child = subprocess.Popen([str(old.python), "-I", "-c", script], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env=runtime.environment())
    try:
        assert child.stdout.readline().strip() == "ready"
        newer = runtime_wheel.prepare(_wheel(isolated, B, "new"), B, timeout=60)
        runtime.publish(newer)
        child.stdin.write("continue\n")
        child.stdin.flush()
        output, errors = child.communicate(timeout=20)
        assert child.returncode == 0, errors
        assert output.strip() == "old"
        assert marker.read_bytes() == baseline
        assert all(path.read_bytes() == content for path, content in old_sources.items())
        assert old.prefix != newer.prefix
        assert runtime.selected() == newer
        monkeypatch.setenv("PYTHONPATH", str(isolated / "changing-checkout"))
        detached = jobs.detach("poolhouse.runtime_probe", [], log=isolated / "new-worker.log")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and "new\n" not in detached.log.read_text():
            time.sleep(0.05)
        assert "new\n" in detached.log.read_text()
        assert detached.command[:4] == (str(newer.python), "-I", "-m", "poolhouse.runtime_probe")
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=10)


@pytest.mark.slow
def test_selected_runtime_rejects_system_sites_bad_marker_and_symlink(isolated):
    chosen = runtime_wheel.prepare(_wheel(isolated, A, "old"), A, timeout=60)
    config = chosen.prefix / "pyvenv.cfg"
    original = config.read_text()
    config.write_text(original.replace("include-system-site-packages = false", "include-system-site-packages = true"))
    with pytest.raises(OSError, match="isolated"):
        runtime.verify(chosen)
    config.write_text(original)
    marker = next(chosen.prefix.rglob("built-from"))
    marker.write_text(B)
    with pytest.raises(OSError, match="do not agree"):
        runtime.publish(chosen)
    marker.write_text(A)
    runtime.publish(chosen)
    target = runtime.directory() / "selected.json"
    saved = target.read_text()
    target.unlink()
    elsewhere = isolated / "elsewhere.json"
    elsewhere.write_text(saved)
    target.symlink_to(elsewhere)
    with pytest.raises(OSError, match="symbolic"):
        runtime.selected()


def test_platform_interpreter_selection_is_separate(isolated, monkeypatch):
    original = runtime.directory()
    monkeypatch.setattr(sys, "platform", "win32")
    assert runtime.directory() != original


def test_frozen_worker_without_a_selected_runtime_is_refused(isolated, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(OSError, match="standalone runtime"):
        runtime.python()


def test_corrupt_or_nonprivate_selection_is_refused(isolated):
    root = runtime.directory()
    root.mkdir(mode=0o700, parents=True)
    target = root / "selected.json"
    target.write_text("{}")
    target.chmod(0o600)
    with pytest.raises(ValueError, match="invalid runtime selection"):
        runtime.selected()
    if os.name != "nt":
        target.chmod(0o644)
        with pytest.raises(OSError, match="private"):
            runtime.selected()


def test_launcher_forward_preserves_arguments_and_removes_source_paths(isolated, monkeypatch):
    chosen = runtime.Runtime(isolated / "runtime", B, "0.1.0", runtime.identity())
    monkeypatch.setattr(runtime, "selected", lambda: chosen)
    monkeypatch.setenv("PYTHONPATH", str(isolated / "source"))
    calls = []
    monkeypatch.setattr(os, "execve", lambda *args: calls.append(args))
    assert runtime.forward("poolhouse.fleet.launch", ["--root", "studio", "--restart"])
    python, argv, environment = calls[0]
    assert python == str(chosen.python)
    assert argv == [python, "-I", "-m", "poolhouse.fleet.launch", "--root", "studio", "--restart"]
    assert "PYTHONPATH" not in environment


def test_selected_restart_uses_a_separate_authenticated_launcher(isolated, monkeypatch):
    from poolhouse.fleet import autostart

    chosen = runtime.Runtime(isolated / "runtime", B, "0.1.0", runtime.identity())
    monkeypatch.setattr(runtime, "selected", lambda: chosen)
    monkeypatch.setattr(sys, "argv", ["daemon", "--root", "studio", "--port", "8771"])
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("live daemon exec bypassed checkpoint"))
    monkeypatch.setattr(autostart, "_run", lambda args: pytest.fail("restarted an old service target"))
    calls = []
    monkeypatch.setattr(jobs, "detach", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert autostart.restart() == "launcher"
    module, argv = calls[0][0]
    assert (module, argv) == ("poolhouse.fleet.launch", ["--restart", "--no-browser", "--root", "studio", "--port", "8771"])
    from poolhouse.fleet import launch
    known, rest = launch._arguments(argv)
    assert known.restart and known.port == 8771 and rest == ["--root", "studio"]


@pytest.mark.slow
def test_installer_ignores_inherited_destinations_and_pip_config(isolated, monkeypatch):
    retained = isolated / 'retained-prefix'
    retained.mkdir()
    sentinel = retained / 'untouched'
    sentinel.write_text('retained')
    config = isolated / 'pip.conf'
    config.write_text(f'[global]\ntarget = {retained}\nprefix = {retained}\nuser = true\n')
    for key in ('PIP_TARGET', 'PIP_PREFIX', 'PIP_ROOT', 'PIP_PYTHON', 'PYTHONUSERBASE'):
        monkeypatch.setenv(key, str(retained))
    monkeypatch.setenv('PIP_USER', '1')
    monkeypatch.setenv('PIP_CONFIG_FILE', str(config))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(isolated))
    user_config = isolated / 'pip' / 'pip.conf'
    user_config.parent.mkdir()
    user_config.write_text(config.read_text())
    chosen = runtime_wheel.prepare(_wheel(isolated, A, 'clean'), A, timeout=60)
    assert runtime.verify(chosen) == chosen
    assert list(retained.iterdir()) == [sentinel]
    assert sentinel.read_text() == 'retained'
    assert runtime.installer_environment()['PIP_NO_INDEX'] == '1'
    assert runtime.installer_environment()['PIP_CONFIG_FILE'] == os.devnull
