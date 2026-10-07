"""Native bundles require artifacts, verified signing and a runnable installer."""

import importlib.util
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def builder(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("desktop_builder", ROOT / "packaging/build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "APP", tmp_path / "app")
    monkeypatch.setattr(module, "DIST", tmp_path / "dist")
    monkeypatch.setattr(module, "target_triple", lambda: "test-host")
    monkeypatch.setattr(module.shutil, "which", lambda _: "npm")
    return module


def test_window_refuses_empty_native_build(builder, tmp_path, monkeypatch):
    sidecar = tmp_path / "daemon"
    sidecar.write_bytes(b"daemon")
    monkeypatch.setattr(builder, "run", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit, match="no application bundle"):
        builder.window(sidecar)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS application bundles")
def test_window_signs_the_built_application_name(builder, tmp_path, monkeypatch):
    sidecar = tmp_path / "daemon"
    sidecar.write_bytes(b"daemon")
    artifact = builder.APP / "src-tauri/target/release/bundle/macos/Poolside.app"
    (artifact / "Contents/MacOS").mkdir(parents=True)
    (artifact / "Contents/MacOS/app").write_bytes(b"native")
    calls = []
    monkeypatch.setattr(builder, "run", lambda argv, **kwargs: calls.append(argv))
    made = builder.window(sidecar)
    assert made == [builder.DIST / "bundle/Poolside.app"]
    assert [argv for argv in calls if argv[0] == "codesign"] == [
        ["codesign", "--force", "--deep", "--sign", "-", str(made[0])],
        ["codesign", "--verify", "--deep", "--strict", str(made[0])],
    ]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS installer")
def test_offline_app_install_preserves_quarantine_and_opens_poolside(tmp_path):
    archive = tmp_path / "demo.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("Poolside.app/Contents/MacOS/app", "native")
    tools = tmp_path / "tools"
    tools.mkdir()
    log = tmp_path / "opened"
    for name, text in {
        "open": '#!/bin/sh\nprintf "%s" "$1" > "$OPEN_LOG"\n',
        "xattr": '#!/bin/sh\nexit 91\n',
    }.items():
        tool = tools / name
        tool.write_text(text)
        tool.chmod(0o755)
    destination = tmp_path / "Applications"
    destination.mkdir()
    done = subprocess.run(["sh", str(ROOT / "packaging/install.sh"), "--app"],
        capture_output=True, text=True, timeout=30, env={
            **os.environ, "HOME": str(tmp_path), "PATH": f"{tools}:/usr/bin:/bin",
            "ML_STACK_OFFLINE_ZIP": str(archive), "ML_STACK_DEST": str(destination),
            "OPEN_LOG": str(log),
        })
    assert done.returncode == 0, done.stdout + done.stderr
    assert (destination / "Poolside.app/Contents/MacOS/app").read_text() == "native"
    assert log.read_text() == str(destination / "Poolside.app")
    assert "xattr" not in (ROOT / "packaging/install.sh").read_text()


@pytest.fixture
def frozen_collector():
    spec = importlib.util.spec_from_file_location("desktop_frozen", ROOT / "packaging/frozen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_collects_wheel_modules_and_dynamic_entrypoints(frozen_collector, monkeypatch):
    from pathlib import PurePosixPath
    from types import SimpleNamespace

    hooks = type(sys)("PyInstaller.utils.hooks")
    calls = []

    def collect_entry_point(group):
        calls.append(group)
        return [("plugin-metadata", group)], ["plugin_provider.routes"]

    hooks.collect_entry_point = collect_entry_point
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    entries = [
        SimpleNamespace(group="console_scripts", module="ml_stack.fleet.daemon"),
        SimpleNamespace(group="ml_stack.workspace_hosts", module="ml_stack.workspace.remote_host"),
        SimpleNamespace(group="ml_stack.ui_routes", module="ml_stack.workspace.agent_routes"),
        SimpleNamespace(group="ml_stack.ui_routes", module="ml_stack.workspace.task_routes"),
    ]
    installed = SimpleNamespace(files=[PurePosixPath(path) for path in [
        "ml_stack/__init__.py", "ml_stack/world/__init__.py", "ml_stack/bench/cli.py",
        "ml_stack/ingest/cli.py", "ml_stack/workspace/remote_host.py",
        "ml_stack/ui/assets/index.html", "ml_stack-1.0.dist-info/METADATA",
        "../bin/ml-stack", "ml_stack/non-module.py",
    ]], entry_points=entries)
    monkeypatch.setattr(frozen_collector, "distribution", lambda name: installed)
    datas, modules = frozen_collector.collect_project()
    assert calls == ["ml_stack.ui_routes", "ml_stack.workspace_hosts"]
    assert datas == [("plugin-metadata", group) for group in calls]
    assert modules == [
        "ml_stack", "ml_stack.bench.cli", "ml_stack.fleet.daemon", "ml_stack.ingest.cli",
        "ml_stack.workspace.agent_routes", "ml_stack.workspace.remote_host",
        "ml_stack.workspace.task_routes", "ml_stack.world", "plugin_provider.routes",
    ]


def test_frozen_spec_includes_project_modules_without_project_exclusions(monkeypatch):
    import runpy
    from types import SimpleNamespace

    hooks = type(sys)("PyInstaller.utils.hooks")
    hooks.collect_dynamic_libs = lambda name: []
    hooks.collect_submodules = lambda name: []
    hooks.copy_metadata = lambda name: []
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    native_modules = [
        "ml_stack.workspace.remote_host", "ml_stack.world", "ml_stack.bench", "ml_stack.ingest",
    ]
    execute = runpy.run_path
    monkeypatch.setattr(runpy, "run_path", lambda path: {
        "collect_project": lambda: ([], native_modules),
    })
    captured = {}

    def analysis(scripts, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(pure=[], scripts=[], binaries=[], datas=[])

    execute(str(ROOT / "packaging/ml-stack.spec"), init_globals={
        "SPECPATH": str(ROOT / "packaging"),
        "Analysis": analysis,
        "PYZ": lambda pure: None,
        "EXE": lambda *args, **kwargs: None,
    })
    assert set(native_modules) <= set(captured["hiddenimports"])
    assert not any(name.startswith("ml_stack") for name in captured["excludes"])
