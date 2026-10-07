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
