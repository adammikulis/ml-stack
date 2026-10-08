"""Desktop origins cannot invoke native shell or filesystem commands, and the window stays on its daemon."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.redteam
ROOT = Path(__file__).resolve().parents[1]
RUSTUP_HOME = os.environ.get("RUSTUP_HOME", str(Path.home() / ".rustup"))
CARGO_HOME = os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))


def test_desktop_grants_only_window_and_close_commands():
    capability = json.loads((ROOT / "app/src-tauri/capabilities/main.json").read_text())
    assert capability["windows"] == ["main"]
    assert capability["remote"]["urls"] == ["http://127.0.0.1:8770/*"]
    assert capability["local"] is False
    assert set(capability["permissions"]) == {
        "core:default", "window-state:default", "allow-close-choice", "allow-on-closing",
    }


@pytest.mark.slow
@pytest.mark.parametrize("selector", ["security_tests", "daemon::tests", "identity::tests"])
def test_native_capability_authority_rejects_hostile_origins_and_commands(tmp_path, selector):
    if sys.platform != "darwin":
        pytest.skip("native desktop capability enforcement runs on macOS")
    cargo = shutil.which("cargo")
    assert cargo, "Cargo is required to verify desktop capabilities"
    project = tmp_path / "desktop"
    shutil.copytree(ROOT / "app/src-tauri", project,
                    ignore=shutil.ignore_patterns("target", "binaries", "gen"))
    native_env = {**os.environ, "RUSTUP_HOME": RUSTUP_HOME, "CARGO_HOME": CARGO_HOME}
    host = subprocess.run(["rustc", "-vV"], env=native_env,
                          capture_output=True, text=True, check=True)
    target = next(line.removeprefix("host: ") for line in host.stdout.splitlines()
                  if line.startswith("host: "))
    binaries = project / "binaries"
    binaries.mkdir()
    (binaries / f"ml-stack-headless-{target}").write_bytes(b"")
    done = subprocess.run(
        [cargo, "test", "--manifest-path", str(project / "Cargo.toml"),
         "--offline", selector, "--", "--nocapture"],
        env={**native_env, "CARGO_TARGET_DIR": os.environ.get(
            "ML_STACK_TAURI_TEST_TARGET", str(tmp_path / "target"))},
        capture_output=True, text=True, timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "test result: ok" in done.stdout and " 0 failed" in done.stdout
    assert "running 0 tests" not in done.stdout
