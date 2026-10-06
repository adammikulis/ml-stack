"""Installed project-host composition for the direct device daemon."""

import importlib.metadata
import subprocess
import sys
import zipfile
from pathlib import Path

from ml_stack.fleet import daemon


def test_injected_workspace_factory_bypasses_installed_provider(monkeypatch):
    projects = object()
    hosted = object()
    def unexpected(**_kwargs):
        raise AssertionError("injected factory must not look up installed providers")
    monkeypatch.setattr(daemon, "entry_points", unexpected)
    assert daemon.workspace_host(projects, lambda received: hosted if received is projects else None) is hosted


def test_direct_daemon_loads_workspace_provider_from_built_wheel(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    built = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation",
                            "--outdir", str(tmp_path), str(repository)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheels[0]) as archive:
        archive.extractall(installed)
    distributions = list(importlib.metadata.distributions(path=[str(installed)]))
    providers = [entry for distribution in distributions for entry in distribution.entry_points
                 if entry.group == "ml_stack.workspace_hosts" and entry.name == "default"]
    assert len(providers) == 1
    assert providers[0].value == "ml_stack.workspace.remote_host:WorkspaceHost"
    monkeypatch.setattr(daemon, "entry_points", lambda **_kwargs: providers)
    projects = object()
    hosted = daemon.workspace_host(projects)
    assert hosted.projects is projects
    assert type(hosted).__name__ == "WorkspaceHost"
