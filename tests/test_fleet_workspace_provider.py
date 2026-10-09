"""Installed project-host composition for the direct device daemon."""

import importlib.metadata
import zipfile

import wheel_cache

from poolhouse.fleet import daemon


def test_injected_workspace_factory_bypasses_installed_provider(monkeypatch):
    projects = object()
    hosted = object()
    def unexpected(**_kwargs):
        raise AssertionError("injected factory must not look up installed providers")
    monkeypatch.setattr(daemon, "entry_points", unexpected)
    assert daemon.workspace_host(projects, lambda received: hosted if received is projects else None) is hosted


def test_direct_daemon_loads_workspace_provider_from_built_wheel(tmp_path, monkeypatch):
    wheels = list(wheel_cache.built().glob("*.whl"))
    assert len(wheels) == 1
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheels[0]) as archive:
        archive.extractall(installed)
    distributions = list(importlib.metadata.distributions(path=[str(installed)]))
    providers = [entry for distribution in distributions for entry in distribution.entry_points
                 if entry.group == "poolhouse.workspace_hosts" and entry.name == "default"]
    assert len(providers) == 1
    assert providers[0].value == "poolhouse.workspace.remote_host:WorkspaceHost"
    monkeypatch.setattr(daemon, "entry_points", lambda **_kwargs: providers)
    projects = object()
    hosted = daemon.workspace_host(projects)
    assert hosted.projects is projects
    assert type(hosted).__name__ == "WorkspaceHost"
