"""Installed source discovery and local Windows checkout registration."""

import subprocess
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from ml_stack.fleet import project_client, projects, runtime_wheel
from ml_stack.fleet.project_source import ProjectError
from ml_stack.http import ServerUnreachable
from ml_stack.net import git


def repository(path):
    path.mkdir()
    git.run(["init"], cwd=path)
    git.run(["remote", "add", "origin", "https://code.example.invalid/team/project.git"], cwd=path)
    return path


def test_recorded_checkout_registered_from_unrelated_runtime_directory(tmp_path, monkeypatch):
    checkout = repository(tmp_path / "source")
    unrelated = tmp_path / "runtime"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    installed = tmp_path / "installed/ml_stack/fleet"
    installed.mkdir(parents=True)
    (installed / runtime_wheel.ORIGIN).write_text(str(checkout))
    monkeypatch.setattr(projects, "__file__", str(installed / "projects.py"))
    monkeypatch.setattr(runtime_wheel, "__file__", str(installed / "runtime_wheel.py"))
    monkeypatch.setattr(projects.source, "build", lambda *args: pytest.fail("source was exported"))
    registry = projects.ProjectRegistry(tmp_path / "registry", "device", projects.local_candidates())
    assert registry.candidates() == [{"id": projects.identity(checkout), "name": "source"}]
    assert registry.boards()[0]["id"] == projects.identity(checkout)
    assert registry.list() == []
    assert not (registry.root / "project-bundles").exists()


def test_absent_installed_source_provenance_keeps_available_candidates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(projects, "__file__", str(tmp_path / "installed/ml_stack/fleet/projects.py"))
    monkeypatch.setattr(runtime_wheel, "source_checkout", lambda: None)
    assert projects.local_candidates()[-1] == tmp_path
    assert projects.ProjectRegistry(tmp_path / "registry", "device", projects.local_candidates()).boards() == []


def test_stale_installed_source_provenance_is_not_a_candidate(tmp_path, monkeypatch):
    module = tmp_path / "runtime_wheel.py"
    module.write_text("")
    (tmp_path / runtime_wheel.ORIGIN).write_text(str(tmp_path / "missing"))
    monkeypatch.setattr(runtime_wheel, "__file__", str(module))
    assert runtime_wheel.source_checkout() is None


@pytest.mark.redteam
def test_windows_registration_in_wsl_uses_literal_argv_and_verified_git_root(tmp_path, monkeypatch):
    checkout = repository(tmp_path / "source")
    calls = []
    requested = r"C:\repos\project $(unexpected)"
    monkeypatch.setattr(projects.wsl_startup, "guest", lambda: True)

    def translate(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=str(checkout) + "\n")

    monkeypatch.setattr(projects.subprocess, "run", translate)
    assert projects.local_root(requested) == checkout
    assert calls == [(["wslpath", "-a", "-u", requested],
                      {"capture_output": True, "text": True, "timeout": 5, "check": True})]
    monkeypatch.undo()
    registry = projects.ProjectRegistry(tmp_path / "registry", "device")
    assert registry.register(checkout, projects.identity(checkout))["id"] == projects.identity(checkout)
    assert registry.list() == []


@pytest.mark.redteam
@pytest.mark.parametrize("matching", [False, True])
def test_sealed_wsl_registration_checks_identity_before_persisting(tmp_path, monkeypatch, matching):
    checkout = repository(tmp_path / "source")
    identifier = projects.identity(checkout) if matching else "a" * 32
    original = subprocess.run
    monkeypatch.setattr(projects.wsl_startup, "guest", lambda: True)
    monkeypatch.setattr(projects.project_enrollment, "local", lambda *args: True)

    def translate(args, **kwargs):
        return SimpleNamespace(stdout=str(checkout) + "\n") if args[0] == "wslpath" else original(args, **kwargs)

    monkeypatch.setattr(projects.subprocess, "run", translate)
    responses = []
    handler = SimpleNamespace(path="/workspace/v1/local-project", client_address=("127.0.0.1", 0),
                              _sealing=lambda: object(), _send=lambda *args: responses.append(args),
                              _object=lambda body: {"root": r"C:\repos\project", "project_id": identifier})
    registry = projects.ProjectRegistry(tmp_path / "registry", "device")
    assert projects.register(handler, registry, tmp_path / "cluster", b"request")
    assert responses[0][0] == (200 if matching else 400)
    assert bool(registry.boards()) is matching
    assert registry.path.exists() is matching
    assert registry.list() == []


@pytest.mark.redteam
def test_registration_rejects_unauthenticated_caller_before_path_translation(tmp_path, monkeypatch):
    monkeypatch.setattr(projects.project_enrollment, "local", lambda *args: False)
    monkeypatch.setattr(projects.subprocess, "run", lambda *args, **kwargs: pytest.fail("unauthorized spawn"))
    responses = []
    handler = SimpleNamespace(path="/workspace/v1/local-project", client_address=("192.168.4.8", 0),
                              _sealing=lambda: object(), _send=lambda *args: responses.append(args))
    registry = projects.ProjectRegistry(tmp_path / "registry", "device")
    assert projects.register(handler, registry, tmp_path / "cluster", b"request")
    assert responses[0][0] == 403
    assert registry.boards() == []


@pytest.mark.redteam
@pytest.mark.parametrize("value", [r"C:relative", r"\\host\share\project", r"\\?\C:\project",
                                  "relative", "/bad\npath", "x" * 2049, None, []])
def test_registration_refuses_invalid_path_without_spawning(tmp_path, monkeypatch, value):
    monkeypatch.setattr(projects.wsl_startup, "guest", lambda: True)
    monkeypatch.setattr(projects.subprocess, "run", lambda *args, **kwargs: pytest.fail("invalid path spawned"))
    with pytest.raises(ProjectError, match="absolute local project path"):
        projects.local_root(value)


@pytest.mark.redteam
@pytest.mark.parametrize("result", ["relative", "", "/bad\npath", "/" + "x" * 2048])
def test_registration_refuses_invalid_wsl_translation(monkeypatch, result):
    monkeypatch.setattr(projects.wsl_startup, "guest", lambda: True)
    monkeypatch.setattr(projects.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=result))
    with pytest.raises(ProjectError, match="absolute local project path"):
        projects.local_root(r"C:\repos\project")


@pytest.mark.redteam
def test_wsl_translation_failure_is_not_silent(monkeypatch):
    monkeypatch.setattr(projects.wsl_startup, "guest", lambda: True)

    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(projects.subprocess, "run", failed)
    with pytest.raises(ProjectError, match="could not be resolved"):
        projects.local_root(r"C:\repos\project")


@pytest.mark.redteam
def test_windows_drive_path_is_not_translated_outside_wsl(monkeypatch):
    monkeypatch.setattr(projects.wsl_startup, "guest", lambda: False)
    monkeypatch.setattr(projects, "Path", PurePosixPath)
    monkeypatch.setattr(projects.subprocess, "run", lambda *args, **kwargs: pytest.fail("not WSL"))
    with pytest.raises(ProjectError, match="absolute local project path"):
        projects.local_root(r"C:\repos\project")


def test_local_registration_requires_available_daemon(tmp_path, monkeypatch):
    def unreachable(*args, **kwargs):
        raise ServerUnreachable("offline")

    monkeypatch.setattr(project_client, "open_stream", unreachable)
    with pytest.raises(ProjectError, match="start ml-stack and retry"):
        project_client.register_local(tmp_path, b"k" * 32, "a" * 32, 8770)
