"""Metadata-only project registration and canonical authority checks."""

from types import SimpleNamespace

import pytest

from ml_stack.files import write_json
from ml_stack.fleet import projects, project_source as source
from ml_stack.fleet.projects import ProjectRegistry, answer, identity
from ml_stack.net import git


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    git.run(["init"], cwd=root)
    git.run(["remote", "add", "origin", "https://code.example.invalid/team/project.git"], cwd=root)
    return root


def test_metadata_registration_never_exports_source(repository, tmp_path, monkeypatch):
    monkeypatch.setattr(source, "build", lambda *args: pytest.fail("metadata built source"))
    registry = ProjectRegistry(tmp_path / "registry", "device", (repository,), "https://device:8770")
    identifier = identity(repository)
    assert registry.candidates() == [{"id": identifier, "name": "project"}]
    assert registry.list() == []
    assert registry.catalogue()["boards"] == []
    assert registry.catalogue(include_boards=True)["boards"] == [{"id": identifier, "name": "project", "machine": "device",
                                                "authority_machine": "", "board_host": ""}]
    assert not (tmp_path / "registry" / "project-bundles").exists()
    with pytest.raises(source.ProjectError, match="not shared"):
        registry.snapshot(identifier, "")
    with pytest.raises(source.ProjectError, match="another device"):
        registry.workspace_base(identifier)
    registry.claim_authority(identifier, expected_machine="device")
    assert registry.workspace_base(identifier).name == identifier
    loaded = ProjectRegistry(tmp_path / "registry", "device", host="https://device:8770")
    assert loaded.get(identifier).board_host == "https://device:8770"
    assert loaded.list() == []


def test_same_git_origin_has_same_board_identity(repository, tmp_path):
    other = tmp_path / "another-checkout"
    other.mkdir()
    git.run(["init"], cwd=other)
    git.run(["remote", "add", "origin", "git@code.example.invalid:team/project.git"], cwd=other)
    assert identity(other) == identity(repository)
    registry = ProjectRegistry(tmp_path / "registry", "device", (repository, other))
    assert len(registry.boards()) == 1


@pytest.mark.redteam
@pytest.mark.parametrize("authority, expected, error", [
    ({}, "foreign", "does not match"),
    ({"machine": "foreign", "host": "https://foreign:8770"}, "device", "another workspace authority"),
    ({"machine": "device", "host": "https://old:8770"}, "device", "another workspace address"),
])
def test_authority_claim_refuses_device_or_host_remapping(repository, tmp_path, authority, expected, error):
    identifier = identity(repository)
    write_json(repository / ".ml-stack-project.json", {"kind": "project-checkout", "project_id": identifier,
                                                        "authority": authority})
    registry = ProjectRegistry(tmp_path / "registry", "device", (repository,), "https://device:8770")
    with pytest.raises(source.ProjectError, match=error):
        registry.claim_authority(identifier, expected_machine=expected)
    project = registry.get(identifier)
    assert project.authority_machine == authority.get("machine", "")
    assert project.board_host == authority.get("host", "")


@pytest.mark.redteam
@pytest.mark.parametrize("authority", [[], {"machine": []}, {"host": "x" * 2049}, {"host": "bad\naddress"}])
def test_malformed_checkout_authority_is_not_registered(repository, tmp_path, authority):
    write_json(repository / ".ml-stack-project.json", {"kind": "project-checkout",
                                                        "project_id": identity(repository),
                                                        "authority": authority})
    registry = ProjectRegistry(tmp_path / "registry", "device", (repository,))
    assert registry.boards() == []
    assert registry.candidates() == []


@pytest.mark.redteam
@pytest.mark.parametrize("visible", [False, True])
def test_catalogue_uses_authenticated_visibility_predicate(repository, tmp_path, monkeypatch, visible):
    registry = ProjectRegistry(tmp_path / "registry", "device", (repository,))
    connection, sealing, cluster_path = object(), object(), tmp_path / "cluster-key"
    checked, responses = [], []

    def predicate(conn, opening, path):
        checked.append((conn, opening, path))
        return visible

    monkeypatch.setattr(projects.project_enrollment, "visible", predicate)
    handler = SimpleNamespace(connection=connection, _sealing=lambda: sealing,
                              _send=lambda code, payload: responses.append((code, payload)))
    assert answer(handler, registry, SimpleNamespace(path="/workspace/v1/projects"),
                  cluster_key_path=cluster_path)
    assert checked == [(connection, sealing, cluster_path)]
    assert bool(responses[0][1]["boards"]) is visible
    assert responses[0][1]["projects"] == []
