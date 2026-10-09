"""Metadata-only project registration and authority checks."""

from types import SimpleNamespace

import pytest

from poolhouse.files import write_json
from poolhouse.fleet import project_client, project_source as source, projects
from poolhouse.fleet.projects import ProjectRegistry, answer, identity
from poolhouse.net import git


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
                                                "board_host": ""}]
    assert not (tmp_path / "registry" / "project-bundles").exists()
    with pytest.raises(source.ProjectError, match="not shared"):
        registry.snapshot(identifier, "")
    with pytest.raises(source.ProjectError, match="another device"):
        registry.workspace_base(identifier)
    registry.claim_authority(identifier)
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
@pytest.mark.parametrize("authority, error", [
    ({"host": "https://foreign:8770"}, "another workspace address"),
    ({"host": "https://old:8770"}, "another workspace address"),
])
def test_authority_claim_refuses_host_remapping(repository, tmp_path, authority, error):
    identifier = identity(repository)
    write_json(repository / ".poolhouse-project.json", {"kind": "project-checkout", "project_id": identifier,
                                                        "authority": authority})
    registry = ProjectRegistry(tmp_path / "registry", "device", (repository,), "https://device:8770")
    with pytest.raises(source.ProjectError, match=error):
        registry.claim_authority(identifier)
    project = registry.get(identifier)
    assert project.board_host == authority.get("host", "")


@pytest.mark.redteam
@pytest.mark.parametrize("authority", [[], {"host": "x" * 2049}, {"host": "bad\naddress"}])
def test_malformed_checkout_authority_is_not_registered(repository, tmp_path, authority):
    write_json(repository / ".poolhouse-project.json", {"kind": "project-checkout",
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


def test_register_native_project_is_metadata_only_and_persistent(repository, tmp_path, monkeypatch):
    monkeypatch.setattr(source, "build", lambda *args: pytest.fail("metadata built source"))
    nested = repository / "nested"
    nested.mkdir()
    identifier = identity(repository)
    registry = ProjectRegistry(tmp_path / "registry", "device", host="https://device:8770")
    assert registry.register(nested, identifier) == {"id": identifier, "name": "project", "machine": "device",
                                                      "board_host": ""}
    assert registry.get(identifier).root == str(repository)
    assert registry.list() == []
    assert not (registry.root / "project-bundles").exists()
    with pytest.raises(source.ProjectError, match="not shared"):
        registry.snapshot(identifier, "")
    registry.claim_authority(identifier)
    assert registry.register(repository, identifier)["board_host"] == "https://device:8770"
    loaded = ProjectRegistry(registry.root, "device")
    assert loaded.get(identifier).board_host == "https://device:8770"
    assert loaded.list() == []


@pytest.mark.redteam
def test_native_registration_refuses_identity_mismatch_before_mutation(repository, tmp_path):
    registry = ProjectRegistry(tmp_path / "registry", "device")
    with pytest.raises(source.ProjectError, match="does not match"):
        registry.register(repository, "a" * 32)
    assert registry.boards() == []
    assert registry.candidates() == []
    assert not registry.path.exists()


@pytest.mark.redteam
@pytest.mark.parametrize("root", ["relative", "/" + "x" * 2048, "/bad\npath"])
def test_native_registration_refuses_unbounded_or_relative_path(tmp_path, root):
    registry = ProjectRegistry(tmp_path / "registry", "device")
    with pytest.raises(source.ProjectError, match="absolute local project path"):
        registry.register(type(tmp_path)(root), "a" * 32)
    assert registry.boards() == []
    assert not registry.path.exists()


@pytest.mark.parametrize('configuration', [
    ('https://device:8770', True),
    ('https://foreign:8770', False),
    ('', False),
])
def test_person_workspace_catalogue_includes_unshared_local_authority_only(repository, tmp_path, monkeypatch,
                                                                          configuration):
    host, visible = configuration
    registry = ProjectRegistry(tmp_path / 'registry', 'device', (repository,), 'https://device:8770')
    identifier = identity(repository)
    registered = registry.get(identifier)
    registered.board_host = host
    monkeypatch.setattr(project_client, 'peers', lambda ui: [])
    monkeypatch.setattr(source, 'build', lambda *args: pytest.fail('workspace catalogue published source'))
    result = project_client.available(SimpleNamespace(projects=registry))
    assert registered.shared is False
    assert result['projects'] == result['local'] == []
    assert registry.catalogue()['projects'] == []
    assert registry.catalogue()['boards'] == []
    assert result['workspaces'] == ([{
        'id': identifier, 'name': 'project', 'machine': 'device',
        'board_host': host,
        'is_self': True, 'local_authority': True,
    }] if visible else [])
    assert not (tmp_path / 'registry' / 'project-bundles').exists()


def test_registry_loads_projects_saved_with_an_authority_machine(tmp_path):
    identifier = "a" * 32
    write_json(tmp_path / "projects.json", {"projects": [{
        "id": identifier, "name": "old", "root": str(tmp_path), "source_machine": "device",
        "authority_machine": "retired", "board_host": "https://device:8770"}]})
    registry = ProjectRegistry(tmp_path, "device", host="https://device:8770")
    assert registry.get(identifier).board_host == "https://device:8770"
    assert registry.hosts(registry.get(identifier).board_host)
    assert not registry.hosts("")
    assert not registry.hosts("https://elsewhere:8770")
