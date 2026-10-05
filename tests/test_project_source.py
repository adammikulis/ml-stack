"""Project publication, hostile archives and canonical checkout authority."""

import hashlib
import io
import json
import runpy
import threading
import zipfile
from types import SimpleNamespace

import pytest

from ml_stack.files import read_json, write_json
from ml_stack.fleet import project_client, project_source as source
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.discovery import derive_token
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.project_client import catalogue, receive
from ml_stack.fleet.projects import ProjectRegistry, bootstrap
from ml_stack.fleet.remote import Peer
from ml_stack.http import Server, ServerError
from ml_stack.net import git

IDENTIFIER = "a" * 32


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    git.run(["init"], cwd=root)
    (root / "run.sh").write_bytes(b"#!/bin/sh\necho hello\n")
    (root / ".gitignore").write_text("ignored.txt\n")
    (root / "ignored.txt").write_text("private\n")
    (root / ".env").write_text("PRIVATE=private\n")
    git.run(["add", "--", "run.sh", ".gitignore", ".env"], cwd=root)
    git.run(["-c", "user.name=Test", "-c", "user.email=test@example.invalid",
             "-c", "commit.gpgsign=false", "commit", "-m", "fixture"], cwd=root)
    return root


def test_index_snapshot_preserves_lf_and_excludes_private_files(repository):
    (repository / "run.sh").write_bytes(b"changed\r\n")
    manifest, packed = source.build(repository, IDENTIFIER)
    _, contents = source.verify(packed, IDENTIFIER, manifest["source_hash"])
    assert contents["run.sh"] == b"#!/bin/sh\necho hello\n"
    assert ".env" not in contents and "ignored.txt" not in contents
    assert manifest["revision"] == git.run(["rev-parse", "HEAD"], cwd=repository).stdout.strip()


@pytest.mark.parametrize("name", ["../bad", "/root", "a\\b", "AUX.txt", "a:stream", "a//b", "a. /b"])
def test_refuses_hostile_source_names(name):
    with pytest.raises(source.ProjectError):
        source.safe_name(name)


def archive(names):
    entries = [{"path": name, "size": 1, "sha256": hashlib.sha256(b"x").hexdigest(),
                "executable": False} for name in names]
    digest = source._hash(entries)
    manifest = {"kind": "tracked-source", "project_id": IDENTIFIER, "files": entries,
                "source_hash": digest, "size_bytes": len(entries)}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as packed:
        packed.writestr("manifest.json", json.dumps(manifest))
        for name in names:
            packed.writestr("files/" + name, b"x")
    return output.getvalue(), digest


@pytest.mark.parametrize("names", [["Readme", "README"], ["a", "a/b"], [".git/config"], [".aws/config"]])
def test_archive_rejected_before_checkout(names, tmp_path):
    packed, digest = archive(names)
    base = tmp_path / "checkouts"
    with pytest.raises(source.ProjectError):
        source.checkout(packed, IDENTIFIER, digest, base)
    assert not base.exists()


def test_checkout_uses_verified_blobs_without_filters(repository, tmp_path):
    (repository / ".gitattributes").write_text("*.sh filter=malicious\n")
    git.run(["add", "--", ".gitattributes"], cwd=repository)
    manifest, packed = source.build(repository, IDENTIFIER)
    result = source.checkout(packed, IDENTIFIER, manifest["source_hash"], tmp_path / "checkout",
                             authority={"machine": "pc", "host": "https://pc:8770"})
    assert (result / "run.sh").read_bytes() == b"#!/bin/sh\necho hello\n"
    assert read_json(result / ".ml-stack-project.json", {})["authority"]["machine"] == "pc"
    assert git.run(["status", "--porcelain"], cwd=result).stdout.strip() == "?? .ml-stack-project.json"


def test_registry_requires_explicit_selection_and_unshare_closes_source(repository, tmp_path):
    registry = ProjectRegistry(tmp_path / "daemon", "pc", (repository,))
    with pytest.raises(source.ProjectError):
        registry.share(str(repository))
    project = registry.share(registry.candidates()[0]["id"])
    assert registry.workspace_base(project.id).name == project.id
    assert registry.snapshot(project.id, project.source_hash)
    registry.unshare(project.id)
    with pytest.raises(source.ProjectError):
        registry.snapshot(project.id, project.source_hash)


def test_remote_checkout_cannot_create_another_authority(repository, tmp_path):
    write_json(repository / ".ml-stack-project.json", {"kind": "project-checkout", "project_id": IDENTIFIER,
                                                        "authority": {"machine": "other"}})
    registry = ProjectRegistry(tmp_path / "daemon", "pc", (repository,))
    with pytest.raises(source.ProjectError, match="authority"):
        registry.share(registry.candidates()[0]["id"])


def test_http_source_requires_auth_and_transfers_verified_checkout(repository, tmp_path):
    registry = ProjectRegistry(tmp_path / "daemon", "pc", (repository,))
    project = registry.share(registry.candidates()[0]["id"])
    runner = JobRunner(tmp_path / "jobs")
    token = derive_token(b"p" * 32)
    server = Server(("127.0.0.1", 0), make_handler(Daemon(runner, tmp_path, token, projects=registry)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(ServerError):
            catalogue(Peer(base, "wrong-token"))
        assert catalogue(Peer(base, token))["projects"][0]["id"] == project.id
        result = receive(Peer(base, token), project.id, tmp_path / "received")
        assert (result / "run.sh").read_bytes().startswith(b"#!/bin/sh\n")
        registry.unshare(project.id)
        with pytest.raises(source.ProjectError):
            receive(Peer(base, token), project.id, tmp_path / "received-again")
    finally:
        server.shutdown()
        server.server_close()
        runner.shutdown()


@pytest.mark.parametrize("payload", [{"action": "share", "candidate_id": "../../etc"},
                                    {"action": "unknown", "candidate_id": "x"}])
def test_ui_project_share_refuses_arbitrary_paths(repository, tmp_path, payload):
    registry = ProjectRegistry(tmp_path / "daemon", "pc", (repository,))
    answers = []
    request = SimpleNamespace(ui=SimpleNamespace(projects=registry), path="/ui/projects", method="POST",
                              body=lambda: payload, send=lambda *args: answers.append(args))
    assert project_client.route(request)
    assert answers[0][0] == 400 and registry.list() == []


def test_git_blob_cannot_be_an_option_or_oversized(repository):
    with pytest.raises(git.GitFailed):
        git.blobs(["--help"], cwd=repository, limit=10, total=10)
    digest = git.run(["rev-parse", "HEAD:run.sh"], cwd=repository).stdout.strip()
    with pytest.raises(git.GitFailed):
        git.blobs([digest], cwd=repository, limit=1, total=1)


def test_standalone_bootstrap_contains_verified_checkout_helpers(repository, tmp_path):
    manifest, packed = source.build(repository, IDENTIFIER)
    script = tmp_path / "bootstrap.py"
    script.write_bytes(bootstrap())
    namespace = runpy.run_path(str(script))
    target = namespace["checkout"](packed, IDENTIFIER, manifest["source_hash"], tmp_path / "bootstrapped")
    assert (target / "run.sh").read_bytes() == b"#!/bin/sh\necho hello\n"
