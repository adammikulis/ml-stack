"""Hostile replies and redirects at the cluster workspace and WSL bridge boundaries."""

import io
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from poolhouse.fleet import project_client, wsl_network
from poolhouse.http import Sealed, ServerError
from poolhouse.workspace import remote, tokens
from poolhouse.workspace.identity import Denied


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "load_cluster_key", lambda _path: bytes(range(32)))
    monkeypatch.setattr(remote, "memberships", lambda _path: [])
    tokens.prepare(tmp_path)
    monkeypatch.setattr(remote.home, "state", lambda *_args: tmp_path)
    return remote.RemoteWorkspace("http://127.0.0.1:8770", "a" * 32)


@pytest.mark.parametrize("target", ["https://8.8.8.8/steal", "http://127.0.0.1:8771/steal",
                                    "http://127.0.0.1:8770/other"])
def test_project_board_redirect_cannot_carry_capabilities(workspace, monkeypatch, target):
    attempted = []

    @contextmanager
    def redirect(_url, **kwargs):
        kwargs["guard"](target)
        attempted.append(target)
        yield None

    monkeypatch.setattr(remote, "open_stream", redirect)
    with pytest.raises(Denied, match="cannot redirect"):
        workspace._request("board", {"agent_token": "private-capability"})
    assert not attempted


@pytest.mark.parametrize("raw", [b"x" * (512 * 1024 + 29), b"not JSON", b"[]", b"null"],
                         ids=["oversized", "malformed", "array", "null"])
def test_project_board_replies_are_bounded_json_objects(workspace, monkeypatch, raw):
    read_sizes = []
    response = io.BytesIO(raw)

    def read(size):
        read_sizes.append(size)
        return response.read(size)

    @contextmanager
    def reply(_url, **_kwargs):
        yield SimpleNamespace(read=read, status=200, headers={remote.sealing.HEADER: "2"},
                              sealed=SimpleNamespace(open=lambda _status, _headers, body: body))

    monkeypatch.setattr(remote, "open_stream", reply)
    with pytest.raises(ValueError):
        workspace._request("board", {"agent_token": "private-capability"})
    assert read_sizes == [512 * 1024 + 29]


def test_project_board_slow_response_has_finite_timeout_and_no_retry(workspace, monkeypatch):
    timeouts = []

    @contextmanager
    def slow(_url, **kwargs):
        timeouts.append(kwargs["timeout"])
        raise TimeoutError("host stopped responding")
        yield

    monkeypatch.setattr(remote, "open_stream", slow)
    with pytest.raises(TimeoutError):
        workspace._request("board", {"agent_token": "private-capability"})
    assert timeouts == [30]


@pytest.mark.parametrize("raw", [b"x" * (wsl_network.LIMIT + 1), b"not JSON\n", b"[]\n", b"null\n", b""],
                         ids=["oversized", "malformed", "array", "null", "closed"])
def test_wsl_control_replies_reject_oversized_and_malformed_frames(raw):
    with pytest.raises((OSError, ValueError)):
        wsl_network._read(io.BytesIO(raw))


@pytest.mark.parametrize("target", ["https://8.8.8.8/steal", "http://127.0.0.1:8771/steal",
                                    "http://127.0.0.1:8770/other"])
def test_project_source_redirect_stays_on_exact_authority(monkeypatch, target):
    followed = []

    @contextmanager
    def redirect(_url, **kwargs):
        kwargs["guard"](target)
        followed.append(target)
        yield None

    monkeypatch.setattr(project_client, "open_stream", redirect)
    with pytest.raises(project_client.ProjectError, match="cannot redirect"):
        project_client.read(SimpleNamespace(base_url="http://127.0.0.1:8770", token="secret", timeout=30), "/workspace/v1/projects", 100)
    assert not followed


@pytest.mark.parametrize("length", [63, 64, 65])
def test_project_source_sealed_wire_and_plaintext_boundary(monkeypatch, length):
    key, nonce, limit = bytes(range(32)), "request-nonce", 64
    raw = remote.sealing.seal(key, b"x" * length, remote.sealing.response_data(nonce, 200))
    sizes = []

    @contextmanager
    def reply(_url, **_kwargs):
        def read(size):
            sizes.append(size)
            return raw[:size]
        yield SimpleNamespace(read=read, status=200, headers={remote.sealing.HEADER: "2"}, sealed=Sealed(key, nonce))

    monkeypatch.setattr(project_client, "open_stream", reply)
    peer = SimpleNamespace(base_url="http://127.0.0.1:8770", token="secret", timeout=30)
    if length > limit:
        with pytest.raises(project_client.ProjectError, match="size limit"):
            project_client.read(peer, "/workspace/v1/projects", limit)
    else:
        assert project_client.read(peer, "/workspace/v1/projects", limit) == b"x" * length
    assert sizes == [limit + remote.sealing.NONCE_BYTES + 16 + 1]


@pytest.mark.parametrize("headers,raw", [({}, b"not authenticated"), ({remote.sealing.HEADER: "2"}, b"forged seal")])
def test_project_source_requires_authenticated_sealed_response(monkeypatch, headers, raw):
    @contextmanager
    def reply(_url, **_kwargs):
        yield SimpleNamespace(read=lambda _size: raw, status=200, headers=headers,
                              sealed=Sealed(bytes(range(32)), "request-nonce"))

    monkeypatch.setattr(project_client, "open_stream", reply)
    with pytest.raises((project_client.ProjectError, ServerError)):
        project_client.read(SimpleNamespace(base_url="http://127.0.0.1:8770", token="secret", timeout=30), "/workspace/v1/projects", 100)


@pytest.mark.parametrize("override", [{"source_hash": "../../escape"}, {"archive_sha256": "wrong"},
    {"id": "../../etc"}, {"name": []}, {"source_machine": {}}, {"board_host": []},
    {"board_host": "http://8.8.8.8"}, {"board_host": "http://user:secret@127.0.0.1"},
    {"files": True}, {"files": 10001}, {"size_bytes": -1}, {"size_bytes": "128"}])
def test_project_catalogue_schema_is_checked_before_snapshot_url(monkeypatch, override):
    project = {"id": "a" * 32, "name": "App", "source_machine": "source-a",
               "board_host": "", "source_hash": "b" * 64,
               "archive_sha256": "c" * 64, "size_bytes": 10, "files": 1, **override}
    monkeypatch.setattr(project_client, "read", lambda *_args, **_kwargs: json.dumps({"machine": "source-a", "projects": [project]}).encode())
    with pytest.raises((project_client.ProjectError, OSError)):
        project_client.catalogue(SimpleNamespace())


@pytest.mark.parametrize("projects", [[None], [1], [{}], ["project"], "projects"])
def test_project_catalogue_rejects_invalid_members_without_type_crash(monkeypatch, projects):
    monkeypatch.setattr(project_client, "read", lambda *_args, **_kwargs: json.dumps({"machine": "source-a", "projects": projects}).encode())
    with pytest.raises(project_client.ProjectError):
        project_client.catalogue(SimpleNamespace())
