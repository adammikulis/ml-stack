"""Paired-device egress and daemon project authority attacks."""

import base64
import threading
from types import SimpleNamespace

import pytest

from ml_stack.fleet import device_auth
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.remote import Peer
from ml_stack.http import Server, ServerError
from ml_stack.workspace import coordinator_client
from ml_stack.workspace.identity import Denied


@pytest.mark.redteam
@pytest.mark.parametrize("change", ["revoked", "certificate", "unpaired", "missing", "ambiguous"])
def test_coordinator_peer_rejects_untrusted_pairing_before_transport(tmp_path, monkeypatch, change):
    device = SimpleNamespace(fingerprint="a" * 64, status="revoked" if change == "revoked" else "active")
    row = {"source": "discovery" if change == "unpaired" else "pairing",
           "fingerprint": device.fingerprint,
           "certificate": "wrong" if change == "certificate" else "trusted",
           "device_secret": "" if change == "missing" else base64.urlsafe_b64encode(b"x" * 32).decode()}
    monkeypatch.setattr(coordinator_client.home, "state", lambda name: tmp_path)
    monkeypatch.setattr(coordinator_client, "Devices", lambda path: SimpleNamespace(all=lambda: [device]))
    monkeypatch.setattr(coordinator_client, "PeerBook", lambda path: SimpleNamespace(
        rows=lambda: [row, dict(row)] if change == "ambiguous" else [row]))
    monkeypatch.setattr(coordinator_client.http, "pin", lambda *args: pytest.fail("untrusted pin installed"))
    monkeypatch.setattr(coordinator_client, "Peer", lambda *args, **kwargs: pytest.fail("untrusted transport opened"))
    with pytest.raises(Denied, match="unique active paired"):
        coordinator_client._device_peer({"endpoint": "https://coordinator.example:8770", "cert": "trusted"})


@pytest.mark.redteam
@pytest.mark.parametrize("credential", ["missing", "cluster", "revoked", "unsealed", "malformed"])
def test_daemon_project_dispatch_refuses_invalid_device_requests(enrolled, tmp_path, credential):
    _kit, device, projects, document = enrolled
    calls = []
    host = SimpleNamespace(answer=lambda *args, **kwargs: calls.append((args, kwargs)))
    runner = JobRunner(tmp_path / "runner")
    server = Server(("127.0.0.1", 0), make_handler(Daemon(
        runner, tmp_path / "files", "cluster-token", projects=projects,
        workspaces=host, devices=lambda: [device])))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    route = "/workspace/v1/projects/" + document["project"]["key"] + "/board"
    try:
        secret = device_auth.secret(device)
        if credential == "revoked":
            device.status = "revoked"
        if credential == "unsealed":
            import http.client
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                connection.request("POST", route, body=b"{}", headers={"Authorization": "Bearer " + secret})
                response = connection.getresponse()
                assert response.status in {400, 401, 403}
                response.read()
            finally:
                connection.close()
            assert calls == []
            return
        if credential == "missing":
            from ml_stack.http import request_bytes
            headers = {"Authorization": "Bearer " + secret} if credential == "unsealed" else {}
            with pytest.raises(ServerError) as refused:
                request_bytes(origin + route, method="POST", data=b"{}", headers=headers)
        else:
            peer = Peer(origin, "cluster-token" if credential == "cluster" else secret)
            with pytest.raises(ServerError) as refused:
                peer._request("POST", route, data=b"[" if credential == "malformed" else b"{}")
        assert refused.value.status in {400, 401, 403}
        assert calls == []
    finally:
        server.shutdown()
        server.server_close()
        runner.shutdown()
        worker.join(timeout=2)
