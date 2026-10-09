"""Android enrollment, scoped sessions and streaming route boundaries."""
import base64
import io
import json
import ssl
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ml_stack.fleet import companion_routes
from ml_stack.fleet.discovery import Membership, derive_token
from ml_stack.fleet.invite_routes import ui_route
from ml_stack.fleet.invites import Invitations, decode, proof

CLUSTER_KEY = base64.urlsafe_b64encode(b"x" * 32).rstrip(b"=")


@pytest.fixture
def enrolled():
    members = [Membership("lab", CLUSTER_KEY)]
    store = Invitations(lambda: members, lambda: ("https://192.168.2.59:8770", "a" * 64))
    clock = [1000]
    store.clock = lambda: clock[0]
    data = json.loads(decode(store.mint("lab", "android")["invite"].split("data=")[1]))
    fields = {"id": data["id"], "kind": "android", "platform": "android",
              "device_name": "test phone", "public_key": ""}
    challenge = store.exchange("challenge", fields)
    fields.update(challenge)
    fields["proof"] = proof(decode(data["secret"]), fields, data["fingerprint"])
    grant = json.loads(decode(store.exchange("redeem", fields)["grant_data"]))
    return store, grant, members, clock


def test_android_grant_contains_only_scoped_session(enrolled):
    store, grant, _, _ = enrolled
    assert set(grant) == {"kind", "device_id", "token", "expires", "capabilities", "endpoint", "fingerprint", "group"}
    assert grant["capabilities"] == ["fleet.status", "chat"]
    assert len(grant["device_id"]) == 32 and len(decode(grant["token"])) == 32
    assert store.authorize(grant["token"])["group"] == "lab"
    assert CLUSTER_KEY not in json.dumps(grant).encode()
    assert grant["token"] not in repr(store.devices)
    assert "token" not in store.active_devices()[0]


@pytest.mark.parametrize("change", ["revoke", "expire", "membership", "certificate", "endpoint"])
def test_session_invalidates_with_authority(enrolled, change):
    store, grant, members, clock = enrolled
    if change == "revoke":
        store.revoke(grant["device_id"])
    if change == "expire":
        clock[0] = grant["expires"]
    if change == "membership":
        members.clear()
    if change == "certificate":
        store.origin = lambda: (grant["endpoint"], "b" * 64)
    if change == "endpoint":
        store.origin = lambda: ("https://192.168.2.60:8770", "a" * 64)
    with pytest.raises(ValueError):
        store.authorize(grant["token"])


def test_invitation_kind_is_bound_and_restart_invalidates(enrolled):
    store, grant, _, _ = enrolled
    data = json.loads(decode(store.mint("lab")["invite"].split("data=")[1]))
    with pytest.raises(ValueError, match="kind changed"):
        store.exchange("challenge", {"id": data["id"], "kind": "android", "platform": "android", "device_name": "phone"})
    replacement = Invitations(store.members, store.origin)
    with pytest.raises(ValueError):
        replacement.authorize(grant["token"])


def handler(grant, path="/companion/v1/status", method="GET"):
    replies = []
    connection = ssl.SSLSocket.__new__(ssl.SSLSocket)
    return SimpleNamespace(path=path, command=method, connection=connection,
        client_address=("192.168.2.8", 4), headers={"Authorization": "Bearer " + grant["token"]},
        _send=lambda code, body, **kw: replies.append((code, body)), replies=replies,
        send_response=lambda code: replies.append((code, None)), send_header=lambda *args: None,
        end_headers=lambda: None, wfile=io.BytesIO())


def ui(store):
    live = SimpleNamespace(models=["model-q"], aliases=["C:/secret/models/model-q"], port=8771)
    return SimpleNamespace(invitations=store, name="test computer", serving=SimpleNamespace(live=lambda: [live]))


def test_status_omits_computer_capabilities_paths_and_secrets(enrolled):
    store, grant, _, _ = enrolled
    request = handler(grant)
    assert companion_routes.answer(ui(store), request)
    assert request.replies == [(200, {"name": "test computer", "models": ["model-q"]})]
    request.path = "/jobs"
    assert not companion_routes.answer(ui(store), request)


@pytest.mark.parametrize("change", ["plain", "public", "missing", "cluster", "revoked"])
def test_route_authentication_refuses_inappropriate_authority(enrolled, change):
    store, grant, _, _ = enrolled
    request = handler(grant)
    if change == "plain":
        request.connection = object()
    if change == "public":
        request.client_address = ("8.8.8.8", 4)
    if change == "missing":
        request.headers = {}
    if change == "cluster":
        request.headers = {"Authorization": "Bearer " + derive_token(CLUSTER_KEY)}
    if change == "revoked":
        store.revoke(grant["device_id"])
    assert companion_routes.answer(ui(store), request)
    assert request.replies[0][0] == 403


def test_owner_controls_android_enrollment_and_revocation(enrolled):
    store, grant, _, _ = enrolled
    replies = []
    owner = ui(store)
    owner.host_ok = lambda host: True
    owner.authed = lambda cookie: True
    owner.sessions = SimpleNamespace(get=lambda cookie: SimpleNamespace(who="passphrase"))
    route = SimpleNamespace(path="/ui/fleet/android-devices", client_ip="127.0.0.1", host_header="localhost",
        ui=owner, cookie="owner", method="GET", header=lambda *args: "100", body=lambda: {"device_id": grant["device_id"]},
        send=lambda code, body, extra=None: replies.append((code, body)))
    assert ui_route(route) and replies[-1][1]["devices"][0]["device_id"] == grant["device_id"]
    route.method = "DELETE"
    owner.authed = lambda cookie: False
    assert ui_route(route) and replies[-1][0] == 401
    assert store.authorize(grant["token"])
    owner.authed = lambda cookie: True
    assert ui_route(route)
    with pytest.raises(ValueError):
        store.authorize(grant["token"])


def test_machine_token_browser_session_cannot_enroll_android(enrolled):
    store, _, _, _ = enrolled
    replies = []
    owner = ui(store)
    owner.host_ok = lambda host: True
    owner.authed = lambda cookie: True
    owner.sessions = SimpleNamespace(get=lambda cookie: SimpleNamespace(who="token"))
    route = SimpleNamespace(path="/ui/fleet/invites", client_ip="127.0.0.1", host_header="localhost",
        ui=owner, cookie="machine", method="POST", header=lambda *args: "100",
        body=lambda: {"group": "lab", "kind": "android"},
        send=lambda code, body, extra=None: replies.append((code, body)))
    before = len(store.rows)
    assert ui_route(route) and replies[-1][0] == 403 and len(store.rows) == before


def test_stream_delivers_actual_deltas_then_completion(enrolled, monkeypatch):
    from ml_stack.fleet import sdk_chat
    store, grant, _, _ = enrolled
    request = handler(grant, "/companion/v1/chat", "POST")
    observed = []
    def stream(target, payload, **kwargs):
        yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
        observed.append(request.wfile.getvalue())
        yield b'data: {"choices":[{"delta":{"content":" second"}}]}\n\n'
    monkeypatch.setattr(sdk_chat, "stream", stream)
    monkeypatch.setattr(companion_routes, "turn", lambda *a, **kw: nullcontext())
    raw = json.dumps({"model": "model-q", "messages": [{"role": "user", "content": "question"}]}).encode()
    assert companion_routes.answer(ui(store), request, raw)
    assert b'first' in observed[0] and b'second' not in observed[0]
    assert b'"done":true' in request.wfile.getvalue()


def test_revocation_closes_stream_before_more_output(enrolled, monkeypatch):
    from ml_stack.fleet import sdk_chat
    store, grant, _, _ = enrolled
    request = handler(grant, "/companion/v1/chat", "POST")
    closed = []
    def stream(target, payload, **kwargs):
        try:
            yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
            store.revoke(grant["device_id"])
            assert kwargs["control"].cancelled.is_set()
            yield b'data: {"choices":[{"delta":{"content":"forbidden"}}]}\n\n'
        finally:
            closed.append(True)
    monkeypatch.setattr(sdk_chat, "stream", stream)
    monkeypatch.setattr(companion_routes, "turn", lambda *a, **kw: nullcontext())
    raw = json.dumps({"model": "model-q", "messages": [{"role": "user", "content": "question"}]}).encode()
    companion_routes.answer(ui(store), request, raw)
    assert closed and b"forbidden" not in request.wfile.getvalue()
    assert b'"error"' in request.wfile.getvalue()


@pytest.mark.parametrize("body", [{"model": "http://attacker", "messages": []},
    {"model": "model-q", "messages": [{"role": "system", "content": "commands"}]},
    {"model": "model-q", "messages": [{"role": "user", "content": "x"}], "tools": []}])
def test_chat_rejects_extra_authority_and_unavailable_targets(enrolled, body):
    store, grant, _, _ = enrolled
    request = handler(grant, "/companion/v1/chat", "POST")
    companion_routes.answer(ui(store), request, json.dumps(body).encode())
    assert request.replies[0][0] == 400


def test_real_tls_dispatcher_refuses_phone_credentials_on_computer_routes(enrolled, tmp_path):
    import http.client

    from ml_stack.fleet import tls
    from ml_stack.fleet.api import Daemon, make_handler
    from ml_stack.fleet.framing import LimitedServer
    from ml_stack.fleet.jobs import JobRunner
    store, grant, _, _ = enrolled
    root = tmp_path / "daemon"
    root.mkdir()
    runner = JobRunner(root)
    interface = ui(store)
    identity = tls.identity(tmp_path / "tls", "test computer")
    server = LimitedServer(("127.0.0.1", 0), make_handler(Daemon(
        runner, root, "computer-token", ui=interface)), tls=tls.server_context(identity))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        cases = [("GET", "/companion/v1/status", None, True, 200),
                 ("GET", "/jobs", None, True, 401),
                 ("GET", "/workspace/v1/projects", None, True, 401),
                 ("GET", "/companion/v1/unknown", None, True, 404),
                 ("GET", "/companion/v1/chat", None, True, 404),
                 ("POST", "/companion/v1/status", b"{}", True, 404),
                 ("POST", "/companion/v1/chat", b"{", True, 400),
                 ("POST", "/companion/v1/chat", b"x" * 65537, True, 413),
                 ("POST", "/companion/v1/chat", b"{}", False, 403),
                 ("GET", "/companion/v1/status", None, False, 403)]
        for method, path, body, authorized, expected in cases:
            connection = http.client.HTTPSConnection("127.0.0.1", server.server_port,
                                                     context=tls.pinned_context(identity.beacon), timeout=5)
            try:
                headers = {"Authorization": "Bearer " + grant["token"]} if authorized else {}
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                assert response.status == expected
                response.read()
            finally:
                connection.close()
    finally:
        runner.shutdown()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


@pytest.mark.redteam
@pytest.mark.parametrize("change", ["remote", "host", "machine", "oversize", "method"])
def test_android_device_controls_reject_foreign_authority_without_revocation(enrolled, change):
    store, grant, _, _ = enrolled
    replies = []
    owner = ui(store)
    owner.host_ok = lambda host: change != "host"
    owner.authed = lambda cookie: True
    owner.sessions = SimpleNamespace(get=lambda cookie: SimpleNamespace(
        who="token" if change == "machine" else "passphrase"))
    route = SimpleNamespace(path="/ui/fleet/android-devices",
        client_ip="192.168.2.8" if change == "remote" else "127.0.0.1",
        host_header="localhost", ui=owner, cookie="owner",
        method="POST" if change == "method" else "DELETE",
        header=lambda *args: "8193" if change == "oversize" else "100",
        body=lambda: {"device_id": grant["device_id"]},
        send=lambda code, body, extra=None: replies.append((code, body)))
    assert ui_route(route)
    assert replies[-1][0] == (400 if change == "oversize" else 405 if change == "method" else 403)
    assert store.authorize(grant["token"])


@pytest.mark.parametrize("address,accepted", [("100.101.1.2", True), ("fd7a:115c:a1e0::1", True), ("8.8.8.8", False)])
def test_tailnet_address_enrolls_and_connects_but_public_does_not(enrolled, address, accepted):
    store, grant, _, _ = enrolled
    host = f"[{address}]" if ":" in address else address
    tailnet = Invitations(lambda: [Membership("lab", CLUSTER_KEY)], lambda: (f"https://{host}:8770", "a" * 64))
    if accepted:
        assert tailnet.mint("lab", "android")["invite"]
    else:
        with pytest.raises(ValueError, match="reachable LAN TLS listener"):
            tailnet.mint("lab", "android")
    request = handler(grant)
    request.client_address = (address, 4)
    assert companion_routes.answer(ui(store), request)
    assert (request.replies[0][0] == 200) is accepted
