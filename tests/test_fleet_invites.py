"""Single-use owner invitations and their TLS route guards."""
import base64
import hashlib
import hmac
import io
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from poolhouse.fleet import routes, tls
from poolhouse.fleet.discovery import Membership
from poolhouse.fleet.invite_routes import public, ui_route
from poolhouse.fleet.invites import Invitations, decode, proof
from poolhouse.fleet.ui import UI


@pytest.fixture
def invitation(tmp_path):
    members = [Membership("lab", base64.urlsafe_b64encode(b"x" * 32).rstrip(b"="))]
    clock = [1000]
    enrolled = []
    store = Invitations(lambda: members, lambda: ("https://192.168.2.59:8770", "a" * 64),
                        enrol=lambda *row: enrolled.append(row))
    store.enrolled = enrolled
    store.clock = lambda: clock[0]
    minted = store.mint("lab")
    data = json.loads(decode(minted["invite"].split("data=")[1]))
    fields = {"id": data["id"], "kind": "computer", "platform": "computer", "device_name": "recipient",
              "public_key": tls.identity(tmp_path / "recipient", "recipient").beacon}
    return store, data, fields, members, clock


def claim(store, data, fields):
    fields = dict(fields)
    fields.update(store.exchange("challenge", fields))
    fields["proof"] = proof(decode(data["secret"]), fields, data["fingerprint"])
    return fields


def test_atomic_single_use_and_no_key_in_invite(invitation):
    store, data, fields, members, _ = invitation
    assert "key" not in data
    fields = claim(store, data, fields)
    def redeem():
        try:
            return store.exchange("redeem", fields)
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(lambda _: redeem(), range(8)))
    success = [answer for answer in answers if answer]
    assert len(success) == 1
    grant = json.loads(decode(success[0]["grant_data"]))
    assert grant["key"].encode() == members[0].key
    assert store.enrolled == [("lab", fields["public_key"], "recipient")]


def test_a_computer_without_a_certificate_cannot_redeem_and_a_phone_may_not_send_one(invitation):
    store, _data, fields, _, _ = invitation
    with pytest.raises(ValueError, match="certificate"):
        store.exchange("challenge", {**fields, "public_key": ""})
    with pytest.raises(ValueError, match="certificate"):
        store.exchange("challenge", {**fields, "public_key": "AAAA"})
    assert store.enrolled == []


def test_expiry_revocation_and_membership_binding(invitation):
    store, data, fields, members, clock = invitation
    fields = claim(store, data, fields)
    members[:] = [Membership("lab", base64.urlsafe_b64encode(b"y" * 32).rstrip(b"="))]
    with pytest.raises(ValueError, match="membership changed"):
        store.exchange("redeem", fields)
    store.revoke(data["id"])
    with pytest.raises(ValueError, match="revoked"):
        store.exchange("challenge", fields)
    members[:] = [Membership("lab", base64.urlsafe_b64encode(b"x" * 32).rstrip(b"="))]
    minted = store.mint("lab")
    clock[0] += 601
    with pytest.raises(ValueError, match="expired"):
        store.exchange("challenge", {**fields, "id": minted["id"]})


def test_wrong_proof_and_changed_fields_cannot_redeem(invitation):
    store, data, fields, _, _ = invitation
    fields = claim(store, data, fields)
    with pytest.raises(ValueError, match="changed"):
        store.exchange("redeem", {**fields, "device_name": "other"})
    fields = claim(store, data, fields)
    with pytest.raises(ValueError, match="proof"):
        store.exchange("redeem", {**fields, "proof": "0" * 64})


def test_expired_challenge_and_certificate_rotation(invitation):
    store, data, fields, _, clock = invitation
    fields = claim(store, data, fields)
    clock[0] += 61
    with pytest.raises(ValueError, match="challenge expired"):
        store.exchange("redeem", fields)
    store.origin = lambda: ("https://192.168.2.59:8770", "b" * 64)
    with pytest.raises(ValueError, match="certificate changed"):
        store.exchange("challenge", fields)


@pytest.mark.parametrize("origin", ["http://192.168.2.59:8770", "https://127.0.0.1:8770", "https://8.8.8.8:8770"])
def test_mint_requires_reachable_private_tls(origin):
    store = Invitations(lambda: [Membership("lab", base64.urlsafe_b64encode(b"x" * 32).rstrip(b"="))],
                        lambda: (origin, "a" * 64))
    with pytest.raises(ValueError):
        store.mint("lab")


def test_public_exchange_refuses_plain_http():
    replies = []
    handler = SimpleNamespace(path="/join/invite/challenge", connection=object(),
                              _send=lambda code, body: replies.append((code, body)))
    assert public(object(), handler, b"{}")
    assert replies[0][0] == 403


def test_ui_mint_requires_local_authenticated_owner():
    replies = []
    route = SimpleNamespace(path="/ui/fleet/invites", client_ip="192.168.2.9", host_header="localhost",
                            ui=SimpleNamespace(host_ok=lambda host: True),
                            send=lambda code, body: replies.append((code, body)))
    assert ui_route(route)
    assert replies[0][0] == 403
    route.client_ip = "127.0.0.1"
    route.ui = SimpleNamespace(authed=lambda cookie: False, host_ok=lambda host: True)
    route.cookie = ""
    assert ui_route(route)
    assert replies[-1][0] == 401


@pytest.mark.parametrize("field,value", [("id", []), ("challenge", {}), ("device_name", "bad\nname"), ("proof", 42)])
def test_malformed_fields_are_refused(invitation, field, value):
    store, _, fields, _, _ = invitation
    with pytest.raises(ValueError, match="invalid invitation field"):
        store.exchange("challenge", {**fields, field: value})


def test_rate_limits_and_challenge_bound(invitation):
    store, _, fields, _, clock = invitation
    for _ in range(20):
        store.permit("192.168.2.7")
    with pytest.raises(ValueError, match="rate limit"):
        store.permit("192.168.2.7")
    clock[0] += 61
    store.permit("192.168.2.7")
    for _ in range(10):
        store.exchange("challenge", fields)
    with pytest.raises(ValueError, match="challenge limit"):
        store.exchange("challenge", fields)


def test_grant_proof_covers_raw_payload(invitation):
    store, data, fields, _, _ = invitation
    fields = claim(store, data, fields)
    answer = store.exchange("redeem", fields)
    raw = decode(answer["grant_data"])
    message = ("poolhouse-invite-grant/v1\n" + fields["challenge"] + "\n").encode() + raw
    assert hmac.compare_digest(answer["proof"], hmac.new(decode(data["secret"]), message, hashlib.sha256).hexdigest())



def test_owner_mint_returns_qr_and_revoke_invalidates(invitation):
    store, _, _, _, _ = invitation
    replies = []
    ui = SimpleNamespace(host_ok=lambda host: True, authed=lambda cookie: True, invitations=store)
    route = SimpleNamespace(path="/ui/fleet/invites", client_ip="127.0.0.1", host_header="localhost",
        cookie="owner", ui=ui, method="POST", header=lambda name, default="": "15",
        body=lambda: {"group": "lab", "kind": "computer"},
        send=lambda code, body, extra=None: replies.append((code, body, extra)))
    assert ui_route(route)
    code, body, extra = replies[-1]
    assert code == 200
    assert body["qr"].startswith("data:image/svg+xml;base64,")
    assert b"<svg" in base64.b64decode(body["qr"].split(",")[1])
    assert extra["Cache-Control"] == "no-store"
    ident = body["id"]
    route.method = "DELETE"
    route.body = lambda: {"id": ident}
    assert ui_route(route)
    assert ident not in store.rows


def test_owner_mint_refuses_hostname_rebinding():
    replies = []
    route = SimpleNamespace(path="/ui/fleet/invites", client_ip="127.0.0.1", host_header="attacker.invalid",
        ui=SimpleNamespace(host_ok=lambda host: False), send=lambda code, body: replies.append(code))
    assert ui_route(route)
    assert replies == [403]


@pytest.mark.parametrize("case", [
    (True, False, True, "localhost", 401),
    (True, True, False, "localhost", 403),
    (False, False, True, "attacker.invalid", 403),
    (False, False, True, "localhost", 200),
    (True, True, True, "localhost", 200),
])
def test_join_invitation_passes_owner_and_setup_router_guards(
        tmp_path, monkeypatch, case):
    joined, authenticated, header, host, expected = case
    ui = UI(name="recipient", cluster_key_path=tmp_path / "cluster.key")
    calls, replies = [], []
    ui.join_guard = nullcontext
    ui.join_invitation = lambda code: calls.append(code) or {"ok": True, "group": "lab"}
    ui.authed = lambda cookie: authenticated
    monkeypatch.setattr(routes, "in_cluster", lambda path: joined)
    body = json.dumps({"invite": "test-invitation"}).encode()
    headers = {"Host": host, "Content-Length": str(len(body))}
    if header:
        headers[routes.UI_HEADER] = "1"
    handler = SimpleNamespace(path="/ui/fleet/join-invite", command="POST", headers=headers,
                              client_address=("127.0.0.1", 1), rfile=io.BytesIO(body))
    route = routes.Router(ui, handler)
    route.send = lambda code, body, extra=None: replies.append((code, body, extra))
    assert route.run()
    assert replies[-1][0] == expected
    assert calls == (["test-invitation"] if expected == 200 else [])
    if expected == 200:
        assert replies[-1][2]["Cache-Control"] == "no-store"
        assert "HttpOnly" in replies[-1][2]["Set-Cookie"]
