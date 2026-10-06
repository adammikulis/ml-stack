"""Dev project enrollment transport and immutable cluster scope."""

import hashlib
import json
import ssl
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ml_stack.fleet import api, project_enrollment as admission, projects
from ml_stack.fleet.discovery import derive_token


@pytest.fixture
def scope(monkeypatch):
    member = SimpleNamespace(group="development", key=b"development-key", mode="dev")
    monkeypatch.setattr(admission, "memberships", lambda path: [member])
    opening = (None, SimpleNamespace(secret=derive_token(member.key)), True, {})
    body = {"cluster": member.group, "cluster_id": hashlib.sha256(member.key).hexdigest()}
    return member, opening, body


def test_dev_requires_verified_transport_and_exact_cluster(scope):
    member, opening, body = scope
    tls = Mock(spec=ssl.SSLSocket)
    assert admission.admit(tls, opening, None, body)
    assert not admission.admit(object(), opening, None, body)
    assert not admission.admit(tls, None, None, body)
    assert not admission.admit(tls, (*opening[:2], False, {}), None, body)
    assert not admission.admit(tls, opening, None, {**body, "cluster": "production"})
    assert not admission.admit(tls, opening, None, {**body, "cluster_id": "0" * 64})
    member.mode = "prod"
    assert not admission.admit(tls, opening, None, body)
    assert not admission.visible(tls, opening, None)


@pytest.mark.redteam
def test_prod_secret_does_not_select_another_dev_membership(scope, monkeypatch):
    dev, opening, body = scope
    prod = SimpleNamespace(group="production", key=b"production-key", mode="prod")
    monkeypatch.setattr(admission, "memberships", lambda path: [dev, prod])
    opening[1].secret = derive_token(prod.key)
    assert admission.authenticated_cluster(opening, None)[0] == "production"
    assert not admission.admit(Mock(spec=ssl.SSLSocket), opening, None, body)


@pytest.mark.redteam
def test_shared_key_alias_cannot_grant_dev_authority(scope, monkeypatch):
    dev, opening, body = scope
    alias = SimpleNamespace(group="production", key=dev.key, mode="prod")
    monkeypatch.setattr(admission, "memberships", lambda path: [dev, alias])
    assert admission.authenticated_cluster(opening, None) == ("", "")
    assert not admission.admit(Mock(spec=ssl.SSLSocket), opening, None, body)


@pytest.mark.redteam
def test_secondary_dev_does_not_override_primary_prod(scope, monkeypatch):
    dev, opening, body = scope
    prod = SimpleNamespace(group="production", key=b"production-key", mode="prod")
    monkeypatch.setattr(admission, "memberships", lambda path: [prod, dev])
    assert not admission.visible(Mock(spec=ssl.SSLSocket), opening, None)
    assert not admission.admit(Mock(spec=ssl.SSLSocket), opening, None, body)


@pytest.mark.redteam
def test_native_registration_requires_local_active_dev_proof(scope):
    member, opening, _ = scope
    assert admission.local(opening, None, "127.0.0.1")
    assert admission.local(opening, None, "::1")
    assert not admission.local(opening, None, "192.168.2.8")
    assert not admission.local(None, None, "127.0.0.1")
    member.mode = "prod"
    assert not admission.local(opening, None, "127.0.0.1")


@pytest.mark.redteam
@pytest.mark.parametrize("attack", ["remote", "unsealed", "prod", "wrong-project"])
def test_local_project_route_refuses_unauthorized_registration(scope, tmp_path, monkeypatch, attack):
    member, opening, _ = scope
    calls = []

    def register(root, project_id):
        calls.append((root, project_id))
        raise ValueError("project identity mismatch")

    monkeypatch.setattr(api.sentinel, "armed", lambda: False)
    daemon = api.Daemon(None, tmp_path, "fixture-secret", projects=SimpleNamespace(register=register))
    handler = object.__new__(api.make_handler(daemon))
    handler.path = "/workspace/v1/local-project"
    handler.client_address = ("192.168.2.8" if attack == "remote" else "127.0.0.1", 1)
    if attack == "prod":
        member.mode = "prod"
    handler._sealing = lambda: (*opening[:2], False, {}) if attack == "unsealed" else opening
    replies = []
    handler._send = lambda status, body: replies.append((status, body))
    assert projects.register(handler, daemon.projects, daemon.cluster_key_path,
                             json.dumps({"root": str(tmp_path), "project_id": "a" * 32}).encode())
    assert replies[0][0] == (400 if attack == "wrong-project" else 403)
    assert len(calls) == (1 if attack == "wrong-project" else 0)
    assert str(tmp_path) not in json.dumps(replies)
