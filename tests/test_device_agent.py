"""Device account enrollment binds authentic local seats independently of models."""
from dataclasses import replace

import pytest
from workspace_kit import Kit

from ml_stack.workspace import device_agent, localagent, tokens
from ml_stack.workspace.device_accounts import account_for
from ml_stack.workspace.identity import Denied


def test_owner_enrollment_survives_worker_revocation_and_model_change(tmp_path, monkeypatch):
    kit = Kit(tmp_path / "ws")
    monkeypatch.setattr(device_agent.home, "machine_id", lambda: "1234567890abcdef")
    first = device_agent.enroll(kit.ws, kit.owner)
    assert device_agent.enroll(kit.ws, kit.owner) == first
    token = kit.agent("worker")
    tokens.store(kit.base, "worker", token)
    worker = localagent.Agent("worker", "model-a", identity="worker")
    localagent.save(kit.ws, worker)
    bound = device_agent.bind_worker(kit.ws, kit.owner, "worker")
    localagent.save(kit.ws, replace(worker, model="model-b"))
    assert device_agent.bind_worker(kit.ws, kit.owner, "worker") == bound
    kit.ws.registry.revoke(kit.ws.auth(kit.owner), "worker")
    assert account_for(kit.reopen(), "worker")["base_id"] == first["base_id"]
    assert account_for(kit.ws, "unknown") is None


def test_agent_cannot_enroll_or_self_assign_device_credit(tmp_path, monkeypatch):
    kit = Kit(tmp_path / "ws")
    monkeypatch.setattr(device_agent.home, "machine_id", lambda: "1234567890abcdef")
    agent = kit.agent("external")
    with pytest.raises(Denied, match="only the person"):
        device_agent.enroll(kit.ws, agent)
    with pytest.raises(Denied, match="only the person"):
        device_agent.bind_worker(kit.ws, agent, "external")
    assert account_for(kit.ws, "external") is None


def test_person_start_binds_model_independent_default(tmp_path, monkeypatch):
    from test_workspace_local_agent import PICK, sleeper

    from ml_stack.workspace import localstart
    kit = Kit(tmp_path / "ws")
    monkeypatch.setattr(device_agent.home, "machine_id", lambda: "1234567890abcdef")
    got = localstart.start(kit.ws, localstart.Ask(), pick=PICK, spawn=sleeper, person_token=kit.owner)
    try:
        assert got.name == "local-agent"
        account = account_for(kit.ws, got.name)
        assert account["device_id"] == "1234567890abcdef"
        with pytest.raises(ValueError, match="same name to change models"):
            localstart.start(kit.ws, localstart.Ask(), pick=replace(PICK, ref="other-model"), spawn=sleeper)
    finally:
        localstart.stop(kit.ws, got.name, wait_s=5)
    assert account_for(kit.ws, got.name) == account
