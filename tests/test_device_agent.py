"""Device account enrollment binds authentic local seats independently of models."""
from dataclasses import replace

import pytest
from workspace_kit import Kit

from ml_stack.workspace import device_agent, localagent, tokens
from ml_stack.workspace.device_accounts import account_for
from ml_stack.workspace.identity import AGENT, Denied


def test_owner_enrollment_survives_worker_revocation_and_model_change(tmp_path, monkeypatch):
    kit = Kit(tmp_path / "ws")
    monkeypatch.setattr(device_agent, "device_id", lambda: "1234567890abcdef")
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
    monkeypatch.setattr(device_agent, "device_id", lambda: "1234567890abcdef")
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
    monkeypatch.setattr(device_agent, "device_id", lambda: "1234567890abcdef")
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


def test_saved_delegated_coding_seat_survives_model_restart(tmp_path, monkeypatch):
    from test_workspace_local_agent import PICK, sleeper

    from ml_stack.workspace import localprofile, localstart
    kit = Kit(tmp_path / "ws")
    parent = kit.agent("parent")
    child = kit.ws.delegate(parent, "worker")
    identity = child["id"]
    localagent.save(kit.ws, localagent.Agent("named-worker", "old-model", identity=identity, profile="coding"))
    monkeypatch.setattr(localprofile, "admit", lambda *args: ("", ""))
    monkeypatch.setattr(localstart.jobs, "detach", sleeper)
    monkeypatch.setattr(device_agent, "device_id", lambda: "1234567890abcdef")
    got = localstart.start(kit.ws, localstart.Ask(name="named-worker", profile="coding"),
                           pick=PICK, person_token=kit.owner)
    try:
        assert localagent.load(kit.ws, got.name).identity == identity
        assert kit.ws.registry.role_of(got.name) == ""
        assert account_for(kit.ws, identity)["base_id"].startswith("local-device-")
        assert kit.ws.registry.info(identity)["model"] == PICK.name
    finally:
        localstart.stop(kit.ws, got.name, wait_s=5)


def owned_launcher(kit, name='launcher'):
    credential = kit.ws.registry._add('local-account', name, AGENT, 0)
    tokens.store(kit.base, name, credential)
    return credential


def delegated_worker(kit, parent, name='worker', project=''):
    identity = kit.ws.delegate(parent, name)['id']
    localagent.save(kit.ws, localagent.Agent(name, 'example-model', identity=identity, project=project))
    return identity


def test_owned_launcher_binds_only_its_delegated_worker_without_person_token(tmp_path, monkeypatch):
    kit = Kit(tmp_path / 'ws')
    monkeypatch.setattr(device_agent, 'device_id', lambda: '1234567890abcdef')
    parent = owned_launcher(kit)
    identity = delegated_worker(kit, parent)
    read = tokens.read_file

    def private_agent_only(path):
        assert path.name != tokens.OWNER_FILE
        return read(path)

    monkeypatch.setattr(tokens, 'read_file', private_agent_only)
    account = device_agent.bind_owned_worker(kit.ws, parent, 'worker')
    assert account['device_id'] == '1234567890abcdef'
    assert {'launcher', identity, account['base_id']} <= set(account['members'])
    assert kit.ws.registry.info(account['base_id'])['role'] == AGENT
    assert device_agent.bind_owned_worker(kit.ws, parent, 'worker') == account


def test_owned_launcher_reuses_existing_person_enrolled_device(tmp_path, monkeypatch):
    kit = Kit(tmp_path / 'ws')
    monkeypatch.setattr(device_agent, 'device_id', lambda: '1234567890abcdef')
    account = device_agent.enroll(kit.ws, kit.owner)
    parent = owned_launcher(kit)
    identity = delegated_worker(kit, parent)
    bound = device_agent.bind_owned_worker(kit.ws, parent, 'worker')
    assert bound['base_id'] == account['base_id']
    assert identity in bound['members']


@pytest.mark.parametrize('parent_kind', ['foreign', 'network', 'wrong-parent', 'revoked'])
def test_untrusted_parent_cannot_assign_local_device_membership(tmp_path, monkeypatch, parent_kind):
    kit = Kit(tmp_path / 'ws')
    monkeypatch.setattr(device_agent, 'device_id', lambda: '1234567890abcdef')
    parent = owned_launcher(kit)
    identity = delegated_worker(kit, parent)
    if parent_kind == 'foreign':
        parent = kit.agent('foreign')
        tokens.store(kit.base, 'foreign', parent)
    elif parent_kind == 'network':
        agents = kit.ws.registry._load()
        agents['launcher']['session_device'] = 'a' * 64
        kit.ws.registry._save(agents)
    elif parent_kind == 'wrong-parent':
        parent = owned_launcher(kit, 'other-launcher')
    else:
        kit.ws.revoke(kit.owner, 'launcher')
    with pytest.raises(Denied):
        device_agent.bind_owned_worker(kit.ws, parent, 'worker')
    assert account_for(kit.ws, identity) is None


def test_revoked_child_or_device_account_cannot_gain_membership(tmp_path, monkeypatch):
    kit = Kit(tmp_path / 'ws')
    monkeypatch.setattr(device_agent, 'device_id', lambda: '1234567890abcdef')
    parent = owned_launcher(kit)
    identity = delegated_worker(kit, parent)
    kit.ws.revoke(parent, identity)
    with pytest.raises(Denied):
        device_agent.bind_owned_worker(kit.ws, parent, 'worker')
    live_identity = delegated_worker(kit, parent, 'live-worker')
    account = device_agent.bind_owned_worker(kit.ws, parent, 'live-worker')
    kit.ws.revoke(kit.owner, account['base_id'])
    with pytest.raises(Denied, match='device account was revoked'):
        device_agent.bind_owned_worker(kit.ws, parent, 'live-worker')
    assert account_for(kit.ws, live_identity)['base_id'] == account['base_id']


def test_worker_project_and_device_must_match_trusted_local_parent(tmp_path, monkeypatch):
    kit = Kit(tmp_path / 'ws')
    monkeypatch.setattr(device_agent, 'device_id', lambda: '1234567890abcdef')
    parent = owned_launcher(kit)
    project = tmp_path / 'other-project'
    project.mkdir()
    identity = delegated_worker(kit, parent, project=str(project))
    with pytest.raises(Denied, match='project differs'):
        device_agent.bind_owned_worker(kit.ws, parent, 'worker')
    assert account_for(kit.ws, identity) is None
    delegated_worker(kit, parent, 'local-worker')
    device_agent.bind_owned_worker(kit.ws, parent, 'local-worker')
    monkeypatch.setattr(device_agent, 'device_id', lambda: 'fedcba0987654321')
    with pytest.raises(Denied, match='another installed device'):
        device_agent.bind_owned_worker(kit.ws, parent, 'local-worker')


def test_interrupted_owned_device_account_creation_recovers_same_identity(tmp_path, monkeypatch):
    kit = Kit(tmp_path / 'ws')
    monkeypatch.setattr(device_agent, 'device_id', lambda: '1234567890abcdef')
    parent = owned_launcher(kit)
    identity = delegated_worker(kit, parent)
    with monkeypatch.context() as interrupted:
        interrupted.setattr(kit.ws.registry, '_add', lambda *args: (_ for _ in ()).throw(OSError('interrupted')))
        with pytest.raises(OSError, match='interrupted'):
            device_agent.bind_owned_worker(kit.ws, parent, 'worker')
    assert account_for(kit.ws, identity) is None
    account = device_agent.bind_owned_worker(kit.ws, parent, 'worker')
    assert account['base_id'] == 'local-device-1234567890abcdef'
    assert kit.ws.registry.info(account['base_id'])['role'] == AGENT


@pytest.mark.parametrize('limit,value', [('mints_per_identity', 1), ('agents_live', 2)])
def test_owned_device_account_mint_limits_precede_membership_writes(tmp_path, monkeypatch, limit, value):
    kit = Kit(tmp_path / 'ws')
    monkeypatch.setattr(device_agent, 'device_id', lambda: '1234567890abcdef')
    parent = owned_launcher(kit)
    identity = delegated_worker(kit, parent)
    kit.limits(**{limit: value})
    before = kit.ws.registry.ids()
    with pytest.raises(Denied, match='the limits are'):
        device_agent.bind_owned_worker(kit.ws, parent, 'worker')
    assert kit.ws.registry.ids() == before
    assert account_for(kit.ws, identity) is None
    with device_agent._graph(kit.ws) as graph:
        assert graph.nodes('device-account') == []


def test_existing_owned_account_membership_does_not_mint_at_capacity(tmp_path, monkeypatch):
    kit = Kit(tmp_path / 'ws')
    monkeypatch.setattr(device_agent, 'device_id', lambda: '1234567890abcdef')
    parent = owned_launcher(kit)
    delegated_worker(kit, parent)
    account = device_agent.bind_owned_worker(kit.ws, parent, 'worker')
    kit.limits(mints_per_identity=2, agents_live=3)
    assert device_agent.bind_owned_worker(kit.ws, parent, 'worker') == account
