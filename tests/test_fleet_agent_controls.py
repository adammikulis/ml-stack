"""Fleet mounts the maintained person agent-start flow and native saved settings."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_fleet_ui import Serving
from test_workspace_board_ui import browser
from test_workspace_local_agent import PICK, sleeper
from workspace_kit import Kit, clean_env

from ml_stack.fleet import extension_routes
from ml_stack.workspace import (
    agent_routes,
    device_agent,
    localagent,
    localmodel,
    localprofile,
    localstart,
    tokens,
)
from ml_stack.workspace.device_accounts import account_for

__all__ = ['browser']


@pytest.fixture
def served(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    tokens.store(kit.base, tokens.OWNER_FILE, kit.owner)
    monkeypatch.setattr(device_agent, 'device_id', lambda: 'abcdef0123456789')
    monkeypatch.setattr(extension_routes, 'entry_points', lambda **kw: [
        SimpleNamespace(name='agents', load=lambda: agent_routes.route)])
    monkeypatch.setattr(localmodel, 'choose', lambda *args, **kw: PICK)
    monkeypatch.setattr(localprofile, 'admit', lambda *args, **kw: ('', ''))
    children = []
    def spawn(*args, **kw):
        job = sleeper(*args, **kw)
        children.append(job.child)
        return job
    monkeypatch.setattr(localstart.jobs, 'detach', spawn)
    device_agent.enroll(kit.ws, kit.owner)
    parent = localstart.launch_parent(kit.ws, str(tmp_path))
    child = kit.ws.delegate(parent, 'worker')
    localagent.save(kit.ws, localagent.Agent('local-worker', PICK.ref, identity=child['id'],
                                           model_name=PICK.name, project=str(tmp_path),
                                           profile='coding', harness='claude', ctx=262144))
    server = Serving(tmp_path, secure=False)
    server.ui.settings.setup_done = True
    yield server, kit, children, child['id']
    server.close()
    for child_process in children:
        child_process.terminate()
        child_process.wait(timeout=5)


@pytest.mark.redteam
def test_actual_fleet_controls_require_auth_origin_header_and_typed_person_payload(served):
    server, kit, children, worker = served
    origin = {'Origin': f'http://127.0.0.1:{server.port}'}
    body = {'name':'local-worker','model':PICK.ref,'profile':'coding','harness':'claude',
            'ctx':'256K','project':str(server.ui.settings_path.parent)}
    assert server.call('/ui/agents/start', method='POST', body=body, ui_header=False, headers=origin)[0] == 403
    assert server.call('/ui/agents/start', method='POST', body=body, headers={'Origin':'http://foreign.invalid'})[0] == 403
    assert server.call('/ui/agents/start', method='PUT', body=body, headers=origin)[0] == 405
    assert server.call('/ui/agents/start', method='POST', body={**body,'identity':worker}, headers=origin)[0] == 400
    assert not children
    status_path=localagent.folder(kit.ws)/"local-worker.status.json"
    status_path.write_text(json.dumps({"state":"stopped","beat":1,"tasks":99,"lease":{"id":"old"}}))
    code, result, _ = server.call('/ui/agents/start', method='POST', body=body, headers=origin)
    assert code == 200, result
    assert len(children) == 1
    assert localagent.load(kit.ws,'local-worker').identity == worker
    assert account_for(kit.ws,worker)['enrolled_by'] == kit.ws.auth(kit.owner).id
    listed = server.call('/ui/agents/list')[1]
    assert listed['agents'][0]['state'] == 'starting'
    assert listed['agents'][0]['tasks'] == 0
    assert listed['agents'][0]['lease'] is False
    agent=localagent.load(kit.ws,'local-worker')
    status_path.write_text(json.dumps({'state':'idle','beat':agent.started+1}))
    assert server.call('/ui/agents/list')[1]['agents'][0]['state'] == 'idle'
    assert listed['saved'][0]['model'] == PICK.ref
    assert listed['saved'][0]['project'] == str(server.ui.settings_path.parent)
    assert 'token' not in listed['saved'][0]


@pytest.mark.slow
def test_person_board_controls_reuse_saved_settings_and_enroll_through_maintained_start(served, browser):
    server, kit, children, worker = served
    page = browser.new_page()
    try:
        page.goto(f'http://127.0.0.1:{server.port}/ui/#board')
        page.locator('#board-agents > summary').click()
        controls = page.locator('ml-agents')
        controls.locator('input#name').wait_for()
        page.wait_for_function("document.querySelector('ml-agents')?.shadowRoot?.querySelector('#name')?.value === 'local-worker'")
        assert controls.locator('input#model').input_value() == PICK.ref
        assert controls.locator('input#project').input_value() == str(server.ui.settings_path.parent)
        assert controls.get_by_text('Selected model:',exact=False).count() == 1
        assert controls.get_by_text('auto uses',exact=False).count() == 0
        assert not controls.locator('details').get_attribute('open')
        assert controls.locator('input#context').input_value() == '262144'
        controls.get_by_role('button',name='Start a local agent',exact=True).click()
        page.wait_for_function("document.querySelector('ml-agents')?.shadowRoot?.querySelector('.note[role=status]')?.textContent.includes('started.')")
        assert len(children) == 1
        assert localagent.load(kit.ws,'local-worker').identity == worker
        assert account_for(kit.ws,worker) is not None
    finally:
        page.close()


def test_auto_model_hints_are_resolved_for_each_work_profile(served, monkeypatch):
    server, _, _, _=served
    monkeypatch.setattr(localmodel,'choose',lambda *args,**kw:replace(PICK,
        name='Qwen3.8-27B' if kw['selection'].coding else 'Qwen3.6-35B-A3B'))
    code,result,_=server.call('/ui/agents/model')
    assert code == 200
    assert result['profiles']['coding']['name'] == 'Qwen3.8-27B'
    assert result['profiles']['chat']['name'] == 'Qwen3.6-35B-A3B'
