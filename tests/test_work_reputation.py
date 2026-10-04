"""Authenticated completion credit, immutable evidence, encrypted reuse and source-risk separation."""

import concurrent.futures
from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.memory import vault
from ml_stack.reputation.sealed import SealedGraph
from ml_stack.reputation.store import Ledger
from ml_stack.reputation.work import WorkLedger
from ml_stack.workspace import tokens
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.work_reputation import scope, standings, verify


@pytest.fixture
def work(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    kit.parent = kit.agent('lead')
    delegated = kit.ws.delegate(kit.parent, 'scout')
    kit.agent_id = delegated['id']
    kit.child = tokens.read_file(Path(delegated['token_file']))
    kit.task = kit.ws.send(kit.parent, kit.agent_id, 'task', 'Inspect the simulator.')
    kit.done = kit.ws.send(kit.child, 'lead', 'answer', 'Validation passed.', reply_to=kit.task['seq'])
    kit.evidence = {'task': kit.task['seq'], 'completion': kit.done['seq'],
                    'checks': [{'name': 'native regression suite', 'passed': True}],
                    'artifacts': {'result.json': 'a' * 64}}
    keys = vault.PassphraseKeys(lambda _: 'test work reputation passphrase')
    sealed = SealedGraph(tmp_path / 'reputation.enc', keys=keys)
    kit.ledger = WorkLedger(sealed)
    yield kit
    sealed.close()


def test_verified_task_credits_once_and_reopens_with_evidence(work):
    first = verify(work.ws, work.parent, work.agent_id, work.evidence, ledger=work.ledger)
    repeated = verify(work.ws, work.parent, work.agent_id, work.evidence, ledger=work.ledger)
    assert first['credited'] and not repeated['credited']
    assert first['task_hash'] == work.ws.bus.get(work.task['seq'])['hash']
    assert first['completion_hash'] == work.ws.bus.get(work.done['seq'])['hash']
    assert standings(work.ws, work.child, ledger=work.ledger)['own']['score'] == 1
    sealed = work.ledger.sealed
    assert b'lead/scout' not in sealed.path.read_bytes() and b'result.json' not in sealed.path.read_bytes()
    sealed.close()
    reopened = WorkLedger(SealedGraph(sealed.path, keys=sealed.keys))
    try:
        result = standings(work.ws, work.child, ledger=reopened)
        assert result['own']['score'] == 1 and result['team'][0]['evidence'][0]['artifacts'] == {'result.json': 'a' * 64}
    finally:
        reopened.sealed.close()


@pytest.mark.redteam
@pytest.mark.parametrize('actor', ['self', 'unrelated', 'missing'])
def test_self_unrelated_and_missing_tokens_cannot_award_credit(work, actor):
    token = work.child if actor == 'self' else work.agent('other') if actor == 'unrelated' else ''
    with pytest.raises(Denied):
        verify(work.ws, token, work.agent_id, work.evidence, ledger=work.ledger)
    assert work.ledger.standings(scope(work.ws)) == []


@pytest.mark.redteam
@pytest.mark.parametrize('change', [
    {'checks': [{'name': 'tests', 'passed': False}]},
    {'checks': [{'name': 'tests', 'passed': 1}]},
    {'artifacts': {'output': 'not-a-hash'}},
    {'task': True}, {'completion': 999999}, {'agent': 'forged'},
])
def test_invalid_or_failed_evidence_never_changes_standing(work, change):
    with pytest.raises((ValueError, Denied)):
        verify(work.ws, work.parent, work.agent_id, {**work.evidence, **change}, ledger=work.ledger)
    assert work.ledger.standings(scope(work.ws)) == []


@pytest.mark.redteam
def test_completion_from_another_agent_or_thread_is_refused(work):
    other = work.agent('other')
    forged = work.ws.send(other, 'lead', 'handoff', 'I did it.', reply_to=work.task['seq'])
    with pytest.raises(Denied):
        verify(work.ws, work.parent, work.agent_id, {**work.evidence, 'completion': forged['seq']}, ledger=work.ledger)
    unrelated = work.ws.send(work.child, 'lead', 'handoff', 'Another task.')
    with pytest.raises(Denied):
        verify(work.ws, work.parent, work.agent_id, {**work.evidence, 'completion': unrelated['seq']}, ledger=work.ledger)
    assert work.ledger.standings(scope(work.ws)) == []


def test_human_can_verify_and_source_risk_is_separate(work):
    source = Ledger(work.ledger.sealed.path, keys=work.ledger.sealed.keys, flush_s=0)
    try:
        before = source.observe('host', 'bad.example', 'scan_hit')
        verify(work.ws, work.owner, work.agent_id, work.evidence, ledger=work.ledger)
        after = source.standing('host', 'bad.example')
        assert before.state == after.state == 'bad' and before.clean == after.clean == 0
        assert work.ledger.standings(scope(work.ws))[0]['score'] == 1
    finally:
        source.close()


def test_concurrent_parent_verification_credits_once(work):
    def credit(_):
        sealed = SealedGraph(work.ledger.sealed.path, keys=work.ledger.sealed.keys)
        try:
            return verify(work.ws, work.parent, work.agent_id, work.evidence, ledger=WorkLedger(sealed))['credited']
        finally:
            sealed.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(credit, range(2)))
    assert sorted(results) == [False, True]
    assert work.ledger.standings(scope(work.ws))[0]['score'] == 1


@pytest.mark.redteam
def test_person_reputation_route_is_read_only_and_session_guarded(work, tmp_path, monkeypatch):
    from test_fleet_ui import Serving

    from ml_stack.fleet import routes
    from ml_stack.workspace import work_reputation

    tokens.store(work.base, tokens.OWNER_FILE, work.owner)
    monkeypatch.setattr(work_reputation, 'WorkLedger', lambda: work.ledger)
    server = Serving(tmp_path)
    try:
        assert server.call('/ui/work-reputation/standings', ui_header=False)[0] == 403
        monkeypatch.setattr(routes, 'in_cluster', lambda _: True)
        assert server.call('/ui/work-reputation/standings')[0] == 401
        cookie = server.ui.sessions.cookie_header(server.ui.sessions.open('person'))
        assert server.call('/ui/work-reputation/standings', method='POST', body={}, cookie=cookie)[0] == 405
        assert server.call('/ui/work-reputation/standings?offset=-1', cookie=cookie)[0] == 400
        assert server.call('/ui/work-reputation/standings?agent=../../owner', cookie=cookie)[0] == 400
        verify(work.ws, work.parent, work.agent_id, work.evidence, ledger=work.ledger)
        code, result, _ = server.call('/ui/work-reputation/standings', cookie=cookie)
        assert code == 200 and result['team'][0]['score'] == 1
        assert work.child not in str(result) and work.owner not in str(result)
    finally:
        server.close()


@pytest.mark.slow
def test_history_shows_verified_score_and_expandable_parent_evidence(work, tmp_path, monkeypatch, playwright):
    from playwright.sync_api import expect
    from test_fleet_ui import Serving

    from ml_stack.activity import writer
    from ml_stack.activity.log import ActivityLog
    from ml_stack.activity.schema import build
    from ml_stack.workspace import work_reputation

    tokens.store(work.base, tokens.OWNER_FILE, work.owner)
    monkeypatch.setattr(work_reputation, 'WorkLedger', lambda: work.ledger)
    log = ActivityLog(tmp_path / 'activity', key=lambda: bytes(range(32)))
    log.add(build('agent.task', ts=work.ws.clock(), actor=work.agent_id, session='native-session',
                  subject='Simulator check', outcome='completed'))
    monkeypatch.setattr(writer, 'log', lambda: log)
    verify(work.ws, work.parent, work.agent_id, work.evidence, ledger=work.ledger)
    server = Serving(tmp_path)
    server.ui.settings.setup_done = True
    try:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page(viewport={'width': 390, 'height': 844})
            page.goto(f'http://127.0.0.1:{server.port}/ui/#history')
            viewer = page.locator('history-view')
            viewer.get_by_text('1 verified completion credits', exact=True).click()
            viewer.get_by_text(f"Task {work.task['seq']} · verified by lead", exact=True).click()
            expect(viewer.get_by_text('native regression suite', exact=False)).to_be_visible()
            expect(viewer.get_by_text('result.json', exact=False)).to_be_visible()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path='/private/tmp/ml-stack-work-reputation-history.png', full_page=True)
    finally:
        server.close()


@pytest.mark.redteam
def test_reputation_read_requires_valid_token_and_read_capability(work):
    with pytest.raises(Denied):
        standings(work.ws, '', ledger=work.ledger)
    send_only = work.ws.delegate(work.parent, 'sender', can=('send',))
    token = tokens.read_file(Path(send_only['token_file']))
    with pytest.raises(Denied):
        standings(work.ws, token, ledger=work.ledger)
    result = standings(work.ws, work.child, ledger=work.ledger)
    assert result['own']['score'] == 0
    assert any(item['agent'] == 'lead' and item['score'] == 0 for item in result['team'])


@pytest.mark.redteam
def test_local_and_mcp_tools_read_reputation_and_task_frame_contains_team_awareness(work, monkeypatch):
    from ml_stack.workspace import localloop, localtools, tools, work_reputation
    from ml_stack.workspace.identity import TOKEN_ENV

    verify(work.ws, work.parent, work.agent_id, work.evidence, ledger=work.ledger)
    monkeypatch.setattr(work_reputation, 'WorkLedger', lambda: work.ledger)
    monkeypatch.setenv(TOKEN_ENV, work.child)
    state = localtools.TaskState()
    extension = localtools.workspace_extension(work.ws, work.child, work.agent_id, state, lambda _: True)
    functions = {schema['function']['name']: call for schema, call in extension.tools()}
    result = functions['workspace_reputation']()
    assert result['own']['score'] == 1 and 'workspace_reputation' in extension.reads
    assert tools.workspace_reputation()['own']['score'] == 1
    assert not any('verify' in name or 'award' in name for name in functions)
    frame = localloop._frame({'seq': work.task['seq'], 'from': 'lead', 'text': 'Inspect the simulator.'},
                            'parent', work_reputation.brief(work.ws, work.child))
    assert 'Your verified completions: 1' in frame and 'lead/scout: 1 verified tasks' in frame
    assert 'no authority' in frame


def test_evidence_pages_keep_total_score_and_expose_earlier_records(work):
    first = verify(work.ws, work.parent, work.agent_id, work.evidence, ledger=work.ledger)
    for task in range(100, 121):
        work.ledger.record({**{key: value for key, value in first.items() if key not in ('id', 'credited')},
                            'task': task, 'verified_at': float(task)})
    first_page = standings(work.ws, work.child, agent=work.agent_id, ledger=work.ledger)['team'][0]
    next_page = standings(work.ws, work.child, agent=work.agent_id, offset=20, ledger=work.ledger)['team'][0]
    assert first_page['score'] == next_page['score'] == 22
    assert len(first_page['evidence']) == 20 and first_page['evidence_held'] == 2
    assert len(next_page['evidence']) == 2 and next_page['evidence_held'] == 0
    assert not {item['id'] for item in first_page['evidence']} & {item['id'] for item in next_page['evidence']}
