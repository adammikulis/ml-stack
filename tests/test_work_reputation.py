"""Authenticated completion credit, immutable evidence, encrypted reuse and source-risk separation."""

import concurrent.futures
from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.memory import vault
from ml_stack.reputation import economy
from ml_stack.reputation.sealed import SealedGraph
from ml_stack.reputation.store import Ledger
from ml_stack.reputation.work import WorkLedger
from ml_stack.workspace import tokens
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.work_reputation import scope, standings


@pytest.fixture
def work(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    kit.parent = kit.agent('lead')
    delegated = kit.ws.delegate(kit.parent, 'scout')
    kit.agent_id = delegated['id']
    kit.child = tokens.read_file(Path(delegated['token_file']))
    kit.task = {'seq': 171}
    kit.done = {'seq': 172}
    kit.evidence = {'task': kit.task['seq'], 'completion': kit.done['seq'],
                    'checks': [{'name': 'native regression suite', 'passed': True}],
                    'artifacts': {'result.json': 'a' * 64}}
    keys = vault.PassphraseKeys(lambda _: 'test work reputation passphrase')
    sealed = SealedGraph(tmp_path / 'reputation.enc', keys=keys)
    kit.ledger = WorkLedger(sealed)
    yield kit
    sealed.close()



def _historical(work, evidence=None, *, ledger=None):
    """Seed already-verified historical graph evidence without inventing canonical task state."""
    raw = dict(work.evidence if evidence is None else evidence)
    return (ledger or work.ledger).record({**raw, 'award': economy.assessment(raw),
                                         'agent': raw.get('agent', work.agent_id), 'verifier': 'lead',
                                         'workspace': scope(work.ws), 'verified_at': work.ws.clock(),
                                         'task_hash': 'b' * 64, 'completion_hash': 'c' * 64})


def test_verified_task_credits_once_and_reopens_with_evidence(work):
    first = _historical(work)
    repeated = _historical(work)
    assert first['credited'] and not repeated['credited']
    assert first['task_hash'] == 'b' * 64
    assert first['completion_hash'] == 'c' * 64
    assert standings(work.ws, work.child, ledger=work.ledger)['own']['verified_tasks'] == 1
    sealed = work.ledger.sealed
    assert b'lead/scout' not in sealed.path.read_bytes() and b'result.json' not in sealed.path.read_bytes()
    sealed.close()
    reopened = WorkLedger(SealedGraph(sealed.path, keys=sealed.keys))
    try:
        result = standings(work.ws, work.child, ledger=reopened)
        assert result['own']['verified_tasks'] == 1 and result['own']['evidence'][0]['artifacts'] == {'result.json': 'a' * 64}
    finally:
        reopened.sealed.close()








def test_historical_credit_and_source_risk_are_separate(work):
    source = Ledger(work.ledger.sealed.path, keys=work.ledger.sealed.keys, flush_s=0)
    try:
        before = source.observe('host', 'bad.example', 'scan_hit')
        _historical(work)
        after = source.standing('host', 'bad.example')
        assert before.state == after.state == 'bad' and before.clean == after.clean == 0
        assert work.ledger.standings(scope(work.ws))[0]['verified_tasks'] == 1
    finally:
        source.close()


def test_concurrent_historical_record_import_is_idempotent(work):
    def credit(_):
        sealed = SealedGraph(work.ledger.sealed.path, keys=work.ledger.sealed.keys)
        try:
            return _historical(work, ledger=WorkLedger(sealed))['credited']
        finally:
            sealed.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(credit, range(2)))
    assert sorted(results) == [False, True]
    assert work.ledger.standings(scope(work.ws))[0]['verified_tasks'] == 1


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
        _historical(work)
        code, result, _ = server.call('/ui/work-reputation/standings', cookie=cookie)
        assert code == 200
        assert next(item for item in result['team'] if item['agent'] == work.agent_id)['verified_tasks'] == 1
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
    from ml_stack.workspace import localagent, work_reputation

    tokens.store(work.base, tokens.OWNER_FILE, work.owner)
    monkeypatch.setattr(work_reputation, 'WorkLedger', lambda: work.ledger)
    log = ActivityLog(tmp_path / 'activity', key=lambda: bytes(range(32)))
    localagent.save(work.ws, localagent.Agent('local-qwen', 'qwen.gguf', identity=work.agent_id))
    log.add(build('agent.task', ts=work.ws.clock(), actor='local-qwen', session='native-session',
                  subject='Simulator check', outcome='completed'))
    monkeypatch.setattr(writer, 'log', lambda: log)
    _historical(work)
    server = Serving(tmp_path)
    server.ui.settings.setup_done = True
    try:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page(viewport={'width': 390, 'height': 844})
            page.goto(f'http://127.0.0.1:{server.port}/ui/#history')
            viewer = page.locator('history-view')
            viewer.locator('#history-credits > details > summary').filter(has_text=work.agent_id).click()
            viewer.get_by_text(f"Task {work.task['seq']} · verified by lead", exact=True).click()
            expect(viewer.get_by_text('native regression suite', exact=False)).to_be_visible()
            expect(viewer.get_by_text('result.json', exact=False)).to_be_visible()
            expect(viewer.get_by_text(f'Account identity: {work.agent_id}', exact=True)).to_be_visible()
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
    assert result['own']['verified_tasks'] == 0
    assert any(item['agent'] == 'lead' and item['verified_tasks'] == 0 for item in result['team'])


@pytest.mark.redteam
def test_local_and_mcp_tools_read_reputation_and_task_frame_contains_team_awareness(work, monkeypatch):
    from ml_stack.workspace import localloop, localtools, tools, work_reputation
    from ml_stack.workspace.identity import TOKEN_ENV

    _historical(work)
    monkeypatch.setattr(work_reputation, 'WorkLedger', lambda: work.ledger)
    monkeypatch.setenv(TOKEN_ENV, work.child)
    state = localtools.TaskState()
    extension = localtools.workspace_extension(work.ws, work.child, work.agent_id, state, lambda _: True)
    functions = {schema['function']['name']: call for schema, call in extension.tools()}
    result = functions['workspace_reputation']()
    assert result['own']['verified_tasks'] == 1 and 'workspace_reputation' in extension.reads
    assert tools.workspace_reputation()['own']['verified_tasks'] == 1
    assert not any('verify' in name or 'award' in name for name in functions)
    frame = localloop._frame({'seq': work.task['seq'], 'from': 'lead', 'text': 'Inspect the simulator.'},
                            'parent', work_reputation.brief(work.ws, work.child))
    assert 'Your verified completions: 1' in frame and 'lead/scout: 1 verified tasks' in frame
    assert 'no authority' in frame


def test_evidence_pages_keep_total_score_and_expose_earlier_records(work):
    first = _historical(work)
    for task in range(100, 121):
        work.ledger.record({**{key: value for key, value in first.items() if key not in ('id', 'credited')},
                            'task': task, 'verified_at': float(task)})
    first_page = standings(work.ws, work.child, agent=work.agent_id, ledger=work.ledger)['team'][0]
    next_page = standings(work.ws, work.child, agent=work.agent_id, offset=20, ledger=work.ledger)['team'][0]
    assert first_page['verified_tasks'] == next_page['verified_tasks'] == 22
    assert len(first_page['evidence']) == 20 and first_page['evidence_held'] == 2
    assert len(next_page['evidence']) == 2 and next_page['evidence_held'] == 0
    assert not {item['id'] for item in first_page['evidence']} & {item['id'] for item in next_page['evidence']}


def test_free_economy_separates_credits_ratings_and_unknown_usage(work):
    first = _historical(work)
    row = standings(work.ws, work.child, ledger=work.ledger)['own']
    assert row['verified_tasks'] == 1 and row['economy']['balance'] == 10
    assert row['economy']['spent'] == 0 and row['economy']['mode'] == 'free'
    assert row['economy']['usage']['recorded'] is False
    assert row['economy']['usage']['tokens_in'] is None
    assert row['work_reputation']['state'] == 'unrated'
    assert row['pricing']['price'] == 0 and not row['pricing']['active_charging']
    assert first['award'] == {'policy': 'completion-quality-v1', 'currency': 'work-credit',
                              'base': 10, 'quality_bonus': 0, 'total': 10, 'quality': []}


def reviewed_evidence(work):
    return {**work.evidence,
            'quality': [{'kind': 'regression', 'reason': 'Reviewer reproduced the fixed failure.',
                         'checks': ['native regression suite'], 'artifacts': ['result.json']}],
            'review': {'quality': 90, 'reliability': 80, 'reason': 'Independent review passed.'},
            'usage': {'source': 'native harness measurement', 'tokens_in': None,
                      'tokens_out': 42, 'wall_seconds': 2.5},
            'provenance': {'model': 'first.gguf', 'harness': 'codex'}}


def test_quality_bonus_requires_reviewed_evidence_and_does_not_multiply_by_test_count(work):
    evidence = reviewed_evidence(work)
    first = _historical(work, evidence)
    repeated = _historical(work, {**evidence, 'quality': []})
    assert first['credited'] and not repeated['credited']
    row = standings(work.ws, work.child, ledger=work.ledger)['own']
    assert row['economy']['balance'] == 15 and row['economy']['quality_credits'] == 5
    assert row['work_reputation']['quality'] == 63.33
    assert row['work_reputation']['reliability'] == 60
    assert row['work_reputation']['samples'] == 1
    assert row['economy']['usage']['tokens_in'] is None
    assert row['economy']['usage']['tokens_out'] == 42
    assert row['economy']['usage']['wall_seconds'] == 2.5
    assert row['pricing']['baseline_guaranteed'] and row['pricing']['price'] == 0


@pytest.mark.redteam
@pytest.mark.parametrize('change', [
    {'quality': [{'kind': 'regression', 'reason': 'Claimed', 'checks': ['invented'], 'artifacts': ['result.json']}]},
    {'quality': [{'kind': 'impact', 'reason': 'Claimed', 'checks': ['native regression suite'], 'artifacts': ['missing']}]},
    {'review': {'quality': True, 'reliability': 80, 'reason': 'Claimed'}},
    {'review': {'quality': 101, 'reliability': 80, 'reason': 'Claimed'}},
    {'usage': {'source': 'native', 'tokens_in': -1, 'tokens_out': 1, 'wall_seconds': 1}},
    {'usage': {'source': 'native', 'tokens_in': 1, 'tokens_out': 1, 'wall_seconds': float('nan')}},
])
def test_unverified_or_invalid_quality_and_resource_claims_never_credit(work, change):
    with pytest.raises(ValueError):
        _historical(work, {**work.evidence, **change})
    assert standings(work.ws, work.child, ledger=work.ledger)['own']['economy']['balance'] == 0


def test_model_switches_keep_one_authenticated_agent_account(work):
    _historical(work, reviewed_evidence(work))
    task, done = {'seq': 173}, {'seq': 174}
    evidence = {**reviewed_evidence(work), 'task': task['seq'], 'completion': done['seq'],
                'provenance': {'model': 'second.gguf', 'harness': 'claude'}}
    _historical(work, evidence)
    result = standings(work.ws, work.child, ledger=work.ledger)
    row = result['own']
    assert row['agent'] == work.agent_id and row['economy']['balance'] == 30
    assert row['verified_tasks'] == 2 and row['work_reputation']['samples'] == 2
    assert {item['provenance']['model'] for item in row['evidence']} == {'first.gguf', 'second.gguf'}
    assert len([item for item in result['team'] if item['agent'] == work.agent_id]) == 1


def test_two_workers_roll_up_to_one_owner_enrolled_device_across_models(work, monkeypatch):
    from ml_stack.workspace import device_agent, localagent

    monkeypatch.setattr(device_agent, 'device_id', lambda: 'physical-one')
    localagent.save(work.ws, localagent.Agent('worker-one', 'first.gguf', identity=work.agent_id))
    account = device_agent.bind_worker(work.ws, work.owner, 'worker-one')
    _historical(work, reviewed_evidence(work))
    another = work.ws.delegate(work.parent, 'second')
    second = another['id']
    second_token = tokens.read_file(Path(another['token_file']))
    localagent.save(work.ws, localagent.Agent('worker-two', 'second.gguf', identity=second))
    device_agent.bind_worker(work.ws, work.owner, 'worker-two')
    task, done = {'seq': 173}, {'seq': 174}
    _historical(work, {**reviewed_evidence(work), 'agent': second, 'task': task['seq'],
                                        'completion': done['seq'],
                                        'provenance': {'model': 'second.gguf', 'harness': 'claude'}},
           ledger=work.ledger)
    row = standings(work.ws, second_token, ledger=work.ledger)['own']
    assert row['agent'] == account['base_id'] and row['device_id'] == 'physical-one'
    assert row['economy']['balance'] == 30 and row['verified_tasks'] == 2
    assert {work.agent_id, second} <= set(row['members'])
    assert {item['agent'] for item in row['evidence']} == {work.agent_id, second}
    assert {'worker-one', 'worker-two'} <= set(row['aliases'])
    assert standings(work.ws, work.child, ledger=work.ledger)['own']['agent'] == row['agent']


@pytest.mark.redteam
def test_foreign_device_rebinding_and_agent_enrollment_cannot_steal_credits(work, monkeypatch):
    from ml_stack.workspace import device_agent, localagent

    localagent.save(work.ws, localagent.Agent('worker', 'model.gguf', identity=work.agent_id))
    monkeypatch.setattr(device_agent, 'device_id', lambda: 'first-device')
    first = device_agent.bind_worker(work.ws, work.owner, 'worker')
    _historical(work)
    with pytest.raises(Denied):
        device_agent.bind_worker(work.ws, work.child, 'worker')
    monkeypatch.setattr(device_agent, 'device_id', lambda: 'second-device')
    with pytest.raises(Denied):
        device_agent.bind_worker(work.ws, work.owner, 'worker')
    second = device_agent.enroll(work.ws, work.owner)
    rows = standings(work.ws, work.owner, ledger=work.ledger)['team']
    assert next(row for row in rows if row['agent'] == first['base_id'])['economy']['balance'] == 10
    assert next(row for row in rows if row['agent'] == second['base_id'])['economy']['balance'] == 0


def test_awards_ratings_and_usage_are_linked_graph_evidence_and_migration_is_once(work, monkeypatch):
    _historical(work, reviewed_evidence(work))
    graph = work.ledger.sealed.graph()
    assert len(graph.nodes('work_award')) == 1
    assert len(graph.nodes('work_quality_review')) == 1
    assert len(graph.nodes('work_rating')) == 1 and len(graph.nodes('work_usage')) == 1
    assert any(edge['rel'] == 'earned' for edge in graph.edges())
    assert any(edge['rel'] == 'supported_by' for edge in graph.edges())
    assert work.ledger.migrate_credit_awards() == 0
    node = graph.nodes('work_evidence')[0]
    historical = {**node['attrs']}
    historical.pop('award')
    def remove_award(current):
        current.upsert_node({**node, 'attrs': historical})
    work.ledger.sealed.edit(remove_award)
    assert standings(work.ws, work.child, ledger=work.ledger)['own']['economy']['pending_awards'] == 1
    assert work.ledger.migrate_credit_awards() == 1
    assert work.ledger.migrate_credit_awards() == 0
    monkeypatch.setattr('ml_stack.reputation.economy.BASE_CREDITS', 1000)
    assert standings(work.ws, work.child, ledger=work.ledger)['own']['economy']['balance'] == 10


def test_forgetting_source_risk_preserves_work_awards_and_discards_stale_backup(work):
    _historical(work)
    source = Ledger(work.ledger.sealed.path, keys=work.ledger.sealed.keys, flush_s=0)
    try:
        source.observe('host', 'bad.example', 'scan_hit')
        assert source.forget_all() == 1
        assert source.sources() == [] and not source.sealed.prev.exists()
        row = standings(work.ws, work.child, ledger=work.ledger)['own']
        assert row['economy']['balance'] == 10 and row['verified_tasks'] == 1
        assert len(work.ledger.sealed.graph().nodes('work_award')) == 1
    finally:
        source.close()
