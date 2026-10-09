"""Actor and exact execution views share independently reviewed outcomes."""

from dataclasses import replace

import pytest
from taskboard_kit import accepted, board as board_fixture
from test_task_credit import ledger as ledger_fixture, reviewed

from poolhouse.graph.store import GraphStore
from poolhouse.workspace import localagent, resource_allocations, task_credit, work_reputation
from poolhouse.workspace.task_provenance import snapshot

board = board_fixture
ledger = ledger_fixture


def test_same_actor_switches_model_without_rewriting_receipts_or_duplicating_credit(board, ledger):
    first_proposal, _ = reviewed(board, {**accepted(), 'verified_usage': {
        'source': 'Independent task review', 'tokens_in': None, 'tokens_out': 24, 'wall_seconds': 2.0}})
    first = task_credit.verify_task(board.ws, board.parent, board.task['id'], ledger=ledger)
    configured = localagent.load(board.ws, 'native-worker')
    localagent.save(board.ws, replace(configured, model='gemma', harness='claude', ctx=65536,
                                      effort='high', max_output_tokens=4096))
    board.status['servers'][0]['model'] = 'gemma'
    board.status['runtime'] = {'state': 'observed', 'source_commit': 'a' * 40, 'protocol': 1,
                              'implementation_sha256': 'b' * 64,
                              'environment': {'package_version': '0.2.2', 'python': '3.13.5'}}
    task = board.board.create(board.parent, {**board.spec, 'source_key': 'second-model'})
    allocation = resource_allocations.assign(board.ws, board.parent, board.worker_id, task['id'], 'native-grant')
    board.board.claim(board.child, task['id'], allocation['allocation_id'])
    proposed = board.board.submit(board.child, task['id'], {
        'artifacts': {'second.json': 'c' * 64}, 'checks': [{'name': 'Worker claims tests', 'passed': True}],
        'provenance': {'model': 'forged-model', 'runtime': 'forged-runtime'}})
    board.board.review(board.parent, task['id'], accepted())
    with GraphStore(board.base / 'coordination.db') as graph:
        node = next(node for node in graph.nodes('task') if node['id'] == task['id'])
        graph.upsert_node({**node, 'attrs': {**node['attrs'], 'state': 'completed'}})
    second = task_credit.verify_task(board.ws, board.parent, task['id'], ledger=ledger)
    again = task_credit.verify_task(board.ws, board.parent, task['id'], ledger=ledger)
    result = work_reputation.standings(board.ws, board.child, ledger=ledger)
    actor = next(row for row in result['dimensions']['agents'] if row['id'] == board.worker_id)
    assert actor['aliases'] == ['native-worker'] and actor['economy']['balance'] == 20
    assert actor['verified_tasks'] == 2 and actor['work_reputation']['reliability_samples'] == 2
    assert actor['economy']['usage']['tokens_out'] == 24
    models = result['dimensions']['models']
    assert len(models) == 2 and sum(row['economy']['balance'] for row in models) == 20
    assert sum(row['economy']['usage']['tokens_out'] or 0 for row in models) == 24
    assert sum(row['economy']['usage']['tokens_out'] or 0 for row in result['team']) == 24
    assert {row['evidence'][0]['provenance']['model'] for row in models} == {'qwen', 'gemma'}
    assert first['provenance'] == first_proposal['provenance']
    assert second['provenance'] == proposed['provenance']
    assert second['provenance']['runtime'] == 'claude' and second['provenance']['alias'] == 'native-worker'
    assert second['provenance']['execution_config']['max_output_tokens'] == 4096
    assert second['provenance']['execution_config']['effort'] == 'high'
    assert second['provenance']['broker_runtime']['source'] == 'broker-observed'
    assert second['provenance']['artifact'] == {'id': None, 'source': 'unknown'}
    assert proposed['claimed_provenance']['model'] == 'forged-model'
    assert not again['credited'] and again['id'] == second['id']
    assert sum(row['economy']['balance'] for row in result['team']) == 20
    assert len(ledger.sealed.graph().nodes('work_award')) == 2
    with GraphStore(board.base / 'coordination.db') as graph:
        assert len(graph.nodes('work-credit-reference')) == 2


@pytest.mark.parametrize('outcome,samples', [('rejected', 1), ('blocked_infrastructure', 0)])
def test_unaccepted_dimensions_keep_existing_review_policy_without_awards(board, ledger, outcome, samples):
    reviewed(board, {'accepted': False, 'outcome': outcome, 'reason': 'Independent outcome.',
                     'checks': [{'name': 'Replay passes', 'passed': False}]})
    task_credit.verify_task(board.ws, board.parent, board.task['id'], ledger=ledger)
    result = work_reputation.standings(board.ws, board.child, ledger=ledger)['dimensions']
    for kind in ('agents', 'models'):
        assert len(result[kind]) == 1
        row = result[kind][0]
        assert row['economy']['balance'] == 0 and row['verified_tasks'] == 0
        assert row['work_reputation']['reliability_samples'] == samples
        assert row['work_reputation']['outcomes'][outcome] == 1
    assert not ledger.sealed.graph().nodes('work_award')


def test_unknown_artifact_and_runtime_are_not_promoted_from_a_local_filename():
    resource = {'model': '/device/models/same.gguf', 'allocation_id': 'allocation:known'}
    result = snapshot('authenticated-worker', resource)
    assert result['model'] == resource['model'] and result['actor'] == 'authenticated-worker'
    assert result['artifact']['source'] == 'unknown' and result['artifact']['id'] is None
    assert result['broker_runtime']['source'] == 'unknown'
    assert result['alias_source'] == 'unknown' and result['alias'] is None


@pytest.mark.redteam
@pytest.mark.parametrize('field,value', [('agent', 'foreign-actor'), ('workspace', 'foreign-workspace'),
                                         ('proposal_id', 'foreign-proposal'), ('completion', 'foreign-review')])
def test_award_reference_cannot_credit_a_different_actor_or_review(board, ledger, field, value):
    reviewed(board)
    task_credit.verify_task(board.ws, board.parent, board.task['id'], ledger=ledger)
    with GraphStore(board.base / 'coordination.db') as graph:
        node = graph.nodes('work-credit-reference')[0]
        graph.upsert_node({**node, 'attrs': {**node['attrs'], field: value}})
    result = work_reputation.standings(board.ws, board.child, ledger=ledger)['dimensions']
    for kind in ('agents', 'models'):
        assert result[kind][0]['verified_tasks'] == 0
        assert result[kind][0]['economy']['balance'] == 0
    assert len(ledger.sealed.graph().nodes('work_award')) == 1


@pytest.mark.redteam
def test_review_cannot_attribute_a_hashed_proposal_to_another_worker(board, ledger):
    from poolhouse.workspace.task_schema import fingerprint

    reviewed(board)
    task_credit.verify_task(board.ws, board.parent, board.task['id'], ledger=ledger)
    with GraphStore(board.base / 'coordination.db') as graph:
        node = graph.nodes('review')[0]
        changed = {**node['attrs'], 'worker': 'foreign-worker'}
        changed['review_hash'] = fingerprint({key: value for key, value in changed.items() if key != 'review_hash'})
        graph.upsert_node({**node, 'attrs': changed})
    result = work_reputation.standings(board.ws, board.child, ledger=ledger)['dimensions']
    assert not result['agents'] and not result['models']


@pytest.mark.slow
def test_history_filters_actor_exact_runtime_and_family_over_the_same_review(board, ledger, tmp_path, monkeypatch, playwright):
    from playwright.sync_api import expect
    from test_fleet_ui import Serving

    from poolhouse.workspace import tokens

    reviewed(board)
    task_credit.verify_task(board.ws, board.parent, board.task['id'], ledger=ledger)
    tokens.store(board.base, tokens.OWNER_FILE, board.owner)
    monkeypatch.setattr(work_reputation, 'Workspace', lambda: board.ws)
    monkeypatch.setattr(work_reputation, 'WorkLedger', lambda: ledger)
    server = Serving(tmp_path)
    server.ui.settings.setup_done = True
    try:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page(viewport={'width': 390, 'height': 844})
            page.goto(f'http://127.0.0.1:{server.port}/ui/#history')
            view = page.locator('history-view')
            view.locator('details.activity-reputation > summary').click()
            credits = view.locator('#history-credits')
            expect(credits.locator('summary').filter(has_text='Qwen · 10 credits')).to_be_visible()
            view.locator('#history-dimension').select_option('agent')
            card = credits.locator('details.history-action').filter(has_text='native-worker · 10 credits')
            card.locator(':scope > summary').click()
            expect(card.get_by_text('These are views of the same contributions; credits are awarded once.', exact=True)).to_be_visible()
            expect(card.get_by_text(f'Account identity: {board.worker_id}', exact=True)).to_be_visible()
            view.locator('#history-dimension').select_option('model')
            card = credits.locator('details.history-action').filter(has_text='qwen · poolhouse-agent · 10 credits')
            card.locator(':scope > summary').click()
            expect(card.get_by_text('Model source: verified-serving-resource · Runtime source: registered-worker-configuration · Artifact: unknown', exact=True)).to_be_visible()
            card.locator('summary').filter(has_text=f'Task {board.task["id"]} · verified by lead').click()
            expect(card.get_by_text('replay.json', exact=False)).to_be_visible()
            assert view.locator('section.workspace').evaluate(
                '(element) => element.scrollWidth <= element.clientWidth && element.clientWidth <= innerWidth')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    finally:
        server.close()
