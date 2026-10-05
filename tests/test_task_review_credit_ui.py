"""Person review persists independently of encrypted credit availability and retries once."""

from types import SimpleNamespace

import pytest
from taskboard_kit import accepted, board as _board_fixture, proposed
from test_fleet_ui import Serving

from ml_stack.fleet import extension_routes
from ml_stack.memory import vault
from ml_stack.reputation.sealed import SealedGraph
from ml_stack.reputation.work import WorkLedger
from ml_stack.workspace import task_credit, task_routes, tokens, work_reputation

board = _board_fixture
pytestmark = pytest.mark.redteam


def test_person_review_records_credit_after_locked_store_retry_without_losing_review(board, monkeypatch, tmp_path):
    proposed(board)
    tokens.store(board.base, tokens.OWNER_FILE, board.owner)
    monkeypatch.setattr(extension_routes, 'entry_points', lambda **kw: [
        SimpleNamespace(name='tasks', load=lambda: task_routes.route)])
    ledger = WorkLedger(SealedGraph(tmp_path / 'credit.enc', keys=vault.PassphraseKeys(lambda _: 'isolated task key')))
    available = [False]
    original = task_credit.verify_task
    def verify(ws, token, ident):
        if not available[0]:
            raise vault.KeyUnavailable('fake locked store')
        return original(ws, token, ident, ledger=ledger)
    monkeypatch.setattr(task_routes.task_credit, 'verify_task', verify)
    server = Serving(tmp_path)
    def post(body):
        return server.call('/ui/tasks', method='POST', body=body,
                           headers={'Origin': f'http://127.0.0.1:{server.port}'})
    try:
        code, result, _ = post({'action': 'review', 'id': board.task['id'], 'decision': accepted()})
        assert code == 200 and result['credit']['state'] == 'pending'
        saved = board.board.get(board.owner, board.task['id'])
        assert saved['state'] == 'completed' and saved['review']['accepted']
        available[0] = True
        code, result, _ = post({'action': 'credit', 'id': board.task['id']})
        assert code == 200 and result['credit']['credited'] and result['credit']['total'] == 10
        assert post({'action': 'credit', 'id': board.task['id']})[1]['credit']['credited'] is False
        standing = work_reputation.standings(board.ws, board.owner, ledger=ledger)
        assert sum(row['economy']['balance'] for row in standing['team']) == 10
        assert post({'action': 'credit', 'id': board.task['id'], 'award': 999})[0] == 400
    finally:
        server.close()
        ledger.sealed.close()
