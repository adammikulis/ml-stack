"""Canonical workers select queued allocations and persist proposals in the graph."""

from taskboard_kit import board

from ml_stack.workspace import task_runtime, task_worker

__all__ = ['board']


def test_actual_graph_assignment_proposal_and_no_repeat(board):
    allocation = task_worker.assigned(board.ws, board.worker_id, board.board)
    assert allocation['allocation_id'] == board.allocation['allocation_id']
    def run(task, project, stopped, checkpoint):
        assert str(project) == allocation['project']
        checkpoint({'summary': 'Native tool completed', 'commit': allocation['baseline_commit']})
        return {'artifacts': {'result.md': 'a' * 64},
                'checks': [{'name': 'Harness returned', 'passed': True}],
                'summary': 'Proposed work', 'provenance': {'commit': 'a' * 40}}
    proposal = task_runtime.execute(board.board, board.worker_id, board.task['id'], allocation['allocation_id'], run)
    task = board.board.get(board.parent, board.task['id'])
    assert task['state'] == 'review'
    assert task['proposal']['id'] == proposal['id']
    assert task['checkpoints'][0]['summary'] == 'Native tool completed'
    assert task_worker.assigned(board.ws, board.worker_id, board.board) is None


def test_actual_graph_block_does_not_reenter_worker(board):
    def run(*_):
        raise RuntimeError('Approval expired')
    task_runtime.execute(board.board, board.worker_id, board.task['id'], board.allocation['allocation_id'], run)
    task = board.board.get(board.parent, board.task['id'])
    assert task['state'] == 'blocked'
    assert task['blocked_reason'] == 'Approval expired'
    assert task_worker.assigned(board.ws, board.worker_id, board.board) is None
