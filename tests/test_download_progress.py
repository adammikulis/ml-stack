"""Measured download rates and combined model/head progress."""

from pathlib import Path
from unittest.mock import patch

from ml_stack.fleet.download_progress import Transfer
from ml_stack.fleet.models import Downloads, Getting, Model


def test_transfer_excludes_resumed_bytes_and_estimates_only_known_totals():
    progress = Transfer()
    with patch('ml_stack.fleet.download_progress.time.monotonic', side_effect=[10, 12, 14]):
        progress.update(100, 1000)
        assert progress.eta is None
        progress.update(300, 1000)
        assert progress.speed == 100
        assert progress.eta == 7
        progress.update(500, 0)
        assert progress.speed == 100
        assert progress.eta is None
        assert 'left' not in progress.text()


def test_model_and_mtp_bytes_accumulate_and_each_phase_is_identified(tmp_path):
    phases = []
    head = tmp_path / 'draft.gguf'
    head.write_bytes(b'12345')
    row = Getting('test', 'model.gguf', pending_draft=True)

    class Store:
        def ensure(self, name, **options):
            options['on_note']('Downloading model.gguf')
            options['on_progress'](100, 100)
            phases.append(row.public())
            return Model(name, Path('model.gguf'), 100, 0)

        def ensure_draft(self, model, draft, **options):
            options['on_progress'](2, 5)
            phases.append(row.public())
            options['on_progress'](5, 5)
            return head

    Downloads(Store())._run(row, None, True, 'hf:fixture/draft')
    assert phases[0]['phase'] == 'model'
    assert phases[0]['eta_s'] is None
    assert phases[1]['phase'] == 'mtp'
    assert phases[1]['done'] == 102
    assert phases[1]['total'] == 105
    assert row.public()['done'] == row.public()['total'] == 105
    assert row.public()['eta_s'] is None
