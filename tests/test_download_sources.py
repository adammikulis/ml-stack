"""Chosen download sources constrain model, head and runtime fetches."""

from unittest.mock import Mock, patch

import pytest

from ml_stack.fleet import llama
from ml_stack.fleet.models import Model, ModelError, Models
from ml_stack.fleet.routes import ModelRoutes
from ml_stack.fleet.settings import Settings


@pytest.mark.parametrize('policy', ['', 'lan'])
def test_model_policy_refuses_online_fallback(tmp_path, policy):
    store = Models([tmp_path], tmp_path, sources=lambda: policy)
    with (patch.object(store, 'where', return_value=[]), patch.object(store, '_from_internet') as online,
          pytest.raises(ModelError, match=r'Choose|Internet downloads are disabled')):
        store.ensure('missing.gguf', source='hf:fixture/model', key=b'key')
    online.assert_not_called()


def test_internet_only_never_discovers_or_copies_from_lan(tmp_path):
    store = Models([tmp_path], tmp_path, sources=lambda: 'internet')
    expected = Model('missing.gguf', tmp_path / 'missing.gguf', 1, 0)
    with patch.object(store, 'where') as peers, patch.object(store, '_from_internet', return_value=expected):
        assert store.ensure('missing.gguf', source='hf:fixture/model', key=b'key') is expected
    peers.assert_not_called()


def test_lan_only_head_cannot_fall_back_to_internet(tmp_path):
    store = Models([tmp_path], tmp_path, sources=lambda: 'lan')
    model = Model('model.gguf', tmp_path / 'model.gguf', 1, 0)
    with (patch.object(store, '_draft_from_peers', return_value=False), patch.object(store, '_from_internet') as online,
          pytest.raises(ModelError, match='MTP head; Internet downloads are disabled')):
        store.ensure_draft(model, 'hf:fixture/head', key=b'key')
    online.assert_not_called()


@pytest.mark.parametrize('policy', ['', 'lan'])
def test_runtime_policy_blocks_release_network_before_fetch(tmp_path, policy):
    with (patch.object(llama, 'find_server', return_value=None), patch.object(llama, '_tokens', return_value=('ubuntu-x64',)),
          patch.object(llama, 'latest') as releases, pytest.raises(llama.LlamaError, match='Choose Internet only or Both')):
        llama.ensure_server(tmp_path, sources=policy)
    releases.assert_not_called()


def test_policy_changes_are_read_for_each_transfer(tmp_path):
    selected = ['lan']
    store = Models([tmp_path], tmp_path, sources=lambda: selected[0])
    with patch.object(store, 'where', return_value=[]), patch.object(store, '_from_internet', return_value=Mock()) as online:
        with pytest.raises(ModelError):
            store.ensure('missing.gguf', source='hf:fixture/model')
        selected[0] = 'both'
        store.ensure('missing.gguf', source='hf:fixture/model')
    online.assert_called_once()


def test_unselected_sources_refuse_ui_download_before_starting_it():
    route = Mock()
    route.ui.settings = Settings()
    route.body.return_value = {'name': 'model.gguf', 'source': 'hf:fixture/model'}
    assert ModelRoutes._get_model(route, None, True)
    assert route.send.call_args.args[0] == 409
    route.ui.downloads.start.assert_not_called()
