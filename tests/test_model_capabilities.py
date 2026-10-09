"""Observed model operations and unknown capability preservation."""

import json
from types import SimpleNamespace

import pytest
from conftest import write_gguf

from ml_stack.hub.capabilities import allows_chat, capabilities
from ml_stack.serve import process


@pytest.mark.parametrize('architecture,pooling', [('bert', 2), ('gemma-embedding', 1), ('other', 3)])
def test_embedding_metadata_excludes_chat_without_using_filename(tmp_path, architecture, pooling):
    path = tmp_path / 'friendly-chat.gguf'
    write_gguf(path, {'general.architecture': architecture, f'{architecture}.pooling_type': pooling})
    found = capabilities(path)
    assert found['chat'] is False and found['embedding'] is True
    assert not allows_chat(found)


def test_speech_config_excludes_chat_and_decoder_remains_eligible(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'model_type': 'whisper', 'architectures': ['WhisperForConditionalGeneration']}))
    assert capabilities(tmp_path)['speech'] is True
    assert not allows_chat(capabilities(tmp_path))
    config.write_text(json.dumps({'model_type': 'qwen3_5', 'architectures': ['QwenForCausalLM']}))
    assert capabilities(tmp_path)['chat'] is True


def test_classification_architecture_does_not_claim_embedding_support(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'architectures': ['OrnithForSequenceClassification']}))
    found = capabilities(tmp_path)
    assert found['chat'] is False and found['embedding'] is None


def test_unknown_and_invalid_metadata_remain_inspectable_and_eligible(tmp_path):
    for content in ('[]', '{', json.dumps({'model_type': 'future_decoder'})):
        (tmp_path / 'config.json').write_text(content)
        found = capabilities(tmp_path)
        assert found['chat'] is None and 'unknown' in found['evidence']
        assert allows_chat(found)
    assert allows_chat(None) and allows_chat({})
    assert allows_chat({'chat': 'false'})


def test_loaded_embedding_flag_overrides_decoder_metadata():
    found = capabilities(architecture='qwen3_5', embedding=True)
    assert not allows_chat(found) and found['evidence'] == 'server embedding mode'
    assert allows_chat(capabilities(architecture='qwen3_5', embedding=None))


@pytest.mark.parametrize('argv,embedding', [
    (['llama-server', '--port', '12345', '--embeddings'], True),
    (['llama-server', '--port', '12345'], False),
    ([], None),
])
def test_process_mode_uses_observed_arguments(monkeypatch, argv, embedding):
    instance = SimpleNamespace(info={'pid': 9876, 'name': 'llama-server', 'cmdline': argv,
                                     'memory_info': None}, status=lambda: 'running')
    monkeypatch.setattr(process.psutil, 'process_iter', lambda *_: [instance])
    assert process.every_server()[0]['embedding'] is embedding


def test_hostile_model_metadata_is_refused_without_a_crash(tmp_path):
    config = tmp_path / 'config.json'
    for content in ('[' * 500000, '{"a":' * 100000, json.dumps({'architectures': [1, None, {}], 'model_type': 7}),
                    json.dumps({'padding': 'x' * (1 << 20)})):
        config.write_text(content)
        found = capabilities(tmp_path)
        assert found['chat'] is None and allows_chat(found)
    (tmp_path / 'broken.gguf').write_bytes(b'GGUF' + b'\xff' * 64)
    assert allows_chat(capabilities(tmp_path / 'broken.gguf'))
