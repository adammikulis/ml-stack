"""Operational model capabilities observed from model files and serving flags."""

import json
from pathlib import Path

from poolhouse.hub.header import meta

CONFIG_LIMIT = 1 << 20
_SPEECH = ('whisper', 'wav2vec', 'parakeet', 'snac')
_EMBEDDING = ('bert', 'roberta', 'nomic-bert', 'gemma-embedding')
_CLASSIFIERS = ('ForSequenceClassification', 'ForTokenClassification', 'ForMaskedLM')
_TEXT = ('llama', 'qwen', 'gemma', 'gpt', 'phi', 'mistral', 'falcon', 'deepseek', 'mllama')


def _metadata(path):
    if path is None:
        return {}
    path = Path(path)
    if path.is_file():
        return meta(path) or {}
    config = path / 'config.json'
    try:
        if config.stat().st_size > CONFIG_LIMIT:
            return {}
        found = json.loads(config.read_text())
        return found if isinstance(found, dict) else {}
    except (OSError, ValueError, RecursionError):
        return {}


def capabilities(path=None, *, architecture='', embedding=None):
    """Return chat eligibility and its evidence; unknown eligibility remains explicit."""
    found = _metadata(path)
    arch = str(found.get('general.architecture') or found.get('model_type') or architecture).lower()
    pooling = found.get(f'{arch}.pooling_type')
    classes = found.get('architectures')
    classes = classes if isinstance(classes, list) else []
    if embedding is True:
        return {'chat': False, 'embedding': True, 'speech': False, 'evidence': 'server embedding mode'}
    if any(arch.startswith(value) for value in _SPEECH):
        return {'chat': False, 'embedding': False, 'speech': True, 'evidence': f'model architecture: {arch}'}
    if (any(arch.startswith(value) for value in _EMBEDDING)
            or (type(pooling) is int and pooling > 0)):
        return {'chat': False, 'embedding': True, 'speech': False, 'evidence': f'model architecture/pooling: {arch}'}
    if any(isinstance(value, str) and value.endswith(_CLASSIFIERS) for value in classes):
        return {'chat': False, 'embedding': None, 'speech': False, 'evidence': 'classification model architecture'}
    if (any(arch.startswith(value) for value in _TEXT)
            or any(isinstance(value, str) and value.endswith('ForCausalLM') for value in classes)):
        return {'chat': True, 'embedding': False, 'speech': False, 'evidence': f'text decoder architecture: {arch}'}
    return {'chat': None, 'embedding': None, 'speech': None,
            'evidence': f'chat capability unknown: {arch or "no model metadata"}'}


def allows_chat(value):
    """Accept unknown capability and exclude explicit evidence of non-chat operation."""
    return not isinstance(value, dict) or value.get('chat') is not False
