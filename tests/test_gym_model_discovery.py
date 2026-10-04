"""Downloaded native vision weights remain visible without claiming runtime support."""

import json

from ml_stack import hub
from ml_stack.gym import models


def test_fastvlm_safetensors_is_discovered_as_vision_and_explicitly_unsupported(tmp_path, monkeypatch):
    folder = tmp_path / 'apple' / 'FastVLM-0.5B'
    folder.mkdir(parents=True)
    config = folder / 'config.json'
    config.write_text(json.dumps({'model_type': 'llava_qwen2', 'mm_vision_tower': 'mobileclip_l_1024'}))
    (folder / 'model.safetensors').write_bytes(b'metadata-only-fixture')
    discovered = hub.discover([tmp_path], kind='vision', refresh=True)
    assert len(discovered) == 1 and discovered[0].path == folder
    assert discovered[0].architecture == 'llava_qwen2'
    config.write_text(json.dumps({'model_type': 'qwen2'}))
    assert hub.discover([tmp_path], kind='vision', refresh=True) == []
    monkeypatch.setattr(models.hub, 'discover', lambda **_: discovered)
    monkeypatch.setattr(models.registry, 'listing', lambda: [])
    row = next(row for row in models.model_choices()['vision'] if 'FastVLM' in row['label'])
    assert row['status'] == 'unsupported' and row['available'] is False and row['backend'] is None
    assert 'not implemented' in row['reason']
