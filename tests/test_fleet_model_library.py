"""Canonical model discovery, shard details and person-facing library filters."""

import os
from types import SimpleNamespace

import pytest
import test_fleet_page as fleet_page
from conftest import write_gguf

from ml_stack.fleet import routes
from ml_stack.fleet.models import Models
from ml_stack.fleet.serving import Serving

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page


def weights(path, name, size=1024 * 1024, modified=100):
    write_gguf(path, {'general.architecture': 'llama', 'general.name': name,
                      'general.file_type': 15, 'llama.context_length': 8192})
    with path.open('ab') as output:
        output.truncate(size)
    os.utime(path, (modified, modified))
    return path


def test_library_uses_canonical_discovery_without_changing_transfer_files(tmp_path):
    first = weights(tmp_path / 'Qwen3-27B-Q4_K_M-00001-of-00002.gguf', 'Qwen3-27B')
    second = weights(tmp_path / 'Qwen3-27B-Q4_K_M-00002-of-00002.gguf', 'Qwen3-27B')
    weights(tmp_path / 'mmproj-Qwen3-27B-F16.gguf', 'Projector')
    weights(tmp_path / 'mtp-Qwen3-27B-Q8_0.gguf', 'Draft head')
    models = Models([tmp_path], tmp_path)
    rows = models.library()
    assert len(rows) == 1
    row = rows[0]
    assert row['path'] == str(first.resolve()) and row['family'] == 'Qwen'
    assert row['shards'] == 2 and row['size_bytes'] == first.stat().st_size + second.stat().st_size
    assert row['servable'] and row['status'] == 'ready'
    assert [file['name'] for file in row['files']] == [first.name, second.name]
    assert row['companion']['name'].startswith('mmproj-')
    assert models.library_model(str(first.resolve())).path == first.resolve()
    assert models.find(second.name).path == second
    assert second.name in {file.name for file in models.all()}
    second.unlink()
    assert models.library()[0]['status'] == 'incomplete'
    assert models.library_model(str(first.resolve())) is None
    assert models.library_model(str(tmp_path / 'mtp-Qwen3-27B-Q8_0.gguf')) is None


def test_non_chat_models_remain_in_library_with_operational_capabilities(tmp_path):
    write_gguf(tmp_path / 'chat-looking.gguf', {'general.architecture': 'bert', 'bert.pooling_type': 2})
    weights(tmp_path / 'qwen.gguf', 'Decoder')
    rows = Models([tmp_path], tmp_path).library()
    assert len(rows) == 2 and all(row['servable'] for row in rows)
    embedding = next(row for row in rows if row['architecture'] == 'bert')
    decoder = next(row for row in rows if row['architecture'] == 'llama')
    assert embedding['capabilities']['chat'] is False
    assert decoder['capabilities']['chat'] is True


@pytest.mark.redteam
def test_serving_refuses_missing_first_shard_and_outside_paths(joined, monkeypatch):
    monkeypatch.setattr(routes, '_can_serve', lambda: True)
    joined.ui.serving = Serving(joined.files.parent / 'serving.json')
    first = weights(joined.files / 'Qwen3-27B-Q4_K_M-00002-of-00002.gguf', 'Qwen3-27B')
    weights(joined.files / 'Qwen3-27B-Q4_K_M-00003-of-00002.gguf', 'Qwen3-27B')
    calls = []
    joined.ui.start_serving = lambda model: calls.append(model) or SimpleNamespace(public=lambda: {'port': 12345})
    for path in (str(first), str(joined.files.parent / 'outside.gguf')):
        status, _, _ = joined.call('/ui/serving', method='POST', body={'path': path}, cookie=joined.cookie)
        assert status == 404
    assert calls == []


@pytest.mark.slow
def test_models_browser_groups_shards_filters_sorts_and_serves_exact_path(joined, open_page, monkeypatch):
    monkeypatch.setattr(routes, '_can_serve', lambda: True)
    joined.ui.serving = Serving(joined.files.parent / 'serving.json')
    first = weights(joined.files / 'Qwen3-27B-Q4_K_M-00001-of-00002.gguf', 'Qwen3-27B')
    weights(joined.files / 'Qwen3-27B-Q4_K_M-00002-of-00002.gguf', 'Qwen3-27B')
    weights(joined.files / 'Ornith-9B-Q8_0.gguf', 'Ornith-9B', modified=200)
    calls = []
    joined.ui.start_serving = lambda model: calls.append(model.path) or SimpleNamespace(public=lambda: {'port': 12345})
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#models')
    library = page.locator('models-library')
    library.locator('article').filter(has_text='Qwen3-27B').wait_for()
    assert library.locator('article').filter(has_text='Qwen3-27B').count() == 1
    row = library.locator('article').filter(has_text='Qwen3-27B')
    assert not row.locator('code').is_visible()
    row.locator('summary').click()
    assert row.locator('li').count() == 2
    assert '2 shards expected' in row.inner_text()
    library.get_by_text('Advanced library filters', exact=True).click()
    library.get_by_label('Family', exact=True).select_option('Qwen')
    assert library.locator('article').count() == 1
    library.get_by_label('Quantization', exact=True).select_option('Q8_0')
    assert library.get_by_text('No installed models match these filters.').is_visible()
    library.get_by_label('Family', exact=True).select_option('')
    assert library.locator('article').count() == 1
    library.get_by_label('Quantization', exact=True).select_option('')
    library.get_by_label('Sort installed models').select_option('size')
    assert 'Qwen3-27B' in library.locator('article').first.inner_text()
    library.get_by_label('Sort installed models').select_option('recent')
    assert 'Ornith-9B' in library.locator('article').first.inner_text()
    library.get_by_label('Search installed models').fill('Qwen3')
    assert library.locator('article').count() == 1
    library.get_by_role('button', name='Serve model').click()
    page.wait_for_selector('#models-note .ok')
    assert calls == [first.resolve()]
    assert not errors
