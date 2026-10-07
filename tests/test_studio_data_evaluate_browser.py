"""Studio dataset and evaluation workflows driven through their controls."""

from __future__ import annotations

import pytest
from playwright.sync_api import expect
from test_fleet_ui import Serving

from ml_stack.scrape.browser import Window, browser

pytestmark = pytest.mark.slow


def test_data_library_upload_preview_and_fine_tune(tmp_path):
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    try:
        with browser(Window(profile=tmp_path / 'browser', headless=True)) as page:
            page.goto(f'http://127.0.0.1:{served.port}/ui/#data')
            page.get_by_role('button', name='+ Add dataset').click()
            page.get_by_label('Destination relative to files root').fill('datasets/sample.jsonl')
            page.get_by_label('Or paste dataset content').fill(
                '{"prompt":"<img src=x onerror=alert(1)>","response":"Sample"}\n'
                '{"prompt":"Next question","response":"Next answer"}\n'
            )
            page.get_by_role('button', name='Upload', exact=True).click()
            expect(page.locator('data-view #preview-title')).to_have_text('sample.jsonl')
            expect(page.locator('data-view #structured-preview tbody tr')).to_have_count(2)
            expect(page.locator('data-view #structured-preview')).to_contain_text(
                '<img src=x onerror=alert(1)>'
            )
            assert page.locator('data-view #structured-preview img').count() == 0
            assert served.files.joinpath('datasets/sample.jsonl').is_file()
            page.get_by_role('button', name='Use for training', exact=True).click()
            expect(page).to_have_url(f'http://127.0.0.1:{served.port}/ui/#training')
            assert page.get_by_label('Dataset path (relative to files root)').input_value() == (
                'datasets/sample.jsonl'
            )
    finally:
        served.close()


def test_data_file_source_and_filter(tmp_path):
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    try:
        with browser(Window(profile=tmp_path / 'browser', headless=True)) as page:
            page.goto(f'http://127.0.0.1:{served.port}/ui/#data')
            page.get_by_role('button', name='+ Add dataset').click()
            page.get_by_role('button', name='Choose a file', exact=True).click()
            page.get_by_label('Choose a text / JSONL file').set_input_files({
                'name': 'examples.json', 'mimeType': 'application/json',
                'buffer': b'[{"prompt":"Example","response":"Answer"}]',
            })
            assert page.get_by_label('Destination relative to files root').input_value() == (
                'datasets/examples.json'
            )
            page.get_by_role('button', name='Upload', exact=True).click()
            expect(page.locator('data-view #structured-preview tbody tr')).to_have_count(1)
            page.get_by_label('Filter files in this folder').fill('missing')
            expect(page.locator('data-view #files')).to_contain_text('No matching files')
            page.get_by_label('Filter files in this folder').fill('examples')
            expect(page.locator('data-view .data-file')).to_have_count(1)
    finally:
        served.close()


def test_evaluate_selection_review_status_and_recorded_results(tmp_path, monkeypatch):
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    fleet = {
        'models': ['sample-qwen.gguf'],
        'peers': [{'name': 'studio', 'is_self': True, 'models': ['sample-qwen.gguf']}],
    }
    monkeypatch.setattr(served.ui, 'fleet', lambda: fleet)
    monkeypatch.setattr(served.ui, 'bench_state', lambda: {
        'available': True, 'text': 'No active measurement', 'measuring': None,
    })
    monkeypatch.setattr(served.ui, 'bench_history', lambda: [
        {'name': 'Recorded comparison', 'started': '2026-10-01T09:00:00',
         'exit': 'done', 'seconds': 12, 'questions': 8, 'kept': ['sample']},
        {'name': 'Unfinished comparison', 'exit': 'unknown', 'seconds': 2},
    ])
    monkeypatch.setattr(served.ui, 'rates', lambda: {'runs': [
        {'label': 'Recorded comparison', 'host': 'studio', 'right': .75,
         'seconds_per_question': 1.5, 'questions': 8},
    ]})
    try:
        with browser(Window(profile=tmp_path / 'browser', headless=True)) as page:
            page.goto(f'http://127.0.0.1:{served.port}/ui/#benchmarks')
            run = page.get_by_role('button', name='Start sweep', exact=True)
            expect(run).to_be_disabled()
            page.get_by_label('sample-qwen.gguf', exact=True).check()
            expect(run).to_be_enabled()
            page.get_by_label('Question set').select_option('limit')
            page.get_by_label('Questions per task', exact=True).fill('7')
            page.get_by_label('Run label', exact=True).fill('test comparison')
            page.get_by_role('button', name='Review command', exact=True).click()
            expect(page.locator('benchmarks-view #command')).to_contain_text('sample-qwen.gguf')
            expect(page.locator('benchmarks-view #command')).to_contain_text('--sample 7')
            expect(page.locator('benchmarks-view #results')).to_contain_text('75.0%')
            expect(page.locator('benchmarks-view #history')).to_contain_text('Complete')
            expect(page.locator('benchmarks-view #history')).to_contain_text('Unknown outcome')
            assert page.locator('benchmarks-view #history-json').is_visible() is False
            page.get_by_label('studio', exact=True).uncheck()
            expect(run).to_be_disabled()
    finally:
        served.close()


@pytest.mark.parametrize('theme', ['light', 'dark', 'custom'])
def test_studio_surfaces_follow_theme_and_keep_text_contrast(tmp_path, monkeypatch, theme):
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    served.files.joinpath('examples.jsonl').write_text('{"prompt":"Sample"}\n')
    monkeypatch.setattr(served.ui, 'fleet', lambda: {
        'models': ['sample-qwen.gguf'],
        'peers': [{'name': 'studio', 'is_self': True, 'models': ['sample-qwen.gguf']}],
    })
    try:
        with browser(Window(profile=tmp_path / 'browser', headless=True)) as page:
            page.goto(f'http://127.0.0.1:{served.port}/ui/#data')
            expect(page.locator('data-view .data-file')).to_have_count(1)
            page.evaluate('''theme => {
                const root = document.documentElement;
                root.dataset.theme = theme === 'custom' ? 'dark' : theme;
                if (theme === 'custom') {
                    const values = {'--bg':'#081c17', '--bg-raised':'#102d25',
                        '--bg-sunken':'#071b14', '--text':'#f5fff8',
                        '--muted':'#b9dac9', '--line':'#527364'};
                    for (const [key, value] of Object.entries(values)) root.style.setProperty(key, value);
                }
            }''', theme)
            page.locator('data-view .data-file').click()
            expect(page.locator('data-view #preview-title')).to_have_text('examples.jsonl')
            page.get_by_role('button', name='+ Add dataset').click()
            metrics = page.evaluate('''() => {
                const rgb = value => value.match(/[\\d.]+/g).slice(0,3).map(Number);
                const luminance = value => rgb(value).map(v => {
                    const channel = v/255;
                    return channel <= .04045 ? channel/12.92 : ((channel+.055)/1.055)**2.4;
                }).reduce((sum, v, i) => sum+v*[.2126,.7152,.0722][i],0);
                const measure = selector => {
                    const node = document.querySelector(selector), style = getComputedStyle(node);
                    let surface = node;
                    while (getComputedStyle(surface).backgroundColor === 'rgba(0, 0, 0, 0)') surface = surface.parentElement;
                    const a = luminance(style.color), b = luminance(getComputedStyle(surface).backgroundColor);
                    return (Math.max(a,b)+.05)/(Math.min(a,b)+.05);
                };
                window.studioThemeContrast = measure;
                const probe = document.createElement('div');
                probe.style.backgroundColor = 'var(--bg-raised)';
                document.body.append(probe);
                const expectedSurface = getComputedStyle(probe).backgroundColor;
                probe.remove();
                return {expectedSurface, panel: getComputedStyle(document.querySelector('data-view .data-panel')).backgroundColor,
                    contrast: ['data-view .data-file strong','data-view .data-file small',
                        'data-view #add-dataset','data-view .data-tabs button[aria-pressed=true]'].map(measure)};
            }''')
            assert metrics['panel'] == metrics['expectedSurface']
            assert min(metrics['contrast']) >= 4.5, metrics
            page.get_by_label('Filter files in this folder').fill('missing')
            assert page.evaluate("window.studioThemeContrast('data-view .data-empty')") >= 4.5
            page.evaluate("window.fleetModel.go('benchmarks')")
            expect(page.locator('benchmarks-view .evaluate-choice')).to_have_count(2)
            page.get_by_label('sample-qwen.gguf', exact=True).check()
            expect(page.locator('benchmarks-view #summary')).not_to_have_text('Checking your pool…')
            surfaces = page.evaluate('''() => {
                const probe = document.createElement('div');
                probe.style.backgroundColor='var(--bg-raised)';document.body.append(probe);
                const expected=getComputedStyle(probe).backgroundColor;probe.remove();
                return {expected, card:getComputedStyle(document.querySelector('benchmarks-view .evaluate-card')).backgroundColor,
                    contrast: ['benchmarks-view .evaluate-choice b','benchmarks-view .evaluate-choice small',
                        'benchmarks-view #summary'].map(window.studioThemeContrast)};
            }''')
            assert surfaces['card'] == surfaces['expected']
            assert min(surfaces['contrast']) >= 4.5, surfaces
    finally:
        served.close()
