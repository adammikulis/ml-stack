"""Authorized History reads and agent-grouped action inspection over a daemon socket."""

from importlib.metadata import entry_points

import pytest
from test_fleet_ui import Serving

from ml_stack.activity import writer
from ml_stack.activity.log import ActivityLog
from ml_stack.fleet import routes
from tests.activity_support import person, ring

__all__ = ['person', 'ring']


@pytest.fixture
def history(tmp_path, person, monkeypatch):
    log = ActivityLog(tmp_path / "activity", key=lambda: bytes(range(32)))
    monkeypatch.setattr(writer, "log", lambda: log)
    assert writer.record('agent.tool_call', actor='agent:scout', subject='<img src=x>', outcome='ok',
                  refs={'task': 'msg:42', 'conversation': 'chat-7', 'project': '/work/demo'},
                  meta={'model': 'qwen', 'arguments': 'private prompt',
                        'issue_url': 'https://github.com/example/repo/issues/42'})
    assert writer.record('workspace.claim', actor='agent:builder', subject='file:demo.py', outcome='claimed')
    server = Serving(tmp_path)
    server.ui.settings.setup_done = True
    try:
        yield server
    finally:
        server.close()


@pytest.mark.redteam
def test_installed_history_extension_reads_redacted_actions(history):
    assert any(entry.value == 'ml_stack.activity.fleet_routes:route'
               for entry in entry_points(group='ml_stack.ui_routes'))
    code, data, _ = history.call('/ui/history/events')
    assert code == 200
    scout = next(event for event in data['events'] if event['actor'] == 'agent:scout')
    assert scout['refs']['task'] == 'msg:42' and scout['meta']['model'] == 'qwen'
    assert 'arguments' not in scout['meta'] and 'private prompt' not in str(data)
    assert history.call('/ui/history/events', method='POST', body={})[0] == 405


@pytest.mark.redteam
def test_history_requires_ui_header_and_joined_session(history, monkeypatch):
    assert history.call('/ui/history/events', ui_header=False)[0] == 403
    monkeypatch.setattr(routes, 'in_cluster', lambda _: True)
    assert history.call('/ui/history/events')[0] == 401
    cookie = history.ui.sessions.cookie_header(history.ui.sessions.open('person'))
    assert history.call('/ui/history/events', cookie=cookie)[0] == 200


@pytest.mark.slow
def test_history_groups_agents_expands_actions_and_filters_on_mobile(history, playwright):
    from playwright.sync_api import expect

    with playwright.chromium.launch(headless=True) as browser:
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{history.port}/ui/#history')
        viewer = page.locator('history-view')
        expect(viewer.locator('.history-agent').filter(has_text='agent:scout')).to_be_visible()
        row = viewer.locator('.history-action').filter(has_text='<img src=x>')
        row.locator('summary').click()
        expect(row.get_by_text('msg:42', exact=True)).to_be_visible()
        expect(row.get_by_role('link', name='issue url')).to_have_attribute(
            'href', 'https://github.com/example/repo/issues/42')
        assert viewer.locator('img').count() == 0
        viewer.locator('#history-refresh').click()
        expect(row.locator('dd').get_by_text('qwen', exact=True)).to_be_visible()
        viewer.locator('#history-agent').select_option('agent:builder')
        expect(viewer.get_by_text('file:demo.py', exact=True)).to_be_visible()
        expect(viewer.get_by_text('<img src=x>', exact=True)).to_have_count(0)
        viewer.locator('#history-search').fill('absent-action')
        expect(viewer.locator('#history-status')).to_have_text('No actions match these filters.')
        viewer.locator('#history-search').fill('')
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path='/private/tmp/ml-stack-agent-history-mobile.png', full_page=True)
        assert not errors
