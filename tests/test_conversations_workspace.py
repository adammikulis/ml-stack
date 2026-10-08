"""Shared team and model conversations driven through the Fleet page."""

import pytest
import test_fleet_chat_browser as chat_fixtures
from workspace_kit import Kit, clean_env

from ml_stack.workspace import tokens

chat_browser = chat_fixtures.chat_browser
pytestmark = pytest.mark.slow


def test_team_composer_survives_history_failure_and_retries_authenticated_send(chat_browser, monkeypatch, tmp_path):
    from playwright.sync_api import expect

    kit = Kit(clean_env(monkeypatch, tmp_path / 'workspace'))
    kit.limits(sends_per_window=1000)
    helper = kit.agent('research-helper')
    tokens.store(kit.base, tokens.OWNER_FILE, kit.owner)
    kit.ws.board.create(helper, '#research')
    for number in range(105):
        kit.ws.send(helper, '#research', 'note', f'Existing research message {number}')
    served, page = chat_browser
    page.goto(f'http://127.0.0.1:{served.port}/ui#board')
    page.get_by_label('Workspace', exact=True).select_option('local')
    page.locator('#conversation-channels').get_by_role('button', name='#research').click()
    board = page.locator('chat-view ml-board')
    editor = board.get_by_role('textbox', name='Message', exact=True)
    expect(editor).to_be_visible()
    editor.fill('Keep this research draft')
    editor.evaluate('(node) => { window.retainedTeamEditor = node; }')
    observed = []

    def fail_history(route):
        observed.append(editor.is_visible())
        observed.append(editor.evaluate('(node) => node === window.retainedTeamEditor'))
        route.abort('failed')

    page.route('**/board/messages?**', fail_history)
    board.get_by_role('button', name='Messages', exact=True).click()
    expect(board.get_by_role('alert')).to_be_visible()
    expect(editor).to_have_value('Keep this research draft')
    assert observed == [True, True]
    assert editor.evaluate('(node) => node === window.retainedTeamEditor')
    page.unroute('**/board/messages?**', fail_history)
    board.get_by_role('button', name='Retry', exact=True).click()
    expect(board.locator('.msg').first).to_be_visible()

    def fail_send(route):
        route.fulfill(status=503, content_type='application/json', body='{"error":"Send temporarily unavailable"}')

    page.route('**/board/post', fail_send)
    board.get_by_role('button', name='Send', exact=True).click()
    expect(board.get_by_role('alert')).to_contain_text('Send temporarily unavailable')
    expect(editor).to_have_value('Keep this research draft')
    page.unroute('**/board/post', fail_send)
    board.get_by_role('button', name='Send', exact=True).click()
    expect(editor).to_have_value('')
    expect(board.locator('.msg pre').last).to_contain_text('Keep this research draft')
    assert kit.ws.board.ui_read(kit.owner, '#research')['messages'][-1]['body'] == 'Keep this research draft'
    board.get_by_role('button', name='Reply in thread').last.click()
    editor.fill('Authenticated thread reply')
    board.get_by_role('button', name='Send', exact=True).click()
    expect(board.locator('.msg pre').last).to_contain_text('Authenticated thread reply')
    assert any(row['body'] == 'Authenticated thread reply'
               for row in kit.ws.board.ui_read(kit.owner, '#research')['messages'])


def test_shared_sidebar_routes_channels_dms_threads_and_saved_model_chats(chat_browser, monkeypatch, tmp_path):
    from playwright.sync_api import expect

    kit = Kit(clean_env(monkeypatch, tmp_path / 'workspace'))
    kit.limits(sends_per_window=1000)
    worker = kit.agent('research-helper')
    tokens.store(kit.base, tokens.OWNER_FILE, kit.owner)
    kit.ws.board.create(worker, '#experiments')
    kit.ws.send(worker, '#experiments', 'note', 'The evaluation is ready.', subject='Model evaluation')
    kit.ws.send(worker, 'owner', 'note', 'Review the experiment criteria.')
    served, page = chat_browser
    saved = served.ui.conversations.start(model='model-a', title='Experiment plan')
    served.ui.conversations.append(saved.id, 'assistant', 'Compare the two evaluation runs.')
    page.goto(f'http://127.0.0.1:{served.port}/ui#chat')
    page.get_by_label('Workspace', exact=True).select_option('local')
    expect(page.locator('#conversation-channels')).to_contain_text('#experiments')
    sidebar = page.locator('chat-view .chats')
    sidebar.get_by_role('button', name='#experiments').click()
    expect(page.locator('#conversation-team')).to_be_visible()
    expect(page.locator('#conversation-model')).to_be_hidden()
    board = page.locator('chat-view ml-board')
    expect(board.locator('.msg pre')).to_contain_text('The evaluation is ready.')
    expect(board.locator('nav')).to_be_hidden()
    editor = board.get_by_role('textbox', name='Message', exact=True)
    editor.fill('Use the held-out dataset.')
    board.get_by_role('button', name='Send', exact=True).click()
    expect(board.locator('.msg pre')).to_contain_text(['The evaluation is ready.', 'Use the held-out dataset.'])
    assert any(message['body'] == 'Use the held-out dataset.' for message in kit.ws.board.ui_read(kit.owner, '#experiments')['messages'])
    board.get_by_role('button', name='Reply in thread').first.click()
    expect(board.get_by_role('button', name='Back to #experiments')).to_be_visible()
    expect(editor).to_be_in_viewport()
    expect(board.get_by_role('button', name='Send', exact=True)).to_be_in_viewport()
    from ml_stack.workspace.agent_display import metadata

    worker_id = kit.ws.auth(worker).id
    worker_label = metadata(kit.ws.registry, worker_id)['display_name']
    peer = sidebar.locator(f'button[title="{worker_id}"]')
    expect(peer).to_contain_text(worker_label)
    expect(peer).to_have_attribute('title', worker_id)
    peer.click()
    assert board.evaluate('(node) => node.target().to') == worker_id
    expect(board.locator('.msg pre')).to_contain_text('Review the experiment criteria.')
    expect(editor).to_be_in_viewport()
    expect(board.get_by_role('button', name='Send', exact=True)).to_be_in_viewport()
    sidebar.get_by_role('link', name='Experiment plan', exact=True).click()
    expect(page.locator('#conversation-model')).to_be_visible()
    expect(page.locator('#chat-messages')).to_contain_text('Compare the two evaluation runs.')
    expect(page.locator('#conversation-model').get_by_role('textbox', name='Message', exact=True)).to_be_in_viewport()
    expect(page.locator('#chat-send')).to_be_in_viewport()
    expect(sidebar).to_be_visible()
    page.screenshot(path='/private/tmp/poolside-unified-conversations.png', full_page=True)


def test_canonical_project_selection_never_falls_back_to_local_board(chat_browser):
    from playwright.sync_api import expect

    served, page = chat_browser
    project = '7' * 32
    seen = []
    page.route('**/ui/projects', lambda route: route.fulfill(json={'workspaces': [
        {'id': project, 'name': 'Experiment workspace', 'is_self': True, 'local_authority': True}]}))

    def unavailable(route):
        seen.append(route.request.url)
        route.fulfill(status=503, json={'error': 'Project Board identity is unavailable.'})

    page.route(f'**/ui/projects/{project}/board/**', unavailable)
    page.goto(f'http://127.0.0.1:{served.port}/ui#chat')
    expect(page.get_by_label('Workspace', exact=True)).to_have_value(project)
    expect(page.locator('#conversation-channels')).to_contain_text('Project Board identity is unavailable')
    assert seen and all(f'/ui/projects/{project}/board/' in url for url in seen)
    assert page.locator('chat-view ml-board').get_attribute('endpoint') == f'/ui/projects/{project}/board'
    assert page.get_by_label('Workspace', exact=True).input_value() != 'local'


def test_new_dm_from_model_chat_opens_full_child_identity_and_survives_reload(chat_browser, monkeypatch, tmp_path):
    from playwright.sync_api import expect

    kit = Kit(clean_env(monkeypatch, tmp_path / 'workspace'))
    kit.limits(sends_per_window=1000)
    parent_name, child_name = 'p' * 48, 'c' * 48
    parent = kit.agent(parent_name)
    child = kit.ws.registry.delegate(kit.ws.auth(parent), child_name, 600, (), 10)
    identity = parent_name + '/' + child_name
    tokens.store(kit.base, tokens.OWNER_FILE, kit.owner)
    kit.ws.send(child, 'owner', 'note', 'Ready to review this experiment.')
    served, page = chat_browser
    page.goto(f'http://127.0.0.1:{served.port}/ui#chat')
    page.get_by_label('Workspace', exact=True).select_option('local')
    direct = page.locator('#conversation-direct')
    direct.locator('summary').get_by_text('+ New direct message', exact=True).click()
    direct.get_by_label('Message an agent', exact=True).select_option(identity)
    direct.get_by_role('button', name='Open', exact=True).click()
    expect(page.locator('#conversation-team')).to_be_visible()
    expect(page.locator('#conversation-model')).to_be_hidden()
    board = page.locator('chat-view ml-board')
    expect(board.locator('.msg pre')).to_contain_text('Ready to review this experiment.')
    assert board.evaluate("node => [...node.feed.childNodes].filter(child => child.nodeType === Node.TEXT_NODE).map(child => child.textContent.trim()).filter(Boolean)") == []
    assert board.evaluate('node => node.view.b') == identity
    board.get_by_label('Message', exact=True).fill('Review the held-out dataset.')
    board.get_by_role('button', name='Send', exact=True).click()
    expect(board.locator('.msg pre')).to_contain_text(['Ready to review this experiment.', 'Review the held-out dataset.'])
    page.reload()
    expect(page.locator('#conversation-team')).to_be_visible()
    expect(board.locator('.msg pre')).to_contain_text(['Ready to review this experiment.', 'Review the held-out dataset.'])
    assert board.evaluate('node => node.view.b') == identity


@pytest.mark.parametrize('theme_tokens', [
    ('light', '#ffffff', '#1b1f3a', '#ff5fa2', '#1b1f3a', 1),
    ('dark', '#242943', '#edf0fa', '#ff5fa2', '#1b1f3a', .8),
    ('custom', '#253d31', '#e8ffef', '#652891', '#ffffff', .65),
])
def test_conversation_surfaces_consume_resolved_theme_tokens(chat_browser, monkeypatch, tmp_path, theme_tokens):
    from playwright.sync_api import expect

    theme, surface, ink, pink, pink_ink, density = theme_tokens

    kit = Kit(clean_env(monkeypatch, tmp_path / 'workspace'))
    tokens.store(kit.base, tokens.OWNER_FILE, kit.owner)
    kit.ws.board.create(kit.agent('theme-reviewer'), '#theme')
    served, page = chat_browser
    page.goto(f'http://127.0.0.1:{served.port}/ui#chat')
    page.locator('#chat-model-button').wait_for(state='visible')
    page.evaluate('''settings => {
      const root = document.documentElement;
      root.dataset.theme = settings.theme;
      root.style.fontSize = '18px';
      for (const [key,value] of Object.entries(settings.tokens)) root.style.setProperty(key,value);
    }''', {'theme': theme, 'tokens': {
        '--ml-bg': surface, '--ml-surface': surface, '--ml-sunken': surface,
        '--ml-text': ink, '--ml-muted': ink, '--ml-accent': pink, '--ml-accent-ink': ink,
        '--ml-on-accent': pink_ink, '--poolside-pink': pink, '--poolside-cyan': '#2de2e6',
        '--poolside-pink-ink': pink_ink, '--ui-density': str(density), '--ml-font': 'Georgia, serif',
    }})
    expected = page.evaluate('''values => {
      const sample = document.createElement('span'); document.body.append(sample);
      const result = values.map(value => {sample.style.color=value; return getComputedStyle(sample).color;});
      sample.remove(); return result;
    }''', [surface, ink, pink, pink_ink])
    actual = page.locator('#conversation-model').evaluate('''node => ({
      background:getComputedStyle(node).backgroundColor, ink:getComputedStyle(node).color,
      font:getComputedStyle(node).fontFamily, padding:parseFloat(getComputedStyle(node.querySelector('.chat-toolbar')).paddingLeft),
      title:parseFloat(getComputedStyle(node.querySelector('h1')).fontSize)
    })''')
    assert actual['background'] == expected[0]
    assert actual['ink'] == expected[1]
    assert 'Georgia' in actual['font']
    assert actual['padding'] == pytest.approx(28 * density)
    assert actual['title'] > 20
    send = page.locator('#chat-send').evaluate('node => ({bg:getComputedStyle(node).backgroundColor,ink:getComputedStyle(node).color})')
    assert send == {'bg': expected[2], 'ink': expected[3]}
    page.locator('#chat-model-button').click()
    expect(page.locator('#chat-model-dialog')).to_be_visible()
    assert page.locator('#chat-model-dialog').evaluate('node => getComputedStyle(node).backgroundColor') == expected[0]
    page.get_by_role('button', name='Close', exact=True).click()
    page.get_by_label('Workspace', exact=True).select_option('local')
    page.locator('#conversation-channels').get_by_role('button', name='#theme', exact=True).click()
    board = page.locator('chat-view ml-board')
    expect(board.get_by_label('Message', exact=True)).to_be_visible()
    assert board.locator('main header').evaluate('node => getComputedStyle(node).backgroundColor') == expected[0]
    assert board.locator('.composer').evaluate('node => getComputedStyle(node).backgroundColor') == expected[0]
    assert board.get_by_label('Message', exact=True).evaluate('node => getComputedStyle(node).color') == expected[1]
    page.screenshot(path=f'/private/tmp/poolside-conversations-theme-{theme}.png', full_page=True)


def test_unshared_canonical_workspace_is_selected_and_team_messages_render(chat_browser, monkeypatch, tmp_path):
    from playwright.sync_api import expect

    from ml_stack.fleet import project_client, project_source
    from ml_stack.fleet.projects import ProjectRegistry, identity
    from ml_stack.net import git
    from ml_stack.workspace.remote_host import WorkspaceHost

    checkout = tmp_path / 'experiment-workspace'
    checkout.mkdir()
    git.run(['init'], cwd=checkout)
    git.run(['remote', 'add', 'origin', 'https://code.example.invalid/team/canonical-demo.git'], cwd=checkout)
    registry = ProjectRegistry(tmp_path / 'registry', 'fixture-device', (checkout,), 'http://127.0.0.1:8770')
    project = identity(checkout)
    host = WorkspaceHost(registry)
    host.prepare(project)
    ws = host.workspace(project)
    owner = tokens.read_file(tokens.directory(ws.base) / tokens.OWNER_FILE)
    worker = ws.mint(owner, 'canonical-demo-helper')
    ws.send(worker, '#general', 'note', 'The canonical team experiment is ready.')
    monkeypatch.setattr(project_client, 'peers', lambda ui: [])
    monkeypatch.setattr(project_source, 'build', lambda *args: pytest.fail('Board chooser published source'))
    served, page = chat_browser
    served.ui.projects, served.ui.workspaces = registry, host
    session = served.ui.sessions.open('fixture-person')
    page.context.add_cookies([{'name': 'ml_stack_ui', 'value': session.sid,
                              'url': f'http://127.0.0.1:{served.port}/ui', 'httpOnly': True}])
    assert registry.get(project).shared is False
    assert registry.list() == []
    page.goto(f'http://127.0.0.1:{served.port}/ui#chat')
    expect(page.get_by_label('Workspace', exact=True)).to_have_value(project)
    board = page.locator('chat-view ml-board')
    assert board.get_attribute('endpoint') == f'/ui/projects/{project}/board'
    expect(page.locator('#conversation-channels').get_by_role('button', name='Join workspace as person')).to_be_visible()
    expect(page.locator('#conversation-direct')).to_have_text('No direct messages yet.')
    page.locator('#conversation-channels').get_by_role('button', name='Join workspace as person').click()
    expect(board.get_by_role('button', name='Join workspace as person')).to_be_visible()
    assert all('person_project' not in row for row in ws.registry._load().values())
    board.get_by_role('button', name='Join workspace as person').click()
    expect(board.locator('.msg pre')).to_contain_text('The canonical team experiment is ready.')
    assert registry.get(project).shared is False
    assert registry.list() == []
    assert not (tmp_path / 'registry' / 'project-bundles').exists()


def _existing_dm_directory(monkeypatch, tmp_path):
    from ml_stack.workspace.agent_display import metadata

    kit = Kit(clean_env(monkeypatch, tmp_path / 'workspace'))
    kit.limits(sends_per_window=1000)
    agents = [kit.agent(f'helper-{number}') for number in range(43)]
    worker = agents[0]
    worker_id = kit.ws.auth(worker).id
    worker_label = metadata(kit.ws.registry, worker_id)['display_name']
    tokens.store(kit.base, tokens.OWNER_FILE, kit.owner)
    kit.ws.board.create(worker, '#experiments')
    kit.ws.send(worker, 'owner', 'note', 'Existing experiment discussion.')
    return kit, agents, worker_id, worker_label


def test_sidebar_keeps_agent_directory_in_new_dm_and_searches_existing_conversations(chat_browser, monkeypatch, tmp_path):
    from playwright.sync_api import expect

    expect.set_options(timeout=30000)  # each of the 43 sessions opens its own store when listed
    kit, agents, worker_id, worker_label = _existing_dm_directory(monkeypatch, tmp_path)
    served, page = chat_browser
    saved = served.ui.conversations.start(model='model-a', title='Experiment plan')
    served.ui.conversations.append(saved.id, 'assistant', 'Recorded model conversation.')
    page.set_viewport_size({'width': 1440, 'height': 900})
    page.goto(f'http://127.0.0.1:{served.port}/ui#chat')
    page.get_by_label('Workspace', exact=True).select_option('local')
    direct = page.locator('#conversation-direct')
    expect(direct.locator(':scope > button')).to_have_count(1)
    expect(direct.locator('[aria-label="1 unread messages"]')).to_be_visible()
    expect(page.locator('#chat-list').get_by_role('link', name='Experiment plan', exact=True)).to_be_visible()
    expect(page.locator('#conversation-models').get_by_role('button', name='● model-a', exact=True)).to_be_in_viewport()
    page.screenshot(path='/private/tmp/poolside-sidebar-existing-dms.png', full_page=True)
    search = page.get_by_label('Search conversations and channels', exact=True)
    expect(search).to_have_attribute('placeholder', 'Search chats & channels')
    search.fill('Experiment')
    expect(page.locator('#conversation-channels').get_by_role('button', name='#experiments', exact=True)).to_be_visible()
    expect(direct.locator(':scope > button')).to_have_count(0)
    expect(page.locator('#chat-list').get_by_role('link', name='Experiment plan', exact=True)).to_be_visible()
    search.fill(worker_label)
    expect(direct.locator(':scope > button')).to_have_count(1)
    expect(page.locator('#conversation-channels > button')).to_have_count(0)
    search.fill('')
    direct.locator('summary').click()
    directory = direct.get_by_label('Message an agent', exact=True)
    assert directory.locator('option').count() >= 44
    selected_id = kit.ws.auth(agents[-1]).id
    directory.select_option(selected_id)
    direct.get_by_role('button', name='Open', exact=True).click()
    board = page.locator('chat-view ml-board')
    expect(page.locator('#conversation-team')).to_be_visible()
    assert board.evaluate('node => node.target().to') == selected_id
    expect(direct.locator(':scope > button')).to_have_count(2)
    active = direct.locator(f':scope > button[title="{selected_id}"]')
    expect(active).to_have_attribute('aria-current', 'true')
    assert len(kit.ws.board.dm_list(kit.owner)) == 1
    page.reload()
    expect(active).to_have_attribute('aria-current', 'true')
    expect(direct.locator(':scope > button')).to_have_count(2)
    assert len(kit.ws.board.dm_list(kit.owner)) == 1
    direct.locator(f':scope > button[title="{worker_id}"]').click()
    expect(board.locator('.msg pre')).to_contain_text('Existing experiment discussion.')
    page.reload()
    expect(direct.locator(':scope > button')).to_have_count(1)
    expect(board.locator('.msg pre')).to_contain_text('Existing experiment discussion.')


def test_channel_history_scrolls_with_composer_visible_at_all_viewport_sizes(chat_browser, monkeypatch, tmp_path):
    from playwright.sync_api import expect

    kit = Kit(clean_env(monkeypatch, tmp_path / 'workspace'))
    kit.limits(sends_per_window=1000)
    worker = kit.agent('history-helper')
    tokens.store(kit.base, tokens.OWNER_FILE, kit.owner)
    for number in range(24):
        kit.ws.send(worker, '#general', 'note', f'History message {number}\nRecorded experiment details.\nRecorded evaluation details.')
    served, page = chat_browser
    page.goto(f'http://127.0.0.1:{served.port}/ui#chat')
    page.get_by_label('Workspace', exact=True).select_option('local')
    page.locator('#conversation-channels').get_by_role('button', name='#general', exact=False).click()
    board = page.locator('chat-view ml-board')
    expect(board.locator('.msg pre')).to_have_count(24)
    composer = board.get_by_role('textbox', name='Message', exact=True)
    for width, height in [(1360, 900), (1360, 640), (1360, 420), (390, 844)]:
        page.set_viewport_size({'width': width, 'height': height})
        expect(composer).to_be_in_viewport()
        expect(board.get_by_role('button', name='Send', exact=True)).to_be_in_viewport()
        bounds = composer.bounding_box()
        assert bounds['y'] >= 0 and bounds['y'] + bounds['height'] <= height
        geometry = board.evaluate('node => ({history:node.feed.scrollHeight, visible:node.feed.clientHeight, height:node.getBoundingClientRect().height})')
        assert 0 < geometry['visible'] < geometry['history']
        assert geometry['height'] <= height
        assert page.evaluate('document.documentElement.scrollHeight <= innerHeight')
        board.evaluate('node => { node.feed.scrollTop = node.feed.scrollHeight; }')
        expect(board.locator('.msg pre').last).to_be_in_viewport()
        expect(composer).to_be_in_viewport()
        composer.fill('Unsent channel draft')
        page.screenshot(path=f'/private/tmp/poolside-channel-fixed-{width}-{height}.png')
        composer.fill('')

    page.get_by_role('button', name='Channels and model chats ▾').click()
    expect(page.locator('chat-view .chats')).to_be_visible()
    expect(composer).to_be_in_viewport()
    expect(board.get_by_role('button', name='Send', exact=True)).to_be_in_viewport()
    assert page.evaluate('document.documentElement.scrollHeight <= innerHeight')
    page.screenshot(path='/private/tmp/poolside-channel-fixed-mobile-expanded.png')
