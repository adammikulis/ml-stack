"""Existing Development pool sign-in over real Fleet sockets."""

import pytest
from test_fleet_ui import Serving

from ml_stack.fleet.discovery import clusters_path, memberships, mint_cluster, primary_ip


@pytest.fixture
def development(tmp_path):
    server = Serving(tmp_path, secure=False)
    mint_cluster('production', server.keyfile, mode='prod')
    mint_cluster('research', server.keyfile, mode='dev')
    mint_cluster('workshop', server.keyfile, mode='dev')
    server.ui.settings.setup_done = True
    try:
        yield server
    finally:
        server.close()


def local_post(server, body, **options):
    return server.call('/ui/setup/local-session', method='POST', body=body,
                       headers={'Origin': f'http://127.0.0.1:{server.port}',
                                'Sec-Fetch-Site': 'same-origin', **options.pop('headers', {})}, **options)


def test_existing_dev_pool_signin_preserves_memberships_and_session_context(development):
    server = development
    original = clusters_path(server.keyfile).read_bytes()
    status, result, _ = server.call('/ui/session')
    assert status == 200 and result['local_pools'] == ['workshop', 'research']
    assert result['signed_in'] is False
    assert 'key' not in str(result) and 'join' not in str(result)
    status, result, headers = local_post(server, {'pool': 'research'})
    assert status == 200 and result['pool'] == 'research'
    cookie = headers['Set-Cookie'].split(';')[0]
    status, session, _ = server.call('/ui/session', cookie=cookie)
    assert status == 200 and session['signed_in'] and session['pool'] == 'research'
    assert memberships(server.keyfile)[0].group == 'workshop'
    assert clusters_path(server.keyfile).read_bytes() == original
    server.call('/ui/session', method='DELETE', cookie=cookie)
    assert server.call('/ui/peers', cookie=cookie)[0] == 401


@pytest.mark.parametrize('body', [{'pool': 'production'}, {'pool': 'missing'}, {'pool': 1},
                                  {'pool': 'research', 'token': 'forged'}, []])
def test_dev_pool_signin_rejects_invalid_selection(development, body):
    original = len(development.ui.sessions)
    assert local_post(development, body)[0] == 400
    assert len(development.ui.sessions) == original


@pytest.mark.parametrize('headers', [{'Origin': 'https://foreign.example'},
    {'Origin': 'http://127.0.0.1:1'}, {'Host': 'foreign.example'},
    {'Host': '127.0.0.1:1'}, {'Sec-Fetch-Site': 'cross-site'},
    {'Authorization': 'Bearer forged'}, {'X-ML-Stack-Agent': 'worker'},
    {'X-ML-Stack-Token': 'forged'}])
def test_dev_pool_signin_rejects_foreign_or_agent_request(development, headers):
    assert local_post(development, {'pool': 'research'}, headers=headers)[0] in (400, 403)
    assert len(development.ui.sessions) == 0


def test_dev_pool_signin_rechecks_current_mode_and_setup(development):
    server = development
    assert server.call('/ui/session')[1]['local_pools']
    server.ui.settings.setup_done = False
    assert server.call('/ui/session')[1]['local_pools'] == []
    assert local_post(server, {'pool': 'research'})[0] == 400
    server.ui.settings.setup_done = True
    mint_cluster('workshop', server.keyfile, mode='prod')
    assert server.call('/ui/session')[1]['local_pools'] == []
    assert local_post(server, {'pool': 'research'})[0] == 400
    assert len(server.ui.sessions) == 0


def test_dev_pool_signin_remote_listener_cannot_list_or_signin(development):
    address = primary_ip()
    if not address or address.startswith('127.'):
        pytest.skip('no LAN address')
    status, result, _ = development.call('/ui/session', host=address)
    assert status in (401, 403) and not result.get('local_pools')
    assert local_post(development, {'pool': 'research'}, host=address)[0] in (401, 403)
    assert len(development.ui.sessions) == 0


def names(body):
    return sorted(row['name'] for row in body['peers'])


def machine(name, *clusters, is_self=False):
    return {'name': name, 'machine': name, 'port': 8770, 'host': '10.0.0.1', 'clusters': list(clusters),
            'device': {}, 'busy': False, 'queued': 0, 'slots': 1, 'free': 1, 'is_self': is_self,
            'base_url': f'http://{name}:8770'}


def test_the_chosen_development_pool_scopes_what_the_session_sees(development):
    import time
    server = development
    server.ui._peers = (time.time() + 600, [
        machine('studio', 'workshop', 'research', is_self=True), machine('lab', 'research'),
        machine('annex', 'workshop'), machine('hall', 'research', 'workshop')])
    everyone = ['annex', 'hall', 'lab', 'studio']
    status, anyone, _ = server.call('/ui/peers', cookie=server.ui.sessions.cookie_header(
        server.ui.sessions.open('setup')).split(';')[0])
    assert status == 200 and names(anyone) == everyone and anyone['group'] == 'workshop'
    cookie = local_post(server, {'pool': 'research'})[2]['Set-Cookie'].split(';')[0]
    status, scoped, _ = server.call('/ui/peers', cookie=cookie)
    assert status == 200 and names(scoped) == ['hall', 'lab', 'studio'] and scoped['group'] == 'research'
    fleet = server.call('/ui/fleet', cookie=cookie)[1]
    assert sorted(row['name'] for row in fleet['peers']) == ['hall', 'lab', 'studio'] and fleet['group'] == 'research'
    other = local_post(server, {'pool': 'workshop'})[2]['Set-Cookie'].split(';')[0]
    assert names(server.call('/ui/peers', cookie=other)[1]) == ['annex', 'hall', 'studio']
    assert names(server.call('/ui/peers', cookie=cookie)[1]) == ['hall', 'lab', 'studio']


@pytest.mark.slow
def test_browser_logout_then_existing_pool_click_signs_in_without_passphrase(development):
    playwright = pytest.importorskip('playwright.sync_api')
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f'http://127.0.0.1:{development.port}/ui#chat')
        page.get_by_role('button', name='Sign out', exact=True).click()
        choices = page.locator('#signin-pools')
        playwright.expect(choices.get_by_role('button', name='research', exact=True)).to_be_visible()
        playwright.expect(page.locator('#p')).to_be_hidden()
        playwright.expect(choices.get_by_role('button', name='production', exact=True)).to_have_count(0)
        choices.get_by_role('button', name='research', exact=True).click()
        playwright.expect(page.locator('#signin')).to_be_hidden()
        session = page.evaluate("async () => (await fetch('/ui/session', {headers:{'X-ML-Stack-UI':'1'}})).json()")
        assert session['signed_in'] and session['pool'] == 'research'
        assert memberships(development.keyfile)[0].group == 'workshop'
        browser.close()
