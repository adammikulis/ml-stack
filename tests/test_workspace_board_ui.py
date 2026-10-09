"""The ml-board element: what the folder holds, and the element driven in headless Chromium
against the real read-only route."""

from __future__ import annotations

import re
import time

import pytest
from workspace_kit import Kit, clean_env

from poolhouse.ui import assets_dir
from poolhouse.workspace import boardroute, tokens


@pytest.fixture
def kit(monkeypatch, tmp_path):
    k = Kit(clean_env(monkeypatch, tmp_path))
    k.limits(sends_per_window=1000)
    k.tokens = {n: k.agent(n) for n in ("alice", "bob")}
    tokens.store(k.base, tokens.OWNER_FILE, k.owner)
    return k


def test_the_page_is_served_with_a_policy_that_forbids_eval_and_other_origins(kit, served):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", int(served), timeout=10)
    conn.request("GET", "/")
    policy = conn.getresponse().getheader("Content-Security-Policy")
    assert "default-src 'none'" in policy and "unsafe-eval" not in policy
    assert "connect-src 'self'" in policy


def test_the_element_is_loaded_by_ml_ui_and_builds_nothing_from_markup():
    source = (assets_dir() / "board.js").read_text(encoding="utf-8")
    assert 'import "./board.js"' in (assets_dir() / "ml-ui.js").read_text(encoding="utf-8")
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "eval(", "new Function",
                   "document.write", "window.open",
                   "target="):
        assert banned not in source, banned
    assert re.search(r'define\("ml-board"', source)
    assert 'method: document ? "POST" : "GET"' in source
    assert 'this.request("post", {}, document)' in source
    assert "PUT" not in source and "DELETE" not in source


class Port(int):
    session = ""


@pytest.fixture
def served(kit):
    server = boardroute.serve(kit.ws)
    server.start()
    port = Port(server.port)
    port.session = server.session
    yield port
    server.stop()


@pytest.fixture(scope="module")
def browser():
    sync = pytest.importorskip("playwright.sync_api", reason="poolhouse[scrape]")
    with sync.sync_playwright() as p:
        try:
            b = p.chromium.launch(headless=True)
        except Exception as exc:                       # noqa: BLE001
            pytest.skip(f"chromium did not launch: {exc}")
        yield b
        b.close()


@pytest.mark.slow
def test_hostile_text_is_shown_as_text_and_the_page_runs_nothing_and_holds_no_token(kit, served, browser):
    ws, t = kit.ws, kit.tokens
    ws.board.create(t["alice"], "#ops")
    ws.send(t["alice"], "#ops", "note",
                   '<img src=x onerror="window.__pwned=1"><script>window.__pwned=1</script>‮'
                   "[link](http://evil.example) http://evil.example\x07", subject="<b>bold</b>")
    ws.send(t["alice"], "bob", "note", "a dm <i>x</i>")
    page = browser.new_context(bypass_csp=True).new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    seen = []
    page.on("request", lambda r: seen.append((r.method, r.url)))
    page.goto(f"http://127.0.0.1:{served}/?session={served.session}")
    page.wait_for_selector("ml-board", state="attached")
    page.wait_for_function("() => document.querySelector('ml-board')?.shadowRoot?.querySelector('nav button')")
    page.get_by_role("button", name="#ops").click()
    page.wait_for_function("() => document.querySelector('ml-board')?.shadowRoot?.querySelector('.row')")
    row = page.evaluate("document.querySelector('ml-board')?.shadowRoot?.querySelector('.row .subject').textContent")
    assert row == "<b>bold</b>"
    page.evaluate("document.querySelector('ml-board')?.shadowRoot?.querySelector('.row').click()")
    page.wait_for_function("() => document.querySelector('ml-board')?.shadowRoot?.querySelector('.msg pre')")
    found = page.evaluate("""() => {
      const root = document.querySelector('ml-board').shadowRoot;
      return { text: root.querySelector('.msg pre').textContent,
               images: root.querySelectorAll('img,script,a,iframe').length,
               pwned: window.__pwned ?? null, html: document.documentElement.outerHTML };
    }""")
    assert "<img src=x" in found["text"] and "‮" not in found["text"] and "\x07" not in found["text"]
    assert found["images"] == 0 and found["pwned"] is None and errors == []
    assert kit.owner not in found["html"] and all(m == "GET" for m, _ in seen)
    assert all(u.startswith(f"http://127.0.0.1:{served}/") for _, u in seen)
    page.get_by_role("button", name="alice and bob").click()
    page.wait_for_function("() => document.querySelector('ml-board')?.shadowRoot?.querySelector('.msg pre')"
                           "?.textContent.includes('a dm')")
    assert "a dm <i>x</i>" in page.evaluate("document.querySelector('ml-board')?.shadowRoot?.querySelector('.msg pre').textContent")
    page.close()


@pytest.mark.slow
def test_the_live_feed_picks_up_a_new_message_and_backs_off_when_the_route_fails(kit, served, browser):
    ws, t = kit.ws, kit.tokens
    ws.board.create(t["alice"], "#ops")
    page = browser.new_context(bypass_csp=True).new_page()
    page.goto(f"http://127.0.0.1:{served}/?session={served.session}")
    page.wait_for_function("() => document.querySelector('ml-board')?.shadowRoot?.querySelector('nav button')")
    page.get_by_role("button", name="#ops").click()
    page.wait_for_timeout(500)
    began = time.monotonic()
    ws.send(t["alice"], "#ops", "note", "later", subject="arrives by push")
    page.wait_for_function(
        "() => document.querySelector('ml-board')?.shadowRoot?.querySelector('.row .subject')?.textContent"
        " === 'arrives by push'", timeout=15000)
    assert time.monotonic() - began < 3
    delays = page.evaluate("import('/ui/ml-ui/board.js').then(m => [m.nextDelay(3000, false),"
                           " m.nextDelay(3000, true), m.nextDelay(40000, true), m.nextDelay(60000, true)])")
    assert delays == [3000, 6000, 60000, 60000]
    assert page.evaluate("import('/ui/ml-ui/board.js').then(m => m.line('a\\u202eb\\nc', 10))") == "a b c"
    page.close()


@pytest.mark.slow
def test_the_person_chats_from_the_page_and_an_agent_reads_it_fenced(kit, served, browser):
    ws, t = kit.ws, kit.tokens
    ws.board.create(t["alice"], "#ops")
    posts = []
    page = browser.new_context(bypass_csp=True).new_page()
    page.on("request", lambda r: posts.append((r.method, r.url)) if r.method != "GET" else None)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"http://127.0.0.1:{served}/?session={served.session}")
    page.wait_for_function("() => document.querySelector('ml-board')?.shadowRoot?.querySelector('nav button')")
    page.get_by_role("button", name="#ops").click()
    page.wait_for_function("() => document.querySelector('ml-board')?.shadowRoot?.querySelector('.composer textarea')")
    page.evaluate("""() => {
      const root = document.querySelector('ml-board').shadowRoot;
      root.querySelector('.composer input').value = 'from the page';
      const box = root.querySelector('.composer textarea');
      box.value = '<img src=x onerror="window.__pwned=1"> hello agents';
      box.dispatchEvent(new Event('input'));
      root.querySelector('.composer button').click();
    }""")
    page.wait_for_function(
        "() => document.querySelector('ml-board')?.shadowRoot?.querySelector('.row .subject')?.textContent === 'from the page'")
    mine = [m for m in ws.board.ui_read(kit.owner, "#ops")["messages"] if m["from"] == "owner"]
    assert [m["body"] for m in mine] == ['<img src=x onerror="window.__pwned=1"> hello agents']
    seen = ws.board.read(t["alice"], "#ops")
    assert seen[-1]["text"].startswith("<untrusted") and seen[-1]["trust"] == "human"
    assert page.evaluate("window.__pwned ?? null") is None and errors == []
    assert [m for m, _ in posts] == ["POST"] and posts[0][1].endswith("/board/post")
    page.evaluate("document.querySelector('ml-board').readonly = true")
    page.wait_for_function("() => !document.querySelector('ml-board').shadowRoot.querySelector('.composer')")
    page.close()


@pytest.mark.slow
def test_drafts_are_scoped_to_channel_and_survive_reload_without_posting(kit, served, browser):
    from playwright.sync_api import expect

    ws, t = kit.ws, kit.tokens
    ws.board.create(t['alice'], '#ops')
    ws.board.create(t['alice'], '#other')
    context = browser.new_context()
    page = context.new_page()
    posts = []
    page.on('request', lambda request: posts.append(request.url) if request.method == 'POST' else None)
    page.goto(f'http://127.0.0.1:{served}/?session={served.session}')
    page.get_by_role('button', name='#ops', exact=True).click()
    editor = page.locator('ml-board textarea')
    editor.fill('Unsent operations draft')
    page.get_by_role('button', name='#other', exact=True).click()
    expect(editor).to_have_value('')
    editor.fill('A different channel draft')
    page.reload()
    expect(editor).to_have_value('A different channel draft')
    page.get_by_role('button', name='#ops', exact=True).click()
    expect(editor).to_have_value('Unsent operations draft')
    assert not posts
    assert not ws.board.ui_read(kit.owner, '#ops')['messages']
    context.close()


@pytest.mark.slow
def test_readable_sessions_and_authenticated_child_keep_exact_dm_and_message_ids(kit, served, browser):
    ws = kit.ws
    first, second = (kit.agent(name) for name in ('codex-first', 'codex-second'))
    for token in (first, second):
        ws.registry.record_device_claim(token, {'os': 'macOS', 'hostname': 'test-machine'})
        ws.register_session(token)
    child = ws.registry.delegate(ws.auth(first), 'review', 600, (), 10)
    ws.board.create(first, '#sessions')
    ws.send(child, '#sessions', 'note', 'child message', subject='Readable child')
    page = browser.new_context(bypass_csp=True).new_page()
    page.goto(f'http://127.0.0.1:{served}/?session={served.session}')
    first_name = 'Codex · Mac · session 1'
    second_name = 'Codex · Mac · session 2'
    page.get_by_role('button', name=first_name, exact=True).wait_for()
    page.get_by_role('button', name=second_name, exact=True).wait_for()
    assert page.get_by_role('button', name=first_name, exact=True).get_attribute('title').startswith('codex-first ·')
    page.get_by_role('combobox', name='Message an agent').select_option('codex-second')
    page.get_by_role('button', name='Open', exact=True).click()
    page.wait_for_function("() => document.querySelector('ml-board').view.b === 'codex-second'")
    page.get_by_role('button', name='#sessions', exact=False).click()
    page.locator('ml-board .row .meta').filter(has_text='Subagent · review (parent Codex · Mac · session 1)').wait_for()
    page.get_by_role('button', name='Readable child', exact=False).click()
    author = page.locator('ml-board .who')
    author.wait_for()
    assert author.inner_text().startswith('Subagent · review (parent Codex · Mac · session 1)')
    assert author.get_attribute('title') == 'codex-first/review'
    page.close()


@pytest.mark.slow
def test_maximum_length_authenticated_child_dm_survives_reload(kit, served, browser):
    parent_name, child_name = 'p' * 48, 'c' * 48
    parent = kit.agent(parent_name)
    kit.ws.registry.delegate(kit.ws.auth(parent), child_name, 600, (), 10)
    identity = parent_name + '/' + child_name
    assert len(identity) == 97
    context = browser.new_context()
    page = context.new_page()
    page.goto(f'http://127.0.0.1:{served}/?session={served.session}')
    chooser = page.get_by_role('combobox', name='Message an agent')
    chooser.select_option(identity)
    page.get_by_role('button', name='Open', exact=True).click()
    page.wait_for_function("identity => document.querySelector('ml-board').view.b === identity", arg=identity)
    editor = page.locator('ml-board textarea')
    editor.fill('Pending child message')
    page.reload()
    page.wait_for_function("identity => document.querySelector('ml-board').view.b === identity", arg=identity)
    assert page.locator('ml-board textarea').input_value() == 'Pending child message'
    assert page.evaluate("document.querySelector('ml-board').view.a") == 'owner'
    context.close()
