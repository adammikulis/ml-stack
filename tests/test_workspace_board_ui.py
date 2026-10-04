"""The ml-board element: what the folder holds, and the element driven in headless Chromium
against the real read-only route."""

from __future__ import annotations

import re
import time

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.ui import assets_dir
from ml_stack.workspace import boardroute, tokens


@pytest.fixture
def kit(monkeypatch, tmp_path):
    k = Kit(clean_env(monkeypatch, tmp_path))
    k.limits(sends_per_window=1000)
    k.tokens = {n: k.agent(n) for n in ("alice", "bob")}
    tokens.store(k.base, tokens.OWNER_FILE, k.owner)
    return k


def test_the_page_is_served_with_a_policy_that_forbids_eval_and_other_origins(kit, served):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", served, timeout=10)
    conn.request("GET", "/")
    policy = conn.getresponse().getheader("Content-Security-Policy")
    assert "default-src 'none'" in policy and "unsafe-eval" not in policy
    assert "connect-src 'self'" in policy


def test_the_element_is_loaded_by_ml_ui_and_builds_nothing_from_markup():
    source = (assets_dir() / "board.js").read_text(encoding="utf-8")
    assert 'import "./board.js"' in (assets_dir() / "ml-ui.js").read_text(encoding="utf-8")
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "eval(", "new Function",
                   "document.write", "localStorage", "sessionStorage", "window.open",
                   "target="):
        assert banned not in source, banned
    assert re.search(r'define\("ml-board"', source)
    assert source.count('method: "POST"') == 1 and source.count("/post`") == 1
    assert "PUT" not in source and "DELETE" not in source


@pytest.fixture
def served(kit):
    server = boardroute.serve(kit.ws)
    server.start()
    yield server.port
    server.stop()


@pytest.fixture(scope="module")
def browser():
    sync = pytest.importorskip("playwright.sync_api", reason="ml-stack[scrape]")
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
    page.goto(f"http://127.0.0.1:{served}/")
    page.wait_for_selector("ml-board", state="attached")
    page.wait_for_function("document.querySelector('ml-board')?.shadowRoot?.querySelector('nav button')")
    page.get_by_role("button", name="#ops").click()
    page.wait_for_function("document.querySelector('ml-board')?.shadowRoot?.querySelector('.row')")
    row = page.evaluate("document.querySelector('ml-board')?.shadowRoot?.querySelector('.row .subject').textContent")
    assert row == "<b>bold</b>"
    page.evaluate("document.querySelector('ml-board')?.shadowRoot?.querySelector('.row').click()")
    page.wait_for_function("document.querySelector('ml-board')?.shadowRoot?.querySelector('.msg pre')")
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
    page.wait_for_function("document.querySelector('ml-board')?.shadowRoot?.querySelector('.msg pre')"
                           "?.textContent.includes('a dm')")
    assert "a dm <i>x</i>" in page.evaluate("document.querySelector('ml-board')?.shadowRoot?.querySelector('.msg pre').textContent")
    page.close()


@pytest.mark.slow
def test_the_live_feed_picks_up_a_new_message_and_backs_off_when_the_route_fails(kit, served, browser):
    ws, t = kit.ws, kit.tokens
    ws.board.create(t["alice"], "#ops")
    page = browser.new_context(bypass_csp=True).new_page()
    page.goto(f"http://127.0.0.1:{served}/")
    page.wait_for_function("document.querySelector('ml-board')?.shadowRoot?.querySelector('nav button')")
    page.get_by_role("button", name="#ops").click()
    page.wait_for_timeout(500)
    began = time.monotonic()
    ws.send(t["alice"], "#ops", "note", "later", subject="arrives by push")
    page.wait_for_function(
        "document.querySelector('ml-board')?.shadowRoot?.querySelector('.row .subject')?.textContent"
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
    page.goto(f"http://127.0.0.1:{served}/")
    page.wait_for_function("document.querySelector('ml-board')?.shadowRoot?.querySelector('nav button')")
    page.get_by_role("button", name="#ops").click()
    page.wait_for_function("document.querySelector('ml-board')?.shadowRoot?.querySelector('.composer textarea')")
    page.evaluate("""() => {
      const root = document.querySelector('ml-board').shadowRoot;
      root.querySelector('.composer input').value = 'from the page';
      const box = root.querySelector('.composer textarea');
      box.value = '<img src=x onerror="window.__pwned=1"> hello agents';
      box.dispatchEvent(new Event('input'));
      root.querySelector('.composer button').click();
    }""")
    page.wait_for_function(
        "document.querySelector('ml-board')?.shadowRoot?.querySelector('.row .subject')?.textContent === 'from the page'")
    mine = [m for m in ws.board.ui_read(kit.owner, "#ops")["messages"] if m["from"] == "owner"]
    assert [m["body"] for m in mine] == ['<img src=x onerror="window.__pwned=1"> hello agents']
    seen = ws.board.read(t["alice"], "#ops")
    assert seen[-1]["text"].startswith("<untrusted") and seen[-1]["trust"] == "human"
    assert page.evaluate("window.__pwned ?? null") is None and errors == []
    assert [m for m, _ in posts] == ["POST"] and posts[0][1].endswith("/board/post")
    page.evaluate("document.querySelector('ml-board').readonly = true")
    page.wait_for_function("!document.querySelector('ml-board').shadowRoot.querySelector('.composer')")
    page.close()
