"""A browser a model drives cannot be pointed at this machine or its network."""

import pytest

from ml_stack.httpguard import Limits
from ml_stack.net import browserguard
from tests.net_site import Site

LOCAL = Limits(allow_hosts=frozenset({"127.0.0.1"}))


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost:8080/", "http://169.254.169.254/latest/", "http://10.0.0.1/",
    "http://[::1]/", "http://192.168.1.1/admin", "file:///etc/passwd", "ftp://example.org/x",
    "ws://127.0.0.1/socket", "chrome://settings", "http://user:pw@8.8.8.8/",
])
def test_requests_to_this_machine_and_its_network_are_not_allowed(url):
    assert browserguard.allowed(url) is False


@pytest.mark.parametrize("url", ["data:text/plain,hi", "blob:https://example.org/abc", "about:blank"])
def test_inline_content_reaches_nowhere_and_is_allowed(url):
    assert browserguard.allowed(url) is True


def test_a_public_address_is_allowed():
    assert browserguard.allowed("http://8.8.8.8/") is True


@pytest.mark.slow
def test_a_real_browser_is_stopped_from_every_request_the_guard_refuses():
    sync = pytest.importorskip("playwright.sync_api")
    with Site() as site, Site() as other:
        other.add("/secret.png", b"\x89PNG\r\n\x1a\n")
        site.add("/page", (
            "<html><body><p>hello</p>"
            f'<img src="http://127.0.0.2:{other.port}/secret.png">'
            f'<img src="http://localhost:{other.port}/secret.png">'
            '<img src="http://169.254.169.254/latest/meta-data/">'
            "</body></html>").encode(), headers={"Content-Type": "text/html"})
        with sync.sync_playwright() as play:
            try:
                browser = play.chromium.launch(headless=True)
            except sync.Error as exc:
                pytest.skip(f"no browser: {exc}")
            context = browser.new_context(accept_downloads=False)
            refused = browserguard.install(context, LOCAL)
            page = context.new_page()
            page.goto(f"{site.base}/page", wait_until="load")
            text = page.inner_text("body")
            browser.close()
        assert "hello" in text
        assert other.routes["/secret.png"].seen == []
        assert any("169.254.169.254" in url for url in refused)
        assert any(f"localhost:{other.port}" in url for url in refused)
