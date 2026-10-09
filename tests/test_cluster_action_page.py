"""Cluster setup labels, required names and explicit join/create requests."""

import json

import pytest

from poolhouse.fleet.page import WEB, render
from poolhouse.ui import assets_dir

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("action", ["join", "create", "missing-name", "join-error", "current-group"])
def test_cluster_setup_sends_only_the_selected_action(action):
    pw = pytest.importorskip("playwright.sync_api")
    posted = []
    rows = [{"group": "lab"}] if action == "current-group" else []

    def route(request):
        path = request.request.url.split("http://cluster.test", 1)[-1]
        if path == "/":
            return request.fulfill(body=render(), content_type="text/html")
        if path.startswith("/ui/ml-ui/"):
            return request.fulfill(path=str(assets_dir() / path.rsplit("/", 1)[-1]))
        if path == "/ui/static/style.css":
            return request.fulfill(path=str(WEB / "style.css"))
        body, status = {"ok": True}, 200
        if path == "/ui/setup":
            body = {"needs_setup": True, "name": "test-machine", "cluster_mode": "prod"}
        elif path == "/ui/clusters":
            body = {"clusters": rows}
        elif path == "/ui/setup/join":
            posted.append(request.request.post_data_json)
            if action == "join-error":
                body, status = {"error": "No machine in this cluster answered."}, 400
            else:
                rows[:] = [{"group": posted[-1]["group"]}]
        request.fulfill(status=status, body=json.dumps(body), content_type="application/json")

    with pw.sync_playwright() as play:
        browser = play.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(5000)
        page.route("http://cluster.test/**", route)
        page.goto("http://cluster.test/")
        screen = page.locator("#first-run")
        screen.get_by_label("Mode", exact=True).select_option("prod")
        screen.get_by_role("button", name="Continue", exact=True).click()
        name = screen.get_by_label("Cluster name (required)", exact=True)
        assert name.input_value() == ("lab" if action == "current-group" else "default")
        mode = screen.get_by_label("What would you like to do?", exact=True)
        assert mode.input_value() == "join"
        words = screen.get_by_label("Cluster passphrase", exact=True)
        words.fill("quince larch marlow")
        if action == "missing-name":
            name.fill("")
            assert screen.get_by_role("button", name="Join existing cluster", exact=True).is_disabled()
            assert posted == []
        else:
            if action == "create":
                mode.select_option("create")
                name.fill("new-lab")
            screen.get_by_role("button", name="Create new cluster" if action == "create"
                               else "Join existing cluster", exact=True).click()
            screen.get_by_role("status").get_by_text("No machine" if action == "join-error"
                else "Created cluster" if action == "create" else "Joined cluster", exact=False).wait_for()
            assert posted == [{"group": "new-lab" if action == "create" else
                               "lab" if action == "current-group" else "default",
                               "passphrase": "quince larch marlow",
                               "mode": "create" if action == "create" else "join", "cluster_mode": "prod"}]
        browser.close()
