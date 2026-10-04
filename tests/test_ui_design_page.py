"""Model browsing, loading feedback and benchmark controls in headless Chromium."""

import json
from urllib.parse import urlsplit

import pytest

from ml_stack.fleet.page import WEB, render
from ml_stack.ui import assets_dir

pytestmark = pytest.mark.slow


def route(request, state):
    path = urlsplit(request.request.url).path
    state["requests"].append(path)
    if path == "/":
        return request.fulfill(body=render(), content_type="text/html")
    if path.startswith("/ui/ml-ui/"):
        return request.fulfill(path=str(assets_dir() / path.rsplit("/", 1)[-1]))
    if path == "/ui/static/style.css":
        return request.fulfill(path=str(WEB / "style.css"))
    method = request.request.method
    if method in {"POST", "DELETE"}:
        state["posts"].append((path, request.request.post_data_json))
    body = {"ok": True}
    if path == "/ui/setup":
        body = {"needs_setup": False, "needs_password": False, "name": "local-test"}
    elif path == "/ui/models":
        body = {"free_gb": 120, "here": [{"name": "quill-27B.gguf", "size": 1000},
                 {"name": "quill-27B.draft.gguf", "size": 50}],
                "elsewhere": [{"name": "marrow-9B.gguf", "peers": ["remote-test"]}],
                "getting": [{"id": "download-a", "name": "another.gguf", "state": "getting",
                             "done": state["done"], "total": 100, "rate_bps": 10, "eta_seconds": 9}]}
    elif path == "/ui/serving":
        if method == "POST":
            if state["delay_serve"]:
                state["pending"] = request
                return
            if state["serve_error"]:
                body = {"error": state["serve_error"]}
            else:
                state["running"] = [{"models": ["quill-27B.gguf"], "port": 9999}]
        elif method == "DELETE":
            state["running"] = []
        body = {**body, "can_serve": True, "running": state["running"]}
    elif path == "/ui/models/popular":
        body = {"models": state["hub_models"] or [{"name": "Hub Qwen", "file": "hub.gguf", "ref": "hf:test/hub",
                             "family": "Qwen", "gb": 5, "what": "A small chat model"}],
                "families": ["Qwen"], "page": 0, "pages": 2}
    elif path == "/ui/fleet":
        body = {"group": "default", "models": ["quill-27B.gguf", "marrow-9B.gguf"],
                "peers": [{"name": "local-test", "is_self": True, "models": ["quill-27B.gguf"], "device": {}},
                          {"name": "remote-test", "models": ["marrow-9B.gguf"], "device": {}}],
                "bench": {"available": True}}
    elif path == "/ui/clusters":
        body = {"clusters": [{"group": "default"}]}
    elif path == "/ui/room":
        body = {"machine": {"supported": False}}
    elif path == "/ui/settings":
        body = {"name": "local-test", "group": "default", "settings": {"download_sources": "both"}}
    elif path == "/ui/fit.json":
        body = {"records": [], "room": 24 * 1024**3, "at_room": 24 * 1024**3,
                "name": "local-test", "steps": [8, 16, 24], "ladder": [2048, 4096, 8192], "vram_gb": [24]}
    elif path == "/ui/bench/sweep":
        body = {"pid": 1234}
    elif path == "/ui/chat" and method == "GET":
        body = {"models": [{"model": "quill-27B", "local": True}] if state["chat_available"] else []}
    elif path == "/ui/conversations" and method == "POST":
        body = {"id": "chat-test"}
    request.fulfill(body=json.dumps(body), content_type="application/json")


@pytest.fixture
def app():
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as play:
        browser = play.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.set_default_timeout(5000)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        state = {"requests": [], "posts": [], "running": [], "done": 10, "delay_serve": False,
                 "serve_error": "", "pending": None, "hub_models": None, "chat_available": False}

        page.route("http://design.test/**", lambda request: route(request, state))
        page.goto("http://design.test/")
        page.locator("#cluster:not([hidden])").wait_for()
        yield page, state, errors
        browser.close()


def models(page):
    page.get_by_role("link", name="Models", exact=True).click()
    page.locator("[data-model='quill-27B.gguf']").wait_for()


def test_network_filter_uses_authenticated_inventory_without_hub_calls(app):
    page, state, errors = app
    models(page)
    assert page.get_by_role("button", name="On network", exact=True).get_attribute("aria-pressed") == "true"
    assert page.locator("#browser-results").get_by_text("marrow-9B.gguf", exact=True).is_visible()
    assert "draft.gguf" not in page.locator("#browser-results").inner_text()
    page.get_by_role("searchbox", name="Search models", exact=True).fill("marrow")
    assert page.locator("#browser-results .model-row").count() == 1
    assert "/ui/models/popular" not in state["requests"]
    page.get_by_role("button", name="Copy here", exact=True).click()
    page.wait_for_function("document.querySelector('models-view').loading === false")
    assert ("/ui/models", {"name": "marrow-9B.gguf", "source": "", "draft": ""}) in state["posts"]
    assert not errors


def test_inventory_poll_preserves_hub_query_focus_filters_and_results(app):
    page, state, errors = app
    models(page)
    page.get_by_role("button", name="Hugging Face", exact=True).click()
    page.get_by_text("Hub Qwen", exact=True).wait_for()
    hunt = page.get_by_role("searchbox", name="Search models", exact=True)
    hunt.fill("qwen")
    page.wait_for_function("document.querySelector('model-browser').hubKey.includes('qwen')")
    hunt.focus()
    before = state["requests"].count("/ui/models/popular")
    page.evaluate("window.savedSearch = document.getElementById('hunt'); window.savedProgress = document.querySelector('#download-rows ml-progress')")
    state["done"] = 60
    page.evaluate("document.querySelector('models-view').draw()")
    page.wait_for_function("document.querySelector('#download-rows ml-progress').done === 60")
    assert hunt.input_value() == "qwen"
    assert hunt.evaluate("e => e === window.savedSearch && e === document.activeElement")
    assert page.locator("#download-rows ml-progress").evaluate("e => e === window.savedProgress")
    assert state["requests"].count("/ui/models/popular") == before
    assert page.get_by_text("Hub Qwen", exact=True).is_visible()
    assert not errors


@pytest.mark.parametrize("failed", [False, True])
def test_run_shows_loading_and_blocks_duplicate_posts_across_poll(app, failed):
    page, state, errors = app
    models(page)
    state["delay_serve"] = True
    row = page.locator("[data-model='quill-27B.gguf']")
    row.get_by_role("button", name="Run", exact=True).click()
    row.get_by_role("button", name="Loading…", exact=True).wait_for()
    assert row.get_by_role("button", name="Loading…", exact=True).is_disabled()
    page.evaluate("document.querySelector('models-view').draw()")
    page.evaluate("document.querySelector('models-view').serve('quill-27B.gguf')")
    assert len([path for path, _ in state["posts"] if path == "/ui/serving"]) == 1
    pending = state["pending"]
    assert pending is not None
    if failed:
        pending.fulfill(body=json.dumps({"error": "Not enough memory"}), content_type="application/json")
        page.get_by_text("Not enough memory", exact=True).wait_for()
        assert row.get_by_role("button", name="Run", exact=True).is_enabled()
    else:
        state["running"] = [{"models": ["quill-27B.gguf"], "port": 9999}]
        pending.fulfill(body=json.dumps({"ok": True}), content_type="application/json")
        page.get_by_text("quill-27B.gguf is ready. Open Chat to use it.", exact=True).wait_for()
        row.get_by_role("button", name="Stop", exact=True).wait_for()
    assert not errors


def test_capacity_lives_in_models_and_uses_q8_geometry(app):
    page, state, errors = app
    models(page)
    assert page.get_by_role("link", name="Fit", exact=True).count() == 0
    assert page.locator("fit-view").count() == 1
    assert "/ui/fit.json" not in state["requests"]
    page.locator("#model-capacity > summary").click()
    page.locator("#fit-controls select").first.wait_for()
    assert "/ui/fit.json" in state["requests"]
    assert page.evaluate("window.fleetModel.ctxRam(8192)") == "289 MB"
    assert not errors


def test_benchmark_has_labelled_selection_and_preserves_form_across_poll(app):
    page, state, errors = app
    page.locator("#cluster-sweep > summary").click()
    page.get_by_label("quill-27B.gguf", exact=False).check()
    page.get_by_label("Questions", exact=True).select_option("limit")
    page.get_by_label("Question limit", exact=True).fill("12")
    page.get_by_label("Run name", exact=False).fill("evening comparison")
    page.evaluate("window.savedRunName = document.getElementById('label'); document.querySelector('cluster-view').draw()")
    page.wait_for_function("document.getElementById('label') === window.savedRunName")
    assert page.get_by_label("Run name", exact=False).input_value() == "evening comparison"
    assert "ml-stack-bench" not in page.locator("#cluster-sweep").inner_text()
    page.get_by_role("button", name="Start benchmark", exact=True).click()
    page.get_by_text("Benchmark started. You can leave this screen while it runs.", exact=True).wait_for()
    assert ("/ui/bench/sweep", {"models": ["quill-27B.gguf"], "peers": ["local-test", "remote-test"],
                                "sample": 12, "label": "evening comparison"}) in state["posts"]
    assert not errors


def test_settings_schedule_and_chat_remain_reachable(app):
    page, _, errors = app
    page.get_by_role("link", name="Chat", exact=True).click()
    page.locator("#chat-none:not([hidden])").wait_for()
    page.get_by_role("link", name="Settings", exact=True).click()
    page.locator("#settings-sub:not(:empty)").wait_for()
    assert page.get_by_role("button", name="Save", exact=True).is_visible()
    assert page.get_by_text("When you need the machine", exact=True).is_visible()
    assert not errors


def test_quant_choice_updates_size_and_download_payload_without_extra_search(app):
    page, state, errors = app
    state["hub_models"] = [
        {"name": "Qwen quill 27B", "file": "Qwen-quill-27B-UD-Q4_K_XL.gguf", "family": "Qwen",
         "ref": "hf:test/Qwen-quill-27B-UD-Q4_K_XL.gguf", "download_gb": 17.6, "draft_download_gb": 1.4,
         "draft_ref": "hf:test/mtp.gguf", "recommended": True},
        {"name": "Qwen quill 27B", "file": "Qwen-quill-27B-UD-IQ4_XS.gguf", "family": "Qwen",
         "ref": "hf:test/Qwen-quill-27B-UD-IQ4_XS.gguf", "download_gb": 14.3, "draft_download_gb": 1.4,
         "draft_ref": "hf:test/mtp.gguf", "estimated_context": 131072},
        {"name": "Qwen quill 27B", "file": "Qwen-quill-27B-GSQ-RCO-IQ3_S.gguf", "family": "Qwen",
         "ref": "hf:test/Qwen-quill-27B-GSQ-RCO-IQ3_S.gguf", "bits_per_weight": 3.5},
    ]
    models(page)
    page.get_by_role("button", name="Hugging Face", exact=True).click()
    selector = page.get_by_role("combobox", name="Quantization for Qwen quill 27B", exact=True)
    selector.wait_for()
    assert page.locator("#browser-results .model-row").count() == 1
    assert selector.input_value().endswith("UD-Q4_K_XL.gguf")
    assert page.get_by_role("button", name="Download", exact=True).is_enabled()
    count = state["requests"].count("/ui/models/popular")
    selector.select_option("Qwen-quill-27B-UD-IQ4_XS.gguf")
    assert "14.3 GB weights + 1.4 GB MTP head · 15.7 GB total download" in page.locator("#browser-results").inner_text()
    page.evaluate("document.querySelector('models-view').draw()")
    assert selector.input_value().endswith("IQ4_XS.gguf")
    page.get_by_role("button", name="Download", exact=True).click()
    assert ("/ui/models", {"name": "Qwen-quill-27B-UD-IQ4_XS.gguf",
                         "source": "hf:test/Qwen-quill-27B-UD-IQ4_XS.gguf", "draft": "hf:test/mtp.gguf"}) in state["posts"]
    page.get_by_role("combobox", name="Quantization bits", exact=True).select_option("3")
    assert selector.locator("option").count() == 1
    assert "size unknown" in selector.inner_text()
    assert "3.5 bits per weight" in page.locator("#browser-results").inner_text()
    assert state["requests"].count("/ui/models/popular") == count
    assert not errors
