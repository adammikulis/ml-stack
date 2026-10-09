"""Optional model downloads in the first-run screen."""

import json

import pytest

from ml_stack.fleet.page import WEB, render
from ml_stack.ui import assets_dir

pytestmark = pytest.mark.slow


def mock_route(action, posted, models):
    def route(request):
        path = request.request.url.split("http://setup.test", 1)[-1]
        if path == "/":
            return request.fulfill(body=render(), content_type="text/html")
        if path.startswith("/ui/ml-ui/"):
            asset = assets_dir() / path.rsplit("/", 1)[-1]
            return request.fulfill(path=str(asset))
        if path == "/ui/static/style.css":
            return request.fulfill(path=str(WEB / "style.css"))
        body = {"ok": True}
        code = 200
        if request.request.method == "POST":
            posted.append((path, request.request.post_data_json))
        if path == "/ui/setup":
            body = {"needs_setup": True, "name": "test-machine"}
        elif path == "/ui/models/startup":
            body = {"models": models, "context": 8192,
                    "memory": {"pool": "VRAM", "capacity_gb": 24, "free_gb": 22.5}}
        elif path == "/ui/models":
            if request.request.method == "POST":
                code = 400 if action == "error" else 202
                body = {"error": "Download refused"} if action == "error" else {
                    "id": "download-test", "name": "large.gguf", "state": "getting"}
            else:
                body = {"getting": [{"id": "download-test", "name": "large.gguf",
                                     "state": action if action in {"failed", "done"} else "getting",
                                     "done": 12, "total": 100,
                                     "error": "Disk full" if action == "failed" else ""}]}
        request.fulfill(status=code, body=json.dumps(body), content_type="application/json")

    return route


@pytest.mark.parametrize("action", ["install", "skip", "later", "error", "failed", "smaller", "done"])
def test_first_run_model_choice_requires_opt_in(action):
    pw = pytest.importorskip("playwright.sync_api")
    posted = []
    models = [
        {"name": "Qwen large", "file": "large.gguf", "ref": "hf:test/large",
         "gb": 16.7, "draft_gb": 0.8, "draft_ref": "hf:test/mtp",
         "recommended": True, "what": "General chat"},
        {"name": "Qwen small", "file": "small.gguf", "ref": "hf:test/small",
         "gb": 5, "draft_gb": 0, "draft_ref": "", "recommended": False, "what": "Quick chat"},
    ]

    route = mock_route(action, posted, models)

    with pw.sync_playwright() as play:
        browser = play.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(5000)
        page.route("http://setup.test/**", route)
        page.goto("http://setup.test/")
        for heading in ("Set up this machine", "Clusters", "What should this machine do?", "When should it start?"):
            page.get_by_role("heading", name=heading, exact=True).wait_for()
            page.get_by_role("button", name="Continue", exact=True).click()
        page.get_by_role("heading", name="Where should downloads come from?").wait_for()
        assert not any(path in {"/ui/models", "/ui/serving/install"} for path, _ in posted)
        save = page.get_by_role("button", name="Save and continue")
        assert save.is_disabled()
        page.locator("#source-both").check()
        save.click()
        choice = page.locator("#startup-model-choice")
        assert choice.input_value() == "large.gguf"
        assert "24.0 GiB capacity, 22.5 GiB free" in page.locator("#startup-model-memory").inner_text()
        assert "16.7 GB download + 0.8 GB MTP draft" in page.locator("#startup-model-detail").inner_text()
        opt = page.locator("#startup-model-download")
        assert not opt.is_checked()
        assert not any(path == "/ui/models" for path, _ in posted)
        if action not in {"skip", "later"}:
            opt.check()
        if action == "smaller":
            choice.select_option("small.gguf")
        page.get_by_role("button", name="Not now" if action == "later" else "Install and continue").click()
        page.get_by_role("heading", name="When you are not using it").wait_for()
        if action == "error":
            page.locator("#startup-model-status").get_by_text("Download refused", exact=False).wait_for()
        downloads = [body for path, body in posted if path == "/ui/models"]
        assert len(downloads) == (0 if action in {"skip", "later"} else 1)
        if downloads:
            preference = next(i for i, (path, body) in enumerate(posted)
                              if path == "/ui/setup/prefs" and body.get("download_sources") == "both")
            assert preference < next(i for i, (path, _) in enumerate(posted) if path == "/ui/models")
            assert downloads[0] == ({"name": "small.gguf", "source": "hf:test/small", "mtp": False, "vision": False}
                                    if action == "smaller" else
                                    {"name": "large.gguf", "source": "hf:test/large", "mtp": True, "vision": False})
        assert not any(path == "/ui/serving" for path, _ in posted)
        if action in {"install", "failed", "smaller", "done"}:
            progress = page.locator("#startup-model-status ml-progress")
            progress.wait_for()
            page.wait_for_function("() => document.querySelector('#startup-model-status ml-progress').total === 100")
            assert progress.evaluate("e => e.state") == (
                action if action in {"failed", "done"} else "running")
            if action == "failed":
                progress.get_by_text("Disk full", exact=True).wait_for()
            if action == "done":
                progress.get_by_text("Downloaded. Open Models to run it.", exact=True).wait_for()
        browser.close()
