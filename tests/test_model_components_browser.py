"""Optional model components selected through the fleet interface."""
from unittest.mock import Mock

import pytest
import test_fleet_page as fleet_page

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page
pytestmark = pytest.mark.slow


def test_download_component_choices_and_installed_component_action(joined, open_page, monkeypatch):
    from playwright.sync_api import expect

    from ml_stack.fleet import catalogue, model_components, routes
    from ml_stack.fleet.models import Getting

    pick = catalogue.Suggestion("Qwen Example 2B", "hf:publisher/example/Example-Q4_K_M.gguf", 1,
                               "Fixture model", draft_ref="hf:publisher/example/mtp-Example.gguf")
    monkeypatch.setattr(catalogue, "popular", lambda *args, **kwargs: [pick])
    monkeypatch.setattr(catalogue, "families", lambda *args, **kwargs: [pick.family])
    monkeypatch.setattr(catalogue, "how_many", lambda *args, **kwargs: 1)
    monkeypatch.setattr(routes, "choices", lambda **kwargs: {"ok": True, "models": []})
    offers = [{"kind": "mtp", "status": "available", "packaging": "separate", "name": "mtp-Example.gguf",
               "ref": pick.draft_ref, "download_bytes": 100, "size_bytes": 100, "error": ""},
              {"kind": "vision", "status": "available", "packaging": "separate", "name": "mmproj-Example.gguf",
               "ref": "hf:publisher/example/mmproj-Example.gguf", "download_bytes": 200, "size_bytes": 200, "error": ""}]
    monkeypatch.setattr(model_components, "catalogue", lambda *args, **kwargs: {"components": offers})
    joined.ui.settings.download_sources = "internet"
    downloads = Mock()
    downloads.active.return_value = []
    downloads.start.return_value = Getting("fixture", pick.file)
    joined.ui.downloads = downloads
    page, errors = open_page(joined, cookie=joined.cookie, path="/ui/#models")
    page.get_by_role("button", name="Hugging Face", exact=True).click()
    page.get_by_role("button", name="Download", exact=True).click()
    mtp = page.get_by_label("Multi-token prediction", exact=False)
    vision = page.get_by_label("Vision", exact=False)
    expect(mtp).to_be_checked()
    mtp.uncheck()
    vision.check()
    page.get_by_role("button", name="Start download", exact=True).click()
    page.wait_for_function("document.querySelector('#models-note').textContent === ''")
    assert downloads.start.call_args.kwargs["components"] == [offers[1]]
    base = joined.files / pick.file
    base.write_bytes(b"fixture")
    monkeypatch.setattr("ml_stack.fleet.models.MIN_SIZE", 0)
    monkeypatch.setattr(joined.ui.models, "library", lambda: [{
        "id": "fixture", "name": "Example", "path": str(base), "family": "Example", "format": "gguf",
        "status": "ready", "servable": True, "shards": 1, "is_complete": True, "files": [],
        "size_bytes": 7, "mtime": 0}])
    downloads.start.reset_mock()
    page.reload()
    page.locator("models-library details summary").click()
    page.get_by_role("button", name="Optional MTP and vision components", exact=True).click()
    page.get_by_role("button", name="Install Vision", exact=True).click()
    expect(page.locator("models-library")).to_contain_text("Download started")
    assert downloads.start.call_args.kwargs["components"] == [offers[1]]
    assert not errors
