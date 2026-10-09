"""History separates earned credits, independent reputation and recorded compute usage."""

import pytest
from test_fleet_ui import Serving

pytestmark = pytest.mark.slow


@pytest.fixture
def standings():
    return {"economy_mode": "free", "team": [{
        "agent": "scout", "members": ["worker-a", "worker-b"], "verified_tasks": 2, "score": 2,
        "economy": {"balance": 35, "earned": 35, "spent": 0,
                    "completion_credits": 20, "quality_credits": 15,
                    "usage": {"tasks": 1, "recorded": True, "tokens_in": 120,
                              "tokens_out": 40, "wall_seconds": 2.5}},
        "work_reputation": {"state": "reviewed", "quality": 70, "reliability": 80,
                            "samples": 1, "confidence": 0.33},
        "evidence": [{"id": "award-1", "task": 42, "verifier": "lead",
                      "verified_at": 1234, "artifacts": {"native.json": "abc123"},
                      "award": {"base": 10, "quality_bonus": 15, "total": 25,
                                "quality": [{"kind": "impact", "reason": "<img src=x> clear improvement",
                                             "checks": ["native regression"], "artifacts": ["native.json"]}]},
                      "review": {"quality": 90, "reliability": 95, "reason": "Independent replay passed"}}]
    }, {"agent": "builder", "verified_tasks": 0, "score": 0,
        "economy": {"balance": 0, "earned": 0, "completion_credits": 0, "quality_credits": 0,
                    "usage": {"tasks": 0, "recorded": False}},
        "work_reputation": {"state": "reviewed", "quality": 50, "reliability": 67, "samples": 0,
                            "quality_samples": 0, "reliability_samples": 1,
                            "outcomes": {"accepted": 1, "rejected": 0, "blocked_infrastructure": 2}},
        "evidence": []}]}


def test_history_credits_without_recent_actions_and_review_evidence(tmp_path, playwright, standings):
    from playwright.sync_api import expect

    server = Serving(tmp_path)
    server.ui.settings.setup_done = True

    try:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page(viewport={"width": 390, "height": 844})
            page.route("**/ui/history/events", lambda route: route.fulfill(json={"events": []}))
            page.route("**/ui/work-reputation/standings", lambda route: route.fulfill(json=standings))
            page.goto(f"http://127.0.0.1:{server.port}/ui/#history")
            viewer = page.locator("history-view")
            expect(viewer.get_by_text("Runs are free.", exact=True)).to_be_visible()
            viewer.get_by_text("scout · 35 credits", exact=True).click()
            expect(viewer.locator("#history-credits > details").filter(has_text="scout · 35 credits").get_by_text("Quality bonus credits", exact=True)).to_be_visible()
            expect(viewer.get_by_text("1 independent reviews", exact=False)).to_be_visible()
            expect(viewer.get_by_text("120 input tokens", exact=False)).to_be_visible()
            viewer.get_by_text("Task 42 · verified by lead", exact=True).click()
            expect(viewer.get_by_text("25 credits earned · 10 completion + 15 quality bonus", exact=True)).to_be_visible()
            expect(viewer.get_by_text("Independent replay passed", exact=True)).to_be_visible()
            expect(viewer.get_by_text("abc123", exact=False)).to_be_visible()
            assert viewer.locator("img").count() == 0
            viewer.locator("#history-refresh").click()
            expect(viewer.get_by_text("Independent replay passed", exact=True)).to_be_visible()
            page.unroute("**/ui/history/events")
            actions = [{"id": "run-a", "ts": 1234, "kind": "agent.run", "actor": "worker-a", "subject": "Model A run", "meta": {"model": "model-a"}},
                       {"id": "run-b", "ts": 1235, "kind": "agent.run", "actor": "worker-b", "subject": "Model B run", "meta": {"model": "model-b"}}]
            page.route("**/ui/history/events", lambda route: route.fulfill(json={"events": actions}))
            viewer.locator("#history-refresh").click()
            viewer.locator("#history-agent").select_option("scout")
            expect(viewer.get_by_text("Model A run", exact=True)).to_be_visible()
            expect(viewer.get_by_text("Model B run", exact=True)).to_be_visible()
            expect(viewer.get_by_text("scout · 35 credits", exact=True)).to_have_count(1)
            viewer.locator("#history-agent").select_option("builder")
            viewer.get_by_text("builder · 0 credits", exact=True).click()
            expect(viewer.get_by_text("Not yet rated", exact=True)).to_be_visible()
            expect(viewer.get_by_text("1 accepted · 0 rejected · 2 infrastructure blockages", exact=True)).to_be_visible()
            expect(viewer.get_by_text("67", exact=True)).to_be_visible()
            expect(viewer.get_by_text("Recorded compute usage", exact=False)).to_have_count(0)
            expect(viewer.get_by_text("scout · 35 credits", exact=True)).to_have_count(0)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path="/private/tmp/ml-stack-history-credit-mobile.png", full_page=True)
    finally:
        server.close()


def test_history_groups_family_credits_across_devices_without_per_model_balances(tmp_path, playwright, standings):
    from playwright.sync_api import expect

    row = standings['team'][0]
    row.update(agent='family-qwen', label='Qwen', family_id='qwen', account_type='model_family',
               devices=['a' * 64, 'b' * 64])
    server = Serving(tmp_path)
    server.ui.settings.setup_done = True
    try:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page()
            page.route('**/ui/history/events', lambda route: route.fulfill(json={'events': []}))
            page.route('**/ui/work-reputation/standings', lambda route: route.fulfill(json=standings))
            page.goto(f'http://127.0.0.1:{server.port}/ui/#history')
            viewer = page.locator('history-view')
            viewer.get_by_text('Qwen · 35 credits', exact=True).click()
            expect(viewer.get_by_text('Account identity: family-qwen · Model family across devices')).to_be_visible()
            viewer.get_by_text('2 contributing devices', exact=True).click()
            expect(viewer.get_by_text('a' * 64, exact=True)).to_be_visible()
            expect(viewer.get_by_text('Workers: worker-a, worker-b', exact=True)).to_be_visible()
            assert viewer.get_by_text('Qwen · 35 credits', exact=True).count() == 1
    finally:
        server.close()
