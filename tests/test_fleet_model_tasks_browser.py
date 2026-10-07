"""Model handoff, task drafts and specialist jobs exercised through the browser."""

import pytest
from playwright.sync_api import expect
from test_fleet_ui import Serving

pytestmark = pytest.mark.slow


@pytest.fixture
def journey_page(tmp_path, playwright):
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        yield served, page, errors
    finally:
        browser.close()
        served.close()


def test_task_review_draft_survives_refresh_and_duplicate_submit(journey_page):
    served, page, errors = journey_page
    task = {"id": "demo-task", "title": "Build an evaluator", "description": "Score held-out cases",
            "state": "review", "acceptance": ["Results include probabilities"], "reviews": [],
            "checkpoints": [], "proposal": {"artifacts": {}}}
    page.route("**/ui/tasks", lambda route: route.fulfill(json={"ok": True, "tasks": [task], "metrics": {}}))
    page.goto(f"http://127.0.0.1:{served.port}/ui/#tasks")
    page.get_by_role("button", name="Build an evaluator").click()
    page.get_by_text("Independently review this proposal", exact=True).click()
    reason = page.get_by_role("textbox", name="Independent review reason")
    reason.fill("Verified the held-out report.")
    reason.blur()
    task["checkpoints"] = [{"message": "Report submitted"}]
    page.evaluate("document.querySelector('tasks-view').load()")
    expect(reason).to_have_value("Verified the held-out report.")
    page.evaluate("""() => {
      const original = window.fleetModel.api;
      window.taskPosts = 0;
      window.fleetModel.api = (path, opts) => path === '/ui/tasks' && opts?.method === 'POST'
        ? (window.taskPosts++, new Promise(resolve => window.finishTask = resolve)) : original(path, opts);
    }""")
    page.get_by_role("button", name="Record independent review", exact=True).click()
    expect(page.get_by_role("button", name="Record independent review", exact=True)).to_be_disabled()
    page.evaluate("document.querySelector('tasks-view .task-detail form').dispatchEvent(new Event('submit',{cancelable:true}))")
    assert page.evaluate("window.taskPosts") == 1
    page.evaluate("window.finishTask({ok:false,status:403,error:'Independent reviewer required'})")
    expect(page.locator("tasks-view .status")).to_contain_text("Independent reviewer required")
    expect(page.get_by_role("button", name="Record independent review", exact=True)).to_be_enabled()
    assert not errors


def test_decider_fetch_validation_and_pending_action(journey_page):
    served, page, errors = journey_page
    page.goto(f"http://127.0.0.1:{served.port}/ui/#tools")
    page.get_by_label("Options, one per line", exact=True).fill("stop")
    page.get_by_role("button", name="Run operation", exact=True).click()
    expect(page.locator("tools-view #decision .status")).to_contain_text("at least two distinct options")
    page.get_by_label("Operation", exact=True).select_option("fetch")
    page.evaluate("""() => {
      const original = window.fleetModel.api;
      window.toolPosts = [];
      window.fleetModel.api = (path, opts) => path === '/ui/workspace/jobs' && opts?.method === 'POST'
        ? (window.toolPosts.push(JSON.parse(opts.body)), new Promise(resolve => window.finishTool = resolve))
        : original(path, opts);
    }""")
    page.get_by_role("button", name="Run operation", exact=True).click()
    expect(page.get_by_role("button", name="Run operation", exact=True)).to_be_disabled()
    page.evaluate("document.querySelector('tools-view').decide()")
    assert page.evaluate("window.toolPosts") == [{"command": "ml-stack-decide", "args": ["fetch", "--yes"], "name": "Decision fetch"}]
    page.evaluate("window.finishTool({ok:false,status:503,error:'Downloads are unavailable'})")
    expect(page.locator("tools-view #decision .status")).to_contain_text("Downloads are unavailable")
    expect(page.get_by_role("button", name="Run operation", exact=True)).to_be_enabled()
    page.get_by_text("Advanced options: command arguments", exact=True).click()
    page.get_by_label("Arguments as JSON array", exact=True).fill('{"bad":"shape"}')
    page.get_by_role("button", name="Review", exact=True).click()
    expect(page.locator("tools-view #decision .status")).to_contain_text("JSON array of strings")
    assert len(page.evaluate("window.toolPosts")) == 1
    assert not errors


def test_installed_model_target_and_details_survive_polling(journey_page):
    served, page, errors = journey_page
    model = {"id": "qwen-demo", "name": "Qwen3.8-demo.gguf", "path": "/tmp/Qwen3.8-demo.gguf",
             "repo": "", "family": "Qwen", "format": "gguf", "quantization": "Q4_K_XL", "kind": "text",
             "status": "installed", "servable": True, "size_bytes": 1000, "mtime": 1,
             "shards": 1, "is_complete": True, "files": []}
    page.route("**/ui/models", lambda route: route.fulfill(json={"library": [model], "here": [], "elsewhere": [], "getting": [], "unfinished": []}))
    page.route("**/ui/serving", lambda route: route.fulfill(json={"running": [], "can_serve": True}))
    page.goto(f"http://127.0.0.1:{served.port}/ui/#models")
    expect(page.get_by_role("button", name="Serve model", exact=True)).to_be_enabled()
    page.evaluate("document.querySelector('models-library').selectModel({path:'/tmp/Qwen3.8-demo.gguf',name:'Qwen3.8-demo.gguf'})")
    expect(page.locator('models-library article[data-selected="true"]')).to_contain_text("Qwen3.8-demo.gguf")
    page.get_by_text("Files and model details", exact=True).click()
    page.evaluate("document.querySelector('models-view').draw()")
    expect(page.locator("models-library article details")).to_have_attribute("open", "")
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert not errors


def test_job_detail_ignores_stale_selection_response(journey_page):
    served, page, errors = journey_page
    page.route("**/ui/workspace/jobs", lambda route: route.fulfill(json={"ok": True, "jobs": [], "capacity": {"running": [], "queued": 0, "slots": 1}}))
    page.goto(f"http://127.0.0.1:{served.port}/ui/#tools")
    page.evaluate("""() => {
      const jobs=document.querySelector('tools-view workspace-jobs'); clearInterval(jobs.timer);
      const original=window.fleetModel.api;
      window.fleetModel.api=(path,opts)=>path.startsWith('/ui/workspace/jobs/')
        ?new Promise(resolve=>{if(path.endsWith('/first'))window.finishFirst=resolve;else window.finishSecond=resolve;})
        :original(path,opts);
      jobs.selected='first';jobs.detail();jobs.selected='second';jobs.detail();
      window.finishSecond({ok:true,job:{id:'second'},metadata:{},metrics:[],log:'Second result'});
    }""")
    expect(page.locator("tools-view workspace-jobs pre")).to_contain_text("Second result")
    page.evaluate("window.finishFirst({ok:true,job:{id:'first'},metadata:{},metrics:[],log:'Stale first result'})")
    expect(page.locator("tools-view workspace-jobs pre")).not_to_contain_text("Stale first result")
    assert not errors
