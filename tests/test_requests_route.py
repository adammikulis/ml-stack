"""The Requests page over a real local server: who may open it, and every way a POST that answers
a request can be refused (Host, Origin, fetch metadata, content type, token, size, rate, replay,
fingerprint). Text is data: the page holds no inline script and sends a content security policy."""

from __future__ import annotations

import http.client
import json
import re
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from ml_stack import person, requests
from ml_stack.inbox import route
from ml_stack.ui import assets_dir
from tests.requests_support import ask, make_server, no_markers

__all__ = ["no_markers"]
pytestmark = pytest.mark.usefixtures("no_markers")
KEY = bytes(range(32))


class Live:
    """A running server and a browser session on it."""

    def __init__(self, tmp_path: Path) -> None:
        self.inbox = requests.Inbox(tmp_path / "inbox", key=lambda: KEY)
        self.app = route.RequestsApp(self.inbox)
        self.server = make_server(self.app)
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.cookie, self.csrf = "", ""

    def call(self, method: str, path: str, body: bytes | str = b"", headers: dict | None = None, *,
             host: str | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        conn.putrequest(method, path, skip_host=True)
        conn.putheader("Host", host or f"127.0.0.1:{self.port}")
        for name, value in (headers or {}).items():
            conn.putheader(name, value)
        data = body.encode() if isinstance(body, str) else body
        conn.putheader("Content-Length", str(len(data)))
        conn.endheaders(data)
        reply = conn.getresponse()
        out = reply.read()
        got = (reply.status, dict(reply.getheaders()), out)
        conn.close()
        return got

    def login(self) -> None:
        status, headers, page = self.call("GET", f"/requests?k={self.app.launch_key}")
        assert status == 200
        self.cookie = headers["Set-Cookie"].split(";")[0]
        self.csrf = re.search(rb'name="ml-requests-csrf" content="([^"]+)"', page).group(1).decode()

    def post(self, data, *, drop: tuple[str, ...] = (), **headers):
        base = {"Cookie": self.cookie, "Content-Type": "application/json", "X-Requests-CSRF": self.csrf,
                "Origin": f"http://127.0.0.1:{self.port}"}
        base.update(headers)
        for name in drop:
            base.pop(name, None)
        return self.call("POST", "/requests/api/answer", data if isinstance(data, (str, bytes)) else json.dumps(data), base)

    def get(self, path: str):
        return self.call("GET", path, headers={"Cookie": self.cookie})

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def live(tmp_path):
    held = Live(tmp_path)
    held.login()
    yield held
    held.stop()


def raised(live: Live, subject: str = "run_shell(x)", **kw):
    return requests.raise_request(ask(subject, **kw), inbox=live.inbox)


def answer_of(handle, choice="allow-once"):
    return {"id": handle.id, "choice": choice, "fingerprint": handle.fingerprint}


# -- who opens the page ------------------------------------------------------------------
def test_the_page_opens_only_with_the_launch_key_once_and_a_session_cookie_after(tmp_path):
    held = Live(tmp_path)
    try:
        assert held.call("GET", "/requests")[0] == 401
        assert held.call("GET", "/requests?k=wrong")[0] == 401
        held.login()
        assert held.call("GET", f"/requests?k={held.app.launch_key}")[0] == 401
        assert held.call("GET", "/requests", headers={"Cookie": held.cookie})[0] == 200
    finally:
        held.stop()


def test_the_cookie_is_http_only_same_site_strict_and_scoped(tmp_path):
    held = Live(tmp_path)
    try:
        _, headers, _ = held.call("GET", f"/requests?k={held.app.launch_key}")
        flags = headers["Set-Cookie"]
        assert "HttpOnly" in flags and "SameSite=Strict" in flags and "Path=/requests" in flags
    finally:
        held.stop()


def test_every_api_route_needs_the_session_and_ignores_a_bearer_token(live):
    handle = raised(live)
    for path in ("/requests/api/list", f"/requests/api/show/{handle.id}", "/requests/api/feed"):
        assert live.call("GET", path)[0] == 401
        assert live.call("GET", path, headers={"Authorization": "Bearer mlsk1.anything"})[0] == 401
        assert live.get(path)[0] == 200


def test_the_csrf_token_and_the_cookie_never_appear_in_any_api_answer(live):
    handle = raised(live)
    seen = b"".join(live.get(p)[2] for p in ("/requests/api/list", f"/requests/api/show/{handle.id}",
                                             "/requests/api/feed"))
    assert live.csrf.encode() not in seen and live.cookie.split("=")[1].encode() not in seen
    assert live.csrf.encode() not in live.post(answer_of(handle), drop=("X-Requests-CSRF",))[2]


def test_the_page_has_no_inline_script_a_strict_policy_and_serves_only_its_own_files(live):
    _status, headers, page = live.call("GET", "/requests", headers={"Cookie": live.cookie})
    text = page.decode()
    assert "default-src 'none'" in headers["Content-Security-Policy"] and "script-src 'self'" in headers["Content-Security-Policy"]
    assert re.findall(r"<script[^>]*>", text) == ['<script type="module" src="/requests/assets/ml-requests.js">']
    assert "</script>" in text and text.count("<script") == 1 and "onclick" not in text
    assert live.call("GET", "/requests/assets/ml-requests.js")[0] == 200
    for name in ("../../graph/guard.py", "ml-ui.js", "gallery.html", "..%2f..%2fcli.py"):
        assert live.call("GET", f"/requests/assets/{name}")[0] == 404


# -- the answer ------------------------------------------------------------------------
def test_an_answer_from_the_page_resolves_the_request_as_the_ui(live):
    handle = raised(live)
    status, _, body = live.post(answer_of(handle))
    assert status == 200 and json.loads(body)["state"] == "approved"
    assert live.inbox.get(handle.id).answered_by == "ui" and handle.wait(timeout=1).approved


@pytest.mark.parametrize("what", ["host", "origin-other", "origin-missing", "origin-null", "site-cross",
                                  "site-same", "type", "type-form", "csrf-missing", "csrf-wrong", "cookie-missing",
                                  "cookie-wrong"])
def test_a_post_that_breaks_any_one_rule_is_refused_and_changes_nothing(live, what):
    handle = raised(live)
    data = json.dumps(answer_of(handle))
    port = live.port
    tries = {
        "host": lambda: live.call("POST", "/requests/api/answer", data, {"Cookie": live.cookie, "Content-Type": "application/json", "X-Requests-CSRF": live.csrf, "Origin": f"http://evil.example:{port}"}, host=f"evil.example:{port}"),
        "origin-other": lambda: live.post(data, Origin="http://evil.example"),
        "origin-missing": lambda: live.post(data, drop=("Origin",)),
        "origin-null": lambda: live.post(data, Origin="null"),
        "site-cross": lambda: live.post(data, **{"Sec-Fetch-Site": "cross-site"}),
        "site-same": lambda: live.post(data, **{"Sec-Fetch-Site": "same-site"}),
        "type": lambda: live.post(data, **{"Content-Type": "text/plain"}),
        "type-form": lambda: live.post("id=x&choice=allow-once", **{"Content-Type": "application/x-www-form-urlencoded"}),
        "csrf-missing": lambda: live.post(data, drop=("X-Requests-CSRF",)),
        "csrf-wrong": lambda: live.post(data, **{"X-Requests-CSRF": "x" * len(live.csrf)}),
        "cookie-missing": lambda: live.post(data, drop=("Cookie",)),
        "cookie-wrong": lambda: live.post(data, Cookie="ml_requests=guess"),
    }
    status = tries[what]()[0]
    assert status in (400, 401, 403, 421), (what, status)
    assert live.inbox.get(handle.id).state == "pending"


def test_a_dns_rebinding_page_is_refused_on_its_host_even_with_a_valid_cookie(live):
    handle = raised(live)
    status, _, _ = live.call("GET", "/requests/api/list", headers={"Cookie": live.cookie}, host=f"rebind.example:{live.port}")
    assert status == 421
    assert live.call("GET", "/requests/api/list", headers={"Cookie": live.cookie}, host=f"127.0.0.1.evil.example:{live.port}")[0] == 421
    assert live.inbox.get(handle.id).state == "pending"


def test_a_body_over_the_limit_is_refused_before_it_is_read(live):
    handle = raised(live)
    big = json.dumps({**answer_of(handle), "pad": "x" * route.MAX_BODY})
    assert live.post(big)[0] == 413
    assert live.inbox.get(handle.id).state == "pending"


def test_an_unknown_id_a_replayed_answer_and_a_changed_request_are_each_refused(live):
    handle = raised(live)
    assert live.post({"id": "rq_nope", "choice": "allow-once", "fingerprint": "f" * 64})[0] == 404
    assert live.post({**answer_of(handle), "fingerprint": "0" * 64})[0] == 409
    assert live.post(answer_of(handle))[0] == 200
    again = live.post(answer_of(handle, "deny"))
    assert again[0] == 409 and b"already resolved" in again[2]
    assert live.post("{not json")[0] == 400 and live.post("[1]")[0] == 400


def test_an_answer_given_at_a_terminal_first_makes_the_pages_answer_resolved(live):
    handle = raised(live)
    requests.answer(handle.id, "deny", handle.fingerprint, "terminal",
                    requests.Context(terminal=(True, True), env={}, inbox=live.inbox))
    status, _, body = live.post(answer_of(handle))
    assert status == 409 and b"already resolved: denied by terminal" in body


def test_a_server_an_agent_started_answers_nothing(live, monkeypatch):
    handle = raised(live)
    monkeypatch.setenv("CLAUDECODE", "1")
    status, _, _body = live.post(answer_of(handle))
    assert status == 403 and live.inbox.get(handle.id).state == "pending"
    monkeypatch.delenv("CLAUDECODE")
    assert person.marked() == ""


def test_answers_are_rate_limited_per_session(live):
    handles = [raised(live, f"c{n}") for n in range(route.ANSWERS_PER_WINDOW + 3)]
    codes = [live.post({"id": h.id, "choice": "deny", "fingerprint": "0" * 64})[0] for h in handles]
    assert codes.count(429) == 3 and codes[:route.ANSWERS_PER_WINDOW] == [409] * route.ANSWERS_PER_WINDOW


# -- groups, bulk and text ---------------------------------------------------------------
def test_the_list_groups_by_agent_and_project_filters_and_puts_pending_first(live):
    a = raised(live, "one", agent="alpha", project="p1")
    raised(live, "two", agent="beta", project="p2")
    requests.answer(a.id, "deny", a.fingerprint, "ui", requests.Context(env={}, inbox=live.inbox))
    rows = json.loads(live.get("/requests/api/list")[2])["requests"]
    assert [r["state"] for r in rows] == ["pending", "denied"]
    only = json.loads(live.get("/requests/api/list?agent=alpha")[2])["requests"]
    assert [r["agent"] for r in only] == ["alpha"]
    assert live.get("/requests/api/list?state=bogus")[0] == 400


def test_approve_all_covers_only_what_is_listed_and_never_a_destructive_or_human_only_kind(live):
    mine = [raised(live, f"safe {n}") for n in range(3)]
    other = raised(live, "not in the list")
    hard = raised(live, "rm -rf", kind="tool_call_destructive")
    held = raised(live, "release it", kind="quarantine_release", choices=("later", "release"))

    def bulk(items, choice="allow-once"):
        return live.call("POST", "/requests/api/answer-kind", json.dumps({"choice": choice, "items": items}),
                         {"Cookie": live.cookie, "Content-Type": "application/json", "X-Requests-CSRF": live.csrf,
                          "Origin": f"http://127.0.0.1:{live.port}"})

    items = [{"id": h.id, "fingerprint": h.fingerprint} for h in mine]
    assert bulk([{"id": hard.id, "fingerprint": hard.fingerprint}])[0] == 400
    assert bulk([{"id": held.id, "fingerprint": held.fingerprint}], "release")[0] == 400
    assert bulk(items, "allow-always")[0] == 400
    assert live.inbox.get(hard.id).state == "pending" and live.inbox.get(held.id).state == "pending"
    status, _, body = bulk(items)
    assert status == 200 and sorted(json.loads(body)["answered"]) == sorted(h.id for h in mine)
    assert live.inbox.get(other.id).state == "pending"


def test_hostile_text_travels_as_json_data_and_the_element_never_builds_markup():
    hostile = "<img src=x onerror=alert(1)>"
    source = (assets_dir() / "ml-requests.js").read_text()
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "href", ".src"):
        assert banned not in source, banned
    assert hostile not in source


def test_the_element_escapes_control_and_bidi_characters_and_bounds_length():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    probe = """
    globalThis.HTMLElement = class {}; globalThis.customElements = { get() {}, define() {} };
    globalThis.CSSStyleSheet = class { replaceSync() {} }; globalThis.document = {};
    const { plain } = await import(process.argv[1]);
    const out = [plain("a\\u001b[31m\\u202eb\\nc"), plain("z".repeat(500), 50), plain("<b>x</b>")];
    console.log(JSON.stringify(out));
    """
    done = subprocess.run([node, "--input-type=module", "-e", probe, str(assets_dir() / "ml-requests.js")],
                          capture_output=True, text=True, timeout=60, check=False)
    got = json.loads(done.stdout)
    assert "\x1b" not in got[0] and "‮" not in got[0] and "\\x1b" in got[0] and "\\u202e" in got[0]
    assert len(got[1]) == 50 and got[2] == "<b>x</b>"
