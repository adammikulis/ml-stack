"""The web interface, against a real daemon on a real socket.

The dangerous route here is first-run setup: a daemon that has not joined a cluster has
no credential to check, and the setup route can create one -- so whoever reaches it first
owns the machine. Those guards are the bulk of what is tested, and they are tested by
making real requests from real addresses rather than by asserting on a function's return.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from ml_stack.fleet.daemon import JobRunner, load_or_create_token, make_handler
from ml_stack.fleet.discovery import in_cluster, primary_ip
from ml_stack.fleet.session import Sessions, Throttle, parse_cookie
from ml_stack.fleet.ui import UI, asset_bytes


def _maybe_json(raw: bytes) -> dict:
    """Routes under /ui serve both JSON and a web page; the page is not a failure."""
    try:
        return json.loads(raw or b"{}")
    except ValueError:
        return {"raw": raw[:400].decode(errors="replace")}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class Serving:
    """A real daemon with the UI mounted, bound on every interface."""

    def __init__(self, tmp_path, name="studio", setup_token=""):
        root = tmp_path / "traind"
        self.files = root / "files"
        self.files.mkdir(parents=True)
        self.keyfile = tmp_path / "cluster.key"
        token = load_or_create_token(root)
        self.runner = JobRunner(root, self.files)
        self.ui = UI(name=name, cluster_key_path=self.keyfile,
                     setup_token=setup_token)
        from ml_stack.fleet.settings import Settings
        self.ui.runner = self.runner
        self.ui.settings = Settings()
        self.ui.settings_path = tmp_path / "settings.json"
        self.ui.report = lambda: {"cpus": 8, "accelerator": False}
        self.port = _free_port()
        self.httpd = ThreadingHTTPServer(
            ("0.0.0.0", self.port),
            make_handler(self.runner, self.files, token, name, ui=self.ui))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def call(self, path, *, method="GET", body=None, host="127.0.0.1",
             headers=None, ui_header=True, cookie=""):
        url = f"http://{host}:{self.port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if ui_header:
            req.add_header("X-ML-Stack-UI", "1")
        if cookie:
            req.add_header("Cookie", cookie)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, _maybe_json(r.read()), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, _maybe_json(e.read()), dict(e.headers)

    def close(self):
        self.runner.shutdown()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serving(tmp_path):
    s = Serving(tmp_path)
    try:
        yield s
    finally:
        s.close()


WORDS = "correct horse battery"


# -- assets --------------------------------------------------------------
class TestAssets:
    def test_the_page_and_its_assets_ship_with_the_package(self):
        for name in ("index.html", "style.css", "app.js"):
            assert asset_bytes(name) is not None, f"{name} is missing from web/"

    def test_every_asset_the_page_asks_for_exists(self):
        """A stylesheet that 404s is a UI that looks broken rather than one that is."""
        html = asset_bytes("index.html")[0].decode()
        import re
        for ref in re.findall(r'(?:src|href)="/ui/static/([^"]+)"', html):
            assert asset_bytes(ref) is not None, f"index.html references missing {ref}"

    def test_there_is_no_python_hiding_in_the_asset_directory(self):
        """web/ is data, not code -- which is what keeps this package device tier. The
        tier check only globs *.py, so it would not notice a module smuggled in here."""
        from ml_stack.fleet.ui import ASSETS
        assert not list(ASSETS.glob("*.py"))

    def test_an_asset_that_was_not_shipped_is_not_served(self, serving):
        status, _, _ = serving.call("/ui/static/../daemon.py")
        assert status == 404

    def test_the_page_is_served_as_html_exactly_once(self, serving):
        """Two Content-Type headers and a browser takes the first, so an HTML page
        announced second arrives as a download rather than a page."""
        status, _, headers = serving.call("/ui/")
        assert status == 200
        assert headers["Content-Type"] == "text/html"


# -- the setup guard -----------------------------------------------------
class TestFirstRunIsNotUpForGrabs:
    def test_a_fresh_daemon_says_it_needs_setting_up(self, serving):
        status, body, _ = serving.call("/ui/setup")
        assert status == 200 and body["needs_setup"] is True

    def test_setup_works_from_the_machine_itself(self, serving):
        status, body, _ = serving.call("/ui/setup/join", method="POST",
                                       body={"passphrase": WORDS, "group": "home"})
        assert status == 200, body
        assert body["in_cluster"] is True and body["group"] == "home"
        assert in_cluster(serving.keyfile)

    def test_setup_is_refused_from_another_machine(self, serving):
        """Whoever reaches an unjoined daemon first would own it. Being on the LAN is
        not enough; you have to be on the box, or use ssh and the CLI."""
        status, body, _ = serving.call("/ui/setup/join", method="POST",
                                       body={"passphrase": WORDS},
                                       host=primary_ip())
        assert status == 403
        assert "ssh" in body["error"]
        assert not in_cluster(serving.keyfile)

    def test_a_request_addressed_to_a_hostname_is_refused(self, serving):
        """DNS rebinding: a page anywhere can point a domain at 127.0.0.1 and POST to
        it. Loopback-only buys nothing without checking what it was addressed to."""
        status, body, _ = serving.call(
            "/ui/setup/join", method="POST", body={"passphrase": WORDS},
            headers={"Host": f"evil.example.com:{serving.port}"})
        assert status == 403
        assert "hostname" in body["error"]
        assert not in_cluster(serving.keyfile)

    def test_the_api_is_unreachable_without_the_ui_header(self, serving):
        """A cross-origin form, image or link cannot set a custom header without a
        preflight -- and the daemon answers no preflight."""
        status, _, _ = serving.call("/ui/setup", ui_header=False)
        assert status == 403

    def test_rejoining_needs_a_session_once_the_box_is_in_a_cluster(self, serving):
        serving.call("/ui/setup/join", method="POST",
                     body={"passphrase": WORDS, "group": "home"})
        status, body, _ = serving.call("/ui/setup/join", method="POST",
                                       body={"passphrase": "different words here"})
        assert status == 401
        assert "sign in" in body["error"]

    def test_a_setup_code_lets_a_headless_box_be_set_up_remotely(self, tmp_path):
        s = Serving(tmp_path, setup_token="abc123xyz")
        try:
            refused, _, _ = s.call("/ui/setup/join", method="POST",
                                   body={"passphrase": WORDS}, host=primary_ip())
            assert refused == 403
            ok, body, _ = s.call("/ui/setup/join", method="POST",
                                 body={"passphrase": WORDS}, host=primary_ip(),
                                 headers={"X-ML-Stack-Setup": "abc123xyz"})
            assert ok == 200, body
        finally:
            s.close()


# -- signing in ----------------------------------------------------------
class TestSignIn:
    @pytest.fixture
    def joined(self, serving):
        serving.call("/ui/setup/join", method="POST",
                     body={"passphrase": WORDS, "group": "home"})
        return serving

    def test_the_passphrase_signs_you_in(self, joined):
        """Typing the words you already know, rather than pasting 43 characters."""
        status, body, headers = joined.call("/ui/session", method="POST",
                                            body={"passphrase": WORDS})
        assert status == 200 and body["signed_in"]
        assert "HttpOnly" in headers["Set-Cookie"]
        assert "SameSite=Strict" in headers["Set-Cookie"]

    def test_the_wrong_passphrase_does_not(self, joined):
        status, _, _ = joined.call("/ui/session", method="POST",
                                   body={"passphrase": "not the words"})
        assert status == 401

    def test_one_typo_does_not_lock_you_out(self, joined):
        """Someone who fumbles a passphrase once and is then told to wait has been
        punished for being the legitimate user."""
        joined.call("/ui/session", method="POST", body={"passphrase": "wrong words"})
        status, body, _ = joined.call("/ui/session", method="POST",
                                      body={"passphrase": WORDS})
        assert status == 200, body

    def test_persistent_guessing_is_slowed_down(self, joined):
        for _ in range(6):
            status, body, _ = joined.call("/ui/session", method="POST",
                                          body={"passphrase": "nope nope nope"})
        assert status == 429
        assert "0s" not in body["error"], "refusing while saying 'wait 0s' reads as a bug"

    def test_the_cluster_cannot_be_browsed_without_signing_in(self, joined):
        status, _, _ = joined.call("/ui/peers")
        assert status == 401

    def test_a_session_opens_the_cluster_view(self, joined):
        _, _, headers = joined.call("/ui/session", method="POST",
                                    body={"passphrase": WORDS})
        cookie = headers["Set-Cookie"].split(";")[0]
        status, body, _ = joined.call("/ui/peers", cookie=cookie)
        assert status == 200
        assert body["group"] == "home"

    def test_signing_out_ends_the_session(self, joined):
        _, _, headers = joined.call("/ui/session", method="POST",
                                    body={"passphrase": WORDS})
        cookie = headers["Set-Cookie"].split(";")[0]
        joined.call("/ui/session", method="DELETE", cookie=cookie)
        status, _, _ = joined.call("/ui/peers", cookie=cookie)
        assert status == 401

    def test_the_ui_cookie_does_not_work_on_the_job_api(self, joined):
        """The cookie is scoped to /ui. A browser session must not become a bearer
        credential for the route that runs commands."""
        _, _, headers = joined.call("/ui/session", method="POST",
                                    body={"passphrase": WORDS})
        cookie = headers["Set-Cookie"].split(";")[0]
        status, _, _ = joined.call("/jobs", cookie=cookie)
        assert status == 401


# -- sessions and throttle, without a socket -----------------------------
class TestSessions:
    def test_a_session_expires(self):
        s = Sessions(ttl_s=-1)
        assert s.get(s.open().sid) is None

    def test_a_ticket_is_single_use(self):
        s = Sessions()
        ticket, _ = s.mint_ticket()
        assert s.spend_ticket(ticket)
        assert not s.spend_ticket(ticket)

    def test_a_restart_forgets_every_session(self):
        """In memory on purpose: a credential persisted to save people a login is one
        that outlives the process holding it, on a box that runs what it is sent."""
        old = Sessions()
        sid = old.open().sid
        assert Sessions().get(sid) is None

    def test_a_cookie_value_is_read_out_of_a_real_header(self):
        assert parse_cookie("other=1; ml_stack_ui=abc; x=2") == "abc"
        assert parse_cookie("nothing=here") == ""


class TestThrottle:
    def test_early_mistakes_cost_nothing(self):
        t = Throttle(free_attempts=3)
        for _ in range(3):
            t.failed("1.2.3.4")
        assert t.blocked_for("1.2.3.4") == 0

    def test_then_the_delay_grows(self):
        t = Throttle(free_attempts=3)
        for _ in range(5):
            t.failed("1.2.3.4")
        first = t.blocked_for("1.2.3.4")
        t.failed("1.2.3.4")
        assert t.blocked_for("1.2.3.4") > first

    def test_a_delay_is_never_reported_as_zero(self):
        t = Throttle(free_attempts=0, base_backoff_s=0.01)
        t.failed("1.2.3.4")
        assert t.blocked_for("1.2.3.4") >= 1.0

    def test_success_clears_it(self):
        t = Throttle(free_attempts=0)
        t.failed("1.2.3.4")
        t.succeeded("1.2.3.4")
        assert t.blocked_for("1.2.3.4") == 0

    def test_one_derivation_at_a_time(self):
        """scrypt is ~64MB a call on an unauthenticated route. Twenty at once is a
        gigabyte on a box whose whole job is to have memory free for training."""
        t = Throttle(slots=1, wait_s=0.05)
        assert t.acquire()
        assert not t.acquire(), "a second derivation ran concurrently"
        t.release()
        assert t.acquire()


class TestPreferences:
    """The wizard's settings step: suggested from the machine, applied live."""

    @pytest.fixture
    def joined(self, serving):
        serving.call("/ui/setup/join", method="POST",
                     body={"passphrase": WORDS, "group": "home"})
        return serving

    def test_a_gpu_machine_is_suggested_for_training(self):
        from ml_stack.fleet.settings import suggest

        got = suggest({"accelerator": True, "gpu": "RTX 4090", "cpus": 16})
        assert got["labels"].value == ["train"]
        assert got["slots"].value == 1
        assert "RTX 4090" in got["labels"].why

    def test_a_machine_with_no_gpu_is_suggested_for_data(self):
        from ml_stack.fleet.settings import suggest

        got = suggest({"accelerator": False, "cpus": 12})
        assert got["labels"].value == ["prep"]
        assert 1 < got["slots"].value <= 8

    def test_a_small_machine_is_not_given_more_jobs_than_cores(self):
        from ml_stack.fleet.settings import suggest

        assert suggest({"accelerator": False, "cpus": 2})["slots"].value == 1

    def test_every_suggestion_carries_a_reason(self):
        from ml_stack.fleet.settings import suggest

        for key, s in suggest({"accelerator": True, "cpus": 8}).items():
            if key == "work_hours" and not s.value:
                continue
            assert s.why, f"{key} was pre-selected with no reason shown"

    def test_settings_survive_a_restart(self, tmp_path):
        from ml_stack.fleet.settings import Settings

        Settings(slots=6, labels=["prep"], on_paused="finish").save(tmp_path / "s.json")
        back = Settings.load(tmp_path / "s.json")

        assert back.slots == 6 and back.labels == ["prep"]
        assert back.on_paused == "finish"

    def test_a_corrupt_settings_file_costs_preferences_not_the_daemon(self, tmp_path):
        from ml_stack.fleet.settings import Settings

        (tmp_path / "s.json").write_text("{ not json")
        assert Settings.load(tmp_path / "s.json").slots == 1

    def test_the_suggestions_describe_the_machine_asking(self, joined):
        status, body, _ = joined.call("/ui/setup/suggest")
        assert status == 200
        assert "machine" in body and "suggest" in body
        assert body["suggest"]["slots"]["value"] >= 1

    def test_choosing_more_jobs_takes_effect_without_a_restart(self, joined):
        _, _, headers = joined.call("/ui/session", method="POST",
                                    body={"passphrase": WORDS})
        cookie = headers["Set-Cookie"].split(";")[0]

        status, body, _ = joined.call("/ui/setup/prefs", method="POST", cookie=cookie,
                                      body={"slots": 4, "labels": ["prep"],
                                            "autostart": "manual"})

        assert status == 200, body
        assert joined.runner.slots == 4
        health = joined.call("/health")[1]
        assert health["slots"] == 4

    def test_lowering_the_count_does_not_interrupt_running_work(self, joined):
        _, _, headers = joined.call("/ui/session", method="POST",
                                    body={"passphrase": WORDS})
        cookie = headers["Set-Cookie"].split(";")[0]
        joined.call("/ui/setup/prefs", method="POST", cookie=cookie,
                    body={"slots": 4, "autostart": "manual"})
        joined.runner.submit("keep", [sys.executable, "-c", "import time;time.sleep(3)"],
                             cwd="")
        time.sleep(1.0)

        joined.call("/ui/setup/prefs", method="POST", cookie=cookie,
                    body={"slots": 1, "autostart": "manual"})

        assert joined.runner.status()["running"], "a running job was cut off by a setting"
