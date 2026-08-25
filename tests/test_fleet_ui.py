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

    def test_every_screen_the_wizard_moves_to_is_defined(self):
        """node --check parses the file; it does not notice a screen that is gone."""
        import re

        js = asset_bytes("app.js")[0].decode()
        defined = set(re.findall(r"^(?:async )?function (\w+)", js, re.M))
        called = set(re.findall(r"return (\w+Step)\(", js))
        assert called, "the wizard moves to no screen at all"
        assert called <= defined, f"screens that are gone: {sorted(called - defined)}"

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


class TestJoiningTwice:
    def test_a_cluster_with_no_name_is_the_same_one_either_way(self, serving):
        """The wizard and the Clusters box must derive the same key from the same
        words, or two machines set up different ways never see each other."""
        from ml_stack.fleet.discovery import key_from_passphrase, memberships

        status, body, headers = serving.call("/ui/setup/join", method="POST",
                                             body={"passphrase": WORDS})
        assert status == 200, body
        cookie = headers["Set-Cookie"].split(";")[0]
        first = memberships(serving.keyfile)[0]

        status, body, _ = serving.call("/ui/clusters", method="POST", cookie=cookie,
                                       body={"passphrase": WORDS})
        assert status == 200, body
        rows = memberships(serving.keyfile)
        assert [m.group for m in rows] == [first.group]
        assert rows[0].key == key_from_passphrase(WORDS, group="ml-stack")


# -- a machine in no cluster ---------------------------------------------
class TestOnItsOwn:
    """Joining a cluster is optional. A machine that skipped it has no password, so it
    answers to itself and to nobody else."""

    def finished(self, serving):
        status, body, _ = serving.call("/ui/setup/done", method="POST")
        assert status == 200, body
        return body

    def test_the_wizard_is_not_shown_again_once_it_is_finished(self, serving):
        assert serving.call("/ui/setup")[1]["needs_setup"] is True
        assert self.finished(serving)["needs_setup"] is False
        assert serving.call("/ui/setup")[1]["needs_setup"] is False
        assert not in_cluster(serving.keyfile)

    def test_finishing_is_remembered_between_runs(self, serving):
        from ml_stack.fleet.settings import Settings

        self.finished(serving)
        assert Settings.load(serving.ui.settings_path).setup_done is True

    def test_there_is_no_password_to_ask_for(self, serving):
        assert serving.call("/ui/setup")[1]["needs_password"] is False
        serving.call("/ui/setup/join", method="POST",
                     body={"passphrase": WORDS, "group": "home"})
        assert serving.call("/ui/setup")[1]["needs_password"] is True

    def test_the_machine_itself_gets_in_without_signing_in(self, serving):
        self.finished(serving)
        status, body, _ = serving.call("/ui/settings")
        assert status == 200, body

    def test_nobody_else_does(self, serving):
        self.finished(serving)
        status, body, _ = serving.call("/ui/settings", host=primary_ip())
        assert status == 403
        assert "ssh" in body["error"]

    def test_a_page_pointing_a_domain_at_loopback_does_not(self, serving):
        self.finished(serving)
        status, body, _ = serving.call(
            "/ui/settings", headers={"Host": f"evil.example.com:{serving.port}"})
        assert status == 403
        assert "hostname" in body["error"]

    def test_finishing_is_refused_from_another_machine(self, serving):
        status, body, _ = serving.call("/ui/setup/done", method="POST",
                                       host=primary_ip())
        assert status == 403
        assert serving.call("/ui/setup")[1]["needs_setup"] is True

    def test_a_cluster_puts_the_password_back(self, serving):
        self.finished(serving)
        serving.call("/ui/setup/join", method="POST",
                     body={"passphrase": WORDS, "group": "home"})
        status, body, _ = serving.call("/ui/settings")
        assert status == 401 and "sign in" in body["error"]


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
        assert "RTX 4090" in got["labels"].why

    def test_a_machine_with_no_gpu_is_suggested_for_data(self):
        from ml_stack.fleet.settings import suggest

        got = suggest({"accelerator": False, "cpus": 12})
        assert got["labels"].value == ["prep"]

    def test_no_machine_is_offered_more_than_one_job_at_a_time(self):
        """Two jobs on one card contend for memory and both get slower, with nothing
        in the logs to say so."""
        from ml_stack.fleet.settings import suggest

        for machine in ({"accelerator": True, "cpus": 16},
                        {"accelerator": False, "cpus": 64}):
            assert "slots" not in suggest(machine)

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

    def test_asking_for_more_jobs_leaves_it_at_one(self, joined):
        _, _, headers = joined.call("/ui/session", method="POST",
                                    body={"passphrase": WORDS})
        cookie = headers["Set-Cookie"].split(";")[0]

        status, body, _ = joined.call("/ui/setup/prefs", method="POST", cookie=cookie,
                                      body={"slots": 4, "labels": ["prep"],
                                            "autostart": "manual"})

        assert status == 200, body
        assert joined.runner.slots == 1
        assert joined.call("/health")[1]["slots"] == 1

    def test_saving_a_preference_does_not_interrupt_running_work(self, joined):
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


class TestClosingTheWindow:
    """Asked once, then remembered if the box was left ticked."""

    class FakeWindow:
        """Records what was asked of it, and on which thread."""

        def __init__(self):
            self.hidden = self.destroyed = False
            self.evaluated = []
            self.threads = []

        def _mark(self):
            self.threads.append(threading.current_thread().name)

        def hide(self):
            self._mark()
            self.hidden = True

        def destroy(self):
            self._mark()
            self.destroyed = True

        def evaluate_js(self, script):
            self._mark()
            self.evaluated.append(script)

    def bridge(self, tmp_path):
        from ml_stack.fleet.app import Bridge

        b = Bridge(tmp_path / "settings.json")
        b.window = self.FakeWindow()
        return b

    def test_a_fresh_machine_has_no_saved_answer(self, tmp_path):
        from ml_stack.fleet.settings import Settings

        assert Settings.load(tmp_path / "settings.json").on_close == ""

    def test_keeping_it_running_hides_the_window(self, tmp_path):
        b = self.bridge(tmp_path)
        b.close_choice("background", remember=False)
        assert b.window.hidden and not b.window.destroyed

    def test_quitting_destroys_it(self, tmp_path):
        b = self.bridge(tmp_path)
        b.close_choice("quit", remember=False)
        assert b.window.destroyed

    def test_unticking_the_box_asks_again_next_time(self, tmp_path):
        from ml_stack.fleet.settings import Settings

        b = self.bridge(tmp_path)
        b.close_choice("quit", remember=False)
        assert Settings.load(tmp_path / "settings.json").on_close == ""

    def test_leaving_the_box_ticked_remembers(self, tmp_path):
        from ml_stack.fleet.settings import Settings

        b = self.bridge(tmp_path)
        b.close_choice("background", remember=True)
        assert Settings.load(tmp_path / "settings.json").on_close == "background"

    def test_the_answer_can_be_changed_later(self, tmp_path):
        from ml_stack.fleet.settings import Settings

        b = self.bridge(tmp_path)
        b.close_choice("background", remember=True)
        b.window = self.FakeWindow()
        b.close_choice("quit", remember=True)
        assert Settings.load(tmp_path / "settings.json").on_close == "quit"

    def test_an_answer_that_is_neither_is_refused(self, tmp_path):
        b = self.bridge(tmp_path)
        assert b.close_choice("explode", True) == {"ok": False}
        assert not b.window.hidden and not b.window.destroyed

    def test_the_question_is_asked_off_the_drawing_thread(self, tmp_path):
        b = self.bridge(tmp_path)
        assert b.on_closing() is False
        b.pending.join(5)
        assert b.window.evaluated == [
            "window.mlStackAskOnClose && window.mlStackAskOnClose()"]
        assert b.window.threads == ["ml-stack-close"]

    def test_keeping_it_running_hides_off_the_drawing_thread(self, tmp_path):
        from ml_stack.fleet.settings import Settings

        b = self.bridge(tmp_path)
        Settings(on_close="background").save(tmp_path / "settings.json")
        assert b.on_closing() is False
        b.pending.join(5)
        assert b.window.hidden and b.window.threads == ["ml-stack-close"]

    def test_a_saved_quit_closes_without_asking(self, tmp_path):
        from ml_stack.fleet.settings import Settings

        b = self.bridge(tmp_path)
        Settings(on_close="quit").save(tmp_path / "settings.json")
        assert b.on_closing() is True
        assert b.window.evaluated == []

    def test_quitting_does_not_ask_the_question_again(self, tmp_path):
        """Closing the window runs the handler a second time."""
        b = self.bridge(tmp_path)
        b.close_choice("quit", remember=False)
        assert b.window.destroyed
        assert b.on_closing() is True
        assert b.window.evaluated == []

    def test_the_page_offers_the_question(self):
        """The native window calls this by name when the close button is clicked."""
        from ml_stack.fleet.ui import asset_bytes

        js = asset_bytes("app.js")[0].decode()
        assert "mlStackAskOnClose" in js
        assert "close_choice" in js
        assert 'id: "remember", checked: "1"' in js, "the box must start ticked"


class TestSettingsScreen:
    @pytest.fixture
    def signed_in(self, serving):
        serving.call("/ui/setup/join", method="POST",
                     body={"passphrase": WORDS, "group": "home"})
        _, _, headers = serving.call("/ui/session", method="POST",
                                     body={"passphrase": WORDS})
        return serving, headers["Set-Cookie"].split(";")[0]

    def test_it_reports_what_the_daemon_is_doing(self, signed_in):
        serving, cookie = signed_in
        status, body, _ = serving.call("/ui/settings", cookie=cookie)
        assert status == 200
        assert body["settings"]["slots"] == serving.runner.slots
        assert body["group"] == "home"
        assert "autostart" in body and "version" in body

    def test_settings_need_a_session(self, serving):
        serving.call("/ui/setup/join", method="POST", body={"passphrase": WORDS})
        assert serving.call("/ui/settings")[0] == 401

    def test_the_settings_screen_cannot_raise_the_job_count(self, signed_in):
        from ml_stack.fleet.settings import Settings

        serving, cookie = signed_in
        serving.call("/ui/settings", method="POST", cookie=cookie,
                     body={"slots": 5, "labels": ["prep"], "autostart": "manual"})

        assert serving.runner.slots == 1
        assert serving.call("/health")[1]["slots"] == 1
        saved = Settings.load(serving.ui.settings_path)
        assert saved.slots == 1 and saved.labels == ["prep"]

    def test_the_close_preference_can_be_set_from_settings(self, signed_in):
        from ml_stack.fleet.settings import Settings

        serving, cookie = signed_in
        serving.call("/ui/settings", method="POST", cookie=cookie,
                     body={"on_close": "background", "autostart": "manual"})
        assert Settings.load(serving.ui.settings_path).on_close == "background"

    def test_automatic_updates_are_on_unless_turned_off(self, signed_in):
        from ml_stack.fleet.settings import Settings

        serving, cookie = signed_in
        assert Settings().auto_update is True
        serving.call("/ui/settings", method="POST", cookie=cookie,
                     body={"auto_update": False, "autostart": "manual"})
        assert Settings.load(serving.ui.settings_path).auto_update is False

    def test_getting_models_automatically_is_on_unless_turned_off(self, signed_in):
        from ml_stack.fleet.settings import Settings

        serving, cookie = signed_in
        assert Settings().autodownload_models is True
        serving.call("/ui/settings", method="POST", cookie=cookie,
                     body={"autodownload_models": False, "autostart": "manual"})
        assert Settings.load(serving.ui.settings_path).autodownload_models is False


class TestTheInterfaceAndTheDaemonAgree:
    """Every address the page calls must be one the daemon answers.

    A screen calling a route nobody wrote returns "no such route" and shows an empty
    panel; no Python test notices, because no Python test asks for that address.
    """

    @pytest.fixture
    def signed_in(self, serving):
        serving.call("/ui/setup/join", method="POST",
                     body={"passphrase": WORDS, "group": "home"})
        _, _, headers = serving.call("/ui/session", method="POST",
                                     body={"passphrase": WORDS})
        return serving, headers["Set-Cookie"].split(";")[0]

    def called_paths(self):
        import re

        asset = asset_bytes("app.js")
        assert asset is not None
        source = asset[0].decode()
        found = set(re.findall(r"""api\(\s*[`"']([^`"']+)""", source))
        found |= set(re.findall(r"""fetch\(\s*[`"']([^`"']+)""", source))
        # Only the path matters here. A query string carries template holes, and
        # any id will do for asking whether the route exists at all.
        cleaned = set()
        for pp in found:
            if not pp.startswith("/ui"):
                continue
            path = pp.split("?", 1)[0]
            cleaned.add(re.sub(r"\$\{[^}]*\}", "x", path))
        return sorted(cleaned)

    def test_the_page_calls_nothing_the_daemon_does_not_answer(self, signed_in,
                                                              monkeypatch):
        import urllib.error

        serving, cookie = signed_in
        from ml_stack.fleet import models as models_mod
        from ml_stack.fleet.conversations import Conversations
        from ml_stack.fleet.models import Models
        serving.ui.conversations = Conversations(serving.files.parent / "chats")
        serving.ui.models = Models([serving.files], serving.files)

        # No hub: the popular route must still answer, and a test must not wait on
        # the internet to find out whether a route exists.
        def unreachable(*a, **k):
            raise urllib.error.URLError("no network in tests")

        monkeypatch.setattr(models_mod, "_hub", unreachable)

        called = self.called_paths()
        assert "/ui/chat" in called and "/ui/models" in called, called

        missing = []
        for path in called:
            for method in ("GET", "POST"):
                status, body, _ = serving.call(path, method=method, cookie=cookie,
                                               body={} if method == "POST" else None)
                if status == 404 and body.get("error") == "no such route":
                    missing.append(f"{method} {path}")
        # A route may refuse a method; it may not be absent for both.
        both = [p for p in called
                if f"GET {p}" in missing and f"POST {p}" in missing]
        assert both == [], f"the page calls addresses nobody answers: {both}"


class TestUpdates:
    def test_versions_compare_numerically(self):
        from ml_stack.fleet.updates import Release

        r = Release(version="0.2.0", url="", notes="", assets=(), checked_at=0)
        assert r.newer_than("0.1.9")
        assert r.newer_than("v0.1.9")
        assert not r.newer_than("0.2.0")
        assert not r.newer_than("1.0.0")

    def test_two_digit_parts_do_not_sort_as_text(self):
        from ml_stack.fleet.updates import Release

        assert Release("0.10.0", "", "", (), 0).newer_than("0.9.0")

    def test_nothing_is_newer_than_a_version_nobody_knows(self):
        """A source tree used to report 0.0.0, so every release looked newer and the
        screen offered to move backwards onto the last tag."""
        from ml_stack.fleet.updates import Release

        older = Release("0.1.3", "", "", (), 0)
        assert older.newer_than("") is False
        assert older.newer_than("   ") is False
        assert older.newer_than("0.1.4") is False
        assert older.newer_than("0.1.2") is True

    def test_a_source_tree_reports_the_version_it_holds(self, monkeypatch):
        import tomllib
        from pathlib import Path as P

        from ml_stack.fleet import updates

        def no_metadata(*a, **k):
            raise LookupError("not installed")

        monkeypatch.setattr("importlib.metadata.version", no_metadata)
        monkeypatch.delenv("ML_STACK_VERSION", raising=False)

        got = updates.current_version()
        here = P(updates.__file__).resolve()
        pyproject = next(
            p / "packages" / "ml-stack-fleet" / "pyproject.toml"
            for p in here.parents
            if (p / "packages" / "ml-stack-fleet" / "pyproject.toml").is_file())
        want = tomllib.loads(pyproject.read_text())["project"]["version"]
        assert got == want, f"reported {got!r}, the checkout says {want!r}"

    def test_being_told_the_version_wins_over_guessing(self, monkeypatch):
        from ml_stack.fleet import updates

        def no_metadata(*a, **k):
            raise LookupError("not installed")

        monkeypatch.setattr("importlib.metadata.version", no_metadata)
        monkeypatch.setenv("ML_STACK_VERSION", "9.9.9")
        assert updates.current_version() == "9.9.9"

    def test_with_no_way_to_tell_it_says_nothing_rather_than_zero(self, monkeypatch):
        from ml_stack.fleet import updates

        def no_metadata(*a, **k):
            raise LookupError("not installed")

        monkeypatch.setattr("importlib.metadata.version", no_metadata)
        monkeypatch.delenv("ML_STACK_VERSION", raising=False)
        monkeypatch.setattr(updates, "_version_in_source", lambda: "")
        assert updates.current_version() == ""

    def test_the_download_for_this_machine_is_picked(self):
        from ml_stack.fleet.updates import Release, asset_for, platform_key

        key = platform_key()
        release = Release("9.9.9", "", "", (
            {"name": "ml-stack-somewhere-else.zip"},
            {"name": f"ml-stack-{key}.zip"},
        ), 0)
        assert asset_for(release)["name"] == f"ml-stack-{key}.zip"

    def test_a_release_with_nothing_for_this_machine_returns_none(self):
        from ml_stack.fleet.updates import Release, asset_for

        assert asset_for(Release("9.9.9", "", "", ({"name": "source.tar.gz"},), 0)) is None

    def test_a_download_whose_digest_is_wrong_is_discarded(self, tmp_path):
        import http.server
        import threading

        from ml_stack.fleet.updates import UpdateError, download

        payload = b"not the right bytes"

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            asset = {"name": "x.zip", "size": len(payload),
                     "browser_download_url": f"http://127.0.0.1:{srv.server_address[1]}/x",
                     "digest": "sha256:" + "0" * 64}
            with pytest.raises(UpdateError, match="digest"):
                download(asset, tmp_path)
            assert not list(tmp_path.iterdir())
        finally:
            srv.shutdown()

    def test_an_archive_that_escapes_its_directory_is_refused(self, tmp_path):
        import zipfile

        from ml_stack.fleet.updates import UpdateError, install

        bad = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("../escaped.txt", "no")
        with pytest.raises(UpdateError, match="refusing"):
            install(bad, app_path=tmp_path / "app")

    def test_a_pip_install_is_told_to_update_with_pip(self, tmp_path):
        from ml_stack.fleet.ui import UI

        ui = UI(name="x", cluster_key_path=tmp_path / "k")
        got = ui.install_update()
        assert not got["ok"] and "pip" in got["error"]


def test_the_interface_script_parses():
    """A duplicate declaration anywhere in this file stops the whole page loading."""
    import shutil
    import subprocess

    from ml_stack.fleet.ui import ASSETS

    node = shutil.which("node")
    if node is None:
        pytest.skip("no node to parse with")
    done = subprocess.run([node, "--check", str(ASSETS / "app.js")],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-400:]



class TestUpdatingItself:
    """auto_update was stored and shown and did nothing: no timer looked at it, and
    relaunch was written but never called."""

    def test_a_newer_release_is_put_on_and_the_copy_restarts(self, monkeypatch,
                                                             tmp_path):
        from ml_stack.fleet import updates

        seen = {}
        monkeypatch.setattr(updates, "running_path", lambda: tmp_path / "ml-stack")
        monkeypatch.setattr(updates, "current_version", lambda: "0.1.0")
        monkeypatch.setattr(updates, "check", lambda **k: updates.Release(
            "0.2.0", "", "", ({"name": "ml-stack-macos-arm64.zip"},), 0))
        monkeypatch.setattr(updates, "asset_for", lambda r, key="": r.assets[0])
        monkeypatch.setattr(updates, "download", lambda a, into, **k: tmp_path / "a.zip")
        monkeypatch.setattr(updates, "install", lambda a: seen.setdefault("put", a))

        got = updates.apply_if_newer()
        assert got == {"ok": True, "installed": True, "version": "0.2.0"}, got
        assert "put" in seen, "it never unpacked anything"

    def test_an_older_release_is_left_alone(self, monkeypatch, tmp_path):
        from ml_stack.fleet import updates

        monkeypatch.setattr(updates, "running_path", lambda: tmp_path / "ml-stack")
        monkeypatch.setattr(updates, "current_version", lambda: "0.2.0")
        monkeypatch.setattr(updates, "check", lambda **k: updates.Release(
            "0.1.0", "", "", (), 0))

        def never(*a, **k):
            raise AssertionError("it tried to install an older release")

        monkeypatch.setattr(updates, "install", never)
        got = updates.apply_if_newer()
        assert got["installed"] is False

    def test_a_pip_install_is_told_to_use_pip(self, monkeypatch):
        from ml_stack.fleet import updates

        monkeypatch.setattr(updates, "running_path", lambda: None)
        got = updates.apply_if_newer()
        assert got["installed"] is False
        assert "pip" in got["error"]

    def test_the_watcher_leaves_a_machine_that_is_working_alone(self, monkeypatch):
        """An update that killed a training run halfway would be worse than waiting.

        The loop swallows exceptions so a bad network does not stop it, so this
        counts the calls rather than raising inside it.
        """
        from ml_stack.fleet import updates

        tried = threading.Event()
        monkeypatch.setattr(updates, "apply_if_newer",
                            lambda: (tried.set(), {"installed": False})[1])
        thread = updates.watch(wanted=lambda: True, idle=lambda: False,
                               every_s=0.02, first_after_s=0.0)
        time.sleep(0.4)
        assert not tried.is_set(), "it updated while a job was running"
        assert thread.is_alive()

    def test_the_watcher_does_nothing_when_it_is_turned_off(self, monkeypatch):
        from ml_stack.fleet import updates

        tried = threading.Event()
        monkeypatch.setattr(updates, "apply_if_newer",
                            lambda: (tried.set(), {"installed": False})[1])
        updates.watch(wanted=lambda: False, idle=lambda: True,
                      every_s=0.02, first_after_s=0.0)
        time.sleep(0.4)
        assert not tried.is_set(), "it updated with the setting off"

    def test_an_idle_machine_with_the_setting_on_updates_and_restarts(self,
                                                                     monkeypatch):
        from ml_stack.fleet import updates

        done = threading.Event()
        monkeypatch.setattr(updates, "apply_if_newer",
                            lambda: {"ok": True, "installed": True, "version": "9.9.9"})
        monkeypatch.setattr(updates, "relaunch",
                            lambda **k: (done.set(), True)[1])
        updates.watch(wanted=lambda: True, idle=lambda: True,
                      every_s=0.05, first_after_s=0.0)
        assert done.wait(3.0), "it never restarted itself"

    def test_relaunch_says_no_when_this_is_not_a_bundle(self, monkeypatch):
        from ml_stack.fleet import updates

        monkeypatch.setattr(updates, "running_path", lambda: None)
        assert updates.relaunch() is False

    def test_relaunch_starts_the_replaced_copy(self, monkeypatch, tmp_path):
        from ml_stack.fleet import updates

        target = tmp_path / "ml-stack"
        target.write_text("#!/bin/sh\n")
        started = []
        monkeypatch.setattr(updates, "running_path", lambda: target)
        monkeypatch.setattr(updates.subprocess, "Popen",
                            lambda argv, **k: started.append(argv))

        assert updates.relaunch(delay_s=0.0, stop=False) is True
        for _ in range(40):
            if started:
                break
            time.sleep(0.05)
        assert started, "it never started the new copy"
        assert str(target) in " ".join(str(x) for x in started[0])
