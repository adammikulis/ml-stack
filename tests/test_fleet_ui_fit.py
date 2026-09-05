"""The Fit view: its two routes, the arithmetic the page does, and ``fit --ui``.

Nothing here reads a real measurement. The autouse fixture points both halves of the
source of truth at ``tmp_path`` -- ``package_file`` is a function so it can be replaced,
and the machine's own half moves with ``$MLSTACK_FIT_FILE`` -- and fills the shipped half
with invented models whose numbers are round enough to check by hand. ``hub.room`` is
replaced too, so no test depends on the machine it runs on.

The test that matters most is `test_the_page_and_fit_py_answer_the_same`: the page composes
`Fit.line`, `Fit.users` and `Fit.longest` itself, in JavaScript, so that dragging a slider
costs no round trip -- and two implementations of one formula drift. That one lifts the
marked block out of the page, runs it over the records the route actually serves, and
compares every answer with fit.py's own. It needs `node` only to run the page's own code;
there is no test runner and no framework in it.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request

import pytest
from ml_stack.fleet.ui import asset_bytes, serve_page
from ml_stack.serve import fit as fit_mod
from ml_stack.serve.fit import Fit

GIB = 1024 ** 3
ROOM = 96 * GIB


# Three invented models, one of each shape the measurement can take: a small dense one, a
# big one with a cache so fat it fits nobody in a small room, and the same small one again
# with a draft head, so the drafted toggle has both records to switch between.
RECORDS = [
    Fit(model="thornfield-8B-Q4_K_M.gguf", weights=5 * GIB, compute=GIB, room=ROOM,
        per_token=32768, per_seq=8 * 1024 * 1024, cache_type="f16", build="a1b2c3d"),
    Fit(model="thornfield-8B-Q4_K_M.gguf", weights=5 * GIB, draft=GIB // 2, compute=GIB,
        room=ROOM, per_token=32768, per_seq=8 * 1024 * 1024, cache_type="f16",
        spec="draft-mtp", build="a1b2c3d"),
    Fit(model="marrowgate-A3B-UD-Q4_K_XL.gguf", weights=60 * GIB, compute=2 * GIB,
        room=ROOM, per_token=4096, per_seq=0, cache_type="q8_0", build="a1b2c3d"),
    # per_token 1 with a room in the tens of gigabytes is where a double's division stops
    # landing on the right side of an integer, which is why the page floors by hand.
    Fit(model="quillhaven-E2B-it-qat-UD-Q4_K_XL.gguf", weights=3 * GIB, compute=1,
        room=ROOM, per_token=1, per_seq=1, cache_type="f16", build="a1b2c3d"),
]


@pytest.fixture(autouse=True)
def _measurements_in_tmp(fit_files):
    """Both halves of the source of truth in ``tmp_path``, and a fixed room."""
    return fit_files(RECORDS, room=ROOM).shipped


class Page:
    """The fit page on a real socket, through the same `routes` the app mounts."""

    def __init__(self, name: str = "atrium") -> None:
        self.httpd = serve_page(name=name)
        self.port = self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def call(self, path: str, *, ui_header: bool = True) -> tuple[int, dict, str]:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
        if ui_header:
            req.add_header("X-ML-Stack-UI", "1")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                raw, status, kind = r.read(), r.status, r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            raw, status, kind = exc.read(), exc.code, exc.headers.get("Content-Type", "")
        try:
            return status, json.loads(raw or b"{}"), kind
        except ValueError:
            return status, {"raw": raw.decode(errors="replace")}, kind

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def page():
    served = Page()
    try:
        yield served
    finally:
        served.close()


# -- the page ----------------------------------------------------------------------------
class TestThePage:
    def test_it_ships_with_the_package(self):
        from ml_stack.fleet.page import COMPONENTS_DIR

        assert (COMPONENTS_DIR / "fit-view.html").is_file(), "fit-view is missing"

    def test_it_is_served_as_html(self, page):
        status, got, kind = page.call("/ui/fit")
        assert status == 200 and kind.startswith("text/html"), (status, kind)
        assert "What fits" in got["raw"]

    def test_a_link_lands_on_it_without_the_ui_header(self, page):
        """The nav is an ordinary link, and a link cannot set a custom header. The page
        carries no data; the route behind it does, and that one is guarded."""
        status, _, kind = page.call("/ui/fit", ui_header=False)
        assert status == 200 and kind.startswith("text/html")

    def test_a_machine_running_no_daemon_gets_the_fit_view_alone(self, page):
        """`ml-stack-serve fit --ui` puts up the page with no cluster behind it, so it is
        assembled without the screens that would ask one questions."""
        _, got, _ = page.call("/ui/fit")
        assert "<fit-view>" in got["raw"]
        for gone in ("<cluster-view>", "<chat-view>", "<models-view>", "<settings-view>"):
            assert gone not in got["raw"], f"{gone} has no daemon to talk to here"

    def test_every_asset_it_asks_for_exists(self, page):
        _, got, _ = page.call("/ui/fit")
        refs = re.findall(r'(?:src|href)="/ui/static/([^"]+)"', got["raw"])
        assert refs, "a page that references nothing would pass the loop below"
        for ref in refs:
            assert asset_bytes(ref) is not None, f"the page references missing {ref}"

    def test_nothing_is_loaded_over_a_network(self, page):
        """This has to open on a machine that has never been online, so there is no d3 and
        no CDN: both panels are hand-drawn SVG."""
        _, got, _ = page.call("/ui/fit")
        assert not re.search(r'(?:src|href)="https?://', got["raw"]), \
            "the page fetches a library"

    def test_the_script_parses(self, tmp_path):
        """A duplicate declaration anywhere in it stops the whole page loading."""
        from ml_stack.fleet.page import COMPONENTS_DIR
        from ml_stack.ui import load

        node = shutil.which("node")
        if node is None:
            pytest.skip("no node to parse with")
        script = load(COMPONENTS_DIR, ["fit-view"])[0].read().script
        path = tmp_path / "fit-view.js"
        path.write_text(script, encoding="utf-8")
        done = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-500:]

    def test_the_app_offers_it_beside_the_cluster_view(self):
        from ml_stack.fleet.page import COMPONENTS_DIR

        nav = (COMPONENTS_DIR / "fleet-nav.html").read_text(encoding="utf-8")
        tabs = re.search(r"const TABS = \[(.+?)\];", nav, re.S)
        assert tabs, "the nav has no TABS"
        assert '"Fit"' in tabs.group(1) and '"fit"' in tabs.group(1)


# -- the data ----------------------------------------------------------------------------
class TestTheDataRoute:
    def test_it_carries_every_measured_record(self, page):
        status, got, _ = page.call("/ui/fit.json")
        assert status == 200, got
        assert [r["model"] for r in got["records"]] == [f.model for f in
                                                        fit_mod.records()]

    def test_it_carries_this_machines_room_and_name(self, page):
        _, got, _ = page.call("/ui/fit.json")
        assert got["room"] == ROOM
        assert got["name"] == "atrium"

    def test_it_carries_the_rooms_the_chart_draws_faintly(self, page):
        """The familiar card sizes come off fit.py rather than being typed twice."""
        _, got, _ = page.call("/ui/fit.json")
        assert got["vram_gb"] == list(fit_mod.COMMON_VRAM_GB)
        assert got["contexts"] == list(fit_mod.PLOT_CONTEXTS)

    def test_each_record_carries_the_numbers_the_page_composes(self, page):
        _, got, _ = page.call("/ui/fit.json")
        assert len(got["records"]) == len(RECORDS), "an empty route would pass the loop below"
        for row in got["records"]:
            for key in ("weights", "draft", "compute", "per_token", "per_seq",
                        "cache_type", "spec"):
                assert key in row, f"{key} is missing, so the page cannot draw the line"

    def test_it_is_unreachable_without_the_ui_header(self, page):
        """A cross-origin form, image or link cannot set a custom header; this is the
        same guard every other route under /ui is behind."""
        status, got, _ = page.call("/ui/fit.json", ui_header=False)
        assert status == 403 and "header" in got["error"]

    def test_a_machine_that_has_measured_nothing_says_so_rather_than_raising(
            self, page, tmp_path, monkeypatch):
        monkeypatch.setattr(fit_mod, "package_file", lambda: tmp_path / "gone.json")
        _, got, _ = page.call("/ui/fit.json")
        assert got["records"] == [] and got["room"] == ROOM


# -- the arithmetic ----------------------------------------------------------------------

class TestTheSeatingIsWorkedOutOnce:
    """One formula, in one language.

    The page draws; `serve.fit` counts. Every seating the view shows -- what a model costs
    loaded, what one more user costs, how many fit at a context, the longest context this
    many could each be given -- arrives worked out on the route, so there is no second copy
    to keep equal.
    """

    ROOMS = [6 * GIB, 24 * GIB, ROOM, 128 * GIB]
    PEOPLE = [1, 2, 7, 64, 1000]

    def component(self) -> str:
        from ml_stack.fleet.page import COMPONENTS_DIR

        return (COMPONENTS_DIR / "fit-view.html").read_text(encoding="utf-8")

    def test_the_view_works_out_no_seating_of_its_own(self):
        """The measured per-user numbers never reach the browser as something to divide by:
        a copy of the formula there is a copy that can disagree."""
        html = self.component()
        for said in ("per_token", "per_seq", "weights_resident", "Math.floor(a / b)"):
            assert said not in html, f"{said} is the page doing fit.py's arithmetic again"

    def test_the_rooms_it_draws_come_off_the_route(self):
        """A fallback list on the page would be a second opinion about `COMMON_VRAM_GB`."""
        html = self.component()
        assert "vram_gb" in html, "the view reads the rooms from somewhere else"
        assert not re.search(r"\[\s*6,\s*8,\s*12,", html), "the rooms are typed twice"

    def test_the_route_seats_exactly_as_fit_py_does(self, page):
        for room in self.ROOMS:
            for people in self.PEOPLE:
                _, got, _ = page.call(f"/ui/fit.json?room={room}&users={people}")
                assert got["records"], "an empty route would pass the loop below"
                ladder = got["ladder"]
                for row in got["records"]:
                    here = Fit.from_dict(row).at_room(room)
                    assert row["loaded"] == here.loaded()
                    assert row["free"] == here.free()
                    assert row["longest"] == here.longest(people)
                    assert row["seats"] == [here.users(c) for c in ladder]
                    assert row["costs"] == [here.cost(c) for c in ladder]

    def test_the_ladder_holds_every_step_the_slider_stops_at(self, page):
        """The view reads a seating off the ladder; a step missing from it would be read at
        the wrong context."""
        _, got, _ = page.call("/ui/fit.json")
        assert set(got["steps"]) <= set(got["ladder"])
        assert got["steps"] == list(fit_mod.SLIDER_CONTEXTS)

    def test_a_room_that_was_not_asked_for_is_this_machines_own(self, page):
        _, got, _ = page.call("/ui/fit.json")
        assert got["room"] == ROOM and got["at_room"] == ROOM

    def test_asking_for_a_smaller_room_seats_fewer(self, page):
        _, big, _ = page.call(f"/ui/fit.json?room={128 * GIB}")
        _, small, _ = page.call(f"/ui/fit.json?room={8 * GIB}")
        at = big["ladder"].index(32768)
        assert sum(r["seats"][at] for r in big["records"]) \
            > sum(r["seats"][at] for r in small["records"])


class TestTheCliFlag:
    def test_fit_ui_serves_the_page_opens_it_and_waits(self, monkeypatch, capsys):
        """`--ui` puts the app's own routes up on loopback and hands the browser at it.

        The fake `open_path` is where the test gets its hands on a running server: it is
        called with the URL before `serve_forever`, so it fetches the page from a thread
        and then stops the server, which is what a Ctrl-C would have done.
        """
        from ml_stack.fleet import ui as ui_mod
        from ml_stack.serve import cli

        made: dict = {}
        real = ui_mod.serve_page

        def watched(**kw):
            made["server"] = real(**kw)
            return made["server"]

        seen: dict = {}

        def opened(where):
            seen["url"] = str(where)

            def visit():
                req = urllib.request.Request(str(where))
                req.add_header("X-ML-Stack-UI", "1")
                with urllib.request.urlopen(req, timeout=10) as r:
                    seen["status"], seen["page"] = r.status, r.read().decode()
                data = urllib.request.Request(str(where) + ".json")
                data.add_header("X-ML-Stack-UI", "1")
                with urllib.request.urlopen(data, timeout=10) as r:
                    seen["records"] = json.loads(r.read())["records"]
                made["server"].shutdown()

            threading.Thread(target=visit, daemon=True).start()
            return "open"

        monkeypatch.setattr(ui_mod, "serve_page", watched)
        monkeypatch.setattr("ml_stack.platform.open_path", opened)

        assert cli.main(["fit", "--ui"]) == 0
        assert seen["status"] == 200 and "What fits" in seen["page"]
        assert [r["model"] for r in seen["records"]] == [f.model for f in fit_mod.records()]
        assert seen["url"].startswith("http://127.0.0.1:")
        said = capsys.readouterr().err
        assert seen["url"] in said, "it never said where the page is"

    def test_the_flag_is_parsed_not_merely_handled(self, monkeypatch):
        """A flag argparse has not been told about is a flag argparse refuses, however
        carefully `cmd_fit` reads for it."""
        from ml_stack.serve import cli

        called: list[argparse.Namespace] = []
        monkeypatch.setattr(cli, "cmd_fit", lambda args: called.append(args) or 0)
        assert cli.main(["fit", "--ui"]) == 0
        assert called and called[0].ui is True
        assert cli.main(["fit"]) == 0 and called[1].ui is False

    def test_a_server_it_puts_up_answers_only_loopback(self):
        """There is no cluster passphrase behind this page, so there had better be no
        address on it either."""
        server = serve_page(name="atrium")
        try:
            assert server.server_address[0] == "127.0.0.1"
        finally:
            server.server_close()


# -- the rates beside it ------------------------------------------------------------------
class TestTheRatesRoute:
    """`ml-stack-bench show --rates` as data, over a store built here.

    Nothing measures anything: two invented runs are written into a store in ``tmp_path``
    and `ml_stack.bench.home_dir()` is pointed at it, so no test can reach the runs this
    machine has kept.
    """

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        from ml_stack.bench.keep import SHORT, save
        from ml_stack.bench.score import Row

        home = tmp_path / "bench"
        home.mkdir()
        monkeypatch.setenv("MLSTACK_BENCH_HOME", str(home))
        where = home / "runs.ladybug"

        def run(label: str, *, seconds: float, tokens: int, right: bool,
                model: str = "", questions: int = SHORT) -> None:
            # a short run's worth: below `SHORT` a run is evidence that something ran, and
            # `composed` leaves it out, so a two-question fixture would compose nothing
            rows = [Row(label=label, question=f"who welds frame {n}?", seconds=seconds,
                        calls=3, processed_tokens=tokens, completion_tokens=64,
                        expected=["person:iris"],
                        shown=["person:iris"] if right or n % 2 else ["topic:welding"])
                    for n in range(questions)]
            save(where, rows, held={"context": 32768, "slots": 2, "host": "atrium",
                                    "model": model or f"{label}-Q4_K_M.gguf",
                                    "kv_and_run_bytes": 4 * GIB})

        # two models, and one of them measured twice: `composed` has something to compose
        run("thornfield-8B", seconds=4.0, tokens=900, right=True)
        run("thornfield-8B-tight", seconds=3.0, tokens=700, right=True,
            model="thornfield-8B-Q4_K_M.gguf")
        run("marrowgate-A3B", seconds=12.0, tokens=2400, right=False)
        return where

    def test_it_carries_every_kept_run_with_its_rates(self, page, store):
        status, got, _ = page.call("/ui/rates.json")
        assert status == 200, got
        labels = {r["label"] for r in got["runs"]}
        assert {"thornfield-8B", "marrowgate-A3B"} <= labels
        for row in got["runs"]:
            for key in ("right", "recall", "precision", "seconds", "paid_tokens",
                        "kv_bytes", "questions", "front"):
                assert key in row, f"{key} is missing, so the page cannot place the point"

    def test_the_rates_are_the_ones_the_command_prints(self, page, store):
        """Not recomputed here: `score.derived` is what both read."""
        from ml_stack.bench.keep import _kept
        from ml_stack.bench.score import derived

        _, got, _ = page.call("/ui/rates.json")
        by_label = {r["label"]: r for r in got["runs"] if not r["composed"]}
        for one in _kept(store):
            mine = derived(one)
            theirs = by_label[one["label"]]
            assert theirs["right"] == pytest.approx(mine["right"])
            assert theirs["seconds"] == pytest.approx(mine["seconds"])
            assert theirs["right_per_minute"] == pytest.approx(mine["right_per_minute"])

    def test_the_frontier_is_marked_for_every_cost(self, page, store):
        """Worked out for all three, so switching the axis on the page fetches nothing."""
        from ml_stack.bench.show import AXES

        _, got, _ = page.call("/ui/rates.json")
        assert set(got["axes"]) == set(AXES)
        for cost in AXES:
            on = [r for r in got["runs"] if cost in r["front"]]
            assert on, f"nothing is on the frontier for {cost}"
            # the most accurate run is on every frontier: nothing beats it on accuracy
            assert max(r["right"] for r in got["runs"]) == max(r["right"] for r in on)

    def test_the_page_draws_the_key_the_frontier_was_worked_out_on(self, page, store):
        """Per question for time and tokens, a total for memory: the same map the command
        uses, sent with the axes so the point and its frontier mark agree."""
        from ml_stack.bench.score import COSTS

        _, got, _ = page.call("/ui/rates.json")
        assert got["keys"] == dict(COSTS)
        assert got["keys"]["seconds"] == "seconds_per_question"
        assert got["keys"]["kv_bytes"] == "kv_bytes"
        assert "per question" in got["axes"]["seconds"]
        for row in got["runs"]:
            for cost, key in got["keys"].items():
                assert row[key] > 0, f"{key} is missing, so the page cannot place the point"

    def test_a_model_composed_is_marked_as_one(self, page, store):
        """A square on the chart, not a circle: accuracy from a model's largest run and
        cost from its fastest that held it is not itself a run anybody made."""
        _, got, _ = page.call("/ui/rates.json")
        assert any(r["composed"] for r in got["runs"])

    def test_a_machine_that_has_measured_nothing_says_so(self, page, tmp_path, monkeypatch):

        monkeypatch.setenv("MLSTACK_BENCH_HOME", str(tmp_path / "empty"))
        status, got, _ = page.call("/ui/rates.json")
        assert status == 200 and got["runs"] == []

    def test_it_is_unreachable_without_the_ui_header(self, page, store):
        status, got, _ = page.call("/ui/rates.json", ui_header=False)
        assert status == 403 and "header" in got["error"]


# -- the telemetry view ------------------------------------------------------------------
#
# The third view: not what a model *would* cost, but what answering has already cost. The
# route is a reader -- of this process, when it answers anything, and otherwise of another
# page's `/metrics`, fetched from here because a page on loopback has no reason to allow a
# cross-origin read. Both halves are driven over a real socket; the second one against a
# real `AskRoutes` server, so what is being read is the format that is actually served and
# not a fixture shaped like it.


class Answering:
    """A real `AskRoutes` server on a free port, answering with no model at all."""

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from ml_stack.graph.serve import AskRoutes

        class Handler(AskRoutes, BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def model_name(self):
                return "quillhaven-E2B-Q4.gguf"

            def do_GET(self):
                if self.path == "/metrics":
                    self.handle_metrics()
                else:
                    self.send_response(404)
                    self.end_headers()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def answering():
    served = Answering()
    try:
        yield served
    finally:
        served.close()


class TestTheTelemetryView:
    def test_it_is_offered_beside_what_fits_and_what_it_cost(self):
        from ml_stack.fleet.page import COMPONENTS_DIR

        html = (COMPONENTS_DIR / "fit-view.html").read_text(encoding="utf-8")
        views = re.search(r"const VIEWS = \[(.+?)\];", html, re.S)
        assert views and '"telemetry"' in views.group(1)
        assert "/ui/telemetry.json" in html, "the view reads no route"

    def test_a_daemon_that_answers_nothing_says_so_rather_than_showing_zeros(self, page):
        """Zeros would read as a server that answered nothing; this one was never asked."""
        status, got, _ = page.call("/ui/telemetry.json")
        assert status == 200 and got["serving"] is False
        assert "answers no questions itself" in got["note"]

    def test_a_host_that_does_answer_hands_over_its_own_record(self, page):
        page.httpd.RequestHandlerClass.ui.answers = lambda: {"answers": 3, "totals": {"calls": 7}}
        try:
            status, got, _ = page.call("/ui/telemetry.json")
            assert status == 200 and got["serving"] is True
            assert got["metrics"]["answers"] == 3 and got["metrics"]["totals"]["calls"] == 7
        finally:
            page.httpd.RequestHandlerClass.ui.answers = None

    def test_a_counter_that_raises_is_a_note_and_not_a_broken_page(self, page):
        def broken():
            raise RuntimeError("the ring went away")

        page.httpd.RequestHandlerClass.ui.answers = broken
        try:
            status, got, _ = page.call("/ui/telemetry.json")
            assert status == 200 and "the ring went away" in got["error"]
        finally:
            page.httpd.RequestHandlerClass.ui.answers = None

    def test_it_reads_another_pages_metrics_from_here_and_not_from_the_browser(
            self, page, answering):
        status, got, _ = page.call(f"/ui/telemetry.json?from={answering.url}/metrics")
        assert status == 200 and got["serving"] is True
        assert got["source"] == f"{answering.url}/metrics"
        assert got["metrics"]["model"] == "quillhaven-E2B-Q4.gguf"
        assert got["metrics"]["answers"] == 0 and got["metrics"]["totals"]["answers"] == 0

    def test_a_page_that_is_not_there_is_a_note_and_keeps_the_view_polling(self, page):
        """The usual reason is that it has not been started yet."""
        status, got, _ = page.call("/ui/telemetry.json?from=http://127.0.0.1:1/metrics")
        assert status == 200 and got["serving"] is False and got["error"]

    def test_only_an_http_address_is_fetched(self, page):
        status, got, _ = page.call("/ui/telemetry.json?from=file:///etc/hosts")
        assert status == 200 and "http://" in got["error"]

    def test_it_is_unreachable_without_the_ui_header(self, page):
        status, got, _ = page.call("/ui/telemetry.json", ui_header=False)
        assert status == 403 and "header" in got["error"]
