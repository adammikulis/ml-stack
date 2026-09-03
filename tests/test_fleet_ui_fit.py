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
from ml_stack.fleet.ui import ASSETS, asset_bytes, serve_page
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
def _measurements_in_tmp(tmp_path, monkeypatch):
    """Both halves of the source of truth in ``tmp_path``, and a fixed room."""
    shipped = tmp_path / "ssot" / "fit.json"
    shipped.parent.mkdir(parents=True, exist_ok=True)
    shipped.write_text(json.dumps([f.as_dict() for f in RECORDS], indent=2), encoding="utf-8")
    monkeypatch.setattr(fit_mod, "package_file", lambda: shipped)
    monkeypatch.setattr(fit_mod, "writable_file", lambda: shipped)
    monkeypatch.setenv("MLSTACK_FIT_FILE", str(tmp_path / "local" / "fit.json"))
    monkeypatch.setattr("ml_stack.hub.room", lambda: ROOM)
    return shipped


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
        assert asset_bytes("fit.html") is not None, "fit.html is missing from web/"

    def test_it_is_served_as_html(self, page):
        status, got, kind = page.call("/ui/fit")
        assert status == 200 and kind == "text/html", (status, kind)
        assert "What fits" in got["raw"]

    def test_a_link_lands_on_it_without_the_ui_header(self, page):
        """The nav is an ordinary link, and a link cannot set a custom header. The page
        carries no data; the route behind it does, and that one is guarded."""
        status, _, kind = page.call("/ui/fit", ui_header=False)
        assert status == 200 and kind == "text/html"

    def test_every_asset_it_asks_for_exists(self):
        html = asset_bytes("fit.html")[0].decode()
        for ref in re.findall(r'(?:src|href)="/ui/static/([^"]+)"', html):
            assert asset_bytes(ref) is not None, f"fit.html references missing {ref}"

    def test_nothing_is_loaded_over_a_network(self):
        """This has to open on a machine that has never been online, so there is no d3 and
        no CDN: both panels are hand-drawn SVG."""
        html = asset_bytes("fit.html")[0].decode()
        assert not re.search(r'(?:src|href)="https?://', html), "the page fetches a library"

    def test_the_script_parses(self):
        """A duplicate declaration anywhere in it stops the whole page loading, and the
        page is one script -- the same guard app.js is under."""
        node = shutil.which("node")
        if node is None:
            pytest.skip("no node to parse with")
        html = asset_bytes("fit.html")[0].decode()
        script = re.search(r"<script>\n(.*)</script>", html, re.S)
        assert script, "the page has no script"
        done = subprocess.run([node, "--check", "-"], input=script.group(1),
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-500:]

    def test_the_app_offers_it_beside_the_cluster_view(self):
        js = asset_bytes("app.js")[0].decode()
        tabs = re.search(r"const TABS = \[(.+?)\];", js, re.S)
        assert tabs, "the nav has no TABS"
        assert '"Fit"' in tabs.group(1) and "/ui/fit" in tabs.group(1)


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

BEGIN = "// BEGIN FIT ARITHMETIC"
END = "// END FIT ARITHMETIC"


def _arithmetic() -> str:
    """The page's own formula, lifted out between its markers."""
    html = (ASSETS / "fit.html").read_text(encoding="utf-8")
    start, stop = html.index(BEGIN), html.index(END)
    return html[start:stop]


class TestThePageAndFitPyAgree:
    """One formula, written twice -- so it is tested as one.

    The page cannot call `Fit.users`: a slider that asked the daemon on every pixel would
    be unusable, and the whole point of the two composing numbers is that the arithmetic
    over them is trivial. What is not trivial is keeping the two copies equal, which is
    what this does: the marked block out of the page, run over the records the route
    serves, against fit.py's own answers.
    """

    ROOMS = [6 * GIB, 24 * GIB, ROOM, 128 * GIB]
    CONTEXTS = [1024, 4096, 32768, 131072, 262144]
    PEOPLE = [1, 2, 7, 64, 1000]

    def test_the_room_sizes_are_not_typed_twice(self):
        """The page falls back to a list of its own when the route says nothing; that
        fallback is fit.py's list, or the fallback is a second opinion."""
        html = (ASSETS / "fit.html").read_text(encoding="utf-8")
        said = re.search(r"const DEFAULT_ROOMS_GB = \[([^\]]+)\]", html)
        assert said, "the page has no room presets"
        assert [int(n) for n in said.group(1).split(",")] == list(fit_mod.COMMON_VRAM_GB)

    def test_the_page_and_fit_py_answer_the_same(self, page, tmp_path):
        node = shutil.which("node")
        if node is None:
            pytest.skip("no node to run the page's own code with")

        _, got, _ = page.call("/ui/fit.json")
        asked = {"records": got["records"], "rooms": self.ROOMS,
                 "contexts": self.CONTEXTS, "people": self.PEOPLE}
        script = tmp_path / "check.js"
        script.write_text(_arithmetic() + """
const asked = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
const out = [];
for (const f of asked.records) {
  for (const room of asked.rooms) {
    for (const ctx of asked.contexts) out.push(Fit.users(f, room, ctx));
    for (const ctx of asked.contexts) out.push(Fit.line(f, ctx)[0], Fit.line(f, ctx)[1]);
    for (const n of asked.people) out.push(Fit.longest(f, room, n));
  }
}
console.log(JSON.stringify(out));
""", encoding="utf-8")
        asked_file = tmp_path / "asked.json"
        asked_file.write_text(json.dumps(asked), encoding="utf-8")
        done = subprocess.run([node, str(script), str(asked_file)],
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-500:]
        theirs = json.loads(done.stdout)

        mine: list[int] = []
        for row in got["records"]:
            for room in self.ROOMS:
                here = Fit.from_dict(row).at_room(room)
                mine += [here.users(c) for c in self.CONTEXTS]
                for context in self.CONTEXTS:
                    mine += list(here.line(context))
                mine += [here.longest(n) for n in self.PEOPLE]

        assert len(theirs) == len(mine)
        for i, (a, b) in enumerate(zip(theirs, mine)):
            assert a == b, f"answer {i}: the page says {a}, fit.py says {b}"


# -- the flag ----------------------------------------------------------------------------
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
    and `ml_stack.graph.bench.HOME` is pointed at it, so no test can reach the runs this
    machine has kept.
    """

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        import ml_stack.graph.bench as bench
        from ml_stack.graph.bench.keep import SHORT, save
        from ml_stack.graph.bench.score import Row

        home = tmp_path / "bench"
        home.mkdir()
        monkeypatch.setattr(bench, "HOME", home)
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
        from ml_stack.graph.bench.keep import _kept
        from ml_stack.graph.bench.score import derived

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
        from ml_stack.graph.bench.show import AXES

        _, got, _ = page.call("/ui/rates.json")
        assert set(got["axes"]) == set(AXES)
        for cost in AXES:
            on = [r for r in got["runs"] if cost in r["front"]]
            assert on, f"nothing is on the frontier for {cost}"
            # the most accurate run is on every frontier: nothing beats it on accuracy
            assert max(r["right"] for r in got["runs"]) == max(r["right"] for r in on)

    def test_a_model_composed_is_marked_as_one(self, page, store):
        """A square on the chart, not a circle: accuracy from a model's largest run and
        cost from its fastest that held it is not itself a run anybody made."""
        _, got, _ = page.call("/ui/rates.json")
        assert any(r["composed"] for r in got["runs"])

    def test_a_machine_that_has_measured_nothing_says_so(self, page, tmp_path, monkeypatch):
        import ml_stack.graph.bench as bench

        monkeypatch.setattr(bench, "HOME", tmp_path / "empty")
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
        html = asset_bytes("fit.html")[0].decode()
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
