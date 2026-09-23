"""``ml-stack-walk``: which screens it walks, what it prints, and a real walk of a daemon.

The browser tests drive headless Chromium against `test_fleet_ui.Serving` -- the daemon on
a real socket with the real routes behind it -- through `ml_stack.walk.ops`, which is the
same path the command takes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_stack.log import to
from ml_stack.walk import ops
from ml_stack.walk.cli import report
from ml_stack.walk.ops import FLEET, GRAPH, Stop, Walk


class TestWhichScreens:
    def test_no_names_walks_every_screen_the_page_has(self):
        assert ops.screens_of("fleet", []) == FLEET
        assert ops.screens_of("graph", []) == GRAPH

    def test_names_come_back_in_the_pages_own_order(self):
        assert ops.screens_of("fleet", ["fit", "cluster"]) == ("cluster", "fit")

    def test_a_screen_the_page_does_not_have_is_refused_by_name(self):
        with pytest.raises(ValueError, match="no screen called 'dashboard'"):
            ops.screens_of("fleet", ["dashboard", "cluster"])

    def test_a_page_nobody_serves_is_refused(self):
        with pytest.raises(ValueError, match="no page called 'wiki'"):
            ops.screens_of("wiki", [])
        with pytest.raises(ValueError, match="no page called 'wiki'"):
            ops.walk(Walk(page="wiki"))


def printed(stops, chars=200):
    """``(exit code, what a person reads)`` for that report."""
    lines = []
    with to(lambda stream, text: lines.append(text)):
        code = report(stops, Path("/shots"), chars)
    return code, "".join(lines)


class TestWhatItPrints:
    def test_a_clean_walk_prints_each_screen_and_exits_zero(self):
        code, said = printed([Stop("cluster", text="Cluster\nOne machine.",
                                   shot=Path("/shots/01-cluster.png"))])
        assert code == 0
        assert "01-cluster.png" in said and "One machine." in said
        assert "1 screen(s) walked, 0 skipped, 0 with errors" in said

    def test_a_screen_that_errored_is_named_and_the_walk_exits_one(self):
        code, said = printed([Stop("chat", errors=("pageerror: nope",))])
        assert code == 1
        assert "pageerror: nope" in said
        assert "1 with errors" in said

    def test_a_skipped_screen_says_why_and_is_not_a_failure(self):
        code, said = printed([Stop("fit", skipped="still in first-run")])
        assert code == 0
        assert "skipped: still in first-run" in said
        assert "0 screen(s) walked, 1 skipped" in said

    def test_a_long_screen_is_cut_where_the_reader_asked(self):
        _, said = printed([Stop("models", text="x" * 500,
                                shot=Path("/shots/01-models.png"))], chars=20)
        assert "x" * 20 + "…" in said
        assert "x" * 21 not in said


# -- a real daemon, driven ------------------------------------------------------------------

pw = pytest.importorskip("playwright.sync_api", reason="ml-stack[scrape]")


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    """A daemon on a real socket that has not been set up, and a cache to walk it from."""
    from test_fleet_ui import Serving

    monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "cache"))
    served = Serving(tmp_path)
    try:
        yield served
    finally:
        served.httpd.shutdown()
        served.httpd.server_close()


def walked(served, tmp_path, play, screens=FLEET, **over):
    """Walk that daemon, skipping when no Chromium will launch."""
    what = Walk(page="fleet", base=f"http://127.0.0.1:{served.port}",
                out=tmp_path / "shots", screens=screens, **over)
    try:
        return ops.fleet(what, play)
    except pw.Error as exc:                            # pragma: no cover - depends on setup
        pytest.skip(f"chromium did not launch: {exc}")


@pytest.mark.slow
class TestWalkingTheDaemon:
    def test_the_wizard_is_walked_as_far_as_it_goes_without_changing_the_machine(
            self, daemon, tmp_path, playwright):
        stops = walked(daemon, tmp_path, playwright)
        by_name = {stop.screen: stop for stop in stops}

        assert [s.screen for s in stops if s.shot] == ["name", "clusters", "job", "start"]
        assert all(stop.ok for stop in stops)
        assert "Set up this machine" in by_name["name"].text
        assert "Clusters" in by_name["clusters"].text
        assert "preferences" in by_name["run"].skipped
        assert "first-run" in by_name["cluster"].skipped

    def test_every_screen_it_walked_left_a_picture(self, daemon, tmp_path, playwright):
        stops = walked(daemon, tmp_path, playwright)
        for stop in stops:
            if stop.shot:
                assert stop.shot.exists() and stop.shot.stat().st_size > 0

    def test_naming_two_screens_walks_only_those(self, daemon, tmp_path, playwright):
        stops = walked(daemon, tmp_path, playwright,
                       screens=ops.screens_of("fleet", ["job", "name"]))
        assert [stop.screen for stop in stops] == ["name", "job"]
        assert "What should this machine do?" in stops[1].text

    def test_a_port_with_nothing_on_it_is_refused_rather_than_walked(self, tmp_path,
                                                                     monkeypatch,
                                                                     playwright):
        import socket

        monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "cache"))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            dead = sock.getsockname()[1]
        with pytest.raises(ops.WalkFailed, match="did not answer"):
            ops.fleet(Walk(page="fleet", base=f"http://127.0.0.1:{dead}",
                           out=tmp_path / "shots", screens=FLEET, timeout_s=5.0),
                      playwright)


# -- a model that streams slowly, behind a real page ----------------------------------------

SLOW = ("Ada Lovelace ", "works ", "on ", "iron", ".")

GRAPH_OF_TWO = {
    "nodes": [{"id": "person:ada", "label": "Ada Lovelace", "kind": "person", "attrs": {},
               "messages": []},
              {"id": "topic:iron", "label": "iron", "kind": "topic", "attrs": {},
               "messages": []}],
    "edges": [{"source": "person:ada", "rel": "works_on", "target": "topic:iron"}],
    "messages": {},
}


@pytest.fixture
def slow_model(monkeypatch):
    """A model streaming for longer than a screen is given to settle."""
    from ml_stack.testing.fakes import FakeLlamaServer, Served

    monkeypatch.setattr(ops, "SETTLE_MS", 300)
    fake = FakeLlamaServer(Served(pieces=SLOW, gap=0.4))
    try:
        yield fake
    finally:
        fake.close()


@pytest.mark.slow
def test_asking_the_graph_page_waits_for_the_whole_streamed_answer(slow_model, tmp_path,
                                                                    monkeypatch, playwright):
    from conftest import threaded_server

    from ml_stack.client import Client
    from ml_stack.graph.page import render
    from ml_stack.graph.serve import Handler

    monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "cache"))
    page = tmp_path / "page.html"
    page.write_text(render(GRAPH_OF_TWO, title="Invented"), encoding="utf-8")

    class Slow(Handler):
        def client_on_slot(self, *, index=0, **over):
            return Client(slow_model.base_url)

        def log_message(self, *args):
            pass

    with threaded_server(Slow.configured(name="Slow", site=page, graph=GRAPH_OF_TWO)) as url:
        what = Walk(page="graph", base=url, out=tmp_path / "shots", screens=("ask",),
                    ask="who works on iron?", answer_s=60.0)
        try:
            stops = ops.graph(what, playwright)
        except pw.Error as exc:                        # pragma: no cover - depends on setup
            pytest.skip(f"chromium did not launch: {exc}")
    assert [s.screen for s in stops] == ["ask"] and stops[0].ok, stops
    assert "Ada Lovelace works on iron." in stops[0].text.splitlines(), stops[0].text


@pytest.mark.slow
def test_saying_something_in_the_fleet_chat_waits_for_the_whole_reply(slow_model, tmp_path,
                                                                      monkeypatch,
                                                                      playwright):
    import time

    from test_fleet_ui import WORDS, Serving

    from ml_stack.fleet.conversations import Conversations
    from ml_stack.fleet.discovery import join_cluster
    from ml_stack.fleet.serving import Serving as Models

    monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "cache"))
    served = Serving(tmp_path, name="laptop")
    try:
        join_cluster(WORDS, group="home", path=served.keyfile)
        served.call("/ui/setup/join", method="POST",
                    body={"passphrase": WORDS, "group": "home"})
        served.call("/ui/setup/done", method="POST")
        served.ui.conversations = Conversations(tmp_path / "chats")
        models = Models(tmp_path / "serving.json")
        models.register(slow_model.port, ["quince-2b.gguf"])
        served.ui.serving = models
        served.ui._peers = (time.time(), [])
        stops = walked(served, tmp_path, playwright, screens=("chat",), passphrase=WORDS,
                       say="hello", answer_s=60.0)
    finally:
        served.httpd.shutdown()
        served.httpd.server_close()
    assert [s.screen for s in stops] == ["chat"] and stops[0].ok, stops
    assert "hello" in stops[0].text and "".join(SLOW) in stops[0].text, stops[0].text
