"""``POST /decide`` on the daemon: the same bearer token as every route, a size limit, and
the decision a served model gives."""

from __future__ import annotations

import json
import threading

import pytest
from decide_fakes import logprob_handler

from poolhouse.fleet.api import Daemon, make_handler
from poolhouse.fleet.daemon import load_or_create_token
from poolhouse.fleet.deciding import MAX_REQUEST, Deciding
from poolhouse.fleet.jobs import JobRunner
from poolhouse.http import Server, ServerError, request_bytes


@pytest.fixture
def api(tmp_path, server):
    chat = server(logprob_handler(lambda user: {"A": 0.1, "B": 0.9}))
    from poolhouse.decide import router
    files = tmp_path / "files"
    files.mkdir()
    runner = JobRunner(tmp_path / "traind")
    decide = Deciding(config=router.Config(backend="logprob", url=chat.base_url))
    good = load_or_create_token(tmp_path / "root")
    httpd = Server(("127.0.0.1", 0), make_handler(Daemon(runner, files, good, decide=decide)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(body, *, token=good, raw=None):
        data = raw if raw is not None else json.dumps(body).encode()
        try:
            got = request_bytes(f"{base}/decide", data=data, method="POST", token=token,
                                headers={"Content-Type": "application/json"}, timeout=10)
        except ServerError as exc:
            return exc.status, json.loads(exc.body or "{}")
        return got.status, json.loads(got.body)

    try:
        yield post, chat, decide
    finally:
        runner.shutdown()
        httpd.shutdown()
        httpd.server_close()


def test_a_decision_comes_back_with_a_probability_for_every_option(api):
    post, chat, _ = api
    code, got = post({"question": "Which?", "state": "s", "options": {"a": "first", "b": ""}})
    assert code == 200
    decision = got["decision"]
    assert decision["choice"] == "b" and decision["scores"]["b"] == pytest.approx(0.9)
    assert decision["backend"] == "logprob"
    assert any(r[1].endswith("completions") for r in chat.requests)


def test_the_bearer_token_is_required(api):
    post, chat, _ = api
    code, _ = post({"question": "q", "options": ["a", "b"]}, token="wrong")
    assert code == 401
    assert not [r for r in chat.requests if r[1].endswith("completions")]


def test_a_request_over_the_limit_is_refused_before_it_is_read(api):
    post, _, _ = api
    code, got = post(None, raw=b'{"state": "' + b"x" * (MAX_REQUEST + 10) + b'"}')
    assert code == 413 and str(MAX_REQUEST) in got["error"]


@pytest.mark.parametrize("body", [
    {"options": ["a", "b"]}, {"question": "q"}, {"question": "q", "options": ["only"]},
    {"question": "q", "options": "ab"}, {"question": "q", "options": ["a", "a"]},
    {"question": "q", "options": ["a", "b"], "abstain_below": "high"}, [1, 2]])
def test_a_malformed_request_is_a_400_that_names_the_problem(api, body):
    post, _, _ = api
    code, got = post(body)
    assert code == 400 and got["error"]


def test_a_body_that_is_not_json_is_a_400(api):
    post, _, _ = api
    assert post(None, raw=b"{nope")[0] == 400


def test_abstain_below_marks_a_low_confidence_answer(api):
    post, _, _ = api
    _, got = post({"question": "q", "options": ["a", "b"], "abstain_below": 0.95})
    assert got["decision"]["abstained"] is True


def test_no_backend_available_is_a_503_that_says_why(tmp_path):
    from poolhouse.decide import router
    answer = Deciding(config=router.Config(url="http://127.0.0.1:9", order=("logprob",))).answer(
        {"question": "q", "options": ["a", "b"]})
    assert answer[0] == 503 and "did not answer" in answer[1]["error"]


def test_a_daemon_without_a_decider_answers_501(tmp_path):
    files = tmp_path / "files"
    files.mkdir()
    runner = JobRunner(tmp_path / "traind")
    good = load_or_create_token(tmp_path / "root")
    httpd = Server(("127.0.0.1", 0), make_handler(Daemon(runner, files, good)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with pytest.raises(ServerError) as err:
            request_bytes(f"http://127.0.0.1:{httpd.server_address[1]}/decide", data=b"{}",
                          method="POST", token=good, timeout=5)
        assert err.value.status == 501
    finally:
        runner.shutdown()
        httpd.shutdown()
        httpd.server_close()


def test_the_decider_for_a_server_is_built_once(api):
    post, _, decide = api
    for _ in range(3):
        post({"question": "q", "options": ["a", "b"]})
    assert len(decide._configs) == 1


def test_the_mcp_tool_decides_with_descriptions_and_an_abstain_threshold(server, monkeypatch):
    from poolhouse import mcp
    from poolhouse.decide import router
    chat = server(logprob_handler(lambda user: {"A": 0.3, "B": 0.7}))
    monkeypatch.setattr(router, "shared", lambda backend, url: router.Config(backend=backend, url=chat.base_url))
    got = mcp.decide("Which?", ["x=first option", "y"], state_text="the state",
                     backend="logprob", abstain_below=0.9)
    assert (got["choice"], got["abstained"]) == ("y", True)
    sent = json.loads(chat.requests[-1][2])["messages"][-1]["content"]
    assert "A. x - first option" in sent and "the state" in sent
