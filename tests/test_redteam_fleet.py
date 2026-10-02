"""The daemon attacks, against the daemon's real request handler on a loopback port."""

from __future__ import annotations

import asyncio

import pytest

from ml_stack.http import request_bytes
from ml_stack.redteam import daemon
from ml_stack.redteam.evidence import Honeypot
from ml_stack.redteam.lab import lab as make_lab
from ml_stack.redteam.report import Report
from ml_stack.redteam.scenarios import Options, fleet


@pytest.fixture
def served(tmp_path):
    with daemon.running(tmp_path) as running, make_lab(
            served=running) as one:
        yield one


def attempts(served, step) -> Report:
    report = Report()
    step(served, report, Options())
    return report


def test_no_protected_route_answers_without_the_right_token(served):
    report = attempts(served, fleet.run_auth)
    assert len(report.attempts) > 80
    assert [a.attack_id for a in report.attempts if a.succeeded] == []


def test_a_correct_token_does_open_a_protected_route(served):
    got = request_bytes(f"{served.daemon_url}/jobs", token=served.token, timeout=10)
    assert got.status == 200


def test_no_path_reaches_a_file_outside_the_files_directory(served):
    report = attempts(served, fleet.run_paths)
    traversal = [a for a in report.attempts if a.attack_class == "path-traversal"]
    assert len(traversal) == 8 and not any(a.succeeded for a in traversal)


def test_a_handler_that_raises_drops_the_connection_and_that_counts_as_a_failure():
    class Raising(Honeypot):
        def get(self, path):
            raise RuntimeError("handler bug")

    server = Raising()
    try:
        got = fleet.raw(server.base_url, fleet.get("/x"))
    finally:
        server.close()
    assert got.status == 0 and fleet.broke(got)


def test_a_refusal_the_daemon_writes_itself_is_not_a_failure(served):
    report = attempts(served, fleet.run_malformed)
    lost = {a.attack_id for a in report.attempts if not a.succeeded}
    assert {"POST/availability:not-json", "POST/availability:empty", "request-line-only",
            "bad-version"} <= lost


def test_broke_says_a_dropped_connection_a_hang_and_a_500_are_failures_and_a_501_is_not():
    assert fleet.broke(fleet.Reply(0, b"", 0.1))
    assert fleet.broke(fleet.Reply(0, b"", 3.0, hung=True))
    assert fleet.broke(fleet.Reply(500, b"x", 0.1))
    assert not fleet.broke(fleet.Reply(501, b"x", 0.1))
    assert not fleet.broke(fleet.Reply(400, b"x", 0.1))
    assert not fleet.broke(fleet.Reply(0, b"<html>", 0.1))


def test_raw_reads_the_status_line_and_the_body_of_a_real_answer(served):
    got = fleet.raw(served.daemon_url, fleet.get("/health"))
    assert got.status == 200 and b'"ok": true' in got.body


def test_the_whole_scenario_runs_on_its_own(served):
    report = Report()
    asyncio.run(fleet.run(served, report, Options(limit=2)))
    assert {a.attack_class for a in report.attempts} >= {
        "auth-bypass", "malformed-request", "oversized-body", "path-traversal", "proxy-route"}
