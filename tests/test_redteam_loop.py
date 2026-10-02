"""The tool-calling loop under attack, with a scripted model that obeys (or ignores) every
instruction it is shown. Needs the redteam extra and ml_stack.agent: run with --redteam."""

from __future__ import annotations

import asyncio

import pytest

from ml_stack.redteam.lab import lab as make_lab
from ml_stack.redteam.report import Report
from ml_stack.redteam.scenarios import Options
from ml_stack.redteam.stub import StubModel

pytestmark = pytest.mark.redteam


def attacked(mode: str) -> Report:
    loop = pytest.importorskip("ml_stack.redteam.scenarios.loop")
    stub = StubModel(mode)
    report = Report()
    try:
        with make_lab(model_url=stub.base_url) as one:
            asyncio.run(loop.run(one, report, Options(limit=1)))
    finally:
        stub.close()
    return report


@pytest.fixture(scope="module")
def gullible() -> Report:
    return attacked("gullible")


def cell(report: Report, attack_class: str, attack_id: str, arm: str):
    found = [a for a in report.attempts
             if (a.attack_class, a.attack_id, a.arm) == (attack_class, attack_id, arm)]
    assert len(found) == 1, (attack_class, attack_id, arm, len(found))
    return found[0]


def test_an_obedient_model_with_no_guard_is_talked_into_each_dangerous_call(gullible):
    for kind in ("write", "send", "exfil", "ssrf"):
        one = cell(gullible, "direct-injection", f"plain-{kind}", "bare")
        assert one.succeeded and one.attempted and not one.blocked, kind


def test_the_web_reader_stops_a_request_for_an_address_on_this_side_of_the_router(gullible):
    one = cell(gullible, "direct-injection", "plain-ssrf", "web-guard")
    assert one.attempted and one.blocked and not one.succeeded


def test_the_web_reader_does_not_stop_a_write_or_a_report(gullible):
    for kind in ("write", "send", "exfil"):
        assert cell(gullible, "direct-injection", f"plain-{kind}", "web-guard").succeeded, kind


def test_the_policy_intervention_stops_every_one_of_them(gullible):
    for kind in ("write", "send", "exfil", "ssrf"):
        one = cell(gullible, "direct-injection", f"plain-{kind}", "web-guard+policy")
        assert one.attempted and one.blocked and not one.succeeded, kind


def test_an_instruction_in_a_visible_page_is_followed_and_then_stopped_the_same_way(gullible):
    assert cell(gullible, "injected-page", "visible-write", "bare").succeeded
    assert cell(gullible, "injected-page", "visible-write", "web-guard+policy").blocked


def test_a_jailbreak_template_around_the_request_makes_no_difference_to_an_obedient_model(
        gullible):
    assert cell(gullible, "jailbreak-template", "aim", "bare").succeeded


def test_a_model_that_obeys_nothing_is_never_attacked_successfully():
    report = attacked("resistant")
    assert report.attempts and not any(a.succeeded or a.attempted for a in report.attempts)
