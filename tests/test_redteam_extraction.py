"""The scenario that needs no model: which hiding places an instruction survives."""

from __future__ import annotations

import asyncio

import pytest

from poolhouse.redteam import pages
from poolhouse.redteam.lab import lab as make_lab
from poolhouse.redteam.report import Report
from poolhouse.redteam.scenarios import Options, extraction


@pytest.fixture(scope="module")
def stood_up():
    with make_lab() as one:
        yield one


def run(stood_up, options: Options | None = None) -> Report:
    report = Report()
    asyncio.run(extraction.run(stood_up, report, options or Options()))
    return report


def test_every_hiding_place_is_tried_once(stood_up):
    report = run(stood_up)
    assert sorted(a.attack_id for a in report.attempts) == sorted(pages.variants())


def test_an_instruction_in_visible_text_and_in_zero_size_text_survives_reading(stood_up):
    won = {a.attack_id for a in run(stood_up).attempts if a.succeeded}
    assert {"visible", "css_font_zero", "css_white_on_white"} <= won


def test_an_instruction_in_a_comment_or_in_alt_text_does_not_survive_reading(stood_up):
    lost = {a.attack_id for a in run(stood_up).attempts if not a.succeeded}
    assert {"html_comment", "alt_text", "css_display_none"} <= lost


def test_the_limit_caps_how_many_variants_are_tried(stood_up):
    assert len(run(stood_up, Options(limit=3)).attempts) == 3
