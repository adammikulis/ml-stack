"""The vocabulary switch: the server's default and the string catalogue.

The page resolves the switch in the browser (address, then storage, then the server default);
``test_vocabulary_browser`` drives that. What is decided here is the server's half: the
environment variable, and a catalogue in which every id has both wordings and every id a screen
uses exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ml_stack.fleet import page, vocabulary
from ml_stack.fleet.vocabulary_strings import CATALOGUE

WEB = Path(page.__file__).parent / "web"
#: ids the page builds from a template string, so no literal in a source names them
BUILT = {f"wizard.stage.{i}" for i in range(9)}
GROUPS = ("nav", "action", "devices", "projects", "settings", "wizard", "dev")
USED = re.compile(r"""["'`]((?:""" + "|".join(GROUPS) + r""")\.[a-z0-9_.]+)["'`]""")


def test_the_default_vocabulary_is_professional():
    assert vocabulary.default_vocabulary({}) == "professional"


@pytest.mark.parametrize("value", ["friendly", " Friendly "])
def test_the_environment_sets_the_server_default(value):
    assert vocabulary.default_vocabulary({"ML_STACK_UI_VOCAB": value}) == "friendly"


@pytest.mark.parametrize("value", ["", "chatty", "poolhouse"])
def test_an_environment_value_that_is_not_a_vocabulary_falls_back_to_the_default(value):
    assert vocabulary.default_vocabulary({"ML_STACK_UI_VOCAB": value}) == "professional"


def test_every_id_has_both_wordings():
    for key, row in CATALOGUE.items():
        assert len(row) == 2, key
        for chosen in vocabulary.VOCABULARIES:
            assert vocabulary.wording(key, chosen).strip(), (key, chosen)


def test_friendly_wording_differs_from_the_plain_word_outside_the_developer_section():
    for key, (plain, friendly) in CATALOGUE.items():
        if not key.startswith("dev."):
            assert plain != friendly, key


def test_the_plain_words_the_owner_named_are_kept():
    assert {CATALOGUE[k][0] for k in ("nav.devices", "nav.projects", "nav.settings", "nav.pool")} == {
        "Devices", "Projects", "Settings", "Pool"}


def test_friendly_wording_never_swaps_out_a_command_or_a_flag():
    for key, row in CATALOGUE.items():
        for text in row:
            assert "--" not in text and "ML_STACK" not in text, key


def test_every_id_a_screen_uses_is_in_the_catalogue():
    seen = set(BUILT)
    for source in [*(WEB / "components").glob("*.html"), WEB / "shell.html"]:
        seen |= set(USED.findall(source.read_text(encoding="utf-8")))
    assert {"nav.settings", "wizard.summary", "dev.vocab"} <= seen
    assert sorted(seen - set(CATALOGUE)) == []


def test_a_placeholder_in_a_wording_is_the_same_set_in_both():
    slots = re.compile(r"\{(\w+)\}")
    for key, row in CATALOGUE.items():
        assert len({frozenset(slots.findall(t)) for t in row}) == 1, key


def test_the_embedded_payload_cannot_close_its_script_element():
    body = vocabulary.payload({"ML_STACK_UI_VOCAB": "friendly"})
    assert "<" not in body and ">" not in body
    data = json.loads(body)
    assert data["vocab"] == "friendly"
    assert data["catalogue"]["nav.pool"][0] == "Pool"


def test_the_rendered_page_carries_the_server_default_and_no_placeholder(monkeypatch):
    monkeypatch.setenv("ML_STACK_UI_VOCAB", "friendly")
    html = page.render()
    assert 'data-vocab="friendly"' in html
    assert "<title>Poolhouse</title>" in html and 'href="/ui/static/poolhouse.svg"' in html
    assert "__VOCAB" not in html


def test_no_brand_switch_is_left_in_the_page_or_the_server():
    html = page.render()
    for left in ("data-brand", "brandResolve", "ML_STACK_UI_BRAND", "pooltable"):
        assert left not in html
    assert not (WEB / "brand-pooltable.css").exists()
