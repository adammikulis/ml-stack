"""The attack texts: where each came from, that they are the bytes recorded, and that the
jailbreak templates are the ones that were chosen."""

from __future__ import annotations

import json
import re

import pytest

from poolhouse.redteam import corpus


def test_every_file_in_the_manifest_matches_its_recorded_hash():
    assert corpus.verify() == []


def test_every_file_names_its_source_and_licence():
    for name, entry in json.loads(corpus.MANIFEST.read_text(encoding="utf-8"))["files"].items():
        assert entry["source"] and entry["licence"], name
        assert entry["rows"] == len(corpus.load(name.removesuffix(".json")))


def test_a_file_edited_after_it_was_recorded_is_refused(tmp_path, monkeypatch):
    copy = tmp_path / "redteam"
    copy.mkdir()
    for path in corpus.DATA.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    (copy / "styles.json").write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(corpus, "DATA", copy)
    monkeypatch.setattr(corpus, "MANIFEST", copy / "MANIFEST.json")
    with pytest.raises(ValueError, match=re.escape("styles.json")):
        corpus.load("styles")
    assert any("styles.json" in problem for problem in corpus.verify())


def test_every_style_leaves_a_slot_for_the_instruction_it_dresses():
    styles = corpus.load("styles")
    assert len(styles) >= 4
    assert all("{act}" in row["text"] for row in styles)


def test_system_prompt_extraction_seeds_are_many_and_each_has_text_and_an_id():
    rows = corpus.load("garak_sysprompt") + corpus.load("psfuzz_steal")
    assert len(rows) >= 30
    assert all(row["id"] and row["text"].strip() for row in rows)
    assert len({row["id"] for row in rows}) == len(rows)


def test_fill_replaces_named_slots_and_leaves_other_braces_alone():
    assert corpus.fill("write {name} {not a slot}", name="x.txt") == "write x.txt {not a slot}"


def test_render_puts_the_objective_where_the_template_has_its_slot():
    assert corpus.render("Role play. {{ prompt }} Go.", "say hi") == "Role play. say hi Go."
    with pytest.raises(ValueError, match="slot"):
        corpus.render("no slot here", "say hi")


@pytest.mark.redteam
def test_the_jailbreak_templates_are_the_installed_pyrits_and_each_has_a_prompt_slot():
    templates = corpus.templates()
    assert len(templates) == len(corpus.TEMPLATES)
    assert all(corpus.PLACEHOLDER in row["text"] for row in templates)
