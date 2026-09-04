"""The vocabulary a shelf accumulates: counted from what each section was read as, kept
in the store, and shown to the next section.

The model is a real HTTP server answering scripted JSON, and the run is the real command.
Every book, concept and verb here is invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_ingest import EMPTY, LATTICE, a_model, a_unit, run
from test_sources_pdf import a_textbook

from ml_stack import ingest
from ml_stack.ingest.vocabulary import DOC, MOST, Vocabulary

pytest.importorskip("pymupdf")


def a_reading(*verbs, kind="structure"):
    return dict(EMPTY, concepts=[{"name": "glimmer node", "kind": kind, "aliases": [],
                                  "definition": ""},
                                 {"name": "vault", "kind": "structure", "aliases": [],
                                  "definition": ""}],
                relations=[{"from": "glimmer node", "rel": verb, "to": "vault"}
                           for verb in verbs])


def test_counting_a_reading_names_what_it_coined_and_leaves_the_core_marked():
    words = Vocabulary()
    assert words.note(a_reading("part_of", "sits_inside")) == ["sits_inside"]
    assert words.note(a_reading("sits_inside", kind="lattice_part")) == ["lattice_part"]

    assert words.verbs["part_of"]["core"] is True and words.verbs["part_of"]["uses"] == 1
    assert words.verbs["sits_inside"] == {"core": False, "uses": 2, "first": "", "gloss": ""}
    assert words.kinds["lattice_part"]["core"] is False
    assert words.coined() == (["sits_inside"], ["lattice_part"])


def test_the_prompt_takes_the_most_used_coined_verbs_and_no_more():
    words = Vocabulary()
    for index in range(MOST + 5):
        words.note(a_reading(*([f"verb_{index}"] * (index + 1))))
    verbs, _ = words.seen()
    assert len(verbs) == MOST
    assert verbs[0] == f"verb_{MOST + 4}", "most used first"
    assert "verb_0" not in verbs, "and the rarest fall off the end"
    assert len(words.seen(most=3)[0]) == 3


def test_the_vocabulary_goes_into_the_store_and_comes_back(tmp_path):
    pytest.importorskip("ladybug")
    from ml_stack.graph.store import GraphStore

    unit = a_unit()
    graph = ingest.fold_source([{"unit": unit.id, "extracted": LATTICE}], {unit.id: unit})
    store = tmp_path / "shelf.ladybug"
    ingest.write(store, graph, source="lattice", title="Lattice Studies")

    words = Vocabulary()
    words.note(a_reading("sits_inside"), "lattice:1:1.1#0")
    words.write(store)

    with GraphStore(store, read_only=True) as held:
        assert held.get_doc(DOC)["verbs"]["sits_inside"]["first"] == "lattice:1:1.1#0"
    back = Vocabulary.read(store)
    assert back.coined() == (["sits_inside"], [])
    assert back.verbs["sits_inside"]["uses"] == 1
    assert back.verbs["part_of"]["core"] is True, "the core lists are always in it"


def test_a_store_with_no_vocabulary_document_is_counted_from_its_reads(tmp_path):
    """A shelf read before the vocabulary was kept, and a run that stopped before its first
    fold, both still know what has been coined."""
    store = tmp_path / "shelf.ladybug"
    reads = {"lattice:1:1.1#0": {"unit": "lattice:1:1.1#0",
                                 "extracted": a_reading("sits_inside", "part_of")}}
    Path(f"{store}.lattice.reads.json").write_text(json.dumps(reads))

    words = Vocabulary.read(store)
    assert words.coined() == (["sits_inside"], [])
    assert words.verbs["part_of"]["uses"] == 1


def test_the_next_section_is_told_what_the_last_one_coined(tmp_path, server):
    pytest.importorskip("ladybug")
    seen: list[str] = []

    def script(prompt):
        seen.append(prompt)
        return a_reading("sits_inside" if len(seen) == 1 else "part_of")

    instance, asked = a_model(server, script)
    book = a_textbook(tmp_path / "lattice.pdf")
    store = tmp_path / "shelf.ladybug"
    assert run([book, "--out", str(store), "--no-tidy",
                "--base-url", instance.base_url]) == 0

    assert len(asked) == 2
    first, second = (str(a["messages"][0]["content"]) for a in asked)
    assert "sits_inside" not in first, "nothing had been coined yet"
    assert "already been read with: sits_inside" in second
    assert "not everything you may use" in second

    from ml_stack.graph.store import GraphStore

    with GraphStore(store, read_only=True) as held:
        assert set(held.get_doc(DOC)["verbs"]) >= {"sits_inside", *ingest.VERBS}


def test_core_only_tells_the_next_section_nothing_that_was_coined(tmp_path, server):
    pytest.importorskip("ladybug")
    instance, asked = a_model(server, lambda prompt: a_reading("part_of"))
    book = a_textbook(tmp_path / "lattice.pdf")
    assert run([book, "--out", str(tmp_path / "shelf.ladybug"), "--core-only",
                "--base-url", instance.base_url]) == 0
    for one in asked:
        assert "already been read with" not in str(one["messages"][0]["content"])


def test_a_grown_vocabulary_does_not_make_a_read_run_again(tmp_path, server):
    """The vocabulary is in the prompt and never in the schema, so the cache key and the
    run record's schema_sha do not move as it grows -- a read answered under an earlier
    vocabulary stays answered."""
    pytest.importorskip("ladybug")

    instance, asked = a_model(server, lambda prompt: a_reading("sits_inside"))
    book = a_textbook(tmp_path / "lattice.pdf")
    store = tmp_path / "shelf.ladybug"
    argv = [book, "--out", str(store), "--base-url", instance.base_url,
            "--cache", str(tmp_path / "cache")]

    assert run(argv) == 0
    assert len(asked) == 2
    assert Vocabulary.read(store).coined() == (["sits_inside"], [])

    assert run(argv) == 0
    assert len(asked) == 2, "the cache answered; the wider vocabulary changed no key"
