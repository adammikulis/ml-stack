"""Documents into a graph: the schema, the fold, the gold scorer, resume and status.

The model is a real HTTP server answering scripted JSON -- the same discipline as the rest
of the client tests, and the reason the extraction path is exercised rather than described.
Every passage, concept and source here is invented; the gold fixture copies the *shape* of a
gold set (passages with triples and aliases) and none of its words.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
from conftest import json_reply
from test_sources_pdf import a_textbook

from ml_stack import ingest, jobs
from ml_stack.contracts import grammar_for

pytest.importorskip("pymupdf")


# What a model is scripted to say about the invented lattice textbook.
LATTICE = {
    "concepts": [
        {"name": "glimmer node", "kind": "structure", "aliases": ["node", "glimmer nodes"],
         "definition": "The smallest part of a lattice that can hold a charge."},
        {"name": "vault", "kind": "structure", "aliases": [], "definition": ""},
        {"name": "vault current", "kind": "process", "aliases": ["current"],
         "definition": "A flow of charge between quickened nodes."},
    ],
    "relations": [
        {"from": "glimmer node", "rel": "part_of", "to": "vault"},
        {"from": "vault current", "rel": "consumes", "to": "charge"},
    ],
    "figures": [
        {"label": "Figure 1.1", "caption": "A glimmer node in cross-section.",
         "shows": "One node with the vault wall around it.",
         "concepts": ["glimmer node", "vault"]},
    ],
    "key_terms": [{"term": "quickened", "definition": "Charged, and passing that charge on."}],
}

EMPTY = {"concepts": [], "relations": [], "figures": [], "key_terms": []}


def a_model(server, script):
    """A server answering every extraction with ``script(prompt)``, counting what it was asked."""
    asked: list[dict] = []

    def handler(method, path, body):
        sent = json.loads(body)
        asked.append(sent)
        prompt = "\n".join(str(m.get("content")) for m in sent.get("messages") or ())
        said = script(prompt)
        # a string is sent verbatim, so a test can script a reply that is not JSON at all
        content = said if isinstance(said, str) else json.dumps(said)
        return json_reply({"choices": [{"message": {"role": "assistant",
                                                    "content": content}}]})

    return server(handler), asked


def a_unit(**over):
    from ml_stack.sources.pdf import Unit

    fields = {"source": "lattice", "book_title": "Lattice Studies", "chapter": "1",
              "chapter_title": "The Glimmer Cascade", "section": "1.1",
              "section_title": "Glimmer Nodes", "first_page": 2, "last_page": 3,
              "text": "Glimmer nodes sit inside vaults."}
    return Unit(**{**fields, **over})


# -- the contract ---------------------------------------------------------------------------


def test_the_document_schema_compiles_to_a_grammar():
    """A schema a grammar cannot be built from is one no constrained decode can hold to."""
    text = grammar_for(ingest.schema())
    assert "root ::=" in text and "concepts" in text


def test_the_schema_takes_a_verb_and_a_kind_the_model_coined_and_refuses_another_shape():
    """A book states relations the core eighteen have no word for. The shape is the fence,
    not the list."""
    validate = pytest.importorskip("jsonschema").validate
    ValidationError = pytest.importorskip("jsonschema").ValidationError
    schema = ingest.schema()

    validate(LATTICE, schema)
    validate(EMPTY, schema)

    coined = json.loads(json.dumps(LATTICE))
    coined["relations"][0]["rel"] = "sits_inside"
    coined["concepts"][0]["kind"] = "lattice_part"
    validate(coined, schema)

    for wrong in ("Sits_Inside", "sits inside", "sits-inside", "_sits", "sits__inside",
                  "s", "sits_inside" * 8):
        bad = json.loads(json.dumps(LATTICE))
        bad["relations"][0]["rel"] = wrong
        with pytest.raises(ValidationError):
            validate(bad, schema)
        bad = json.loads(json.dumps(LATTICE))
        bad["concepts"][0]["kind"] = wrong
        with pytest.raises(ValidationError):
            validate(bad, schema)

    extra = json.loads(json.dumps(LATTICE))
    extra["concepts"][0]["colour"] = "blue"
    with pytest.raises(ValidationError):
        validate(extra, schema)


def test_core_only_fences_the_verbs_and_the_kinds_back_to_the_core_lists():
    validate = pytest.importorskip("jsonschema").validate
    ValidationError = pytest.importorskip("jsonschema").ValidationError
    schema = ingest.schema(core_only=True)

    validate(LATTICE, schema)
    assert ingest.closed(schema) and not ingest.closed(ingest.schema())
    assert ingest.fenced(schema) == set(ingest.VERBS)
    assert ingest.fenced(ingest.schema()) == set(), "an open shape fences nothing"

    coined = json.loads(json.dumps(LATTICE))
    coined["relations"][0]["rel"] = "sits_inside"
    with pytest.raises(ValidationError):
        validate(coined, schema)

    other = json.loads(json.dumps(LATTICE))
    other["concepts"][0]["kind"] = "lattice_part"
    with pytest.raises(ValidationError):
        validate(other, schema)

    assert "root" in grammar_for(schema), "and the grammar builder takes the fenced one"


def test_the_instructions_follow_the_shape_the_section_is_read_under():
    """A prompt telling the model to use the schema's verbs and no other, sent with a shape
    that takes any verb, would fence by suggestion alone."""
    unit = a_unit()
    open_turns, _ = ingest.prompt_for(unit)
    closed_turns, _ = ingest.prompt_for(unit, core_only=True)
    assert "and no other" in closed_turns[0]["content"]
    assert "and no other" not in open_turns[0]["content"]
    assert "name the relation the text states" in open_turns[0]["content"]
    for kind, gloss in ingest.CORE_KINDS.items():
        assert f"{kind} -- {gloss}" in open_turns[0]["content"]


def test_the_schema_mirrors_the_message_one_so_the_same_fold_takes_both():
    """`entities.fold_edges` keys on from/rel/to. A document relation named its fields
    differently would need a second reader, and the two would drift."""
    from ml_stack.contracts import load

    relation = ingest.schema()["properties"]["relations"]["items"]["properties"]
    assert set(relation) == set(load("extraction.schema.json")
                                ["properties"]["relations"]["items"]["properties"])


# -- one extraction as a graph ------------------------------------------------------------------


def test_an_extraction_becomes_nodes_and_edges_that_say_where_they_came_from():
    nodes, edges = ingest.build(LATTICE, a_unit(), book_title="Lattice Studies")

    node = nodes["concept:glimmer-node"]
    assert node["kind"] == "structure" and node["label"] == "glimmer node"
    assert node["attrs"]["definition"].startswith("The smallest part")
    assert node["attrs"]["aliases"] == ["node", "glimmer nodes"]
    assert node["attrs"]["defined_in"] == "lattice:1:1.1", "a pointer, resolved by located()"
    assert node["provenance"] == ["lattice:1:1.1"]
    assert not {"chapter", "section", "page", "source"} & set(node["attrs"]), "pointers only"

    assert ("concept:glimmer-node", "part_of", "concept:vault") in edges
    edge = edges[("concept:glimmer-node", "part_of", "concept:vault")]
    assert edge["provenance"] == ["lattice:1:1.1"] and "where" not in edge


def test_a_figure_becomes_a_node_joined_to_what_it_illustrates():
    nodes, edges = ingest.build(LATTICE, a_unit())
    figure = next(n for n in nodes.values() if n["kind"] == "figure")
    assert figure["attrs"]["caption"] == "A glimmer node in cross-section."
    assert (figure["id"], "illustrates", "concept:glimmer-node") in edges
    assert (figure["id"], "illustrates", "concept:vault") in edges


def test_a_key_term_is_a_concept_and_is_marked_as_one():
    nodes, _ = ingest.build(LATTICE, a_unit())
    assert nodes["concept:quickened"]["attrs"]["key_term"] is True
    assert nodes["concept:glimmer-node"]["attrs"]["key_term"] is False


def test_a_relation_naming_the_same_concept_twice_is_dropped():
    said = dict(EMPTY, relations=[{"from": "vault", "rel": "part_of", "to": "vault"}])
    _, edges = ingest.build(said, a_unit())
    assert edges == {}


# -- the fold across two sections ----------------------------------------------------------------


def test_two_sections_naming_the_same_concept_make_one_node_with_both_behind_it():
    first, second = a_unit(), a_unit(section="1.2", section_title="Vaults", first_page=4,
                                     last_page=5)
    reads = [{"unit": first.id, "extracted": LATTICE},
             {"unit": second.id, "extracted": dict(EMPTY, concepts=[
                 {"name": "vault", "kind": "structure", "aliases": [],
                  "definition": "A shell that holds many nodes."}])}]
    graph = ingest.fold_source(reads, {first.id: first, second.id: second})

    vault = next(n for n in graph["nodes"] if n["id"] == "concept:vault")
    assert vault["mentions"] == 4, "named in the relations and the figure of one, and again"
    assert vault["provenance"] == ["lattice:1:1.1", "lattice:1:1.2"]
    assert vault["attrs"]["definition"] == "A shell that holds many nodes."


def test_one_relationship_spelled_twice_is_one_edge():
    """The whole reason `entities.fold` is here: a model coins the relation as it goes."""
    first, second = a_unit(), a_unit(section="1.2", first_page=4, last_page=5)
    reads = [
        {"unit": first.id, "extracted": dict(EMPTY, relations=[
            {"from": "glimmer node", "rel": "part_of", "to": "vault"}])},
        {"unit": second.id, "extracted": dict(EMPTY, relations=[
            {"from": "glimmer node", "rel": "partof", "to": "vault"}])},
    ]
    graph = ingest.fold_source(reads, {first.id: first, second.id: second})

    joined = [e for e in graph["edges"] if e["source"] == "concept:glimmer-node"]
    assert len(joined) == 1 and joined[0]["rel"] == "part_of" and joined[0]["weight"] == 2
    assert joined[0]["provenance"] == ["lattice:1:1.1", "lattice:1:1.2"]
    assert graph["folds"]["relations"], "a fold is on record, not silent"


def test_a_plural_and_its_singular_fold_into_the_name_the_source_uses_more():
    first, second = a_unit(), a_unit(section="1.2", first_page=4, last_page=5)
    reads = [
        {"unit": first.id, "extracted": dict(EMPTY, relations=[
            {"from": "glimmer node", "rel": "part_of", "to": "vault"},
            {"from": "glimmer node", "rel": "requires", "to": "charge"}])},
        {"unit": second.id, "extracted": dict(EMPTY, relations=[
            {"from": "glimmer nodes", "rel": "part_of", "to": "vault"}])},
    ]
    graph = ingest.fold_source(reads, {first.id: first, second.id: second})

    ids = {n["id"] for n in graph["nodes"]}
    assert "concept:glimmer-node" in ids and "concept:glimmer-nodes" not in ids
    kept = next(n for n in graph["nodes"] if n["id"] == "concept:glimmer-node")
    assert "glimmer nodes" in kept["attrs"]["aliases"]


# -- writing a store -------------------------------------------------------------------------------


def test_a_source_is_written_as_a_node_everything_it_holds_hangs_off(tmp_path):
    pytest.importorskip("ladybug")
    from ml_stack.graph.store import GraphStore

    unit = a_unit()
    graph = ingest.fold_source([{"unit": unit.id, "extracted": LATTICE}], {unit.id: unit})
    counts = ingest.write(tmp_path / "sources.ladybug", graph, source="lattice",
                          title="Lattice Studies",
                          docs={"ingest:unit:lattice:1:1.1": {"extracted": LATTICE}})

    assert counts["nodes"] == len(graph["nodes"]) + 1
    with GraphStore(tmp_path / "sources.ladybug", read_only=True) as store:
        labels = {n["id"]: n["label"] for n in store.nodes()}
        assert labels["source:lattice"] == "Lattice Studies"
        assert labels["concept:glimmer-node"] == "glimmer node"
        read_from = [e for e in store.edges("read_from")]
        assert {e["target"] for e in read_from} == {"source:lattice"}
        assert store.get_doc("ingest:unit:lattice:1:1.1")["extracted"] == LATTICE


def test_a_verb_and_a_kind_from_outside_the_core_lists_survive_the_fold_and_the_store(tmp_path):
    """A book's own vocabulary is a finding: which relations and kinds came out of the core
    lists and which the reader named itself reaches the store."""
    pytest.importorskip("ladybug")
    from ml_stack.graph.store import GraphStore

    said = json.loads(json.dumps(LATTICE))
    said["relations"].append({"from": "glimmer node", "rel": "sits_inside", "to": "vault"})
    said["concepts"][1]["kind"] = "lattice_part"
    unit = a_unit()
    graph = ingest.fold_source([{"unit": unit.id, "extracted": said}], {unit.id: unit})

    assert {e["rel"] for e in graph["edges"] if e.get("extension")} == {"sits_inside"}
    assert all("extension" not in e for e in graph["edges"]
               if e["rel"] in ("part_of", "consumes", "illustrates"))
    kinds = {n["label"]: (n["attrs"] or {}).get("extension_kind") for n in graph["nodes"]
             if n["id"].startswith("concept:")}
    assert kinds["vault"] is True and kinds["glimmer node"] is None

    store = tmp_path / "shelf.ladybug"
    ingest.write(store, graph, source="lattice", title="Lattice Studies")
    with GraphStore(store, read_only=True) as held:
        back = {e["rel"]: e.get("extension") for e in held.edges()}
        assert back["sits_inside"] is True
        assert back["part_of"] is None and back["read_from"] is None, \
            "a core verb and the graph's own links are not the book's vocabulary"
        kinds = {n["label"]: (n["attrs"] or {}).get("extension_kind") for n in held.nodes()}
        assert kinds["vault"] is True and kinds["glimmer node"] is None
        assert kinds["Lattice Studies"] is None, "nor is the book itself"


def test_a_kind_a_later_read_takes_from_the_core_list_stops_being_an_extension(tmp_path):
    """The mark is derived from the kind the store ends up holding, never merged into it."""
    pytest.importorskip("ladybug")
    from ml_stack.graph.store import GraphStore

    unit = a_unit()
    coined = json.loads(json.dumps(LATTICE))
    coined["concepts"][1]["kind"] = "lattice_part"
    store = tmp_path / "shelf.ladybug"
    for said in (coined, LATTICE):
        graph = ingest.fold_source([{"unit": unit.id, "extracted": said}], {unit.id: unit})
        ingest.write(store, graph, source="lattice", title="Lattice Studies")
    with GraphStore(store, read_only=True) as held:
        vault = next(n for n in held.nodes() if n["label"] == "vault")
        assert vault["kind"] == "structure"
        assert "extension_kind" not in (vault["attrs"] or {})


# -- the gold set ------------------------------------------------------------------------------------


# The shape of a gold set, with invented passages. Aliases are the point: an extractor that
# writes the singular where the gold writes the plural is right, and a scorer without them
# would report a failure that is not one.
GOLD = {
    "name": "lattice",
    "passages": [
        {"passage_id": "lattice-nodes",
         "source": "Lattice Studies -- The Glimmer Cascade",
         "text": "Glimmer nodes are structures found inside a vault. Each node holds a "
                 "charge, and a quickened node passes that charge to the vault wall.",
         "triples": [
             {"subject": "glimmer node", "predicate": "part_of", "object": "vault",
              "subject_aliases": ["glimmer nodes", "node"],
              "predicate_aliases": ["inside", "located_in"]},
             {"subject": "glimmer node", "predicate": "produces", "object": "charge",
              "subject_aliases": ["glimmer nodes"], "object_aliases": ["a charge"]},
         ]},
        {"passage_id": "lattice-currents",
         "source": "Lattice Studies -- Vault Currents",
         "text": "A vault current runs between quickened nodes and consumes charge.",
         "triples": [
             {"subject": "vault current", "predicate": "consumes", "object": "charge",
              "subject_aliases": ["current"]},
         ]},
    ],
}


def a_gold_file(tmp_path):
    where = tmp_path / "gold.json"
    where.write_text(json.dumps(GOLD))
    return where


def test_a_gold_set_is_read_back_and_an_empty_one_is_refused(tmp_path):
    assert len(ingest.read_gold(a_gold_file(tmp_path))) == 2
    (tmp_path / "none.json").write_text(json.dumps({"passages": []}))
    with pytest.raises(ValueError):
        ingest.read_gold(tmp_path / "none.json")


def test_a_perfect_reading_scores_one_and_names_no_misses(server, tmp_path):
    def script(prompt):
        if "vault current" in prompt:
            return dict(EMPTY, relations=[
                {"from": "vault current", "rel": "consumes", "to": "charge"}])
        return dict(EMPTY, relations=[
            {"from": "glimmer nodes", "rel": "located_in", "to": "vault"},
            {"from": "glimmer node", "rel": "produces", "to": "a charge"}])

    instance, asked = a_model(server, script)
    from ml_stack.client import Client

    scored = ingest.gold_score(Client(instance.base_url), ingest.read_gold(a_gold_file(tmp_path)),
                               ingest.schema())

    assert len(asked) == 2, "every passage goes through the same extraction the run uses"
    assert scored.recall == 1.0 and scored.precision == 1.0 and scored.f1 == 1.0
    assert scored.misses == [] and scored.spurious == []


def test_a_miss_and_an_invention_are_both_counted_and_both_listed(server, tmp_path):
    def script(prompt):
        if "vault current" in prompt:
            return dict(EMPTY, relations=[
                {"from": "vault current", "rel": "regulates", "to": "the lattice"}])
        return dict(EMPTY, relations=[
            {"from": "glimmer node", "rel": "part_of", "to": "vault"}])

    instance, _ = a_model(server, script)
    from ml_stack.client import Client

    scored = ingest.gold_score(Client(instance.base_url), ingest.read_gold(a_gold_file(tmp_path)),
                               ingest.schema())

    assert (scored.wanted, scored.found, scored.matched) == (3, 2, 1)
    assert scored.recall == round(1 / 3, 4) and scored.precision == 0.5
    assert [m["triple"] for m in scored.misses] == [
        "glimmer node produces charge", "vault current consumes charge"]
    assert [m["triple"] for m in scored.spurious] == ["vault current regulates the lattice"]
    assert any("missed" in line for line in ingest.gold_lines(scored))


def test_a_gold_predicate_the_schema_has_no_word_for_is_named_rather_than_hidden(server,
                                                                                  tmp_path):
    """A gold set written before the vocabulary was closed drags recall down for a reason
    that is nothing to do with the model, and nobody can see it from the number alone."""
    older = json.loads(json.dumps(GOLD))
    older["passages"][1]["triples"][0]["predicate"] = "drains"
    older["passages"][1]["triples"][0]["predicate_aliases"] = ["drain"]
    where = tmp_path / "older.json"
    where.write_text(json.dumps(older))

    instance, _ = a_model(server, lambda prompt: EMPTY)
    from ml_stack.client import Client

    scored = ingest.gold_score(Client(instance.base_url), ingest.read_gold(where),
                               ingest.schema(core_only=True))
    assert [m["predicate"] for m in scored.unsayable] == ["drains"]
    assert any("no word for" in line for line in ingest.gold_lines(scored))

    plain = ingest.gold_score(Client(instance.base_url), ingest.read_gold(a_gold_file(tmp_path)),
                              ingest.schema())
    assert plain.unsayable == []


def test_matching_forgives_a_plural_and_an_alias_but_not_a_different_concept():
    assert ingest._same("glimmer nodes", "glimmer node", [])
    assert ingest._same("node", "glimmer node", ["node"])
    assert ingest._same("the vault wall", "vault wall", [])
    assert not ingest._same("vault", "glimmer node", ["node"])


# -- the whole command ----------------------------------------------------------------------------


def a_reading(tmp_path, server, script=lambda prompt: LATTICE):
    instance, asked = a_model(server, script)
    source = a_textbook(tmp_path / "lattice.pdf")
    return source, instance, asked


def run(argv):
    return ingest.main(argv)


def test_a_source_is_read_section_by_section_into_a_store(tmp_path, server, capsys):
    pytest.importorskip("ladybug")
    source, instance, asked = a_reading(tmp_path, server)
    store = tmp_path / "sources.ladybug"

    assert run([source, "--out", str(store), "--base-url", instance.base_url]) == 0
    said = capsys.readouterr().out
    assert len(asked) == 2, "one call per section"
    assert "2 section(s) of 1 source(s)" in said
    done = ingest.Progress(ingest.Progress.beside(store)).state["sources"]["lattice"]["done"]
    assert set(done) == {"lattice:1:1.1", "lattice:2:2.1"}

    from ml_stack.graph.store import GraphStore
    with GraphStore(store, read_only=True) as held:
        assert {n["id"] for n in held.nodes()} >= {"source:lattice", "concept:glimmer-node"}


def test_resume_skips_what_is_already_done_and_asks_the_model_nothing_more(tmp_path, server):
    pytest.importorskip("ladybug")
    source, instance, asked = a_reading(tmp_path, server)
    store = tmp_path / "sources.ladybug"

    run([source, "--out", str(store), "--base-url", instance.base_url])
    assert len(asked) == 2

    run([source, "--out", str(store), "--base-url", instance.base_url, "--resume"])
    assert len(asked) == 2, "a section already read is not read again"

    run([source, "--out", str(store), "--base-url", instance.base_url])
    assert len(asked) == 4, "without --resume the whole source is read again"


def test_resume_still_folds_what_an_earlier_run_extracted(tmp_path, server):
    """A resumed run that folded only the sections it read itself would write a graph
    missing everything the run before it found."""
    pytest.importorskip("ladybug")
    from ml_stack.graph.store import GraphStore

    source, instance, _ = a_reading(tmp_path, server)
    store = tmp_path / "sources.ladybug"
    run([source, "--out", str(store), "--base-url", instance.base_url])
    with GraphStore(store, read_only=True) as held:
        first = {n["id"] for n in held.nodes()}

    run([source, "--out", str(store), "--base-url", instance.base_url, "--resume"])
    with GraphStore(store, read_only=True) as held:
        # every run leaves its own hidden run node; the source's nodes are the same
        assert ({n["id"] for n in held.nodes() if not n["id"].startswith("run:")}
                == {i for i in first if not i.startswith("run:")})


def test_sample_reads_only_the_first_sections(tmp_path, server):
    pytest.importorskip("ladybug")
    source, instance, asked = a_reading(tmp_path, server)
    run([source, "--out", str(tmp_path / "sources.ladybug"), "--base-url", instance.base_url,
         "--sample", "1"])
    assert len(asked) == 1


def test_a_chapter_reads_only_that_chapter(tmp_path, server):
    pytest.importorskip("ladybug")
    source, instance, asked = a_reading(tmp_path, server)
    run([source, "--out", str(tmp_path / "sources.ladybug"), "--base-url", instance.base_url,
         "--chapter", "2"])
    assert len(asked) == 1


def test_status_reports_the_sources_the_sections_and_the_rate(tmp_path, server, capsys):
    pytest.importorskip("ladybug")
    source, instance, _ = a_reading(tmp_path, server)
    store = tmp_path / "sources.ladybug"
    run([source, "--out", str(store), "--base-url", instance.base_url])
    capsys.readouterr()

    assert run(["status", "--out", str(store)]) == 0
    said = capsys.readouterr().out
    assert "2 of 2 sections in 1 source(s)" in said
    assert "lattice" in said and "s/section" in said


def test_status_on_a_store_nothing_was_ingested_into_says_so(tmp_path, capsys):
    assert run(["status", "--out", str(tmp_path / "empty.ladybug")]) == 1
    assert "nothing ingested" in capsys.readouterr().out


def test_a_failed_section_is_recorded_and_the_next_one_is_still_read(tmp_path, server, capsys):
    pytest.importorskip("ladybug")

    def script(prompt):
        return LATTICE if "1.1" in prompt else "not json at all"

    source, instance, asked = a_reading(tmp_path, server, script)
    store = tmp_path / "sources.ladybug"
    assert run([source, "--out", str(store), "--base-url", instance.base_url]) == 0
    assert len(asked) == 2, "the section after the failure was still read"
    assert "1 failed" in capsys.readouterr().out

    progress = ingest.Progress(ingest.Progress.beside(store))
    assert progress.totals()["failed"] == 1


# -- the gold gate, through the command -------------------------------------------------------


def test_gold_through_the_command_prints_the_rates(tmp_path, server, capsys):
    instance, _ = a_model(server, lambda prompt: dict(EMPTY, relations=[
        {"from": "glimmer node", "rel": "part_of", "to": "vault"}]))
    assert run(["--base-url", instance.base_url, "--gold", str(a_gold_file(tmp_path))]) == 0
    assert "recall 33%" in capsys.readouterr().out


def test_core_only_reads_a_section_under_the_core_lists_alone(tmp_path, server, capsys):
    """The two ways of reading, through the command: what the server is sent says which."""
    pytest.importorskip("ladybug")

    def sent_rel(argv):
        source, instance, asked = a_reading(tmp_path, server)
        assert run([source, "--out", str(tmp_path / f"{len(argv)}.ladybug"),
                    "--base-url", instance.base_url, *argv]) == 0
        shape = asked[0]["response_format"]["json_schema"]["schema"]
        return (shape["properties"]["relations"]["items"]["properties"]["rel"],
                shape["properties"]["concepts"]["items"]["properties"]["kind"],
                str(asked[0]["messages"][0]["content"]))

    rel, kind, told = sent_rel(["--core-only"])
    assert rel["enum"] == list(ingest.VERBS) and "pattern" not in rel
    assert kind["enum"] == list(ingest.CORE_KINDS)
    assert "and no other" in told

    rel, kind, told = sent_rel([])
    assert "enum" not in rel and rel["pattern"] and "enum" not in kind
    assert "name the relation the text states" in told


def test_fail_under_is_a_gate(tmp_path, server, capsys):
    instance, _ = a_model(server, lambda prompt: EMPTY)
    argv = ["--base-url", instance.base_url, "--gold", str(a_gold_file(tmp_path))]
    assert run([*argv, "--fail-under", "0.8"]) == 1
    assert run([*argv, "--fail-under", "0.0"]) == 0


# -- the parser ---------------------------------------------------------------------------------


def test_naming_no_document_and_no_gold_is_an_error(tmp_path, capsys):
    assert run(["--out", str(tmp_path / "store")]) == 2
    assert "name at least one document" in capsys.readouterr().err


def test_reading_a_document_without_a_store_to_write_it_into_is_an_error(tmp_path, capsys):
    assert run([str(tmp_path / "source.pdf")]) == 2
    assert "needs --out STORE" in capsys.readouterr().err


def test_status_without_a_store_is_an_error(capsys):
    assert run(["status"]) == 2
    assert "status needs --out STORE" in capsys.readouterr().err


def test_a_document_that_is_not_there_is_said_and_the_run_still_ends(tmp_path, server, capsys):
    instance, _ = a_model(server, lambda prompt: LATTICE)
    assert run([str(tmp_path / "missing.pdf"), "--out", str(tmp_path / "sources.ladybug"),
                "--base-url", instance.base_url]) == 2
    assert "no such document" in capsys.readouterr().err


def test_the_parser_refuses_an_abbreviated_flag_rather_than_guessing():
    with pytest.raises(SystemExit):
        ingest.parser().parse_args(["source.pdf", "--out", "s", "--resu"])


def test_the_images_go_to_the_model_as_pictures_only_when_asked_for(tmp_path):
    from ml_stack.sources import pdf

    document = pdf.read(a_textbook(tmp_path / "lattice.pdf"), images=True)
    unit = pdf.units(document)[0]

    plain, shown = ingest.prompt_for(unit)
    assert shown == 0 and all(isinstance(t["content"], str) for t in plain)

    seen, shown = ingest.prompt_for(unit, images=True)
    assert shown == 1
    parts = seen[-1]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/")


def test_a_section_with_no_rendered_figure_sends_no_picture_and_claims_none(tmp_path):
    from ml_stack.sources import pdf

    unit = pdf.units(pdf.read(a_textbook(tmp_path / "lattice.pdf")))[0]
    turns, shown = ingest.prompt_for(unit, images=True)
    assert shown == 0 and len(turns) == 2


def test_what_a_section_cost_is_kept_call_by_call(tmp_path, server):
    from ml_stack.client import Client
    from ml_stack.sources import pdf

    instance, _ = a_model(server, lambda prompt: LATTICE)
    unit = pdf.units(pdf.read(a_textbook(tmp_path / "lattice.pdf")))[0]
    row = ingest.extract_unit(Client(instance.base_url), unit, ingest.schema())

    assert row.concepts == 3 and row.relations == 2 and row.figures == 1
    assert len(row.calls) == 1 and row.calls[0]["tool"] == "extract"
    assert row.calls[0]["seconds"] >= 0.0 and not row.error


def test_the_core_verbs_and_kinds_are_the_glossed_ones_and_the_instructions_name_each():
    """A core verb without a gloss would be used for whatever the model guessed."""
    assert list(ingest.VERBS) == ingest.core_verbs(ingest.schema())
    assert list(ingest.CORE_KINDS) == ingest.core_kinds(ingest.schema())
    assert ingest.core_verbs(ingest.schema(core_only=True)) == list(ingest.VERBS)
    for verb, gloss in ingest.VERBS.items():
        assert f"{verb} -- {gloss}" in ingest.INSTRUCTIONS
    for kind, gloss in ingest.CORE_KINDS.items():
        assert f"{kind} -- {gloss}" in ingest.INSTRUCTIONS


def test_a_gold_triple_written_the_other_way_round_is_still_found():
    """`charter created_by orlan vesk` says `orlan vesk authored charter`; the first gold run
    counted that a miss and the predicate unsayable."""
    said = {"from": "Charter of Velthorne", "rel": "created_by", "to": "Orlan Vesk"}
    gold = {"subject": "orlan vesk", "predicate": "authored",
            "object": "charter of velthorne",
            "predicate_aliases": ["author_of", "wrote"]}
    assert ingest._matches(said, gold)
    assert not ingest._matches({**said, "rel": "adopted_by"}, gold), "only its own inverse"
    assert not ingest._matches({"from": "Orlan Vesk", "rel": "created_by",
                                "to": "Charter of Velthorne"}, gold), \
        "created_by the right way round is the wrong fact"


def test_a_coined_verb_that_states_a_core_one_backwards_is_still_found():
    """An open vocabulary lets the model write `veltrose produced_by spindrel` where the
    gold says `spindrel produces veltrose`. Same fact, and the gate counted it a miss."""
    gold = {"subject": "spindrel", "predicate": "produces", "object": "veltrose"}
    assert ingest._matches({"from": "veltrose", "rel": "produced_by", "to": "spindrel"}, gold)
    assert not ingest._matches({"from": "spindrel", "rel": "produced_by", "to": "veltrose"},
                               gold), "the same verb the right way round is the wrong fact"
    assert ingest._matches({"from": "thornwick", "rel": "contains", "to": "amberine"},
                           {"subject": "amberine", "predicate": "part_of",
                            "object": "thornwick"})
    assert not ingest._matches({"from": "veltrose", "rel": "dissolves_in", "to": "spindrel"},
                               gold), "a verb no core one stands behind matches nothing"


def test_every_inverse_names_a_core_verb():
    assert set(ingest.INVERSES) <= set(ingest.core_verbs(ingest.schema()))


def test_a_unit_that_failed_is_not_done_so_resume_reads_it_again(tmp_path):
    """Chapter 2 of a biology source lost one unit to a timeout, and --resume would have
    skipped it forever: written down is not the same as finished."""
    progress = ingest.Progress(tmp_path / "sources.progress.json")
    progress.source("velthorne", title="Velthorne", path="v.pdf", sections=2)
    fields = {"source": "velthorne", "chapter": "1", "section": "1.1", "title": "Vault Currents"}
    progress.note("velthorne", ingest.Read(unit="velthorne:1:1.1#0", seconds=1.0, concepts=3,
                                           relations=2, **fields))
    progress.note("velthorne", ingest.Read(unit="velthorne:1:1.1#1", seconds=300.0,
                                           error="ServerUnreachable: timed out", **fields))
    assert progress.done("velthorne", "velthorne:1:1.1#0")
    assert not progress.done("velthorne", "velthorne:1:1.1#1")
    assert progress.totals()["failed"] == 1, "and status still says it failed"


def test_plurals_fold_into_their_singular_and_nothing_else_does():
    got = ingest.plurals(["acid", "Acids", "hydrogen ion", "hydrogen ions", "species",
                          "base", "bases", "bus", "vertebrae", "Currents"])
    assert got == {"acids": "acid", "hydrogen ions": "hydrogen ion", "bases": "base"}


def test_the_source_fold_joins_a_plural_to_its_singular():
    """acid and acids, each with its own edges, are one concept in the folded source."""
    unit = a_unit()
    reads = [{"unit": unit.id, "extracted": {
        "concepts": [{"name": "acid", "kind": "substance", "definition": "", "aliases": []},
                     {"name": "acids", "kind": "substance", "definition": "", "aliases": []},
                     {"name": "base", "kind": "substance", "definition": "", "aliases": []}],
        "relations": [{"from": "acid", "rel": "contrasts_with", "to": "base"},
                      {"from": "acids", "rel": "produces", "to": "base"}],
        "figures": [], "key_terms": []}}]
    graph = ingest.fold_source(reads, {unit.id: unit}, book_title="Velthorne")
    labels = {n["label"] for n in graph["nodes"] if n["kind"] != "figure"}
    assert "acids" not in labels and "acid" in labels
    sources = {e["source"] for e in graph["edges"]}
    assert "concept:acids" not in sources


def test_extraction_serves_one_slot_with_the_whole_context(monkeypatch, tmp_path):
    """Adam: "we shouldn't be handling parallel requests while extracting ... we should never
    be splitting the GPU like that" -- and measured: two workers averaged 140 s a unit
    against 86 alone. One seat, and --context is all its own."""
    from ml_stack.serve import Run, Shape

    seen = {}

    class Found:
        def run(self, port, seats, resolve=True, n_predict=16384, timeout=300.0):
            seen["seats"] = seats
            return Run(shape=Shape(model="x.gguf", port=port, seats=seats,
                                   seat_context=16384))

        def said(self):
            return "measured"

    monkeypatch.setattr("ml_stack.serve.profile.profile_for", lambda m: Found())
    monkeypatch.setattr("ml_stack.serve.profile.said", lambda m: "measured")
    monkeypatch.setattr(ingest, "_find_model", lambda m: "x.gguf")

    def fake_serve(model, manager=None, **lease):
        seen["lease"] = lease
        raise SystemExit(0)

    monkeypatch.setattr("ml_stack.serve.manager.serve", fake_serve)
    (tmp_path / "g.json").write_text(
        '{"passages": [{"passage_id": "p", "text": "Vault currents flow.", "triples": []}]}')
    import contextlib
    with contextlib.suppress(SystemExit):
        ingest.main(["--gold", str(tmp_path / "g.json"), "--model", "x", "--context", "32768"])
    assert seen["seats"] == 1 and seen["lease"]["parallel"] == 1
    assert seen["lease"]["context"] == 32768
    with pytest.raises(SystemExit):
        ingest.parser().parse_args(["source.pdf", "--out", "s", "--workers", "2"])


def test_the_ingest_leases_one_run_and_the_record_reads_the_serving_off_it(monkeypatch,
                                                                          tmp_path):
    """Shape, ceiling, timeout and sampling are one `Run` laid over the model's measured
    profile -- not a lease built in one place and a client built beside it. The record says
    what was actually asked for, so the two can never disagree."""
    import contextlib
    from dataclasses import replace

    from ml_stack.serve.profile import Profile

    measured = Profile(model="kestrel-8B-UD-Q4_K_XL.gguf", seat_context=16384, parallel=4,
                       cache_type="q8_0", sampling={"temperature": 0.0})
    monkeypatch.setattr("ml_stack.serve.profile.profile_for",
                        lambda m: replace(measured, served=str(m)))
    monkeypatch.setattr(ingest, "_find_model", lambda m: "kestrel-8B-UD-Q4_K_XL.gguf")
    seen = {}

    class Up:
        base_url = "http://127.0.0.1:8099"

    @contextlib.contextmanager
    def fake_serve(model, manager=None, **lease):
        seen["model"], seen["lease"] = model, lease
        yield Up()

    monkeypatch.setattr("ml_stack.serve.manager.serve", fake_serve)
    args = ingest.parser().parse_args(
        ["--out", str(tmp_path / "sources"), "--model", "kestrel", "--context", "40000",
         "--per-section", "120", "--n-predict", "999", "--top-k", "20"])

    with ingest._serving(args, say=lambda line: None) as client:
        assert client.base_url == "http://127.0.0.1:8099"
        assert client.n_predict == 999 and client.timeout == 120.0
        assert client.slot == 0, "one seat, and it is sat in"
        assert client.asked_temperature == 0.0, "the profile measured it"
        assert client.asked_top_k == 20, "and the command line laid its own over"

    assert seen["model"] == "kestrel-8B-UD-Q4_K_XL.gguf"
    lease = seen["lease"]
    assert lease["parallel"] == 1, "extraction never splits the GPU, whatever measured"
    assert lease["context"] == 65536, \
        "one seat holding the whole cache the record measured across four"
    assert lease["cache_type_k"] == lease["cache_type_v"] == "q8_0", "from the profile"
    assert (lease["timeout"], lease["cache_reuse"], lease["warmup"]) == (900.0, 256, False)

    said = ingest._serving_said(args)
    assert "context 65536, parallel 1" in said and "kestrel-8B" in said


def test_stop_ends_the_recorded_run_and_says_so_when_there_is_none(tmp_path, capsys):
    import subprocess
    import sys

    assert ingest.stop(home=tmp_path) == 1
    assert "no detached ingest" in capsys.readouterr().out
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    jobs.record("ingest", pid=child.pid, home=tmp_path / "jobs")
    try:
        assert ingest.stop(home=tmp_path) == 0
        child.wait(timeout=10)          # stop waits for it, and reaps it if it is a child
        assert not (tmp_path / "jobs" / "ingest.json").exists()
        assert "--resume" in capsys.readouterr().out
    finally:
        if child.poll() is None:
            child.kill()


def test_a_run_recorded_before_the_record_moved_into_jobs_is_still_stopped(tmp_path, capsys):
    """An ``ingesting.json`` written by the code that kept its own record: `stop` adopts it
    as this machine's ``ingest`` job and ends it."""
    import json as _json
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (tmp_path / "ingesting.json").write_text(_json.dumps({"pid": child.pid, "argv": []}))
    try:
        assert ingest.stop(home=tmp_path) == 0
        child.wait(timeout=10)
        assert "--resume" in capsys.readouterr().out
        assert not (tmp_path / "ingesting.json").exists(), "the old record is taken over"
        assert not (tmp_path / "jobs" / "ingest.json").exists()
    finally:
        if child.poll() is None:
            child.kill()


def test_a_failed_unit_keeps_the_whole_reply_for_reading_later(monkeypatch):
    """The server cut the reply; the row keeps every character it did write."""
    from ml_stack.client import Client
    from ml_stack.client.chat import Reply

    client = Client("http://127.0.0.1:1")
    half = '{"concepts": [{"name": "Vault Currents", "kind": "concept"}' * 40

    def cut(*a, **k):
        return Reply(content=half, finish_reason="length")

    monkeypatch.setattr(client, "chat", cut)
    row = ingest.extract_unit(client, a_unit(), ingest.schema())
    assert row.error.startswith("ServerError: the reply was cut off")
    assert row.raw == half


def test_n_max_lengthens_the_profiles_draft_for_the_run(monkeypatch, tmp_path):
    from ml_stack.serve import Run, Shape

    seen = {}

    class Found:
        def run(self, port, seats, resolve=True, n_predict=16384, timeout=300.0):
            return Run(shape=Shape(model="x.gguf", port=port, seats=seats,
                                   seat_context=16384, draft="mtp.gguf", draft_n_max=4))

        def said(self):
            return "measured"

    monkeypatch.setattr("ml_stack.serve.profile.profile_for", lambda m: Found())
    monkeypatch.setattr("ml_stack.serve.profile.said", lambda m: "measured")
    monkeypatch.setattr(ingest, "_find_model", lambda m: "x.gguf")

    def fake_serve(model, manager=None, **lease):
        seen["lease"] = lease
        raise SystemExit(0)

    monkeypatch.setattr("ml_stack.serve.manager.serve", fake_serve)
    (tmp_path / "g.json").write_text(
        '{"passages": [{"passage_id": "p", "text": "Vault currents flow.", "triples": []}]}')
    import contextlib
    with contextlib.suppress(SystemExit):
        ingest.main(["--gold", str(tmp_path / "g.json"), "--model", "x", "--n-max", "12"])
    assert seen["lease"]["spec_draft_max"] == 12 and seen["lease"]["draft"] == "mtp.gguf"


def test_a_unit_that_failed_twice_is_left_alone_and_status_says_so(tmp_path, capsys):
    progress = ingest.Progress(ingest.Progress.beside(tmp_path / "sources"))
    progress.source("velthorne", title="Velthorne", path="v.pdf", sections=1)
    fields = {"source": "velthorne", "chapter": "1", "section": "1.1", "title": "Vault Currents"}
    bad = ingest.Read(unit="velthorne:1:1.1#0", seconds=700.0,
                      error="ServerError: the reply was cut off (finish_reason=length)", **fields)
    progress.note("velthorne", bad)
    assert not progress.done("velthorne", bad.unit), "one failure: read it again"
    progress.note("velthorne", bad)
    assert progress.done("velthorne", bad.unit), "two: leave it"
    assert progress.totals()["given_up"] == 1
    ingest.status(tmp_path / "sources")
    assert "given up" in capsys.readouterr().out


def test_the_schema_caps_every_list_so_a_greedy_decode_cannot_circle():
    """One unit wrote 378 relations, 282 of them distinct, until n_predict cut it."""
    from ml_stack.contracts.jsonschema import grammar_for

    shape = ingest.schema()
    for name in ("concepts", "relations", "figures", "key_terms"):
        assert shape["properties"][name].get("maxItems"), name
    assert "root" in grammar_for(shape), "and the grammar builder accepts the caps"


def test_retry_frees_the_units_given_up_on(tmp_path, capsys):
    progress = ingest.Progress(ingest.Progress.beside(tmp_path / "sources"))
    progress.source("velthorne", title="Velthorne", path="v.pdf", sections=1)
    fields = {"source": "velthorne", "chapter": "1", "section": "1.1", "title": "Vault Currents"}
    bad = ingest.Read(unit="velthorne:1:1.1#0", seconds=700.0, error="cut off", **fields)
    progress.note("velthorne", bad)
    progress.note("velthorne", bad)
    assert progress.done("velthorne", bad.unit)
    assert ingest.retry(tmp_path / "sources") == 0
    assert "1 unit(s)" in capsys.readouterr().out
    again = ingest.Progress(ingest.Progress.beside(tmp_path / "sources"))
    assert not again.done("velthorne", bad.unit)
    assert ingest.main(["retry"]) == 2, "retry needs --out"


# -- a source folded while it is still being read -------------------------------------------


def _until(ready, what, seconds=30.0):
    """Wait for a child to say it is up, so a signal is not sent before it can take one."""
    deadline = time.time() + seconds
    while not ready():
        assert time.time() < deadline, f"timed out waiting until {what}"
        time.sleep(0.02)


def a_read(unit, *, source="velthorne-open-texts", chapter="1", section="1.1",
           title="Vault Currents", pages=(2, 3), extracted=None, error=""):
    """One row of a reads file, in the shape the run writes it."""
    return {"unit": unit, "source": source, "chapter": chapter, "section": section,
            "title": title, "pages": list(pages), "seconds": 86.0, "concepts": 0,
            "relations": 0, "figures": 0, "images": 0, "timed_out": False, "error": error,
            "raw": "", "calls": [], "extracted": dict(extracted or {})}


def said(*names, relations=(), figures=(), key_terms=()):
    """An extraction naming ``names`` as concepts and ``relations`` between them."""
    return {"concepts": [{"name": n, "kind": "structure", "definition": f"What a {n} is.",
                          "aliases": []} for n in names],
            "relations": [{"from": a, "rel": r, "to": b} for a, r, b in relations],
            "figures": list(figures), "key_terms": list(key_terms)}


def a_part_read_source(tmp_path, *, slug="velthorne-open-texts", rows=None,
                     title="Velthorne Open Texts", sections=4, store=None):
    """A reads file and a progress file beside a store, as a half-read source leaves them."""
    where = store or (tmp_path / "sources.ladybug")
    rows = rows if rows is not None else [
        a_read(f"{slug}:1:1.1", source=slug,
               extracted=said("vault", "charge", relations=[("vault", "produces", "charge")])),
        a_read(f"{slug}:1:1.2", source=slug, section="1.2", pages=(4, 5),
               extracted=said("vault", "seam wall",
                              relations=[("seam wall", "part_of", "vault")]))]
    ingest._write_json(ingest.reads_path(where, slug), {r["unit"]: r for r in rows})
    progress = ingest.Progress(ingest.Progress.beside(where))
    held = progress.source(slug, title=title, path=f"{slug}.pdf", sections=sections)
    for row in rows:
        held["done"][row["unit"]] = {"seconds": row["seconds"], "error": row["error"],
                                     "attempts": 1}
    progress.save()
    return where


def in_store(where):
    with ingest.Sources(where).store() as held:
        return {n["id"] for n in held.nodes()}


def test_fold_writes_a_part_read_source_into_the_store_and_says_it_is_partial(tmp_path, capsys):
    pytest.importorskip("ladybug")
    store = a_part_read_source(tmp_path)

    assert ingest.main(["fold", "--out", str(store)]) == 0
    line = capsys.readouterr().out.strip()
    assert line.count("\n") == 2 and line.endswith("reads back whole"), \
        "a line for what it did, one for what the hygiene pass kept, and the check"
    assert "Velthorne Open Texts" in line and "2 of 4 units read" in line
    assert "partial" in line
    assert "kept: merged 0 node(s)" in line, "nothing was settled, so nothing is replayed"
    assert in_store(store) >= {"source:velthorne-open-texts", "concept:vault",
                               "concept:seam-wall"}


def test_fold_twice_leaves_the_store_exactly_as_it_was(tmp_path):
    pytest.importorskip("ladybug")
    store = a_part_read_source(tmp_path)
    ingest.fold(store, say=lambda _: None)
    with ingest.Sources(store).store() as held:
        first = (held.nodes(), held.edges())

    ingest.fold(store, say=lambda _: None)
    with ingest.Sources(store).store() as held:
        assert (held.nodes(), held.edges()) == first


def test_a_second_fold_adds_to_the_source_and_only_rebuild_takes_anything_out(tmp_path):
    """Adam: "if the source already exists, it should append new nodes/connect new edges.
    additive." A section re-read into something else adds what it now says; what the
    first fold wrote stays until a person asks for a rebuild."""
    pytest.importorskip("ladybug")
    slug = "velthorne-open-texts"
    store = a_part_read_source(tmp_path)
    ingest.fold(store, say=lambda _: None)
    assert "concept:seam-wall" in in_store(store)

    ingest._write_json(ingest.reads_path(store, slug), {
        f"{slug}:1:1.1": a_read(f"{slug}:1:1.1", source=slug,
                                extracted=said("vault", "flux ring",
                                               relations=[("vault", "produces", "flux ring")]))})
    ingest.fold(store, say=lambda _: None)

    held = in_store(store)
    assert "concept:flux-ring" in held, "added"
    assert {"concept:seam-wall", "concept:charge"} <= held, "and nothing taken out"
    with ingest.Sources(store).store() as store_handle:
        rels = {(e["source"], e["rel"], e["target"]) for e in store_handle.edges()}
    assert ("concept:vault", "produces", "concept:charge") in rels
    assert ("concept:vault", "produces", "concept:flux-ring") in rels

    ingest.fold(store, rebuild=True, say=lambda _: None)
    held = in_store(store)
    assert "concept:flux-ring" in held and "concept:seam-wall" not in held \
        and "concept:charge" not in held, "a rebuild is the full fold from the reads, alone"


def test_a_concept_two_sources_name_survives_one_of_them_being_rebuilt(tmp_path):
    pytest.importorskip("ladybug")
    store = tmp_path / "sources.ladybug"
    a_part_read_source(tmp_path, store=store)
    a_part_read_source(tmp_path, store=store, slug="lattice-studies", title="Lattice Studies",
                     sections=2, rows=[a_read("lattice-studies:1:1.1", source="lattice-studies",
                                              extracted=said("vault", "glimmer node",
                                                             relations=[("glimmer node",
                                                                         "part_of", "vault")]))])
    ingest.fold(store, say=lambda _: None)
    assert {"source:velthorne-open-texts", "source:lattice-studies"} <= in_store(store)

    ingest._write_json(ingest.reads_path(store, "velthorne-open-texts"), {
        "velthorne-open-texts:1:1.1": a_read("velthorne-open-texts:1:1.1",
                                             extracted=said("flux ring"))})
    ingest.fold(store, source="velthorne-open-texts", rebuild=True, say=lambda _: None)

    held = in_store(store)
    assert "concept:vault" in held, "the other source still names it"
    assert "concept:glimmer-node" in held
    with ingest.Sources(store).store() as handle:
        sources = {(e["source"], e["target"]) for e in handle.edges("read_from")}
    assert ("concept:vault", "source:lattice-studies") in sources
    assert ("concept:vault", "source:velthorne-open-texts") not in sources


def test_a_unit_that_failed_contributes_nothing_to_the_fold(tmp_path):
    """What a failed extraction wrote is kept for reading, not for believing."""
    slug = "velthorne-open-texts"
    rows = [a_read(f"{slug}:1:1.1", extracted=said("vault")),
            a_read(f"{slug}:1:1.2", section="1.2", error="ServerError: the reply was cut off",
                   extracted=said("half a concept"))]
    store = a_part_read_source(tmp_path, rows=rows)
    graph = ingest.Sources(store).graph(slug)
    assert {n["id"] for n in graph["nodes"]} == {"concept:vault"}


def test_fold_on_a_store_nothing_was_read_into_says_so(tmp_path, capsys):
    assert ingest.main(["fold", "--out", str(tmp_path / "empty.ladybug")]) == 1
    assert "nothing to fold" in capsys.readouterr().out
    assert ingest.main(["fold"]) == 2


# -- the sources, as an application reads it ----------------------------------------------------


def test_the_sources_view_names_every_one_and_how_much_of_it_is_read(tmp_path):
    store = a_part_read_source(tmp_path)
    a_part_read_source(tmp_path, store=store, slug="lattice-studies", title="Lattice Studies",
                     sections=1, rows=[a_read("lattice-studies:1:1.1", source="lattice-studies",
                                              extracted=said("glimmer node"))])
    sources = {b.slug: b for b in ingest.Sources(store).sources()}

    assert set(sources) == {"velthorne-open-texts", "lattice-studies"}
    partial = sources["velthorne-open-texts"]
    assert (partial.units, partial.read, partial.wanted) == (2, 2, 4)
    assert partial.partial and partial.title == "Velthorne Open Texts"
    assert not sources["lattice-studies"].partial
    assert partial.per_unit == 86.0 and partial.left == 2 * 86.0


def test_a_part_read_source_folds_with_no_store_and_no_pdf(tmp_path):
    store = a_part_read_source(tmp_path)
    graph = ingest.Sources(store).graph("velthorne-open-texts")

    assert {n["id"] for n in graph["nodes"]} == {"concept:vault", "concept:charge",
                                                 "concept:seam-wall"}
    vault = next(n for n in graph["nodes"] if n["id"] == "concept:vault")
    assert vault["provenance"] == ["velthorne-open-texts:1:1.1",
                                   "velthorne-open-texts:1:1.2"]
    assert "page" not in vault["attrs"] and "section" not in vault["attrs"], "pointers only"
    assert not (store.exists() or ingest.Progress.beside(store).with_suffix(".none").exists())


def test_a_unit_read_in_parts_keeps_the_id_its_provenance_names(tmp_path):
    """A section split into parts has a `#2` in its unit id, and rebuilding that id from
    the row's fields would name a unit nothing was read from."""
    row = a_read("velthorne-open-texts:1:1.1#2", pages=(6, 7))
    unit = ingest.unit_of(row)
    assert unit.id == "velthorne-open-texts:1:1.1#2"
    assert unit.where == {"source": "velthorne-open-texts", "chapter": "1", "section": "1.1",
                          "page": 6, "pages": [6, 7],
                          "unit": "velthorne-open-texts:1:1.1#2"}


def test_the_sources_view_opens_the_store_read_only_while_a_writer_has_it_open(tmp_path):
    """An application reads a sources the run is still writing into."""
    pytest.importorskip("ladybug")
    from ml_stack.graph.store import GraphStore

    store = a_part_read_source(tmp_path)
    ingest.fold(store, say=lambda _: None)
    with GraphStore(store) as writer:
        writer.upsert_node({"id": "concept:ward", "kind": "concept", "label": "ward"})
        with ingest.Sources(store).store() as reader:
            assert reader.read_only
            assert "concept:vault" in {n["id"] for n in reader.nodes()}


# -- folding as the run goes ------------------------------------------------------------------


def test_the_store_holds_the_first_chapter_before_the_second_is_read(tmp_path, server,
                                                                    monkeypatch):
    """The whole point: a sources that takes days is answerable while it is being read."""
    pytest.importorskip("ladybug")
    monkeypatch.setattr(ingest, "FOLD_EVERY", 1)
    store = tmp_path / "sources.ladybug"
    seen: list[set] = []

    def script(prompt):
        if "2.1" in prompt:
            seen.append(in_store(store))
        return LATTICE

    source, instance, _ = a_reading(tmp_path, server, script)
    assert run([source, "--out", str(store), "--base-url", instance.base_url]) == 0
    assert seen and "concept:glimmer-node" in seen[0], \
        "chapter 1 was in the store before chapter 2 was asked for"
    assert "figure:lattice:2:2.1:1" not in seen[0], "and chapter 2 was not"


def test_the_progress_file_records_how_far_each_source_is_folded(tmp_path, server):
    pytest.importorskip("ladybug")
    source, instance, _ = a_reading(tmp_path, server)
    store = tmp_path / "sources.ladybug"
    run([source, "--out", str(store), "--base-url", instance.base_url])

    held = json.loads(ingest.Progress.beside(store).read_text())["sources"]["lattice"]
    assert held["folded_at"] == 2 and held["folded_nodes"] > 0 and held["folded_edges"] > 0
    assert held["sections"] == 2, "and no existing key was renamed"
    assert set(held["done"]) == {"lattice:1:1.1", "lattice:2:2.1"}


def test_a_chapter_ends_a_fold_and_a_long_chapter_folds_inside_itself():
    assert not ingest._time_to_fold(ingest.FOLD_EVERY - 1, True), "too soon to pay for one"
    assert ingest._time_to_fold(ingest.FOLD_EVERY, True)
    assert not ingest._time_to_fold(ingest.FOLD_EVERY, False), "no chapter has ended"
    assert ingest._time_to_fold(2 * ingest.FOLD_EVERY, False)


def test_folding_a_few_hundred_units_of_one_source_costs_a_second_or_two(tmp_path):
    """What the interval is chosen from. The fold itself is `entities.fold_names`, which
    grows with the square of the vocabulary; the write grows with the units."""
    pytest.importorskip("ladybug")
    slug = "velthorne-open-texts"
    words = [f"vault {n}" for n in range(120)] + [f"seam {n}" for n in range(120)]
    rows = []
    for n in range(300):
        chapter = str(n // 15 + 1)
        picks = [words[(n * 5 + k) % len(words)] for k in range(5)]
        rows.append(a_read(f"{slug}:{chapter}:{chapter}.{n % 15 + 1}", chapter=chapter,
                           section=f"{chapter}.{n % 15 + 1}", pages=(n * 2, n * 2 + 1),
                           extracted=said(*picks, relations=[
                               (picks[i], "part_of", picks[i + 1]) for i in range(4)])))
    store = a_part_read_source(tmp_path, rows=rows, sections=515)

    began = time.time()
    got = ingest.fold_into(store, slug)
    spent = time.time() - began

    assert got["units"] == 300 and got["nodes"] > 200 and got["partial"]
    assert spent < 20.0, f"a fold of 300 units took {spent:.1f}s"


# -- stopping -----------------------------------------------------------------------------------


def test_a_run_told_to_stop_folds_what_it_read_and_ends_cleanly(tmp_path):
    """SIGTERM mid-source: the unit in flight is lost, the units before it are in the store."""
    pytest.importorskip("ladybug")
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler

    from conftest import REPO, threaded_server

    store = tmp_path / "sources.ladybug"
    source = a_textbook(tmp_path / "lattice.pdf")
    second = threading.Event()
    asked: list[bytes] = []

    class Model(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            asked.append(self.rfile.read(int(self.headers.get("content-length") or 0)))
            if len(asked) > 1:
                second.set()
                time.sleep(10)          # long enough for the stop to land inside this unit
            body = json.dumps({"choices": [{"message": {
                "role": "assistant", "content": json.dumps(LATTICE)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with threaded_server(Model) as url:
        child = subprocess.Popen(
            [sys.executable, "-m", "ml_stack.ingest", source, "--out", str(store),
             "--base-url", url],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONUNBUFFERED": "1"})
        try:
            assert second.wait(60), "the run reached its second unit"
            child.terminate()
            out = child.communicate(timeout=60)[0].decode()
        finally:
            if child.poll() is None:            # pragma: no cover - only on a hung run
                child.kill()

    assert child.returncode == 0, out
    assert "stopped:" in out and "--resume" in out
    assert "concept:glimmer-node" in in_store(store), "the first unit is in the store"
    assert ingest.Sources(store).source("lattice").folded_at == 1


def test_stop_waits_for_the_run_and_says_the_fold_landed(tmp_path, capsys):
    import subprocess

    store = tmp_path / "sources.ladybug"
    a_part_read_source(tmp_path, slug="lattice-studies", title="Lattice Studies", store=store,
                     sections=4, rows=[a_read("lattice-studies:1:1.1", source="lattice-studies",
                                              extracted=said("vault"))])
    folding = (
        "import json, pathlib, signal, sys, time\n"
        "p, ready = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])\n"
        "def landed(*a):\n"
        "    held = json.loads(p.read_text())\n"
        "    held['sources']['lattice-studies']['folded_at'] = 1\n"
        "    p.write_text(json.dumps(held))\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, landed)\n"
        "ready.write_text('up')\n"
        "time.sleep(60)\n")
    ready = tmp_path / "ready"
    child = subprocess.Popen([sys.executable, "-c", folding,
                              str(ingest.Progress.beside(store)), str(ready)])
    jobs.record("ingest", pid=child.pid, argv=["source.pdf", "--out", str(store)],
                home=tmp_path / "jobs")
    try:
        _until(ready.is_file, "the run installed its stop handler")
        assert ingest.stop(home=tmp_path) == 0
        line = capsys.readouterr().out.strip()
        assert line.count("\n") == 0, "one line for what it did"
        assert "folded lattice-studies at unit 1" in line and "--resume" in line
    finally:
        if child.poll() is None:                # pragma: no cover - only on a hung child
            child.kill()


def test_stop_says_so_when_the_run_will_not_end(tmp_path, capsys):
    import subprocess

    deaf = ("import pathlib, signal, sys, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "pathlib.Path(sys.argv[1]).write_text('up')\n"
            "time.sleep(60)\n")
    ready = tmp_path / "ready"
    child = subprocess.Popen([sys.executable, "-c", deaf, str(ready)])
    jobs.record("ingest", pid=child.pid, argv=[], home=tmp_path / "jobs")
    try:
        _until(ready.is_file, "the run is ignoring SIGTERM")
        assert ingest.stop(home=tmp_path, wait=1.0) == 1
        assert "had not ended after 1s" in capsys.readouterr().out
    finally:
        child.kill()
        child.wait(timeout=10)


# -- what was read ------------------------------------------------------------------------------


def test_show_prints_the_concepts_the_relations_and_the_folds_of_each_source(tmp_path, capsys):
    slug = "velthorne-open-texts"
    rows = [a_read(f"{slug}:1:1.1", extracted={
        "concepts": [{"name": "vault", "kind": "structure", "aliases": [],
                      "definition": "A shell that holds many nodes."},
                     {"name": "vaults", "kind": "structure", "aliases": [],
                      "definition": ""}],
        "relations": [{"from": "vault", "rel": "produces", "to": "charge"},
                      {"from": "vaults", "rel": "requires", "to": "charge"}],
        "figures": [{"label": "Figure 1.1", "caption": "A vault in cross-section.",
                     "shows": "One vault.", "concepts": ["vault"]}],
        "key_terms": []})]
    store = a_part_read_source(tmp_path, rows=rows)

    assert ingest.main(["show", "--out", str(store)]) == 0
    out = capsys.readouterr().out
    assert "Velthorne Open Texts (velthorne-open-texts): 1 of 4 units read, partial" in out
    assert "1 figure(s)" in out
    assert "vault [structure] -- A shell that holds many nodes." in out
    assert "vault --produces--> charge   p.2" in out and "(velthorne-open-texts:1:1.1)" in out
    assert "read by" in out
    assert "names joined: vaults -> vault" in out


def test_show_takes_one_source_and_a_sample_size(tmp_path, capsys):
    slug = "velthorne-open-texts"
    store = a_part_read_source(tmp_path)
    a_part_read_source(tmp_path, store=store, slug="lattice-studies", title="Lattice Studies",
                     sections=1, rows=[a_read("lattice-studies:1:1.1", source="lattice-studies",
                                              extracted=said("glimmer node"))])

    assert ingest.main(["show", "--out", str(store), "--source", slug, "--sample", "1"]) == 0
    out = capsys.readouterr().out
    assert "Lattice Studies" not in out
    assert "... and 2 more" in out and "... and 1 more" in out


def test_show_on_a_store_nothing_was_read_into_says_so(tmp_path, capsys):
    assert ingest.main(["show", "--out", str(tmp_path / "empty.ladybug")]) == 1
    assert "nothing read into" in capsys.readouterr().out


def test_status_says_what_is_in_the_store_and_how_long_the_rest_will_take(tmp_path, capsys):
    store = a_part_read_source(tmp_path, sections=6)
    assert ingest.status(store) == 0
    out = capsys.readouterr().out
    assert "in store: nothing folded yet" in out
    assert "~6 min left" in out, "four units at the 86 s each this source measured"

    pytest.importorskip("ladybug")
    ingest.fold(store, say=lambda _: None)
    capsys.readouterr()
    ingest.status(store)
    out = capsys.readouterr().out
    assert "folded at unit 2 of 6" in out and "in store: " in out


# -- the files a kill can land in the middle of -----------------------------------------------


def test_a_reads_file_that_cannot_be_written_is_left_as_it_was(tmp_path):
    """A rename is what makes a kill mid-write survivable; a write in place is not."""
    path = tmp_path / "sources.velthorne-open-texts.reads.json"
    ingest._write_json(path, {"velthorne-open-texts:1:1.1": {"unit": "1.1"}})
    before = path.read_text()

    with pytest.raises(TypeError):
        ingest._write_json(path, {"unit": object()})

    assert path.read_text() == before
    assert list(tmp_path.glob("*.part")) == [], "and no half-written file is left behind"


def _gold_with_a_fake_model(tmp_path, monkeypatch):
    """A `--gold --model` run whose server is a fake and whose scoring is a no-op."""
    import contextlib

    seen = {}

    class Up:
        base_url = "http://127.0.0.1:8099"

    @contextlib.contextmanager
    def fake_serve(model, manager=None, **lease):
        seen["lease"] = lease
        yield Up()

    monkeypatch.setattr("ml_stack.serve.manager.serve", fake_serve)
    monkeypatch.setattr("ml_stack.serve.profile.profile_for", lambda m: None)
    monkeypatch.setattr(ingest, "_find_model", lambda m: "x.gguf")
    monkeypatch.setattr("ml_stack.ingest.cli.gold_score",
                        lambda *a, **k: ingest.Scored())
    monkeypatch.setattr("ml_stack.ingest.cli.gold_lines", lambda scored: [])
    (tmp_path / "g.json").write_text(
        '{"passages": [{"passage_id": "p", "text": "Vault currents flow.", "triples": []}]}')
    return ["--gold", str(tmp_path / "g.json"), "--model", "x"], seen


def test_the_ingest_takes_the_benchs_measuring_lock_and_never_roams(tmp_path, monkeypatch,
                                                                    capsys):
    """One job on the GPU: the ingest waits on the same lock file the bench takes, says it
    is waiting for the bench, and leases on its own port rather than beside whatever the
    bench left there."""
    import contextlib

    import ml_stack.graph.bench as bench
    import ml_stack.lock

    taken = {}

    @contextlib.contextmanager
    def recording(what, *, wait=True, timeout=0.0, announce=print):
        taken["path"], taken["wait"], taken["announce"] = Path(what), wait, announce
        yield Path(what)

    monkeypatch.setattr(ml_stack.lock, "only_one", recording)
    argv, seen = _gold_with_a_fake_model(tmp_path, monkeypatch)
    assert ingest.main(argv) == 0
    assert taken["path"] == bench.HOME / "measuring.lock"
    assert taken["wait"] is True
    assert seen["lease"]["roam"] is False
    taken["announce"]("waiting for measuring.lock, held by pid 7")
    assert "waiting for the bench" in capsys.readouterr().out


def test_no_queue_is_refused_at_once_while_the_bench_holds_the_lock(tmp_path, monkeypatch,
                                                                    capsys):
    import ml_stack.graph.bench as bench
    from ml_stack.lock import only_one

    argv, seen = _gold_with_a_fake_model(tmp_path, monkeypatch)
    with only_one(bench.HOME / "measuring.lock", wait=False, announce=lambda *a: None):
        assert ingest.main([*argv, "--no-queue"]) == 3
    err = capsys.readouterr().err
    assert "held by pid" in err and "--no-queue" in err
    assert "lease" not in seen, "nothing was served"
    assert ingest.main(argv) == 0, "and with the lock free the same run goes through"
