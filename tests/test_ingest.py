"""Documents into a graph: the schema, the fold, the gold scorer, resume and status.

The model is a real HTTP server answering scripted JSON -- the same discipline as the rest
of the client tests, and the reason the extraction path is exercised rather than described.
Every passage, concept and book here is invented; the gold fixture copies the *shape* of a
gold set (passages with triples and aliases) and none of its words.
"""

from __future__ import annotations

import json

import pytest
from conftest import json_reply
from test_sources_pdf import a_textbook

from ml_stack import ingest
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

    fields = {"book": "lattice", "book_title": "Lattice Studies", "chapter": "1",
              "chapter_title": "The Glimmer Cascade", "section": "1.1",
              "section_title": "Glimmer Nodes", "first_page": 2, "last_page": 3,
              "text": "Glimmer nodes sit inside vaults."}
    return Unit(**{**fields, **over})


# -- the contract ---------------------------------------------------------------------------


def test_the_document_schema_compiles_to_a_grammar():
    """A schema a grammar cannot be built from is one no constrained decode can hold to."""
    text = grammar_for(ingest.schema())
    assert "root ::=" in text and "concepts" in text


def test_the_schema_accepts_an_extraction_and_refuses_one_outside_the_vocabulary():
    validate = pytest.importorskip("jsonschema").validate
    ValidationError = pytest.importorskip("jsonschema").ValidationError
    schema = ingest.schema()

    validate(LATTICE, schema)
    validate(EMPTY, schema)

    invented = json.loads(json.dumps(LATTICE))
    invented["relations"][0]["rel"] = "sits_inside"
    with pytest.raises(ValidationError):
        validate(invented, schema)

    extra = json.loads(json.dumps(LATTICE))
    extra["concepts"][0]["colour"] = "blue"
    with pytest.raises(ValidationError):
        validate(extra, schema)


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
    assert node["attrs"]["chapter"] == "1" and node["attrs"]["section"] == "1.1"
    assert node["attrs"]["page"] == 2 and node["provenance"] == ["lattice:1:1.1"]

    assert ("concept:glimmer-node", "part_of", "concept:vault") in edges
    assert edges[("concept:glimmer-node", "part_of", "concept:vault")]["where"]["section"] == "1.1"


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
    graph = ingest.fold_book(reads, {first.id: first, second.id: second})

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
    graph = ingest.fold_book(reads, {first.id: first, second.id: second})

    joined = [e for e in graph["edges"] if e["source"] == "concept:glimmer-node"]
    assert len(joined) == 1 and joined[0]["rel"] == "part_of" and joined[0]["weight"] == 2
    assert joined[0]["provenance"] == ["lattice:1:1.1", "lattice:1:1.2"]
    assert graph["folds"]["relations"], "a fold is on record, not silent"


def test_a_plural_and_its_singular_fold_into_the_name_the_book_uses_more():
    first, second = a_unit(), a_unit(section="1.2", first_page=4, last_page=5)
    reads = [
        {"unit": first.id, "extracted": dict(EMPTY, relations=[
            {"from": "glimmer node", "rel": "part_of", "to": "vault"},
            {"from": "glimmer node", "rel": "requires", "to": "charge"}])},
        {"unit": second.id, "extracted": dict(EMPTY, relations=[
            {"from": "glimmer nodes", "rel": "part_of", "to": "vault"}])},
    ]
    graph = ingest.fold_book(reads, {first.id: first, second.id: second})

    ids = {n["id"] for n in graph["nodes"]}
    assert "concept:glimmer-node" in ids and "concept:glimmer-nodes" not in ids
    kept = next(n for n in graph["nodes"] if n["id"] == "concept:glimmer-node")
    assert "glimmer nodes" in kept["attrs"]["aliases"]


# -- writing a store -------------------------------------------------------------------------------


def test_a_book_is_written_as_a_node_everything_it_holds_hangs_off(tmp_path):
    pytest.importorskip("ladybug")
    from ml_stack.graph.store import GraphStore

    unit = a_unit()
    graph = ingest.fold_book([{"unit": unit.id, "extracted": LATTICE}], {unit.id: unit})
    counts = ingest.write(tmp_path / "shelf.ladybug", graph, book="lattice",
                          title="Lattice Studies",
                          docs={"ingest:unit:lattice:1:1.1": {"extracted": LATTICE}})

    assert counts["nodes"] == len(graph["nodes"]) + 1
    with GraphStore(tmp_path / "shelf.ladybug", read_only=True) as store:
        labels = {n["id"]: n["label"] for n in store.nodes()}
        assert labels["book:lattice"] == "Lattice Studies"
        assert labels["concept:glimmer-node"] == "glimmer node"
        read_from = [e for e in store.edges("read_from")]
        assert {e["target"] for e in read_from} == {"book:lattice"}
        assert store.get_doc("ingest:unit:lattice:1:1.1")["extracted"] == LATTICE


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
                               ingest.schema())
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


def a_shelf(tmp_path, server, script=lambda prompt: LATTICE):
    instance, asked = a_model(server, script)
    book = a_textbook(tmp_path / "lattice.pdf")
    return book, instance, asked


def run(argv):
    return ingest.main(argv)


def test_a_book_is_read_section_by_section_into_a_store(tmp_path, server, capsys):
    pytest.importorskip("ladybug")
    book, instance, asked = a_shelf(tmp_path, server)
    store = tmp_path / "shelf.ladybug"

    assert run([book, "--out", str(store), "--base-url", instance.base_url]) == 0
    said = capsys.readouterr().out
    assert len(asked) == 2, "one call per section"
    assert "2 section(s) of 1 book(s)" in said
    assert "1.1" in said and "1.2" not in said

    from ml_stack.graph.store import GraphStore
    with GraphStore(store, read_only=True) as held:
        assert {n["id"] for n in held.nodes()} >= {"book:lattice", "concept:glimmer-node"}


def test_resume_skips_what_is_already_done_and_asks_the_model_nothing_more(tmp_path, server):
    pytest.importorskip("ladybug")
    book, instance, asked = a_shelf(tmp_path, server)
    store = tmp_path / "shelf.ladybug"

    run([book, "--out", str(store), "--base-url", instance.base_url])
    assert len(asked) == 2

    run([book, "--out", str(store), "--base-url", instance.base_url, "--resume"])
    assert len(asked) == 2, "a section already read is not read again"

    run([book, "--out", str(store), "--base-url", instance.base_url])
    assert len(asked) == 4, "without --resume the whole book is read again"


def test_resume_still_folds_what_an_earlier_run_extracted(tmp_path, server):
    """A resumed run that folded only the sections it read itself would write a graph
    missing everything the run before it found."""
    pytest.importorskip("ladybug")
    from ml_stack.graph.store import GraphStore

    book, instance, _ = a_shelf(tmp_path, server)
    store = tmp_path / "shelf.ladybug"
    run([book, "--out", str(store), "--base-url", instance.base_url])
    with GraphStore(store, read_only=True) as held:
        first = {n["id"] for n in held.nodes()}

    run([book, "--out", str(store), "--base-url", instance.base_url, "--resume"])
    with GraphStore(store, read_only=True) as held:
        assert {n["id"] for n in held.nodes()} == first


def test_sample_reads_only_the_first_sections(tmp_path, server):
    pytest.importorskip("ladybug")
    book, instance, asked = a_shelf(tmp_path, server)
    run([book, "--out", str(tmp_path / "shelf.ladybug"), "--base-url", instance.base_url,
         "--sample", "1"])
    assert len(asked) == 1


def test_a_chapter_reads_only_that_chapter(tmp_path, server):
    pytest.importorskip("ladybug")
    book, instance, asked = a_shelf(tmp_path, server)
    run([book, "--out", str(tmp_path / "shelf.ladybug"), "--base-url", instance.base_url,
         "--chapter", "2"])
    assert len(asked) == 1


def test_status_reports_the_books_the_sections_and_the_rate(tmp_path, server, capsys):
    pytest.importorskip("ladybug")
    book, instance, _ = a_shelf(tmp_path, server)
    store = tmp_path / "shelf.ladybug"
    run([book, "--out", str(store), "--base-url", instance.base_url])
    capsys.readouterr()

    assert run(["status", "--out", str(store)]) == 0
    said = capsys.readouterr().out
    assert "2 of 2 sections in 1 book(s)" in said
    assert "lattice" in said and "s/section" in said


def test_status_on_a_store_nothing_was_ingested_into_says_so(tmp_path, capsys):
    assert run(["status", "--out", str(tmp_path / "empty.ladybug")]) == 1
    assert "nothing ingested" in capsys.readouterr().out


def test_a_failed_section_is_recorded_and_the_next_one_is_still_read(tmp_path, server, capsys):
    pytest.importorskip("ladybug")

    def script(prompt):
        return LATTICE if "1.1" in prompt else "not json at all"

    book, instance, asked = a_shelf(tmp_path, server, script)
    store = tmp_path / "shelf.ladybug"
    assert run([book, "--out", str(store), "--base-url", instance.base_url]) == 0
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
    assert run([str(tmp_path / "book.pdf")]) == 2
    assert "needs --out STORE" in capsys.readouterr().err


def test_status_without_a_store_is_an_error(capsys):
    assert run(["status"]) == 2
    assert "status needs --out STORE" in capsys.readouterr().err


def test_a_document_that_is_not_there_is_said_and_the_run_still_ends(tmp_path, server, capsys):
    instance, _ = a_model(server, lambda prompt: LATTICE)
    assert run([str(tmp_path / "missing.pdf"), "--out", str(tmp_path / "shelf.ladybug"),
                "--base-url", instance.base_url]) == 2
    assert "no such document" in capsys.readouterr().err


def test_the_parser_refuses_an_abbreviated_flag_rather_than_guessing():
    with pytest.raises(SystemExit):
        ingest.parser().parse_args(["book.pdf", "--out", "s", "--resu"])


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


def test_the_schema_verbs_are_the_glossed_verbs_and_the_instructions_name_each():
    """A verb the schema allows without a gloss would be used for whatever the model guessed."""
    allowed = ingest.schema()["properties"]["relations"]["items"]["properties"]["rel"]["enum"]
    assert list(ingest.VERBS) == allowed
    for verb, gloss in ingest.VERBS.items():
        assert f"{verb} -- {gloss}" in ingest.INSTRUCTIONS


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


def test_every_inverse_names_a_verb_the_schema_has():
    allowed = ingest.schema()["properties"]["relations"]["items"]["properties"]["rel"]["enum"]
    assert set(ingest.INVERSES) <= set(allowed)


def test_a_unit_that_failed_is_not_done_so_resume_reads_it_again(tmp_path):
    """Chapter 2 of a biology book lost one unit to a timeout, and --resume would have
    skipped it forever: written down is not the same as finished."""
    progress = ingest.Progress(tmp_path / "shelf.progress.json")
    progress.book("velthorne", title="Velthorne", path="v.pdf", sections=2)
    fields = {"book": "velthorne", "chapter": "1", "section": "1.1", "title": "Vault Currents"}
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


def test_the_book_fold_joins_a_plural_to_its_singular():
    """acid and acids, each with its own edges, are one concept in the folded book."""
    unit = a_unit()
    reads = [{"unit": unit.id, "extracted": {
        "concepts": [{"name": "acid", "kind": "substance", "definition": "", "aliases": []},
                     {"name": "acids", "kind": "substance", "definition": "", "aliases": []},
                     {"name": "base", "kind": "substance", "definition": "", "aliases": []}],
        "relations": [{"from": "acid", "rel": "contrasts_with", "to": "base"},
                      {"from": "acids", "rel": "produces", "to": "base"}],
        "figures": [], "key_terms": []}}]
    graph = ingest.fold_book(reads, {unit.id: unit}, book_title="Velthorne")
    labels = {n["label"] for n in graph["nodes"] if n["kind"] != "figure"}
    assert "acids" not in labels and "acid" in labels
    sources = {e["source"] for e in graph["edges"]}
    assert "concept:acids" not in sources


def test_context_is_per_worker_so_the_lease_is_that_many_times_over(monkeypatch, tmp_path):
    """Four figures, 2,500 tokens of text and a long reply overran a 16k seat on the first
    shelf night; each worker's seat gets the whole --context."""
    from ml_stack.serve import Shape

    seen = {}

    class Found:
        def shape(self, port, seats):
            return Shape(model="x.gguf", port=port, seats=seats, seat_context=16384)

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
        ingest.main(["--gold", str(tmp_path / "g.json"), "--model", "x",
                     "--workers", "2", "--context", "32768"])
    assert seen["lease"]["context"] == 65536 and seen["lease"]["parallel"] == 2


def test_stop_ends_the_recorded_run_and_says_so_when_there_is_none(tmp_path, capsys):
    import json as _json
    import subprocess
    import sys

    assert ingest.stop(home=tmp_path) == 1
    assert "no detached ingest" in capsys.readouterr().out
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (tmp_path / "ingesting.json").write_text(_json.dumps({"pid": child.pid}))
    try:
        assert ingest.stop(home=tmp_path) == 0
        assert child.wait(timeout=10) != 0
        assert not (tmp_path / "ingesting.json").exists()
        assert "--resume" in capsys.readouterr().out
    finally:
        if child.poll() is None:
            child.kill()


def test_a_failed_unit_keeps_the_whole_reply_for_reading_later(monkeypatch):
    from ml_stack.client.http import ServerError

    class Client:
        def extract(self, *a, **k):
            raise ServerError("the reply was cut off (finish_reason=length)",
                              body='{"concepts": [{"name": "Vault Currents"' * 40)

    row = ingest.extract_unit(Client(), a_unit(), ingest.schema())
    assert row.error.startswith("ServerError: the reply was cut off")
    assert row.raw.count("Vault Currents") == 40
