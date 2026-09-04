"""A nodes/edges CSV pair another extractor wrote, brought onto the shelf as one book.

The pair here is written by the test, in the shape the material has: the same columns, the
same JSON in `source_context` and `source_metadata`, the same open predicate vocabulary.
Every book, concept, predicate and file name is invented.
"""

from __future__ import annotations

import csv
import json

import pytest

from ml_stack import ingest

pytest.importorskip("ladybug")


NODE_COLUMNS = ["node_id", "label", "category", "definition", "aliases", "source_text",
                "source_type", "source_uri", "source_locator", "source_context",
                "source_metadata", "source_citations", "confidence", "model",
                "extraction_method", "extraction_version", "run_id", "created_by",
                "created_at", "metadata"]
EDGE_COLUMNS = ["edge_id", "subject_id", "predicate", "object_id", "evidence_texts",
                "source_type", "source_uri", "source_locator", "source_context",
                "source_metadata", "source_citations", "confidence", "model",
                "extraction_method", "extraction_version", "run_id", "created_by",
                "created_at", "metadata"]

URI = "Velthorne_Open_Texts.pdf"


def _context(chapter=1, chapter_title="The Glimmer Cascade", section="1.1",
             section_title="Glimmer Nodes"):
    return json.dumps({"chapter_number": chapter, "chapter_title": chapter_title,
                       "section_title": section_title, "section_number": section,
                       "segment_type": "content", "excerpt": "Glimmer nodes sit in vaults."})


def _provenance(chapter=1, section="1.1", pages=(2, 3), confidence="high",
                validation="provisional", section_title="Glimmer Nodes"):
    return {"source_text": "Glimmer nodes sit in vaults.", "source_type": "book",
            "source_uri": URI, "source_locator": f"chapter {chapter}; section {section}",
            "source_context": _context(chapter=chapter, section=section,
                                       section_title=section_title),
            "source_metadata": json.dumps({"page_start": pages[0], "page_end": pages[1]}),
            "source_citations": "[]", "confidence": confidence, "model": "quill:8b",
            "extraction_method": "orchard.textbook_ingest", "extraction_version": "0.1.0",
            "run_id": "velthorne_extract_2026-01-14T01:17:49", "created_by": "quill_extraction",
            "created_at": "2026-01-14T01:19:52", "metadata": json.dumps(
                {"validation_status": validation, "provisional": validation == "provisional"})}


def a_node(node_id, label, *, category="Concept", definition="", aliases=(), **over):
    return {"node_id": node_id, "label": label, "category": category,
            "definition": definition, "aliases": json.dumps(list(aliases)),
            **{**_provenance(), **over}}


def an_edge(edge_id, subject, predicate, obj, **over):
    row = {"edge_id": edge_id, "subject_id": subject, "predicate": predicate,
           "object_id": obj, "evidence_texts": json.dumps(["The book says so."]),
           **{**_provenance(), **over}}
    return {k: v for k, v in row.items() if k in EDGE_COLUMNS}


def a_pair(tmp_path, nodes, edges, *, name="extraction"):
    """One nodes/edges CSV pair on disk, in the columns the material has."""
    where = tmp_path / name
    where.mkdir(parents=True, exist_ok=True)
    for path, columns, rows in ((where / "nodes.csv", NODE_COLUMNS, nodes),
                                (where / "edges.csv", EDGE_COLUMNS, edges)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return where


def a_small_book(tmp_path, **over):
    """Three concepts and the relations between them, as the other extractor wrote them."""
    nodes = [a_node("N:1", "glimmer node", category="Structure",
                    definition="The smallest part of a lattice that holds a charge.",
                    aliases=["node"]),
             a_node("N:2", "vault", category="Structure"),
             a_node("N:3", "vault current", category="Process")]
    edges = [an_edge("E:1", "N:1", "part_of", "N:2"),
             an_edge("E:2", "N:2", "includes", "N:3"),
             an_edge("E:3", "N:3", "related_to", "N:1"),
             an_edge("E:4", "N:2", "shelters", "N:1")]
    return a_pair(tmp_path, nodes, edges, **over)


# -- the table between the two vocabularies -------------------------------------------------


def test_the_table_maps_a_predicate_each_way_round():
    """`includes` is `has_part` as it stands; `defines` is `defined_by` the other way."""
    assert ingest.verb_for("part_of") == ("part_of", False)
    assert ingest.verb_for("includes") == ("has_part", False)
    assert ingest.verb_for("defines") == ("defined_by", True)
    assert ingest.verb_for("caused_by") == ("causes", True)
    assert ingest.verb_for("enables") == ("requires", True)
    # the third person 's' is not part of the word
    assert ingest.verb_for("include") == ("has_part", False)
    assert ingest.verb_for("contribute_to") == ingest.verb_for("contributes_to")


def test_every_verb_the_table_normalises_onto_is_one_this_library_sets():
    """The left-hand side of the table is open; the right-hand side is `fold.CORE`."""
    written = {found[0] for found in ingest.RELATIONS.values() if found}
    assert written <= ingest.CORE, written - ingest.CORE
    assert set(ingest.schema()["properties"]["relations"]["items"]
               ["properties"]["rel"]["enum"]) <= ingest.CORE


def test_a_flipped_predicate_is_written_with_its_ends_the_other_way(tmp_path):
    nodes = [a_node("N:1", "vault"), a_node("N:2", "glimmer node")]
    edges = [an_edge("E:1", "N:1", "defines", "N:2")]
    got = ingest.imported(*_files(a_pair(tmp_path, nodes, edges)))
    written = [r for row in got.rows for r in row["extracted"]["relations"]]
    assert written == [{"from": "glimmer node", "rel": "defined_by", "to": "vault"}]


def _files(where):
    return where / "nodes.csv", where / "edges.csv"


# -- what it makes of a pair ----------------------------------------------------------------


def test_a_predicate_outside_the_core_verbs_is_written_as_it_stands(tmp_path):
    got = ingest.imported(*_files(a_small_book(tmp_path)))
    verbs = sorted(r["rel"] for row in got.rows for r in row["extracted"]["relations"])
    assert verbs == ["has_part", "part_of", "related_to", "shelters"]
    # both are counted by name: one the table names and has no counterpart for, one it
    # does not name at all
    assert got.extensions == {"shelters": 1, "related_to": 1}
    assert got.unnamed == {"shelters": 1}
    assert got.core == 2
    said = ingest.import_lines(got)
    assert any("shelters" in line for line in said)
    assert any("related_to" in line for line in said)


def test_core_only_leaves_the_predicates_outside_those_verbs(tmp_path):
    got = ingest.imported(*_files(a_small_book(tmp_path)), core_only=True)
    verbs = sorted(r["rel"] for row in got.rows for r in row["extracted"]["relations"])
    assert verbs == ["has_part", "part_of"]
    assert got.extensions == {} and got.left == {"shelters": 1, "related_to": 1}
    assert any("--core-only" in line for line in ingest.import_lines(got))


def test_the_confidence_filter_leaves_the_rows_under_it(tmp_path):
    nodes = [a_node("N:1", "vault"), a_node("N:2", "glimmer node", confidence="low")]
    edges = [an_edge("E:1", "N:1", "part_of", "N:2", confidence="low"),
             an_edge("E:2", "N:2", "part_of", "N:1", confidence="high")]
    both = ingest.imported(*_files(a_pair(tmp_path, nodes, edges)), confidence="low")
    assert both.concepts == 2 and both.relations == 2
    sure = ingest.imported(*_files(a_pair(tmp_path, nodes, edges)), confidence="high")
    assert sure.concepts == 1 and sure.relations == 0
    assert sure.dropped["a concept under high confidence"] == 1
    assert sure.dropped["a relation under high confidence"] == 1
    # the high-confidence relation went because the nodes file no longer holds an end
    assert sure.dropped["a relation whose ends the nodes file does not hold"] == 1


def test_no_provisional_leaves_the_rows_their_extractor_left_provisional(tmp_path):
    nodes = [a_node("N:1", "vault", metadata=json.dumps({"validation_status": "validated"})),
             a_node("N:2", "glimmer node")]
    edges = [an_edge("E:1", "N:2", "part_of", "N:1",
                     metadata=json.dumps({"validation_status": "validated"}))]
    where = a_pair(tmp_path, nodes, edges)
    assert ingest.imported(*_files(where)).concepts == 2
    strict = ingest.imported(*_files(where), provisional=False)
    assert strict.concepts == 1 and strict.dropped["a provisional concept"] == 1


def test_a_malformed_row_is_refused_by_name(tmp_path):
    nodes = [a_node("N:1", "vault"), a_node("N:2", "")]
    with pytest.raises(ValueError, match=r"nodes\.csv:3: label is empty"):
        ingest.imported(*_files(a_pair(tmp_path, nodes, [])))
    broken = [a_node("N:1", "vault", source_context="{not json")]
    with pytest.raises(ValueError, match=r"nodes\.csv:2: source_context is not JSON"):
        ingest.imported(*_files(a_pair(tmp_path, broken, [], name="broken")))
    where = a_pair(tmp_path, nodes[:1], [], name="short")
    (where / "edges.csv").write_text("edge_id,predicate\nE:1,part_of\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no subject_id, object_id column"):
        ingest.imported(*_files(where))


# -- onto the shelf -------------------------------------------------------------------------


def test_the_imported_book_is_on_the_shelf_beside_a_read_one(tmp_path, capsys):
    store = tmp_path / "shelf.ladybug"
    where = a_small_book(tmp_path)
    assert ingest.bring(store, [str(where)]) == 0
    shelf = ingest.Shelf(store)
    book = shelf.book("velthorne-open-texts")
    assert book is not None and book.title == "Velthorne Open Texts"
    assert book.units == 1 and book.read == 1 and not book.partial
    # every unit is written down as done, so `status` counts the book whole
    assert shelf.progress.done("velthorne-open-texts", "velthorne-open-texts:1:1.1")
    with shelf.store() as held:
        ids = {n["id"] for n in held.nodes()}
        kinds = {n["id"]: n["kind"] for n in held.nodes()}
        edges = {(e["source"], e["rel"], e["target"]) for e in held.edges()}
        node = next(n for n in held.nodes() if n["id"] == "concept:glimmer-node")
    assert "book:velthorne-open-texts" in ids
    assert kinds["concept:glimmer-node"] == "structure"
    assert kinds["concept:vault-current"] == "process"
    assert ("concept:glimmer-node", "read_from", "book:velthorne-open-texts") in edges
    assert ("concept:glimmer-node", "part_of", "concept:vault") in edges
    assert node["attrs"]["definition"].startswith("The smallest part")
    assert node["attrs"]["aliases"] == ["node"]
    assert shelf.shared()["books"][0]["book"] == "velthorne-open-texts"


def test_an_extension_verb_is_marked_on_its_edge_and_a_core_one_is_not(tmp_path):
    store = tmp_path / "shelf.ladybug"
    assert ingest.bring(store, [str(a_small_book(tmp_path))]) == 0
    with ingest.Shelf(store).store() as held:
        marked = {(e["source"], e["rel"], e["target"]): bool(e.get("extension"))
                  for e in held.edges()}
        predicates = held.get_doc("ingest:predicates:velthorne-open-texts")
    assert marked[("concept:glimmer-node", "part_of", "concept:vault")] is False
    assert marked[("concept:vault", "shelters", "concept:glimmer-node")] is True
    assert marked[("concept:vault-current", "related_to", "concept:glimmer-node")] is True
    assert marked[("concept:vault", "read_from", "book:velthorne-open-texts")] is False
    assert predicates["core"]["includes"] == {"verb": "has_part", "flipped": False,
                                              "relations": 1}
    assert predicates["extensions"] == {"related_to": 1, "shelters": 1}
    assert predicates["unnamed"] == {"shelters": 1}


def test_the_provenance_points_at_units_the_way_a_read_book_does(tmp_path):
    store = tmp_path / "shelf.ladybug"
    assert ingest.bring(store, [str(a_small_book(tmp_path))]) == 0
    with ingest.Shelf(store).store() as held:
        node = next(n for n in held.nodes() if n["id"] == "concept:vault")
        where = ingest.located(held, node)
        origin = ingest.origin(held, node)
    assert node["provenance"] == ["velthorne-open-texts:1:1.1"]
    assert where == [{"unit": "velthorne-open-texts:1:1.1", "book": "velthorne-open-texts",
                      "title": "Glimmer Nodes", "chapter": "1", "section": "1.1",
                      "pages": [2, 3]}]
    assert origin[0]["model"] == "quill:8b" and origin[0]["started"] == "2026-01-14T01:19:52"


def test_a_dry_run_says_what_it_would_write_and_writes_nothing(tmp_path, capsys):
    store = tmp_path / "shelf.ladybug"
    assert ingest.bring(store, [str(a_small_book(tmp_path))], dry_run=True) == 0
    said = capsys.readouterr().out
    assert "part_of" in said and "shelters" in said and "related_to" in said
    assert "nothing written" in said
    assert not store.exists()
    assert not list(tmp_path.glob("shelf.ladybug*"))


def test_two_units_of_one_book_become_two_units_on_the_shelf(tmp_path):
    nodes = [a_node("N:1", "vault"),
             a_node("N:2", "spore bloom",
                    **_provenance(chapter=2, section="2.1", pages=(6, 7),
                                  section_title="Spore Blooms"))]
    edges = [an_edge("E:1", "N:2", "part_of", "N:1",
                     **_provenance(chapter=2, section="2.1", pages=(6, 7),
                                   section_title="Spore Blooms"))]
    store = tmp_path / "shelf.ladybug"
    assert ingest.bring(store, [str(a_pair(tmp_path, nodes, edges))]) == 0
    with ingest.Shelf(store).store() as held:
        bloom = next(n for n in held.nodes() if n["id"] == "concept:spore-bloom")
        vault = next(n for n in held.nodes() if n["id"] == "concept:vault")
    assert bloom["provenance"] == ["velthorne-open-texts:2:2.1"]
    # the section that states a relation names both its ends
    assert sorted(vault["provenance"]) == ["velthorne-open-texts:1:1.1",
                                           "velthorne-open-texts:2:2.1"]


def test_the_slug_names_the_book_and_can_be_given(tmp_path):
    store = tmp_path / "shelf.ladybug"
    assert ingest.bring(store, [str(a_small_book(tmp_path))], slug="lattice-studies") == 0
    assert {b.slug for b in ingest.Shelf(store).books()} == {"lattice-studies"}


def test_the_command_takes_a_directory_or_the_two_files(tmp_path, capsys):
    store = tmp_path / "shelf.ladybug"
    nodes, edges = _files(a_small_book(tmp_path))
    assert ingest.main(["import", str(nodes), str(edges), "--out", str(store),
                        "--dry-run"]) == 0
    assert "velthorne-open-texts" in capsys.readouterr().out
    assert ingest.main(["import", str(tmp_path / "extraction"), "--out", str(store)]) == 0
    assert ingest.Shelf(store).book("velthorne-open-texts") is not None
    assert ingest.main(["import", str(nodes), "--out", str(store)]) == 2
    assert "nodes.csv and edges.csv" in capsys.readouterr().out
    assert ingest.main(["import", str(tmp_path / "extraction")]) == 2
