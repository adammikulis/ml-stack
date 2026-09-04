"""A store written when a source was called a book, brought up to date.

The old store is built here by hand, in the shape the fold wrote before the rename, so
the migration is measured against a fixture rather than against its own inverse. Every
title, concept and slug is invented.
"""

from __future__ import annotations

from importlib import import_module

import pytest

from ml_stack import ingest

pytest.importorskip("ladybug")

migrate_module = import_module("ml_stack.ingest.migrate")

SLUG = "velthorne-open-texts"
TITLE = "Velthorne Open Texts"
UNITS = (f"{SLUG}:1:1.1", f"{SLUG}:1:1.2")


def an_old_store(tmp_path, *, name="sources.ladybug", slug=SLUG, title=TITLE):
    """A store, a progress file and a reads file as the fold left them before the rename."""
    from ml_stack.graph.store import GraphStore

    where = tmp_path / name
    book_id = f"book:{slug}"
    nodes = [
        {"id": book_id, "kind": "book", "label": title, "mentions": 1,
         "attrs": {"book": slug}},
        {"id": "concept:vault", "kind": "structure", "label": "vault", "mentions": 2,
         "attrs": {"definition": "What a vault is.", "aliases": [], "key_term": False},
         "provenance": list(UNITS)},
        {"id": "concept:seam-wall", "kind": "structure", "label": "seam wall", "mentions": 1,
         "attrs": {"definition": "", "aliases": [], "key_term": False},
         "provenance": [UNITS[1]]},
    ]
    edges = [
        {"source": "concept:seam-wall", "rel": "part_of", "target": "concept:vault",
         "weight": 1, "provenance": [UNITS[1]]},
        {"source": "concept:vault", "rel": "read_from", "target": book_id, "weight": 1},
        {"source": "concept:seam-wall", "rel": "read_from", "target": book_id, "weight": 1},
    ]
    with GraphStore(where) as store:
        store.write({"nodes": nodes, "edges": edges})
        for index, unit in enumerate(UNITS, start=1):
            store.put_doc(f"ingest:unit:{unit}", {
                "unit": unit, "book": slug,
                "where": {"book": slug, "chapter": "1", "section": f"1.{index}",
                          "page": 2 * index, "pages": [2 * index, 2 * index + 1],
                          "unit": unit},
                "title": "Vault Currents", "chapter_title": "", "run": "",
                "extracted": {}, "calls": [], "seconds": 86.0, "error": ""})
        store.put_doc(f"ingest:folds:{slug}", {"concepts": [], "relations": []})
        store.put_doc(f"ingest:shares:{slug}", {"concept:vault": 2, "concept:seam-wall": 1})
    ingest._write_json(ingest.reads_path(where, slug), {
        unit: {"unit": unit, "book": slug, "chapter": "1", "section": f"1.{index}",
               "title": "Vault Currents", "pages": [2 * index, 2 * index + 1],
               "seconds": 86.0, "concepts": 0, "relations": 0, "figures": 0, "images": 0,
               "timed_out": False, "error": "", "raw": "", "calls": [], "extracted": {}}
        for index, unit in enumerate(UNITS, start=1)})
    ingest._write_json(ingest.Progress.beside(where), {
        "started": "2026-01-01T00:00:00",
        "books": {slug: {"title": title, "path": f"{slug}.pdf", "sections": 4,
                         "done": {unit: {"seconds": 86.0, "error": "", "attempts": 1}
                                  for unit in UNITS},
                         "folded_at": 2, "folded_nodes": 3, "folded_edges": 3}}})
    return where


def beside(where):
    """Every file the migration touches, by name, as bytes."""
    return {p.name: p.read_bytes() for p in
            [ingest.Progress.beside(where), *ingest.reads_beside(where)] if p.is_file()}


def quiet(where):
    said = []
    code = ingest.migrate(where, say=said.append)
    return code, "\n".join(said)


# -- the graph ------------------------------------------------------------------------------


def test_the_book_node_becomes_a_source_node_and_the_edges_follow_it(tmp_path):
    from ml_stack.graph.store import GraphStore

    where = an_old_store(tmp_path)
    with GraphStore(where, read_only=True) as store:
        before = store.counts()

    assert quiet(where)[0] == 0

    with GraphStore(where, read_only=True) as store:
        nodes = {n["id"]: n for n in store.nodes()}
        edges = {(e["source"], e["rel"], e["target"]) for e in store.edges()}
        assert store.counts() == before, "one node renamed, none added or lost"
        assert store.check() == []
    assert not [n for n in nodes if n.startswith("book:")]
    assert nodes[f"source:{SLUG}"]["kind"] == "source"
    assert nodes[f"source:{SLUG}"]["attrs"] == {"source": SLUG}
    assert nodes[f"source:{SLUG}"]["label"] == TITLE
    assert ("concept:vault", "read_from", f"source:{SLUG}") in edges
    assert ("concept:seam-wall", "read_from", f"source:{SLUG}") in edges
    assert ("concept:seam-wall", "part_of", "concept:vault") in edges, "left alone"


def test_the_unit_documents_say_which_source_they_were_read_from(tmp_path):
    from ml_stack.graph.store import GraphStore

    where = an_old_store(tmp_path)

    assert quiet(where)[0] == 0

    with GraphStore(where, read_only=True) as store:
        doc = store.get_doc(f"ingest:unit:{UNITS[0]}")
        shares = store.get_doc(f"ingest:shares:{SLUG}")
    assert doc["source"] == SLUG and "book" not in doc
    assert doc["where"]["source"] == SLUG and "book" not in doc["where"]
    assert doc["where"]["pages"] == [2, 3], "the rest of the provenance is untouched"
    assert shares == {"concept:vault": 2, "concept:seam-wall": 1}


def test_the_reads_and_the_progress_file_beside_the_store_are_rewritten(tmp_path):
    where = an_old_store(tmp_path)

    assert quiet(where)[0] == 0

    rows = ingest._read_json(ingest.reads_path(where, SLUG))
    assert all(row["source"] == SLUG and "book" not in row for row in rows.values())
    held = ingest._read_json(ingest.Progress.beside(where))
    assert "books" not in held and set(held["sources"]) == {SLUG}
    assert held["sources"][SLUG]["folded_at"] == 2


def test_the_migrated_store_reads_back_through_the_sources_view(tmp_path):
    where = an_old_store(tmp_path)

    assert quiet(where)[0] == 0

    listed = ingest.Sources(where).sources()
    assert [one.slug for one in listed] == [SLUG]
    assert (listed[0].title, listed[0].units, listed[0].read, listed[0].wanted) == \
        (TITLE, 2, 2, 4)
    shared = ingest.Sources(where).shared()
    assert [one["source"] for one in shared["sources"]] == [SLUG]
    assert shared["sources"][0]["nodes"] == 2, "the two concepts read from it"


def test_a_fold_after_the_migration_finds_the_source_the_migration_wrote(tmp_path):
    where = an_old_store(tmp_path)
    assert quiet(where)[0] == 0

    assert ingest.fold(where, say=lambda _: None) == 0

    with ingest.Sources(where).store() as store:
        kinds = {n["id"] for n in store.nodes(kind="source")}
    assert kinds == {f"source:{SLUG}"}, "folded onto the migrated node, not beside it"


# -- running it twice, and running it on a store that is already migrated ---------------------


def test_running_the_migration_twice_changes_nothing_the_second_time(tmp_path):
    from ml_stack.graph.store import GraphStore

    where = an_old_store(tmp_path)
    assert quiet(where)[0] == 0
    with GraphStore(where, read_only=True) as store:
        after = (store.nodes(), store.edges(), store.docs())
    files = beside(where)

    code, said = quiet(where)

    assert code == 0 and "nothing to migrate" in said
    with GraphStore(where, read_only=True) as store:
        assert (store.nodes(), store.edges(), store.docs()) == after
    assert beside(where) == files


def a_new_store(tmp_path, *, name="sources.ladybug"):
    """A store the fold wrote since the rename, from reads that already say `source`."""
    where = tmp_path / name
    ingest._write_json(ingest.reads_path(where, SLUG), {
        unit: {"unit": unit, "source": SLUG, "chapter": "1", "section": f"1.{index}",
               "title": "Vault Currents", "pages": [2 * index, 2 * index + 1],
               "seconds": 86.0, "error": "", "calls": [],
               "extracted": {"concepts": [{"name": "vault", "kind": "structure",
                                           "definition": "", "aliases": []}],
                             "relations": [], "figures": [], "key_terms": []}}
        for index, unit in enumerate(UNITS, start=1)})
    progress = ingest.Progress(ingest.Progress.beside(where))
    held = progress.source(SLUG, title=TITLE, path=f"{SLUG}.pdf", sections=4)
    for unit in UNITS:
        held["done"][unit] = {"seconds": 86.0, "error": "", "attempts": 1}
    progress.save()
    ingest.fold(where, say=lambda _: None)
    return where


def test_a_store_written_since_the_rename_is_left_alone(tmp_path):
    where = a_new_store(tmp_path)

    code, said = quiet(where)

    assert code == 0 and "nothing to migrate" in said
    assert not (tmp_path / "_backups").exists(), "a store it will not touch is not copied"


def test_the_migration_on_a_store_that_is_not_there_says_so(tmp_path):
    code, said = quiet(tmp_path / "nothing.ladybug")
    assert code == 1 and "no store at" in said


# -- what it does when it cannot finish -------------------------------------------------------


def test_a_store_that_does_not_read_back_whole_is_put_back(tmp_path, monkeypatch):
    """The verification is what the copy exists for: a store that fails it is restored,
    with the files beside it, and nothing is left half-renamed."""
    from ml_stack.graph.store import GraphStore

    where = an_old_store(tmp_path)
    before = (ingest._read_json(ingest.reads_path(where, SLUG)),
              ingest._read_json(ingest.Progress.beside(where)))
    monkeypatch.setattr(GraphStore, "check", lambda self: ["invented finding"])

    code, said = quiet(where)

    assert code == 1 and "did not verify" in said and "as it was" in said
    monkeypatch.undo()
    with GraphStore(where, read_only=True) as store:
        assert {n["id"] for n in store.nodes()} >= {f"book:{SLUG}"}
        assert store.get_doc(f"ingest:unit:{UNITS[0]}")["book"] == SLUG
    assert (ingest._read_json(ingest.reads_path(where, SLUG)),
            ingest._read_json(ingest.Progress.beside(where))) == before


def test_a_migration_that_stops_partway_puts_the_files_beside_the_store_back(tmp_path,
                                                                            monkeypatch):
    from ml_stack.graph.store import GraphStore

    where = an_old_store(tmp_path)

    def explode(path):
        raise OSError("the disk went away")

    monkeypatch.setattr(migrate_module, "_rewrite_reads", explode)

    code, said = quiet(where)

    assert code == 1 and "the disk went away" in said
    with GraphStore(where, read_only=True) as store:
        assert f"book:{SLUG}" in {n["id"] for n in store.nodes()}
    rows = ingest._read_json(ingest.reads_path(where, SLUG))
    assert all("book" in row for row in rows.values())
    assert "books" in ingest._read_json(ingest.Progress.beside(where))


# -- the store's own name ---------------------------------------------------------------------


def test_a_store_called_shelf_moves_to_sources_with_the_files_beside_it(tmp_path):
    where = an_old_store(tmp_path, name="shelf.ladybug")

    code, said = quiet(where)

    moved = tmp_path / "sources.ladybug"
    assert code == 0 and "moved shelf.ladybug" in said
    assert moved.exists() and not where.exists()
    assert (tmp_path / "sources.ladybug.ingest.json").is_file()
    assert (tmp_path / f"sources.ladybug.{SLUG}.reads.json").is_file()
    assert not (tmp_path / "shelf.ladybug.ingest.json").exists()
    listed = ingest.Sources(moved).sources()
    assert [one.slug for one in listed] == [SLUG] and listed[0].units == 2


def test_a_store_already_called_sources_is_left_where_it_is(tmp_path):
    where = an_old_store(tmp_path)

    code, said = quiet(where)

    assert code == 0 and f"moved {where.name}" not in said
    assert where.exists() and sorted(p.name for p in tmp_path.glob("*.ladybug")) == \
        ["sources.ladybug"]


# -- the command ------------------------------------------------------------------------------


def test_the_command_migrates_a_store_and_prints_what_it_changed(tmp_path, capsys):
    where = an_old_store(tmp_path)

    assert ingest.main(["migrate", "--out", str(where)]) == 0

    said = capsys.readouterr().out
    assert "1 node(s) renamed" in said and "2 edge(s) moved" in said
    assert "2 reads row(s)" in said and "1 progress entry(s)" in said
    assert "reads back whole" in said


def test_the_command_needs_a_store(capsys):
    assert ingest.main(["migrate"]) == 2
    assert "migrate needs --out STORE" in capsys.readouterr().err
