"""A shelf of books read into one store: what it holds together, what it cost, how often it
is folded, and what happens when the server resets inside one read.

Every book, concept and relation here is invented. Nothing opens a port a model could be
behind and nothing reads the machine this runs on.
"""

from __future__ import annotations

import json
import time

import pytest
from test_ingest import a_part_read_book, a_read, in_store, said

from ml_stack import ingest

pytest.importorskip("ladybug")

FIELD_GUIDE = "ambleford-field-guide"
OPEN_TEXTS = "velthorne-open-texts"


def a_second_book(store, *, slug=FIELD_GUIDE, title="Ambleford Field Guide"):
    """A second book beside the first, naming one concept the first also names."""
    rows = [a_read(f"{slug}:1:1.1", book=slug, title="Spore Blooms",
                   extracted=said("vault", "spore bloom",
                                  relations=[("spore bloom", "part_of", "vault")])),
            a_read(f"{slug}:2:2.1", book=slug, chapter="2", section="2.1", pages=(6, 7),
                   title="Blooming", extracted=said("spore bloom", "damp season",
                                                    relations=[("spore bloom", "requires",
                                                                "damp season")]))]
    return a_part_read_book(store.parent, slug=slug, rows=rows, title=title, sections=6,
                            store=store)


def a_shelf_of_two(tmp_path):
    """Two books folded into one store, sharing the concept both of them name."""
    store = a_part_read_book(tmp_path)
    a_second_book(store)
    assert ingest.fold(store, say=lambda _: None) == 0
    return store


def a_shelf_spelled_apart(tmp_path):
    """Two books folded into one store, one spelling `seam wal` what the other spells
    `seam wall`: a letter apart, so the fold leaves them two nodes for the hygiene pass."""
    store = a_part_read_book(tmp_path)
    rows = [a_read(f"{FIELD_GUIDE}:1:1.1", book=FIELD_GUIDE, title="Spore Blooms",
                   extracted=said("spore bloom", "seam wal",
                                  relations=[("spore bloom", "part_of", "seam wal")]))]
    a_part_read_book(tmp_path, slug=FIELD_GUIDE, rows=rows, title="Ambleford Field Guide",
                     sections=6, store=store)
    assert ingest.fold(store, say=lambda _: None) == 0
    return store


# -- the shelf at a glance -----------------------------------------------------------------


def test_the_shelf_names_each_book_and_what_the_store_holds_for_it(tmp_path):
    store = a_shelf_of_two(tmp_path)

    got = ingest.Shelf(store).shared()

    books = {b["book"]: b for b in got["books"]}
    assert set(books) == {OPEN_TEXTS, FIELD_GUIDE}
    assert books[OPEN_TEXTS]["title"] == "Velthorne Open Texts"
    assert books[OPEN_TEXTS]["read"] == 2 and books[OPEN_TEXTS]["wanted"] == 4
    # vault, charge, seam wall -- and the second book's own three
    assert books[OPEN_TEXTS]["nodes"] == 3 and books[FIELD_GUIDE]["nodes"] == 3
    assert books[OPEN_TEXTS]["edges"] == 2 and books[FIELD_GUIDE]["edges"] == 2


def test_a_concept_two_books_name_is_listed_with_the_books_that_name_it(tmp_path):
    store = a_shelf_of_two(tmp_path)

    got = ingest.Shelf(store).shared()

    assert [one["label"] for one in got["shared"]] == ["vault"]
    assert got["shared"][0]["id"] == "concept:vault"
    assert got["shared"][0]["books"] == [FIELD_GUIDE, OPEN_TEXTS]


def test_an_edge_whose_ends_came_from_different_books_is_a_relation_between_them(tmp_path):
    """What the hygiene pass leaves when it joins one book's name to another's: an edge
    with a concept of each book at its ends."""
    from ml_stack.graph.store import GraphStore

    store = a_shelf_of_two(tmp_path)
    with GraphStore(store) as held:
        held.write({"nodes": [], "edges": [{"source": "concept:spore-bloom", "rel": "requires",
                                            "target": "concept:charge", "weight": 3}]})

    got = ingest.Shelf(store).shared()

    assert len(got["between"]) == 1
    one = got["between"][0]
    assert one["source"] == "concept:spore-bloom" and one["target"] == "concept:charge"
    assert one["rel"] == "requires"
    assert one["books"] == [[FIELD_GUIDE], [OPEN_TEXTS]]


def test_the_pairs_the_hygiene_pass_judged_are_counted(tmp_path):
    from ml_stack.graph.store import GraphStore
    from ml_stack.graph.tidy import DECISIONS

    store = a_shelf_of_two(tmp_path)
    with GraphStore(store) as held:
        held.put_doc(DECISIONS, {"pairs": {
            "concept:vault|concept:vaults": {"verdict": "same", "why": "a plural"},
            "concept:charge|concept:charges": {"verdict": "different", "why": "two things"},
            "concept:seam-wall|concept:seam-walls": {"verdict": "unsure", "why": ""}},
            "hidden": True})

    judged = ingest.Shelf(store).shared()["decisions"]

    assert judged == {"pairs": 3, "same": 1, "different": 1, "unsure": 1}


def test_a_store_no_pass_has_judged_counts_no_pairs(tmp_path):
    store = a_shelf_of_two(tmp_path)

    assert ingest.Shelf(store).shared()["decisions"]["pairs"] == 0


# -- a name joined across books on the way in ----------------------------------------------


def a_shelf_folded_in_turn(tmp_path):
    """One book folded, then a second whose unit names the first book's `seam wall` in the
    plural: the fold lands `seam walls` on the node the store already holds."""
    store = a_part_read_book(tmp_path)
    assert ingest.main(["fold", "--out", str(store)]) == 0
    rows = [a_read(f"{FIELD_GUIDE}:1:1.1", book=FIELD_GUIDE, title="Spore Blooms",
                   extracted=said("spore bloom", "seam walls",
                                  relations=[("spore bloom", "part_of", "seam walls")]))]
    a_part_read_book(tmp_path, slug=FIELD_GUIDE, rows=rows, title="Ambleford Field Guide",
                     sections=6, store=store)
    return store


def test_the_fold_writes_each_name_it_lands_on_an_existing_node_with_the_units_of_both(
        tmp_path):
    from ml_stack.graph.store import GraphStore
    from ml_stack.graph.tidy import MERGES

    store = a_shelf_folded_in_turn(tmp_path)
    assert ingest.main(["fold", "--out", str(store)]) == 0

    with GraphStore(store, read_only=True) as held:
        merges = held.get_doc(MERGES)["merges"]
        assert "concept:seam-walls" not in {n["id"] for n in held.nodes()}
    assert len(merges) == 1
    one = merges[0]
    assert (one["kept"], one["kept_label"]) == ("concept:seam-wall", "seam wall")
    assert (one["gone"], one["gone_label"]) == ("concept:seam-walls", "seam walls")
    assert one["kind"] == "structure"
    assert one["edges_moved"] == 1, "the relation; the edge to its book is written after"
    assert one["kept_from"] == [f"{OPEN_TEXTS}:1:1.2"]
    assert one["gone_from"] == [f"{FIELD_GUIDE}:1:1.1"]
    assert one["at"]


def test_a_fold_repeated_writes_the_landing_once(tmp_path):
    from ml_stack.graph.store import GraphStore
    from ml_stack.graph.tidy import MERGES

    store = a_shelf_folded_in_turn(tmp_path)
    ingest.main(["fold", "--out", str(store)])
    ingest.main(["fold", "--out", str(store)])

    with GraphStore(store, read_only=True) as held:
        assert len(held.get_doc(MERGES)["merges"]) == 1


def test_a_name_the_fold_landed_across_two_books_is_between_them_and_the_command_prints_it(
        tmp_path, capsys):
    store = a_shelf_folded_in_turn(tmp_path)
    ingest.main(["fold", "--out", str(store)])

    assert ingest.Shelf(store).between_books() == [
        {"a_label": "seam wall", "a_book": OPEN_TEXTS, "b_label": "seam walls",
         "b_book": FIELD_GUIDE, "kind": "structure", "weight": 5}], \
        "the joined node's four mentions, two from each book, and the one edge rewired"
    assert ingest.main(["shelf", "--out", str(store)]) == 0
    said_out = capsys.readouterr().out
    assert "between books (1)" in said_out
    assert f"seam wall ({OPEN_TEXTS}) = seam walls ({FIELD_GUIDE})  structure  5" in said_out


def test_a_book_folded_again_keeps_what_the_other_book_gave_a_shared_node(tmp_path):
    """The fold of every book runs in slug order, so the first book is folded again after
    the second landed on its node: the node keeps both books' units, mentions and names."""
    from ml_stack.graph.store import GraphStore

    store = a_shelf_folded_in_turn(tmp_path)
    ingest.main(["fold", "--out", str(store)])
    ingest.fold_into(store, OPEN_TEXTS)
    ingest.fold_into(store, OPEN_TEXTS)

    with GraphStore(store, read_only=True) as held:
        wall = next(n for n in held.nodes() if n["id"] == "concept:seam-wall")
    assert wall["mentions"] == 4
    assert wall["provenance"] == [f"{FIELD_GUIDE}:1:1.1", f"{OPEN_TEXTS}:1:1.2"]
    assert wall["attrs"]["aliases"] == ["seam walls"]
    assert ingest.Shelf(store).between_books()[0]["weight"] == 5


def test_a_book_read_further_grows_its_own_share_of_a_shared_node(tmp_path):
    from ml_stack.graph.store import GraphStore

    store = a_shelf_folded_in_turn(tmp_path)
    ingest.main(["fold", "--out", str(store)])
    rows = ingest.Shelf(store).reads(OPEN_TEXTS)
    rows.append(a_read(f"{OPEN_TEXTS}:2:2.1", chapter="2", section="2.1", pages=(8, 9),
                       extracted=said("seam wall")))
    ingest._write_json(ingest.reads_path(store, OPEN_TEXTS), {r["unit"]: r for r in rows})
    ingest.fold_into(store, OPEN_TEXTS)

    with GraphStore(store, read_only=True) as held:
        wall = next(n for n in held.nodes() if n["id"] == "concept:seam-wall")
    assert wall["mentions"] == 5, "two from the field guide, three of its own"
    assert wall["provenance"] == [f"{FIELD_GUIDE}:1:1.1", f"{OPEN_TEXTS}:1:1.2",
                                  f"{OPEN_TEXTS}:2:2.1"]


def test_a_fold_in_memory_a_dry_fold_and_a_read_only_absorb_write_no_landing(tmp_path):
    from ml_stack.graph.store import GraphStore
    from ml_stack.graph.tidy import MERGES, absorb

    store = a_shelf_folded_in_turn(tmp_path)
    shelf = ingest.Shelf(store)

    graph = shelf.graph(FIELD_GUIDE)
    assert {n["id"] for n in graph["nodes"]} >= {"concept:seam-walls"}
    got = ingest.fold_into(store, FIELD_GUIDE, dry_run=True)
    assert got["nodes"] == 2
    assert absorb(store, graph).mapped_plural == 1

    with GraphStore(store, read_only=True) as held:
        assert held.get_doc(MERGES) is None
        assert "concept:seam-walls" not in {n["id"] for n in held.nodes()}


# -- a name joined across books by the hygiene pass ------------------------------------------


WRITTEN = {"seam wal": "seam wall"}
"""What a person hands the hygiene pass about the two spellings."""


def test_tidy_writes_each_merge_it_makes_to_the_store_with_the_units_both_names_came_from(
        tmp_path):
    from ml_stack.graph.store import GraphStore
    from ml_stack.graph.tidy import MERGES, tidy

    store = a_shelf_spelled_apart(tmp_path)
    assert tidy(store, dry_run=False, written=WRITTEN).merged_nodes == 1

    with GraphStore(store, read_only=True) as held:
        merges = held.get_doc(MERGES)["merges"]
        assert "concept:seam-wal" not in {n["id"] for n in held.nodes()}
    assert len(merges) == 1
    one = merges[0]
    assert (one["kept"], one["kept_label"]) == ("concept:seam-wall", "seam wall")
    assert (one["gone"], one["gone_label"]) == ("concept:seam-wal", "seam wal")
    assert one["kind"] == "structure"
    assert one["edges_moved"] == 2, "the relation, and the read_from edge to its book"
    assert one["kept_from"] == [f"{OPEN_TEXTS}:1:1.2"]
    assert one["gone_from"] == [f"{FIELD_GUIDE}:1:1.1"]
    assert one["at"]


def test_a_dry_tidy_writes_no_merge(tmp_path):
    from ml_stack.graph.store import GraphStore
    from ml_stack.graph.tidy import MERGES, tidy

    store = a_shelf_spelled_apart(tmp_path)
    assert tidy(store, written=WRITTEN).merged_nodes == 1

    with GraphStore(store, read_only=True) as held:
        assert held.get_doc(MERGES) is None


def test_a_merge_is_written_once_however_often_tidy_runs(tmp_path):
    from ml_stack.graph.store import GraphStore
    from ml_stack.graph.tidy import MERGES, tidy

    store = a_shelf_spelled_apart(tmp_path)
    tidy(store, dry_run=False, written=WRITTEN)
    with GraphStore(store) as held:
        held.write({"nodes": [{"id": "concept:seam-wal", "kind": "concept",
                               "label": "seam wal", "mentions": 1,
                               "attrs": {}, "provenance": [f"{FIELD_GUIDE}:2:2.1"]}],
                    "edges": []})
    tidy(store, dry_run=False, written=WRITTEN)

    with GraphStore(store, read_only=True) as held:
        merges = held.get_doc(MERGES)["merges"]
    assert [(m["kept"], m["gone"]) for m in merges] == [("concept:seam-wall",
                                                          "concept:seam-wal")]


def test_a_name_merged_across_two_books_is_between_them_with_a_weight(tmp_path):
    from ml_stack.graph.tidy import tidy

    store = a_shelf_spelled_apart(tmp_path)
    tidy(store, dry_run=False, written=WRITTEN)

    got = ingest.Shelf(store).shared()

    assert got["merged"] == ingest.Shelf(store).between_books()
    assert got["merged"] == [{"a_label": "seam wall", "a_book": OPEN_TEXTS,
                              "b_label": "seam wal", "b_book": FIELD_GUIDE,
                              "kind": "structure", "weight": 6}], \
        "the joined node's four mentions, two from each book, and the two edges moved"


def test_a_name_merged_within_one_book_is_not_between_books(tmp_path):
    from ml_stack.graph.tidy import tidy

    from ml_stack.graph.store import GraphStore

    store = a_part_read_book(tmp_path)
    ingest.fold(store, say=lambda _: None)
    with GraphStore(store) as held:
        held.write({"nodes": [{"id": "concept:seam-wal", "kind": "structure",
                               "label": "seam wal", "mentions": 1, "attrs": {},
                               "provenance": [f"{OPEN_TEXTS}:1:1.1"]}],
                    "edges": [{"source": "concept:seam-wal", "rel": "read_from",
                               "target": f"book:{OPEN_TEXTS}", "weight": 1}]})
    assert tidy(store, dry_run=False, written=WRITTEN).merged_nodes == 1

    assert ingest.Shelf(store).between_books() == []


def test_a_shelf_never_tidied_has_nothing_between_books(tmp_path):
    assert ingest.Shelf(a_shelf_spelled_apart(tmp_path)).between_books() == []


def a_shelf_a_plural_apart(tmp_path):
    """Two books folded into one store, one naming `seam walls` what the other names
    `seam wall`: the fold's own absorb lands the second on the first, with no model."""
    store = a_part_read_book(tmp_path)
    rows = [a_read(f"{FIELD_GUIDE}:1:1.1", book=FIELD_GUIDE, title="Spore Blooms",
                   extracted=said("spore bloom", "seam walls",
                                  relations=[("spore bloom", "part_of", "seam walls")]))]
    a_part_read_book(tmp_path, slug=FIELD_GUIDE, rows=rows, title="Ambleford Field Guide",
                     sections=6, store=store)
    return store


def test_a_fold_logs_the_name_it_lands_on_another_book_s_node(tmp_path, capsys):
    from ml_stack.graph.store import GraphStore
    from ml_stack.graph.tidy import MERGES

    store = a_shelf_a_plural_apart(tmp_path)
    assert not store.exists(), "nothing has written the log, or anything else, yet"

    assert ingest.fold(store, say=lambda _: None) == 0

    with GraphStore(store, read_only=True) as held:
        assert len(held.get_doc(MERGES)["merges"]) == 1
    assert ingest.main(["shelf", "--out", str(store)]) == 0
    said_out = capsys.readouterr().out
    assert "between books (1)" in said_out
    assert f"seam walls ({FIELD_GUIDE}) = seam wall ({OPEN_TEXTS})" in said_out


def test_the_shelf_command_prints_the_names_joined_across_books(tmp_path, capsys):
    from ml_stack.graph.tidy import tidy

    store = a_shelf_spelled_apart(tmp_path)
    tidy(store, dry_run=False, written=WRITTEN)

    assert ingest.main(["shelf", "--out", str(store)]) == 0

    said_out = capsys.readouterr().out
    assert "between books (1)" in said_out
    assert f"seam wall ({OPEN_TEXTS}) = seam wal ({FIELD_GUIDE})  structure  6" in said_out


def test_the_shelf_command_prints_the_books_the_shared_concepts_and_the_judged_pairs(
        tmp_path, capsys):
    store = a_shelf_of_two(tmp_path)

    assert ingest.main(["shelf", "--out", str(store)]) == 0

    said_out = capsys.readouterr().out
    assert "2 book(s)" in said_out
    assert OPEN_TEXTS in said_out and "Ambleford Field Guide" in said_out
    assert "concepts in more than one book (1)" in said_out
    assert "vault" in said_out and f"{FIELD_GUIDE}, {OPEN_TEXTS}" in said_out
    assert (f"between books (0): no log of the names the books share; "
            f"ml-stack-ingest fold --out {store} re-folds each book from its reads and "
            f"writes one") in said_out
    assert "relations between books (0)" in said_out
    assert "judged: nothing" in said_out


def test_the_shelf_command_asks_for_a_tidy_once_a_merge_is_logged_within_one_book(
        tmp_path, capsys):
    from ml_stack.graph.store import GraphStore
    from ml_stack.graph.tidy import MERGES, tidy

    store = a_part_read_book(tmp_path)
    ingest.fold(store, say=lambda _: None)
    with GraphStore(store) as held:
        held.write({"nodes": [{"id": "concept:seam-wal", "kind": "structure",
                               "label": "seam wal", "mentions": 1, "attrs": {},
                               "provenance": [f"{OPEN_TEXTS}:1:1.1"]}],
                    "edges": [{"source": "concept:seam-wal", "rel": "read_from",
                               "target": f"book:{OPEN_TEXTS}", "weight": 1}]})
    assert tidy(store, dry_run=False, written=WRITTEN).merged_nodes == 1
    with GraphStore(store, read_only=True) as held:
        assert len(held.get_doc(MERGES)["merges"]) == 1, "a merge is logged, none across books"

    assert ingest.main(["shelf", "--out", str(store)]) == 0

    assert (f"between books (0): no concept merged across books yet; "
            f"ml-stack-ingest tidy --out {store}") in capsys.readouterr().out


def test_the_shelf_command_takes_a_sample_size(tmp_path, capsys):
    store = a_shelf_of_two(tmp_path)
    a_part_read_book(tmp_path, slug="kessleton-mills", title="Kessleton Mills", store=store,
                     rows=[a_read("kessleton-mills:1:1.1", book="kessleton-mills",
                                  extracted=said("vault", "charge"))])
    ingest.fold(store, say=lambda _: None)

    assert ingest.main(["shelf", "--out", str(store), "--sample", "1"]) == 0

    said_out = capsys.readouterr().out
    assert "concepts in more than one book (2)" in said_out
    assert "... and 1 more" in said_out


def test_the_shelf_command_on_a_store_that_is_not_there_says_so(tmp_path, capsys):
    assert ingest.main(["shelf", "--out", str(tmp_path / "nothing.ladybug")]) == 1
    assert "no store at" in capsys.readouterr().out


def test_the_shelf_command_needs_a_store(capsys):
    assert ingest.main(["shelf"]) == 2
    assert "shelf needs --out STORE" in capsys.readouterr().err


# -- what each unit cost -------------------------------------------------------------------


def a_call(prompt, completion):
    return {"prompt_tokens": prompt, "completion_tokens": completion, "seconds": 1.0}


def test_a_books_tokens_are_added_up_over_the_calls_each_unit_made(tmp_path):
    rows = [a_read(f"{OPEN_TEXTS}:1:1.1", extracted=said("vault")),
            a_read(f"{OPEN_TEXTS}:1:1.2", section="1.2", extracted=said("charge"))]
    rows[0]["calls"] = [a_call(1200, 400), a_call(300, 90)]
    rows[1]["calls"] = [a_call(1100, 350)]
    store = a_part_read_book(tmp_path, rows=rows)

    book = ingest.Shelf(store).book(OPEN_TEXTS)

    assert (book.prompt_tokens, book.completion_tokens) == (2600, 840)
    assert book.tokens == 3440 and book.tokens_per_unit == 1720
    assert book.per_unit == 86.0


def test_a_read_from_before_the_calls_were_kept_costs_nothing_rather_than_failing(tmp_path):
    """Old records stay readable: a row with no `calls` is zero, not an error."""
    rows = [a_read(f"{OPEN_TEXTS}:1:1.1", extracted=said("vault"))]
    rows[0].pop("calls")
    store = a_part_read_book(tmp_path, rows=rows)

    book = ingest.Shelf(store).book(OPEN_TEXTS)

    assert (book.prompt_tokens, book.completion_tokens, book.tokens_per_unit) == (0, 0, 0)


def test_status_says_what_each_book_cost_and_what_the_shelf_cost(tmp_path, capsys):
    rows = [a_read(f"{OPEN_TEXTS}:1:1.1", extracted=said("vault")),
            a_read(f"{OPEN_TEXTS}:1:1.2", section="1.2", extracted=said("charge"))]
    for row in rows:
        row["calls"] = [a_call(1000, 250)]
    store = a_part_read_book(tmp_path, rows=rows)

    assert ingest.main(["status", "--out", str(store)]) == 0

    said_out = capsys.readouterr().out
    assert "cost: 2,000 read + 500 written token(s) over 2 unit(s)" in said_out
    assert "86.0 s/unit, 1,250 tokens/unit" in said_out
    assert "shelf: 2,000 read + 500 written token(s) over 2 unit(s)" in said_out


def test_status_totals_the_cost_over_every_book_on_the_shelf(tmp_path, capsys):
    store = a_part_read_book(tmp_path)
    held = ingest._read_json(ingest.reads_path(store, OPEN_TEXTS))
    for row in held.values():
        row["calls"] = [a_call(500, 100)]
    ingest._write_json(ingest.reads_path(store, OPEN_TEXTS), held)
    a_second_book(store)

    assert ingest.main(["status", "--out", str(store)]) == 0

    said_out = capsys.readouterr().out
    assert "shelf: 1,000 read + 200 written token(s) over 4 unit(s)" in said_out


# -- how often a book is folded ------------------------------------------------------------


def test_the_interval_is_what_it_always_was_until_a_fold_has_been_measured():
    assert ingest._fold_interval(0.0) == ingest.FOLD_EVERY
    assert ingest._fold_interval(ingest.FOLD_SECONDS) == ingest.FOLD_EVERY
    assert not ingest._time_to_fold(ingest.FOLD_EVERY - 1, True)
    assert ingest._time_to_fold(ingest.FOLD_EVERY, True)


def test_the_interval_widens_with_what_the_last_fold_took():
    """A fold of a minute is not paid at every chapter end of a long book."""
    assert ingest._fold_interval(21.0) == 2 * ingest.FOLD_EVERY
    assert ingest._fold_interval(82.0) == 5 * ingest.FOLD_EVERY

    assert not ingest._time_to_fold(ingest.FOLD_EVERY, True, seconds=82.0)
    assert ingest._time_to_fold(5 * ingest.FOLD_EVERY, True, seconds=82.0)
    assert not ingest._time_to_fold(5 * ingest.FOLD_EVERY, False, seconds=82.0)
    assert ingest._time_to_fold(10 * ingest.FOLD_EVERY, False, seconds=82.0)


def test_the_progress_file_records_what_the_fold_took(tmp_path):
    store = a_part_read_book(tmp_path)

    got = ingest.fold_into(store, OPEN_TEXTS)

    held = json.loads(ingest.Progress.beside(store).read_text())["books"][OPEN_TEXTS]
    assert held["folded_seconds"] == got["seconds"] >= 0.0


def test_a_dry_run_says_what_the_fold_would_cost_and_writes_nothing(tmp_path):
    store = a_part_read_book(tmp_path)

    got = ingest.fold_into(store, OPEN_TEXTS, dry_run=True)

    assert got["seconds"] >= 0.0
    assert "folded_seconds" not in json.loads(
        ingest.Progress.beside(store).read_text())["books"][OPEN_TEXTS]


def a_wide_book(tmp_path, units, *, slug=OPEN_TEXTS, store=None):
    """``units`` units of a book whose vocabulary is as wide as the book is long."""
    words = [f"vault {n}" for n in range(units // 2)] + [f"seam {n}" for n in range(units // 2)]
    rows = []
    for n in range(units):
        chapter = str(n // 15 + 1)
        picks = [words[(n * 5 + k) % len(words)] for k in range(5)]
        rows.append(a_read(f"{slug}:{chapter}:{chapter}.{n % 15 + 1}", book=slug,
                           chapter=chapter, section=f"{chapter}.{n % 15 + 1}",
                           pages=(n * 2, n * 2 + 1),
                           extracted=said(*picks, relations=[(picks[i], "part_of", picks[i + 1])
                                                             for i in range(4)])))
    return a_part_read_book(tmp_path, slug=slug, rows=rows, sections=units * 2, store=store)


@pytest.mark.slow
def test_what_a_fold_costs_as_a_book_grows_is_what_the_interval_is_chosen_from(tmp_path,
                                                                              capsys):
    """The measurement `FOLD_SECONDS` and `_fold_interval` are set from.

    `fold_book` folds every name against every other, so it grows with the square of the
    vocabulary; the write grows with the units. Measured here on this machine, and printed,
    because the numbers in `FOLD_SECONDS` are of one laptop and one afternoon.
    """
    measured = {}
    for units in (300, 1000, 3000):
        store = tmp_path / f"{units}.ladybug"
        a_wide_book(tmp_path, units, store=store)
        rows = ingest.Shelf(store).reads(OPEN_TEXTS)
        began = time.time()
        graph = ingest.fold_book(rows, ingest.units_of(rows))
        folding = time.time() - began
        got = ingest.fold_into(store, OPEN_TEXTS)
        measured[units] = (folding, got["seconds"], len(graph["nodes"]))
        with capsys.disabled():
            print(f"\n  {units:>5} units, {len(graph['nodes']):>5} concepts: "
                  f"fold {folding:6.2f}s, fold and write {got['seconds']:6.2f}s "
                  f"-- folds {ingest._fold_interval(got['seconds'])} units apart")

    assert measured[3000][0] > measured[300][0], "a wider vocabulary costs more to fold"
    assert measured[3000][1] > ingest.FOLD_SECONDS, "which is why the interval widens"
    assert ingest._fold_interval(measured[300][1]) == ingest.FOLD_EVERY
    assert ingest._fold_interval(measured[3000][1]) > ingest.FOLD_EVERY


# -- a server that reset inside one read ----------------------------------------------------


def a_reset(unit):
    return ingest.Read(unit=unit.id, book=unit.book, chapter=unit.chapter,
                       section=unit.section, title=unit.section_title,
                       error="ServerUnreachable: cannot reach it (Connection reset by peer)")


def test_a_reset_mid_request_is_read_once_more_and_the_second_answer_stands(monkeypatch):
    """The server still answers, so the connection dropped inside the request."""
    tries = []

    def once(client, unit, shape, **kw):
        tries.append(unit.id)
        if len(tries) == 1:
            return a_reset(unit)
        return ingest.Read(unit=unit.id, book=unit.book, chapter=unit.chapter,
                           section=unit.section, title=unit.section_title, concepts=3)

    monkeypatch.setattr(ingest, "extract_unit", once)
    monkeypatch.setattr(ingest, "_alive", lambda client: True)
    from test_ingest import a_unit

    row = ingest.read_unit(object(), a_unit(), {})

    assert tries == ["lattice:1:1.1", "lattice:1:1.1"]
    assert row.error == "" and row.concepts == 3 and row.retried


def test_the_second_failure_is_the_units_and_it_is_not_read_a_third_time(monkeypatch):
    tries = []

    def always(client, unit, shape, **kw):
        tries.append(unit.id)
        return a_reset(unit)

    monkeypatch.setattr(ingest, "extract_unit", always)
    monkeypatch.setattr(ingest, "_alive", lambda client: True)
    from test_ingest import a_unit

    row = ingest.read_unit(object(), a_unit(), {})

    assert len(tries) == 2, "one retry, not a loop"
    assert row.error.startswith("ServerUnreachable") and row.retried


def test_a_server_that_is_gone_is_not_read_again(monkeypatch):
    """The dead-server path is unchanged: one failure, and the run stops on it."""
    tries = []

    def gone(client, unit, shape, **kw):
        tries.append(unit.id)
        return a_reset(unit)

    monkeypatch.setattr(ingest, "extract_unit", gone)
    monkeypatch.setattr(ingest, "_alive", lambda client: False)
    from test_ingest import a_unit

    row = ingest.read_unit(object(), a_unit(), {})

    assert len(tries) == 1 and not row.retried


def test_a_run_whose_server_reset_reads_the_unit_again_and_carries_on(tmp_path, server,
                                                                     monkeypatch, capsys):
    """A reset on the first unit costs one extra read; the book is still finished."""
    from test_ingest import a_shelf

    book, instance, _asked = a_shelf(tmp_path, server)
    store = tmp_path / "shelf.ladybug"
    tries: list[str] = []
    real = ingest.extract_unit

    def resetting(client, unit, shape, **kw):
        tries.append(unit.id)
        if len(tries) == 1:
            return a_reset(unit)
        return real(client, unit, shape, **kw)

    monkeypatch.setattr(ingest, "extract_unit", resetting)
    monkeypatch.setattr(ingest, "_alive", lambda client: True)
    assert ingest.main([book, "--out", str(store), "--base-url", instance.base_url]) == 0

    assert len(tries) == 3, "two units, one of them read twice"
    assert "read again after a reset" in capsys.readouterr().out
    assert in_store(store) >= {"book:lattice", "concept:glimmer-node"}
