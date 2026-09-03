"""The "Ingest" section of ``ml-stack-bench report``: one row per book on a ``--shelf``.

An ingest run and a bench run are kept in different places -- ``<store>.ingest.json``,
``<store>.<slug>.reads.json`` and the store's own ``ingest:unit:`` documents and hidden
``run:`` nodes, rather than a `bench.save`d row -- so this section is read straight off
`ml_stack.ingest.Shelf` and never off ``kept``. What is tested here is the reading and the
arranging: that a book's counts, cost and run(s) land in the right cells, that a shelf's
total line sums what its books measured, that the store's judged-decisions count is read
when the hygiene pass has run, and that a ``--shelf`` naming nothing prints no section at
all -- the same convention `report()` already keeps for an empty ``extracted``.

Everything is built in ``tmp_path`` with `test_ingest`'s own helpers
(`a_part_read_book`, `a_read`) and invented books; nothing here reads ``~/.ml-stack``,
serves a model or touches a GPU.
"""

from __future__ import annotations

import pathlib

import pytest
from test_ingest import a_part_read_book, a_read

from ml_stack import ingest
from ml_stack.graph.bench.report import report

pytest.importorskip("ladybug")

RUN = "run:20260101T090000"
RUN2 = "run:20260101T113000"


def said(*names, relations=()):
    """An extraction naming ``names`` as concepts and ``relations`` between them."""
    return {"concepts": [{"name": n, "kind": "structure", "definition": f"What a {n} is.",
                          "aliases": []} for n in names],
            "relations": [{"from": a, "rel": r, "to": b} for a, r, b in relations],
            "figures": [], "key_terms": []}


def a_run_read(unit, *, run=RUN, prompt=0, completion=0, seconds=40.0, **kw):
    """One row of a reads file, pointed at a run and carrying one telemetry call."""
    row = a_read(unit, **kw)
    row["seconds"] = seconds
    row["run"] = run
    row["calls"] = [{"prompt_tokens": prompt, "completion_tokens": completion,
                     "seconds": seconds}]
    return row


def a_shelf(tmp_path: pathlib.Path, *, slug: str = "velthorne-open-texts",
           title: str = "Velthorne Open Texts", sections: int = 4, run: str = RUN,
           rows=None, store=None, write_run_node: bool = True, decisions=None):
    """A folded book on a shelf, with a run node and (optionally) judged decisions."""
    rows = rows if rows is not None else [
        a_run_read(f"{slug}:1:1.1", book=slug, run=run, prompt=800, completion=200,
                  seconds=40.0,
                  extracted=said("vault", "charge",
                                 relations=[("vault", "produces", "charge")])),
        a_run_read(f"{slug}:1:1.2", book=slug, section="1.2", pages=(4, 5), run=run,
                  prompt=700, completion=150, seconds=46.0,
                  extracted=said("vault", "seam wall",
                                 relations=[("seam wall", "part_of", "vault")])),
    ]
    where = a_part_read_book(tmp_path, slug=slug, title=title, sections=sections, rows=rows,
                             store=store)
    ingest.fold(where, say=lambda _: None)
    if write_run_node:
        ingest.write_run(where, {"id": run, "label": "an ingest run", "model": "kestrel-8B",
                                 "serving": "llama.cpp current",
                                 "started": "2026-01-01T09:00:00"})
    if decisions is not None:
        from ml_stack.graph.store import GraphStore

        with GraphStore(where) as handle:
            handle.put_doc("tidy:decisions", {"pairs": decisions, "hidden": True})
    return where


# -- the table -----------------------------------------------------------------------------

def test_a_book_is_tabled_with_its_counts_cost_and_run(tmp_path):
    store = a_shelf(tmp_path)
    book = ingest.Shelf(store).book("velthorne-open-texts")
    assert book is not None                                # the fixture folded it

    body = report([], shelves=[str(store)])
    assert "## Ingest" in body
    head = ("| book | read | failed | given up | nodes | edges | seconds | s/unit "
            "| prompt tok | completion tok | tok/unit | run(s) |")
    assert head in body
    assert (f"| Velthorne Open Texts | 2/4 | 0 | 0 | {book.folded_nodes} "
            f"| {book.folded_edges} | 86 | 43.0 | 1500 | 350 | 925 "
            "| kestrel-8B / llama.cpp current (2026-01-01T09:00:00) |") in body


def test_failed_and_given_up_units_are_counted_apart(tmp_path):
    slug = "velthorne-open-texts"
    rows = [
        a_run_read(f"{slug}:1:1.1", book=slug,
                  extracted=said("vault"), prompt=400, completion=100, seconds=30.0),
        a_run_read(f"{slug}:1:1.2", book=slug, section="1.2", pages=(4, 5),
                  error="ServerError: the reply was cut off", extracted={},
                  prompt=100, completion=0, seconds=12.0),
    ]
    store = a_shelf(tmp_path, rows=rows)
    # one attempt already failed; a second attempt at or past GIVE_UP is left alone
    progress = ingest.Progress(ingest.Progress.beside(store))
    progress.book(slug)["done"][f"{slug}:1:1.2"]["attempts"] = ingest.GIVE_UP
    progress.save()

    body = report([], shelves=[str(store)])
    assert "| Velthorne Open Texts | 1/4 | 1 | 1 |" in body


def test_a_book_with_no_run_recorded_shows_a_dash(tmp_path):
    slug = "velthorne-open-texts"
    rows = [a_read(f"{slug}:1:1.1", book=slug, extracted=said("vault"))]  # no "run" key
    store = a_shelf(tmp_path, rows=rows, write_run_node=False)

    body = report([], shelves=[str(store)])
    lines = [line for line in body.splitlines() if line.startswith("| Velthorne")]
    assert len(lines) == 1
    assert lines[0].rstrip().endswith("| - |")


def test_a_run_id_the_store_holds_no_node_for_prints_the_bare_id(tmp_path):
    store = a_shelf(tmp_path, write_run_node=False)
    body = report([], shelves=[str(store)])
    assert f"| {RUN} |" in body


def test_two_runs_reading_the_same_book_are_both_named(tmp_path):
    slug = "velthorne-open-texts"
    rows = [
        a_run_read(f"{slug}:1:1.1", book=slug, run=RUN, extracted=said("vault")),
        a_run_read(f"{slug}:1:1.2", book=slug, section="1.2", pages=(4, 5), run=RUN2,
                  extracted=said("charge")),
    ]
    store = a_shelf(tmp_path, rows=rows, write_run_node=False)
    ingest.write_run(store, {"id": RUN, "model": "kestrel-8B", "serving": "llama.cpp",
                             "started": "2026-01-01T09:00:00"})
    ingest.write_run(store, {"id": RUN2, "model": "ember-2B", "serving": "llama.cpp",
                             "started": "2026-01-01T11:30:00"})

    body = report([], shelves=[str(store)])
    assert "kestrel-8B / llama.cpp (2026-01-01T09:00:00)" in body
    assert "ember-2B / llama.cpp (2026-01-01T11:30:00)" in body


# -- the total line and the decisions count -------------------------------------------------

def test_the_shelf_total_line_sums_across_its_books(tmp_path):
    store = tmp_path / "shelf.ladybug"
    a_shelf(tmp_path, store=store)
    a_shelf(tmp_path, store=store, slug="lattice-studies", title="Lattice Studies",
           sections=2, run=RUN,
           rows=[a_run_read("lattice-studies:1:1.1", book="lattice-studies", run=RUN,
                            prompt=300, completion=100, seconds=20.0,
                            extracted=said("glimmer node"))])

    body = report([], shelves=[str(store)])
    assert ("2 book(s), 3/6 unit(s) read, 0 failed (0 given up), 106s (35.3 s/unit), "
            "1800 prompt and 450 completion token(s) (750 tok/unit).") in body


def test_the_judged_decisions_count_is_read_when_the_hygiene_pass_has_run(tmp_path):
    store = a_shelf(tmp_path, decisions={
        "concept:vault|concept:reservoir": {"verdict": "different", "why": "distinct parts"}})
    body = report([], shelves=[str(store)])
    assert "1 pair(s) of names judged for merge." in body


def test_no_decisions_document_says_nothing_about_judging(tmp_path):
    store = a_shelf(tmp_path)
    body = report([], shelves=[str(store)])
    assert "judged for merge" not in body


# -- what prints no section -------------------------------------------------------------

def test_no_shelf_argument_prints_no_section():
    body = report([])
    assert "## Ingest" not in body


def test_a_shelf_naming_nothing_prints_no_section(tmp_path):
    """A ``--shelf`` pointed at a store with no book behind it -- nothing ever ingested --
    is left out entirely, the way an empty ``extracted`` leaves out "Extraction": a
    heading over nothing reads as a book that was read and found empty."""
    body = report([], shelves=[str(tmp_path / "nothing-here.ladybug")])
    assert "## Ingest" not in body


def test_one_empty_shelf_beside_one_real_one_only_tables_the_real_one(tmp_path):
    store = a_shelf(tmp_path)
    body = report([], shelves=[str(tmp_path / "nothing-here.ladybug"), str(store)])
    assert body.count("## Ingest") == 1
    assert "nothing-here" not in body
    assert "Velthorne Open Texts" in body


# -- alongside the rest of the document ---------------------------------------------------

def test_ingest_sits_beside_answering_runs_in_the_same_document(tmp_path):
    pytest.importorskip("ladybug")
    from ml_stack.graph import bench

    answering_store = str(tmp_path / "runs.ladybug")
    rows = [bench.Row(label="kestrel-plain", question=f"who runs the vault, q{n}?",
                      expected=["person:marisol-quen"], shown=["person:marisol-quen"],
                      seconds=10.0, calls=1, answer_chars=80, processed_tokens=400,
                      completion_tokens=50) for n in range(8)]
    bench.save(answering_store, rows, held={"model": "kestrel.gguf", "context": 32768,
                                            "slots": 1,
                                            "binary": "/builds/current/llama-server"})
    kept = bench.runs(answering_store)

    shelf_store = a_shelf(tmp_path, store=tmp_path / "shelf.ladybug")
    body = report(kept, shelves=[str(shelf_store)])
    assert "## Answering, per model" in body
    assert "## Ingest" in body
    assert body.index("## Answering, per model") < body.index("## Ingest")


def test_the_text_rendering_carries_the_same_numbers_without_pipes(tmp_path):
    store = a_shelf(tmp_path)
    body = report([], shelves=[str(store)], md=False)
    assert "|" not in body
    assert "INGEST" in body
    assert "Velthorne Open Texts" in body
    assert "kestrel-8B / llama.cpp current (2026-01-01T09:00:00)" in body
