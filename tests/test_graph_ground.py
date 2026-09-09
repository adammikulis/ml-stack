"""Citing what an answer is built on, and checking that it was built on anything.

`tools_for(cite=True)` shows the model where every entry it reads came from and gives it
the passage behind one; `grounded` says, afterwards, which of the entries the answer is
about had a source and which the model never read.
"""

from ml_stack.graph.ask import Answer, look_around, look_at, quotes, tools_for
from ml_stack.graph.asking import Asking
from ml_stack.graph.ground import grounded, ungrounded

PASSAGE = ("A glimmer node is a point of the lattice that holds charge between pulses. "
           "Every glimmer node sits inside a cinder vault.")

GRAPH = {
    "nodes": [
        {"id": "concept:glimmer-node", "kind": "concept", "label": "glimmer node",
         "mentions": 6, "attrs": {"definition": "a point of the lattice that holds charge "
                                                "between pulses"},
         "provenance": ["lattice:2:2.1"], "spans": {"lattice:2:2.1": [18, 73]}},
        {"id": "concept:cinder-vault", "kind": "structure", "label": "cinder vault",
         "mentions": 3, "attrs": {}, "provenance": ["lattice:2:2.1"]},
        {"id": "concept:thrum-coil", "kind": "concept", "label": "thrum coil",
         "mentions": 1, "attrs": {}},
    ],
    "edges": [{"source": "concept:glimmer-node", "rel": "part_of",
               "target": "concept:cinder-vault", "weight": 2,
               "provenance": ["lattice:2:2.1"]}],
    "units": {"lattice:2:2.1": {"source": "lattice-studies", "title": "Glimmer Nodes",
                                "chapter": "2", "section": "2.1", "pages": [40, 42]}},
    "texts": {"lattice:2:2.1": PASSAGE},
}


def _named(pairs):
    return {schema["function"]["name"]: fn for schema, fn in pairs}


# -- citing -----------------------------------------------------------------------------


def test_look_at_says_where_an_entry_was_read_only_when_asked_to():
    said = look_at(GRAPH, ["concept:glimmer-node"], cite=True)
    assert "lattice-studies 2.1 Glimmer Nodes, pp. 40-42" in said
    assert "read at" not in look_at(GRAPH, ["concept:glimmer-node"])


def test_look_around_says_it_too():
    said = look_around(GRAPH, ["concept:glimmer-node"], cite=True)
    assert "lattice-studies 2.1 Glimmer Nodes, pp. 40-42" in said
    assert "read at" not in look_around(GRAPH, ["concept:glimmer-node"])


def test_the_citation_is_on_the_line_that_names_the_entry():
    """A cut takes the tail of an entry, never the source it was read at."""
    line = look_at(GRAPH, ["concept:glimmer-node"], cite=True).splitlines()[0]
    assert line.startswith("- glimmer node") and "read at lattice-studies" in line


def test_an_entry_that_points_at_nothing_is_cited_as_nothing():
    said = look_at(GRAPH, ["concept:thrum-coil"], cite=True)
    assert "thrum coil" in said and "read at" not in said


def test_a_tool_result_is_still_trimmed_to_the_budget_with_citations_on():
    whole = look_at(GRAPH, ["concept:glimmer-node", "concept:cinder-vault"], cite=True)
    cut = look_at(GRAPH, ["concept:glimmer-node", "concept:cinder-vault"], cite=True,
                  budget=20)
    assert len(cut) < len(whole)
    # the entry that survived kept its citation rather than being read out bare
    assert "read at lattice-studies" in cut


def test_quote_gives_the_source_words_the_span_points_at():
    rows = quotes(GRAPH, ["concept:glimmer-node"])
    assert rows[0]["quote"] == "a point of the lattice that holds charge between pulses"
    assert rows[0]["verbatim"] is True
    assert rows[0]["read_at"] == "lattice-studies 2.1 Glimmer Nodes, pp. 40-42"


def test_quote_falls_back_to_what_the_entry_holds_and_says_it_is_not_the_source():
    rows = quotes({**GRAPH, "texts": {}}, ["concept:glimmer-node"])
    assert rows[0]["quote"] == "a point of the lattice that holds charge between pulses"
    assert rows[0]["verbatim"] is False
    assert quotes(GRAPH, ["concept:cinder-vault"])[0]["quote"] == ""


def test_a_definition_the_fold_could_not_find_in_the_source_is_not_quotable():
    """The one thing worse than no quote is an invented one in quotation marks."""
    made_up = {"id": "concept:sablon", "kind": "substance", "label": "sablon", "mentions": 1,
               "attrs": {"definition": "the grey mineral nobody mentions", "unsourced": True},
               "provenance": ["lattice:2:2.1"]}
    graph = {**GRAPH, "nodes": [*GRAPH["nodes"], made_up]}
    row = quotes(graph, ["concept:sablon"])[0]
    assert row["quote"] == "" and row["verbatim"] is False


def test_a_text_the_graph_looks_up_rather_than_holds_is_read_the_same_way():
    graph = {**GRAPH, "texts": lambda unit_id: PASSAGE if unit_id == "lattice:2:2.1" else ""}
    assert quotes(graph, ["concept:glimmer-node"])[0]["quote"].startswith("a point of the")


def test_quote_is_offered_only_when_citing_is_on():
    assert "quote" not in _named(tools_for(GRAPH))
    offered = _named(tools_for(GRAPH, cite=True))
    assert "quote" in offered
    assert offered["quote"]({"ids": ["concept:glimmer-node"]})[0]["label"] == "glimmer node"


def test_the_read_tools_cite_when_the_pairs_were_built_to():
    plain, citing = _named(tools_for(GRAPH)), _named(tools_for(GRAPH, cite=True))
    args = {"ids": ["concept:glimmer-node"]}
    assert "read at" not in plain["look_at"](args)
    assert "read at lattice-studies" in citing["look_at"](args)
    assert "read at lattice-studies" in citing["look_around"](args)


def test_the_asking_carries_citing_into_the_tools_it_asks_for():
    assert "quote" in _named(tools_for(GRAPH, **Asking(cite=True).tools()))
    assert Asking(cite=True).said()["cite"] is True


# -- grounding --------------------------------------------------------------------------


def test_an_entry_the_answer_is_about_with_no_source_is_named():
    answer = Answer(show=["concept:glimmer-node", "concept:thrum-coil"],
                    read=["concept:glimmer-node"])
    rows = grounded(answer, GRAPH)
    assert [r["id"] for r in rows] == ["concept:glimmer-node", "concept:thrum-coil"]
    assert rows[0]["grounded"] and rows[0]["read"] and rows[0]["known"]
    assert rows[0]["units"] == ["lattice:2:2.1"]
    assert rows[0]["spans"] == {"lattice:2:2.1": [18, 73]}
    assert not rows[1]["grounded"] and rows[1]["known"]
    assert ungrounded(rows) == "thrum coil (no source)"


def test_an_entry_the_answer_is_about_that_it_never_read_says_so():
    rows = grounded(Answer(show=["concept:cinder-vault"], read=["concept:glimmer-node"]),
                    GRAPH)
    assert rows[0]["grounded"] and not rows[0]["read"]


def test_an_id_the_graph_does_not_have_is_neither_known_nor_grounded():
    rows = grounded(Answer(show=["concept:sablon"]), GRAPH)
    assert rows == [{"id": "concept:sablon", "label": "", "known": False,
                     "grounded": False, "read": False, "units": [], "spans": {}}]
    assert ungrounded(rows) == "concept:sablon (not in the graph)"


def test_an_answer_that_selected_nothing_grounds_nothing():
    assert grounded(Answer(), GRAPH) == []
    assert ungrounded([]) == ""


def test_a_store_is_read_the_same_way_as_a_graph(tmp_path):
    import pytest

    pytest.importorskip("ladybug")
    from ml_stack.graph.store import GraphStore

    with GraphStore(tmp_path / "sources.ladybug") as store:
        store.write({"nodes": GRAPH["nodes"], "edges": GRAPH["edges"]})
        rows = grounded(Answer(show=["concept:glimmer-node", "concept:thrum-coil"],
                               read=["concept:glimmer-node"]), store)
    assert [r["grounded"] for r in rows] == [True, False]
    assert rows[0]["spans"] == {"lattice:2:2.1": [18, 73]}
