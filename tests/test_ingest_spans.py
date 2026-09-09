"""Character spans: an extracted definition found back in the words it was read from.

A citation needs the sentence, not the section, so `ingest.spans` locates the source's own
words and `build` records the offsets beside the unit id a node already points at.
"""

import json

import pytest

from ml_stack import ingest
from ml_stack.ingest.spans import locate, sentence_span, spans_for
from tests.test_ingest import a_unit

PASSAGE = (
    "2.1 Glimmer Nodes\n\n"
    "A glimmer node is a point of the lattice that holds charge between pulses. "
    "Every glimmer node sits inside a cinder vault, which is sealed with spun sablon. "
    "The vault current runs the length of a vault and never leaves it.\n\n"
    "Thrum coils are wound around a sablon rod."
)
DEFINITION = "a point of the lattice that holds charge between pulses"


def _unit(text=PASSAGE, **over):
    return a_unit(text=text, **over)


def _extraction(concepts=(), relations=(), key_terms=(), figures=()):
    return {"concepts": [{"name": n, "kind": "concept", "definition": d, "aliases": []}
                         for n, d in concepts],
            "relations": [{"from": a, "rel": r, "to": b} for a, r, b in relations],
            "key_terms": [{"term": t, "definition": d} for t, d in key_terms],
            "figures": list(figures)}


# -- locate ---------------------------------------------------------------------------


def test_a_definition_quoted_verbatim_is_found_where_it_stands():
    span = locate(PASSAGE, DEFINITION)
    assert span is not None
    assert PASSAGE[span[0]:span[1]] == DEFINITION


def test_a_definition_respaced_and_repunctuated_is_still_found():
    """A model rewrites the whitespace and the dashes; the words are the source's."""
    said = "a  point of the\nlattice — that holds charge  between pulses"
    span = locate(PASSAGE, said)
    assert span is not None and PASSAGE[span[0]:span[1]] == DEFINITION


def test_a_definition_the_source_never_gave_is_not_found():
    assert locate(PASSAGE, "a bead of spare sablon left on a plate after grinding") is None


def test_the_floor_says_how_close_is_close_enough():
    said = "a point of the lattice that holds warmth between winters"
    assert locate(PASSAGE, said) is None
    assert locate(PASSAGE, said, least=0.5) is not None


def test_nothing_to_look_in_or_for_is_nothing_found():
    assert locate("", DEFINITION) is None and locate(PASSAGE, "") is None


def test_a_sentence_is_found_by_the_names_it_carries():
    span = sentence_span(PASSAGE, "glimmer node", "cinder vault")
    assert span is not None
    assert PASSAGE[span[0]:span[1]].startswith("Every glimmer node sits inside a cinder vault")
    assert sentence_span(PASSAGE, "glimmer node", "thrum coil") is None


# -- spans_for ------------------------------------------------------------------------


def test_every_defined_concept_gets_a_span_and_an_invented_one_gets_none():
    found = spans_for(_extraction(
        concepts=[("glimmer node", DEFINITION),
                  ("cinder vault", "a sealed chamber nobody has ever opened")],
        key_terms=[("vault current", "runs the length of a vault")]), _unit())
    assert PASSAGE[slice(*found["glimmer node"])] == DEFINITION
    assert found["cinder vault"] is None, "the failure is visible, not swallowed"
    assert found["vault current"] is not None
    assert sum(1 for span in found.values() if span is None) == 1


def test_a_unit_with_no_text_judges_nothing():
    """No text is not the same as a definition that was not there."""
    assert spans_for(_extraction(concepts=[("glimmer node", DEFINITION)]),
                     _unit(text="")) == {}


# -- through the fold -----------------------------------------------------------------


def test_a_node_carries_the_span_beside_the_unit_it_points_at():
    unit = _unit()
    nodes, _edges = ingest.build(
        _extraction(concepts=[("glimmer node", DEFINITION)]), unit)
    node = nodes["concept:glimmer-node"]
    assert node["provenance"] == [unit.id]
    assert PASSAGE[slice(*node["spans"][unit.id])] == DEFINITION
    assert "unsourced" not in node["attrs"]


def test_an_edge_carries_the_sentence_that_states_it():
    unit = _unit()
    _nodes, edges = ingest.build(
        _extraction(concepts=[("glimmer node", DEFINITION)],
                    relations=[("glimmer node", "part_of", "cinder vault")]), unit)
    edge = edges[("concept:glimmer-node", "part_of", "concept:cinder-vault")]
    said = PASSAGE[slice(*edge["spans"][unit.id])]
    assert "glimmer node" in said and "cinder vault" in said


def test_a_figure_carries_the_span_of_its_caption():
    text = PASSAGE + "\n\nFIGURE 2.9 A cinder vault cut open along its long axis."
    unit = _unit(text=text)
    nodes, _edges = ingest.build(
        {"concepts": [], "relations": [], "key_terms": [],
         "figures": [{"label": "FIGURE 2.9",
                      "caption": "A cinder vault cut open along its long axis.",
                      "shows": "a vault", "concepts": []}]}, unit)
    node = nodes[f"figure:{unit.id}:1"]
    assert text[slice(*node["spans"][unit.id])].startswith("A cinder vault cut open")


def test_an_invented_definition_is_marked_and_counted():
    nodes, _edges = ingest.build(
        _extraction(concepts=[("glimmer node", DEFINITION),
                              ("cinder vault", "a sealed chamber nobody has ever opened")]),
        _unit())
    assert nodes["concept:cinder-vault"]["attrs"]["unsourced"] is True
    assert ingest.unsourced(nodes.values()) == ["cinder vault"]


def test_a_unit_with_no_text_records_no_spans_and_marks_nothing():
    nodes, _edges = ingest.build(
        _extraction(concepts=[("glimmer node", "a thing the fold cannot check")]),
        _unit(text=""))
    node = nodes["concept:glimmer-node"]
    assert "spans" not in node and "unsourced" not in node["attrs"]


def test_two_units_each_add_their_own_span():
    first, second = _unit(section="2.1"), _unit(section="2.2")
    reads = [{"unit": u.id, "source": u.source, "chapter": u.chapter, "section": u.section,
              "title": u.section_title, "pages": [u.first_page, u.last_page],
              "extracted": _extraction(concepts=[("glimmer node", DEFINITION)])}
             for u in (first, second)]
    graph = ingest.fold_source(reads, {first.id: first, second.id: second})
    node = next(n for n in graph["nodes"] if n["id"] == "concept:glimmer-node")
    assert sorted(node["spans"]) == sorted([first.id, second.id])


# -- into the store -------------------------------------------------------------------


def _keep(tmp_path, slug, reads):
    (tmp_path / f"sources.{slug}.reads.json").write_text(
        json.dumps({r["unit"]: r for r in reads}))


def test_the_store_reads_the_spans_back_and_the_pointers_resolve(tmp_path):
    pytest.importorskip("ladybug")
    from ml_stack.graph.store import GraphStore

    unit = _unit()
    _keep(tmp_path, "lattice", [{
        "unit": unit.id, "source": unit.source, "chapter": unit.chapter,
        "section": unit.section, "title": unit.section_title, "run": "run:one",
        "pages": [unit.first_page, unit.last_page],
        "extracted": _extraction(concepts=[("glimmer node", DEFINITION)],
                                 relations=[("glimmer node", "part_of", "cinder vault")])}])
    out = tmp_path / "sources"
    ingest.Progress(ingest.Progress.beside(out)).source("lattice", title="Lattice Studies",
                                                        sections=1)
    ingest.fold_into(out, "lattice", units_by_id={unit.id: unit})

    with GraphStore(out, read_only=True) as store:
        node = next(n for n in store.nodes() if n["id"] == "concept:glimmer-node")
        assert node["spans"][unit.id] == list(locate(PASSAGE, DEFINITION))
        rows = ingest.located(store, node)
        assert rows[0]["unit"] == unit.id and rows[0]["pages"] == [2, 3]
        assert rows[0]["span"] == node["spans"][unit.id]
        said = ingest.quote(store, node, texts={unit.id: PASSAGE})
        assert said[0]["quote"] == DEFINITION
        assert said[0]["section"] == unit.section


def test_a_node_with_no_span_quotes_what_it_was_given(tmp_path):
    pytest.importorskip("ladybug")

    node = {"id": "concept:sablon", "label": "sablon", "provenance": ["lattice:2:2.4"],
            "attrs": {"definition": "the grey mineral the plates are ground from"}}

    class _NoDocs:
        path = tmp_path / "sources"

        def get_doc(self, key, default=None):
            return default

    said = ingest.quote(_NoDocs(), node)
    assert said[0]["quote"] == "the grey mineral the plates are ground from"
    assert said[0]["span"] is None


def test_a_book_read_folded_and_asked_carries_its_own_words_all_the_way(tmp_path):
    """The whole path, driven the way a run drives it: a document, a fold that re-reads it
    for the text, a store, and a model that reads a cited entry and quotes the passage."""
    pytest.importorskip("ladybug")
    pytest.importorskip("pymupdf")
    from ml_stack.graph.ask import tools_for
    from ml_stack.ingest.ask import graph_of
    from ml_stack.sources import pdf
    from tests.test_sources_pdf import a_textbook

    book = a_textbook(tmp_path / "lattice.pdf")
    document = pdf.read(book)
    unit = next(u for u in pdf.units(document) if u.section == "2.1")
    said = "run between quickened nodes"
    assert said in unit.text, "the fixture stopped saying what the extraction claims"

    out = tmp_path / "sources"
    kept = ingest.Progress(ingest.Progress.beside(out))
    kept.source(document.slug, title=document.title, path=book, sections=1)
    kept.save()
    _keep(tmp_path, document.slug, [{
        "unit": unit.id, "source": unit.source, "chapter": unit.chapter,
        "section": unit.section, "title": unit.section_title, "run": "run:one",
        "pages": [unit.first_page, unit.last_page],
        "extracted": _extraction(concepts=[("vault current", said)])}])

    # the fold gets no units in memory, so the spans come from reading the document again
    ingest.fold_into(out, document.slug)
    graph = graph_of(out, cite=True)
    node = next(n for n in graph["nodes"] if n["id"] == "concept:vault-current")
    assert node["spans"][unit.id], "no span survived the fold and the store"

    tools = {s["function"]["name"]: fn for s, fn in tools_for(graph, cite=True)}
    read = tools["look_at"]({"ids": ["concept:vault-current"]})
    assert f"read at {document.slug} 2.1 Currents in Practice" in read
    assert tools["quote"]({"ids": ["concept:vault-current"]})[0]["quote"] == said


def test_a_model_answering_with_citing_on_is_told_to_cite_and_can_quote():
    from ml_stack.client import Reply
    from ml_stack.graph.ask import CITE_SYSTEM_SENTENCE, converse
    from ml_stack.graph.asking import Asking

    graph = {"nodes": [{"id": "concept:vault-current", "kind": "concept",
                        "label": "vault current", "mentions": 3,
                        "attrs": {"definition": "run between quickened nodes"},
                        "provenance": ["lattice-studies:2:2.1"],
                        "spans": {"lattice-studies:2:2.1": [0, 26]}}],
             "edges": [],
             "units": {"lattice-studies:2:2.1": {"source": "lattice-studies",
                                                 "title": "Currents in Practice",
                                                 "chapter": "2", "section": "2.1",
                                                 "pages": [4, 4]}},
             "texts": {"lattice-studies:2:2.1": "run between quickened nodes and vaults"}}

    def _call(name, args):
        return {"id": name, "function": {"name": name, "arguments": json.dumps(args)}}

    class _Model:
        model = "quill:8b"

        def __init__(self):
            self.seen, self.turn = [], 0

        def chat(self, messages, tools=None, **_kw):
            self.seen.append(messages)
            self.turn += 1
            if self.turn == 1:
                return Reply(content="", tool_calls=[
                    _call("look_at", {"ids": ["concept:vault-current"]})])
            if self.turn == 2:
                return Reply(content="", tool_calls=[
                    _call("quote", {"ids": ["concept:vault-current"]})])
            if self.turn == 3:
                return Reply(content="", tool_calls=[
                    _call("show", {"ids": ["concept:vault-current"]})])
            return Reply(content="Vault currents run between quickened nodes "
                                 "(lattice 2.1, p. 4).")

    model = _Model()
    answer = converse("what is a vault current?", graph, model, asking=Asking(cite=True))
    system = model.seen[0][0]["content"]
    assert CITE_SYSTEM_SENTENCE in system
    said = [m["content"] for turn in model.seen for m in turn if m.get("role") == "tool"]
    assert any("read at lattice-studies 2.1" in one for one in said)
    assert any("run between quickened nodes" in one for one in said)
    assert "quoted 1 entry" in answer.steps
