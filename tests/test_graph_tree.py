"""A hierarchy read out of prose or a picture.

An org chart, a family tree and a parts breakdown are one object with three shapes, so they
are tested as one. Every name is invented.
"""

from __future__ import annotations

import pytest

from ml_stack.graph.tree import (FAMILY, ORG, PARTS, PictureUnreadable, Shape, cycles, read,
                                 roots, schema_for, to_graph, transcribe)

# A real 1x1 PNG, written out here rather than fetched or faked. A test that hands in
# `b"pretend-png"` fails three layers down with `any(...) is False`, because an unreadable
# image is dropped rather than refused -- which is how a test failure becomes an afternoon.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d49444154789c6360000002000100ffff03000006"
    "00057b7d5a0000000049454e44ae426082")


@pytest.fixture(scope="module", autouse=True)
def _needs_pillow():
    """Say why, rather than failing oddly, when the image library is not installed."""
    pytest.importorskip("PIL", reason="image tests need Pillow: pip install -e '.[vision]'")

CHART = [
    {"name": "Wren Halloway", "title": "Head of Works", "team": "", "above": []},
    {"name": "Hollis Fen", "title": "Shift Lead", "team": "Assembly", "above": ["Wren Halloway"]},
    {"name": "Pell Grantham", "title": "Fitter", "team": "Assembly", "above": ["Hollis Fen"]},
]


def test_an_org_chart_becomes_entries_and_links():
    g = to_graph(CHART, ORG)
    by = {n["id"]: n for n in g["nodes"]}
    assert set(by) == {"person:wrenhalloway", "person:hollisfen", "person:pellgrantham"}
    assert by["person:hollisfen"]["attrs"] == {"title": "Shift Lead", "team": "Assembly"}
    assert [(e["source"], e["rel"], e["target"]) for e in g["edges"]] == [
        ("person:hollisfen", "reports_to", "person:wrenhalloway"),
        ("person:pellgrantham", "reports_to", "person:hollisfen")]
    assert roots(g, ORG) == ["person:wrenhalloway"]
    assert cycles(g, ORG) == []


def test_a_family_tree_is_the_same_object_with_two_parents():
    """The shape is the only difference: what an entry is, what the link means, how many."""
    rows = [{"name": "Elowen Trewin", "born": "1901", "above": ["Osric Trewin", "Verity Trewin"]},
            {"name": "Osric Trewin", "born": "1870", "above": []}]
    g = to_graph(rows, FAMILY)
    links = {(e["source"], e["target"]) for e in g["edges"]}
    assert links == {("person:elowentrewin", "person:osrictrewin"),
                     ("person:elowentrewin", "person:veritytrewin")}
    assert all(e["rel"] == "child_of" for e in g["edges"])
    # an org chart keeps one, because a second manager there is a misreading
    assert len(to_graph([rows[0]], ORG)["edges"]) == 1


def test_somebody_named_only_as_a_parent_still_becomes_an_entry():
    """A chart that says "reports to the board" and never draws the board is about the board."""
    g = to_graph([{"name": "Hollis Fen", "above": ["The Board"]}], ORG)
    assert {n["id"] for n in g["nodes"]} == {"person:hollisfen", "person:theboard"}
    assert roots(g, ORG) == ["person:theboard"]


def test_a_name_written_two_ways_is_one_entry():
    g = to_graph([{"name": "Jo  Ash", "above": []},
                  {"name": "jo ash", "title": "Lead", "above": []},
                  {"name": "Pell Grantham", "above": ["JO ASH"]}], ORG)
    assert len([n for n in g["nodes"] if n["id"] == "person:joash"]) == 1
    assert [n for n in g["nodes"] if n["id"] == "person:joash"][0]["attrs"]["title"] == "Lead"
    assert g["edges"][0]["target"] == "person:joash"


def test_somebody_who_manages_themselves_is_not_linked_to_themselves():
    g = to_graph([{"name": "Wren Halloway", "above": ["Wren Halloway"]}], ORG)
    assert g["edges"] == []


def test_a_loop_is_found_rather_than_walked_for_ever():
    """A model reading a crowded chart will draw one, and "who is above me" then runs until
    it runs out of graph."""
    g = to_graph([{"name": "A One", "above": ["B Two"]},
                  {"name": "B Two", "above": ["C Three"]},
                  {"name": "C Three", "above": ["A One"]}], ORG)
    found = cycles(g, ORG)
    assert len(found) == 1
    assert set(found[0]) == {"person:aone", "person:btwo", "person:cthree"}
    assert roots(g, ORG) == []          # a loop has no top, which is the point


def test_several_tops_is_ordinary(_=None):
    g = to_graph([{"name": "A One", "above": []}, {"name": "B Two", "above": []}], ORG)
    assert roots(g, ORG) == ["person:aone", "person:btwo"]


def test_the_schema_asks_for_what_the_shape_wants():
    org = schema_for(ORG)["properties"]["entries"]["items"]["properties"]
    assert set(org) == {"name", "title", "team", "above"}
    fam = schema_for(FAMILY)["properties"]["entries"]["items"]["properties"]
    assert "born" in fam and "title" not in fam
    assert set(schema_for(PARTS)["properties"]["entries"]["items"]["properties"]) \
        == {"name", "quantity", "code", "above"}


class Chat:
    def __init__(self, entries):
        self.entries, self.seen = entries, {}

    def extract(self, text, schema, **kw):
        self.seen = {"text": text, **kw}
        return {"entries": self.entries}


def test_reading_prose_asks_the_model_for_the_shape():
    model = Chat([{"name": "Wren Halloway", "above": []}])
    rows = read(model, ORG, text="Wren runs the works.")
    assert rows == [{"name": "Wren Halloway", "above": []}]
    assert "org chart" in model.seen["instructions"]
    assert model.seen["messages"] is None


def test_an_entry_with_no_name_is_dropped():
    model = Chat([{"name": "", "above": []}, {"name": "  ", "above": []},
                  {"name": "Hollis Fen", "above": []}, "not a row"])
    assert [r["name"] for r in read(model, ORG, text="x")] == ["Hollis Fen"]


def test_reading_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing to read"):
        read(Chat([]), ORG)


class Reader:
    def __init__(self, said):
        self.said, self.asked = said, None

    def chat(self, messages, **kw):
        self.asked = messages
        return type("R", (), {"content": self.said})()


def test_a_document_model_transcribes_and_a_chat_model_structures():
    """The best model at reading a crowded chart is rarely the best at deciding what it
    means, so either can be a different one."""
    eyes = Reader("Wren Halloway\n  Hollis Fen")
    model = Chat([{"name": "Wren Halloway", "above": []},
                  {"name": "Hollis Fen", "above": ["Wren Halloway"]}])
    rows = read(model, ORG, images=[PNG], reader=eyes)

    assert [r["name"] for r in rows] == ["Wren Halloway", "Hollis Fen"]
    assert "Hollis Fen" in model.seen["text"], "the transcription must reach the structurer"
    assert model.seen["messages"] is None, "the picture must not be sent twice"
    assert eyes.asked[0]["role"] == "user"
    assert any(p["type"] == "image_url" for p in eyes.asked[0]["content"])


def test_without_a_reader_the_picture_goes_to_the_model_itself():
    model = Chat([{"name": "Wren Halloway", "above": []}])
    read(model, ORG, images=[PNG])
    assert model.seen["messages"] is not None
    assert any(p["type"] == "image_url" for p in model.seen["messages"][0]["content"])


def test_transcribing_asks_for_layout_not_a_summary():
    eyes = Reader("some text")
    assert transcribe(eyes, [PNG]) == "some text"
    asked = eyes.asked[0]["content"][0]["text"]
    assert "summarise" in asked and "indent" in asked


def test_a_shape_can_be_invented_for_anything():
    mine = Shape(name="build graph", kind="thing", relation="depends_on",
                 above="what it needs", entry="package", attributes=("version",))
    g = to_graph([{"name": "app", "version": "2.0", "above": ["libcore"]}], mine)
    assert [(e["source"], e["rel"], e["target"]) for e in g["edges"]] \
        == [("thing:app", "depends_on", "thing:libcore")]


def test_a_picture_that_cannot_be_read_says_so(tmp_path):
    """Silently dropping it means asking a model to read a chart with no chart attached --
    it answers about nothing, and the reason is in a warnings list nobody reads."""
    with pytest.raises(PictureUnreadable, match="unrecognised image format"):
        transcribe(Reader("x"), [b"this is not a png"])

    broken = tmp_path / "chart.png"
    broken.write_bytes(b"nor is this")
    with pytest.raises(PictureUnreadable) as raised:
        read(Chat([]), ORG, images=[broken])
    assert "chart.png" in str(raised.value) or "image 0" in str(raised.value)


def test_a_readable_picture_is_not_refused():
    """The guard must not fire on a real image, or it is just a new way to fail."""
    eyes = Reader("Wren Halloway")
    assert transcribe(eyes, [PNG]) == "Wren Halloway"
