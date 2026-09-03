"""``page.render`` as a string: what a kind is painted, and how many messages ride along.

No browser here; these read the html back. The graph is invented (tests/known-fixtures.txt).
"""

from __future__ import annotations

import json
import re

from ml_stack.graph import page as graph_page

SHIPPED = ("person", "org", "place", "topic", "opportunity")


def graph_with(*kinds: str) -> dict:
    nodes = [{"id": f"{kind}:{i}", "label": f"{kind} {i}", "kind": kind, "attrs": {},
              "messages": []} for i, kind in enumerate(kinds)]
    return {"nodes": nodes, "edges": [], "messages": {}}


def payload_of(html: str) -> dict:
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert match, "the page carries no DATA payload"
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_every_kind_gets_a_colour_and_a_given_one_is_kept_verbatim():
    html = graph_page.render(
        graph_with("person", "group", "event"),
        kinds=[{"k": "person", "label": "People", "shape": "circle"},
               {"k": "group", "label": "Groups", "shape": "square", "colour": "#7a5af5"},
               {"k": "event", "label": "Events", "shape": "diamond"}])
    assert "--k-group:#7a5af5" in html
    assert re.search(r"--k-event:#[0-9a-fA-F]{6}", html)
    kinds = {k["k"]: k for k in payload_of(html)["kinds"]}
    assert kinds["group"]["colour"] == "#7a5af5"
    assert re.fullmatch(r"#[0-9a-fA-F]{6}", kinds["event"]["colour"])
    assert kinds["person"]["colour"]


def test_a_colour_chosen_for_a_kind_is_none_of_the_five_shipped_ones():
    template = (graph_page.WEB / "graph.html").read_text()
    shipped = set(re.findall(r"--k-(?:person|org|place|topic|opportunity): (#[0-9a-f]{6})",
                             template))
    html = graph_page.render(graph_with("group", "event", "product", "department"))
    kinds = payload_of(html)["kinds"]
    chosen = [k["colour"].lower() for k in kinds]
    assert len(set(chosen)) == len(chosen), "two kinds share a colour"
    assert not set(chosen) & shipped


def test_a_colour_one_kind_was_given_is_not_chosen_for_another():
    html = graph_page.render(
        graph_with("group", "event"),
        kinds=[{"k": "group", "label": "Groups", "shape": "square",
                "colour": graph_page.PALETTE[0]},
               {"k": "event", "label": "Events", "shape": "diamond"}])
    kinds = {k["k"]: k["colour"] for k in payload_of(html)["kinds"]}
    assert kinds["event"] != kinds["group"]


def test_the_colour_rule_comes_after_the_templates_three_palette_blocks():
    html = graph_page.render(graph_with("group"))
    last_palette = html.rfind("--k-opportunity: #")
    assert html.find("--k-group:") > last_palette


def test_shipped_kinds_without_a_colour_keep_the_templates_dark_and_light_pair():
    html = graph_page.render(graph_with("person", "group"))
    rule = html[html.find("--k-group:") - 200:html.find("--k-group:")]
    assert "--k-person" not in rule


def messages(n: int) -> dict:
    return {f"m{i}": {"text": f"message {i}", "ts": str(1700000000 + i * 3600),
                      "sender": "Ada Lovelace"} for i in range(n)}


def test_most_messages_keeps_the_newest_and_says_how_many_were_left_out():
    graph = {"nodes": [{"id": "person:ada", "label": "Ada Lovelace", "kind": "person",
                        "attrs": {}, "messages": ["m0", "m1", "m2", "m3", "m4"]}],
             "edges": [], "messages": messages(5)}
    got = payload_of(graph_page.render(graph, most_messages=2))
    assert set(got["graph"]["messages"]) == {"m3", "m4"}
    assert got["messagesLeftOut"] == 3
    assert got["graph"]["nodes"][0]["messages"] == ["m3", "m4"]


def test_without_a_limit_every_message_stays_and_nothing_was_left_out():
    graph = {"nodes": [], "edges": [], "messages": messages(5)}
    got = payload_of(graph_page.render(graph))
    assert len(got["graph"]["messages"]) == 5
    assert got["messagesLeftOut"] == 0
