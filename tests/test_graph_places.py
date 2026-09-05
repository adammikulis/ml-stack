"""Places on a graph: a name given a point, and `near` edges between the closest.

Nothing here reaches Nominatim: every test hands `geocode` its own `lookup`.
"""

from __future__ import annotations

import json

import pytest

from ml_stack.geo import CACHE_VERSION
from ml_stack.graph.places import geocode, kilometres, places_in, points

# invented coordinates for real places, so no test depends on a gazetteer answering
WHERE = {"Turin": (45.07, 7.69), "Lyon": (45.76, 4.84), "Oslo": (59.91, 10.75),
         "Kyoto": (35.01, 135.77)}


def a_lookup(asked: list[str] | None = None):
    def lookup(place, **_kwargs):
        if asked is not None:
            asked.append(place)
        found = WHERE.get(place)
        return None if found is None else {"lat": found[0], "lon": found[1],
                                           "display": place, "type": "city"}
    return lookup


def a_graph():
    return {
        "nodes": [
            {"id": "person:ada", "kind": "person", "label": "Ada of Turin",
             "attrs": {"place": "Turin"}},
            {"id": "person:bea", "kind": "person", "label": "Bea of Turin",
             "attrs": {"place": "Lyon"}},
            {"id": "person:cyd", "kind": "person", "label": "Cyd Marek",
             "attrs": {"place": "Kyoto"}},
            {"id": "person:dor", "kind": "person", "label": "Dorian Vale", "attrs": {}},
            {"id": "place:oslo", "kind": "place", "label": "Oslo", "attrs": {}},
            {"id": "topic:iron", "kind": "topic", "label": "iron", "attrs": {}},
        ],
        "edges": [{"source": "person:ada", "target": "topic:iron", "rel": "works_on"}],
    }


def test_a_place_is_read_from_an_attribute_or_from_a_place_nodes_own_label():
    assert places_in(a_graph()) == {"person:ada": "Turin", "person:bea": "Lyon",
                                   "person:cyd": "Kyoto", "place:oslo": "Oslo"}


def test_every_entry_that_named_a_place_comes_back_with_a_point(tmp_path):
    placed = geocode(a_graph(), tmp_path / "cache.json", lookup=a_lookup(), log=lambda _: None)
    attrs = {n["id"]: n.get("attrs") or {} for n in placed["nodes"]}
    assert (attrs["person:ada"]["lat"], attrs["person:ada"]["lon"]) == WHERE["Turin"]
    assert (attrs["place:oslo"]["lat"], attrs["place:oslo"]["lon"]) == WHERE["Oslo"]
    assert "lat" not in attrs["person:dor"], "nobody who said nothing is placed"
    assert "lat" not in attrs["topic:iron"]


def test_the_points_the_map_draws_are_one_per_placed_entry(tmp_path):
    placed = geocode(a_graph(), tmp_path / "cache.json", lookup=a_lookup(), log=lambda _: None)
    drawn = points(placed)
    assert [p["id"] for p in drawn] == ["person:ada", "person:bea", "person:cyd", "place:oslo"]
    assert drawn[0] == {"id": "person:ada", "label": "Ada of Turin", "place": "Turin",
                        "lat": 45.07, "lon": 7.69}


def test_a_hidden_node_is_never_on_the_map(tmp_path):
    graph = a_graph()
    graph["nodes"].append({"id": "run:1", "kind": "run", "label": "a run",
                           "attrs": {"hidden": True, "lat": 0.0, "lon": 0.0}})
    assert "run:1" not in [p["id"] for p in points(graph)]


def test_a_place_is_asked_about_once_and_the_cache_answers_the_next_run(tmp_path):
    asked: list[str] = []
    cache = tmp_path / "cache.json"
    graph = a_graph()
    graph["nodes"].append({"id": "person:eve", "kind": "person", "label": "Vera Lund",
                           "attrs": {"place": "Turin"}})
    geocode(graph, cache, lookup=a_lookup(asked), log=lambda _: None)
    assert sorted(asked) == ["Kyoto", "Lyon", "Oslo", "Turin"]

    geocode(graph, cache, lookup=a_lookup(asked), log=lambda _: None)
    assert sorted(asked) == ["Kyoto", "Lyon", "Oslo", "Turin"], "the cache answered the second"
    assert json.loads(cache.read_text(encoding="utf-8"))["_v"] == CACHE_VERSION


def test_a_place_nothing_is_found_for_leaves_the_node_alone(tmp_path):
    graph = a_graph()
    graph["nodes"][0]["attrs"]["place"] = "Nowhereshire"
    placed = geocode(graph, tmp_path / "cache.json", lookup=a_lookup(), log=lambda _: None)
    assert "lat" not in (placed["nodes"][0].get("attrs") or {})


class TestNear:
    def _near(self, placed):
        return {(e["source"], e["target"]): e["weight"]
                for e in placed["edges"] if e["rel"] == "near"}

    def test_each_placed_entry_is_joined_to_its_closest(self, tmp_path):
        placed = geocode(a_graph(), tmp_path / "cache.json", near=1,
                         lookup=a_lookup(), log=lambda _: None)
        joined = self._near(placed)
        assert {frozenset(pair) for pair in joined} == {
            frozenset({"person:ada", "person:bea"}),      # Turin and Lyon
            frozenset({"place:oslo", "person:bea"}),      # Oslo's nearest of the four
            frozenset({"person:cyd", "place:oslo"}),      # Kyoto is nearest to Oslo
        }

    def test_the_weight_falls_off_with_the_kilometres(self, tmp_path):
        placed = geocode(a_graph(), tmp_path / "cache.json", near=1,
                         lookup=a_lookup(), log=lambda _: None)
        joined = self._near(placed)
        close = next(w for pair, w in joined.items()
                     if set(pair) == {"person:ada", "person:bea"})
        far = next(w for pair, w in joined.items()
                   if set(pair) == {"person:cyd", "place:oslo"})
        assert close > far
        assert close == pytest.approx(
            1.0 / (1.0 + kilometres(WHERE["Turin"], WHERE["Lyon"])), rel=1e-4)

    def test_the_edges_the_graph_had_are_kept_and_a_second_run_adds_no_more(self, tmp_path):
        once = geocode(a_graph(), tmp_path / "cache.json", near=1,
                       lookup=a_lookup(), log=lambda _: None)
        twice = geocode(once, tmp_path / "cache.json", near=1,
                        lookup=a_lookup(), log=lambda _: None)
        assert len(twice["edges"]) == len(once["edges"])
        assert {"source": "person:ada", "target": "topic:iron", "rel": "works_on"} \
            in twice["edges"]

    def test_without_near_no_edges_are_added_or_taken_away(self, tmp_path):
        placed = geocode(a_graph(), tmp_path / "cache.json",
                         lookup=a_lookup(), log=lambda _: None)
        assert placed["edges"] == a_graph()["edges"]


def test_kilometres_is_the_distance_over_the_ground():
    assert kilometres(WHERE["Turin"], WHERE["Turin"]) == 0.0
    assert kilometres(WHERE["Turin"], WHERE["Lyon"]) == pytest.approx(233, abs=8)
    assert kilometres((0.0, 0.0), (0.0, 180.0)) == pytest.approx(20015, abs=5)


class TestCommand:
    def test_geocode_writes_the_points_and_the_near_edges_back(self, tmp_path, monkeypatch,
                                                               capsys):
        from ml_stack.graph import places, serve

        monkeypatch.setattr(places.geo, "lookup", a_lookup())
        source = tmp_path / "graph.json"
        source.write_text(json.dumps(a_graph()), encoding="utf-8")
        out = tmp_path / "placed.json"
        code = serve.main(["geocode", "--graph", str(source), "--cache",
                           str(tmp_path / "cache.json"), "--near", "1", "--out", str(out)])
        assert code == 0
        written = json.loads(out.read_text(encoding="utf-8"))
        assert len(points(written)) == 4
        assert [e for e in written["edges"] if e["rel"] == "near"]
        assert "4 entr(ies) placed" in capsys.readouterr().out
        assert json.loads(source.read_text(encoding="utf-8")) == a_graph()

    def test_without_an_out_the_graph_is_written_over(self, tmp_path, monkeypatch):
        from ml_stack.graph import places, serve

        monkeypatch.setattr(places.geo, "lookup", a_lookup())
        source = tmp_path / "graph.json"
        source.write_text(json.dumps(a_graph()), encoding="utf-8")
        assert serve.main(["geocode", "--graph", str(source),
                           "--cache", str(tmp_path / "cache.json")]) == 0
        assert len(points(json.loads(source.read_text(encoding="utf-8")))) == 4
