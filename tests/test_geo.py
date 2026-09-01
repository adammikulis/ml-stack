"""Which of several same-named places was meant, and asking about each only once."""

import json
import urllib.parse

from conftest import json_reply

from ml_stack.geo import CACHE_VERSION, best, expand, geocode_all


def row(name, rank, kind="administrative", category="boundary", importance=0.5, short=None):
    return {"display_name": name, "name": short if short is not None else name.split(",")[0],
            "place_rank": rank, "type": kind, "category": category,
            "importance": importance, "lat": "0", "lon": "0"}


def test_a_region_beats_the_city_that_shares_its_name():
    """"Ontario" is where someone lives, not the city in California Nominatim ranks first."""
    rows = [row("Ontario, California", 16, importance=0.62), row("Ontario, Canada", 8, importance=0.78)]
    assert best(rows, "Ontario")["display_name"] == "Ontario, Canada"


def test_a_country_wins_outright():
    rows = [row("Canada, Kentucky", 18), row("Canada", 4)]
    assert best(rows, "Canada")["display_name"] == "Canada"


def test_a_plain_city_is_still_the_city():
    rows = [row("San Francisco, California", 16)]
    assert best(rows, "San Francisco")["display_name"] == "San Francisco, California"


def test_an_administrative_match_beats_a_landmark_that_ranks_higher():
    rows = [row("Colorado River", 10, kind="river", category="waterway", importance=0.9),
            row("Colorado, United States", 8)]
    assert best(rows, "Colorado")["display_name"] == "Colorado, United States"


def test_nothing_found_is_none():
    assert best([], "anywhere") is None


def test_the_place_called_what_was_asked_beats_a_wider_one_that_is_not():
    """Nominatim answers "Raleigh" with Raleigh County before the city of that name."""
    rows = [row("Raleigh County, West Virginia", 12, short="Raleigh County"),
            row("Raleigh, North Carolina", 16, short="Raleigh", importance=0.72)]
    assert best(rows, "Raleigh")["display_name"] == "Raleigh, North Carolina"


def test_the_best_known_of_two_places_with_one_name_wins():
    """A hamlet in Newfoundland is also called Raleigh; nobody writing it means that one."""
    rows = [row("Raleigh, Newfoundland, Canada", 19, short="Raleigh", importance=0.21),
            row("Raleigh, North Carolina", 16, short="Raleigh", importance=0.72)]
    assert best(rows, "Raleigh")["display_name"] == "Raleigh, North Carolina"


def test_a_city_with_a_state_after_it_is_still_the_city():
    rows = [row("Houston County, Texas", 12, short="Houston County"),
            row("Houston, Harris County, Texas", 16, short="Houston", importance=0.8)]
    assert best(rows, "Houston, TX")["display_name"] == "Houston, Harris County, Texas"


def test_shorthand_is_expanded_before_the_lookup():
    assert expand("MD") == "Maryland"
    assert expand("sf") == "San Francisco"
    assert expand("Ohio") == "Ohio"


def test_a_community_adds_its_own_shorthand_without_losing_the_rest():
    ours = {"the bay": "San Francisco Bay Area", "md": "Marlow Dell"}
    assert expand("The Bay", ours) == "San Francisco Bay Area"
    assert expand("MD", ours) == "Marlow Dell"
    assert expand("TX", ours) == "Texas"


def _asked(server):
    return [urllib.parse.parse_qs(urllib.parse.urlparse(path).query)["q"][0]
            for _, path, _ in server.requests]


def test_each_place_is_asked_about_once_and_the_answer_is_kept(tmp_path, server):
    """Expanded before it is sent, cached once it comes back, and nothing re-asked."""
    answers = {"Maryland": [row("Maryland, United States", 8)], "Nowhere Much": []}

    def handle(method, path, body):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)["q"][0]
        return json_reply(answers[q])

    srv = server(handle)
    cache = tmp_path / "geo.json"
    said = []
    got = geocode_all(["MD", "Nowhere Much", ""], cache, url=srv.base_url, sleep=0,
                      log=said.append)
    assert got["MD"] == {"lat": 0.0, "lon": 0.0, "display": "Maryland, United States",
                         "type": "administrative"}
    assert got["Nowhere Much"] is None
    assert json.loads(cache.read_text())["_v"] == CACHE_VERSION
    assert _asked(srv) == ["Maryland", "Nowhere Much"]

    geocode_all(["MD", "Nowhere Much"], cache, url=srv.base_url, sleep=0, log=said.append)
    assert _asked(srv) == ["Maryland", "Nowhere Much"], "the cache answered the second time"


def test_a_place_the_service_chokes_on_is_logged_and_the_rest_still_go(tmp_path, server):
    def handle(method, path, body):
        if "Texas" in urllib.parse.unquote(path):
            return 500, b"no"
        return json_reply([row("Ohio, United States", 8)])

    srv = server(handle)
    said = []
    got = geocode_all(["TX", "Ohio"], tmp_path / "geo.json", url=srv.base_url, sleep=0,
                      log=said.append)
    assert "TX" not in got and got["Ohio"]["display"] == "Ohio, United States"
    assert any("geocode failed for 'TX'" in s for s in said)


def test_a_cache_from_an_older_ranking_is_thrown_away(tmp_path, server):
    cache = tmp_path / "geo.json"
    cache.write_text(json.dumps({"_v": CACHE_VERSION - 1, "Ohio": {"lat": 1, "lon": 1}}))
    srv = server(lambda m, p, b: json_reply([row("Ohio, United States", 8)]))
    got = geocode_all(["Ohio"], cache, url=srv.base_url, sleep=0, log=lambda s: None)
    assert got["Ohio"]["lat"] == 0.0 and _asked(srv) == ["Ohio"]
