"""Getting from one node to another."""

from ml_stack.entities.paths import between, shortest_path


def edges(*rows):
    return [{"source": a, "target": b, "weight": w} for a, b, w in rows]


def test_the_best_evidenced_way_wins_over_the_shortest():
    """One hop seen once is a worse answer than two hops everyone agrees on."""
    g = edges(("a", "z", 1), ("a", "m", 9), ("m", "z", 9))
    assert between(g, "a", "z") == ["a", "m", "z"]
    # make the direct link well attested and it wins
    g = edges(("a", "z", 20), ("a", "m", 9), ("m", "z", 9))
    assert between(g, "a", "z") == ["a", "z"]


def test_no_way_across_is_no_path():
    g = edges(("a", "b", 3), ("y", "z", 3))
    assert shortest_path(g, "a", "z") == []
    assert between(g, "a", "z") == []
    assert between(g, "a", "nobody") == []


def test_a_node_reaches_itself_without_walking():
    g = edges(("a", "b", 1))
    assert shortest_path(g, "a", "a") == []
    assert between(g, "a", "a") == ["a"]


def test_the_walk_is_the_edges_in_order():
    g = edges(("a", "b", 5), ("b", "c", 5), ("c", "d", 5))
    walk = shortest_path(g, "a", "d")
    assert [(e["source"], e["target"]) for e in walk] == [("a", "b"), ("b", "c"), ("c", "d")]
    assert between(g, "a", "d") == ["a", "b", "c", "d"]


def test_a_long_way_round_is_refused_past_the_hop_limit():
    g = edges(*[(str(i), str(i + 1), 5) for i in range(10)])
    assert between(g, "0", "10", hops=3) == []
    assert between(g, "0", "3", hops=3) == ["0", "1", "2", "3"]
