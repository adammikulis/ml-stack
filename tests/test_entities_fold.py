"""A coined vocabulary folds to one word, and only where the guess is safe."""

from ml_stack.entities.fold import ESTABLISHED, dead_keys, fold_edges, fold_names


def edge(source, rel, target, weight, said):
    return {(source, rel, target): {"source": source, "rel": rel, "target": target,
                                    "weight": weight, "messages": list(said)}}


def test_the_same_word_typed_twice_folds_into_the_heavier_spelling():
    canonical, folds = fold_names({"works_at": 4, "worksat": 1, "mentors": 2})
    assert canonical == {"works_at": "works_at", "worksat": "works_at", "mentors": "mentors"}
    assert folds == [{"from": "worksat", "into": "works_at", "written": False}]


def test_a_written_map_settles_two_genuinely_different_words():
    """"mentors" and "advises" share no spelling; only a decision can join them."""
    canonical, folds = fold_names({"mentors": 2, "advises": 1}, {"mentors": "advises"})
    assert canonical["mentors"] == "advises"
    assert folds == [{"from": "mentors", "into": "advises", "written": True}]


def test_two_established_names_refuse_to_fold_and_say_so():
    lines: list[str] = []
    canonical, folds = fold_names({"works_at": ESTABLISHED + 1, "worksat": ESTABLISHED},
                                  log=lines.append, label="relations",
                                  settles="write it down")
    assert folds == []
    assert canonical == {"works_at": "works_at", "worksat": "worksat"}
    assert lines == ["relations: 'worksat' (3) and 'works_at' (4) are both established, "
                     "so neither folds; write it down"]

    # a written entry is a decision, not a guess, and folds regardless of the floor
    canonical, folds = fold_names({"works_at": ESTABLISHED + 1, "worksat": ESTABLISHED},
                                  {"worksat": "works_at"})
    assert canonical["worksat"] == "works_at" and folds[0]["written"] is True


def test_a_fold_is_logged_rather_than_silent():
    """A wrong fold merges two different relationships; the least it owes anyone is a line."""
    lines: list[str] = []
    fold_names({"works_at": 4, "worksat": 1}, log=lines.append)
    assert lines == ["'worksat' (1) folded into 'works_at' (4)"]


def test_folded_edges_join_their_weight_and_their_provenance():
    edges = {**edge("p:ada", "works_at", "o:pellard", 4, ["m1"]),
             **edge("p:ada", "worksat", "o:pellard", 1, ["m2"]),
             **edge("p:bea", "mentors", "p:cyd", 2, ["m3"])}
    out, folds = fold_edges(edges)
    assert set(out) == {("p:ada", "works_at", "o:pellard"), ("p:bea", "mentors", "p:cyd")}
    joined = out[("p:ada", "works_at", "o:pellard")]
    assert joined["weight"] == 5 and joined["messages"] == ["m1", "m2"]
    assert joined["rel"] == "works_at", "the edge repeats its relation and must be rewritten"
    assert folds == [{"from": "worksat", "into": "works_at", "written": False}]


def test_an_edge_that_does_not_fold_is_left_exactly_as_it_was():
    edges = edge("p:ada", "mentors", "p:bea", 2, ["m1"])
    out, folds = fold_edges(edges)
    assert out == edges and folds == []


def test_a_map_key_nothing_produces_any_more_is_reported_per_map():
    dead = dead_keys({
        "topics": ({"practical ai": "AI implementations", "job opportunities": "hiring"},
                   {"practical ai", "compilers"}),
        "relations": ({"worksat": "works_at"}, {"worksat"}),
        "aliases": ({}, set()),
    })
    assert dead == {"topics": ["job opportunities"]}, "only the maps with a dead key at all"


def test_a_key_is_matched_casefolded_against_what_is_alive():
    assert dead_keys({"topics": ({"Practical AI": "x"}, {"practical ai"})}) == {}
