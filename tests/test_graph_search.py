"""Finding things in a graph three ways at once."""

from ml_stack.graph.search import hybrid, lexical, rrf

GRAPH = {
    "nodes": [
        {"id": "t:compilers", "kind": "topic", "label": "compilers", "mentions": 9, "attrs": {},
         "messages": ["m1"]},
        {"id": "p:ada", "kind": "person", "label": "Ada Lovelace", "mentions": 4,
         "attrs": {"role": "analyst"}, "messages": ["m1"]},
        {"id": "t:robotics", "kind": "topic", "label": "robotics", "mentions": 2, "attrs": {},
         "messages": ["m2"]},
        {"id": "p:bea", "kind": "person", "label": "Bea Marlow", "mentions": 1, "attrs": {},
         "messages": ["m2"]},
    ],
    "messages": {"m1": {"text": "I have spent years on compilers."},
                 "m2": {"text": "I fix machines for a living."}},
    "edges": [],
}


class Store:
    """A store that answers the two questions a store can answer."""

    def __init__(self, by_word=(), by_meaning=()):
        self.by_word, self.by_meaning = list(by_word), list(by_meaning)
        self.asked = []

    def search(self, text, limit=10):
        self.asked.append(("search", text))
        return [{"id": i} for i in self.by_word][:limit]

    def similar(self, vector, model="", limit=10):
        self.asked.append(("similar", tuple(vector)))
        return [{"id": i} for i in self.by_meaning][:limit]


def test_fusion_prefers_what_two_lists_agree_on():
    """Second in both beats first in one, which is the whole point of fusing."""
    assert rrf(["a", "b"], ["c", "b"], limit=3) == ["b", "a", "c"]
    assert rrf([], [], limit=3) == []
    assert rrf(["a"], limit=3) == ["a"]


def test_matching_characters_ranks_the_label_above_what_was_said():
    assert lexical(GRAPH, "compilers")[0] == "t:compilers"
    assert lexical(GRAPH, "analyst") == ["p:ada"]          # an attribute
    # only in what was said, and the better-mentioned of the two goes first
    assert lexical(GRAPH, "machines") == ["t:robotics", "p:bea"]
    assert lexical(GRAPH, "  ") == []
    assert lexical(GRAPH, "nothing at all") == []


def test_all_three_vote_and_the_result_is_fused():
    store = Store(by_word=["t:robotics"], by_meaning=["p:bea", "t:robotics"])
    hits = hybrid(GRAPH, "robotics", store=store, vector=[0.1, 0.2], limit=3)
    assert [h["id"] for h in hits][:2] == ["t:robotics", "p:bea"]
    assert hits[0]["label"] == "robotics" and hits[0]["kind"] == "topic"
    assert ("search", "robotics") in store.asked
    assert ("similar", (0.1, 0.2)) in store.asked


def test_a_way_that_is_unavailable_simply_does_not_vote():
    class Broken(Store):
        def search(self, text, limit=10):
            raise RuntimeError("no index built yet")

        def similar(self, vector, model="", limit=10):
            raise RuntimeError("nothing embedded yet")

    hits = hybrid(GRAPH, "compilers", store=Broken(), vector=[0.1], limit=3)
    # the characters still found it: the label, then the person whose message says the word
    assert [h["id"] for h in hits] == ["t:compilers", "p:ada"]
    assert [h["id"] for h in hybrid(GRAPH, "compilers", limit=3)] == ["t:compilers", "p:ada"]


def test_meaning_finds_what_the_characters_cannot():
    """"who fixes machines" shares no word with "robotics"."""
    store = Store(by_word=[], by_meaning=["t:robotics"])
    hits = hybrid(GRAPH, "who fixes machines", store=store, vector=[0.3], limit=3)
    assert "t:robotics" in [h["id"] for h in hits]


def test_an_id_the_store_knows_but_the_graph_does_not_is_dropped():
    store = Store(by_word=["t:gone"], by_meaning=["t:gone"])
    assert "t:gone" not in [h["id"] for h in hybrid(GRAPH, "compilers", store=store, vector=[0.1])]


def test_fusion_can_say_what_placed_each_id():
    """The ids alone are what `rrf` always gave; the scores are for a hit to say how well
    it did, and the two must agree."""
    from ml_stack.graph.search import rrf_scored

    scored = rrf_scored(["a", "b"], ["c", "b"], limit=3)
    assert [i for i, _ in scored] == rrf(["a", "b"], ["c", "b"], limit=3)
    assert scored[0][1] > scored[1][1] > 0


def test_without_rich_a_hit_carries_exactly_what_it_always_did():
    """A benchmark of the current behaviour is about to run; the flag off must be invisible."""
    store = Store(by_word=["t:compilers"], by_meaning=["p:ada"])
    for hit in hybrid(GRAPH, "compilers", store=store, vector=[0.1]):
        assert set(hit) == {"id", "label", "kind"}
    assert lexical(GRAPH, "compilers") == ["t:compilers", "p:ada"]


def test_the_characters_say_what_they_matched_on():
    assert lexical(GRAPH, "analyst", rich=True) == [
        {"id": "p:ada", "score": 2, "matched": ["attribute"]}]
    assert lexical(GRAPH, "compilers", rich=True)[0] == \
        {"id": "t:compilers", "score": 4, "matched": ["label"]}
    assert lexical(GRAPH, "machines", rich=True)[1]["matched"] == ["said"]


def test_a_rich_hit_names_the_voters_that_actually_fired():
    """An exact label and one word in one quote both come back as hits; what tells them
    apart is which of the three ways found them, so that is said."""
    # the characters alone: the label found the topic, and a quote found the person
    alone = hybrid(GRAPH, "compilers", rich=True, limit=3)
    assert [(h["id"], h["matched"]) for h in alone] == \
        [("t:compilers", ["label"]), ("p:ada", ["said"])]
    assert all(isinstance(h["score"], float) and h["score"] > 0 for h in alone)
    # the characters and the word index, no vectors
    words = Store(by_word=["t:compilers"])
    with_words = hybrid(GRAPH, "compilers", store=words, rich=True, limit=3)
    assert with_words[0]["matched"] == ["label", "words"]
    assert with_words[0]["score"] == round(2 / 61, 3)
    assert with_words[1]["matched"] == ["said"]
    # all three, and one the vectors alone found
    both = Store(by_word=["t:compilers"], by_meaning=["t:compilers", "p:bea"])
    with_meaning = hybrid(GRAPH, "compilers", store=both, vector=[0.1], rich=True)
    by_id = {h["id"]: h["matched"] for h in with_meaning}
    assert by_id["t:compilers"] == ["label", "words", "meaning"]
    assert by_id["p:bea"] == ["meaning"]
    assert by_id["p:ada"] == ["said"]
