"""The embed backend: zero-shot ranking, trained heads, and the file a head is saved in."""

from __future__ import annotations

import numpy as np
import pytest
from decide_fakes import embedder
from safetensors.numpy import save_file

from poolhouse.decide.cases import Case
from poolhouse.decide.embed import EmbedDecider, Head, Training, fit_head
from poolhouse.decide.types import DecideError, options_of

TEAMS = {"billing": "payment invoice refund charge", "technical": "error crash bug timeout",
         "sales": "price quote upgrade discount"}

STATES = {
    "billing": ["my invoice shows a double charge", "refund for the last payment please",
                "payment failed and the charge is pending", "invoice total looks wrong"],
    "technical": ["the app crash every time I open it", "got a timeout error calling the api",
                  "bug in the export, error on save", "crash with error code 500"],
    "sales": ["what is the price for the team plan", "can I get a quote for 50 seats",
              "is there a discount if I upgrade", "upgrade price for enterprise"],
}


def cases(parts=(0, 1, 2, 3)) -> list[Case]:
    return [Case("Which team?", STATES[label][i], options_of(TEAMS), label, id=f"{label}{i}")
            for label in TEAMS for i in parts]


def test_with_no_head_options_are_ranked_by_similarity_to_the_state():
    d = EmbedDecider(embedder)
    got = d.decide("Which team?", "my invoice has a wrong charge", TEAMS)
    assert got.choice == "billing"
    assert got.backend == "embed" and got.details["head"] == "zero-shot"
    assert sum(got.scores.values()) == pytest.approx(1.0)


def test_a_pairwise_head_trained_on_a_few_cases_beats_zero_shot_on_held_out_ones():
    train, test = cases((0, 1, 2)), cases((3,))
    head = fit_head(embedder, train)
    zero, trained = EmbedDecider(embedder), EmbedDecider(embedder, head=head)
    right = [sum(d.decide(c.question, c.state, c.options).choice == c.label for c in test)
             for d in (zero, trained)]
    assert right[1] >= right[0]
    assert right[1] == len(test)


def test_a_pairwise_head_still_works_on_options_it_never_saw():
    head = fit_head(embedder, cases((0, 1, 2)))
    got = EmbedDecider(embedder, head=head).decide(
        "Which team?", "the app has an error and crash", {"outage": "error crash timeout",
                                                         "pricing": "price quote discount"})
    assert got.choice == "outage"


def test_a_classes_head_decides_only_its_own_options_in_any_order():
    head = fit_head(embedder, cases((0, 1, 2)), Training("classes"))
    d = EmbedDecider(embedder, head=head)
    reordered = list(reversed(list(TEAMS)))
    assert d.decide("Which team?", STATES["sales"][3], reordered).choice == "sales"
    with pytest.raises(DecideError, match="decides among"):
        d.decide("Which team?", "x", ["billing", "legal"])


def test_a_classes_head_needs_the_same_options_in_every_case():
    mixed = [Case("q", "s", options_of(["a", "b"]), "a"), Case("q", "s", options_of(["a", "c"]), "a")]
    with pytest.raises(ValueError, match="same options"):
        fit_head(embedder, mixed, Training("classes"))


def test_a_head_round_trips_through_safetensors_with_its_description(tmp_path):
    head = fit_head(embedder, cases(), Training("classes", embedder_id="fake", data_hash="abc"))
    path = tmp_path / "head.safetensors"
    head.save(path)
    back = Head.load(path)
    assert (back.kind, back.options, back.embedder, back.data_hash, back.n) == (
        "classes", tuple(TEAMS), "fake", "abc", 12)
    assert pytest.approx(head.W, abs=1e-5) == back.W


def test_a_file_that_is_not_a_head_or_has_the_wrong_shapes_is_refused(tmp_path):
    odd = tmp_path / "odd.safetensors"
    save_file({"w": np.zeros(3, dtype="float32")}, str(odd))
    with pytest.raises(DecideError, match="not an embed head"):
        Head.load(odd)
    wrong = tmp_path / "wrong.safetensors"
    save_file({"w": np.zeros(3, dtype="float32")}, str(wrong), metadata={
        "format": "poolhouse-embed-head/1", "kind": "pairwise", "dim": "4", "options": "[]"})
    with pytest.raises(DecideError, match="do not match"):
        Head.load(wrong)


def test_a_head_trained_for_another_embedding_size_is_refused():
    head = fit_head(embedder, cases())
    with pytest.raises(DecideError, match="dimensions"):
        EmbedDecider(lambda texts: [[1.0, 0.0]] * len(texts), head=head).decide("q", "s",
                                                                              ["a", "b"])


def test_option_embeddings_are_asked_for_once():
    asked: list[str] = []

    def counting(texts):
        asked.extend(texts)
        return embedder(texts)

    d = EmbedDecider(counting)
    d.decide("q", "one", TEAMS)
    d.decide("q", "two", TEAMS)
    assert sum(t.startswith("billing") for t in asked) == 1
