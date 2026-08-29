"""Entity resolution: stems that agree, and folds that only join what belongs together."""

from __future__ import annotations

import pytest
from ml_stack.entities import canonical, fold_duplicates, fold_key, looks_like_handle, stem

# a word and its inflections have to reduce to one form, or a fold key never matches
SAME = [
    ("agentic", "agents"),
    ("service", "services"),
    ("engineer", "engineering"),
    ("operations", "operational"),
    ("implementation", "implementations"),
    ("automation", "automating"),
    ("analytic", "analytics"),
]

DIFFERENT = [
    ("finance", "financial"),
    ("legal", "logical"),
    ("retail", "repair"),
    ("audit", "audio"),
    ("python", "pytorch"),
]


@pytest.mark.parametrize("a,b", SAME)
def test_inflections_share_a_stem(a, b):
    assert stem(a) == stem(b)


@pytest.mark.parametrize("a,b", DIFFERENT)
def test_unrelated_words_keep_their_own_stem(a, b):
    assert stem(a) != stem(b)


def test_short_words_survive_stemming():
    for w in ("ai", "sap", "rag", "mcp", "sales", "legal"):
        assert stem(w)


def test_fold_key_drops_stopwords_and_ignores_order():
    assert fold_key("the future of work") == fold_key("work future")
    assert fold_key("AI Agents") == fold_key("agentic ai")
    assert fold_key("retail", stopwords={"industry"}) == fold_key("retail industry", stopwords={"industry"})
    assert fold_key("retail") != fold_key("retail industry")


def test_canonical_matches_case_insensitively_and_tidies_the_rest():
    aliases = {"ada": "Ada Lovelace"}
    assert canonical("@Ada", aliases) == "Ada Lovelace"
    assert canonical(" ADA ", aliases) == "Ada Lovelace"
    assert canonical("Grace  Hopper,", aliases) == "Grace Hopper"


def test_looks_like_handle():
    assert looks_like_handle("night owl")
    assert looks_like_handle("a.person")
    assert not looks_like_handle("Ada Lovelace")
    assert not looks_like_handle("Alan T")


def node(nid, kind, label, weight=1):
    return {"id": nid, "kind": kind, "label": label, "weight": weight}


RANK = {"person": 0, "org": 1, "place": 2, "topic": 3}


def test_variants_fold_onto_the_heavier_record():
    fold = fold_duplicates(
        [node("t:agents", "topic", "AI agents", weight=4), node("t:agentic", "topic", "agentic AI")],
        rank=RANK, weak_kinds={"topic"})
    assert fold == {"t:agentic": "t:agents"}


def test_a_weak_kind_folds_into_a_stronger_one():
    fold = fold_duplicates(
        [node("o:harbor", "org", "Harbor"), node("t:harbor", "topic", "harbor", weight=9)],
        rank=RANK, weak_kinds={"topic"})
    assert fold == {"t:harbor": "o:harbor"}


def test_two_strong_kinds_stay_apart():
    fold = fold_duplicates([node("o:york", "org", "York"), node("p:york", "place", "York")],
                           rank=RANK, weak_kinds={"topic"})
    assert fold == {}


def test_a_two_word_topic_absorbs_its_longer_forms():
    fold = fold_duplicates(
        [node("t:ai-agents", "topic", "AI agents"), node("t:ai-agent-deploy", "topic", "AI agent deployment")],
        rank=RANK, weak_kinds={"topic"})
    assert fold == {"t:ai-agent-deploy": "t:ai-agents"}


def test_a_one_word_topic_joins_its_only_superset():
    fold = fold_duplicates(
        [node("t:onboarding", "topic", "onboarding"), node("t:community", "topic", "community onboarding")],
        rank=RANK, weak_kinds={"topic"})
    assert fold == {"t:onboarding": "t:community"}


def test_a_one_word_topic_with_rival_supersets_stays_put():
    fold = fold_duplicates(
        [node("t:ai", "topic", "AI"), node("t:agents", "topic", "AI agents"),
         node("t:training", "topic", "AI training")],
        rank=RANK, weak_kinds={"topic"})
    assert "t:ai" not in fold


def test_same_kind_folds_without_weak_kinds():
    assert fold_duplicates([node("t:a", "topic", "AI agents"), node("t:b", "topic", "agentic AI")],
                           rank=RANK) == {"t:b": "t:a"}


def test_a_record_never_folds_onto_itself():
    fold = fold_duplicates([node("t:x", "topic", "sales"), node("t:y", "topic", "sale")],
                           rank=RANK, weak_kinds={"topic"})
    assert all(k != v for k, v in fold.items())
