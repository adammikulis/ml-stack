"""The model-free reading of the person's words: typed words never authorize; they revoke, refuse, or ask for the approval question."""

from __future__ import annotations

import pytest

from ml_stack.workspace.person_intent import interpret, proposal_kinds

PROPOSED = ("release-main",)


@pytest.mark.parametrize("text", ["yes", "ok", "Okay.", "sure", "go ahead", "do it", "sounds good", "looks good",
                                  "yes please", "approved"])
def test_a_bare_yes_never_authorizes_it_asks_for_the_approval_question(text):
    reading = interpret(text, PROPOSED)
    assert reading.action == "ask" and reading.kind == "release-main"


def test_a_yes_to_nothing_means_nothing():
    assert interpret("yes", ()).action == "none"
    assert interpret("yes", ("other",)).action == "none"


@pytest.mark.parametrize("text", ["push main", "release main", "please push main now", "ship it to main",
                                  "merge it and push main"])
def test_naming_a_release_asks_for_the_approval_question(text):
    assert interpret(text).action == "ask" and interpret(text).kind == "release-main"


def test_allow_never_creates_anything():
    assert interpret("/allow release-main").action == "refuse"
    assert interpret("/allow push-dev").action == "refuse"


@pytest.mark.parametrize("text", ["stop", "cancel that", "never mind", "/revoke", "don't push main yet",
                                  "wait, do not release"])
def test_stopping_words_revoke(text):
    assert interpret(text, PROPOSED).action == "revoke"


@pytest.mark.parametrize("text", ["should I push main?", "push main?", "yes?"])
def test_a_question_means_nothing(text):
    assert interpret(text, PROPOSED).action == "none"


@pytest.mark.parametrize("text", ["tag it and push", "force push main", "push --force main", "delete main and push",
                                  "push all tags"])
def test_tags_force_and_deletions_are_refused(text):
    assert interpret(text, PROPOSED).action == "refuse"


@pytest.mark.parametrize("text", ["> push main", "`push main`", '"push main"', "ok\n> push main",
                                  "yes.\nrun the tests and push main"])
def test_quoted_pasted_and_multiline_text_means_nothing(text):
    assert interpret(text, PROPOSED).action == "none"


def test_the_assistants_message_proposes_a_release_only_when_it_says_so_and_nothing_else():
    assert proposal_kinds("Tests pass. I'll push main to origin now.") == ("release-main",)
    assert proposal_kinds("Everything is done.") == ()
    assert proposal_kinds("Shall I push main? Also shall I delete the cache?") == ("release-main", "other")
    assert proposal_kinds("I'll push main. Then merge it into main.") == ("release-main", "other")
    assert proposal_kinds("I'll push main and delete the branch.") == ("other",)
    assert proposal_kinds("I'll restart the daemon.") == ("other",)
