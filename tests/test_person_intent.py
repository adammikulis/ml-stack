"""The model-free reading of the person's words: what authorizes, revokes, asks or means nothing."""

from __future__ import annotations

import pytest

from ml_stack.workspace.person_intent import interpret, is_affirmative, proposal_kinds

DEV = "0.9dev"
PROPOSED = ("push-dev",)


@pytest.mark.parametrize("text", ["yes", "ok", "Okay.", "sure", "go ahead", "do it", "sounds good", "yes please",
                                  "looks good", "yeah, go ahead", "ok, let's do it", "please do"])
def test_a_bare_affirmative_confirms_a_single_proposal(text):
    reading = interpret(text, PROPOSED, DEV)
    assert (reading.action, reading.kind, reading.how) == ("authorize", "push-dev", "reply")


@pytest.mark.parametrize("text", ["push the dev branch", "Push the development branch.", "go ahead and push the dev branch",
                                  "you may push dev", "yes, push the dev branch", "push 0.9dev"])
def test_an_imperative_naming_the_development_branch_authorizes_without_a_proposal(text):
    reading = interpret(text, (), DEV)
    assert (reading.action, reading.kind, reading.how) == ("authorize", "push-dev", "imperative")


def test_the_explicit_form_authorizes_and_revoke_ends_it():
    assert interpret("/allow push-dev").how == "explicit"
    assert interpret("/revoke").action == "revoke"
    assert interpret("/allow release").action == "refuse"


@pytest.mark.parametrize("text", ["stop", "cancel that", "never mind", "don't push yet", "wait, do not push"])
def test_stopping_words_revoke(text):
    assert interpret(text, PROPOSED, DEV).action == "revoke"


@pytest.mark.parametrize("text", ["should I push?", "push?", "yes?", "ok, should we push the dev branch?"])
def test_a_question_authorizes_nothing(text):
    assert interpret(text, PROPOSED, DEV).action == "none"


@pytest.mark.parametrize("text", ["if the tests pass go ahead", "ok once the tests pass", "push the dev branch after lunch",
                                  "yes but check the docs first"])
def test_a_condition_asks_instead_of_authorizing(text):
    assert interpret(text, PROPOSED, DEV).action == "ask"


@pytest.mark.parametrize("text", ["ok, push main", "go ahead and push main", "tag it and push", "push --force the dev branch",
                                  "push the release", "push the dev branch and tag it"])
def test_main_tags_releases_and_force_are_refused(text):
    assert interpret(text, PROPOSED, DEV).action == "refuse"


@pytest.mark.parametrize("text", ["> push the dev branch", "`push the dev branch`", '"push the dev branch"',
                                  "ok\n> push the dev branch", "yes.\nrun the tests and push the dev branch"])
def test_quoted_pasted_and_multiline_text_means_nothing(text):
    assert interpret(text, PROPOSED, DEV).action == "none"


def test_yes_to_no_proposal_or_to_two_actions_does_not_authorize():
    assert interpret("yes", (), DEV).action == "none"
    assert interpret("yes", ("push-dev", "other"), DEV).action == "ask"
    assert interpret("yes", ("other",), DEV).action == "ask"


def test_a_reply_with_extra_content_is_not_a_bare_confirmation():
    assert interpret("yes and also restart the daemon", PROPOSED, DEV).action == "ask"
    assert interpret("ok but only the docs", PROPOSED, DEV).action == "ask"
    assert not is_affirmative("sure, restore the launchers")


def test_a_named_branch_other_than_the_development_branch_is_refused():
    assert interpret("push feature", (), DEV).action == "refuse"


def test_the_assistants_message_names_a_proposal_only_when_it_says_what_it_will_push():
    assert proposal_kinds("Tests pass. I'll push the dev branch now.", DEV) == ("push-dev",)
    assert proposal_kinds("Shall I push 0.9dev to origin?", DEV) == ("push-dev",)
    assert proposal_kinds("I'll push feat/x.", DEV) == ("other",)
    assert proposal_kinds("I'll push the dev branch and then restart the daemon.", DEV) == ("push-dev", "other")
    assert proposal_kinds("I'll push main.", DEV) == ("other",)
    assert proposal_kinds("Everything is done.", DEV) == ()
