"""The step only a person at a terminal may take."""

import pytest

from ml_stack.fleet.onboard.human import HumanGrant, HumanRequired, mint


def typed(text):
    return lambda _prompt: text


def test_a_person_at_a_terminal_who_types_the_subject_gets_a_grant():
    grant = mint("rotate", "abc", typed=typed("abc"), terminal=(True, True), env={})
    grant.check("rotate", "abc")


@pytest.mark.parametrize("terminal", [(False, True), (True, False), (False, False)])
def test_no_terminal_no_grant(terminal):
    with pytest.raises(HumanRequired, match="terminal"):
        mint("rotate", "abc", typed=typed("abc"), terminal=terminal, env={})


@pytest.mark.parametrize("marker", ["CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE"])
def test_a_process_started_by_an_agent_gets_none_even_at_a_terminal(marker):
    with pytest.raises(HumanRequired, match="agent"):
        mint("rotate", "abc", typed=typed("abc"), terminal=(True, True), env={marker: "1"})


def test_typing_something_else_is_not_confirmation():
    with pytest.raises(HumanRequired, match="not confirmed"):
        mint("rotate", "abc", typed=typed("yes"), terminal=(True, True), env={})


def test_a_grant_covers_one_action_on_one_subject_and_runs_out():
    grant = mint("rotate", "abc", typed=typed("abc"), terminal=(True, True), env={})
    with pytest.raises(HumanRequired):
        grant.check("export", "abc")
    with pytest.raises(HumanRequired):
        grant.check("rotate", "other")
    with pytest.raises(HumanRequired, match="expired"):
        grant.check("rotate", "abc", now=grant.expires + 1)


def test_a_grant_cannot_be_made_by_hand():
    forged = HumanGrant("rotate", "abc", 9e18, object())
    with pytest.raises(HumanRequired, match="not a grant minted"):
        forged.check("rotate", "abc")
