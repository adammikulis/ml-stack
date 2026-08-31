"""Known names are swapped for stable placeholders."""

from ml_stack.redact import Redactor, tag


def test_the_same_name_always_gives_the_same_tag():
    assert tag("Ada Lovelace") == tag("  ada lovelace ")
    assert tag("Ada Lovelace") != tag("Bea Marlow")
    assert tag("Ada Lovelace").startswith("person#")


def test_names_are_replaced_wherever_they_appear():
    hide = Redactor({"Ada Lovelace", "Bea Marlow"})
    out = hide("Ada Lovelace wrote to bea marlow about Ada Lovelace")
    assert "Ada Lovelace" not in out and "marlow" not in out.casefold()
    assert out.count(tag("Ada Lovelace")) == 2
    assert tag("Bea Marlow") in out


def test_the_longest_name_wins_when_one_contains_another():
    hide = Redactor({"Pellard", "Pellard Foundry"})
    assert hide("Pellard Foundry ships") == f"{tag('Pellard Foundry')} ships"


def test_a_name_inside_a_word_is_left_alone():
    hide = Redactor({"Pellard"})
    assert hide("the Pellardesque style") == "the Pellardesque style"


def test_no_names_means_no_change():
    assert Redactor([])("Ada Lovelace said nothing") == "Ada Lovelace said nothing"


def test_anything_is_first_made_text():
    hide = Redactor({"Ada Lovelace"})
    assert hide({"who": "Ada Lovelace"}) == "{'who': '" + tag("Ada Lovelace") + "'}"


def test_the_prefix_names_what_kind_of_thing_was_hidden():
    assert tag("Quenlow Robotics", "org").startswith("org#")
    hide = Redactor({"Quenlow Robotics"}, prefix="org")
    assert hide("Quenlow Robotics hires") == f"{tag('Quenlow Robotics', 'org')} hires"
