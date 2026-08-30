"""One word written two ways."""

from ml_stack.entities.spelling import budget, close, collapsed, distance, nearest, spelled_in


def test_distance_gives_up_past_the_ceiling():
    assert distance("pelard", "pellard") == 1
    assert distance("kitten", "sitting", ceiling=3) == 3
    # two words nothing like each other cost nothing to reject
    assert distance("pelard", "encyclopaedia", ceiling=2) == 3


def test_a_doubled_letter_is_the_same_word_at_any_length():
    assert collapsed("Pellard") == collapsed("Pelard") == "pelard"
    assert close("Pellard", "Pelard")
    # short enough that distance alone would refuse it
    assert close("Vitt", "Vit")


def test_a_short_word_may_not_be_substituted():
    """Two people can be four letters apart by one letter and still be two people."""
    assert not close("Bram", "Brem")
    assert not close("Dana", "Dan")
    assert not close("AI", "Al")


def test_a_longer_word_affords_more():
    assert close("robotics", "robtics")
    assert close("Quenlow", "Quenlaw")
    assert not close("Vantwill", "Ventrall")
    assert budget("hi") == 0 and budget("robotics") == 2


def test_nearest_finds_the_word_a_name_was_typed_as():
    assert nearest("Pellard", "getting the duck but pelard looks cool") == ("pelard", 1)
    assert nearest("Pellard", "a sentence about something else entirely") is None
    # an exact word wins outright, even when a near one is also there
    assert nearest("Pellard", "the pellard and pelard spellings both") == ("pellard", 0)


def test_spelled_in_reads_a_multi_word_name_word_by_word():
    assert spelled_in("Pellard Foundry", "my consultancy pelard foundry does grant work")
    assert not spelled_in("Quenlow Robotics", "an unrelated line of text")
