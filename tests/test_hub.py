"""What a model card asks for, read out of prose."""
from ml_stack.hub import advice


def test_advice_reads_the_settings_a_card_names():
    said = """
    ## Best Practices
    ### 1. Sampling Parameters
    Use the following standardized sampling configuration across all use cases:
    * `temperature=1.0`
    * `top_p=0.95`
    * `top_k=64`
    """
    assert advice(said) == {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0}


def test_advice_reads_the_several_ways_a_card_writes_them():
    assert advice('"temperature": 0.7') == {"temperature": 0.7}
    assert advice("temp: 0.6, top-k: 20") == {"temperature": 0.6, "top_k": 20.0}
    assert advice("repetition_penalty = 1.05") == {"repeat_penalty": 1.05}
    # a card that only shows a command line still reads
    assert advice("llama-server --temp 0.6 --top-k 20") == {"temperature": 0.6, "top_k": 20.0}


def test_the_first_mention_wins_inside_the_recommending_section():
    """A card names its recommendation, then shows variations of it."""
    assert advice("temperature=1.0 ... later, try temperature=0.3") == {"temperature": 1.0}


def test_the_recommending_section_is_read_before_the_rest():
    """Otherwise a card that opens by warning against a setting reads as recommending it.

    "First mention wins" is only safe once you know which part is doing the recommending.
    """
    said = """
    # Notes
    Do not use `temperature=0.0` with this model; it degenerates.

    ## Recommended sampling
    * `temperature=0.8`
    * `top_p=0.9`

    ## Troubleshooting
    If output repeats, try `temperature=1.4`.
    """
    assert advice(said) == {"temperature": 0.8, "top_p": 0.9}


def test_a_card_with_no_marked_section_falls_back_to_the_whole_document():
    assert advice("Just use temperature=0.7 and you'll be fine.") == {"temperature": 0.7}


def test_a_card_that_says_nothing_says_nothing():
    """Empty is a real answer: nobody chose, so the caller's default stands. Inventing a
    number here would be putting words in a publisher's mouth."""
    assert advice("") == {}
    assert advice("This model is good at reasoning.") == {}
    assert advice("gpt-oss supports configurable reasoning effort: low, medium, high.") == {}
