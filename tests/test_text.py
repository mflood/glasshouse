import pytest

from glasshouse.text import carries_a_claim, sentences, split_sentences


@pytest.mark.parametrize(
    "text,expected",
    [
        ("One. Two. Three.", 3),
        ("Revenue fell 3.5% in Q2. It recovered later.", 2),
        ("Dr. Chen signed it. She left in May.", 2),
        ("The vendor (e.g. Acme) was late. We rebid.", 2),
        ("J. R. Hartley wrote it. Nobody read it.", 2),
        ("Is it done? Yes! Finally.", 3),
        ('She said "it slipped." Then she left.', 2),
        ("See www.example.com for details. It is stale.", 2),
        ("Approx. 40 units shipped. The rest were held.", 2),
        ("No trailing terminator here", 1),
    ],
)
def test_sentence_counts(text, expected):
    assert len(sentences(text)) == expected


def test_offsets_point_at_the_original_text():
    text = "  The rollout slipped.   Engineering blamed the vendor.  "
    for span in split_sentences(text):
        assert text[span.start : span.end] == span.text


def test_a_decimal_does_not_split():
    assert sentences("It cost 2.1 million dollars.") == [
        "It cost 2.1 million dollars."
    ]


def test_empty_and_whitespace_produce_nothing():
    assert sentences("") == []
    assert sentences("   \n  ") == []


def test_lowercase_continuation_is_not_a_boundary():
    """A period followed by a lowercase word is an abbreviation we do not know."""
    assert len(sentences("Filed under sec. 12 of the act. Then appealed.")) == 2


@pytest.mark.parametrize(
    "sentence",
    [
        "The rollout slipped by six weeks.",
        "Engineering blamed a vendor firmware revision.",
        "It cost 2.1 million dollars.",
    ],
)
def test_real_claims_are_kept(sentence):
    assert carries_a_claim(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "Here is what I found:",
        "Based on the provided context, here are the details.",
        "I hope this helps.",
        "---",
        "Sure!",
    ],
)
def test_scaffolding_is_not_a_claim(sentence):
    """Scoring boilerplate pollutes the numbers in both directions."""
    assert not carries_a_claim(sentence)


def test_ambiguous_sentences_are_kept_rather_than_dropped():
    """A false negative hides a claim from judgement; a false positive costs one row."""
    assert carries_a_claim("The context suggests a longer delay than reported.")


def test_a_summary_still_carries_its_claim():
    """"In summary, X" asserts X, and dropping it would hide a real claim."""
    assert carries_a_claim("In summary, the project was late.")
