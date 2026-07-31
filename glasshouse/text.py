"""Sentence segmentation, done carefully because everything downstream is a sentence.

glasshouse judges an answer one sentence at a time. If the splitter breaks
``"Revenue fell 3.5% in Q2."`` into two fragments, one of them carries a number
with no subject and the grounding verdict for it is meaningless. So this is a
real splitter with a real abbreviation list rather than ``text.split(".")``.

It is still a heuristic. It is not a parser, it only knows English, and the
tests pin down the cases it is known to get right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Words that end in a period without ending a sentence. Stored without the
#: trailing period; multi-dot forms like "e.g" keep their internal dots.
ABBREVIATIONS = frozenset(
    """
    mr mrs ms mss dr prof rev hon sr jr st vs etc al
    e.g i.e cf ca approx est no nos fig figs eq eqs
    inc ltd co corp llc plc dept div univ assn bros
    vol vols pp ch chs sec secs ed eds trans repr
    jan feb mar apr jun jul aug sept sep oct nov dec
    mon tue tues wed thu thurs fri sat sun
    u.s u.k e.u a.m p.m ph.d b.a m.a m.d d.c
    """.split()
)

#: A run of terminators, plus any closing punctuation that belongs with it.
_TERMINATOR = re.compile(r"[.!?]+[\"'\)\]”’]*")

#: The word immediately before a terminator, allowing internal periods.
_TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)$")

#: Leading list markers and enumerations that should not start a new sentence
#: on their own.
_WHITESPACE = re.compile(r"\s")


@dataclass(frozen=True)
class Span:
    """A piece of text and where it came from."""

    text: str
    start: int
    end: int

    def __len__(self) -> int:
        return len(self.text)


def _is_boundary(text: str, match: re.Match) -> bool:
    """Decide whether a run of terminators actually ends a sentence."""
    after = text[match.end() :]

    if not after.strip():
        return True

    # "3.5" or "www.example.com" -- a terminator glued to what follows is
    # punctuation inside a token, not the end of a thought.
    if not _WHITESPACE.match(after[0]):
        return False

    nxt = after.lstrip()
    # A lowercase continuation almost always means the period was an
    # abbreviation we do not know about.
    if nxt and nxt[0].islower():
        return False

    before = text[: match.start() + 1]
    word_match = _TRAILING_WORD.search(before.rstrip(".") + ".")
    if word_match:
        word = word_match.group(1).rstrip(".").lower()
        if word in ABBREVIATIONS:
            return False
        # A single letter before a period is an initial: "J. R. Tolkien".
        if len(word) == 1 and word.isalpha():
            return False

    return True


def split_sentences(text: str) -> list[Span]:
    """Split ``text`` into sentences, keeping offsets into the original."""
    if not text or not text.strip():
        return []

    spans: list[Span] = []
    start = 0

    for match in _TERMINATOR.finditer(text):
        if not _is_boundary(text, match):
            continue
        spans.append(_span(text, start, match.end()))
        start = match.end()

    if start < len(text):
        spans.append(_span(text, start, len(text)))

    return [s for s in spans if s.text]


def _span(text: str, start: int, end: int) -> Span:
    """Trim surrounding whitespace while keeping the offsets truthful."""
    raw = text[start:end]
    lead = len(raw) - len(raw.lstrip())
    trail = len(raw) - len(raw.rstrip())
    return Span(raw.strip(), start + lead, end - trail)


def sentences(text: str) -> list[str]:
    """Just the sentence strings, for when offsets do not matter."""
    return [s.text for s in split_sentences(text)]


def word_count(text: str) -> int:
    return len(text.split())


# In verbose mode every literal space is discarded, so whitespace between
# words has to be written out. Getting this wrong is silent: the pattern still
# compiles and simply never matches.
_NON_CLAIM = re.compile(
    r"""^(?:
        here (?: 's | \s+ (?: is | are ) ) \b .*
      | the \s+ (?: context | documents? | sources? | passages? | excerpts? ) \b
            .{0,60} (?: does \s+ not | do \s+ not | don't | doesn't | never ) \b .*
      | i \s+ (?: hope | can | could | cannot | can't | don't | do \s+ not
                | was | am ) \b .*
      | let \s+ me \s+ know \b .*
      | \W*                                  # punctuation or a rule, no words
    )$""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# Models occasionally ignore the prompt's request not to mention their input.
# A source-framing prefix is not itself a claim, but the sentence after its
# comma often is. Strip only the prefix before classification; rejecting the
# whole sentence silently hides assertions such as "Based on the excerpts, the
# crew included Amara." from ablation.
_SOURCE_FRAME = re.compile(
    r"""^(?:
        based \s+ on \s+ (?: the \s+ )?
            (?: provided \s+ | given \s+ | supplied \s+ )?
            (?: context | documents? | sources? | passages? | excerpts? )
      | according \s+ to \s+ (?: the \s+ )?
            (?: provided \s+ | given \s+ | supplied \s+ )?
            (?: context | documents? | sources? | passages? | excerpts? )
    ) \s* [,;:\-—] \s*""",
    re.IGNORECASE | re.VERBOSE,
)


def carries_a_claim(sentence: str) -> bool:
    """Is this sentence asserting something about the world?

    Answers contain scaffolding -- "Here is what I found:", "I hope this
    helps." -- that is neither grounded nor hallucinated, and scoring it
    pollutes the numbers in both directions. Anything genuinely ambiguous is
    kept, because a false ``True`` merely produces one extra verdict while a
    false ``False`` silently hides a claim from judgement.
    """
    stripped = sentence.strip()
    if not stripped:
        return False
    frame = _SOURCE_FRAME.match(stripped)
    if frame:
        stripped = stripped[frame.end() :].strip()
        if not stripped:
            return False
    if _NON_CLAIM.match(stripped):
        return False
    # A fragment with no verb-like content and almost no words is a heading.
    return word_count(stripped) >= 3
