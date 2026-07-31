"""Prompting.

The prompt is part of the measurement apparatus, not a detail. Three properties
below exist specifically so that ablation means something:

**The model must not be told to refuse when evidence is thin.** A prompt that
says "answer only from the context" produces "I cannot answer that" the moment
a chunk is withheld, which makes every claim look grounded and the whole
measurement circular. glasshouse asks for a natural answer and *measures*
whether the evidence was used, rather than instructing the model to pretend it
was.

**Excerpts are numbered but not labelled with an identity the model can cite.**
Inviting citations would let the model claim support it did not use, and the
citation would then be mistaken for evidence.

**Removal has to be invisible.** Excerpts are renumbered after a chunk is
withheld, so the model cannot infer that something was taken away and change
its behaviour in response. A gap at excerpt 3 is a signal, and a model that
notices signals is a model that is no longer answering the same question.
"""

from __future__ import annotations

from typing import Sequence

from .models import Chunk

SYSTEM = (
    "You answer questions using the excerpts you are given. "
    "Write plainly, in full sentences, and keep it under six sentences. "
    "Use the excerpts where they are relevant. Where they are not, answer "
    "from your own knowledge as you normally would. "
    "Do not mention the excerpts, do not refer to them by number, and do not "
    "describe what you were or were not given -- just answer the question."
)

CLOSED_BOOK_SYSTEM = (
    "Answer the question from your own knowledge. "
    "Write plainly, in full sentences, and keep it under six sentences. "
    "Do not mention what you do or do not have access to -- just answer."
)


def build_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    """Render the user turn for a set of excerpts."""
    if not chunks:
        return question.strip()

    parts = ["Excerpts:", ""]
    for number, chunk in enumerate(chunks, 1):
        parts.append("[%d] %s" % (number, chunk.text.strip()))
        parts.append("")
    parts.append("Question: %s" % question.strip())
    return "\n".join(parts)


def system_for(chunks: Sequence[Chunk]) -> str:
    return SYSTEM if chunks else CLOSED_BOOK_SYSTEM


INJECT_SYSTEM = (
    "You produce test data for a hallucination detector. "
    "Given a passage, insert exactly one additional sentence that is "
    "plausible, on-topic, specific, and NOT supported by the passage -- the "
    "kind of confident detail a language model invents. Keep every original "
    "sentence exactly as written, in order. Return only the modified passage."
)


def build_injection_prompt(answer: str) -> str:
    return (
        "Passage:\n%s\n\n"
        "Insert one unsupported sentence, then return the whole passage."
        % answer.strip()
    )
