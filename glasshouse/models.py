"""The nouns.

Everything that crosses a module boundary in glasshouse is one of these. They
are plain frozen dataclasses with no behaviour beyond what is needed to keep a
representation honest -- most of the interesting logic lives in functions that
take these and return new ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Sequence


@dataclass(frozen=True)
class Document:
    """A source document, before it is cut up."""

    doc_id: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable span of a document.

    ``start``/``end`` are character offsets into the parent document's text, so
    the UI can highlight the exact span that supported a claim rather than an
    approximate re-match of the chunk text.
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    start: int
    end: int
    ordinal: int = 0

    def excerpt(self, limit: int = 220) -> str:
        if len(self.text) <= limit:
            return self.text
        return self.text[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class Retrieved:
    """A chunk together with why it was retrieved."""

    chunk: Chunk
    score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass(frozen=True)
class Usage:
    """Token and money accounting for one model call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            cached=self.cached and other.cached,
        )


ZERO_USAGE = Usage()


@dataclass(frozen=True)
class Completion:
    """What an LLM gives back."""

    text: str
    usage: Usage = ZERO_USAGE
    model: str = "unknown"


class RunKind(str, Enum):
    """Why a particular generation was run.

    FULL is the answer the user sees. Every other kind exists only to be
    compared against it.
    """

    FULL = "full"
    #: Same chunks as FULL, presented in a different order. Measures how much
    #: the answer moves for reasons that have nothing to do with evidence.
    CONTROL = "control"
    #: All chunks but one.
    LEAVE_ONE_OUT = "loo"
    #: A specific set of chunks removed together, for claims that survive
    #: every single-chunk removal because two chunks say the same thing.
    COALITION = "coalition"
    #: No chunks at all. What the model says from memory.
    CLOSED_BOOK = "closed"


@dataclass(frozen=True)
class Run:
    """One generation, and the evidence it was allowed to see."""

    run_id: str
    kind: RunKind
    #: Chunk ids withheld from this run. Empty for FULL and CONTROL.
    removed: tuple[str, ...]
    answer: str
    sentences: tuple[str, ...]
    usage: Usage = ZERO_USAGE


class Verdict(str, Enum):
    """What we concluded about one sentence of the answer."""

    #: Removing specific evidence measurably changed this sentence.
    GROUNDED = "grounded"
    #: Nothing we removed changed it, and the model says the same thing with
    #: no documents at all. It may be true. It did not come from your corpus.
    MODEL_MEMORY = "model_memory"
    #: Nothing explains it: not the evidence, not the model's memory. Usually
    #: connective tissue, occasionally a claim we could not attribute.
    UNSUPPORTED = "unsupported"
    #: The sentence carries no checkable claim ("Here is what I found:").
    NO_CLAIM = "no_claim"
    #: Ablation was cut short by the budget before this sentence was resolved.
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class Support:
    """How much one chunk mattered to one sentence.

    Every retrieved chunk gets one of these for every sentence, credited or
    not. The uncredited ones are not waste: they are what the UI's heatmap
    draws, and a reader who disagrees with a verdict can see the numbers it
    was drawn from instead of being asked to trust it.
    """

    chunk_id: str
    #: Similarity lost when this chunk was withheld, above the noise floor.
    effect: float
    #: Raw similarity lost, before the control correction.
    raw_drop: float
    #: Whether this chunk cleared the threshold and was credited.
    credited: bool = False
    #: Credited only as a member of a group that had to be removed together.
    joint: bool = False


@dataclass(frozen=True)
class ClaimVerdict:
    """The finding for a single sentence of the answer."""

    index: int
    text: str
    verdict: Verdict
    #: Every retrieved chunk's effect on this sentence, strongest first.
    support: tuple[Support, ...] = ()
    #: How much this sentence moved between two runs that saw identical
    #: evidence. The threshold every effect has to clear.
    noise_floor: float = 0.0
    #: Similarity to the closed-book answer.
    memory: float = 0.0
    #: Human-readable reason, shown in the UI.
    note: str = ""

    @property
    def credited(self) -> tuple[Support, ...]:
        return tuple(s for s in self.support if s.credited)

    @property
    def strongest(self) -> Support | None:
        return self.credited[0] if self.credited else None


@dataclass(frozen=True)
class Report:
    """Everything one question produced."""

    question: str
    answer: str
    retrieved: tuple[Retrieved, ...]
    claims: tuple[ClaimVerdict, ...]
    runs: tuple[Run, ...]
    usage: Usage = ZERO_USAGE
    elapsed_s: float = 0.0
    truncated: bool = False
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def grounded_count(self) -> int:
        return sum(1 for c in self.claims if c.verdict is Verdict.GROUNDED)

    @property
    def checkable(self) -> tuple[ClaimVerdict, ...]:
        """Claims worth judging -- excludes pure connective sentences."""
        return tuple(c for c in self.claims if c.verdict is not Verdict.NO_CLAIM)

    @property
    def corpus_contributed(self) -> bool:
        """Did the documents affect this answer at all?

        This is the check that catches retrieval having returned nothing
        relevant. A retriever cannot tell you that -- cosine similarity is
        positive between any two texts, so something is always returned and
        ranked. Ablation answers it directly: if withholding every chunk in
        turn changed nothing, the corpus was not used, whatever the retrieval
        scores looked like.
        """
        return any(c.verdict is Verdict.GROUNDED for c in self.claims)

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        for r in self.retrieved:
            if r.chunk_id == chunk_id:
                return r.chunk
        return None


def renumber(claims: Sequence[ClaimVerdict]) -> tuple[ClaimVerdict, ...]:
    """Reassign contiguous indices, for when claims are filtered."""
    return tuple(replace(c, index=i) for i, c in enumerate(claims))
