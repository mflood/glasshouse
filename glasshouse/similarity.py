"""Did this sentence survive?

Every verdict glasshouse produces reduces to one comparison: the answer said X
with all the evidence present; when evidence was withheld, did anything like X
come back? That is near-duplicate detection between one sentence and a short
passage, and it is deliberately not the same problem as semantic search.

Two signals are combined with ``max``:

*Semantic* similarity catches rewording -- "it cost roughly $2M" against "the
cost was about two million dollars".

*Lexical* character-n-gram similarity catches the case semantic embeddings are
worst at: two sentences that differ in one crucial token. "The rollout slipped
six weeks" and "the rollout slipped sixteen weeks" embed almost identically,
and treating them as the same claim would be wrong -- but for *survival* the
question is only whether the claim reappeared at all, and near-identical
wording is strong evidence that it did.

Taking the maximum rather than the mean is a deliberate asymmetry. A high
survival score suppresses a grounding claim; a low one produces one. Since
falsely reporting "this sentence is ungrounded" is the expensive error --
it accuses the model of making something up -- glasshouse errs toward
concluding that a sentence survived.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .embed import Embedder, NgramEmbedder, cosine


class Matcher:
    """Scores whether sentences reappeared in an alternative answer."""

    def __init__(self, embedder: Embedder, lexical: Embedder | None = None):
        self.embedder = embedder
        # A second, purely local model. Free, deterministic, and measuring
        # something the first one is bad at.
        self.lexical = lexical or NgramEmbedder(dimensions=1024)

    def survival(
        self, reference: Sequence[str], candidate: Sequence[str]
    ) -> np.ndarray:
        """For each reference sentence, its best match in ``candidate``.

        Returns values in ``[0, 1]``: 1.0 means the sentence came back
        essentially unchanged, 0.0 means nothing in the alternative answer
        resembles it.
        """
        reference = list(reference)
        candidate = list(candidate)
        if not reference:
            return np.zeros(0, dtype=np.float32)
        if not candidate:
            # An empty answer means every sentence was lost. That is a real
            # measurement, not a missing one.
            return np.zeros(len(reference), dtype=np.float32)

        semantic = self._best(self.embedder, reference, candidate)
        lexical = self._best(self.lexical, reference, candidate)
        return np.clip(np.maximum(semantic, lexical), 0.0, 1.0)

    @staticmethod
    def _best(
        embedder: Embedder, reference: Sequence[str], candidate: Sequence[str]
    ) -> np.ndarray:
        similarity = cosine(
            embedder.embed(reference), embedder.embed(candidate)
        )
        return similarity.max(axis=1).astype(np.float32)

    def rank_by_similarity(
        self, sentence: str, texts: Sequence[str]
    ) -> list[tuple[int, float]]:
        """Order ``texts`` by how much they look like ``sentence``.

        Used to decide which chunks are worth testing together when
        leave-one-out came back empty -- searching every subset is
        exponential, and the chunks that resemble the claim are where a
        redundant pair is actually likely to be.
        """
        if not texts:
            return []
        semantic = cosine(
            self.embedder.embed([sentence]), self.embedder.embed(list(texts))
        )[0]
        lexical = cosine(
            self.lexical.embed([sentence]), self.lexical.embed(list(texts))
        )[0]
        scores = np.maximum(semantic, lexical)
        return sorted(
            ((index, float(score)) for index, score in enumerate(scores)),
            key=lambda pair: -pair[1],
        )
