"""Hybrid retrieval: BM25, dense vectors, and rank fusion.

Both retrievers are here rather than imported because each contributes a
failure mode the other covers. BM25 cannot match "vendor delays" to "supplier
scheduling problems"; dense retrieval routinely misses an exact identifier like
``CVE-2019-14287`` because the embedding smooths it away. Fusing their rankings
is the cheapest large win in retrieval and needs no extra model.

There is one glasshouse-specific reason to care about retrieval quality beyond
answer quality, and it drives the MMR step below: **redundant chunks destroy
attribution**. If two retrieved chunks say the same thing, removing either one
alone changes nothing, and a perfectly grounded claim looks unsupported. See
:mod:`glasshouse.ablate` for the coalition search that cleans up whatever
diversification does not prevent.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .embed import Embedder, cosine
from .models import Chunk, Retrieved

_WORD = re.compile(r"[a-z0-9][a-z0-9'_-]*")

#: Words too common to discriminate. Deliberately short -- an aggressive list
#: throws away exactly the function words that make a phrase query work.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of to in on at by
    for with from as is are was were be been being it its
    """.split()
)

K1 = 1.5
B = 0.75
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


class BM25:
    """Okapi BM25 over a fixed collection."""

    def __init__(self, documents: Sequence[str], k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(d) for d in documents]
        self._lengths = np.asarray([len(t) for t in self._tokens], dtype=np.float32)
        self._avg_length = float(self._lengths.mean()) if len(self._tokens) else 0.0
        self._counts = [Counter(t) for t in self._tokens]

        frequency: Counter = Counter()
        for tokens in self._tokens:
            frequency.update(set(tokens))
        total = len(self._tokens)
        # The +0.5 smoothing keeps the idf of a term appearing in every
        # document at a small positive number rather than a negative one,
        # which would make a common word *reduce* a document's score.
        self._idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        out = np.zeros(len(self._tokens), dtype=np.float32)
        if not terms or not self._avg_length:
            return out
        for index, counts in enumerate(self._counts):
            length = self._lengths[index]
            total = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self._avg_length
                )
                total += self._idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            out[index] = total
        return out


@dataclass(frozen=True)
class RetrievalPolicy:
    """Knobs for the retrieval stage."""

    top_k: int = 6
    #: How many candidates each retriever contributes before fusion.
    candidates: int = 24
    #: 0 keeps pure relevance order; 1 maximises diversity. See the module
    #: docstring for why glasshouse leans toward diversity.
    diversity: float = 0.35
    #: How many standard deviations above the corpus mean a chunk's dense
    #: similarity must sit before it counts as a match on its own. See
    #: :meth:`HybridIndex._dense_candidates` for why this is relative.
    min_z: float = 1.0
    #: Below this many chunks the z-score is computed from too few samples to
    #: mean anything, so the floor is not applied at all.
    min_corpus_for_z: int = 8
    #: Bring nearby chunks from the same document into the candidate pool.
    #: Questions often match an entity in one paragraph and ask for an outcome
    #: two paragraphs later; pure chunk retrieval sees only the entity.
    neighbor_window: int = 2
    #: Rank-fusion score retained per step away from the matching chunk.
    neighbor_decay: float = 0.82

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.diversity <= 1.0:
            raise ValueError("diversity must be between 0 and 1")
        if self.neighbor_window < 0:
            raise ValueError("neighbor_window cannot be negative")
        if not 0.0 < self.neighbor_decay <= 1.0:
            raise ValueError("neighbor_decay must be in (0, 1]")


class HybridIndex:
    """BM25 and dense retrieval over the same chunks, fused by RRF."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder):
        if not chunks:
            raise ValueError("cannot index an empty corpus")
        self.chunks = list(chunks)
        self.embedder = embedder
        self._bm25 = BM25([c.text for c in self.chunks])
        self._vectors = embedder.embed([c.text for c in self.chunks])

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def vectors(self) -> np.ndarray:
        """The chunk embedding matrix, one unit row per chunk."""
        return self._vectors

    def search(
        self, query: str, policy: RetrievalPolicy | None = None
    ) -> list[Retrieved]:
        policy = policy or RetrievalPolicy()

        lexical = self._rank(self._bm25.scores(query), policy.candidates)
        query_vector = self.embedder.embed([query])
        dense = self._dense_candidates(query_vector, policy)

        fused = _reciprocal_rank_fusion(lexical, dense)
        fused, context_promoted = self._expand_neighbors(fused, policy)
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])
        selected = self._diversify(
            [i for i, _ in ordered], query_vector, policy, context_promoted
        )

        lexical_position = {index: rank for rank, index in enumerate(lexical)}
        dense_position = {index: rank for rank, index in enumerate(dense)}
        return [
            Retrieved(
                chunk=self.chunks[index],
                score=float(fused[index]),
                lexical_rank=lexical_position.get(index),
                dense_rank=dense_position.get(index),
            )
            for index in selected
        ]

    def _expand_neighbors(
        self, fused: dict[int, float], policy: RetrievalPolicy
    ) -> tuple[dict[int, float], set[int]]:
        """Promote context around a matching chunk without faking a retriever hit.

        Neighbor chunks get no lexical or dense rank in the returned metadata;
        the UI can therefore distinguish "matched this" from "included for
        context". Expansion happens before MMR, so neighbors still compete for
        the fixed top-k budget rather than silently increasing model cost.
        """
        if not fused or policy.neighbor_window == 0:
            return dict(fused), set()

        expanded = dict(fused)
        context_promoted: set[int] = set()
        by_location = {
            (chunk.doc_id, chunk.ordinal): index
            for index, chunk in enumerate(self.chunks)
        }
        for index, score in list(fused.items()):
            chunk = self.chunks[index]
            for distance in range(1, policy.neighbor_window + 1):
                promoted_score = score * (policy.neighbor_decay**distance)
                for ordinal in (chunk.ordinal - distance, chunk.ordinal + distance):
                    neighbor = by_location.get((chunk.doc_id, ordinal))
                    if (
                        neighbor is not None
                        and promoted_score > expanded.get(neighbor, 0.0)
                    ):
                        expanded[neighbor] = promoted_score
                        context_promoted.add(neighbor)
        return expanded, context_promoted

    def vectors_for(self, chunk_ids: Sequence[str]) -> np.ndarray:
        position = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        return np.stack([self._vectors[position[cid]] for cid in chunk_ids])

    @staticmethod
    def _rank(scores: np.ndarray, limit: int) -> list[int]:
        order = np.argsort(-scores)
        return [int(i) for i in order[:limit] if scores[i] > 0]

    def _dense_candidates(
        self, query_vector: np.ndarray, policy: RetrievalPolicy
    ) -> list[int]:
        """Dense matches that stand out from the corpus, not merely score well.

        A dense retriever has no natural notion of "nothing matched": cosine
        similarity between any two texts is some positive number, so an
        absolute floor has to be tuned per embedding model and silently stops
        working when the model changes. Measured on this corpus, a nonsense
        query scores 0.25 against a chunk while a good query scores 0.29 --
        no constant separates them.

        What does separate them is the *shape* of the score distribution. A
        real query is much closer to a few chunks than to the rest; a nonsense
        query is equally far from everything. So the test is relative: how many
        standard deviations above the corpus mean does this chunk sit.

        Below :attr:`RetrievalPolicy.min_corpus_for_z` chunks the statistic has
        too few samples to be trustworthy and is skipped -- with a handful of
        documents there is little to be confused about, and dropping the floor
        is the failure that leaves evidence out.

        This reduces junk; it cannot eliminate it, because some chunk is always
        above the mean. Deciding that a corpus contributed *nothing* is left to
        ablation, which answers it directly -- see
        :attr:`glasshouse.models.Report.corpus_contributed`.
        """
        similarity = cosine(query_vector, self._vectors)[0].astype(np.float32)
        ranked = self._rank(similarity, policy.candidates)

        if len(self.chunks) < policy.min_corpus_for_z:
            return ranked

        spread = float(similarity.std())
        if spread == 0:
            return ranked
        floor = float(similarity.mean()) + policy.min_z * spread
        return [i for i in ranked if similarity[i] >= floor]

    def _diversify(
        self,
        candidates: list[int],
        query_vector: np.ndarray,
        policy: RetrievalPolicy,
        context_promoted: set[int] | None = None,
    ) -> list[int]:
        """Maximal marginal relevance over the fused ranking.

        Standard MMR, with a non-standard motivation: near-duplicate evidence
        is not merely redundant here, it is actively misleading, because
        leave-one-out ablation cannot see through it.
        """
        if not candidates:
            return []
        if policy.diversity == 0:
            return candidates[: policy.top_k]

        pool = candidates[: max(policy.candidates, policy.top_k)]
        context_promoted = context_promoted or set()
        vectors = self._vectors[pool]
        relevance = cosine(query_vector, vectors)[0]

        chosen: list[int] = [0]
        while len(chosen) < min(policy.top_k, len(pool)):
            remaining = [i for i in range(len(pool)) if i not in chosen]
            redundancy = cosine(vectors[remaining], vectors[chosen]).max(axis=1)
            # Adjacent chunks were promoted precisely because their neighbour
            # matched. Do not make that same relationship disqualify them as
            # redundant; otherwise neighbour expansion can never beat MMR.
            context_mask = np.asarray(
                [pool[i] in context_promoted for i in remaining], dtype=bool
            )
            redundancy[context_mask] *= 0.1
            adjusted = (
                1 - policy.diversity
            ) * relevance[remaining] - policy.diversity * redundancy
            chosen.append(remaining[int(np.argmax(adjusted))])

        return [pool[i] for i in chosen]


def _reciprocal_rank_fusion(*rankings: Sequence[int], k: int = RRF_K) -> dict[int, float]:
    """Combine rankings by position rather than by score.

    Scores from BM25 and cosine similarity live on incomparable scales, and
    normalising them requires assumptions about their distributions that do not
    survive a change of corpus. Ranks are directly comparable, which is why RRF
    keeps beating carefully tuned score blending.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for position, index in enumerate(ranking):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + position + 1)
    return fused
