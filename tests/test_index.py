import numpy as np
import pytest

from glasshouse import ChunkingPolicy, Document, HybridIndex, RetrievalPolicy, chunk_all
from glasshouse.embed import NgramEmbedder
from glasshouse.index import BM25, _reciprocal_rank_fusion, tokenize


@pytest.fixture
def index(documents):
    chunks = chunk_all(documents, ChunkingPolicy(target_words=18))
    return HybridIndex(chunks, NgramEmbedder())


def test_tokenize_drops_stopwords_and_case():
    assert tokenize("The Vendor and the Firmware") == ["vendor", "firmware"]


def test_bm25_prefers_the_document_that_says_the_word():
    bm25 = BM25(["the vendor was late", "the budget was fine", "nothing here"])

    assert int(np.argmax(bm25.scores("vendor"))) == 0


def test_bm25_idf_is_never_negative():
    """A term in every document should be uninformative, not harmful.

    Without the +0.5 smoothing its idf goes negative and a common word
    actively lowers the score of documents that contain it.
    """
    bm25 = BM25(["shared term here", "shared term there", "shared term everywhere"])

    assert all(value >= 0 for value in bm25._idf.values())
    assert (bm25.scores("shared") >= 0).all()


def test_bm25_scores_nothing_for_an_unknown_term():
    bm25 = BM25(["alpha beta", "gamma delta"])

    assert bm25.scores("epsilon").sum() == 0


def test_rrf_ranks_a_consensus_pick_above_either_leader():
    """Something ranked second by both beats something ranked first by one."""
    fused = _reciprocal_rank_fusion([0, 2], [1, 2])

    assert max(fused, key=fused.get) == 2


def test_search_finds_the_relevant_chunk(index):
    results = index.search("why did the schedule slip?")

    assert results
    assert any("slip" in r.chunk.text.lower() for r in results)


def test_search_reports_which_retriever_found_each_chunk(index):
    """The UI shows this, and it is how a reader sees hybrid retrieval working."""
    results = index.search("firmware revision")

    assert any(r.lexical_rank is not None for r in results)
    assert any(r.dense_rank is not None for r in results)


def test_top_k_is_respected(index):
    assert len(index.search("meridian", RetrievalPolicy(top_k=2))) == 2


def test_the_best_match_comes_first(index):
    """Results are in MMR selection order, not score order.

    Only the first pick is guaranteed to be the top-scoring candidate; after
    that diversity deliberately reorders, which is the whole point of it.
    """
    results = index.search("meridian budget")

    assert results[0].score == max(r.score for r in results)


def test_diversity_breaks_up_near_duplicates():
    """Redundant evidence is what defeats leave-one-out ablation.

    Four near-identical chunks plus one different one: with diversity off the
    top-2 are duplicates of each other, and neither can be shown to matter.
    """
    docs = [
        Document(doc_id="dup%d" % i, title="dup", text="The rollout slipped six weeks.")
        for i in range(4)
    ]
    docs.append(
        Document(doc_id="other", title="other", text="The rollout slipped six weeks badly.")
    )
    index = HybridIndex(chunk_all(docs), NgramEmbedder())

    greedy = index.search("rollout slip", RetrievalPolicy(top_k=2, diversity=0.0))
    diverse = index.search("rollout slip", RetrievalPolicy(top_k=2, diversity=0.9))

    assert len({r.chunk.text for r in greedy}) == 1
    assert len({r.chunk.text for r in diverse}) == 2


def test_an_empty_corpus_is_rejected():
    with pytest.raises(ValueError, match="empty corpus"):
        HybridIndex([], NgramEmbedder())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"diversity": 1.5},
        {"neighbor_window": -1},
        {"neighbor_decay": 0.0},
        {"neighbor_decay": 1.1},
    ],
)
def test_incoherent_retrieval_policies_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RetrievalPolicy(**kwargs)


@pytest.fixture
def wide_index():
    """Enough chunks for the relative dense floor to be meaningful."""
    docs = [
        Document(
            doc_id="d%d" % i,
            title="doc %d" % i,
            text="Topic %d covers %s in detail across the region."
            % (i, "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()[i]),
        )
        for i in range(10)
    ]
    return HybridIndex(chunk_all(docs), NgramEmbedder())


def test_junk_retrieves_far_less_than_a_real_query(wide_index):
    """Cosine similarity is positive between any two texts.

    Without a floor a nonsense query still retrieves its six nearest chunks
    and they look like sources. The relative floor cuts most of that away --
    it cannot cut all of it, since some chunk is always above the mean, which
    is why deciding a corpus contributed nothing is left to ablation.
    """
    junk = wide_index.search("zzzz qqqq wwww vvvv")
    real = wide_index.search("what does topic 3 cover")

    assert len(junk) < len(real)
    assert len(junk) <= 1


def test_without_the_floor_junk_looks_like_a_normal_result(wide_index):
    """The failure the floor exists to prevent."""
    unfloored = wide_index.search(
        "zzzz qqqq wwww vvvv", RetrievalPolicy(min_z=0.0)
    )

    assert len(unfloored) == 6


def test_a_real_query_still_survives_the_floor(wide_index):
    assert wide_index.search("what does topic 3 cover")


def test_the_floor_is_skipped_for_a_small_corpus(index):
    """With three documents the z-score has too few samples to trust."""
    assert index.search("meridian")


def test_a_match_promotes_later_context_from_the_same_document():
    """The entity can be introduced before the paragraph with the answer."""
    docs = [
        Document(
            "incident",
            "Incident",
            "Cormorant left the support ship. "
            "Telemetry remained nominal during descent. "
            "It surfaced 4.1 nautical miles east of the support ship.",
        ),
        Document("other", "Other", "An unrelated maintenance memo."),
    ]
    chunks = chunk_all(docs, ChunkingPolicy(target_words=5))
    index = HybridIndex(chunks, NgramEmbedder())

    without = index.search(
        "Where did Cormorant end up?",
        RetrievalPolicy(top_k=3, neighbor_window=0),
    )
    with_context = index.search(
        "Where did Cormorant end up?",
        RetrievalPolicy(top_k=3, neighbor_window=2),
    )

    assert not any("4.1 nautical" in r.chunk.text for r in without)
    assert any("4.1 nautical" in r.chunk.text for r in with_context)


def test_neighbor_expansion_keeps_the_top_k_budget():
    docs = [Document("long", "Long", "One. Two. Three. Four. Five. Six.")]
    index = HybridIndex(
        chunk_all(docs, ChunkingPolicy(target_words=1)), NgramEmbedder()
    )

    assert len(index.search("One", RetrievalPolicy(top_k=2))) == 2


def test_neighbor_expansion_never_crosses_document_boundaries():
    docs = [
        Document("alpha", "Alpha", "Target phrase. Alpha neighbor."),
        Document("beta", "Beta", "Beta neighbor. Another beta paragraph."),
    ]
    index = HybridIndex(
        chunk_all(docs, ChunkingPolicy(target_words=2)), NgramEmbedder()
    )

    seed = next(i for i, chunk in enumerate(index.chunks) if chunk.chunk_id == "alpha#0")
    expanded, _ = index._expand_neighbors(
        {seed: 1.0}, RetrievalPolicy(neighbor_window=2)
    )

    assert all(index.chunks[i].doc_id == "alpha" for i in expanded)
