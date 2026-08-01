import asyncio
import json
import time

import httpx
import numpy as np
import pytest

from glasshouse import Document, build
from glasshouse.corpus import chunk_all
from glasshouse.embed import (
    EmbedderConfig,
    NgramEmbedder,
    OpenAIEmbedder,
    create_embedder,
    identity,
)
from glasshouse.llm import ScriptedLLM
from glasshouse.index import HybridIndex, RetrievalPolicy


def test_auto_provider_is_explicitly_lexical_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    embedder = create_embedder(EmbedderConfig())

    assert identity(embedder) == "ngram-local"


def test_explicit_semantic_provider_never_silently_downgrades(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        create_embedder(EmbedderConfig(provider="openai"))


def test_offline_rejects_a_remote_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    with pytest.raises(RuntimeError, match="offline"):
        create_embedder(EmbedderConfig(provider="openai", offline=True))


def test_retrieval_and_survival_models_are_independent():
    lab = build(
        [Document("one", "One", "A document")],
        ScriptedLLM(lambda request: ""),
        retrieval_embedder=NgramEmbedder(dimensions=64),
        survival_embedder=NgramEmbedder(dimensions=128),
    )

    assert lab.index.embedder.dimensions == 64
    assert lab.matcher.embedder.dimensions == 128
    assert lab.metadata == {
        "retrieval_embedder": "ngram-local",
        "survival_embedder": "ngram-local",
    }


def test_semantic_retrieval_finds_a_paraphrase_without_token_overlap():
    class ParaphraseEmbedder:
        name = "fixture-semantic"
        dimensions = 2

        def embed(self, texts):
            rows = []
            for text in texts:
                lowered = text.lower()
                semantic = any(
                    phrase in lowered
                    for phrase in ("supplier scheduling", "vendor delays")
                )
                rows.append([1.0, 0.0] if semantic else [0.0, 1.0])
            return np.asarray(rows, dtype=np.float32)

    chunks = chunk_all(
        [
            Document("answer", "Answer", "Supplier scheduling became unreliable."),
            Document("distractor", "Distractor", "Invoices were paid promptly."),
        ]
    )
    index = HybridIndex(chunks, ParaphraseEmbedder())

    results = index.search(
        "What caused the vendor delays?",
        RetrievalPolicy(top_k=1, min_z=0.0, neighbor_window=0),
    )

    assert results[0].chunk.doc_id == "answer"


def test_openai_embedder_works_with_a_compatible_remote_server(monkeypatch):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        data = [
            {"index": i, "embedding": [1.0, float(i + 1)]}
            for i, _ in enumerate(payload["input"])
        ]
        return httpx.Response(200, json={"data": data})

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    embedder = OpenAIEmbedder(
        model="semantic-test",
        dimensions=2,
        endpoint="https://embedding.test/v1/embeddings",
        transport=httpx.MockTransport(handler),
    )
    vectors = embedder.embed(["supplier scheduling", "vendor delays"])

    assert requests[0]["model"] == "semantic-test"
    assert requests[0]["dimensions"] == 2
    assert vectors.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)


@pytest.mark.asyncio
async def test_synchronous_embedding_does_not_block_the_event_loop():
    class SlowEmbedder(NgramEmbedder):
        name = "slow-semantic-test"

        def embed(self, texts):
            time.sleep(0.08)
            return super().embed(texts)

    lab = build(
        [Document("one", "One", "supplier scheduling problems")],
        ScriptedLLM(lambda request: ""),
        retrieval_embedder=SlowEmbedder(dimensions=32),
    )
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(heartbeat())
    try:
        await lab.ask("vendor delays")
    finally:
        task.cancel()

    assert ticks >= 5
