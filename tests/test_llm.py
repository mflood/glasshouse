import json

import pytest

from glasshouse.cassette import Cassette, MissingRecording, fingerprint
from glasshouse.embed import (
    CachingEmbedder,
    FrozenEmbedder,
    NgramEmbedder,
    cosine,
)
from glasshouse.embed import MissingRecording as MissingVector
from glasshouse.llm import (
    RecordingLLM,
    ReplayLLM,
    Request,
    ScriptedLLM,
    price,
)


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def test_key_ordering_does_not_change_the_fingerprint():
    """Dictionary order is an implementation detail, not part of the request."""
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_a_changed_prompt_is_a_different_fingerprint():
    """An edited template must be a cache miss, not a stale hit."""
    base = Request(system="s", prompt="p")

    assert base.key() != Request(system="s", prompt="p ").key()
    assert base.key() != Request(system="s2", prompt="p").key()
    assert base.key() != Request(system="s", prompt="p", max_tokens=1).key()


# ---------------------------------------------------------------------------
# Cassettes
# ---------------------------------------------------------------------------


async def test_a_recording_replays_exactly(tmp_path):
    cassette = Cassette(path=tmp_path / "c.json")
    inner = ScriptedLLM(lambda r: "the recorded answer")
    request = Request(system="s", prompt="p")

    await RecordingLLM(inner, cassette).complete(request)
    cassette.save()

    replayed = await ReplayLLM.load(tmp_path / "c.json").complete(request)
    assert replayed.text == "the recorded answer"


async def test_a_replay_never_calls_the_model(tmp_path):
    cassette = Cassette(path=tmp_path / "c.json")
    inner = ScriptedLLM(lambda r: "recorded")
    request = Request(system="s", prompt="p")
    await RecordingLLM(inner, cassette).complete(request)

    await ReplayLLM(cassette).complete(request)

    assert len(inner.calls) == 1


async def test_recording_the_same_request_twice_calls_the_model_once():
    """Ablation repeats requests constantly; the second is a lookup."""
    cassette = Cassette()
    inner = ScriptedLLM(lambda r: "answer")
    recorder = RecordingLLM(inner, cassette)
    request = Request(system="s", prompt="p")

    await recorder.complete(request)
    await recorder.complete(request)

    assert len(inner.calls) == 1


async def test_replaying_something_unrecorded_fails_loudly(tmp_path):
    """Silently degrading would make the demo lie about what it is."""
    replay = ReplayLLM(Cassette())

    with pytest.raises(MissingRecording, match="not in the recording"):
        await replay.complete(Request(system="s", prompt="never recorded"))


async def test_a_replayed_completion_is_marked_cached(tmp_path):
    """So the cost meter can show the demo is free rather than inventing a price."""
    cassette = Cassette()
    await RecordingLLM(ScriptedLLM(lambda r: "x"), cassette).complete(
        Request(system="s", prompt="p")
    )

    completion = await ReplayLLM(cassette).complete(Request(system="s", prompt="p"))

    assert completion.usage.cached


def test_a_saved_cassette_is_readable_json(tmp_path):
    """It is committed to the repo, so a reader should be able to open it."""
    cassette = Cassette(path=tmp_path / "c.json")
    cassette.put("abc", {"text": "hello"}, request={"prompt": "hi"})
    path = cassette.save()

    payload = json.loads(path.read_text())
    assert payload["entries"]["abc"]["text"] == "hello"
    assert payload["entries"]["abc"]["_request"]["prompt"] == "hi"


def test_a_cassette_with_no_path_cannot_be_saved():
    with pytest.raises(ValueError, match="no path"):
        Cassette().save()


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def test_a_known_model_is_priced():
    assert price("claude-haiku-4-5-20251001", 1_000_000, 0) == pytest.approx(1.00)


def test_an_unknown_model_prices_at_zero_rather_than_guessing():
    assert price("some-future-model", 1_000_000, 1_000_000) == 0.0


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------


def test_embeddings_are_unit_vectors():
    import numpy as np

    vectors = NgramEmbedder().embed(["alpha", "beta gamma"])

    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_the_local_embedder_is_deterministic():
    """CI on another machine must get the same numbers."""
    import numpy as np

    a = NgramEmbedder().embed(["the rollout slipped"])
    b = NgramEmbedder().embed(["the rollout slipped"])

    assert np.array_equal(a, b)


def test_similar_text_scores_higher_than_unrelated_text():
    embedder = NgramEmbedder()
    vectors = embedder.embed(
        [
            "the rollout slipped by six weeks",
            "the rollout slipped by six weeks in total",
            "unrelated commentary about marine biology",
        ]
    )
    similarity = cosine(vectors[:1], vectors[1:])[0]

    assert similarity[0] > similarity[1]


def test_the_cache_avoids_re_embedding():
    inner = NgramEmbedder()
    cached = CachingEmbedder(inner)

    cached.embed(["a", "b"])
    cached.embed(["a", "b", "c"])

    assert cached.misses == 3
    assert cached.hits == 2


def test_the_cache_returns_vectors_in_the_order_asked_for():
    import numpy as np

    cached = CachingEmbedder(NgramEmbedder())
    first = cached.embed(["alpha", "beta"])
    second = cached.embed(["beta", "alpha"])

    assert np.array_equal(first[0], second[1])
    assert np.array_equal(first[1], second[0])


def test_a_cache_round_trips_through_disk(tmp_path):
    import numpy as np

    path = tmp_path / "vectors.json"
    first = CachingEmbedder(NgramEmbedder(), path)
    expected = first.embed(["alpha"])
    first.save()

    second = CachingEmbedder(NgramEmbedder(), path)
    assert np.allclose(second.embed(["alpha"]), expected)
    assert second.misses == 0


def test_a_cache_from_a_different_model_is_ignored(tmp_path):
    """Mixing vector spaces would corrupt every similarity silently."""
    path = tmp_path / "vectors.json"
    CachingEmbedder(NgramEmbedder(dimensions=64), path).embed(["alpha"])
    CachingEmbedder(NgramEmbedder(dimensions=64), path).save()

    reloaded = CachingEmbedder(NgramEmbedder(dimensions=128), path)
    reloaded.embed(["alpha"])

    assert reloaded.misses == 1


def test_a_frozen_embedder_refuses_unrecorded_text(tmp_path):
    frozen = FrozenEmbedder({}, dimensions=8, name="test")

    with pytest.raises(MissingVector, match="fixed recording"):
        frozen.embed(["something nobody recorded"])
