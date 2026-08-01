"""Embeddings, behind an interface small enough to substitute.

Two very different jobs in glasshouse use vectors, and conflating them is a
mistake worth naming:

*Retrieval* needs semantic similarity -- "vendor delays" should find a passage
about "supplier scheduling problems". That genuinely needs a trained model.

*Survival scoring* needs near-duplicate detection -- did this sentence, or a
lightly reworded version of it, come back when we withheld a chunk? That is a
much easier problem, and a purely local character-n-gram model does it well.

Keeping them behind one interface means the demo can run with no API key at
all, the tests can run with no network, and the live path can use a real model,
without any of the code in between knowing which is in play.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    """Turn text into unit vectors."""

    name: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dimensions)`` array of unit vectors."""


def identity(embedder: Embedder) -> str:
    """The underlying model identity, without operational wrappers."""
    inner = getattr(embedder, "inner", None)
    return identity(inner) if inner is not None else embedder.name


@dataclass(frozen=True)
class EmbedderConfig:
    """User-facing selection for one embedding role."""

    provider: str = "auto"
    model: str = "text-embedding-3-small"
    dimensions: int = 512
    cache_path: Path | None = None
    offline: bool = False
    endpoint: str | None = None


def create_embedder(config: EmbedderConfig) -> Embedder:
    """Resolve an explicit provider, never silently downgrading semantics."""
    provider = config.provider.lower()
    if provider not in {"auto", "openai", "lexical"}:
        raise ValueError("embedding provider must be auto, openai, or lexical")
    if config.offline and provider == "openai":
        raise RuntimeError("--offline cannot be combined with --embedding-provider openai")
    if provider == "auto":
        provider = "lexical" if config.offline or not os.environ.get("OPENAI_API_KEY") else "openai"
    if provider == "openai":
        inner: Embedder = OpenAIEmbedder(
            model=config.model,
            dimensions=config.dimensions,
            endpoint=config.endpoint,
        )
    else:
        inner = NgramEmbedder(dimensions=config.dimensions)
    return CachingEmbedder(inner, config.cache_path)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of ``a`` and rows of ``b``.

    Inputs are expected to be unit vectors already; this is a matrix product
    with a defensive renormalisation for anything that slipped through.
    """
    a = _unit(np.atleast_2d(a))
    b = _unit(np.atleast_2d(b))
    return a @ b.T


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ---------------------------------------------------------------------------
# Local, dependency-free
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


class NgramEmbedder:
    """Hashed character n-grams. No model, no network, no key.

    This is a lexical model wearing a vector's clothes. It cannot tell you that
    "car" and "automobile" are related, so it is a poor retriever. It is very
    good at "is this the same sentence, slightly reworded", which is the
    question survival scoring actually asks, and it is completely deterministic
    -- so the test suite gets identical numbers on every machine.
    """

    name = "ngram-local"

    def __init__(self, dimensions: int = 512, sizes: Sequence[int] = (3, 4, 5)):
        self.dimensions = dimensions
        self.sizes = tuple(sizes)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for gram, weight in self._grams(text).items():
                bucket = (
                    int.from_bytes(
                        hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest(),
                        "big",
                    )
                    % self.dimensions
                )
                out[row, bucket] += weight
        return _unit(out)

    def _grams(self, text: str) -> dict[str, float]:
        normalised = " ".join(_TOKEN.findall(text.lower()))
        grams: dict[str, float] = {}
        for size in self.sizes:
            padded = " %s " % normalised
            for i in range(max(0, len(padded) - size + 1)):
                gram = padded[i : i + size]
                grams[gram] = grams.get(gram, 0.0) + 1.0
        # Sub-linear damping, so one repeated word cannot dominate a sentence.
        return {g: 1.0 + math.log(c) for g, c in grams.items()}


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class CachingEmbedder:
    """Memoises an embedder, optionally onto disk.

    Ablation embeds the same sentences over and over -- every leave-one-out
    answer repeats most of the full answer verbatim. Without this the vector
    bill is quadratic in the number of runs for no new information.
    """

    def __init__(self, inner: Embedder, path: Path | None = None):
        self.inner = inner
        self.name = "cached(%s)" % inner.name
        self.dimensions = inner.dimensions
        self.path = Path(path) if path else None
        self._memory: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        if self.path and self.path.exists():
            self._load()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self._memory]
        if missing:
            fresh = self.inner.embed(missing)
            for text, vector in zip(missing, fresh):
                self._memory[self._key(text)] = vector
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        return np.stack([self._memory[self._key(t)] for t in texts])

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _load(self) -> None:
        payload = json.loads(self.path.read_text())
        if (
            payload.get("dimensions") != self.dimensions
            or payload.get("model") != self.inner.name
        ):
            return  # A cache from a different model is not usable.
        for key, values in payload["vectors"].items():
            self._memory[key] = np.asarray(values, dtype=np.float32)

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "model": self.inner.name,
                    "dimensions": self.dimensions,
                    "vectors": {k: v.tolist() for k, v in self._memory.items()},
                }
            )
        )


class FrozenEmbedder:
    """Replays recorded vectors and refuses to invent new ones.

    This is what powers the offline demo. If a visitor types a question the
    recording does not cover, the honest answer is "this demo cannot do that",
    not a silently degraded vector from a different model.
    """

    def __init__(self, vectors: dict[str, list[float]], dimensions: int, name: str):
        self._vectors = {k: np.asarray(v, dtype=np.float32) for k, v in vectors.items()}
        self.dimensions = dimensions
        self.name = "frozen(%s)" % name

    @classmethod
    def load(cls, path: Path) -> "FrozenEmbedder":
        payload = json.loads(Path(path).read_text())
        return cls(
            payload["vectors"], payload["dimensions"], payload.get("model", "unknown")
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
            if key not in self._vectors:
                raise MissingRecording(
                    "no recorded embedding for %r -- this corpus is a fixed "
                    "recording, so it can only answer the questions it was "
                    "recorded with" % _clip(text)
                )
            rows.append(self._vectors[key])
        return np.stack(rows)


class MissingRecording(KeyError):
    """Asked to replay something that was never recorded."""

    def __str__(self) -> str:  # KeyError repr adds quotes that hurt readability
        return self.args[0]


def _clip(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Remote
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    """``text-embedding-3-*`` over plain HTTP.

    Deliberately not using the SDK: one POST with three fields does not justify
    a dependency, and the absence keeps ``pip install glasshouse`` small.
    """

    name = "openai"
    ENDPOINT = "https://api.openai.com/v1/embeddings"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 512,
        api_key: str | None = None,
        timeout: float = 30.0,
        endpoint: str | None = None,
        transport: Any = None,
    ):
        self.model = model
        self.name = "openai:%s" % model
        self.dimensions = dimensions
        self.timeout = timeout
        self.endpoint = endpoint or self.ENDPOINT
        self.transport = transport
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Run the demo instead "
                "(`glasshouse serve --demo`), which needs no keys."
            )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import httpx

        vectors: list[list[float]] = []
        # The endpoint accepts batches; keep them modest so one oversized
        # request cannot fail an entire ablation sweep.
        for start in range(0, len(texts), 96):
            batch = list(texts[start : start + 96])
            with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint,
                    headers={"Authorization": "Bearer %s" % self._api_key},
                    json={
                        "model": self.model,
                        "input": batch,
                        "dimensions": self.dimensions,
                    },
                )
            response.raise_for_status()
            payload = response.json()
            ordered = sorted(payload["data"], key=lambda d: d["index"])
            vectors.extend(item["embedding"] for item in ordered)
        return _unit(np.asarray(vectors, dtype=np.float32))
