"""Corpus in, report out.

This is the only module that knows the whole shape of the thing: chunk, index,
retrieve, generate, ablate, score. Everything it uses is an interface, so the
same pipeline serves the live path, the recorded demo, and the eval harness
without a branch anywhere in it.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .ablate import AblationPolicy, Ablator
from .corpus import (
    ChunkingPolicy,
    chunk_all,
    load_documents,
    validate_chunk_ids,
    validate_documents,
)
from .embed import CachingEmbedder, Embedder, FrozenEmbedder, NgramEmbedder, identity
from .events import Emitter, emit
from .index import HybridIndex, RetrievalPolicy
from .llm import LLM, ReplayLLM
from .models import Document, Report
from .serialize import retrieved_json
from .similarity import Matcher


@dataclass
class Lab:
    """A corpus, a model, and the settings to interrogate them with."""

    index: HybridIndex
    llm: LLM
    matcher: Matcher
    retrieval: RetrievalPolicy = field(default_factory=RetrievalPolicy)
    ablation: AblationPolicy = field(default_factory=AblationPolicy)
    documents: tuple[Document, ...] = ()

    async def ask(self, question: str, emitter: Emitter | None = None) -> Report:
        question = question.strip()
        if not question:
            raise ValueError("ask something")

        await emit(emitter, "question", question=question)

        # Embedding may perform synchronous HTTP. Keep that work off the event
        # loop so another request can continue streaming while it is in flight.
        retrieved, projection = await asyncio.to_thread(
            self._retrieve, question
        )
        await emit(
            emitter,
            "retrieved",
            chunks=[retrieved_json(r) for r in retrieved],
            projection=projection,
        )

        if not retrieved:
            await emit(emitter, "empty", reason="nothing in the corpus matched")

        ablator = Ablator(self.llm, self.matcher, self.ablation, emitter)
        report = await ablator.analyse(question, retrieved)
        report = replace(report, metadata=self.metadata)

        await emit(
            emitter,
            "done",
            runs=len(report.runs),
            cost_usd=round(report.usage.cost_usd, 6),
            input_tokens=report.usage.input_tokens,
            output_tokens=report.usage.output_tokens,
            elapsed_s=round(report.elapsed_s, 3),
            truncated=report.truncated,
        )
        return report

    def _retrieve(self, question: str):
        retrieved = self.index.search(question, self.retrieval)
        return retrieved, self.projection(question)

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "retrieval_embedder": identity(self.index.embedder),
            "survival_embedder": identity(self.matcher.embedder),
        }

    def projection(self, question: str | None = None) -> dict:
        """A 2-D view of the retrieved region of embedding space, for the UI.

        Principal components of the chunk vectors, which is the honest cheap
        option: it is a linear projection, so a reader can be told exactly what
        they are looking at, and unlike t-SNE or UMAP it will not invent
        clusters that are not in the data. The variance it keeps is reported
        alongside so nobody over-reads the picture.
        """
        vectors = self.index.vectors
        labels = [c.chunk_id for c in self.index.chunks]
        query_vector = None
        if question:
            query_vector = self.index.embedder.embed([question])

        stack = vectors if query_vector is None else np.vstack([vectors, query_vector])
        centred = stack - stack.mean(axis=0)
        # Full SVD on a matrix this small is instant, and avoids the extra
        # dependency a dedicated PCA would bring.
        _, singular, components = np.linalg.svd(centred, full_matrices=False)
        coords = centred @ components[:2].T

        total = float((singular**2).sum()) or 1.0
        explained = float((singular[:2] ** 2).sum() / total)

        return {
            "explained_variance": round(explained, 4),
            "chunks": [
                {"chunk_id": label, "x": float(coords[i, 0]), "y": float(coords[i, 1])}
                for i, label in enumerate(labels)
            ],
            "query": (
                {"x": float(coords[-1, 0]), "y": float(coords[-1, 1])}
                if query_vector is not None
                else None
            ),
        }


def build(
    documents: Sequence[Document],
    llm: LLM,
    embedder: Embedder | None = None,
    retrieval_embedder: Embedder | None = None,
    survival_embedder: Embedder | None = None,
    chunking: ChunkingPolicy | None = None,
    retrieval: RetrievalPolicy | None = None,
    ablation: AblationPolicy | None = None,
) -> Lab:
    """Assemble a lab from documents."""
    documents = tuple(documents)
    validate_documents(list(documents))
    if embedder is not None and retrieval_embedder is not None:
        raise ValueError("pass embedder or retrieval_embedder, not both")
    retrieval_embedder = retrieval_embedder or embedder or NgramEmbedder()
    if not isinstance(retrieval_embedder, CachingEmbedder):
        retrieval_embedder = CachingEmbedder(retrieval_embedder)
    survival_embedder = survival_embedder or NgramEmbedder(dimensions=1024)
    if not isinstance(survival_embedder, CachingEmbedder):
        survival_embedder = CachingEmbedder(survival_embedder)
    chunks = chunk_all(list(documents), chunking)
    validate_chunk_ids(chunks)
    if not chunks:
        raise ValueError("the corpus produced no chunks -- are the documents empty?")
    return Lab(
        index=HybridIndex(chunks, retrieval_embedder),
        llm=llm,
        matcher=Matcher(survival_embedder),
        retrieval=retrieval or RetrievalPolicy(),
        ablation=ablation or AblationPolicy(),
        documents=tuple(documents),
    )


def from_path(path: Path, llm: LLM, **kwargs) -> Lab:
    return build(load_documents(Path(path)), llm, **kwargs)


# ---------------------------------------------------------------------------
# The recorded demo
# ---------------------------------------------------------------------------

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo" / "cormorant"


@dataclass(frozen=True)
class Demo:
    """A lab that runs entirely from a recording."""

    lab: Lab
    questions: tuple[str, ...]
    title: str
    blurb: str
    #: Ground truth for the attribution evaluation; see
    #: demo/cormorant/source/probes.json.
    probes: tuple[dict, ...] = ()
    #: Independent sentence labels used by the injection/classification eval.
    labels: tuple[dict, ...] = ()


def load_demo(directory: Path | None = None, delay: float = 0.35) -> Demo:
    """Load the committed recording: corpus, vectors, and model responses.

    ``delay`` is artificial latency. It is there because the demo's whole
    purpose is to show the ablation resolving run by run, and a replay that
    completes in four milliseconds shows nothing at all.
    """
    directory = Path(directory or DEMO_DIR)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "no demo recording at %s -- run `glasshouse record` to make one" % directory
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    documents = load_documents(directory / "corpus.jsonl")
    validate_documents(documents)
    embedder = FrozenEmbedder.load(directory / "vectors.json")
    llm = ReplayLLM.load(directory / "cassette.json", delay=delay)

    chunks = chunk_all(documents, ChunkingPolicy(**manifest.get("chunking", {})))
    validate_chunk_ids(chunks)
    lab = Lab(
        index=HybridIndex(chunks, embedder),
        llm=llm,
        matcher=Matcher(embedder),
        retrieval=RetrievalPolicy(**manifest.get("retrieval", {})),
        ablation=AblationPolicy(**manifest.get("ablation", {})),
        documents=tuple(documents),
    )
    return Demo(
        lab=lab,
        questions=tuple(manifest["questions"]),
        title=manifest.get("title", "demo corpus"),
        blurb=manifest.get("blurb", ""),
        probes=tuple(manifest.get("probes", ())),
        labels=tuple(manifest.get("labels", ())),
    )
