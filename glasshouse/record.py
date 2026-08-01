"""Build the committed demo recording.

Runs the real pipeline against a real model, capturing every request/response
pair and every embedding into files the repository ships. A visitor with no API
key then gets the genuine interface driven by genuine model output, and the
test suite gets a deterministic fixture made of real answers rather than of
whatever a stub was written to return.

Everything the demo needs is recorded here, including the closed-book runs and
the coalition runs -- which is why the recording is built by running the actual
analysis rather than by asking the model a list of questions.
"""

from __future__ import annotations

import json
from pathlib import Path

from .ablate import AblationPolicy
from .cassette import Cassette
from .corpus import ChunkingPolicy, chunk_all, load_documents
from .embed import CachingEmbedder, NgramEmbedder
from .index import HybridIndex, RetrievalPolicy
from .llm import AnthropicLLM, DEFAULT_MODEL, RecordingLLM
from .pipeline import Lab
from .similarity import Matcher

SOURCE = (
    Path(__file__).resolve().parent.parent / "demo" / "cormorant" / "source"
)

CHUNKING = ChunkingPolicy(target_words=90)
RETRIEVAL = RetrievalPolicy(top_k=6)
ABLATION = AblationPolicy(max_runs=28)


async def record_demo(out: Path, model: str | None = None, source: Path | None = None) -> int:
    out = Path(out)
    source = Path(source or SOURCE)
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("no demo source at %s" % manifest_path)

    spec = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = load_documents(source / "corpus")
    questions = spec["questions"]

    embedder = CachingEmbedder(NgramEmbedder())
    cassette = Cassette(
        about={
            "what": "Real responses from a live model, captured so the demo "
            "needs no API key.",
            "model": model or DEFAULT_MODEL,
            "questions": questions,
        }
    )
    llm = RecordingLLM(AnthropicLLM(), cassette)

    chunks = chunk_all(documents, CHUNKING)
    lab = Lab(
        index=HybridIndex(chunks, embedder),
        llm=llm,
        matcher=Matcher(embedder),
        retrieval=RETRIEVAL,
        ablation=AblationPolicy(
            support_threshold=ABLATION.support_threshold,
            memory_threshold=ABLATION.memory_threshold,
            max_runs=ABLATION.max_runs,
            model=model or DEFAULT_MODEL,
        ),
        documents=tuple(documents),
    )

    probes_path = source / "probes.json"
    probes = (
        json.loads(probes_path.read_text(encoding="utf-8"))["probes"]
        if probes_path.exists()
        else []
    )
    labels_path = source / "labels.json"
    labels = (
        json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
        if labels_path.exists()
        else []
    )
    # The probe questions are recorded too, so the attribution evaluation --
    # the one with hand-written ground truth -- reproduces without a key.
    to_record = list(questions) + [p["question"] for p in probes]

    spent = 0.0
    baselines = {}
    for question in to_record:
        report = await lab.ask(question)
        baselines[question] = report
        spent += report.usage.cost_usd
        print(
            "  %-58s %d/%d grounded  %2d runs  $%.4f"
            % (
                _clip(question, 58),
                report.grounded_count,
                len(report.checkable),
                len(report.runs),
                report.usage.cost_usd,
            )
        )

    # Record the independent falsification run as well: for each grounded
    # claim, remove every document it was credited to and ask again. The eval
    # can then be reproduced from the cassette without an API key.
    from .evaluate import counterfactual_suite

    counterfactual = await counterfactual_suite(
        lab, documents, questions, baselines=baselines
    )

    # Embed the evaluation's injection sentences as well. They are never sent
    # to the model, but the injection suite has to embed them to score them,
    # and the frozen embedder in the demo will (correctly) refuse anything it
    # was not given. Recording them is what lets a visitor who cloned the
    # repository reproduce the README's numbers with no API key.
    embedder.embed([label["sentence"] for label in labels])

    out.mkdir(parents=True, exist_ok=True)
    _write_corpus(out / "corpus.jsonl", documents)
    cassette.save(out / "cassette.json")
    embedder.path = out / "vectors.json"
    embedder.save()
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "title": spec["title"],
                "blurb": spec["blurb"],
                "questions": questions,
                "chunking": {"target_words": CHUNKING.target_words},
                "retrieval": {"top_k": RETRIEVAL.top_k},
                "ablation": {"max_runs": ABLATION.max_runs},
                "recorded_model": model or DEFAULT_MODEL,
                "retrieval_embedder": embedder.inner.name,
                "survival_embedder": embedder.inner.name,
                "probes": probes,
                "labels": labels,
                "counterfactual": counterfactual.report(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nrecorded %d question(s), %d model responses, %d vectors -- "
        "$%.4f baseline/probe cost (counterfactual calls not included)"
        % (len(to_record), len(cassette), len(embedder._memory), spent)
    )
    return 0


def _write_corpus(path: Path, documents) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                {"doc_id": d.doc_id, "title": d.title, "text": d.text},
                ensure_ascii=False,
            )
            for d in documents
        )
        + "\n",
        encoding="utf-8",
    )


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
