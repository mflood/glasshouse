import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from glasshouse import AblationPolicy, ChunkingPolicy, Document, build
from glasshouse.evaluate import (
    ClassificationSuite,
    Outcome,
    _labels_beside,
    Suite,
    _probes_beside,
    attribution_suite,
    counterfactual_suite,
    run_suite,
)
from glasshouse.llm import Request, ScriptedLLM
from glasshouse.models import (
    Chunk,
    ClaimVerdict,
    Report,
    Retrieved,
    Support,
    Verdict,
)


@pytest.mark.asyncio
async def test_attribution_names_the_document_that_uniquely_contains_the_fact():
    docs = [
        Document("alpha", "Alpha log", "The launch code was cobalt seven."),
        Document("beta", "Beta log", "The weather was clear all afternoon."),
    ]

    def respond(request: Request) -> str:
        return (
            "The launch code was cobalt seven."
            if "cobalt seven" in request.prompt
            else "Nothing in the context answers that."
        )

    lab = build(
        docs,
        ScriptedLLM(respond),
        chunking=ChunkingPolicy(target_words=40),
        ablation=AblationPolicy(max_runs=8),
    )
    suite = await attribution_suite(
        lab,
        [{"question": "What was the launch code?", "document": "alpha"}],
    )

    assert suite.rate == 1.0
    assert suite.outcomes[0].passed
    assert "alpha" in suite.outcomes[0].note


@pytest.mark.asyncio
async def test_attribution_fails_when_no_claim_is_grounded():
    docs = [Document("alpha", "Alpha log", "The launch code was cobalt seven.")]
    lab = build(
        docs,
        ScriptedLLM(lambda _request: "The launch code was vermilion nine."),
        chunking=ChunkingPolicy(target_words=40),
        ablation=AblationPolicy(max_runs=6),
    )

    suite = await attribution_suite(
        lab,
        [{"question": "What was the launch code?", "document": "alpha"}],
    )

    assert suite.rate == 0.0
    assert suite.outcomes[0].note == "nothing was grounded at all"


def test_probe_key_loads_beside_a_corpus(tmp_path):
    payload = {"probes": [{"question": "q", "document": "alpha"}]}
    (tmp_path / "probes.json").write_text(json.dumps(payload))

    assert _probes_beside(tmp_path) == payload["probes"]


def test_missing_probe_key_is_empty(tmp_path):
    assert _probes_beside(tmp_path) == []


def test_independent_labels_load_beside_a_corpus(tmp_path):
    payload = {"labels": [{"question": "q", "sentence": "s", "grounded": False}]}
    (tmp_path / "labels.json").write_text(json.dumps(payload))

    assert _labels_beside(tmp_path) == payload["labels"]


def test_mark_everything_grounded_fails_false_positive_control():
    suite = ClassificationSuite()
    suite.expected = [True, False]
    suite.predicted = [True, True]
    suite.outcomes = [Outcome("q", "positive", True), Outcome("q", "negative", False)]

    assert suite.false_positive_rate == 1.0
    assert suite.precision == 0.5
    assert suite.report()["false_positive"] == 1


def test_mark_nothing_grounded_fails_recall_control():
    suite = ClassificationSuite()
    suite.expected = [True, False]
    suite.predicted = [False, False]
    suite.outcomes = [Outcome("q", "positive", False), Outcome("q", "negative", True)]

    assert suite.recall == 0.0
    assert suite.false_negative_rate == 1.0
    assert suite.report()["false_negative"] == 1


@pytest.mark.asyncio
async def test_counterfactual_can_reuse_a_baseline_instead_of_paying_twice():
    docs = [
        Document("alpha", "Alpha", "The launch code was cobalt seven."),
        Document("beta", "Beta", "The weather was clear."),
    ]

    def respond(request: Request) -> str:
        return (
            "The launch code was cobalt seven."
            if "cobalt seven" in request.prompt
            else "Nothing answers the question."
        )

    llm = ScriptedLLM(respond)
    lab = build(
        docs,
        llm,
        chunking=ChunkingPolicy(target_words=40),
        ablation=AblationPolicy(max_runs=8),
    )
    question = "What was the launch code?"
    baseline = await lab.ask(question)
    calls_after_baseline = len(llm.calls)

    suite = await counterfactual_suite(
        lab, docs, [question], baselines={question: baseline}
    )

    assert suite.outcomes
    assert suite.rate == 1.0
    # One reduced-corpus analysis was added; the baseline was not repeated.
    assert len(llm.calls) > calls_after_baseline


@pytest.mark.asyncio
async def test_counterfactual_preserves_both_embedding_roles(monkeypatch):
    class ExactEmbedder:
        name = "exact-retrieval"
        dimensions = 2

        def embed(self, texts):
            import numpy as np

            return np.asarray(
                [[1.0, 0.0] if "cobalt" in text.lower() else [0.0, 1.0] for text in texts],
                dtype=np.float32,
            )

    class SurvivalEmbedder(ExactEmbedder):
        name = "independent-survival"

    docs = [
        Document("alpha", "Alpha", "The launch code was cobalt seven."),
        Document("beta", "Beta", "The weather was clear."),
    ]
    llm = ScriptedLLM(
        lambda request: (
            "The launch code was cobalt seven."
            if "cobalt seven" in request.prompt
            else "Nothing answers the question."
        )
    )
    lab = build(
        docs,
        llm,
        retrieval_embedder=ExactEmbedder(),
        survival_embedder=SurvivalEmbedder(),
        chunking=ChunkingPolicy(target_words=40),
        ablation=AblationPolicy(max_runs=8),
    )
    question = "What was the launch code?"
    baseline = await lab.ask(question)
    passed = []

    def capture_build(*args, **kwargs):
        passed.append(kwargs)
        return build(*args, **kwargs)

    monkeypatch.setattr("glasshouse.evaluate.build", capture_build)

    suite = await counterfactual_suite(
        lab, docs, [question], baselines={question: baseline}
    )

    assert suite.outcomes
    assert passed[0]["retrieval_embedder"] is lab.index.embedder
    assert passed[0]["survival_embedder"] is lab.matcher.embedder


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim, regenerated",
    [
        (
            "Revenue grew 3.5% this year.",
            "Revenue grew 3.5% this year. It beat forecasts.",
        ),
        (
            "Dr. Chen approved the launch.",
            "Dr. Chen approved the launch. It proceeded.",
        ),
        (
            "J. R. Hartley approved the launch.",
            "J. R. Hartley approved the launch. It proceeded.",
        ),
        ('She asked, "Is it ready?"', 'She asked, "Is it ready?" Then she waited.'),
        ("Is the launch ready?", "Is the launch ready? It is."),
        ("The launch is ready!", "The launch is ready! It proceeds today."),
    ],
    ids=[
        "decimal",
        "title",
        "initials",
        "quoted-terminator",
        "question",
        "exclamation",
    ],
)
async def test_counterfactual_rejects_surviving_claims_for_all_sentence_punctuation(
    monkeypatch, claim, regenerated
):
    """The old ``split('. ')`` fragmented or joined every case in this table."""
    chunk = Chunk("alpha:0", "alpha", "Alpha", claim, 0, len(claim))
    support = Support("alpha:0", effect=1.0, raw_drop=1.0, credited=True)
    baseline = Report(
        question="What happened?",
        answer=claim,
        retrieved=(Retrieved(chunk, score=1.0),),
        claims=(ClaimVerdict(0, claim, Verdict.GROUNDED, support=(support,)),),
        runs=(),
    )

    class ExactMatcher:
        embedder = object()

        def survival(self, references, candidates):
            import numpy as np

            return np.asarray(
                [float(reference in candidates) for reference in references],
                dtype=np.float32,
            )

    lab = SimpleNamespace(
        index=SimpleNamespace(embedder=object()),
        matcher=ExactMatcher(),
        llm=object(),
        retrieval=object(),
        ablation=object(),
    )

    async def ask(_question):
        return SimpleNamespace(answer=regenerated)

    smaller = SimpleNamespace(ask=ask)
    monkeypatch.setattr("glasshouse.evaluate.build", lambda *_args, **_kwargs: smaller)

    suite = await counterfactual_suite(
        lab,
        [
            Document("alpha", "Alpha", claim),
            Document("beta", "Beta", "Other evidence."),
        ],
        ["What happened?"],
        baselines={"What happened?": baseline},
    )

    assert len(suite.outcomes) == 1
    assert not suite.outcomes[0].passed
    assert suite.outcomes[0].note.endswith("the claim scores 1.00")


@pytest.mark.asyncio
async def test_attribution_is_a_real_cli_branch(monkeypatch, capsys, tmp_path):
    """Regression: the parser accepted attribution but ran thresholds."""
    fake = Suite("attribution")
    fake.outcomes.append(
        __import__("glasshouse.evaluate", fromlist=["Outcome"]).Outcome(
            question="q", detail="alpha", passed=True, note="credited alpha"
        )
    )

    async def evaluate(lab, probes):
        assert probes == [{"question": "q", "document": "alpha"}]
        return fake

    class Demo:
        class Lab:
            documents = ()

        lab = Lab()
        questions = ()
        probes = ({"question": "q", "document": "alpha"},)

    monkeypatch.setattr("glasshouse.pipeline.load_demo", lambda delay=0: Demo())
    monkeypatch.setattr("glasshouse.evaluate.attribution_suite", evaluate)
    args = Namespace(demo=True, corpus=None, suite="attribution", out=tmp_path / "out.json")

    assert await run_suite(args) == 0
    assert "document attribution 100%" in capsys.readouterr().out
    assert json.loads(args.out.read_text())["suite"] == "attribution"
