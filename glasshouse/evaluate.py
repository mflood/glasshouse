"""Does any of this actually work?

A hallucination detector that has not been measured is a claim, not a tool. Two
evaluations run here, and they are deliberately different in kind.

**Injection** scores hand-labelled grounded and unsupported sentences against
the ablation runs for a question.  The labels live in a committed fixture and
never depend on a verdict emitted by glasshouse, so both always-grounded and
never-grounded implementations are visible in the confusion matrix.

**Attribution** is the one with ground truth. The demo corpus was written so
that particular facts appear in exactly one document, which makes "did it name
the right document" checkable against a hand-written key rather than against a
model's opinion.

**Counterfactual** is the one worth trusting most, because it needs no labels
at all. When glasshouse says a sentence is grounded in a chunk, that is a
falsifiable prediction: delete the document containing that chunk, ask again,
and the sentence should not come back unchanged. The deletion is an independent
intervention, while grading deliberately reuses production sentence splitting
and survival matching so it measures reappearance the same way.

The numbers these produce are in the README, including the unflattering ones.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .models import Document, Verdict
from .pipeline import Lab, build
from .similarity import Matcher
from .text import sentences


@dataclass
class Outcome:
    """One trial."""

    question: str
    detail: str
    passed: bool
    note: str = ""


@dataclass
class Suite:
    """A named set of trials and what they came to."""

    name: str
    outcomes: list[Outcome] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.passed for o in self.outcomes) / len(self.outcomes)

    def report(self) -> dict:
        return {
            "suite": self.name,
            "trials": len(self.outcomes),
            "passed": sum(o.passed for o in self.outcomes),
            "rate": round(self.rate, 4),
            "skipped": len(self.skipped),
            "outcomes": [asdict(o) for o in self.outcomes],
            "skipped_detail": self.skipped,
        }


@dataclass
class ClassificationSuite:
    """Independent binary labels and the detector predictions made for them."""

    name: str = "independent-sentence-classification"
    outcomes: list[Outcome] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    expected: list[bool] = field(default_factory=list)
    predicted: list[bool] = field(default_factory=list)

    def _count(self, expected: bool, predicted: bool) -> int:
        return sum(
            want is expected and got is predicted
            for want, got in zip(self.expected, self.predicted)
        )

    @property
    def precision(self) -> float:
        tp, fp = self._count(True, True), self._count(False, True)
        return tp / (tp + fp) if tp + fp else 0.0

    @property
    def recall(self) -> float:
        tp, fn = self._count(True, True), self._count(True, False)
        return tp / (tp + fn) if tp + fn else 0.0

    @property
    def false_positive_rate(self) -> float:
        fp, tn = self._count(False, True), self._count(False, False)
        return fp / (fp + tn) if fp + tn else 0.0

    @property
    def false_negative_rate(self) -> float:
        return 1.0 - self.recall

    def report(self) -> dict:
        return {
            "suite": self.name,
            "trials": len(self.outcomes),
            "skipped": len(self.skipped),
            "true_positive": self._count(True, True),
            "true_negative": self._count(False, False),
            "false_positive": self._count(False, True),
            "false_negative": self._count(True, False),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "outcomes": [asdict(o) for o in self.outcomes],
            "skipped_detail": self.skipped,
        }


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


async def injection_suite(
    lab: Lab, labels: Sequence[dict], reports: dict[str, object] | None = None
) -> ClassificationSuite:
    """Classify sentences whose truth labels were written outside the detector."""
    suite = ClassificationSuite()
    reports = reports if reports is not None else {}
    for label in labels:
        question = label["question"]
        expected = label["grounded"]
        sentence = label["sentence"]
        report = reports.get(question)
        if report is None:
            report = await lab.ask(question)
            reports[question] = report
        if not report.checkable:
            suite.skipped.append("%s -- no checkable claims" % question)
            continue
        verdict = await _verdict_for_label(lab, report, sentence)
        if verdict is None:
            suite.skipped.append("%s -- could not score labelled sentence" % question)
            continue
        predicted = verdict is Verdict.GROUNDED
        suite.expected.append(expected)
        suite.predicted.append(predicted)
        suite.outcomes.append(
            Outcome(
                question=question,
                detail=sentence,
                passed=predicted is expected,
                note="expected %s; reported %s%s"
                % (
                    "grounded" if expected else "unsupported",
                    verdict.value,
                    "; source %s" % label["document"] if label.get("document") else "",
                ),
            )
        )
    return suite


async def _verdict_for_label(lab: Lab, report, sentence: str):
    """Score one externally labelled sentence using runs already paid for.

    The ablation runs for this question exist; the injected sentence just needs
    scoring against them. Re-running the whole sweep would multiply the cost of
    the evaluation for no additional information.
    """
    from .ablate import Ablator

    ablator = Ablator(lab.llm, lab.matcher, lab.ablation)
    claims = [sentence]

    survival = {
        run.run_id: lab.matcher.survival(claims, run.sentences) for run in report.runs
    }
    others = [r for r in report.runs if r.run_id != "full"]
    if not others:
        return None

    noise = ablator._noise_floor(claims, others, survival)
    memory = ablator._memory(claims, others, survival)
    supports = []
    for run in others:
        if run.kind.value != "loo" or not run.removed:
            continue
        raw = float(1.0 - survival[run.run_id][0])
        from .models import Support

        supports.append(
            Support(
                chunk_id=run.removed[0],
                effect=max(0.0, raw - float(noise[0])),
                raw_drop=raw,
            )
        )
    supports.sort(key=lambda s: -s.effect)

    decided = ablator._decide(0, sentence, supports, float(noise[0]), float(memory[0]))
    return ablator._settle(decided, truncated=False).verdict


# ---------------------------------------------------------------------------
# Attribution against a hand-written key
# ---------------------------------------------------------------------------


async def attribution_suite(lab: Lab, probes: Sequence[dict]) -> Suite:
    """Did it name the right document?

    Each probe is a question whose answer appears in exactly one document of
    the demo corpus, which was written that way on purpose. The key is
    hand-written, so this measures attribution against something outside the
    system rather than against another model's opinion.

    A probe passes if *any* grounded claim in the answer is credited to the
    expected document. That is deliberately lenient about which sentence
    carries the fact -- the model phrases things differently every run, and
    pinning the expectation to a sentence would be measuring the phrasing.
    """
    suite = Suite("attribution")

    for probe in probes:
        question, expected = probe["question"], probe["document"]
        report = await lab.ask(question)
        grounded = [c for c in report.checkable if c.verdict is Verdict.GROUNDED]

        if not grounded:
            suite.outcomes.append(
                Outcome(
                    question=question,
                    detail=expected,
                    passed=False,
                    note="nothing was grounded at all",
                )
            )
            continue

        credited = set()
        for claim in grounded:
            for support in claim.credited:
                chunk = report.chunk_by_id(support.chunk_id)
                if chunk:
                    credited.add(chunk.doc_id)

        suite.outcomes.append(
            Outcome(
                question=question,
                detail=expected,
                passed=expected in credited,
                note="credited %s" % (", ".join(sorted(credited)) or "nothing"),
            )
        )

    return suite


# ---------------------------------------------------------------------------
# Counterfactual
# ---------------------------------------------------------------------------


async def counterfactual_suite(
    lab: Lab,
    documents: Sequence[Document],
    questions: Sequence[str],
    baselines: dict[str, object] | None = None,
) -> Suite:
    """Test every attribution by deleting the document it points at.

    The document deletion is independent of the ablation that produced the
    attribution. Grading deliberately shares production's sentence splitter
    and survival matcher: glasshouse predicts "this sentence came from that
    document"; the test removes the document, asks again, and looks for the
    sentence using the same definition of reappearance.
    """
    suite = Suite("counterfactual")
    by_id = {d.doc_id: d for d in documents}
    baselines = baselines or {}
    reduced_answers: dict[tuple[str, tuple[str, ...]], object] = {}

    grading_matcher = lab.matcher
    from .embed import FrozenEmbedder, NgramEmbedder

    if isinstance(grading_matcher.embedder, FrozenEmbedder):
        # The demo froze deterministic n-gram vectors for the production runs.
        # A newly corrected sentence boundary creates a new input string, so
        # reconstruct that same local embedder instead of requiring a new LLM
        # recording merely to cache another deterministic vector.
        grading_matcher = Matcher(
            NgramEmbedder(dimensions=grading_matcher.embedder.dimensions),
            lexical=grading_matcher.lexical,
        )

    for question in questions:
        report = baselines.get(question) or await lab.ask(question)
        grounded = [c for c in report.checkable if c.verdict is Verdict.GROUNDED]
        if not grounded:
            suite.skipped.append("%s -- nothing grounded to test" % question)
            continue

        for claim in grounded:
            chunk = report.chunk_by_id(claim.strongest.chunk_id)
            if chunk is None or chunk.doc_id not in by_id:
                continue

            # Every document the claim was credited to, not just the strongest:
            # a jointly-credited claim survives removing one of a redundant
            # pair, and calling that a failure would punish the coalition
            # search for being right.
            credited_docs = set()
            for support in claim.credited:
                other = report.chunk_by_id(support.chunk_id)
                if other is not None:
                    credited_docs.add(other.doc_id)

            reduced = [d for d in documents if d.doc_id not in credited_docs]
            if not reduced:
                suite.skipped.append(
                    "%s -- removing the evidence empties the corpus" % question
                )
                continue

            key = (question, tuple(sorted(credited_docs)))
            after = reduced_answers.get(key)
            if after is None:
                retrieval_embedder = lab.index.embedder
                survival_embedder = lab.matcher.embedder
                if isinstance(retrieval_embedder, FrozenEmbedder):
                    # The committed cassette recorded reduced-corpus retrieval
                    # with the deterministic local model. Frozen vectors only
                    # contain the full demo index, not newly chunked variants.
                    retrieval_embedder = NgramEmbedder(
                        dimensions=retrieval_embedder.dimensions
                    )
                if isinstance(survival_embedder, FrozenEmbedder):
                    survival_embedder = NgramEmbedder(
                        dimensions=survival_embedder.dimensions
                    )
                smaller = build(
                    reduced,
                    lab.llm,
                    retrieval_embedder=retrieval_embedder,
                    survival_embedder=survival_embedder,
                    retrieval=lab.retrieval,
                    ablation=lab.ablation,
                )
                after = await smaller.ask(question)
                reduced_answers[key] = after
            survived = float(
                grading_matcher.survival([claim.text], sentences(after.answer))[0]
            )

            suite.outcomes.append(
                Outcome(
                    question=question,
                    detail=claim.text,
                    # 0.85 is near-verbatim reappearance. Below it the claim
                    # changed or vanished, which is what a real dependence on
                    # the removed evidence predicts.
                    passed=survived < 0.85,
                    note="without %s the claim scores %.2f"
                    % (", ".join(sorted(credited_docs)), survived),
                )
            )

    return suite


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


async def threshold_sweep(
    lab: Lab, documents: Sequence[Document], labels: Sequence[dict]
) -> dict:
    """Select a support threshold against the independent sentence labels."""
    from .ablate import AblationPolicy

    rows = []
    reports: dict[str, object] = {}
    for support in (0.06, 0.09, 0.12, 0.15, 0.20, 0.25):
        lab.ablation = AblationPolicy(
            support_threshold=support,
            memory_threshold=lab.ablation.memory_threshold,
            max_runs=lab.ablation.max_runs,
            model=lab.ablation.model,
        )
        suite = await injection_suite(lab, labels, reports=reports)
        rows.append(
            {
                "support_threshold": support,
                "precision": round(suite.precision, 4),
                "recall": round(suite.recall, 4),
                "false_positive_rate": round(suite.false_positive_rate, 4),
                "false_negative_rate": round(suite.false_negative_rate, 4),
                "trials": len(suite.outcomes),
                "skipped": len(suite.skipped),
            }
        )
    eligible = [row for row in rows if row["false_positive_rate"] <= 0.10]
    selected = max(
        eligible or rows,
        key=lambda row: (row["recall"], row["precision"], -row["support_threshold"]),
    )
    return {
        "objective": "maximise recall, then precision, subject to FPR <= 0.10",
        "selected_support_threshold": selected["support_threshold"],
        "sweep": rows,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_suite(args) -> int:
    from .pipeline import load_demo

    if args.demo or not args.corpus:
        demo = load_demo(delay=0.0)
        lab, documents, questions = demo.lab, list(demo.lab.documents), list(demo.questions)
        probes = list(demo.probes)
        labels = list(getattr(demo, "labels", ()))
    else:
        from .corpus import load_documents
        from .llm import AnthropicLLM

        documents = load_documents(args.corpus)
        lab = build(documents, AnthropicLLM())
        questions = _questions_beside(args.corpus)
        probes = _probes_beside(args.corpus)
        labels = _labels_beside(args.corpus)

    if args.suite == "attribution":
        if not probes:
            raise FileNotFoundError(
                "the attribution suite needs probes.json beside the corpus"
            )
        suite = await attribution_suite(lab, probes)
        payload = suite.report()
        print(
            "document attribution %.0f%% (%d probes)"
            % (suite.rate * 100, len(suite.outcomes))
        )
    elif args.suite == "injection":
        if not labels:
            raise FileNotFoundError("the injection suite needs labels.json beside the corpus")
        suite = await injection_suite(lab, labels)
        payload = suite.report()
        print(
            "precision %.0f%%   recall %.0f%%   FPR %.0f%%   FNR %.0f%% (%d trials, %d skipped)"
            % (
                suite.precision * 100, suite.recall * 100,
                suite.false_positive_rate * 100, suite.false_negative_rate * 100,
                len(suite.outcomes), len(suite.skipped),
            )
        )
    elif args.suite == "counterfactual":
        suite = await counterfactual_suite(lab, documents, questions)
        payload = suite.report()
        print(
            "attributions confirmed %.0f%% (%d tested, %d skipped)"
            % (suite.rate * 100, len(suite.outcomes), len(suite.skipped))
        )
    else:
        if not labels:
            raise FileNotFoundError("the threshold sweep needs labels.json beside the corpus")
        payload = await threshold_sweep(lab, documents, labels)
        print("%-20s %-10s %-10s %s" % ("support_threshold", "precision", "recall", "FPR"))
        for row in payload["sweep"]:
            print(
                "%-20.2f %-10.2f %-10.2f %.2f"
                % (row["support_threshold"], row["precision"], row["recall"], row["false_positive_rate"])
            )
        print("selected %.2f" % payload["selected_support_threshold"])

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print("written to %s" % args.out, file=sys.stderr)
    return 0


def _questions_beside(corpus: Path) -> list[str]:
    """Read ``questions.txt`` next to the corpus, one per line."""
    path = Path(corpus)
    candidate = (path if path.is_dir() else path.parent) / "questions.txt"
    if not candidate.exists():
        raise FileNotFoundError(
            "an evaluation needs questions; put one per line in %s" % candidate
        )
    return [
        line.strip()
        for line in candidate.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _probes_beside(corpus: Path) -> list[dict]:
    """Read a hand-written attribution key next to a live corpus."""
    path = Path(corpus)
    candidate = (path if path.is_dir() else path.parent) / "probes.json"
    if not candidate.exists():
        return []
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    return list(payload.get("probes", ()))


def _labels_beside(corpus: Path) -> list[dict]:
    """Read independent sentence-level truth labels next to a live corpus."""
    path = Path(corpus)
    candidate = (path if path.is_dir() else path.parent) / "labels.json"
    if not candidate.exists():
        return []
    return list(json.loads(candidate.read_text(encoding="utf-8")).get("labels", ()))
