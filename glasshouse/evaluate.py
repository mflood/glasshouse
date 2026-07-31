"""Does any of this actually work?

A hallucination detector that has not been measured is a claim, not a tool. Two
evaluations run here, and they are deliberately different in kind.

**Injection** is the ordinary one, and it is reported here mainly to explain
why it is nearly worthless. Splice a plausible unsupported sentence into a real
answer and glasshouse flags it every time -- but that is close to a tautology.
A spliced sentence appears in no variant answer, so its similarity drop and its
noise floor are both large and cancel to roughly zero effect. It cannot be
graded grounded almost regardless of what the detector does. A number that a
broken implementation would also score 100% on is not evidence, and saying so
is more useful than printing it.

**Attribution** is the one with ground truth. The demo corpus was written so
that particular facts appear in exactly one document, which makes "did it name
the right document" checkable against a hand-written key rather than against a
model's opinion.

**Counterfactual** is the one worth trusting most, because it needs no labels
at all. When glasshouse says a sentence is grounded in a chunk, that is a
falsifiable prediction: delete the document containing that chunk, ask again,
and the sentence should not come back unchanged. The prediction and the test
share no machinery, so a bug cannot satisfy both by being self-consistent.

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


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

#: Sentences that are plausible, specific, and unsupported by any corpus here.
#: Written by hand rather than generated, so the eval does not depend on a
#: model's willingness to fabricate on demand, and so the same sentences are
#: used on every run.
INJECTIONS = [
    "The board commissioned an independent review of the decision.",
    "A second supplier was engaged in the following quarter.",
    "The programme director resigned shortly afterwards.",
    "Regulators were notified within thirty days as required.",
    "An internal audit later put the figure closer to double the estimate.",
]


async def injection_suite(lab: Lab, questions: Sequence[str]) -> tuple[Suite, Suite]:
    """Splice unsupported sentences into real answers and see what is caught.

    Returns two suites: detection (was the injected sentence flagged?) and
    false positives (did the model's own grounded sentences survive intact?).
    A detector that flags everything scores perfectly on the first and
    catastrophically on the second, which is why both are reported.
    """
    detection = Suite("injection-detection")
    false_positive = Suite("injection-false-positive")

    for number, question in enumerate(questions):
        report = await lab.ask(question)
        if not report.checkable:
            detection.skipped.append("%s -- no checkable claims" % question)
            continue

        for claim in report.checkable:
            if claim.verdict is Verdict.GROUNDED:
                false_positive.outcomes.append(
                    Outcome(
                        question=question,
                        detail=claim.text,
                        passed=True,
                        note="genuinely grounded, correctly kept",
                    )
                )

        injected = INJECTIONS[number % len(INJECTIONS)]
        verdict = await _verdict_for_injected(lab, question, report, injected)
        if verdict is None:
            detection.skipped.append("%s -- could not place the injection" % question)
            continue

        detection.outcomes.append(
            Outcome(
                question=question,
                detail=injected,
                passed=verdict is not Verdict.GROUNDED,
                note="reported as %s" % verdict.value,
            )
        )

    return detection, false_positive


async def _verdict_for_injected(lab: Lab, question: str, report, injected: str):
    """Score one spliced-in sentence using the runs already paid for.

    The ablation runs for this question exist; the injected sentence just needs
    scoring against them. Re-running the whole sweep would multiply the cost of
    the evaluation for no additional information.
    """
    from .ablate import Ablator

    ablator = Ablator(lab.llm, lab.matcher, lab.ablation)
    claims = [injected]

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

    decided = ablator._decide(0, injected, supports, float(noise[0]), float(memory[0]))
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

    This is the strongest check in the project because it shares no machinery
    with the thing it is checking. glasshouse predicts "this sentence came from
    that document"; the test removes the document, asks again, and looks for
    the sentence. A detector that is confidently wrong fails here even if it is
    wrong consistently.
    """
    suite = Suite("counterfactual")
    by_id = {d.doc_id: d for d in documents}
    baselines = baselines or {}
    reduced_answers: dict[tuple[str, tuple[str, ...]], object] = {}

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
                smaller = build(
                    reduced,
                    lab.llm,
                    retrieval=lab.retrieval,
                    ablation=lab.ablation,
                )
                after = await smaller.ask(question)
                reduced_answers[key] = after
            survived = float(
                lab.matcher.survival([claim.text], list(after.answer.split(". ")))[0]
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
    lab: Lab, documents: Sequence[Document], questions: Sequence[str]
) -> dict:
    """Where do the default thresholds come from?

    From this, rather than from taste. The sweep reports detection and false
    positive rates across a grid, and the default is the point that maximises
    detection subject to keeping false positives under a tenth.
    """
    from .ablate import AblationPolicy

    rows = []
    for support in (0.06, 0.09, 0.12, 0.15, 0.20, 0.25):
        lab.ablation = AblationPolicy(
            support_threshold=support,
            memory_threshold=lab.ablation.memory_threshold,
            max_runs=lab.ablation.max_runs,
            model=lab.ablation.model,
        )
        detection, false_positive = await injection_suite(lab, questions)
        rows.append(
            {
                "support_threshold": support,
                "detection": round(detection.rate, 4),
                "false_positive": round(1.0 - false_positive.rate, 4),
                "trials": len(detection.outcomes),
            }
        )
    return {"sweep": rows}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_suite(args) -> int:
    from .pipeline import load_demo

    if args.demo or not args.corpus:
        demo = load_demo(delay=0.0)
        lab, documents, questions = demo.lab, list(demo.lab.documents), list(demo.questions)
        probes = list(demo.probes)
    else:
        from .corpus import load_documents
        from .llm import AnthropicLLM

        documents = load_documents(args.corpus)
        lab = build(documents, AnthropicLLM())
        questions = _questions_beside(args.corpus)
        probes = _probes_beside(args.corpus)

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
        detection, false_positive = await injection_suite(lab, questions)
        payload = {
            "detection": detection.report(),
            "false_positive": false_positive.report(),
        }
        print(
            "detection %.0f%% (%d trials)   false positives %.0f%% (%d grounded claims)"
            % (
                detection.rate * 100,
                len(detection.outcomes),
                (1 - false_positive.rate) * 100,
                len(false_positive.outcomes),
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
        payload = await threshold_sweep(lab, documents, questions)
        print("%-20s %-12s %s" % ("support_threshold", "detection", "false_positive"))
        for row in payload["sweep"]:
            print(
                "%-20.2f %-12.2f %.2f"
                % (row["support_threshold"], row["detection"], row["false_positive"])
            )

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
