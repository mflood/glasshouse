"""Tests for the ablation engine.

Every test here works by making the scripted model's answer *depend* on which
excerpts it can see, then asserting glasshouse recovers that dependency. A stub
returning a fixed string would pass a suite that tests nothing, because the
whole method is a measurement of change.
"""

import pytest

from glasshouse import AblationPolicy, ChunkingPolicy, Document, Verdict, build
from glasshouse.ablate import Budget, _rotate
from glasshouse.events import Collector
from glasshouse.models import Chunk, RunKind

#: Supported by exactly one chunk (``budget#0``), so leave-one-out alone can
#: attribute it. Triggered by a phrase that appears nowhere else in the corpus.
COST = "The project consumed 2.1 million dollars."
COST_TRIGGER = "2.1 million"

#: Supported by two chunks that say the same thing (``rollout#0`` and
#: ``memo#0``), so no single removal changes it. This is the case leave-one-out
#: cannot see and the coalition search exists for.
SLIP = "The rollout slipped by six weeks."

#: Said whatever the model is shown, including nothing at all.
INVENTED = "The chief executive resigned in protest."


def lab_for(documents, rules, fallback="", **policy):
    """A lab whose model says a sentence only when a trigger word is visible."""
    from glasshouse.llm import Request, ScriptedLLM

    def respond(request: Request) -> str:
        said = [text for trigger, text in rules if trigger in request.prompt]
        return " ".join(said + ([fallback] if fallback else []))

    return build(
        documents,
        ScriptedLLM(respond),
        chunking=ChunkingPolicy(target_words=40),
        ablation=AblationPolicy(**policy),
    )


async def test_a_sentence_that_depends_on_a_chunk_is_grounded(documents):
    """The central claim: remove the evidence, the sentence goes away."""
    lab = lab_for(documents, [(COST_TRIGGER, COST)])

    report = await lab.ask("what did it cost?")

    claim = next(c for c in report.checkable if c.text == COST)
    assert claim.verdict is Verdict.GROUNDED
    assert claim.strongest is not None


async def test_the_credited_chunk_is_the_one_that_mattered(documents):
    lab = lab_for(documents, [(COST_TRIGGER, COST)])

    report = await lab.ask("what did it cost?")

    claim = next(c for c in report.checkable if c.text == COST)
    assert claim.strongest.chunk_id == "budget#0"


async def test_a_sentence_no_chunk_affects_is_not_grounded(documents):
    """The model says this no matter what it is shown, so nothing supports it."""
    lab = lab_for(documents, [(COST_TRIGGER, COST)], fallback=INVENTED)

    report = await lab.ask("what did it cost?")

    claim = next(c for c in report.checkable if c.text == INVENTED)
    assert claim.verdict is not Verdict.GROUNDED


async def test_a_sentence_the_model_says_with_no_documents_is_model_memory(documents):
    """The distinction the whole tool exists to draw."""
    lab = lab_for(documents, [], fallback=INVENTED)

    report = await lab.ask("what happened?")

    claim = next(c for c in report.checkable if c.text == INVENTED)
    assert claim.verdict is Verdict.MODEL_MEMORY
    assert claim.memory > 0.9


async def test_grounded_and_ungrounded_are_separated_in_one_answer(documents):
    """The demo case: a mixed answer, correctly split."""
    lab = lab_for(documents, [(COST_TRIGGER, COST)], fallback=INVENTED)

    report = await lab.ask("what did it cost?")

    verdicts = {c.text: c.verdict for c in report.checkable}
    assert verdicts[COST] is Verdict.GROUNDED
    assert verdicts[INVENTED] is Verdict.MODEL_MEMORY


# ---------------------------------------------------------------------------
# The control run
# ---------------------------------------------------------------------------


async def test_a_control_run_is_fired(documents):
    lab = lab_for(documents, [(COST_TRIGGER, COST)])

    report = await lab.ask("what did it cost?")

    assert any(r.kind is RunKind.CONTROL for r in report.runs)


async def test_the_control_run_sees_every_chunk(documents):
    """It measures order sensitivity, so it must withhold nothing."""
    lab = lab_for(documents, [(COST_TRIGGER, COST)])

    report = await lab.ask("what did it cost?")

    control = next(r for r in report.runs if r.kind is RunKind.CONTROL)
    assert control.removed == ()


async def test_a_jittery_model_does_not_produce_grounding(documents):
    """Movement unrelated to evidence must not be read as evidence.

    This model rephrases on every call for reasons that have nothing to do
    with the excerpts. Without the noise floor subtracted, every chunk looks
    load-bearing for every sentence.
    """
    from glasshouse.llm import Request, ScriptedLLM

    counter = {"n": 0}

    def respond(request: Request) -> str:
        counter["n"] += 1
        return "Something happened in a way numbered %d." % counter["n"]

    lab = build(
        documents,
        ScriptedLLM(respond),
        chunking=ChunkingPolicy(target_words=40),
    )
    report = await lab.ask("what happened?")

    assert all(c.verdict is not Verdict.GROUNDED for c in report.checkable)
    assert all(c.noise_floor > 0 for c in report.checkable)


# ---------------------------------------------------------------------------
# Redundant evidence
# ---------------------------------------------------------------------------


async def test_duplicated_evidence_defeats_leave_one_out(documents):
    """Two chunks saying the same thing: removing either alone changes nothing.

    This is the failure the coalition search exists for, asserted here as a
    property of leave-one-out so the next test has something to improve on.
    """
    lab = lab_for(
        documents,
        [("six week", SLIP), ("six weeks", SLIP)],
        coalition_size=2,
        max_runs=6,  # full + closed + control + three loo, and no more
    )

    report = await lab.ask("how late was it?")

    claim = next(c for c in report.checkable if c.text == SLIP)
    assert claim.verdict is not Verdict.GROUNDED


async def test_the_coalition_search_recovers_redundant_support(documents):
    """Removing the pair together reveals the support neither showed alone."""
    lab = lab_for(
        documents,
        [("six week", SLIP), ("six weeks", SLIP)],
        coalition_size=2,
        coalition_candidates=3,
    )

    report = await lab.ask("how late was it?")

    claim = next(c for c in report.checkable if c.text == SLIP)
    assert claim.verdict is Verdict.GROUNDED
    assert any(s.joint for s in claim.support)
    assert "same thing" in claim.note


async def test_coalitions_only_run_for_unresolved_claims(documents):
    """They are the expensive path; a resolved claim must not pay for them."""
    lab = lab_for(documents, [(COST_TRIGGER, COST)])

    report = await lab.ask("what did it cost?")

    assert not [r for r in report.runs if r.kind is RunKind.COALITION]


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_a_budget_refuses_to_overspend():
    budget = Budget(3)

    assert budget.take(2)
    assert not budget.take(2)
    assert budget.remaining == 1


async def test_the_run_budget_is_honoured(documents):
    lab = lab_for(documents, [(COST_TRIGGER, COST)], max_runs=4)

    report = await lab.ask("what did it cost?")

    assert len(report.runs) <= 4


async def test_a_truncated_report_says_so_rather_than_guessing(documents):
    """An unresolved claim must not be silently rounded to 'unsupported'."""
    lab = lab_for(documents, [(COST_TRIGGER, COST)], max_runs=3)

    report = await lab.ask("what did it cost?")

    assert report.truncated
    assert any(c.verdict is Verdict.UNDETERMINED for c in report.checkable)


async def test_the_closed_book_run_is_never_the_one_dropped(documents):
    """It is what separates grounded from model memory, so it goes first."""
    lab = lab_for(documents, [(COST_TRIGGER, COST)], max_runs=2)

    report = await lab.ask("what did it cost?")

    assert any(r.kind is RunKind.CLOSED_BOOK for r in report.runs)


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


async def test_scaffolding_is_not_judged(documents):
    lab = lab_for(documents, [(COST_TRIGGER, COST)], fallback="I hope this helps.")

    report = await lab.ask("what did it cost?")

    scaffold = next(c for c in report.claims if c.text == "I hope this helps.")
    assert scaffold.verdict is Verdict.NO_CLAIM
    assert scaffold not in report.checkable


async def test_every_retrieved_chunk_gets_an_effect_number(documents):
    """The UI heatmap draws the uncredited ones too."""
    lab = lab_for(documents, [(COST_TRIGGER, COST)])

    report = await lab.ask("what did it cost?")

    claim = next(c for c in report.checkable if c.text == COST)
    assert len(claim.support) == len(report.retrieved)
    assert len(claim.credited) < len(claim.support)


async def test_cost_is_summed_across_every_run(documents):
    lab = lab_for(documents, [(COST_TRIGGER, COST)])

    report = await lab.ask("what did it cost?")

    assert report.usage.input_tokens == sum(r.usage.input_tokens for r in report.runs)


async def test_events_arrive_in_a_usable_order(documents):
    """The UI needs the answer before the verdicts that annotate it."""
    lab = lab_for(documents, [(COST_TRIGGER, COST)])
    collector = Collector()

    await lab.ask("what did it cost?", emitter=collector)

    types = collector.types()
    assert types.index("retrieved") < types.index("answer")
    assert types.index("answer") < types.index("verdicts")
    assert types[-1] == "done"


async def test_a_run_event_is_emitted_as_each_finishes(documents):
    """This is what fills the ablation matrix in live."""
    lab = lab_for(documents, [(COST_TRIGGER, COST)])
    collector = Collector()

    report = await lab.ask("what did it cost?", emitter=collector)

    assert len(collector.of("run")) == len(report.runs) - 1  # the full run is 'answer'


async def test_corpus_contribution_is_reported(documents):
    lab = lab_for(documents, [(COST_TRIGGER, COST)])
    grounded = await lab.ask("what did it cost?")

    memory_only = lab_for(documents, [], fallback=INVENTED)
    ungrounded = await memory_only.ask("what happened?")

    assert grounded.corpus_contributed
    assert not ungrounded.corpus_contributed


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_rotation_moves_every_chunk_but_keeps_them_all():
    chunks = [
        Chunk("c%d" % i, "d", "t", "text %d" % i, 0, 1, i) for i in range(4)
    ]

    rotated = _rotate(chunks)

    assert {c.chunk_id for c in rotated} == {c.chunk_id for c in chunks}
    assert [c.chunk_id for c in rotated] != [c.chunk_id for c in chunks]


def test_rotating_one_chunk_is_a_no_op():
    chunks = [Chunk("c0", "d", "t", "text", 0, 1, 0)]

    assert _rotate(chunks) == chunks


@pytest.mark.parametrize(
    "kwargs",
    [
        {"support_threshold": 0},
        {"support_threshold": 1.5},
        {"memory_threshold": 0},
        {"max_runs": 1},
        {"coalition_size": 1},
    ],
)
def test_incoherent_ablation_policies_are_rejected(kwargs):
    with pytest.raises(ValueError):
        AblationPolicy(**kwargs)


async def test_an_empty_question_is_rejected(documents):
    lab = lab_for(documents, [])

    with pytest.raises(ValueError, match="ask something"):
        await lab.ask("   ")
