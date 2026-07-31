import pytest

from glasshouse import Document
from glasshouse.llm import Request, ScriptedLLM


@pytest.fixture
def documents():
    """A tiny corpus with a deliberate structure.

    ``rollout`` and ``memo`` overlap: both state the six-week slip. That
    redundancy is not accidental -- it is the case leave-one-out cannot see,
    and several tests depend on it being there.
    """
    return [
        Document(
            doc_id="rollout",
            title="Rollout retrospective",
            text=(
                "The Meridian rollout slipped by six weeks. "
                "Engineering attributed the delay to a vendor firmware "
                "revision that arrived in March. "
                "The team shipped on the ninth of June."
            ),
        ),
        Document(
            doc_id="memo",
            title="Vendor memo",
            text=(
                "Our supplier confirmed a six week slip to the Meridian "
                "schedule. "
                "The firmware revision was not available until March. "
                "No penalty clause was triggered."
            ),
        ),
        Document(
            doc_id="budget",
            title="Budget note",
            text=(
                "Meridian consumed 2.1 million dollars of the capital "
                "budget. "
                "Contract staffing accounted for most of the overrun. "
                "The remaining balance was returned to the pool in July."
            ),
        ),
    ]


@pytest.fixture
def scripted():
    """Build a ScriptedLLM whose answer depends on which excerpts it saw.

    Ablation is a measurement of *change*, so a stub that ignores its input
    cannot exercise any of it. This helper lets a test say "assert this
    sentence only when the prompt mentions firmware", which is exactly the
    causal structure the analysis is supposed to recover.
    """

    def make(rules, fallback="Nothing further is known."):
        def respond(request: Request) -> str:
            said = [text for trigger, text in rules if trigger in request.prompt]
            return " ".join(said) if said else fallback

        return ScriptedLLM(respond)

    return make
