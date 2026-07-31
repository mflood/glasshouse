"""glasshouse -- see which sentences of an answer actually came from your documents.

    from glasshouse import build, Document
    from glasshouse.llm import AnthropicLLM

    lab = build(documents, AnthropicLLM())
    report = await lab.ask("why did the rollout slip?")

    for claim in report.checkable:
        print(claim.verdict.value, claim.text)
"""

from .ablate import AblationPolicy, Ablator
from .corpus import ChunkingPolicy, chunk_all, chunk_document, load_documents
from .events import Collector, Event
from .index import HybridIndex, RetrievalPolicy
from .models import (
    Chunk,
    ClaimVerdict,
    Document,
    Report,
    Retrieved,
    Run,
    RunKind,
    Support,
    Usage,
    Verdict,
)
from .pipeline import Demo, Lab, build, from_path, load_demo
from .similarity import Matcher

__version__ = "0.1.0"

__all__ = [
    "AblationPolicy",
    "Ablator",
    "Chunk",
    "ChunkingPolicy",
    "ClaimVerdict",
    "Collector",
    "Demo",
    "Document",
    "Event",
    "HybridIndex",
    "Lab",
    "Matcher",
    "Report",
    "RetrievalPolicy",
    "Retrieved",
    "Run",
    "RunKind",
    "Support",
    "Usage",
    "Verdict",
    "__version__",
    "build",
    "chunk_all",
    "chunk_document",
    "from_path",
    "load_demo",
    "load_documents",
]
