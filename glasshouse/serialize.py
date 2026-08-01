"""Turning a report into JSON.

Kept apart from both the models and the API so the shape the browser sees is
defined in exactly one place, and so a script can produce the same JSON without
starting a web server.
"""

from __future__ import annotations

from .models import ClaimVerdict, Report, Retrieved, Run


def retrieved_json(r: Retrieved) -> dict:
    return {
        "chunk_id": r.chunk.chunk_id,
        "doc_id": r.chunk.doc_id,
        "doc_title": r.chunk.doc_title,
        "text": r.chunk.text,
        "start": r.chunk.start,
        "end": r.chunk.end,
        "score": round(r.score, 5),
        "lexical_rank": r.lexical_rank,
        "dense_rank": r.dense_rank,
    }


def claim_json(c: ClaimVerdict) -> dict:
    return {
        "index": c.index,
        "text": c.text,
        "verdict": c.verdict.value,
        "note": c.note,
        "noise_floor": round(c.noise_floor, 4),
        "memory": round(c.memory, 4),
        "support": [
            {
                "chunk_id": s.chunk_id,
                "effect": round(s.effect, 4),
                "raw_drop": round(s.raw_drop, 4),
                "credited": s.credited,
                "joint": s.joint,
            }
            for s in c.support
        ],
    }


def run_json(run: Run) -> dict:
    return {
        "run_id": run.run_id,
        "kind": run.kind.value,
        "removed": list(run.removed),
        "answer": run.answer,
        "cost_usd": round(run.usage.cost_usd, 8),
        "input_tokens": run.usage.input_tokens,
        "output_tokens": run.usage.output_tokens,
        "cached": run.usage.cached,
    }


def report_json(report: Report) -> dict:
    return {
        "metadata": dict(report.metadata),
        "question": report.question,
        "answer": report.answer,
        "retrieved": [retrieved_json(r) for r in report.retrieved],
        "claims": [claim_json(c) for c in report.claims],
        "runs": [run_json(r) for r in report.runs],
        "summary": {
            "grounded": report.grounded_count,
            "checkable": len(report.checkable),
            "runs": len(report.runs),
            "cost_usd": round(report.usage.cost_usd, 6),
            "input_tokens": report.usage.input_tokens,
            "output_tokens": report.usage.output_tokens,
            "cached": report.usage.cached,
            "elapsed_s": round(report.elapsed_s, 3),
            "truncated": report.truncated,
            "corpus_contributed": report.corpus_contributed,
        },
    }
