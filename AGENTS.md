# Codex project guidance

Read `MISSION.md` before making architectural or feature changes.

## Core boundary

- Treat counterfactual source attribution through evidence ablation as the
  product's core.
- Admit core changes only when they improve the validity, efficiency,
  reproducibility, or inspectability of that measurement.
- Source credit must come from a controlled intervention, never retrieval
  rank, semantic similarity, model self-report, or generated citations alone.
- Preserve the distinction between source dependence, relevance, entailment,
  truth, reliability, and authority.
- Preserve negative outcomes such as model memory, unsupported, undetermined,
  and no corpus contribution.
- Keep the core domain-independent. It may preserve opaque provenance metadata
  and exact spans but must not interpret domain-specific metadata.

## Demos and integrations

- Put ingestion, crawling, document clustering, citation formatting, source
  comparison, and domain-specific workflows under `demo/` or in a separate
  integration.
- Core modules must never import demo or integration code.
- Follow `demo/CHARTER.md` when adding or changing a demo.
- Downstream transformations occur only after Glasshouse has completed source
  attribution and may not influence source-credit decisions.

## Change standard

- For a proposed core feature, complete: "Without this change, Glasshouse
  mismeasures or cannot adequately inspect source dependence because ____."
- Add tests or evaluations for changes to attribution behavior.
- Keep experimental measurements visible rather than replacing them with an
  unexplained score or citation badge.
- Do not hide unfavorable evaluation results.
