# Claude project guidance

The canonical project mission and architectural boundary are in `MISSION.md`.
Read it before proposing or implementing features.

## Required principles

- Glasshouse measures whether generated claims depend on supplied evidence by
  removing evidence and observing changes beyond generation variance.
- Source credit must be experimentally derived. Retrieval scores, semantic
  relevance, attention, model explanations, and generated citations do not
  independently establish source use.
- Glasshouse measures dependence, not truth, entailment, reliability, or
  authority. Do not blur these concepts in code, UI text, or documentation.
- `model_memory`, `unsupported`, `undetermined`, and no corpus contribution are
  valid outcomes and must remain visible.
- Core changes must improve the validity, efficiency, reproducibility, or
  inspectability of the ablation experiment across domains.

## Scope control

- Keep crawling, ingestion, clustering, citation formatting, source comparison,
  and domain workflows in `demo/` or separate integrations.
- The core may carry opaque provenance metadata and exact source spans, but it
  must not acquire content or interpret domain-specific metadata.
- Core code must never depend on a demo or integration.
- Follow `demo/CHARTER.md` for demo work.
- Any downstream transformation must occur after attribution and must not
  influence which evidence is credited.

Before adding a core feature, answer: "Without this change, Glasshouse
mismeasures or cannot adequately inspect source dependence because ____."
If there is no convincing answer, keep the feature downstream.
