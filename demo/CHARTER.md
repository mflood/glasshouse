# Demo charter

## Purpose

The `demo/` tree demonstrates and evaluates Glasshouse's core primitive:
counterfactual source attribution through evidence ablation. Demo code, data,
ingestion, metadata interpretation, and presentation are not part of the core
library merely because they appear in this repository.

The current Cormorant demo uses a deliberately fictional incident corpus and a
recorded run from a live model. Its controlled facts support reproducible
attribution, counterfactual, and independently labelled evaluations. It is an
evaluation fixture, not a submersible-analysis workflow and not evidence that
Glasshouse determines truth.

## What every demo must declare

Each new demo must document:

1. **Purpose:** the concrete downstream use being illustrated.
2. **Core capability exercised:** the ablation, control, coalition,
   closed-book, source-span, or evaluation behavior under examination.
3. **Downstream behavior:** ingestion, formatting, domain logic, or interface
   behavior added outside the core.
4. **Supported conclusions:** what the Glasshouse experiment itself establishes.
5. **Unsupported conclusions:** what the demo must not claim, including truth,
   source reliability, authority, or complete entailment unless another clearly
   labelled system evaluates it.
6. **Provenance:** origin, license or permission, version, transformations, and
   retrieval date for real source material.
7. **Reproducibility:** frozen inputs, questions, recordings where applicable,
   and independent evaluation keys.

## Dependency rule

Demos depend on the public Glasshouse API. The Glasshouse core never imports
from a demo, and a demo must not introduce domain-specific dependencies into the
core package.

If a demo reveals a generally useful improvement, it may be proposed for the
core only when it passes the admission test in `MISSION.md` and includes an
evaluation showing how source-dependence measurement improves.

## Downstream result handling

A demo may transform a completed Glasshouse report into citations,
visualizations, research notes, or another application artifact. That
transformation must occur after source credit is decided and must remain
visibly separate from the attribution mechanism.

For example, a citation demo may:

```text
read credited chunk IDs and source spans
    -> consolidate overlapping spans
    -> attach bibliographic metadata
    -> render citation markers and references
```

It may not use a generated citation, publisher identity, or presentation rule
to decide that a source deserves credit.

## Demo acceptance checklist

- The demo exercises a named core measurement behavior.
- Downstream code is contained under `demo/` or a separate integration.
- Core code has no dependency on the demo.
- Claims about dependence are traceable to ablation results.
- Claims about truth, authority, or adequacy are excluded or separately tested.
- Negative, unsupported, memory-derived, and undetermined results remain visible.
- Real data includes provenance and rights information.
- The demo remains understandable and reproducible from its committed inputs.
