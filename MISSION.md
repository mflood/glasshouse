# Glasshouse mission

## Purpose

Glasshouse is an experimental instrument for measuring whether generated claims
depend on supplied evidence.

It establishes source dependence through controlled removal of evidence and
comparison of the resulting generations. Its primary output is an inspectable,
experimentally derived relationship between a generated claim and the source
material that affected it.

> Glasshouse tests whether generated claims actually depend on their purported
> evidence.
>
> Glasshouse reveals source dependence by removing evidence and measuring what
> changes.
>
> The result is not a model's citation claim, but experimentally measured
> provenance of generated claims.

## Core mandates

### Source credit must result from intervention

Glasshouse may credit evidence to a claim only when that credit is derived from
an ablation or another explicitly controlled intervention. Retrieval rank,
embedding similarity, attention, model explanation, and model-generated
citations may be diagnostics; none independently establishes source dependence.

### Ablation remains the center of the execution model

Every core analysis includes a full generation and one or more counterfactual
generations in which evidence availability changes. A capability that can
operate entirely without counterfactual generation is unlikely to belong in the
core.

### Dependence remains distinct from truth

Glasshouse measures dependence, not factual correctness, entailment, source
quality, reliability, or authority. A false source can ground a false claim. A
true statement from model memory is not evidence from the supplied corpus.

### Results expose the experiment

Every attribution remains inspectable through the intervention, observed
change, control variance, and decision threshold that produced it. The core
must not collapse these measurements into an unexplained citation badge or
confidence score.

### Negative results are first-class

`model_memory`, `unsupported`, `undetermined`, and "the corpus did not
contribute" are valid results. They must not be hidden, repaired, or relabelled
to make an answer appear grounded.

### The core remains domain-independent

The core may preserve opaque source identifiers, spans, and provenance metadata
needed to inspect an experiment. It does not acquire content, interpret
domain-specific metadata, judge publishers, or format downstream products.

## Non-goals of the core

The Glasshouse core does not:

- acquire, crawl, license, or monitor content;
- cluster documents into stories or topics;
- compare publishers, rank sources, or characterize bias;
- determine whether a source is trustworthy or authoritative;
- determine whether a generated claim is true or fully entailed;
- resolve contradictions among sources;
- generate bibliographies or choose citation styles;
- provide domain-specific research workflows;
- replace retrieval evaluation or fact-checking systems; or
- instruct the model to declare which sources it used.

These capabilities may consume a Glasshouse report in a demo or integration.
They do not participate in source-credit decisions and must not become
dependencies of the core library.

## Core admission test

A proposed core change should satisfy all of the following:

1. It improves the validity, efficiency, reproducibility, or inspectability of
   the ablation experiment.
2. It is useful across domains and corpus types.
3. Its absence would impair accurate measurement of source dependence.
4. Its result can be explained through an intervention and an observed change.
5. It preserves the distinction between dependence, relevance, entailment, and
   truth.

The proposal should be able to complete this sentence:

> Without this change, Glasshouse mismeasures or cannot adequately inspect
> source dependence because ______.

If it cannot, the feature belongs in a demo, adapter, integration, or separate
application.

Examples of core work include control-run noise estimation, coalition search,
cross-generation sentence alignment, ablation scheduling, experiment
serialization, source-span preservation, and attribution evaluation.

Examples of downstream work include PDF or web ingestion, news-story
clustering, citation formatting, publisher comparison, source-reliability
scoring, and domain-specific research interfaces.

## Dependency boundary

Dependencies point inward:

```text
demos and integrations  --->  glasshouse core
ingestion adapters      --->  glasshouse core
evaluation suites       --->  glasshouse core
```

The core library must never import from demos, integrations, or domain-specific
adapters. Optional ingestion and presentation dependencies stay outside the
core dependency set. General improvements discovered through a demo move into
the core only after passing the core admission test.

## Citations

Glasshouse may emit claim-to-source attribution suitable for citation.
Citation selection, numbering, formatting, bibliography generation, and
publication workflow are deterministic downstream transformations of that
attribution.

The intended sequence is:

```text
generate without citations
    -> run controlled ablations
    -> credit source spans
    -> optionally verify citation adequacy downstream
    -> render citations downstream
```

Citation code must not participate in deciding whether a chunk is credited.
The generating model must not be asked to provide source-use claims that are
then mistaken for experimental attribution.

## Contribution standard

Changes to attribution behavior require tests or evaluations showing which
measurement property improved. Negative and unflattering results remain
visible. New downstream behavior is isolated in a demo or integration and
documents which conclusions come from Glasshouse versus the downstream layer.

The enduring test is simple:

> Does this make Glasshouse better at measuring what evidence caused a
> generated claim?
