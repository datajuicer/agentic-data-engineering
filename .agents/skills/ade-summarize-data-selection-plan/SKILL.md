---
name: ade-summarize-data-selection-plan
description: Create a source-linked derived summary for one Long-CoT SFT data-selection plan. Use only when ADE Agent Runtime explicitly invokes the Plan Summarizer with an isolated, versioned package of accepted findings, token and coverage evidence, failed trials, contradictions, and plan state.
---

# Summarize Data Selection Plan

Update one Long-CoT SFT Plan Memory after the current Trial.

Read only `input/`. A summary is a derived view, never state authority.

Scope contract: summarize only evidence about the fixed-pool selection logic.
Keep training, evaluation, extraction, checkpoint, and infrastructure issues
as out-of-scope caveats; they are not next-Plan search directions.

Only Task/Harness-declared constraints are fixed controls. A property or value
observed in this or a prior realization remains evidence; do not promote it in
Memory into a future constraint, target, quota, or matching requirement merely
for historical comparability. Record any comparability concern as a confounder
or measurement need. A future Plan may constrain that property only through
its own explicit falsifiable hypothesis, with the Coordinator treating the
choice as current experimental design rather than inherited control.

## Execute

1. Read `references/input-guide.md`, then validate `input/manifest.json`,
   `input/action.json`, and `input/task.json`
   against `references/input-contract.md`.
2. Follow `references/summarization-workflow.md`.
3. Write one complete, self-contained `output/MEMORY.md`, preserving still
   valid parent knowledge and integrating the current Trial facts/findings.
   Harness owns PM identity, source binding, manifests, digests, and ledgers.

Treat `input/subject/plan.md` as the immutable pre-experiment commitment. The
Analyzer findings were produced without this commitment and must remain an
independent interpretation of the evidence. For each precommitted claim,
compare expected and rejecting observations with the objective outcome and
source-linked findings. Read final realization, frozen planning comparators,
and Harness-owned objective comparison from the current Trial Record. Use its
`hypothesis_comparator.metric_relations` for each expected primary/secondary
observation; do not infer that aggregate `improved` satisfies both axes.
Adjudicate the Plan hypothesis as `supported|rejected|inconclusive`; never
recompute or rewrite the portfolio result, and never revise the hypothesis to
fit the result.
Never revise the original hypothesis.

End the current Trial update with exactly one `## Trial Conclusions` section
containing `Realization status: ...`, `Hypothesis result: ...`, and `Portfolio
result: ...` using the closed values in the output contract.

Use only accepted sources. Preserve failures, contradictions, uncertainty, and
the next unresolved question. Do not create new experimental findings.

## Retry

When `input/retry.json` exists, restore `input/prior-delivery/`, change only
the named `feedback_paths` and preserve all unaffected memory content.
