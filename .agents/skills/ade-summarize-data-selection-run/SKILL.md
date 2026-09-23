---
name: ade-summarize-data-selection-run
description: Create a source-linked derived summary for one Long-CoT SFT data-selection run. Use only when ADE Agent Runtime explicitly invokes the Run Summarizer with an isolated, versioned package of accepted global insights, plan summaries, ranking, contradictions, and run state.
---

# Summarize Data Selection Run

Update the complete Run Memory by merging one queue-head Plan Memory.

Read only `input/`. A summary is a derived view, never state authority.

Scope contract: compare only fixed-pool selection logic across Plans. Training,
evaluation/extraction, checkpoint, and infrastructure behavior is fixed
context, not an additional optimization axis or diagnosis target.

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
   valid parent knowledge and integrating the queue-head PM result. Harness
   owns RM identity, source binding, manifests, ranking facts, and ledgers.

Do not re-summarize all Plans or gate the accepted PM. Preserve basis revision,
stable source citations, competing plans, contradictions,
uncertainty, ranking limitations, and the next global question.
Preserve the queue-head PM's current `## Trial Conclusions` block exactly:
Run Summary does not re-adjudicate hypothesis or recompute realization and
portfolio results.

## Retry

When `input/retry.json` exists, restore `input/prior-delivery/`, change only
the named `feedback_paths` and preserve all unaffected memory content.
