---
name: ade-summarize-reward-design-run
description: Create a source-linked derived summary for one GRPO RFT reward-design run. Use only when ADE Agent Runtime explicitly invokes the Run Summarizer with an isolated, versioned package of accepted global insights, rollout-group and replay findings, independent evaluation, plan summaries, ranking, contradictions, and run state.
---

# Summarize Reward Design Run

Update the complete Run Memory by merging one queue-head reward Plan Memory.

Read only `input/`. A summary is a derived view, never state authority.

Answer extraction/equivalence, the versioned `TrainingOutcomeAdapter`,
evaluation parsers/graders, ground truth, and checkpoint selection are given
Task/Engine fixed controls. Preserve their observed defects as measurement
caveats, but never convert their modification into reward-search guidance.

Scope contract: compare only reward-function behavior across Plans. Fixed
training, evaluation/extraction, checkpoint, and infrastructure behavior is
not an additional diagnosis or optimization axis.

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
   owns RM identity, accepted source binding, manifests, ranking and ledgers.

Do not re-summarize all Plans or gate the accepted PM. Preserve stable source
citations, reward and replay contradictions, ranking limitations,
uncertainty, remaining risks, and the next global question.
Preserve the queue-head PM's current `## Trial Conclusions` values exactly;
Run Summary does not re-adjudicate hypothesis or recompute Harness results.

For every v2 queue-head PM preserve the distinction among Engine-owned raw
evidence, Artifact projection/pre-group reward, final or assigned reward,
realized advantage from the actual trainer tensor, outcome-only
counterfactual, task-alignment direction, and matched downstream result.
Disabled Group Credit still carries complete siblings but no assignment
decision. Do not promote density, process score, non-flat group count,
variance, discrimination, or non-zero advantage count to cross-Plan improvement
without downstream evidence. Retain legacy interpretation only for an
explicitly declared legacy PM.

## Retry

When `input/retry.json` exists, restore `input/prior-delivery/`, change only
the named `feedback_paths` and preserve all unaffected memory content.
