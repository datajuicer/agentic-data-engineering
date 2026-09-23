---
name: ade-summarize-reward-design-plan
description: Create a source-linked derived summary for one GRPO RFT reward-design plan. Use only when ADE Agent Runtime explicitly invokes the Plan Summarizer with an isolated, versioned package of accepted reward, rollout-group, independent-validation, replay, exploit, and failure findings.
---

# Summarize Reward Design Plan

Update one GRPO RFT reward-design Plan Memory after the current Trial.

Read only `input/`. A summary is a derived view, never state authority.

Answer extraction/equivalence, the versioned `TrainingOutcomeAdapter`,
evaluation parsers/graders, ground truth, and checkpoint selection are given
Task/Engine fixed controls. Preserve their observed defects as measurement
caveats, but never convert their modification into next-Plan guidance.

Scope contract: summarize only evidence about the reward function. Training,
evaluation/extraction, checkpoint, and infrastructure issues remain fixed
context and cannot become a new reward-design direction.

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
   valid parent knowledge and integrating current Trial facts/findings.
   Harness owns PM identity, accepted source binding, manifests and ledgers.

Treat `input/subject/plan.md` as the immutable pre-experiment commitment. The
Analyzer findings were produced without this commitment and must remain an
independent interpretation of the evidence. For each precommitted claim,
compare expected and rejecting observations with the objective outcome and
source-linked findings. Read final realization, frozen planning comparators,
and Harness-owned objective comparison from the Trial Record. Use its
`hypothesis_comparator.metric_relations` for each expected primary/secondary
observation; aggregate `improved` does not satisfy a regressed or unavailable
axis. Adjudicate the Plan hypothesis as `supported|rejected|inconclusive`;
never recompute or rewrite portfolio result, and never revise the hypothesis to
fit the result. Never revise the original hypothesis.

End the current Trial update with exactly one `## Trial Conclusions` section
containing the closed realization, hypothesis, and portfolio lines from the
output contract.

Preserve reward failures, replay failures, contradictions, uncertainty, and
the next unresolved question. Do not invent findings absent from accepted
current-Trial inputs.

For every current v2 Trial preserve the full adjudication chain: Engine-owned
raw evidence; Artifact projection/pre-group reward; assigned or final training
reward; realized advantage from the actual trainer tensor; outcome-only
counterfactual; and matched downstream validation. Group Credit disabled has
no assignment decision but still requires complete siblings. Reward density,
process score, non-flat group count, additional non-zero advantages, or
discrimination alone cannot support the Plan commitment. For a declared legacy
Trial retain its per-response interpretation; do not infer fields absent from
accepted findings.

## Retry

When `input/retry.json` exists, restore `input/prior-delivery/`, change only
the named `feedback_paths` and preserve all unaffected memory content.
