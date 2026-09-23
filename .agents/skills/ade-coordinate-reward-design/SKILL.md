---
name: ade-coordinate-reward-design
description: Propose exactly one evidence-based, directional GRPO RFT reward-design plan as structured Markdown for an ADE run. Use only when ADE Agent Runtime explicitly invokes the Coordinator with an isolated package containing the current revision, accepted insights, rollout-group evidence, independent evaluation, reward constraints, and allowed references. Diagnose one credit defect and set a falsifiable evidence hypothesis, guardrails, evaluation questions, and rejection gates without preselecting process reward or choosing implementation-level dimensions, projections, formulas, thresholds, rubric text, labels, or branch logic.
---

# Coordinate Reward Design

Plan one reward change specifically for GRPO RFT.

Read only `input/`. Do not inspect the repository, `runs/`, Engine logs, or
operator-only data.

Scope contract: this task searches only the reward function. Answer
extraction/equivalence, the versioned `TrainingOutcomeAdapter`, evaluation
parsers/graders, ground truth, checkpoint selection, training controls, and
infrastructure are fixed controls. Observe them only as context; do not turn
their variance into new work. Every Plan must be implementable only in
`reward.py`.

Control authority belongs to the Task and Harness: only constraints they
explicitly declare are inherited fixed controls. Any property or value
observed in a prior realization or recommended by prior Memory remains
evidence. Do not preserve, match, bound, or optimize it merely because a
previous Run had it. Introduce an additional constraint only when the current
Plan's explicit falsifiable hypothesis requires it, and identify that choice as
part of the current intervention or experimental design, not as an inherited
control.

When resolved training enables `rft.group_credit`, the Plan must treat
`compute_score` as per-response evidence construction and
`assign_group_credit` as the complete-sibling credit decision. Its experimental
goal is a task-aligned change in the realized advantage produced by the fixed
GRPO estimator that improves matched downstream validation. Require evidence
that can confirm or refute that causal chain. A raw-reward change that
normalizes to the same advantage, or many non-zero advantages assigned in the
wrong direction, is not improvement. Do not prescribe a ranking formula,
threshold, tie policy, group type, or evidence combination; those remain the
hypothesis and Builder implementation.

Use `input/task/resolved.json` at
`resolved.training.rft.group_credit.enabled` as the switch authority. When it
is false or absent, propose only a per-response `compute_score` intervention
under the existing fallback contract; do not propose, require, or attribute an
effect to `assign_group_credit`, even if an inherited reference file contains
a dormant identity implementation.

Use `input/task/resolved.json#judge_enrichment.enabled` as the only Judge switch.
When false, the artifact must not declare or call Judge. When true, the fixed
deployment-local `llm_judge` is required exactly once for every valid rollout,
including `outcome_score == 0`; the resulting Engine-owned structured evidence
must remain separate from any Artifact projection. Evidence acquisition does
not require a scalar process component and does not require reward or group
intervention. Define the credit defect, evidence hypothesis, success criteria,
and observations that would refute it, then delegate fixed-dimension selection,
projection, per-row reward use, and group intervention/tie/abstention to the
Artifact Builder. Prohibit additional judges, providers, arbitrary evidence
schemas, or free-form metadata.

## Execute

1. Read `references/input-guide.md`, `references/input-contract.md`, and
   `references/output-contract.md`, then
   validate `input/manifest.json`,
   `input/action.json`, and `input/task.json`.
2. Read the frozen Memory manifest and synthesis, Plan Catalog, Ranking,
   portfolio target, budget, and task views. Before citing a completed Plan,
   comparator, or contradiction, inspect its attributed Memory sidecars.
   Other declared sidecars remain available for targeted verification and do
   not need to be mechanically reread when unrelated to the proposal. Read no
   undeclared source.
3. Follow `references/planning-workflow.md`.
4. Compare the proposal with every accepted direction in
   `subject/PLAN_CATALOG.md`, including active Plans that have no Trial result.
   State the controlled-variable and hypothesis difference explicitly; do not
   treat an active Plan as experimental evidence or a relation source.
5. Write one directional proposal to `output/plan.md`. Separate observations,
   hypothesis, expected results, risks, rejection criteria, and fixed controls.
   Delegate implementation weights, formulas, thresholds, rubric text, labels,
   and branch logic to the Artifact Builder.
6. Choose the Plan relation and explain the hypothesis expectation against its
   primary source. Harness mechanically resolves that source's representative
   Trial as the hypothesis comparator and freezes the portfolio target. Do not
   write either comparator SubjectRef.
7. Write only relation and the hypothesis reason/expected observation under
   `design.comparisons` to `output/decision.json`. Harness owns both comparator
   identities and metric values, Plan identity, basis, manifest, and delivery
   metadata.
8. Run
   `python .agents/skills/ade-coordinate-reward-design/scripts/validate_output.py output`
   and do not deliver unless it succeeds.

Do not deliver if the task identity, basis revision, or manifest is
inconsistent. Separate observed evidence from the proposed hypothesis.

## Retry

When `input/retry.json` exists, restore all files from
`input/prior-delivery/`, change only `feedback_paths`, and reproduce unaffected
semantic files byte-for-byte. Re-read `references/output-contract.md`, run the
packaged output validator, and deliver only `plan.md` and `decision.json`.
