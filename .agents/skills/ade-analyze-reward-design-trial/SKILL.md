---
name: ade-analyze-reward-design-trial
description: Design and synthesize evidence-bound analysis for one GRPO RFT reward-design trial in two Harness-controlled stages.
---

# Analyze Reward Design Trial

Validate `input/manifest.json`, `input/action.json`, and `input/task.json`.
Read `input/experiment/EVIDENCE_GUIDE.md` and
`input/experiment/evidence-catalog.json` before opening evidence artifacts;
the catalog is the current Trial's pool, identity, grouping, position, field,
and coverage authority. Follow `references/input-guide.md` and
`references/input-contract.md`.

For a Search Trial, read the declared final realization and direct-reference
replay materials as the authority for actual reward behavior before training.
The canonical P000 bootstrap baseline (`plan_id=p000`,
`trial_id=p000-t000-baseline`) has no Builder realization; analyze its declared
executed reward, training, and evaluation evidence directly and do not reject
it for the intentionally absent realization directory. The accepted Plan, PM,
comparator bindings, and expected observations are intentionally absent; do
not infer them or adjudicate the precommitted hypothesis.

Analyze only `reward.py` under the fixed GRPO RFT trainer, TrainingOutcome,
versioned `TrainingOutcomeAdapter`, evaluation parser/grader, ground truth,
dataset, checkpoint selection policy, and resources. Read
`references/reward-analysis.md` for task-specific scientific guidance and
`references/analysis-workflow.md` for the staged workflow.

For `review_design`, inspect complete statistics and telemetry plus
representative records. Write exactly `output/review-plan.json`. Review only
`training_rollout` and `offline_validation`; online evaluation results are not
Analyzer evidence. Use `groups` to compare responses to the same prompt and
`records` for hypothesis-driven disagreements or anomalies. Follow the
catalog's coverage unit: legacy v1 pools require global row coverage and allow
incomplete groups; Group Credit v2 pools require global group coverage and
every selected group's complete sibling set. Use `all` for offline validation.
No individual position has its own quota. Never call Review Labor or copy
response text.

For `synthesis`, require the frozen packet and coverage. Compare within-group
quality/reward/outcome, early and late positions, complete telemetry, and
offline transfer, then write exactly `output/analysis.md` and
`output/findings.md`. Cite `[[artifact:<id>]]` and `[[review:<review-id>]]`.

For every trial, keep six layers distinct: Engine-owned raw process evidence;
Artifact projection and pre-group reward; assigned/final training reward;
realized advantage from the trainer's actual tensor; the same-group
outcome-only counterfactual; and downstream validation. Group Credit disabled
still requires complete authoritative siblings and all layers except an
assignment decision. Ask whether reward changes survive fixed normalization,
point toward task success rather than verifier-wrong responses, and align over
steps with matched downstream validation. Reward density, process score,
non-flat group count, or newly non-zero advantages alone are never improvement.

For a v2 `training_rollout` pool, treat the catalog's
`context_fields` as the field authority. Read, when present, `group_type`,
`group_credit_enabled`, process evidence `status`, `adapter_id`,
`schema_version`, and `dimensions`, `artifact_projection`, `pre_group_reward`,
`final_training_reward`, `assigned_training_reward`, `realized_grpo_advantage`,
`counterfactual_identity_grpo_advantage`, `group_credit_mode`,
`group_credit_source`, `group_credit_reason`, declared evidence sources and
process dimensions, in addition to outcome, correctness, rule evidence,
position, and sample identity.
A selected group expands to complete sibling response units; Review Labor still
judges one response per request. Reassemble those independent reviews by
`prompt_group_id`, using `response_index` and `group_size` to verify the sibling
set before making a group-level comparison. Do not assume these v2-only fields
exist in a legacy v1 pool.

On retry, repair only the current stage. Harness owns selector expansion,
coverage, provider failures, usage, and mechanical sidecars.
