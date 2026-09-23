# Input contract

Require `input/manifest.json`, `input/action.json`, and `input/task.json` with
schema version `1`, role `plan_summarizer`, task ID `reward_design`, plan
subject ID, and basis revision.

Require the exact parent `input/memory/MEMORY.md`, immutable
`input/subject/plan.md`, current `input/subject/trial/manifest.json`,
`TRIAL.md`, and `outcome/outcome.json`. Read current `analysis.md`,
`findings.md`, `evidence.json`, and `review-coverage.json` only when declared;
otherwise preserve the declared `failure.json` or unavailable status. Never
read historical Trial snapshots or raw Review scratch. Reject inconsistent
identity, missing declared sources, stale revision, or unaccepted analysis.

Require `input/subject/trial/realization/final-realization.json`,
`comparisons/planning-comparators.json`, and
`comparisons/objective-comparison.json`. Copy realization status and portfolio
result from these facts; use its primary/secondary hypothesis metric relations,
the immutable Plan, and Analyzer findings to judge the hypothesis without
recomputing mechanical comparisons.

For GRPO RFT, require source-linked evidence for score bounds, complete sibling
groups, Engine-owned raw evidence, Artifact projection/pre-group reward, final
or assigned reward, actual realized advantage, outcome-only counterfactual,
non-finite handling, exploit probes, representative replay, and independent
evaluation. Preserve missing evidence as uncertainty.
