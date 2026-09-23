# Input contract

Require `input/manifest.json`, `input/action.json`, and `input/task.json` with
schema version `1`, role `plan_summarizer`, task ID `data_selection`, plan
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
`comparisons/objective-comparison.json`. The first fixes realization status;
the last fixes portfolio result and publishes primary/secondary hypothesis
metric relations. Use those per-axis relations, Analyzer findings, and the
immutable Plan to judge hypothesis support without recomputing either
mechanical artifact.

For Long-CoT SFT, require source-linked evidence for realization statistics,
trajectory-group integrity, correctness/quality, contamination, duplication,
truncation, and the controlled variable used for comparison. Preserve
unavailable statistics as explicit unknowns.
