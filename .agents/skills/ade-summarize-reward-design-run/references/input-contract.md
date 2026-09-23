# Input contract

Require `input/manifest.json`, `input/action.json`, and `input/task.json` with
schema version `1`, role `run_summarizer`, task ID `reward_design`, run subject
ID, and basis revision.

Require the exact parent RM, one queue-head complete PM,
`subject/PLAN_UPDATE.md`, and same-basis `run/manifest.json`, Plan Catalog, and
Ranking. Do not read all latest Plan snapshots. Reject inconsistent identity,
missing sources, stale revision, operator-only evidence, or a non-head PM.

For GRPO RFT, require comparable rollout-group settings, reward bounds,
Engine-owned evidence, Artifact projection/pre-group reward, final or assigned
reward, actual and counterfactual advantage, non-finite events, exploit-probe
and replay outcomes, and an independent evaluation metric. State configuration
differences before ranking plans.

Treat answer extraction/equivalence, the versioned `TrainingOutcomeAdapter`,
evaluation parsers/graders, ground truth, and checkpoint selection as given
Task/Engine fixed controls. Their defects affect interpretation but are not
reward Plan interventions.
