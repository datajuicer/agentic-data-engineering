# Input contract

Require `input/manifest.json`, `input/action.json`, and `input/task.json` with
schema version `1`, role `run_summarizer`, task ID `data_selection`, run subject
ID, and basis revision.

Require the exact parent RM, one queue-head complete PM,
`subject/PLAN_UPDATE.md`, and same-basis `run/manifest.json`, Plan Catalog, and
Ranking. Do not read all latest Plan snapshots. Reject inconsistent identity,
missing sources, stale revision, operator-only evidence, or a non-head PM.

For Long-CoT SFT, require the declared sources to expose comparable token
budgets, tokenizer identity, trajectory-group policies, quality and correctness
evidence, length/difficulty coverage, contamination and truncation checks, and
independent evaluation metrics. Do not compare plans whose controls are
materially different without naming the confounder.
