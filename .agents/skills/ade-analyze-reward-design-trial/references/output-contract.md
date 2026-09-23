# Output contract

For `stage=review_design`, write only `output/review-plan.json` with schema
`ade.analysis_review_plan.v2`, non-empty hypotheses, and hypothesis-bound
batches. Each batch has `batch_id`, `pool`, `investigation_purpose`, one
selection, and non-empty rubrics. Each rubric is an object with a unique
`rubric_id`, a non-empty `instruction`, and at least two unique string
`labels`:

```json
{
  "rubric_id": "correctness",
  "instruction": "Assess whether the response reaches the correct answer with valid reasoning.",
  "labels": ["fully_correct", "partially_correct", "incorrect"]
}
```

Selection is exactly one of:

```json
{"mode":"all","source_artifact_ids":["optional-position-artifact"]}
{"mode":"records","source_artifact_ids":["artifact-id"],"record_ids":["record-id"]}
{"mode":"groups","source_artifact_ids":["optional-position-artifact"],"group_ids":["catalog-group-id"]}
```

Omit `source_artifact_ids` when no narrowing is needed. `all` carries no record
or group IDs; `records` and `groups` carry only their matching non-empty list.
Use `all` for `offline_validation`. Training-rollout batches collectively meet
the catalog's global row coverage.

For `stage=synthesis`, write only non-empty `output/analysis.md` and
`output/findings.md`; findings are the PM-relative knowledge delta. Harness
writes all mechanical sidecars.
