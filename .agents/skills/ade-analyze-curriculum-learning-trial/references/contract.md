# Analyzer contract

The action stage determines the exact output: `review_design` emits only `review-plan.json`; `synthesis` emits only `analysis.md` and `findings.md`. Use only catalog identities and citations. Training coverage is complete prompt groups; offline coverage is all eligible records. Recommendations remain within schedule logic.

## Review-design output

`review-plan.json` has exactly this Harness-owned shape:

```json
{
  "schema_version": "ade.analysis_review_plan.v2",
  "hypotheses": ["A concrete hypothesis testable from the selected evidence."],
  "batches": [
    {
      "batch_id": "training-progression",
      "pool": "training_rollout",
      "investigation_purpose": "Explain what this review tests.",
      "hypothesis_ref": "Optional matching hypothesis text.",
      "selection": {
        "mode": "groups",
        "source_artifact_ids": ["optional catalog artifact IDs"],
        "group_ids": ["catalog group IDs"]
      },
      "rubrics": [
        {
          "rubric_id": "mathematical_quality",
          "instruction": "Assess the response against the problem and reference evidence.",
          "labels": ["incorrect", "partially_correct", "correct"]
        }
      ]
    },
    {
      "batch_id": "offline-all",
      "pool": "offline_validation",
      "investigation_purpose": "Inspect all fixed-checkpoint offline behavior.",
      "selection": {"mode": "all"},
      "rubrics": [
        {
          "rubric_id": "mathematical_quality",
          "instruction": "Assess the response against the problem and reference evidence.",
          "labels": ["incorrect", "partially_correct", "correct"]
        }
      ]
    }
  ]
}
```

The only selection keys are `mode`, `source_artifact_ids`, `record_ids`, and `group_ids`. `groups` requires non-empty `group_ids` and no `record_ids`; `records` requires non-empty `record_ids` and no `group_ids`; `all` carries neither. Each rubric requires a non-empty unique `rubric_id`, a non-empty instruction, and at least two unique labels. Do not add task identity, catalog bindings, coverage summaries, copied records, or copied group members: those belong to Harness inputs and the compiled Review command.

For this task, select enough `training_rollout` group IDs to satisfy the catalog's one global group fraction, with every chosen ID representing its complete sibling group. Select `offline_validation` using `all`.

## Synthesis output

`analysis.md` must contain these sections:

- `## Analysis Scope and Evidence`
- `## Direct Inspection`
- `## Curriculum Schedule Diagnosis`
- `## Training Dynamics and Validation Response`
- `## Findings`
- `## Contradictions and Uncertainty`
- `## Recommendations`

Use `[[review:<review_id>]]` for Review citations and `[[artifact:<artifact_id>]]` only for artifact IDs declared in the package manifest. `findings.md` contains the concise findings delta with the same citation syntax. The Harness derives evidence and coverage sidecars; do not write them.
