# Output contract

For `stage=review_design`, write only `output/review-plan.json`:

```json
{
  "schema_version": "ade.analysis_review_plan.v2",
  "hypotheses": ["Evidence-grounded hypothesis"],
  "batches": [
    {
      "batch_id": "selected-all",
      "pool": "selected_examples",
      "investigation_purpose": "Test a concrete selection-quality hypothesis.",
      "selection": {"mode": "all"},
      "rubrics": [
        {"rubric_id": "criterion", "instruction": "One observable requirement.", "labels": ["absent", "present"]}
      ]
    }
  ]
}
```

Provide batches for both Review pools, `selected_examples` and
`offline_validation`; each uses `mode=all` without record or group IDs.
`candidate_data` is supporting input only and must not appear in `batches`.
For `stage=synthesis`, write only non-empty `output/analysis.md` and
`output/findings.md`; findings contain the PM-relative knowledge delta.
Harness writes evidence, coverage, usage, provenance, manifest, and delivery
sidecars.
