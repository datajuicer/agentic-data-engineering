# Output contract

Write exactly two semantic files under `output/`:

- `plan.md`: a self-contained controlled data-selection proposal covering the
  objective, accepted evidence, hypothesis, intervention, expected observation,
  fixed controls, rejection criteria, risks, uncertainty, and a `Difference
  from Accepted Plans` comparison against relevant catalog directions.
- `decision.json`: the relation and hypothesis expectation:

```json
{
  "schema_version": "1",
  "relation": {"kind": "new_direction", "related_plans": []},
  "design": {
    "comparisons": {
      "hypothesis": {
        "reason": "direct controlled baseline",
        "expected_observation": {
          "primary": "strict_improvement",
          "secondary": "no_regression"
        }
      }
    }
  }
}
```

Harness resolves the hypothesis comparator from the relation's primary source:
P000 for `new_direction`, the sole related Plan for `revisit` and
`contradiction`, and the first related Plan for `combine`. It binds that Plan's
source-eligible representative Trial and the frozen portfolio target directly.
Expected relations are `strict_improvement|no_regression|diagnostic_only`.
Realization statistics belong in evidence and analysis, not in structured
admission fields.

For `revisit` and `contradiction`, `relation.related_plans` contains exactly one
fully qualified `<run>/cNNN/pNNN` SubjectRef; `combine` contains at least two.
Do not use short `cNNN/pNNN` keys. `design.comparisons` contains exactly the
`hypothesis` expectation, with no comparator SubjectRef. Do not add realization
requirements or implementation details to the structured decision. Partial
Judge fallback is realization telemetry governed by the selector's
ranking/backfill policy.
