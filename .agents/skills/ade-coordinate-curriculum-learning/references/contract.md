# Output contract

Deliver only `plan.md` and `decision.json`. Schedule is the only proposed intervention. Put the directional hypothesis, expected and rejecting observations, risks, fixed controls, and schedule reasoning in `plan.md`; Harness binds that complete document as the hypothesis text.

`decision.json` must use this Harness-owned shape:

```json
{
  "schema_version": "1",
  "relation": {"kind": "new_direction", "related_plans": []},
  "design": {
    "judge_enrichment": true,
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

Harness resolves the relation primary source's representative Trial as the hypothesis comparator and binds `subject/portfolio-target.json` directly. Do not copy either SubjectRef. Expected relations are only `strict_improvement`, `no_regression`, or `diagnostic_only`.

Relation kind is `new_direction`, `revisit`, `contradiction`, or `combine`. `new_direction` has no related Plan; `revisit` and `contradiction` have exactly one; `combine` has at least two. Related Plans are unique full `<run>/cNNN/pNNN` SubjectRefs, never short IDs. Harness owns the target Plan identity and basis; do not add them to this JSON.
