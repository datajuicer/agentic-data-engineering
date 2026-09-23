# Output contract

Write exactly `output/plan.md` and `output/decision.json`.

`plan.md` is a directional GRPO reward proposal. Cover observations, the
falsifiable hypothesis, the single controlled reward direction, expected
training/validation behavior, exploit risks, fixed controls, and rejection
criteria. Include `Difference from Accepted Plans`, comparing the controlled
variable and claim with every relevant direction in `PLAN_CATALOG.md`. Leave
weights, formulas, thresholds, rubric wording, labels, and
branch logic to the Artifact Builder.

The controlled direction must be expressible inside `reward.py`. List answer
extraction, `TrainingOutcomeAdapter`, evaluation parsers/graders, ground truth,
and checkpoint selection among fixed controls. A discovered evaluator defect
may be a caveat but must not be the proposed intervention.

Preserve the single-switch Judge boundary: when `judge_enrichment.enabled` is
false prohibit Judge declaration/calls; when true require exactly one call to
the injected deployment-local `llm_judge` for every valid rollout, including
zero outcome. Require Engine-owned structured evidence to remain distinct from
Artifact projection, prohibit additional judges/providers and arbitrary
evidence fields, and delegate dimension selection and all reward/group use to
the Artifact Builder. Do not prescribe a scalar process reward.

`decision.json` contains the relation and hypothesis expectation:

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
`contradiction`, and the first related Plan for `combine`. It resolves that
Plan's source-eligible representative Trial and binds the frozen portfolio
target without asking the Coordinator to reproduce either SubjectRef. Missing
secondary score is retained as unavailable rather than making the primary
comparison inadmissible. Replay publishes observed reward, advantage, sign,
and Judge availability facts without Coordinator targets or satisfaction
labels. Harness owns target Plan ID, comparator identities and metric values,
basis, provenance, and delivery.

For `revisit` and `contradiction`, `relation.related_plans` contains exactly one
fully qualified `<run>/cNNN/pNNN` SubjectRef; `combine` contains at least two.
Do not use short `cNNN/pNNN` keys. `design.comparisons` contains exactly the
`hypothesis` expectation and no comparator identity or realization requirement.
