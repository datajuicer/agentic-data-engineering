# Artifact contract

Deliver only non-empty `curriculum.py` and `design.md`. The Python module has one top-level async entrypoint with arguments `candidate_inventory, total_steps, prompts_per_step, judge_batch`. Harness performs restricted-Python admission, Judge request validation, ID authorization, shape validation, canonicalization, and final realization.

For a declared reflection round, `design.md` includes exactly one `Reflection
decision: finalize` or `Reflection decision: revise` line. Realized schedule
statistics remain observations rather than acceptance targets.

## Judge request contract

Each `judge_batch` request has exactly `question`, `response`, and `rubric`. The
rubric is a JSON string with exactly these top-level keys:

- `template`: non-empty text containing both `{{question}}` and `{{response}}`;
- `required_variables`: exactly `["question", "response"]`;
- `output_schema`: a closed JSON Schema object whose only required property is
  the closed `scores_by_dimension` object;
- `projection`: a non-empty `dimensions` list. Dimension IDs are unique, weights
  sum to `1.0`, and each dimension's score levels exactly match its schema enum.

A minimal valid one-dimension rubric string is:

```python
rubric = '''{
  "template": "Question: {{question}}\\nResponse: {{response}}",
  "required_variables": ["question", "response"],
  "output_schema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "scores_by_dimension": {
        "type": "object",
        "additionalProperties": false,
        "properties": {"difficulty": {"enum": [0.0, 1.0]}},
        "required": ["difficulty"]
      }
    },
    "required": ["scores_by_dimension"]
  },
  "projection": {
    "dimensions": [{
      "id": "difficulty",
      "criterion": "The problem requires multi-step reasoning.",
      "weight": 1.0,
      "score_levels": [
        {"value": 0.0, "standard": "simple"},
        {"value": 1.0, "standard": "complex"}
      ]
    }]
  }
}'''
```

The result corresponding to each request is evidence shaped like
`{"status": "completed", "scores_by_dimension": {"difficulty": 1.0},
"projected_score": 1.0, "fallback": false}`. Use these scores as deterministic
features in the schedule policy. The Judge does not return prose markers or a
direct schedule selection. Treat fallback/unavailable evidence explicitly; it
is not a genuine zero score.
