---
name: ade-build-data-selection-artifact
description: Build one executable Long-CoT SFT data-selection experiment from an accepted ADE Plan snapshot. Use only when ADE Agent Runtime invokes the Artifact Builder with the current Plan snapshot, frozen seed Plan snapshots, and resolved task configuration.
---

# Build Data Selection Artifact

Produce one executable selection function and its experiment design.

Read `input/manifest.json`, `input/action.json`, `input/task.json`, the files
under `input/snapshots/current-plan/`, any declared
`input/snapshots/seed-plans/`, the declared `input/sources/`, and
`input/task/resolved.json`. The primary source contains the real parent
`selection.py` and its canonical realized `selection-result.json`; donor
sources, when present, use the same pair. Read dataset
paths only when `input/task/resolved.json` explicitly authorizes them. Do not
inspect test data, Engine logs, neighboring files, or another Agent workspace.
Before implementing selection logic, read the authorized training-data and
candidate-inventory evidence (at least representative rows and their field
shapes). Derive field names, nested conversation structure, and metadata
values from those inputs; do not infer them from a dataset name or from a
historical artifact.

Scope contract: implement only the accepted selection logic for the fixed
candidate pool. Training, model, optimizer, evaluation and
checkpoint controls are immutable; do not repair or work around their variance
inside `selection.py`. The required experiment-level
`judge_enrichment.enabled` flag controls Judge use: when false, do not declare
or call Judge; when true, the selection execution must make at least one Judge
call on a valid path.

## Execute

1. Read `references/input-guide.md` and `references/input-contract.md`, then
   validate the package identity.
2. Follow `references/artifact-workflow.md` to choose one controlled change.
3. Write `output/design.md` with the expected outline in
   `references/output-contract.md`; record source inheritance, implementation,
   fixed controls, development observations, fallbacks, risks, and limitations.
4. Write `output/selection.py` with the async entrypoint
`select_trajectories(candidate_inventory, select_size, judge)`.
`judge(requests)` is an async batch capability. `requests` is a list of
objects with `question`, `response`, and `rubric`; it returns one structured
evaluation evidence object per request with `status`, `scores_by_dimension`,
and `projected_score`, not a selection decision. Every call must be awaited;
the Engine may split a call into bounded provider batches. Calls may
be nested or routed through multiple rubric branches, but must remain inside
the single selection entrypoint.

`selection.py` runs in ADE's restricted Python entrypoint. `import json` is
allowed for JSON serialization; `from ... import ...`, other modules,
file/network access, and other external side effects are not. Use the safe
builtins provided by the entrypoint and the injected `judge` capability;
express static rubric data as Python literals. The compiler checks this
restriction before the artifact can be admitted.

Source files are authoring-time evidence only and are not shipped beside the
accepted artifact. When the intervention inherits the parent's realized
selection, read `selected_ids` from the declared primary
`selection-result.json` and encode the necessary stable IDs or equivalent
deterministic logic inside the delivered entrypoint. Do not open, import, or
otherwise depend on `input/sources/` at selection runtime. The Plan is a design
proposal: if implementation evidence justifies a different inheritance choice,
state that choice and its consequence in `design.md`; do not claim parent-set
inheritance when the executable artifact does not implement it.

Treat `candidate_inventory` metadata as the contract-provided values, not as
upstream labels that may be inferred. Read the actual inventory evidence and
resolved task inputs before branching on fields such as `domain` or
`difficulty`; do not presume labels or invent values that are absent from the
inventory.

Every rubric value must be a complete JSON string accepted by
ade.rubric_jobs.v1, with exactly these top-level keys:
template, required_variables, output_schema, and projection.
Use required_variables: ["question", "response"]; the template must
contain {{question}} and {{response}}; output_schema must be an object
schema whose required field is scores_by_dimension; and projection must
declare weighted dimensions whose weights sum to 1.0, with score levels and
matching numeric enums in the output schema. Every score-level value and
matching enum value must be a finite number in the inclusive range `[0.0,
1.0]`; for three ordered levels use values such as `0.0`, `0.5`, and `1.0`,
not `0.0`, `1.0`, and `2.0`. A descriptive ad-hoc rubric string or a different
JSON shape is not valid.

The smallest valid one-dimension rubric has this shape (the numeric enum and
score levels must match):

    {
      "template": "Question: {{question}} Response: {{response}}",
      "required_variables": ["question", "response"],
      "output_schema": {
        "type": "object",
        "additionalProperties": false,
        "required": ["scores_by_dimension"],
        "properties": {
          "scores_by_dimension": {
            "type": "object",
            "additionalProperties": false,
            "required": ["quality"],
            "properties": {"quality": {"enum": [0.0, 1.0]}}
          }
        }
      },
      "projection": {
        "dimensions": [{
          "id": "quality",
          "criterion": "quality",
          "weight": 1.0,
          "score_levels": [
            {"value": 0.0, "standard": "weak"},
            {"value": 1.0, "standard": "strong"}
          ]
        }]
      }
    }

The rubric score is evidence for selection; it does not mean “select” or
“reject”. Combine Judge dimension scores and projected scores with declared
heuristics such as keyword, length, difficulty, domain, and trajectory-group
features. The selection logic may build tiers, rank candidates, route different
canonical rubrics, and then apply group constraints. Do not
turn a Judge result into a direct binary select/reject rule unless that is an
explicitly justified selection strategy.

Historical realization statistics are evidence, not inherited constraints.
Implement only constraints declared by the Task Contract or the current Plan's
selection mechanism.

The entrypoint may contain a scoring/enrichment layer followed by bucket,
    routing, and deterministic selection logic. It may use rule-based logic alone,
or combine it with multiple conditional Judge rubrics when enrichment is
enabled; do not add complexity unless the Plan needs it. The final selection
returns stable trajectory IDs. It must return exactly `select_size` IDs on the
canonical fixed pool. If a bucket quota, tier, or preferred subset does not
have enough candidates, deterministically fill the deficit from the remaining
authorized candidates (preserving complete trajectory groups) before
returning. Never return a short list because a preferred bucket is sparse;
the Engine treats any result whose length differs from `select_size` as an
artifact failure. Harness records per-candidate Judge status and Engine
verifies exact selection size, group integrity, and authorized IDs.

When `input/reflection/` is declared, this is the next normal round of the same
Builder Call, not a retry. Read `round.json`, the exact prior delivery,
`realization-report.json`, `selection-result.json`, complete selected examples,
and Judge evidence/manifest. Add exactly one `Reflection decision: finalize`
or `Reflection decision: revise` line to `design.md`. Revise only when the
artifact does not faithfully implement the accepted Plan; observed composition
or scale is not by itself a requirement. Preserve uncertainty instead of
inventing evidence. Harness reruns realization after a revision; do not submit
Judge work or create realization sidecars.

## Retry

When `input/retry.json` exists, start from `input/prior-delivery/`. Modify only
the named `feedback_paths`, preserve unaffected artifacts byte-for-byte,
deliver only `selection.py` and `design.md`; Harness generates all sidecars.
