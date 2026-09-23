---
name: ade-analyze-data-selection-trial
description: Design and synthesize evidence-bound analysis for one Long-CoT SFT data-selection trial in two Harness-controlled stages.
---

# Analyze Data Selection Trial

Validate `input/manifest.json`, `input/action.json`, and `input/task.json`.
Read `input/experiment/EVIDENCE_GUIDE.md` and
`input/experiment/evidence-catalog.json` before opening any evidence artifact;
the catalog is the pool, identity, field-binding, and coverage authority for
this Trial and the manifest allowlists readable artifacts. Read the input rules
in `references/input-guide.md` and `references/input-contract.md`.

For a Search Trial, read
`input/experiment/realization/final-realization.json` and its declared
selection/training materials as the authority for what the artifact actually
did. The canonical P000 bootstrap baseline (`plan_id=p000`,
`trial_id=p000-t000-baseline`) has no Builder realization; analyze its declared
executed selection, training, and evaluation evidence directly and do not
reject it for the intentionally absent realization files. The accepted Plan,
comparator identities, expected observation, PM, and planning rationale are
intentionally unavailable; do not infer or request them, and do not adjudicate
the precommitted hypothesis.

Analyze only the mutable selection strategy. Training recipe, model, candidate
population, evaluation task, parser/grader, checkpoint policy, and resources
are fixed controls. Read `references/data-selection-analysis.md` when forming
task-specific hypotheses and `references/analysis-workflow.md` for the two
stages.

For `review_design`, inspect the declared population statistics, telemetry,
candidate supporting artifacts, and representative complete records. Write
exactly `output/review-plan.json`. Create hypothesis-bound batches only for
`selected_examples` and `offline_validation`; use `selection.mode=all` so
Harness, not the Agent, expands every eligible record. `candidate_data` remains
readable supporting input for population and selection-bias analysis, but is
not a Review pool and must not receive a Review batch. Never call Review Labor
or copy record text into the plan.

For `synthesis`, require the frozen packet and coverage, reconnect Review
results to candidate → selected → optimization telemetry → offline behavior,
then write exactly `output/analysis.md` and `output/findings.md`. Cite declared
sources as `[[artifact:<id>]]` and usable review batches as
`[[review:<review-id>]]`. Do not request supplemental Review work.

On retry, repair only the current stage's output. Harness owns expansion,
coverage, provider failures, usage, and mechanical sidecars.
