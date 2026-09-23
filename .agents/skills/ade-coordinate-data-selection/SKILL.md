---
name: ade-coordinate-data-selection
description: Propose exactly one evidence-based Long-CoT SFT data-selection plan for an ADE run. Use only when ADE Agent Runtime explicitly invokes the Coordinator with an isolated data-selection package containing the current run revision, accepted insights, ranking, trajectory constraints, and allowed references.
---

# Coordinate Data Selection

Plan data selection specifically for Long-CoT SFT.

Read only `input/`. Do not inspect the repository, `runs/`, Engine logs, or
operator-only data.

Scope contract: this task searches only the selection logic over the fixed,
config-declared candidate pool. Training recipe, model, optimizer,
evaluation/extraction, checkpoint policy, and infrastructure are fixed
controls. Realization statistics are evidence, not inherited constraints.
Do not turn fixed-control variance or possible defects into a selection
direction.

Control authority belongs to the Task and Harness: only constraints they
explicitly declare are inherited fixed controls. Any property or value
observed in a prior realization or recommended by prior Memory remains
evidence. Do not preserve, match, bound, or optimize it merely because a
previous Run had it. Introduce an additional constraint only when the current
Plan's explicit falsifiable hypothesis requires it, and identify that choice as
part of the current intervention or experimental design, not as an inherited
control.

## Execute

1. Read `references/input-guide.md`, `references/input-contract.md`, and
   `references/output-contract.md`, then
   validate `input/manifest.json`,
   `input/action.json`, and `input/task.json` before reasoning.
2. Read the frozen Memory manifest and synthesis, Plan Catalog, Ranking,
   portfolio target, budget, and task views. Before citing a completed Plan,
   comparator, or contradiction, inspect its attributed Memory sidecars.
   Other declared sidecars remain available for targeted verification and do
   not need to be mechanically reread when unrelated to the proposal. Treat
   undeclared files as unavailable.
3. Follow `references/planning-workflow.md`.
4. Compare the proposal with every accepted direction in
   `subject/PLAN_CATALOG.md`, including active Plans with no Trial result, and
   explicitly distinguish the controlled variable and hypothesis. Do not use
   an active Plan as evidence or a relation source.
5. Read the Harness-owned `budget` view in the declared run state. In
   `output/plan.md`, state the remaining Plan/Trial budget and explain the
   exploration-versus-exploitation posture it implies. No relation kind is
   prescribed as a default; explain the selected relation from the evidence
   and budget, and never modify or allocate the budget.
6. Choose one falsifiable selection direction and describe its coherent
   controlled changes in
   `output/plan.md`. Cover the objective, evidence, proposed intervention,
   expected observation, risks, rejection criteria, and fixed controls.
7. Explain the hypothesis expectation against the relation's primary source.
   Harness resolves that source's representative Trial and the frozen portfolio
   target; do not write either comparator SubjectRef.
8. Write the relation plus the hypothesis reason/expected observation under
   `design.comparisons` to `output/decision.json`. Harness owns comparator
   identities and metric values, target Plan identity, basis, provenance,
   manifest, and delivery metadata.
9. Run
   `python .agents/skills/ade-coordinate-data-selection/scripts/validate_output.py output`
   and do not deliver unless it succeeds.

Do not deliver if input identity, task ID, basis revision, or declared hashes
are inconsistent. Do not invent missing evidence.

## Retry

When `input/retry.json` exists:

1. Read its `feedback_paths` and violations.
2. Restore the previous delivery from `input/prior-delivery/`.
3. Change only the named semantic files and preserve unaffected content.
4. Re-read `references/output-contract.md`, run the packaged output validator,
   and deliver exactly `plan.md` and `decision.json`; do not create sidecars.
