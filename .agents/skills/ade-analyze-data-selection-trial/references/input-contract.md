# Input contract

Require `input/manifest.json`, `input/action.json`, `input/task.json`,
`input/experiment/manifest.json`, `input/experiment/evidence-catalog.json`, and
`input/experiment/EVIDENCE_GUIDE.md`. Run/Coordinator/Plan/Trial identity and
basis must agree. Require the root manifest to declare exactly every other
Agent-readable input path with matching digest and source binding. The Analyzer
profile is `data_selection`; the experiment
manifest schema is `ade.trial_artifacts.v1` and its unique `artifacts` list is
the complete readable allowlist.

For every Search Trial, require declared
`input/experiment/realization/final-realization.json` and the task-owned final
selection/training materials. The canonical P000 bootstrap baseline is exactly
`plan_id=p000` and `trial_id=p000-t000-baseline`; its Engine Experiment Package
intentionally has no Builder realization. For that Trial only, require and use
the manifest-declared executed baseline selection, training, and evaluation
artifacts instead. A missing realization on any other Trial is invalid. Reject
any Analyzer-visible accepted Plan, PM, planning comparator, expected
observation, or `objective-comparison.json`; those belong to Plan
Summarization.

Artifact paths are package-owned paths under `input/`; resolve only those
listed in the trimmed manifest. Treat status and digest/size binding literally.
Do not scan host artifact roots, Engine logs, other Trials, Review scratch,
model weights, or undeclared paths. Online evaluation results are not Analyzer
evidence and must not be inferred from checkpoint selection facts.

In `synthesis`, also require `input/review/packet.json` and
`input/review/coverage.json`. They are immutable Harness inputs.

On retry, require declared `input/retry.json` and `input/prior-delivery/`.
Reject an undeclared file, a retry overlay absent from the root manifest, or any
Agent-visible `input/prior-reviews/`.
