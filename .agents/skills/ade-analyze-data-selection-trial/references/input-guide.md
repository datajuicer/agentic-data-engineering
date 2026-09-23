# Analyzer input guide

Read inputs in this order:

1. `input/manifest.json`, `input/action.json`, and `input/task.json` for scope.
   The root manifest is the exhaustive file inventory for this Attempt: every
   Agent-readable input except the manifest itself is declared there with its
   digest and source binding.
2. `input/experiment/EVIDENCE_GUIDE.md` for the current Trial's material map.
3. `input/experiment/evidence-catalog.json` for actual pools, artifact paths,
   identities, field bindings, counts, subset relations, and coverage.
4. Declared statistics, selection artifacts, full telemetry, and records needed
   to form or test the current hypotheses.
5. In synthesis only, the frozen Review packet and coverage.

`input/experiment/manifest.json` is narrower than the root manifest: it is the
Trial evidence allowlist, not the complete Attempt inventory. In
`review_design`, no Review packet is present. In `synthesis`, require exactly
`input/review/packet.json` and `input/review/coverage.json` as the additional
stage inputs.

On retry, also require `input/retry.json` and the current stage's prior outputs
under `input/prior-delivery/`. Read every violation and use prior delivery only
as the repair basis; it is not new accepted evidence. Modify only the paths
listed by `retry.json#feedback_paths`. Harness-internal Review history is not an
Agent input and must not appear under `input/`.

The guide and catalog are generated from this Trial. Their counts and field
bindings supersede assumptions from prior Runs or Skill examples.

For a Search Trial,
`input/experiment/realization/final-realization.json` plus the declared
selection result, selected examples, and training binding describe actual
pre-training realization. The canonical P000 bootstrap baseline has no Builder
realization directory; use its declared selection, training, and evaluation
artifacts as the executed baseline facts. No planning comparator or accepted
Plan file is authorized in an Analyzer package.
