# Input contract

Require `input/manifest.json`, `input/action.json`, `input/task.json`,
`input/experiment/manifest.json`, `input/experiment/evidence-catalog.json`, and
`input/experiment/EVIDENCE_GUIDE.md`. Identities and basis must agree. The
root manifest must declare exactly every other Agent-readable input path with
matching digest and source binding. The Analyzer profile is `reward_design`;
the experiment manifest schema is
`ade.trial_artifacts.v1` and its unique `artifacts` list is the complete
readable allowlist.

Artifact paths are package-owned paths under `input/`; resolve only manifest
entries and honor their status and digest/size binding. Do not scan a host
artifact root, execute reward code, inspect model weights, or read Engine logs,
other Trials, Review scratch, operator-only data, or undeclared paths. Online
evaluation result content is intentionally absent; checkpoint selection facts
do not authorize reconstructing it.

For every Search Trial, require declared final realization and direct-reference
replay/group/Judge materials under `input/experiment/realization/`. The
canonical P000 bootstrap baseline is exactly `plan_id=p000` and
`trial_id=p000-t000-baseline`; its Engine Experiment Package intentionally has
no Builder realization. For that Trial only, require and use the
manifest-declared executed baseline reward, training, and evaluation artifacts
instead. A missing realization on any other Trial is invalid. Reject any
Analyzer-visible accepted Plan, PM, planning comparator, expected observation,
or `objective-comparison.json`.

In synthesis, also require the immutable Review packet and coverage.

On retry, require declared `input/retry.json` and `input/prior-delivery/`.
Reject an undeclared file, a retry overlay absent from the root manifest, or any
Agent-visible `input/prior-reviews/`.
