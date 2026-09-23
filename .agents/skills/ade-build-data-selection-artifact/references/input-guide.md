# Artifact Builder input guide

The package is one immutable Trial-scoped build request. Read
`input/manifest.json` first: it declares every available file, digest, and
source reference. A path not declared there is unavailable.

- `input/action.json`: fixed Run/Coordinator/Plan/Trial identity,
  `source_artifact_ref_ids`, and the parent artifact binding. Do not choose a
  different parent or identity.
- `input/task.json`: task identity and domain.
- `input/task/resolved.json`: resolved candidate-pool, dataset, token-budget,
  and fixed training/evaluation configuration. Dataset paths are authorized
  only when this file explicitly authorizes them; read the exact paths only.
- `input/snapshots/current-plan/`: the accepted Plan hypothesis, constraints,
  and provenance at the frozen basis. This is the primary design input.
- `input/snapshots/seed-plans/`: optional immutable reference Plan snapshots;
  compare them as evidence, not as editable workspaces.
- `input/sources/manifest.json`: authoritative mapping from the action's source
  artifact IDs to the materialized primary and optional donor paths.
- `input/sources/primary/`: the real parent `selection.py` and the canonical
  `selection-result.json` produced by its accepted Trial realization.
- `input/sources/donors/`: optional donor source code and canonical realization
  files, organized by artifact ID.
- `input/reference/` and `input/compliance/`: other task-specific declared
  inputs when present; for data selection, use the resolved candidate inventory
  and fixed ABI; do not inspect neighboring dataset files.
- `input/retry.json` and `input/prior-delivery/`: retry reason and prior
  artifact. Change only declared feedback paths.
- `input/reflection/`: Harness-owned execution of the preceding proposal
  against the same frozen pool/Judge binding. It contains round identity,
  prior delivery, realized selection, examples, Judge evidence, and expectation
  comparison. Reflection is normal experiment work, not validation retry.

Snapshots are materialized read-only evidence, not RunState authority,
checkpoints, or another Agent's workspace. Read snapshot manifests and only
their declared files. Write only the declared `output/` artifacts. Never read
repository code, Engine logs, test data, or undeclared paths.
