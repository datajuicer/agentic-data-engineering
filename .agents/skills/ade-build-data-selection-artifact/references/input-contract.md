# Input contract

Require schema version `1`, role `artifact_builder`, task ID
`data_selection`, run ID, Coordinator ID, Plan ID, Trial ID, basis revision,
and declarations for every input.

Require `input/action.json#source_artifact_ref_ids` as a non-empty unique
string list set by ADE Harness. For the first Trial it contains the frozen
Plan-relation seed artifacts. For every later Trial it contains exactly the
preceding Trial artifact. Treat it as authoritative lineage, not a request to
choose a different parent.

Require `input/manifest.json`, `input/action.json`, `input/task.json`,
`input/task/resolved.json`, and an accepted immutable snapshot under
`input/snapshots/current-plan/`. When frozen seed Plans are declared, require
their immutable snapshots under `input/snapshots/seed-plans/`. Treat these as
read-only evidence. Never search outside declared paths.

Require `input/sources/manifest.json` and require its ordered source artifact
IDs to match `input/action.json#source_artifact_ref_ids`. Require the primary
source under `input/sources/primary/`; when donors are declared, require them
under `input/sources/donors/<artifact-ref-id>/`. Every source contains the real
accepted `selection.py` and the canonical `selection-result.json` from the
Trial that produced that artifact. The latter records realized `selected_ids`;
it is not a Builder output and must not be recreated by rerunning parent code.

Resolve the candidate inventory and training dataset only from the resolved
task config. The paths are authorized read-only inputs, but their parent
directories are not. Require stable candidate, problem, and trajectory IDs,
tokenizer-derived token counts, length, difficulty, and domain bins, and
complete trajectory groups.

The current Plan snapshot is the source of the accepted hypothesis and
constraints. Frozen seed snapshots are reference evidence, while
`source_artifact_ref_ids` is the immutable parent binding. Reject missing
identities, unavailable declared inputs, inconsistent
snapshot manifests, or a Plan whose required inputs cannot be represented by
the fixed selection function ABI.

Treat all files under `input/sources/` as authoring-time evidence. The delivered
`selection.py` must remain self-contained under the fixed three-argument ABI;
it cannot read or import a source file at Harness or Engine runtime. When the
implementation needs the parent's realized identities, encode the necessary
stable IDs or equivalent deterministic logic inside the entrypoint.

When declared, require `input/reflection/round.json`, `prior-delivery/`,
`realization-report.json`, `selection-result.json`,
`selected-examples.jsonl`, and the Judge evidence/manifest. Their pool and
Judge binding remain frozen across rounds. Use these accepted facts as the
revision basis; do not confuse `reflection_index` with validation
`retry_index`.
