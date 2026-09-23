# Artifact Builder input guide

The package is one immutable Trial-scoped build request. Read
`input/manifest.json` first: it declares every available file, digest, and
source reference. A path not declared there is unavailable.

- `input/action.json`: fixed Run/Coordinator/Plan/Trial identity,
  `source_artifact_ref_ids`, and immutable parent binding. Do not choose a
  different parent or identity.
- `input/task.json`: task identity and domain.
- `input/task/resolved.json`: resolved reward, rollout, training, evaluation,
  and fixed-control configuration. It is authoritative for runtime controls.
- `input/snapshots/current-plan/`: the accepted reward-design hypothesis and
  constraints at the frozen basis. This is the primary design input.
- `input/snapshots/seed-plans/`: optional immutable reference Plan snapshots;
  use them only as declared comparison evidence.
- `input/reference/manifest.json`: maps the parent artifact IDs to the primary
  reward and optional donor rewards. `reference/primary/reward.py` is the
  first editing source; donor files are comparison sources.
- `input/compliance/manifest.json` and `records.jsonl`: the fixed 32-group
  direct-reference realization population from the actual primary source
  Trial. It validates realized behavior and is not evidence for selecting a
  different planning direction.
- `input/reflection/`: prior delivery plus persisted direct source groups,
  group/Judge results, manifest, and realization report for the same frozen
  binding. Reflection is normal experiment work, not validation retry.
- `input/retry.json` and `input/prior-delivery/`: retry reason and prior
  artifact. Change only declared feedback paths.

Snapshots and reference artifacts are read-only. The injected Judge is a
runtime capability, not an input file or provider client. Never read
repository code, Engine logs, test data, undeclared code, or another Agent's
workspace; write only the declared `output/` artifacts.
