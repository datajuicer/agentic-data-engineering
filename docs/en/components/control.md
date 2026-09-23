# Control, scheduling and Reducer

[Components](README.md)

## Implementation and contract

`WorkflowDriver.reconcile_once(run_id)` is the complete workflow entry: it reloads durable state and advances the currently admissible work. `ControlLoop.tick` handles Coordinator planning and catalog acceptance; it is not the whole training lifecycle. `Planner` selects actions and reserves Plan slots deterministically from budget, Coordinator availability and bootstrap status. `TrialLifecycle` compiles/adopts artifacts and accepts Engine, analysis and summary results.

`StateCoordinator.apply` turns a typed Outcome into one committed transition through `Reducer.apply` and the repository. Outcomes carry the relevant scope and basis; Reducer checks lifecycle constraints and returns a new state. Workers report execution facts rather than mutating state themselves. Revision is a count of committed transitions, not a polling counter.

Inputs are RunState plus completed deliveries/receipts. Outputs are admitted actions, updated RunState, accepted artifact/snapshot references and derived records. `ade/core/outcomes.py` defines the concrete Outcome types; `ade/core/trial.py` separates phase, outcome and archive status.

## State invariants

Control is the sole writer of accepted Run state. Reducer applies typed Outcomes without external service calls; Agents, Engine and Review submit deliveries or receipts through their own interfaces. A successful process exit is not scientific acceptance.

`run.json` is the continuation authority. Each committed transition advances one revision; polling alone does not. History preserves accepted boundaries and their evidence. An `AcceptedBoundary` is an immutable `run-id@rev-NNNNNN` whose transition identifies its fact class (`audit`, `operational` or `scientific`), logical work, scope, accepted facts and source receipts. A log line or failed Attempt is not such a scientific boundary.

Use full subjects such as `<run-id>/cNNN/pNNN/tNNN` across queues, receipts, Memory and tracking; local Plan/Trial ordinals are only unique within their parent. Coordinators advance independently. Memory updates use accepted evidence and explicit basis revisions, with Run Memory merges serialized across Coordinators. A Trial's outcome, lifecycle phase and archival state remain separate; terminal Run state also requires the documented completion and cleanup boundaries.

## Configuration and usage

Experiment `search.coordinator_count`, `plans_per_coordinator` and `trials_per_plan` establish search capacity. `bootstrap`, `artifact_builder.max_reflections` and Engine recovery settings control their respective paths. Use `ade run start` through [operations](../operations.md); the Supervisor runs Control and workers together. Low-level `ade control step` can advance real work and is not a read-only preview.

For the existing local demo, observe committed state without advancing it:

```bash
.unified-vllm-0.19.1-verl-venv/bin/ade --runs-root runs/quickstart-demo/runs run history scripted-demo
```

Run the [CPU demo](../cpu-demo.md) first if that directory does not exist. Each revision contains transition kind, scope and accepted references. The complete assembly can be read in `examples/scripted_run.py`; constructing `ControlLoop` alone omits Trial execution.

## Diagnose

For stalled planning, inspect bootstrap status, Plan budgets, planning queue and active Coordinator calls. For an invalid Outcome, compare its scope/basis and the current phase; do not force a revision or weaken Reducer guards. `suspended` requires the [recovery](recovery.md) path. Follow accepted references to the Trial Record before assuming a worker's successful exit was accepted.

## Implementation entry points

| Source | Responsibility |
| --- | --- |
| [ade/controller/workflow_driver.py](../../../ade/controller/workflow_driver.py) | WorkflowDriver |
| [ade/controller/planner.py](../../../ade/controller/planner.py) | Deterministic action and slot planning |
| [ade/controller/control.py](../../../ade/controller/control.py) | Coordinator planning loop |
| [ade/controller/trial_lifecycle.py](../../../ade/controller/trial_lifecycle.py) | Trial admission and lifecycle |
| [ade/controller/state.py](../../../ade/controller/state.py) | Accepted-outcome commit path |
| [ade/controller/reducer.py](../../../ade/controller/reducer.py) | State transition rules |
| [ade/core/outcomes.py](../../../ade/core/outcomes.py) | Typed Outcome definitions |
