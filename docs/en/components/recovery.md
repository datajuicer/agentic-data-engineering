# Monitoring, recovery, Seed and fork

[Components](README.md)

## Implementation boundaries

Supervisor owns the foreground process group and checks Control/worker exits. `RunResourceMonitor` samples Run/resource state and tracking health; telemetry and W&B streams are observations, not a second recovery authority. `run.json`, typed failures and accepted references determine continuation. `run_cleanup.py` handles scoped process/resource cleanup and writes receipts.

| Action | Meaning |
| --- | --- |
| Automatic retry | New physical attempt for the same logical work within the configured recovery budget |
| Pause | Persist stop request, stop submitting new external work and drain to resumable state |
| Resume | Continue the same persisted Run after pause/suspension and readiness checks |
| Seed | Import an accepted completed source boundary into a new experiment |
| Fork | Create a new Run from a typed, evidence-backed replay boundary |
| Cancel | Abandon this Run terminally; continuation needs a typed fork |

`engine.automatic_recovery` bounds attempts and dependency-readiness waiting. Retryable failures need not exit the Supervisor for manual intervention. Exhausted budgets or unresolved integrity/execution conditions can suspend the Run. Do not equate elapsed training time with a dead worker; heartbeat/progress and actual execution identity matter.

## Automatic recovery conditions

Prefer recovery within the same Run, then repair/resume a suspended Run. Use a typed fork only for the causes below. Changing scientific configuration, inputs or frozen execution semantics requires a new experiment.

An Engine Command freezes logical inputs; each physical Attempt has its own execution identity, workspace and terminal receipt. Retry requires a retryable infrastructure/dependency failure before scientific acceptance, unchanged frozen inputs, a terminal or explicitly fenced old Attempt, scoped resource reclamation, healthy admission and remaining recovery budget. Only the current active Attempt can supply an accepted receipt; late output from a replaced Attempt is audit-only. Other Coordinators continue independently.

`max_attempts` includes the initial Attempt and does not refund Plan/Trial budget. `dependency_readiness_timeout_seconds` bounds dependency waiting, not training duration. A failed training Attempt restarts the logical Command from its frozen inputs; ADE does not resume intermediate optimizer/RNG state. Pause drains active external execution before becoming resumable. Automatic recovery exhaustion or an unresolved execution boundary suspends the Run for repair.

## Use

Use the complete pause/resume/cancel and Seed commands in [operations](../operations.md). To inspect the local example without starting services:

```bash
.unified-vllm-0.19.1-verl-venv/bin/ade --runs-root runs/quickstart-demo/runs run observe scripted-demo --queue-root runs/quickstart-demo/queue
```

A formal Run uses deployment roots from config inspection. Inspect `services/supervisor/`, `reports/worker-provenance.json`, tracking health and usage alongside active calls/commands. `run observe` is a state/queue view; `ade monitor run` is a service and can connect to Ray/tracking, so it is not an interchangeable read-only command.

## Fork and Seed contracts

`RunSeedImporter` resolves `bootstrap` or Coordinator Plan frontiers from a completed source, checks scientific compatibility and materializes accepted references. A source Run ID must be discoverable with its actual artifacts under deployment-scoped storage.

`RunForkService.fork` accepts one of `invalid_accepted_fact`, `unfenceable_execution` or `continue_cancelled`. The CLI options are `--invalidate`, `--unfenceable`, `--continue-cancelled`; each takes an exact evidence revision selector such as `run-id@rev-000027`, not a bare artifact ID. The selected revision must contain the typed evidence required by that cause. The service resolves the replay boundary and generates the target Run ID and lineage. It is not a general “copy any checkpoint” command.

| Fork option | Required evidence revision |
| --- | --- |
| `--invalidate` | A scientific AcceptedBoundary with accepted fact references for the invalid logical work |
| `--unfenceable` | A suspended Run with `failure.code=unfenceable_execution`; its `attempt:<logical-work-ref>/<attempt-id>@writer:<backend-writer-ref>` evidence must also appear in transition `origin_refs` and identify exactly one active Attempt |
| `--continue-cancelled` | The terminal `run_cancelled` transition |

First inspect the source history using its deployment Control root, then pass the evidence selector:

```bash
ade --runs-root "$ADE_CONTROL_ROOT" run history "$ADE_RUN_ID"
# Replace the selector with the actual cancelled transition from that history.
ade --runs-root "$ADE_CONTROL_ROOT" run fork --continue-cancelled source-run@rev-000027
```

The selector supplies evidence, not an arbitrary replay checkpoint. ADE chooses the boundary and child ID. The child keeps accepted facts before that boundary and records lineage; active execution, leases and backend handles are not inherited. Fork creation does not dispatch work. Activating the child still requires resource admission; attachment handoff requires a safe direct source boundary and matching deployment/Judge ownership. A fork does not authorize interrupting a live source or taking another Run's resources.

## Diagnose closure

First identify the latest transition and exact active logical work/attempt. Inspect its receipt, heartbeat, worker provenance and real process status. Resume may refuse live or unresolved claims; repair that execution rather than deleting queues. The first Supervisor SIGINT/SIGTERM requests pause/drain; a second forces exit. Confirm terminal state, Trial archival and `reports/cleanup/run-cleanup.json` separately. A completed Run or finished W&B stream does not prove all owned resources were released.

## Implementation entry points

| Source | Responsibility |
| --- | --- |
| [ade/harness/supervisor.py](../../../ade/harness/supervisor.py) | Process supervision and drain |
| [ade/harness/run_monitor.py](../../../ade/harness/run_monitor.py) | Resource and tracking observations |
| [ade/harness/run_cleanup.py](../../../ade/harness/run_cleanup.py) | Scoped cleanup receipts |
| [ade/controller/workflow_driver.py](../../../ade/controller/workflow_driver.py) | Automatic recovery reconciliation |
| [ade/controller/run_seed.py](../../../ade/controller/run_seed.py) | Accepted frontier import |
| [ade/controller/fork.py](../../../ade/controller/fork.py) | Typed fork boundary and lineage |
| [ade/core/failures.py](../../../ade/core/failures.py) | Failure types |
