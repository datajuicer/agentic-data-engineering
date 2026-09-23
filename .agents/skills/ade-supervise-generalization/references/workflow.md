# Standalone Generalization Supervisor Workflow

This workflow is the complete Operator procedure for one post-hoc evaluation
matrix. The Operation Prompt supplies facts and authorization; it does not
duplicate this lifecycle.

## Required request

Require one fenced YAML mapping with all of these facts:

- `kind: ade.generalization_operation`, a path-safe `evaluation_id`, and
  `run_mode: AUTO` for restart-safe one-prompt operation;
- the exact absolute repository path and its full environment at
  `<repository>/.unified-vllm-0.19.1-verl-venv`, as required by the request compiler;
- one existing deployment YAML plus the expected Ray master IP;
- an explicit resource-authorization sentence beginning with `AUTHORIZED`;
- the versioned generalization contract catalog;
- explicit `full_matrix_after_admission`, retry/worker restart budgets, polling,
  snapshot, and in-flight settings;
- suites containing a frozen `contract_profile`, unique checkpoint IDs plus
  explicit `base`/`baseline`/`n1`/`n3` variants, one or more requested checkpoint
  entries with SubjectRef/path/protocol role, and explicit dataset ID/K entries.
  Base and baseline are optional, with at most one of each per suite. A
  supplemental request can contain only new Run-best checkpoints; it must use
  its own evaluation identity rather than mutate an existing frozen matrix.
  Only listed checkpoints are evaluated. Missing comparison arms remain blank
  with `missing:...` status in comparison.csv; this does not make successfully
  completed requested units fail or authorize rerunning absent controls.

`max_in_flight` accepts `1`, `2`, or `3`. One supervisor owns that many independent
Engine processes, `c001` through `c003`, sharing an evaluation-specific atomic
queue. Each process executes one checkpoint-by-dataset unit at a time. Resource
admission requires the sum of concurrent GPU allocations; three 24-GPU workers
require 72 available GPUs and a runtime defining three 24-GPU resource groups.
Do not start concurrent supervisors for the same exclusive cluster. With fewer
eligible units (including the admission wave), fewer workers will be busy.

Each claim records its worker PID and ID. A lost worker only terminalizes its
own claimed attempts; its exact coordinator leases are released before its slot
is reused. Other workers continue. `max_worker_restarts` applies per worker.
AUTO reuses each surviving process; terminal cleanup attempts every worker.

Stop before contacting Ray when any required fact or explicit authorization is
missing. A value such as `NOT AUTHORIZED`, an unresolved placeholder, a missing
checkpoint, or a master-IP/deployment mismatch is a blocker.

## Materialize and inspect

Use the request's `evaluation_id` as the filename and copy its YAML block
verbatim:

```bash
# Set PROJECT_ROOT to the exact absolute repository path from the request.
# The compiler requires this repository-local environment for standalone matrices.
ADE_ENV="$PROJECT_ROOT/.unified-vllm-0.19.1-verl-venv"
ADE_PYTHON="$ADE_ENV/bin/python"
REQUEST_ROOT="$PROJECT_ROOT/runs/generalization-operation-requests"
REQUEST="$REQUEST_ROOT/<evaluation-id>.yaml"
test -x "$ADE_PYTHON"
```

Create only the exact request file; do not edit the contract catalog or an
existing immutable request to make admission pass. Then run the no-Ray prepare
step:

```bash
"$ADE_PYTHON" -m ade.harness.cli \
  --project-root "$PROJECT_ROOT" \
  evaluate prepare "$REQUEST"
```

Verify its JSON output before starting:

- deployment/Ray master and `authorized: true` match the Operation Prompt;
- resolved unit count equals the checkpoint × dataset cross-products;
- expected generation count is the sum of rows × K across all units;
- `checkpoint_unavailable_units` is empty;
- the expanded config and operation digest are frozen under the printed
  deployment-scoped `request_root`.

`prepare` does not connect Ray, create an evaluation, submit a command, or use a
GPU. Any discrepancy blocks startup.

## Start or resume the one supervisor

Run exactly:

```bash
"$ADE_PYTHON" -m ade.harness.cli \
  --project-root "$PROJECT_ROOT" \
  evaluate supervise "$REQUEST"
```

Keep the process handle. The command performs, in order:

1. local authorization, checkpoint, benchmark artifact, and exclusive ADE Run
   checks;
2. direct Ray/GCS admission against the deployment address;
3. standalone evaluation creation, immutable inputs, and supervisor lease;
4. startup or reuse of the evaluation-owned Engine worker pool with the exact
   `RAY_ADDRESS`;
5. the cheapest dataset for each task/checkpoint entry as the admission wave;
6. the remaining matrix when every admission unit succeeds and
   `full_matrix_after_admission: true`;
7. automatic retry only for receipts with `retryable: true`, within
   `max_attempts`;
8. table regeneration on every durable state change;
9. shutdown of every worker and terminal Ray resource inspection.

Do not start another Engine worker or manually submit/retry units. The
evaluation-specific queue isolates these commands from formal ADE Run queues.

## Monitor

Do not abandon a live process because it produces little terminal output. At
least every ten minutes, and after any process event, inspect:

```text
runs/deployments/<deployment>/evaluations/<evaluation-id>/supervisor/state.json
runs/deployments/<deployment>/evaluations/<evaluation-id>/monitor.md
runs/deployments/<deployment>/evaluations/<evaluation-id>/snapshots/
analysis/generalization/<evaluation-id>/status.csv
analysis/generalization/<evaluation-id>/results.csv
analysis/generalization/<evaluation-id>/comparison.csv
analysis/generalization/<evaluation-id>/family_summary.csv
analysis/generalization/<evaluation-id>/failed_units.csv
analysis/generalization/<evaluation-id>/summary.md
```

Report concise counts, current checkpoint/variant/dataset unit, attempt and
heartbeat, Ray available/total GPUs, retry events, and blockers. Do not treat
log text, GPU utilization, or an
empty queue as proof of success; receipts and supervisor state are authoritative.

If the supervisor process exits unexpectedly while state is nonterminal, inspect
the latest state, snapshot, preflight receipt, and exact `supervisor/engine-worker-cNNN.log`. When
the request and resource authorization remain valid, rerun the same `AUTO`
request. Its immutable operation/config digests, lease, unit state, receipts,
and attempt history prevent duplicate scientific work.

## Blockers and recovery

- Missing checkpoint or tokenizer/chat-template mismatch: stop for the Operator;
  never substitute a checkpoint.
- Dataset/catalog/row-count or prompt/model contract failure: stop; do not change
  the experiment based on observed scores.
- Active exclusive ADE Run, placement group, or GPU lease: stop for a resource
  handoff/cleanup decision.
- Retryable worker/claim failure: let the supervisor create the next physical
  attempt automatically.
- After a verified executor repair, an already authorized finite retry of a
  failed full-matrix unit in a cleaned-up `completed_degraded` request can use
  `evaluate supervise "$REQUEST" --retry-repaired-unit <unit-id>`. This performs
  fresh resource admission, preserves the original receipt and successful units,
  and consumes the original attempt/worker budgets. Do not use it to override a
  scientific contract failure. Resume any interrupted repair with the same AUTO
  request; its recorded repair intent is completed without a duplicate attempt.
- Non-retryable admission-unit failure or exhausted retry/worker budget: accept
  `blocked` as the durable outcome and report the exact receipt/evidence path.
- `awaiting_full_matrix_authorization`: stop. Continuing requires a new exact
  Operation Request/transition implemented by the control plane; do not bypass
  it with manual `evaluate run` commands.

## Terminal acceptance

Accept only when all applicable checks hold:

1. supervisor status is `complete` or `completed_degraded`;
2. every logical unit is terminal and has at most one successful attempt;
3. aggregate worker cleanup is `stopped`, every member is stopped/already exited
   (or absent before startup), and terminal Ray inspection is
   `complete`;
4. `status.csv` and `results.csv` contain exactly the resolved unit count;
5. `comparison.csv` contains one row per task/dataset and computes deltas only
   from complete, contract-matched arms;
6. incomplete/failed units are present in `failed_units.csv` and explicitly
   named in the handoff; completed invalid generations remain explicit in
   `results.csv` and count as incorrect rather than making the unit incomplete;
7. no cross-task score was generated and source ADE audit/ranking files were not
   modified.

For `completed_degraded`, describe the matrix as degraded rather than complete.
For `blocked`, report the durable blocker and keep the objective unfinished
unless the Operator chooses to terminate the operation.
