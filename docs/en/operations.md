# Run a real experiment

[README](../../README.md) | [Results](results.md)

See [service configuration and lifecycle](service-lifecycle.md) for configuration ownership, process startup, pause/resume and Judge cleanup.

The normal entry is the [Operation Prompt handoff](quickstart.md): one supervising Agent follows the repository Skill through the full Run. This page is its CLI reference and an operator diagnosis guide. Do not execute the startup commands alongside an Agent already owning that Run.

## Prepare the deployment

Complete [installation](installation.md), [input preparation](data-preparation.md) and [deployment](deployment.md). The selected runtime must have its configured Ray capacity, accessible backend environments and model/data paths. Configure the Agent backend and its authentication, W&B settings and Judge credentials. The formal configurations enable these services. Config inspection does not check every service's readiness.

Run from the repository root on the deployment head, using the full environment. Keep the supervisor attached to a persistent terminal session. This example uses the math SFT family; the same baseline/ADE pairing is listed for other tasks in [the configuration matrix](../../configs/README.md).

```bash
source .unified-vllm-0.19.1-verl-venv/bin/activate
ADE_DEPLOYMENT=configs/deployments/local.yaml
ADE_BASELINE_CONFIG=configs/experiments/openthoughts-math-sft-data-selection-baseline-formal.yaml
ADE_CONFIG=configs/experiments/openthoughts-math-sft-data-selection-ADE-formal-n1.yaml
ADE_BASELINE_ID=math-sft-baseline-001
ADE_RUN_ID=math-sft-ade-001
```

The IDs above illustrate CLI syntax only. In the supervised workflow the Skill generates the actual ID mechanically from experiment identity, UTC timestamp and a unique suffix. Use that actual ID when observing or continuing a Run. Shell variables here are conveniences, not YAML interpolation.

## Inspect the resolved configuration

```bash
ade experiment inspect "$ADE_BASELINE_CONFIG" \
  --deployment "$ADE_DEPLOYMENT" --run-id "$ADE_BASELINE_ID"
ade experiment inspect "$ADE_CONFIG" \
  --deployment "$ADE_DEPLOYMENT" --run-id "$ADE_RUN_ID"
```

These commands compile configuration and print `resolved` and `runtime_roots` without creating a Run or starting services. They may read tokenizer and dataset inputs. Resolve missing-file and contract errors before launch. Use the returned `runtime_roots.control` and `runtime_roots.queue` in monitoring commands. For default deployment roots:

```bash
# Replace this name with the deployment ID printed by inspection.
ADE_DEPLOYMENT_ID=your-deployment-name
ADE_CONTROL_ROOT="$PWD/runs/deployments/$ADE_DEPLOYMENT_ID/control"
ADE_QUEUE_ROOT="$PWD/runs/deployments/$ADE_DEPLOYMENT_ID/queue"
```

Global `--runs-root` goes **before** the subcommand. Its CLI default is `runs`, which is not the deployment-scoped Control root. The supervised `run start`/`resume` commands derive their roots from the deployment; do not use the global option to relocate them. This guide retains default roots because Seed source discovery searches `runs/deployments/`.

## Run the baseline, then import it into ADE

The baseline configuration has `bootstrap.reference.enabled: false` and `stop_after_baseline: true`. It executes the baseline and closes that Run:

```bash
ade run start "$ADE_BASELINE_CONFIG" \
  --deployment "$ADE_DEPLOYMENT" --run-id "$ADE_BASELINE_ID"
```

This is a foreground supervisor that launches Control, Agent, Engine and Review processes and manages Run resources including the Judge. It can consume GPUs and invoke external services. One deployment is exclusive to a supervised Run; wait for baseline completion and resource cleanup before starting ADE. Check its status, Trial outcome, evidence and cleanup receipt as described below.

The ADE configuration enables reference bootstrap. Inspect the completed source before importing it:

```bash
ade run seed inspect "$ADE_CONFIG" \
  --deployment "$ADE_DEPLOYMENT" \
  --source-run "$ADE_BASELINE_ID" --frontier bootstrap
ade run start "$ADE_CONFIG" \
  --deployment "$ADE_DEPLOYMENT" --run-id "$ADE_RUN_ID" \
  --initial-state-reference "$ADE_BASELINE_ID" \
  --initial-state-frontier bootstrap
```

Seed inspection validates the source boundary and compatibility without launching training. The source must be present in a deployment-scoped repository under this checkout, with its referenced artifacts available and a unique Run ID across deployments. A private historical Run ID by itself is insufficient. Import reuses the accepted baseline; ADE then executes new Trials. The scripted CPU demo's synthetic baseline is not a formal Seed source.

## Independent Curriculum baseline

The Curriculum baseline uses `bootstrap.reference.enabled: false` and trains its own P000 from the configured base model. Its task-owned baseline generates a deterministic random schedule from the Run seed, materializes the schedule, and trains with the fixed outcome reward. It then completes evaluation, Curriculum analysis and Memory publication.

```bash
ADE_CURRICULUM_CONFIG=configs/experiments/math-math-rft-curriculum-learning-baseline-formal.yaml
ade run start "$ADE_CURRICULUM_CONFIG" \
  --deployment "$ADE_DEPLOYMENT" --run-id curriculum-baseline-001
```

Choose an unused target ID. In the baseline Operation Prompt, set both Initial State Reference and Initial State Frontier to `NONE`. Once the baseline completes and cleanup is accepted, use its actual Run ID as the source for Curriculum ADE N=1/N=3 ADE with frontier `bootstrap`, following the ADE import commands above. Reward Design baselines are not compatible Curriculum Seed sources.

## Monitor and locate evidence

From another terminal with the same environment and variables:

```bash
ade --runs-root "$ADE_CONTROL_ROOT" status "$ADE_RUN_ID"
ade --runs-root "$ADE_CONTROL_ROOT" run observe "$ADE_RUN_ID" \
  --queue-root "$ADE_QUEUE_ROOT"
ade --runs-root "$ADE_CONTROL_ROOT" inspect "$ADE_RUN_ID"
ade --runs-root "$ADE_CONTROL_ROOT" run history "$ADE_RUN_ID"
```

Use `$ADE_BASELINE_ID` instead while monitoring the baseline. `status` summarizes completion and budgets; `observe` adds Trial phases, active work and queue information; `inspect` returns the persisted RunState; `history` exposes revision transitions. A poll with no new accepted fact need not create a revision. Locate service logs under `<control>/<run>/services/supervisor/`, timeline under `reports/timeline.csv`, and exact evidence through the [results guide](results.md).

## Pause and resume

```bash
ade --runs-root "$ADE_CONTROL_ROOT" pause "$ADE_RUN_ID" \
  --reason "Planned maintenance"
```

Pause is a durable request. Control stops submitting new external work and drains active work; wait until status becomes `paused` and the supervisor exits. The first SIGINT/SIGTERM to the supervisor also requests pause and drain; a second signal forces shutdown. Do not treat Ctrl-C as successful cancellation or proof that resources are already released.

After resolving the cause of a pause or suspension, resume with the same Run ID, experiment configuration, deployment and roots:

```bash
ade run resume "$ADE_CONFIG" \
  --deployment "$ADE_DEPLOYMENT" --run-id "$ADE_RUN_ID"
```

For a baseline, substitute its configuration and ID. Do not pass Seed arguments again on resume. Resume continues persisted work and checks its configuration binding. Ordinary retryable failures already have bounded automatic recovery; manual resume is for paused/suspended execution after diagnosis. If live claims or orphaned work prevent resume, inspect provenance and logs and resolve those exact executions first. Do not erase queues or edit `run.json` to bypass the boundary.

## Finish early or cancel

To lower each ADE Coordinator's Plan limit while allowing admitted work to close:

```bash
ade --runs-root "$ADE_CONTROL_ROOT" run finish "$ADE_RUN_ID" \
  --plans-per-coordinator 1 --reason "Finish after the admitted Plan"
```

The requested limit must be compatible with already allocated work; choose it from `status`, not blindly from this example. This is graceful early completion, not immediate termination.

To abandon the Run:

```bash
ade --runs-root "$ADE_CONTROL_ROOT" cancel "$ADE_RUN_ID" \
  --reason "Stop this experiment"
```

Cancellation is terminal. Cleanup closes both queued and claimed Engine/Review commands owned by the Run, so later workers cannot execute its leftover work; sibling Run commands remain untouched. It initiates Run-scoped cleanup; inspect `<control>/<run>/reports/cleanup/run-cleanup.json` and remaining execution before declaring resources released. A cancelled Run cannot be resumed in place. Continuing it requires a typed fork with evidence; see the [recovery rules](components/recovery.md). Do not use a new Run ID to disguise continuation of unresolved work.

## Completion

Check persisted `status`, `completion_kind`, each Trial's `outcome` and `archive_status`, accepted evidence, current Memory heads, and the cleanup receipt. A completed Run does not imply every hypothesis improved the objective, nor does it by itself prove cleanup succeeded. The [results guide](results.md) explains how to read those distinctions.

## Final generalization

After selecting checkpoints using in-loop validation, follow the [generalization guide](generalization.md) to submit a separate evaluation request. Run-owned operator test does not launch this matrix automatically.
