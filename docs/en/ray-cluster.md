# Ray cluster setup and lifecycle

[Installation](installation.md) | [Deployment](deployment.md)

`ade cluster` manages Ray on the **current node** using the Ray executable installed beside ADE's Python interpreter. It reads the head address from the existing deployment YAML. It does not SSH into other nodes, install environments, download inputs or start Judge servers.

## Setup order

1. Use the same source/runtime on every participating node. Install with `scripts/recreate_unified_vllm_env.sh` into the standard repository-owned environment. Compatible nodes sharing the checkout at the same absolute path use one installation; activate it on each node. Otherwise install on each node as described in [installation](installation.md).
2. Fill `configs/deployments/local.yaml` with the real head address/port and Judge settings. Make source, model/data and configured runtime paths accessible on their execution nodes. Configure the Agent backend and credentials on the host running ADE.
3. Activate that node's full environment. Start the designated head, then join each allocated worker.
4. Inspect Ray status, then inspect the selected ADE experiment. Ray membership alone does not check model files, Judge loading or experiment readiness.
5. Start the Run Supervisor. It owns ADE worker processes and Judge lifecycle; Ray is prepared separately and can outlive a Run.

## Preview before starting

From the repository root:

```bash
ade cluster head --deployment configs/deployments/local.yaml --gpus 0 --dry-run
ade cluster worker --deployment configs/deployments/local.yaml --node-ip 192.0.2.11 --gpus 8 --cpus 80 --dry-run
```

Replace the example worker IP and capacities with your allocated node. Preview prints the exact executable and argument list and executes nothing. Use the unified environment for both preview and real operations. No model or dataset loading is needed to construct these commands.

## Start nodes

On the head:

```bash
ade cluster head --deployment configs/deployments/local.yaml --gpus 0
```

The head IP/host and port come from `run_resources.ray_cluster.address`. Use the head's reachable node address, not `auto`, a Ray Client URL or an unrelated localhost address. The command advertises the explicit `--gpus` allocation (the example uses a CPU-only head); a Judge actor reserves eight GPUs in that same cluster. The dashboard is disabled. Head startup enables `kill_child_processes_on_worker_exit_with_raylet_subreaper` to reap vLLM descendants after actor failure. This is cluster startup configuration: an operator must arrange rebuilding an existing cluster when it has no work. Run this on the configured head machine.

On each worker:

```bash
ade cluster worker --deployment configs/deployments/local.yaml --node-ip 192.0.2.11 --gpus 8 --cpus 80
```

`--node-ip` is required; ADE does not infer a multi-interface machine's address. `--gpus` and optional `--cpus` are per-node Ray resource declarations, not a cluster-wide total. Omit `--cpus` to let Ray detect CPU capacity. The current ADE execution allocator requests 10 Ray CPUs per GPU for relevant GPU placement; provision actual CPU capacity accordingly rather than over-advertising it. If only some GPUs belong to this cluster, set `CUDA_VISIBLE_DEVICES` to that allocation before starting Ray; `--gpus` alone declares a count and does not choose device IDs.

These commands start local Ray processes and return. They do not stop or replace an existing Ray instance. Use the existing Ray diagnostics if a port or node is already active.

## Status and shutdown

```bash
ade cluster status --deployment configs/deployments/local.yaml
```

This explicitly queries the configured head, avoiding an unrelated cluster selected through ambient defaults. Node networking must support Ray's worker communication as well as the head port. Judge gateway connectivity is a separate requirement described in [deployment](deployment.md).

Finish or cancel ADE Runs, inspect their cleanup receipts, and confirm no other workload needs this node's Ray processes before stopping. On each worker, then on the head:

```bash
ade cluster stop --local --dry-run
ade cluster stop --local
```

Stop calls ordinary `ray stop` on the current machine. It affects that machine's Ray processes; it is neither Run-scoped cancellation nor remote cluster shutdown. Stopping Ray on the Judge node terminates its actor and service; use normal Run cleanup first. The required `--local` makes this scope explicit. There is no automatic restart or forced stop.

## Troubleshooting

- **Ray executable missing:** activate the full ADE environment; the command intentionally does not use another `ray` from `PATH`.
- **Worker absent:** check the explicit node/head addresses, environment versions, network and the Ray command's output.
- **Ray reports GPUs but ADE cannot admit work:** compare free GPU/CPU capacity and the selected runtime allocations; inspect existing leases/workloads.
- **Run exits but Ray stays up:** expected ownership separation; Run cleanup and node shutdown are distinct.
