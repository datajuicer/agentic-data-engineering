# Deployment

[Installation](installation.md) | [Operations](operations.md) | [Results](results.md)

See [service configuration and lifecycle](service-lifecycle.md) for configuration ownership, process startup, pause/resume and Judge cleanup.

An experiment config owns scientific parameters and selects its runtime allocation. A deployment config binds that experiment to a Ray cluster and the deployment-local Judge. Keep machine addresses and service credentials out of scientific configurations.

## Resource requirements

Use the runtime selected by your experiment and the Judge settings in your deployment to determine capacity and placement. Resource counts and model choices belong to these configs.

| Configuration | What to check |
| --- | --- |
| Experiment `search` | Coordinator, Plan and Trial counts |
| Runtime `cluster`, `coordinator_capacity_gpus` | Cluster capacity and each Coordinator's allocation |
| Runtime `allocations`, `concurrency` | Training/evaluation GPU requests and which jobs can overlap |
| Deployment `run_resources.local_judge` | Judge model path, GPU request and service settings |

Ray places the Judge and workers within the configured cluster. Nodes must expose the source, environment, model/data and runtime-state paths needed by their workloads; Ray does not copy these files. Admission checks available resources and records the selected service placement.

## Configure your deployment

From the repository root:

```bash
test -e configs/deployments/local.yaml || cp configs/deployments/example.yaml configs/deployments/local.yaml
test -e .env || cp .env.example .env
```

Use fresh local files; preserve any existing `.env` or `local.yaml` instead of overwriting it. Both local files are ignored by Git. Edit `local.yaml`:

| Field | Set to |
| --- | --- |
| `name`, `ray_cluster.cluster_id` | The same unique deployment name |
| `ray_cluster.address` | Your Ray head's reachable address and port |
| `local_judge.model_path` | Path to the Judge model selected in the deployment |
| `local_judge.model_digest` | The model identity used by the existing binding; keep it consistent with the selected model |
| `local_judge.gateway_port` | Gateway port (default 8899); the host comes from Ray placement |
| `local_judge.vllm.executable` | Absolute path to the installed `ade-judge-vllm` executable on every eligible Judge node |
| `local_judge.authorization_env` | Environment variable holding the gateway authorization value |

The example uses `192.0.2.10` and `/opt/ade` as placeholders. YAML values do not expand `$VARIABLE` or `~`. Prepare the model selected in your deployment before starting the Run.

Declare each node’s assigned GPUs with `ade cluster head/worker --gpus`. Ensure the allocation satisfies the selected runtime and Judge configuration. Keep protocol and generation settings in the deployment YAML; the compiler supplies the experiment seed.

The Judge executable loads its dependencies from its own environment's `judge-packages`. Ordinary network environment settings can be passed through `vllm.environment`; do not put secrets there because deployment configuration is persisted with the Run.

If you installed with a custom `ENV_DIR`, update the Judge executable and the RFT backend `environment` in the selected `configs/runtime/*.yaml`. Absolute RFT paths are supported; relative paths resolve from the project root. SFT starts LlamaFactory with the Ray worker’s Python interpreter; `runtime.backends.sft.environment` records metadata and does not select another interpreter. Start Ray on every node and invoke ADE from the complete runtime environment.

## Environment variables

Create `.env` in the **repository root** from [`.env.example`](../../.env.example), then fill the following fields. These are the four environment fields used by the supplied formal configurations:

| Field | Required / default | What to put here |
| --- | --- | --- |
| `WANDB_ENTITY` | Required | Your W&B username or team slug that owns the projects, such as `your-team`; not a project name or URL. |
| `WANDB_BASE_URL` | Default: `https://api.wandb.ai` | The API server for your W&B instance. Keep the default for hosted W&B; use your instance's server URL for a self-hosted deployment. |
| `WANDB_API_KEY` | Required | Copy an API key from your account/API-key settings on that W&B instance. It must have write access to `WANDB_ENTITY`. |
| `ADE_LOCAL_JUDGE_AUTHORIZATION` | Required | A private random string you generate for the local Judge gateway. It is not a W&B, model-provider or Agent API key. Callers and the gateway use the same value. |

For example, replace all angle-bracket placeholders before running ADE:

```dotenv
WANDB_ENTITY=<YOUR_WANDB_USER_OR_TEAM>
WANDB_BASE_URL=https://api.wandb.ai
WANDB_API_KEY=<YOUR_WANDB_API_KEY>
ADE_LOCAL_JUDGE_AUTHORIZATION=<YOUR_RANDOM_JUDGE_SECRET>
```

You can generate the Judge secret with `python -c "import secrets; print(secrets.token_urlsafe(32))"` and paste the result into its field. In deployment YAML, `run_resources.local_judge.authorization_env` contains the **variable name** `ADE_LOCAL_JUDGE_AUTHORIZATION`, not the generated value. Experiment `tracking.entity_env`, `base_url_env` and `api_key_env` likewise name the W&B variables. If you change a variable name in YAML, use that exact name in `.env`.

ADE CLI reads `<project-root>/.env` before executing commands. Exported environment variables take precedence, including an exported empty value; unset an old shell variable to use the file's value. Write one `KEY=value` per line, put comments on separate lines, and use matching quotes if needed. The loader does not execute `export`, expand `$VARIABLE` or `~`, or evaluate shell commands. Keep real values in `.env` or the process environment; do not put them in YAML, Operation Prompts or committed files.

Agent authentication is separate: configure the operating account's Codex login as described in [Quickstart](quickstart.md). Merely creating `.env` does not log the outer Codex session in. ADE generates `WANDB_PROJECT`, run/group IDs and `RAY_ADDRESS` from the accepted Run and deployment; do not fill them manually for the supervised workflow. Installer environment settings such as `ENV_DIR` and `CUDA_HOME` are described in [installation](installation.md); export them when invoking the installer, which does not read `.env`.

Check that required values are present without printing them, starting from the repository root:

```bash
.unified-vllm-0.19.1-verl-venv/bin/python - <<'PYENV'
import os
from pathlib import Path
from ade.harness.environment import load_project_environment

load_project_environment(Path.cwd())
fields = ("WANDB_ENTITY", "WANDB_BASE_URL", "WANDB_API_KEY", "ADE_LOCAL_JUDGE_AUTHORIZATION")
missing = [name for name in fields if not os.environ.get(name, "").strip()
           or os.environ[name].strip().startswith("<")]
for name in fields:
    print(f"{name}: {'MISSING_OR_PLACEHOLDER' if name in missing else 'SET'}")
raise SystemExit(1 if missing else 0)
PYENV
```

This checks field presence only. Service connectivity and credentials are checked during Run preflight. Prepare the model/tokenizer, training data and benchmarks referenced by the selected task using [input preparation](data-preparation.md).

## Ray setup on allocated nodes

Use the [Ray cluster guide](ray-cluster.md) for node installation, startup, status, and local shutdown. The public entry point reads this deployment file:

```bash
ade cluster head --deployment configs/deployments/local.yaml --gpus 0 --dry-run
ade cluster worker --deployment configs/deployments/local.yaml --node-ip 192.0.2.11 --gpus 8 --cpus 80 --dry-run
```

Replace the worker address/capacity and preview on the intended node. Remove `--dry-run` to start Ray after installation and node allocation are ready. Head GPUs are advertised to the same cluster; Judge placement follows the deployment configuration. ADE Supervisor starts and manages Judge through Ray; do not start duplicate vLLM servers manually.

## Inspect before starting an experiment

After preparing model/tokenizer and dataset inputs:

```bash
source .unified-vllm-0.19.1-verl-venv/bin/activate
ade --project-root "$PWD" experiment inspect \
  configs/experiments/openthoughts-math-sft-data-selection-baseline-formal.yaml \
  --deployment configs/deployments/local.yaml \
  --run-id installation-check
```

This compiles the actual experiment and prints resolved configuration and runtime paths without starting a Run or services. Missing data or tokenizer files are actionable preparation errors. Schema acceptance alone does not verify network connectivity, free GPUs or model loading. Formal execution additionally needs the selected Run's resource authorization and supervision inputs.

Run state and workers' artifacts use deployment-scoped roots under `runs/deployments/<deployment>/` by default. Keep those generated files out of source control.
