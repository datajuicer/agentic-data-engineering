# Frequently asked questions

[README](../../README.md)

## What can I run without GPUs or credentials?

The [CPU demo](cpu-demo.md) runs the workflow with scripted Agent, training, evaluation and Review responses. Its scores and baseline are synthetic; its output is not a Seed for real experiments. Installing its packages requires network access, but running the example needs no model download or service.

## Does N=1 mean one GPU?

N counts search Coordinators. The current formal math SFT N=1 configuration requests 24 training/evaluation GPUs plus an eight-GPU Judge reservation in the same Ray cluster. Other task capacities are listed in [deployment](deployment.md).

## Which environment should I use?

Use `.unified-vllm-0.19.1-verl-venv` for every ADE command, including tests and the CPU demo. Install it with `bash scripts/recreate_unified_vllm_env.sh`; do not create a separate `.venv` or use `uv sync`. The installer includes both training backends and the Judge dependency overlay. FlashAttention compilation needs CUDA development tools, or a matching wheel supplied through `FLASH_ATTN_WHEEL`. See [installation](installation.md).

## Why does experiment inspection report missing inputs?

`ade experiment inspect` reads real tokenizer and task inputs. Prepare the model, candidate pool and selected benchmarks using the [data guide](data-preparation.md). YAML does not expand shell variables or `~`. Replace example deployment addresses and paths with actual values. Schema validation alone does not establish network, GPU or model readiness.

## Do I start Agent and Judge workers myself?

Prepare Ray separately through [the cluster CLI](ray-cluster.md). `ade run start` manages ADE workers and the deployment Judge. Configure Agent backend authentication separately before starting a Run. See [service lifecycle](service-lifecycle.md) for configuration ownership and shutdown behavior.

## Can I resume a Run with changed YAML or use the demo baseline?

Resume continues the persisted scientific configuration; editing YAML does not change that contract. Formal search requires a compatible accepted baseline. Follow [operations](operations.md) to run the baseline, inspect its Seed frontier and import it into search. The CPU demo baseline is incompatible with this purpose.

## Where do I find results and confirm cleanup?

Use the resolved `runtime_roots.control` path and the [results guide](results.md). Inspect Run state, Trial outcomes, accepted evidence and Memory. A terminal Run alone does not prove GPU cleanup; inspect `reports/cleanup/run-cleanup.json`. Pause retains the deployment Judge and does not release its GPUs.

## Why use an Operation Prompt instead of only a start command?

The prompt fixes the config, deployment, environment and authorization for one supervising Agent. The Agent owns preflight, monitoring, recovery and terminal acceptance through the Run Skill. A start command alone does not provide that operator-side lifecycle. See [Quickstart](quickstart.md) and [supervision](components/supervision.md).
