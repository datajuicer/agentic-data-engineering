# Installation

[Quickstart](quickstart.md) | [Models and data](data-preparation.md) | [Deployment](deployment.md)

The installer sets up ADE, LlamaFactory, VERL and the Local Judge using the pinned dependencies in `requirements/unified-vllm-0.19.1-verl-venv.txt`. Run the commands below from the repository root.

## Prerequisites

- Linux x86-64 and Python 3.11. The environment script uses uv-managed Python by default, avoiding DSW-specific interpreter and system-library paths.
- Install `uv` and Git. For example, use `python3 -m pip install uv` in your tool environment.
- For the full runtime: an NVIDIA driver compatible with the pinned Torch/CUDA stack, CUDA development tools including `nvcc`, and a C/C++ compiler when building FlashAttention. The runtime snapshot contains Torch 2.10.0 and CUDA 12.8 libraries.
- Access to the package index and sufficient disk space for the environment, model weights, datasets and checkpoints. Model weights and datasets are not included in this repository.

Work from the repository root. All repository commands use `<repository>/.unified-vllm-0.19.1-verl-venv`. On compatible nodes sharing the checkout at the same absolute path, install once and activate it on each node. Otherwise install at the configured path on each node. A custom node-local `ENV_DIR` requires matching backend environment paths; do not concurrently build one shared environment.

## Install the unified environment

The single full-runtime installation entry point is below. Use `bash scripts/recreate_unified_vllm_env.sh --help` to view its settings without installing anything. Missing `uv`, `timeout`, a supplied wheel, or the CUDA compiler for a source build is reported before environment installation.

Install with:

```bash
bash scripts/recreate_unified_vllm_env.sh
source .unified-vllm-0.19.1-verl-venv/bin/activate
ade --help
```

The script installs the version snapshot, builds FlashAttention after Torch, installs this checkout's ADE/LlamaFactory/VERL in editable mode, applies the TransferQueue namespace patch, and installs the separate Judge packages. It then checks package dependencies, scoring imports, task registration and CLI startup. These checks launch no GPU jobs or services.

The script does not download models or datasets, authenticate Codex, start Ray, or start the Judge. Re-running installs into the selected environment; it does not delete that environment. Use a fresh target when validating a clean installation. Do not run it against an environment serving active jobs.

When compatible Linux nodes share this checkout at the same absolute path, install the shared environment once. Both `.python/` and the virtual environment must be accessible on every node; then activate the environment on each node. A custom `PYTHON_BIN` or `UV_PYTHON_INSTALL_DIR` must also be accessible there. Do not run simultaneous installers against one shared environment.

| Setting | Default / usage |
| --- | --- |
| `ENV_DIR` | `<repository>/.unified-vllm-0.19.1-verl-venv`; use an absolute path for a custom node-local environment |
| `PYTHON_BIN` | Unset: uv-managed Python 3.11. Set to a standard Python 3.11 executable to use your own interpreter |
| `UV_PYTHON_INSTALL_DIR` | `<repository>/.python`, keeping the managed interpreter alongside the environment |
| `UV_INDEX_URL` | `https://pypi.org/simple`; can be set to your package mirror |
| `FLASH_ATTN_WHEEL` | Optional local wheel for FlashAttention 2.8.3.post1, matching Python, Torch and CUDA ABI |
| `CUDA_HOME` | Set when your CUDA toolkit is outside its normal discovery path |
| `MAX_JOBS` | `4`, for extension builds |
| `VALIDATE_TIMEOUT_SECONDS` | `120`, for import and Judge CLI checks |

Example with a local wheel:

```bash
FLASH_ATTN_WHEEL=/path/to/matching-flash-attn.whl \
  bash scripts/recreate_unified_vllm_env.sh
```

Build failures are reported directly; packages are not copied from another checkout. The installer respects your network/proxy configuration. The old cloned-environment VERL setup entry has been removed; the reconstruction script installs both included backends.

The backend sources are managed under `third_party/llamafactory/` and
`third_party/verl/`, with ADE adapters under `ade/engine/backends/`. If you already
installed the full environment before this directory layout, refresh its editable
bindings without rebuilding the runtime:

```bash
uv pip install --python .unified-vllm-0.19.1-verl-venv/bin/python \
  --no-deps --no-build-isolation \
  --editable third_party/llamafactory --editable third_party/verl
```

## Dependency ownership

| Component | Version/source |
| --- | --- |
| ADE package metadata | `pyproject.toml`; installed editable by the unified installer |
| Torch / vLLM / Ray | 2.10.0 / 0.19.1 / 2.55.1 |
| Training Transformers | 4.56.1 |
| FlashAttention / PyArrow | 2.8.3.post1 / 24.0.0 |
| Math scoring | math-verify 0.6.0, latex2sympy2-extended 1.0.9, word2number 1.1 |
| LlamaFactory / VERL | Source included in `third_party/llamafactory/` and `third_party/verl/` |
| Judge Transformers / FastAPI / Starlette | 4.57.1 / 0.115.0 / 0.38.6 in `<environment>/judge-packages` |

`ade-judge-vllm` loads only the Judge-specific overlay in its own process and passes it to vLLM child processes. Training uses the same Python environment without this process-local overlay; `judge-packages` is a subdirectory, not another virtual environment. The launcher retains the existing Transformers helper and metrics-routing adaptations for vLLM 0.19.1.

Use the reconstruction script to install or update this environment. `uv` project management is disabled in `pyproject.toml`; do not use `uv sync` or create a separate `.venv`. For commands, activate the environment or use its executable path directly. Do not install upstream VERL's optional `vllm` extra over this frozen environment.

## Next steps

Next, try the [CPU demo](cpu-demo.md), or prepare [models and datasets](data-preparation.md) and your [deployment](deployment.md) for a formal experiment.

Once the environment is accessible on each node, follow [Ray cluster setup](ray-cluster.md), then the [Run operations guide](operations.md). The installer starts neither Ray nor Judge.

Run `bash scripts/recreate_unified_vllm_env.sh --help` to list settings without installing. Missing `uv`, `timeout`, a supplied wheel or the CUDA compiler required for a source build is reported before environment installation. Relative `ENV_DIR` and `FLASH_ATTN_WHEEL` paths are resolved from the repository root.

## Hand the runtime to the supervising Agent

Installation does not start an experiment. Put this checkout's absolute path and the installed full environment's absolute path into the [Operation Prompt](quickstart.md). Prepare backend authentication, inputs, deployment and assigned Ray nodes before live handoff. The outer Agent uses the included `ade-supervise-run` Skill and the same unified environment. If its host does not discover repository Skills, provide `.agents/skills/ade-supervise-run/SKILL.md` explicitly. See the [walkthrough](walkthrough.md).
