#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Usage: bash scripts/recreate_unified_vllm_env.sh

Install ADE, vendored LlamaFactory/VERL and the Judge dependency overlay.
Install once per environment; compatible nodes may share it at the same path.
No Ray/Judge services are started.

Settings (environment variables):
  ENV_DIR          Target environment (default: <repo>/.unified-vllm-0.19.1-verl-venv)
  PYTHON_BIN       Optional Python 3.11 executable; otherwise uv-managed Python
  UV_PYTHON_INSTALL_DIR  Managed Python location (default: <repo>/.python)
  UV_INDEX_URL     Package index (default: https://pypi.org/simple)
  FLASH_ATTN_WHEEL Optional matching local FlashAttention 2.8.3.post1 wheel
  CUDA_HOME        CUDA toolkit location when not discoverable
  MAX_JOBS         Extension build parallelism (default: 4)
  VALIDATE_TIMEOUT_SECONDS  Import/CLI check timeout (default: 120)

Prerequisites: uv, timeout, and CUDA build tools when building FlashAttention.
Relative ENV_DIR and FLASH_ATTN_WHEEL paths are resolved from the repository root.
HELP
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "Unknown arguments. Use --help; configure installation with environment variables." >&2
  exit 2
fi
for requirement in uv timeout; do
  if ! command -v "${requirement}" >/dev/null 2>&1; then
    echo "Missing prerequisite: ${requirement}. See docs/en/installation.md." >&2
    exit 1
  fi
done

# Install the frozen ADE training/evaluation stack without using another checkout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_DIR="${ENV_DIR:-${ROOT}/.unified-vllm-0.19.1-verl-venv}"
UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.org/simple}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${ROOT}/.python}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export MAX_JOBS="${MAX_JOBS:-4}"

cd "${ROOT}"
if [[ -n "${FLASH_ATTN_WHEEL}" && ! -f "${FLASH_ATTN_WHEEL}" ]]; then
  echo "FLASH_ATTN_WHEEL does not exist: ${FLASH_ATTN_WHEEL}" >&2
  exit 1
fi
if [[ -z "${FLASH_ATTN_WHEEL}" ]]; then
  if [[ -n "${CUDA_HOME:-}" ]]; then
    NVCC="${CUDA_HOME}/bin/nvcc"
  else
    NVCC="$(command -v nvcc || true)"
  fi
  if [[ -z "${NVCC}" || ! -x "${NVCC}" ]]; then
    echo "FlashAttention build requires nvcc. Set CUDA_HOME or provide FLASH_ATTN_WHEEL." >&2
    exit 1
  fi
fi
if [[ -n "${PYTHON_BIN:-}" ]]; then
  uv venv --allow-existing --python "${PYTHON_BIN}" "${ENV_DIR}"
else
  # A managed Python avoids dependencies on DSW's relocated interpreter/libs.
  uv venv --allow-existing --managed-python --python 3.11 "${ENV_DIR}"
fi
PYTHON="${ENV_DIR}/bin/python"
"${PYTHON}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit("ADE requires Python 3.11")
PY

# Build FlashAttention only after the matching Torch and build tools exist.
FILTERED_SNAPSHOT="$(mktemp)"
trap 'rm -f "${FILTERED_SNAPSHOT}"' EXIT
grep -Eiv '^flash[_-]attn==' \
  "${ROOT}/requirements/unified-vllm-0.19.1-verl-venv.txt" > "${FILTERED_SNAPSHOT}"
uv pip install --python "${PYTHON}" --index-url "${UV_INDEX_URL}" \
  -r "${FILTERED_SNAPSHOT}"
FLASH_ATTN_SPEC="$(grep -Ei '^flash[_-]attn==' "${ROOT}/requirements/unified-vllm-0.19.1-verl-venv.txt")"
if [[ -n "${FLASH_ATTN_WHEEL}" ]]; then
  uv pip install --python "${PYTHON}" --no-deps "${FLASH_ATTN_WHEEL}"
else
  uv pip install --python "${PYTHON}" --index-url "${UV_INDEX_URL}" \
    --no-deps --no-build-isolation "${FLASH_ATTN_SPEC}"
fi

# Use the vendored backends without re-resolving their optional training stacks.
uv pip install --python "${PYTHON}" --no-deps --no-build-isolation \
  --editable "${ROOT}/third_party/verl" \
  --editable "${ROOT}/third_party/llamafactory" \
  --editable "${ROOT}"
"${PYTHON}" "${ROOT}/scripts/patch_transfer_queue_namespace.py"
uv pip check --python "${PYTHON}"

# Keep the existing production Judge versions outside the training import path.
uv pip install --python "${PYTHON}" --index-url "${UV_INDEX_URL}" \
  --no-deps --target "${ENV_DIR}/judge-packages" \
  -r "${ROOT}/requirements/judge.txt"

# Import/CLI checks allocate no GPUs and launch no Ray or Judge services.
CUDA_VISIBLE_DEVICES="" timeout "${VALIDATE_TIMEOUT_SECONDS:-120}" "${PYTHON}" - <<'PY'
import importlib.metadata
import ade
import flash_attn
import pyarrow
import torch
import vllm
from ade.engine.eval.utils.qwen_math.parser import extract_answer
from ade.tasks.registry import default_task_registry
for name in ("agentic-data-engineering", "llamafactory", "verl", "torch", "vllm", "flash-attn", "pyarrow", "math-verify"):
    print(f"{name}={importlib.metadata.version(name)}")
assert importlib.metadata.version("flash-attn") == "2.8.3.post1"
assert extract_answer(r"\boxed{42}", "math") == "42"
registry = default_task_registry()
for task_id in registry.task_ids():
    registry.get(task_id)
PY
"${ENV_DIR}/bin/ade" --help
CUDA_VISIBLE_DEVICES="" timeout "${VALIDATE_TIMEOUT_SECONDS:-120}" \
  "${ENV_DIR}/bin/ade-judge-vllm" serve --help >/dev/null
echo "ADE environment installed: ${ENV_DIR}"
