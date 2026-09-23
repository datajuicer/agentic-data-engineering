from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    for env_name in ("ADE_PROJECT_ROOT", "WORK_DIR"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _project_root()
BASE_MODEL = PROJECT_ROOT / "Qwen" / "Qwen2.5-7B-Instruct"
LLAMA_FACTORY_ROOT = PROJECT_ROOT / "third_party/llamafactory"
DATA_ROOT = PROJECT_ROOT / "data"
RUNS_ROOT = PROJECT_ROOT / "runs"
TASK_QUEUE_ROOT = RUNS_ROOT / "queue"
DEFAULT_SYSTEM_PROMPT = PROJECT_ROOT / "prompts" / "system" / "sys_math.txt"
