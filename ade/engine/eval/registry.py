from __future__ import annotations

from typing import Any

from .tasks.aime24 import AIME24Task
from .tasks.aime24_smoke16 import AIME24Smoke16Task
from .tasks.aime25 import AIME25Task
from .tasks.aime25_smoke16 import AIME25Smoke16Task
from .tasks.aime26 import AIME26Task
from .utils.base import EvalTask
from .tasks.codeelo import CodeELODatasetTask
from .tasks.codeforces import CodeforcesDatasetTask
from .tasks.hmmt import HMMTTask
from .tasks.humaneval_plus import HumanEvalPlusDatasetTask
from .tasks.livecodebench_gt_2024_03_30 import LiveCodeBenchGT20240330Task
from .tasks.livecodebench_le_2024_03_30 import LiveCodeBenchLE20240330Task
from .tasks.math_100 import Math100Task
from .tasks.math_500 import Math500DatasetTask
from .tasks.math_500_smoke16 import Math500Smoke16Task
from .tasks.math_val import MathValTask
from .tasks.mbpp_plus import MBPPPlusDatasetTask
from .tasks.math_val_smoke16 import MathValSmoke16Task


_TASKS: dict[str, type[EvalTask]] = {
    "aime24": AIME24Task,
    "aime24_smoke16": AIME24Smoke16Task,
    "math_100": Math100Task,
    "math_val": MathValTask,
    "math_val_smoke16": MathValSmoke16Task,
    "math_500": Math500DatasetTask,
    "math_500_smoke16": Math500Smoke16Task,
    "aime25": AIME25Task,
    "aime25_smoke16": AIME25Smoke16Task,
    "aime26": AIME26Task,
    "hmmt": HMMTTask,
    "livecodebench_le_2024_03_30": LiveCodeBenchLE20240330Task,
    "livecodebench_gt_2024_03_30": LiveCodeBenchGT20240330Task,
    "humaneval_plus": HumanEvalPlusDatasetTask,
    "mbpp_plus": MBPPPlusDatasetTask,
    "codeelo": CodeELODatasetTask,
    "codeforces": CodeforcesDatasetTask,
}

CANONICAL_TASK_IDS = tuple(_TASKS)


def task_type_from_config(
    dataset: dict[str, Any] | None = None,
    request: dict[str, Any] | None = None,
    answer_format: str | None = None,
) -> str:
    del request, answer_format
    dataset = dataset or {}
    dataset_id = dataset.get("name")
    task_type = dataset.get("task_type")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("evaluation dataset requires canonical name")
    if not isinstance(task_type, str) or not task_type:
        raise ValueError(f"evaluation dataset {dataset_id!r} requires canonical task_type")
    if task_type != dataset_id:
        raise ValueError(
            f"evaluation dataset {dataset_id!r} must bind its same-name task, got {task_type!r}"
        )
    if task_type not in _TASKS:
        raise ValueError(f"unsupported canonical evaluation task_type={task_type!r}")
    return task_type


def get_task(task_type: str | None) -> EvalTask:
    if not isinstance(task_type, str) or task_type not in _TASKS:
        raise ValueError(
            f"unsupported canonical evaluation task_type={task_type!r}; "
            f"supported={list(CANONICAL_TASK_IDS)!r}"
        )
    return _TASKS[task_type]()
