from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import EvalExample
from ..utils.qwen_math.task import QWEN_PROMPT, QwenMathEvaluationTask


class MathValTask(QwenMathEvaluationTask):
    """Canonical math validation parquet task."""

    task_type = "math_val"
    parser_data_name = "math"

    def load_examples(self, path: str | Path, *, dataset_name: str | None = None,
                      dataset_config: dict[str, Any] | None = None) -> list[EvalExample]:
        del dataset_config
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ImportError("Math validation parquet requires pyarrow in the ADE runtime") from exc
        return [self.row_to_example(row, idx, dataset_name=dataset_name)
                for idx, row in enumerate(parquet.read_table(str(path)).to_pylist())]

    def row_to_example(self, row: dict[str, Any], idx: int, *, dataset_name: str | None = None) -> EvalExample:
        prompt = ""
        prompt_messages = row.get("prompt")
        if isinstance(prompt_messages, list):
            for message in reversed(prompt_messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    prompt = str(message.get("content") or "")
                    break
        if not prompt:
            extra_info = row.get("extra_info")
            if isinstance(extra_info, dict) and extra_info.get("question"):
                prompt = QWEN_PROMPT.format(problem=extra_info["question"])
        if not prompt:
            raise ValueError(f"{self.task_type} dataset row {idx} has no user prompt or extra_info.question")
        reward_model = row.get("reward_model")
        if not isinstance(reward_model, dict) or reward_model.get("ground_truth") is None:
            raise ValueError(f"{self.task_type} dataset row {idx} is missing reward_model.ground_truth")
        unique_id = row.get("unique_id")
        if not isinstance(unique_id, str) or not unique_id:
            raise ValueError(f"{self.task_type} dataset row {idx} is missing stable unique_id")
        return EvalExample(
            id=unique_id,
            prompt=prompt,
            reference=str(reward_model["ground_truth"]),
            row_index=idx,
            meta={"dataset_name": dataset_name or "math_val",
                  "data_source": row.get("data_source"),
                  "reference_source": "reward_model.ground_truth"},
        )
