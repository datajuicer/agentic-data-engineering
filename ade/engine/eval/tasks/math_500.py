from __future__ import annotations

from typing import Any

from ..contracts import EvalExample
from ..utils.base import first_present, require_first_present
from ..utils.qwen_math import extract_answer
from ..utils.qwen_math.task import QWEN_PROMPT, QwenMathEvaluationTask


class Math500DatasetTask(QwenMathEvaluationTask):
    """Canonical MATH500 task using the SimpleRL Qwen Math evaluator."""

    task_type = "math_500"
    parser_data_name = "math500"

    def row_to_example(
        self,
        row: dict[str, Any],
        idx: int,
        *,
        dataset_name: str | None = None,
    ) -> EvalExample:
        problem = require_first_present(
            row,
            ("problem", "question"),
            task_type=self.task_type,
            row_index=idx,
            field_name="problem",
        )
        solution = require_first_present(
            row,
            ("solution",),
            task_type=self.task_type,
            row_index=idx,
            field_name="solution",
        )
        reference = extract_answer(str(solution), self.parser_data_name)
        example_id = str(
            first_present(row, ("unique_id", "id", "index", "task_id"))
            or f"{dataset_name or 'math_500'}/{idx}"
        )
        return EvalExample(
            id=example_id,
            prompt=QWEN_PROMPT.format(problem=problem),
            reference=reference,
            row_index=idx,
            meta={
                "dataset_name": dataset_name or "math_500",
                "reference_source": "solution:qwen_math.extract_answer",
                "reference_full": str(solution),
                "subject": row.get("subject"),
                "level": row.get("level"),
            },
        )
