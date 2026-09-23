from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalExample:
    id: str
    prompt: str
    reference: Any
    row_index: int
    choices: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutRecord:
    example: EvalExample
    repeat_idx: int
    output: str
    seed: int | None = None
    thinking_content: str = ""
    answer_content: str = ""
    num_output_tokens: int | None = None
    finish_reason: str | None = None
    stop_reason: Any = None


@dataclass
class EvalScore:
    task_type: str
    primary_metric: str
    score: float
    metrics: dict[str, Any]
    details: list[dict[str, Any]]


def stable_eval_case_id(
    *,
    task_type: str,
    dataset_name: str | None,
    repeat_index: int,
    row_index: int,
    example_id: str,
) -> str:
    raw = f"{task_type}:{dataset_name or ''}:{repeat_index}:{row_index}:{example_id}"
    return "case_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def make_eval_case(
    *,
    task_type: str,
    dataset_name: str | None,
    repeat_index: int,
    bucket: str,
    index: int,
    example_id: str,
    input: dict[str, Any] | None = None,
    gold: dict[str, Any] | None = None,
    prediction: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": stable_eval_case_id(
            task_type=task_type,
            dataset_name=dataset_name,
            repeat_index=repeat_index,
            row_index=index,
            example_id=example_id,
        ),
        "bucket": bucket,
        "index": index,
        "example_id": example_id,
        "input": dict(input or {}),
        "gold": dict(gold or {}),
        "prediction": dict(prediction or {}),
        "diagnostics": dict(diagnostics or {}),
        "metadata": dict(metadata or {}),
    }
