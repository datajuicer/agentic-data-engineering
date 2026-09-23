"""Build the tokenizer-eligible RFT problem inventory used by Curriculum Learning."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


def build_fixed_pool_from_task(
    task_config: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    data = task_config.get("data")
    rft = task_config.get("rft")
    model_path = task_config.get("base_model")
    if (
        not isinstance(data, Mapping)
        or not isinstance(rft, Mapping)
        or not isinstance(model_path, str)
    ):
        raise ValueError("Curriculum Learning fixed pool binding is incomplete")
    source_path = data.get("train")
    max_prompt_length = rft.get("max_prompt_length")
    if not isinstance(source_path, str) or type(max_prompt_length) is not int:
        raise ValueError("Curriculum Learning requires train and max_prompt_length")
    return build_fixed_pool(source_path, model_path, max_prompt_length)


@lru_cache(maxsize=8)
def build_fixed_pool(
    source_path: str,
    model_path: str,
    max_prompt_length: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive")
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    table = pq.read_table(Path(source_path))
    required = {"unique_id", "subject", "level", "question", "prompt", "answer"}
    missing = required - set(table.schema.names)
    if missing:
        raise ValueError(f"Curriculum training parquet is missing fields: {sorted(missing)}")
    inventory: list[dict[str, object]] = []
    seen: set[str] = set()
    overlength = 0
    for source_order, row in enumerate(table.to_pylist()):
        problem_id = _text(row.get("unique_id"), "unique_id")
        if problem_id in seen:
            raise ValueError(f"duplicate Curriculum problem ID: {problem_id}")
        seen.add(problem_id)
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"Curriculum problem {problem_id} has invalid prompt")
        token_count = len(
            tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        if token_count > max_prompt_length:
            overlength += 1
            continue
        inventory.append(
            {
                "problem_id": problem_id,
                "source_order": source_order,
                "subject": _text(row.get("subject"), "subject"),
                "level": row.get("level"),
                "prompt_tokens": token_count,
                "question": _text(row.get("question"), "question"),
                "reference_answer": _string(row.get("answer"), "answer"),
            }
        )
    if not inventory:
        raise ValueError("Curriculum eligible inventory is empty")
    return inventory, {
        "source_rows": table.num_rows,
        "eligible_rows": len(inventory),
        "overlength_rows": overlength,
    }


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Curriculum inventory requires {label}")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Curriculum inventory requires string {label}")
    return value
