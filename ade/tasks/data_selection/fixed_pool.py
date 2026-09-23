"""Build the canonical full OpenThoughts SFT selection pool."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping


CANONICAL_SFT_CUTOFF_LEN = 16_384
THINKING_TOKENS = ("<think>", "</think>")


def build_fixed_pool_from_task(
    task_config: Mapping[str, object],
) -> tuple[bytes, bytes, dict[str, int]]:
    data = task_config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Data Selection task config requires data")
    source_path = data.get("fixed_training_data")
    pool_size = data.get("pool_size")
    model_path = task_config.get("base_model")
    prompt_protocol = task_config.get("prompt_protocol")
    if (
        not isinstance(source_path, str)
        or not isinstance(model_path, str)
        or not isinstance(prompt_protocol, Mapping)
        or prompt_protocol.get("mode") != "chat_template"
        or not isinstance(prompt_protocol.get("system_prompt"), Mapping)
    ):
        raise ValueError("Data Selection fixed pool binding is incomplete")
    system_prompt = prompt_protocol["system_prompt"].get("content")
    if not isinstance(system_prompt, str) or not system_prompt:
        raise ValueError("Data Selection fixed pool system prompt is missing")
    inventory, training, stats = build_fixed_pool(source_path, model_path, system_prompt)
    if type(pool_size) is not int or pool_size <= 0:
        raise ValueError("Data Selection task.data.pool_size must be positive")
    if stats.get("source_rows") != pool_size:
        raise ValueError(
            "Data Selection fixed pool row count does not match task.data.pool_size: "
            f"expected={pool_size}, actual={stats.get('source_rows')}"
        )
    return inventory, training, stats


@lru_cache(maxsize=8)
def build_fixed_pool(
    source_path: str,
    model_path: str,
    system_prompt: str,
    cutoff_len: int = CANONICAL_SFT_CUTOFF_LEN,
) -> tuple[bytes, bytes, dict[str, int]]:
    """Return all source rows; LlamaFactory owns cutoff-time truncation."""

    if cutoff_len != CANONICAL_SFT_CUTOFF_LEN:
        raise ValueError(
            f"canonical SFT cutoff_len must be {CANONICAL_SFT_CUTOFF_LEN}"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(THINKING_TOKENS)}
    )
    source = Path(source_path)
    inventory: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    total_rows = 0
    truncated_rows = 0
    for row_index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        total_rows += 1
        row = json.loads(line)
        conversations = row.get("conversations") if isinstance(row, dict) else None
        messages, assistant_target = _messages(conversations, row_index)
        rendered_messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]
        token_ids = tokenizer.apply_chat_template(
            rendered_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        rendered_token_count = len(token_ids)
        will_truncate = rendered_token_count > cutoff_len
        if will_truncate:
            truncated_rows += 1
        effective_token_count = min(rendered_token_count, cutoff_len)
        trajectory_id = f"openthoughts-{row_index:04d}"
        inventory.append(
            {
                "candidate_id": trajectory_id,
                "problem_id": trajectory_id,
                "trajectory_ids": [trajectory_id],
                "token_count": effective_token_count,
                "domain": "math_code_science",
                "difficulty": "unlabeled",
                "length_bin": _length_bin(effective_token_count),
                "quality_score": 1.0,
            }
        )
        training.append(
            {
                "trajectory_id": trajectory_id,
                "conversations": [
                    {"from": message["role"], "value": message["content"]}
                    for message in messages
                ],
                "target_contract": {
                    "starts_with_open_tag": assistant_target.startswith("<think>"),
                    "paired_thinking_tags": True,
                    "nonempty_final_answer": True,
                    "source_rendered_token_count": rendered_token_count,
                    "training_cutoff_len": cutoff_len,
                    "will_truncate": will_truncate,
                },
            }
        )
    if not inventory:
        raise ValueError("canonical SFT fixed pool has no rows")
    return (
        _jsonl_bytes(inventory),
        _jsonl_bytes(training),
        {
            "source_rows": total_rows,
            "training_rows": len(inventory),
            "truncated_rows": truncated_rows,
            "cutoff_len": cutoff_len,
        },
    )


def _messages(
    conversations: object,
    row_index: int,
) -> tuple[list[dict[str, str]], str]:
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"OpenThoughts row {row_index} has no conversations")
    messages: list[dict[str, str]] = []
    assistant_targets: list[str] = []
    role_map = {
        "user": "user",
        "assistant": "assistant",
    }
    for message in conversations:
        if not isinstance(message, dict):
            raise ValueError(f"OpenThoughts row {row_index} has invalid message")
        role = role_map.get(message.get("from"))
        content = message.get("value")
        if role is None or not isinstance(content, str):
            raise ValueError(f"OpenThoughts row {row_index} has invalid role/content")
        messages.append({"role": role, "content": content})
        if role == "assistant":
            assistant_targets.append(content)
    if [message["role"] for message in messages] != ["user", "assistant"]:
        raise ValueError(
            f"OpenThoughts row {row_index} must contain exactly user -> assistant"
        )
    if len(assistant_targets) != 1:
        raise ValueError(
            f"OpenThoughts row {row_index} must have exactly one assistant target"
        )
    target = assistant_targets[0]
    if (
        not target.startswith("<think>")
        or target.count("<think>") != 1
        or target.count("</think>") != 1
        or not target.split("</think>", 1)[1].strip()
    ):
        raise ValueError(
            f"OpenThoughts row {row_index} violates the thinking target contract"
        )
    return messages, target


def _length_bin(token_count: int) -> str:
    if token_count <= 4_096:
        return "le_4096"
    if token_count <= 8_192:
        return "4097_8192"
    return "8193_16384"


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode("utf-8")
