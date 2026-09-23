"""Engine-only execution of an admitted Data Selection script."""

from __future__ import annotations

import asyncio
from collections import Counter
import json

from ade.tasks.data_selection.selection_contract import (
    validate_selection_source,
)

_MAX_SCRIPT_BYTES = 65_536
_MAX_SOURCE_BYTES = 128 * 1024 * 1024
_MAX_INVENTORY_GROUPS = 100_000


def execute_selection(
    script: bytes,
    candidate_inventory: bytes,
    training_dataset: bytes,
    *,
    select_size: int,
    judge_batch=None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    return asyncio.run(execute_selection_async(
        script, candidate_inventory, training_dataset,
        select_size=select_size, judge_batch=judge_batch,
    ))


async def execute_selection_async(
    script: bytes,
    candidate_inventory: bytes,
    training_dataset: bytes,
    *,
    select_size: int,
    judge_batch=None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not script or len(script) > _MAX_SCRIPT_BYTES:
        raise ValueError("selection.py exceeds the Engine size limit")
    if len(candidate_inventory) > _MAX_SOURCE_BYTES or len(training_dataset) > _MAX_SOURCE_BYTES:
        raise ValueError("Data Selection source exceeds the Engine size limit")
    try:
        source = script.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("selection.py must be UTF-8") from error
    tree = validate_selection_source(source)
    inventory = _jsonl(candidate_inventory, "candidate_inventory")
    if len(inventory) > _MAX_INVENTORY_GROUPS:
        raise ValueError("candidate_inventory exceeds the Engine group limit")
    namespace: dict[str, object] = {}
    namespace["json"] = json
    exec(compile(tree, "selection.py", "exec"), namespace, namespace)
    entrypoint = namespace.get("select_trajectories")
    if not callable(entrypoint):
        raise ValueError("selection.py entrypoint is unavailable")
    if type(select_size) is not int or select_size <= 0:
        raise ValueError("Data Selection select_size must be positive")
    if judge_batch is None:
        async def judge_batch(*_args, **_kwargs):
            raise RuntimeError("Judge enrichment is disabled for this selection artifact")
    training_rows = _jsonl(training_dataset, "training_dataset")
    enriched_inventory = _with_training_examples(inventory, training_rows)
    selected = await entrypoint(
        json.loads(json.dumps(enriched_inventory)), select_size, judge_batch
    )
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(item, str) or not item for item in selected)
        or len(selected) != len(set(selected))
    ):
        raise ValueError("selection.py must return unique non-empty trajectory IDs")
    if len(selected) != select_size:
        raise ValueError("selection.py returned the wrong number of trajectory IDs")
    return _materialize(inventory, _jsonl(training_dataset, "training_dataset"), selected)


def _with_training_examples(
    inventory: list[dict[str, object]],
    training: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_trajectory: dict[str, dict[str, object]] = {}
    for row in training:
        trajectory_id = _text(row.get("trajectory_id"), "trajectory_id")
        if trajectory_id in by_trajectory:
            raise ValueError(f"duplicate training trajectory_id: {trajectory_id}")
        by_trajectory[trajectory_id] = row
    enriched: list[dict[str, object]] = []
    for group in inventory:
        trajectory_ids = group.get("trajectory_ids")
        if not isinstance(trajectory_ids, list):
            raise ValueError("candidate inventory trajectory_ids must be a list")
        examples = []
        for trajectory_id in trajectory_ids:
            if not isinstance(trajectory_id, str) or trajectory_id not in by_trajectory:
                raise ValueError(
                    f"candidate inventory trajectory lacks training example: {trajectory_id}"
                )
            examples.append(dict(by_trajectory[trajectory_id]))
        enriched.append({**group, "training_examples": examples})
    return enriched


def _materialize(
    inventory: list[dict[str, object]],
    training: list[dict[str, object]],
    selected_ids: list[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    groups: dict[str, dict[str, object]] = {}
    trajectory_to_group: dict[str, str] = {}
    for row in inventory:
        candidate_id = _text(row.get("candidate_id"), "candidate_id")
        trajectory_ids = row.get("trajectory_ids")
        if (
            not isinstance(trajectory_ids, list)
            or not trajectory_ids
            or any(not isinstance(item, str) or not item for item in trajectory_ids)
            or len(trajectory_ids) != len(set(trajectory_ids))
        ):
            raise ValueError(f"candidate {candidate_id} has invalid trajectory_ids")
        if candidate_id in groups:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        if set(trajectory_ids).intersection(trajectory_to_group):
            raise ValueError("trajectory IDs appear in multiple candidate groups")
        token_count = row.get("token_count")
        if type(token_count) is not int or token_count <= 0:
            raise ValueError(f"candidate {candidate_id} has invalid token_count")
        group = {
            "candidate_id": candidate_id,
            "problem_id": _text(row.get("problem_id"), "problem_id"),
            "trajectory_ids": list(trajectory_ids),
            "token_count": token_count,
            "domain": _text(row.get("domain"), "domain"),
            "difficulty": _text(row.get("difficulty"), "difficulty"),
            "length_bin": _text(row.get("length_bin"), "length_bin"),
        }
        groups[candidate_id] = group
        trajectory_to_group.update((item, candidate_id) for item in trajectory_ids)
    unknown = sorted(set(selected_ids) - set(trajectory_to_group))
    if unknown:
        raise ValueError(f"unknown trajectory IDs: {unknown}")
    selected_group_ids = {trajectory_to_group[item] for item in selected_ids}
    selected_groups = [groups[item] for item in sorted(selected_group_ids)]
    complete_ids = {
        item for group in selected_groups for item in group["trajectory_ids"]
    }
    if complete_ids != set(selected_ids):
        raise ValueError("Data Selection must select complete trajectory groups")
    total_tokens = sum(int(group["token_count"]) for group in selected_groups)
    training_by_id: dict[str, dict[str, object]] = {}
    for row in training:
        trajectory_id = _text(row.get("trajectory_id"), "trajectory_id")
        if trajectory_id in training_by_id:
            raise ValueError(f"duplicate training trajectory_id: {trajectory_id}")
        conversations = row.get("conversations")
        if conversations is not None:
            normalized_conversations = _normalize_conversations(
                conversations, trajectory_id
            )
            row = {**row, "conversations": normalized_conversations}
        else:
            for field in ("instruction", "input", "output"):
                if not isinstance(row.get(field), str):
                    raise ValueError(f"training_dataset rows require string {field}")
        training_by_id[trajectory_id] = row
    missing = [item for item in selected_ids if item not in training_by_id]
    if missing:
        raise ValueError(f"selected trajectories are not present in training_dataset: {missing}")
    selected_rows = [
        {
            key: value
            for key, value in training_by_id[item].items()
            if key not in {"trajectory_id", "target_contract"}
        }
        for item in selected_ids
    ]
    canonical_groups = [
        {
            "candidate_id": group["candidate_id"],
            "problem_id": group["problem_id"],
            "trajectory_ids": group["trajectory_ids"],
        }
        for group in selected_groups
    ]
    selection = {
        "schema_version": "1",
        "selected_ids": selected_ids,
        "selection_size": len(selected_ids),
        "selected_groups": canonical_groups,
        "total_tokens": total_tokens,
        "statistics": {
            "length_bins": dict(sorted(Counter(group["length_bin"] for group in selected_groups).items())),
            "difficulty_bins": dict(sorted(Counter(group["difficulty"] for group in selected_groups).items())),
            "domain_counts": dict(sorted(Counter(group["domain"] for group in selected_groups).items())),
        },
        "validation": {
            "unique_ids": True,
            "group_integrity": True,
        },
    }
    return selection, selected_rows


def _normalize_conversations(
    value: object,
    trajectory_id: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"training trajectory {trajectory_id} has invalid conversations")
    normalized: list[dict[str, str]] = []
    role_map = {
        "user": "user",
        "assistant": "assistant",
    }
    for message in value:
        if not isinstance(message, dict):
            raise ValueError(f"training trajectory {trajectory_id} has invalid message")
        role = role_map.get(message.get("from"))
        content = message.get("value")
        if role is None or not isinstance(content, str):
            raise ValueError(f"training trajectory {trajectory_id} has invalid role/content")
        normalized.append({"from": role, "value": content})
    if [message["from"] for message in normalized] != ["user", "assistant"]:
        raise ValueError(
            f"training trajectory {trajectory_id} must contain exactly user -> assistant"
        )
    target = normalized[1]["value"]
    if (
        not target.startswith("<think>")
        or target.count("<think>") != 1
        or target.count("</think>") != 1
        or not target.split("</think>", 1)[1].strip()
    ):
        raise ValueError(f"training trajectory {trajectory_id} violates thinking target contract")
    return normalized


def _jsonl(content: bytes, label: str) -> list[dict[str, object]]:
    try:
        values = [
            json.loads(line)
            for line in content.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be UTF-8 JSONL") from error
    if not values or any(not isinstance(item, dict) for item in values):
        raise ValueError(f"{label} must contain JSON objects")
    return values


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Data Selection source requires {label}")
    return value
