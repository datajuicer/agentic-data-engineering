"""Task-owned deterministic random baseline for Data Selection."""

from __future__ import annotations

import json
import random
from typing import Mapping

from ade.tasks.contracts import BaselineArtifactRequest, CompiledArtifact
from ade.tasks.data_selection.fixed_pool import build_fixed_pool_from_task


def build_baseline(request: BaselineArtifactRequest) -> CompiledArtifact:
    groups = _canonical_singleton_groups(request.task_config)
    if groups is None:
        inventory, _training, _stats = build_fixed_pool_from_task(request.task_config)
        groups = _groups(inventory)
    selection_size = _positive_int(request.task_config.get("select_size"), "select_size")
    selected = _sample_complete_groups(groups, selection_size, request.seed)
    selected_ids = [
        trajectory_id
        for group in selected
        for trajectory_id in group["trajectory_ids"]
    ]
    source = (
        "async def select_trajectories(candidate_inventory, select_size, judge):\n"
        f"    return {selected_ids!r}\n"
    ).encode()
    return CompiledArtifact(
        path="selection.py",
        kind="data_selection",
        content=source,
    )


def _canonical_singleton_groups(
    task_config: Mapping[str, object],
) -> list[dict[str, object]] | None:
    data = task_config.get("data")
    if not isinstance(data, Mapping) or "fixed_training_data" not in data:
        return None
    pool_size = _positive_int(data.get("pool_size"), "data.pool_size")
    return [
        {
            "candidate_id": trajectory_id,
            "trajectory_ids": [trajectory_id],
        }
        for row_index in range(pool_size)
        for trajectory_id in (f"openthoughts-{row_index:04d}",)
    ]


def _sample_complete_groups(
    groups: list[dict[str, object]],
    selection_size: int,
    seed: int,
) -> list[dict[str, object]]:
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    choices: dict[int, list[dict[str, object]]] = {0: []}
    for group in shuffled:
        size = len(group["trajectory_ids"])
        for current in sorted(tuple(choices), reverse=True):
            total = current + size
            if total <= selection_size and total not in choices:
                choices[total] = [*choices[current], group]
    selected = choices.get(selection_size)
    if selected is None:
        raise ValueError(
            "Data Selection baseline cannot satisfy select_size with complete groups"
        )
    return sorted(selected, key=lambda group: str(group["candidate_id"]))


def _groups(content: bytes) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen_candidates: set[str] = set()
    seen_trajectories: set[str] = set()
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"candidate inventory line {line_number} is invalid JSON"
            ) from error
        if not isinstance(row, dict):
            raise ValueError("candidate inventory rows must be objects")
        candidate_id = _text(row.get("candidate_id"), "candidate_id")
        problem_id = _text(row.get("problem_id"), "problem_id")
        trajectory_ids = row.get("trajectory_ids")
        if (
            not isinstance(trajectory_ids, list)
            or not trajectory_ids
            or any(not isinstance(item, str) or not item for item in trajectory_ids)
            or len(trajectory_ids) != len(set(trajectory_ids))
        ):
            raise ValueError(f"candidate {candidate_id} has invalid trajectory_ids")
        if candidate_id in seen_candidates:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        overlap = seen_trajectories.intersection(trajectory_ids)
        if overlap:
            raise ValueError(f"trajectory IDs appear in multiple groups: {sorted(overlap)}")
        token_count = row.get("token_count")
        if type(token_count) is not int or token_count <= 0:
            raise ValueError(f"candidate {candidate_id} has invalid token_count")
        result.append(
            {
                "candidate_id": candidate_id,
                "problem_id": problem_id,
                "trajectory_ids": list(trajectory_ids),
                "token_count": token_count,
                "domain": _text(row.get("domain"), "domain"),
                "difficulty": _text(row.get("difficulty"), "difficulty"),
                "length_bin": _text(row.get("length_bin"), "length_bin"),
            }
        )
        seen_candidates.add(candidate_id)
        seen_trajectories.update(trajectory_ids)
    if not result:
        raise ValueError("candidate inventory is empty")
    return result


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"Data Selection baseline requires a positive {label}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate inventory requires {label}")
    return value
