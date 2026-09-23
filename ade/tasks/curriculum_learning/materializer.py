"""Materialize a canonical problem schedule as sequential VERL parquet rows."""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Mapping


def materialize_scheduled_parquet(
    source_path: str | Path,
    schedule: Mapping[str, object],
) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    steps = schedule.get("steps")
    prompts_per_step = schedule.get("prompts_per_step")
    total_steps = schedule.get("total_steps")
    if (
        schedule.get("schema_version") != "ade.curriculum_schedule.v1"
        or type(prompts_per_step) is not int
        or type(total_steps) is not int
        or not isinstance(steps, list)
        or len(steps) != total_steps
    ):
        raise ValueError("Curriculum schedule is incomplete")
    source = pq.read_table(Path(source_path))
    if "unique_id" not in source.schema.names or "extra_info" not in source.schema.names:
        raise ValueError("Curriculum source parquet lacks unique_id/extra_info")
    by_id: dict[str, dict[str, object]] = {}
    for row in source.to_pylist():
        problem_id = row.get("unique_id")
        if not isinstance(problem_id, str) or not problem_id or problem_id in by_id:
            raise ValueError("Curriculum source unique_id is invalid or duplicated")
        by_id[problem_id] = row
    scheduled: list[dict[str, object]] = []
    for expected_step, step in enumerate(steps, 1):
        problem_ids = step.get("problem_ids") if isinstance(step, dict) else None
        if (
            step.get("step") != expected_step
            or not isinstance(problem_ids, list)
            or len(problem_ids) != prompts_per_step
        ):
            raise ValueError(f"Curriculum step {expected_step} is invalid")
        for slot, problem_id in enumerate(problem_ids):
            source_row = by_id.get(problem_id)
            if source_row is None:
                raise ValueError(
                    f"Curriculum step {expected_step} references unknown problem"
                )
            row = copy.deepcopy(source_row)
            extra = row.get("extra_info")
            if not isinstance(extra, dict):
                raise ValueError(f"Curriculum problem {problem_id} lacks extra_info")
            row["extra_info"] = {
                **extra,
                "sample_uid": problem_id,
                "curriculum_step": expected_step,
                "curriculum_slot": slot,
            }
            scheduled.append(row)
    expected_rows = total_steps * prompts_per_step
    if len(scheduled) != expected_rows:
        raise ValueError("Curriculum scheduled row count is inconsistent")
    table = pa.Table.from_pylist(scheduled)
    output = io.BytesIO()
    pq.write_table(table, output)
    return output.getvalue()
