"""Execute an admitted Curriculum policy once against frozen inputs."""

from __future__ import annotations

import ast
import asyncio
import copy
from collections import Counter
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping

from ade.rubric_jobs.process import canonical_process_rubric
from ade.tasks.curriculum_learning.schedule_contract import (
    canonical_step_lists,
    validate_curriculum_source,
)


JudgeBatch = Callable[
    [list[dict[str, object]]], Awaitable[list[dict[str, object]]]
]


@dataclass(frozen=True)
class CurriculumRealization:
    schedule: dict[str, object]
    judge_evidence: tuple[dict[str, object], ...]


async def realize_curriculum(
    script: bytes,
    inventory: list[dict[str, object]],
    *,
    total_steps: int,
    prompts_per_step: int,
    rollout_n: int,
    pool_stats: Mapping[str, int],
    judge_batch: JudgeBatch | None,
) -> CurriculumRealization:
    try:
        source = script.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("curriculum.py must be UTF-8") from error
    tree = validate_curriculum_source(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "judge_batch"
    ]
    for call in calls:
        if call.keywords or len(call.args) != 1 or not any(
            isinstance(node, ast.Await) and node.value is call
            for node in ast.walk(tree)
        ):
            raise ValueError(
                "curriculum.py Judge calls must await judge_batch with one request list"
            )
    evidence_log: list[dict[str, object]] = []

    async def admitted_judge(
        requests: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if judge_batch is None:
            raise ValueError("curriculum.py called Judge while enrichment is unavailable")
        if not isinstance(requests, list) or not requests:
            raise ValueError("Curriculum Judge batch must be a non-empty list")
        normalized: list[dict[str, object]] = []
        for request in requests:
            if not isinstance(request, dict) or set(request) != {
                "question",
                "response",
                "rubric",
            }:
                raise ValueError(
                    "Curriculum Judge requests require question, response, and rubric"
                )
            question = request["question"]
            response = request["response"]
            if not isinstance(question, str) or not isinstance(response, str):
                raise ValueError("Curriculum Judge question/response must be strings")
            normalized.append(
                {
                    "question": question,
                    "response": response,
                    "rubric": canonical_process_rubric(request["rubric"]),
                }
            )
        results = await judge_batch(copy.deepcopy(normalized))
        if (
            not isinstance(results, list)
            or len(results) != len(normalized)
            or any(not isinstance(result, dict) for result in results)
        ):
            raise ValueError("Curriculum Judge evidence does not match requests")
        evidence_log.extend(
            {"request": request, "evidence": copy.deepcopy(result)}
            for request, result in zip(normalized, results, strict=True)
        )
        return copy.deepcopy(results)

    namespace: dict[str, object] = {}
    exec(compile(tree, "curriculum.py", "exec"), namespace, namespace)
    entrypoint = namespace.get("build_curriculum")
    if not callable(entrypoint):
        raise ValueError("curriculum.py entrypoint is unavailable")
    try:
        proposed = await entrypoint(
            copy.deepcopy(inventory),
            total_steps,
            prompts_per_step,
            admitted_judge,
        )
    except Exception as error:
        raise ValueError(
            f"curriculum.py realization failed: {type(error).__name__}: {error}"
        ) from error
    steps = canonical_step_lists(
        proposed,
        inventory=inventory,
        total_steps=total_steps,
        prompts_per_step=prompts_per_step,
    )
    by_id = {str(row["problem_id"]): row for row in inventory}
    reuse = Counter(problem_id for step in steps for problem_id in step)
    subject_counts: list[dict[str, int]] = []
    level_counts: list[dict[str, int]] = []
    for step in steps:
        subject_counts.append(
            dict(sorted(Counter(str(by_id[item]["subject"]) for item in step).items()))
        )
        level_counts.append(
            dict(sorted(Counter(str(by_id[item]["level"]) for item in step).items()))
        )
    schedule = {
        "schema_version": "ade.curriculum_schedule.v1",
        "selection_unit": "problem",
        "total_steps": total_steps,
        "prompts_per_step": prompts_per_step,
        "rollout_n": rollout_n,
        "candidate_pool": dict(pool_stats),
        "candidates": [
            {
                key: row[key]
                for key in (
                    "problem_id",
                    "source_order",
                    "subject",
                    "level",
                    "prompt_tokens",
                )
            }
            for row in inventory
        ],
        "steps": [
            {"step": index, "problem_ids": problem_ids}
            for index, problem_ids in enumerate(steps, 1)
        ],
        "summary": {
            "unique_problem_count": len(reuse),
            "reuse_count_distribution": dict(
                sorted(Counter(reuse.values()).items())
            ),
            "per_step_subject_counts": subject_counts,
            "per_step_level_counts": level_counts,
        },
    }
    return CurriculumRealization(schedule, tuple(evidence_log))


def realize_curriculum_sync(*args, **kwargs) -> CurriculumRealization:
    return asyncio.run(realize_curriculum(*args, **kwargs))
