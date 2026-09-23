"""Task-owned deterministic random schedule baseline."""

from __future__ import annotations

import random

from ade.tasks.contracts import BaselineArtifactRequest, CompiledArtifact
from ade.tasks.curriculum_learning.fixed_pool import build_fixed_pool_from_task


def build_baseline(request: BaselineArtifactRequest) -> CompiledArtifact:
    inventory, _stats = build_fixed_pool_from_task(request.task_config)
    rft = request.task_config.get("rft")
    if not isinstance(rft, dict):
        raise ValueError("Curriculum baseline requires task.rft")
    total_steps = _positive_int(rft.get("total_training_steps"), "total_training_steps")
    prompts_per_step = _positive_int(rft.get("gen_batch_size"), "gen_batch_size")
    if len(inventory) < prompts_per_step:
        raise ValueError("Curriculum eligible pool cannot fill one step")
    problem_ids = [str(row["problem_id"]) for row in inventory]
    rng = random.Random(request.seed)
    steps: list[list[str]] = []
    while len(steps) < total_steps:
        epoch = list(problem_ids)
        rng.shuffle(epoch)
        complete = len(epoch) // prompts_per_step
        for offset in range(complete):
            start = offset * prompts_per_step
            steps.append(epoch[start : start + prompts_per_step])
            if len(steps) == total_steps:
                break
    source = (
        "async def build_curriculum(candidate_inventory, total_steps, "
        "prompts_per_step, judge_batch):\n"
        "    del candidate_inventory, judge_batch\n"
        f"    schedule = {steps!r}\n"
        "    if total_steps != len(schedule):\n"
        "        raise ValueError('resolved total_steps does not match P000')\n"
        "    if any(len(step) != prompts_per_step for step in schedule):\n"
        "        raise ValueError('resolved prompts_per_step does not match P000')\n"
        "    return schedule\n"
    ).encode()
    return CompiledArtifact("curriculum.py", "curriculum_learning", source)


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"Curriculum baseline requires positive {label}")
    return value
