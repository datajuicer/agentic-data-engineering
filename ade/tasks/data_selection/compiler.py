"""Trusted static compiler for Data Selection scripts."""

from __future__ import annotations

import asyncio
import ast
import json

from ade.tasks.data_selection.selection_contract import (
    validate_selection_source,
)
from ade.tasks.data_selection.fixed_pool import build_fixed_pool_from_task
from ade.tasks.data_selection.selection_runtime import _with_training_examples
from ade.tasks.contracts import ArtifactCompilationRequest, CompiledArtifact
from ade.rubric_jobs.process import parse_process_rubric


def compile_artifact(request: ArtifactCompilationRequest) -> CompiledArtifact:
    delivery = request.delivery
    if delivery.kind != "data_selection_proposal":
        raise ValueError("Data Selection compiler requires data_selection_proposal")
    try:
        source = delivery.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("selection proposal is not valid Python") from error
    validate_selection_source(source)
    inventory, _training, _stats = build_fixed_pool_from_task(request.task_config)
    compiled = source.encode()
    if not compiled.endswith(b"\n"):
        compiled += b"\n"
    enrichment = request.task_config.get("judge_enrichment")
    enabled = enrichment.get("enabled") if isinstance(enrichment, dict) else None
    if type(enabled) is not bool:
        raise ValueError("task.judge_enrichment.enabled must be explicit boolean")
    _probe_selection(
        source,
        inventory,
        _training,
        expected_size=request.task_config.get("select_size"),
        judge_enabled=enabled,
    )
    return CompiledArtifact("selection.py", "data_selection", compiled)


def _probe_selection(
    source: str,
    inventory: bytes,
    training: bytes,
    *,
    expected_size: object,
    judge_enabled: bool,
) -> None:
    rows = [
        json.loads(line)
        for line in inventory.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("candidate_inventory must contain JSON objects")
    training_rows = [
        json.loads(line)
        for line in training.decode("utf-8").splitlines()
        if line.strip()
    ]
    rows = _with_training_examples(rows, training_rows)
    namespace: dict[str, object] = {}
    tree = validate_selection_source(source)
    judge_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "judge"
    ]
    if judge_enabled and not judge_calls:
        raise ValueError("judge_enrichment requires selection.py to call judge")
    if not judge_enabled and judge_calls:
        raise ValueError("judge_enrichment is disabled but selection.py calls judge")
    for call in judge_calls:
        if call.keywords or len(call.args) != 1 or not any(
            isinstance(node, ast.Await) and node.value is call
            for node in ast.walk(tree)
        ):
            raise ValueError(
                "selection.py Judge calls must await the injected batch capability with one request-list argument"
            )
    namespace["json"] = json
    exec(compile(tree, "selection.py", "exec"), namespace, namespace)
    entrypoint = namespace.get("select_trajectories")
    if not callable(entrypoint):
        raise ValueError("selection.py entrypoint is unavailable")
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError("task.select_size must be a positive integer")
    async def probe_judge(requests):
        if not isinstance(requests, list):
            raise ValueError("selection.py Judge batch must be a list")
        evidence = []
        for request in requests:
            if not isinstance(request, dict):
                raise ValueError("selection.py Judge requests must be objects")
            rubric = parse_process_rubric(request.get("rubric"), required=True)
            assert rubric is not None
            dimensions = rubric["projection"]["dimensions"]
            scores = {
                dimension["id"]: dimension["score_levels"][0]["value"]
                for dimension in dimensions
            }
            projected = sum(
                dimension["weight"] * scores[dimension["id"]]
                for dimension in dimensions
            )
            evidence.append({
                "status": "completed",
                "scores_by_dimension": scores,
                "projected_score": projected,
                "fallback": False,
            })
        return evidence
    try:
        selected = asyncio.run(
            entrypoint(json.loads(json.dumps(rows)), expected_size, probe_judge)
        )
    except Exception as error:
        raise ValueError(
            f"selection.py probe failed: {type(error).__name__}: {error}"
        ) from error
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(item, str) or not item for item in selected)
        or len(selected) != len(set(selected))
    ):
        raise ValueError("selection.py must return unique non-empty trajectory IDs")
    allowed = {
        item
        for row in rows
        for item in row.get("trajectory_ids", ())
        if isinstance(item, str) and item
    }
    if not set(selected).issubset(allowed):
        raise ValueError("selection.py returns unauthorized trajectory IDs")
    selected_set = set(selected)
    for row in rows:
        group = row.get("trajectory_ids")
        if not isinstance(group, list):
            continue
        overlap = selected_set.intersection(group)
        if overlap and overlap != set(group):
            raise ValueError("selection.py must select complete trajectory groups")
    if len(selected) != expected_size:
        raise ValueError(
            f"selection.py must return exactly task.select_size={expected_size} IDs"
        )
