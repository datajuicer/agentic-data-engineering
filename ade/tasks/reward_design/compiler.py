"""Trusted static compiler for GRPO reward proposals."""

from __future__ import annotations

import asyncio
import ast
import math

from ade.tasks.contracts import ArtifactCompilationRequest, CompiledArtifact
from ade.tasks.reward_design.rewards.contracts import (
    enforce_outcome_component,
    validate_reward_result,
)
from ade.tasks.safety_policy import validate_python_safety
_REWARD_ARGUMENTS = (
    "question_prompt",
    "response_content",
    "extracted_answer",
    "outcome_score",
    "response_length_tokens",
    "max_response_length_tokens",
)


def compile_artifact(request: ArtifactCompilationRequest) -> CompiledArtifact:
    delivery = request.delivery
    if delivery.kind != "reward_design_proposal":
        raise ValueError("Reward Design compiler requires reward_design_proposal")
    try:
        source = delivery.content.decode("utf-8")
        tree = ast.parse(source, filename=delivery.path)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError("reward proposal is not valid Python") from error
    _validate_restricted_python(tree)
    rft = request.task_config.get("rft")
    if not isinstance(rft, dict):
        raise ValueError("Reward Design task config requires rft")
    bounds = _bounds(rft.get("reward_range"), "task.rft")
    if _bounds(delivery.metadata.get("score_bounds"), "proposal metadata") != bounds:
        raise ValueError("proposal score bounds do not match fixed task config")
    judge_enabled = _judge_enrichment_enabled(request.task_config)
    group_credit_enabled = _group_credit_enabled(request.task_config)
    _validate_judge_usage(
        tree,
        enabled=judge_enabled,
        group_credit_enabled=group_credit_enabled,
    )
    if judge_enabled:
        _probe_process_reward_for_zero_outcome(
            tree,
            group_credit_enabled=group_credit_enabled,
            artifact_acquires_judge=not group_credit_enabled,
        )
    if group_credit_enabled:
        _admit_group_credit(
            tree,
            observe_process_bank=judge_enabled,
        )

    compiled = source.encode()
    if not compiled.endswith(b"\n"):
        compiled += b"\n"
    return CompiledArtifact("reward.py", "reward_design", compiled)


def _probe_process_reward_for_zero_outcome(
    tree: ast.Module,
    *,
    group_credit_enabled: bool,
    artifact_acquires_judge: bool,
) -> None:
    namespace: dict[str, object] = {}
    exec(compile(tree, "reward.py", "exec"), namespace, namespace)
    compute_score = namespace.get("compute_score")
    if not callable(compute_score):
        raise ValueError("reward proposal compute_score is unavailable")
    judge_calls = 0

    async def llm_judge(
        question_prompt: str,
        response_content: str,
    ) -> dict[str, object]:
        nonlocal judge_calls
        judge_calls += 1
        return {
            "status": "available",
            "adapter_id": "math_process_evidence.v1",
            "schema_version": "ade.process_evidence.v1",
            "dimensions": {
                "derivation_soundness": 1.0,
                "conclusion_support": 1.0,
                "substantive_relevance": 1.0,
            },
        }

    if artifact_acquires_judge:
        namespace["llm_judge"] = llm_judge
    result = asyncio.run(
        compute_score(
            "ADE process-reward admission question",
            "ADE process-reward admission response",
            "",
            0.0,
            1,
            1,
        )
    )
    expected_judge_calls = 1 if artifact_acquires_judge else 0
    if judge_calls != expected_judge_calls:
        if artifact_acquires_judge:
            raise ValueError(
                "judge enrichment requires exactly one llm_judge call on valid reward execution"
            )
        raise ValueError(
            "reward artifact Judge acquisition does not match the resolved Group Credit mode"
        )
    normalized = enforce_outcome_component(
        validate_reward_result(
            result,
            fallback=False,
            group_credit_enabled=group_credit_enabled,
        ),
        0.0,
    )
    if not math.isfinite(normalized["score"]):
        raise ValueError("reward proposal score must remain finite after Judge evidence")


def _judge_enrichment_enabled(task_config: dict[str, object]) -> bool:
    value = task_config.get("judge_enrichment")
    if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
        raise ValueError("task.judge_enrichment.enabled is required")
    return bool(value["enabled"])


def _group_credit_enabled(task_config: dict[str, object]) -> bool:
    rft = task_config.get("rft")
    group_credit = rft.get("group_credit") if isinstance(rft, dict) else None
    return isinstance(group_credit, dict) and group_credit.get("enabled") is True


def _admit_group_credit(
    tree: ast.Module, *, observe_process_bank: bool
) -> None:
    namespace: dict[str, object] = {}
    exec(compile(tree, "reward.py", "exec"), namespace, namespace)
    function = namespace.get("assign_group_credit")
    from ade.tasks.reward_design.group_credit import admit_group_credit_function

    admit_group_credit_function(
        function,
        observe_process_bank=observe_process_bank,
    )


def _validate_judge_usage(
    tree: ast.Module,
    *,
    enabled: bool,
    group_credit_enabled: bool,
) -> None:
    declarations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "llm_judge"
    ]
    references = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "llm_judge"
    ]
    if group_credit_enabled and (declarations or references):
        raise ValueError(
            "Group Credit enabled requires Engine-owned Judge acquisition; "
            "reward.py cannot declare or reference llm_judge"
        )
    compute = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "compute_score"
    )
    calls = [
        node for node in ast.walk(compute)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "llm_judge"
    ]
    artifact_acquires_judge = enabled and not group_credit_enabled
    if not artifact_acquires_judge and calls:
        raise ValueError("judge_enrichment is disabled but reward.py calls llm_judge")
    if artifact_acquires_judge and not calls:
        raise ValueError("judge_enrichment requires reward.py to call llm_judge")
    if artifact_acquires_judge and len(calls) != 1:
        raise ValueError("judge_enrichment requires exactly one llm_judge call site")
    for call in calls:
        if call.keywords or len(call.args) != 2 or not any(
            isinstance(node, ast.Await) and node.value is call
            for node in ast.walk(compute)
        ):
            raise ValueError("reward.py Judge calls must await the injected two-argument capability")


def _validate_restricted_python(tree: ast.Module) -> None:
    if any(not _safe_module_statement(node) for node in tree.body):
        raise ValueError("reward proposal contains module-level executable code")
    compute_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "compute_score"
    ]
    fallback_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "compute_fallback_score"
    ]
    if (
        len(compute_functions) != 1
        or not isinstance(compute_functions[0], ast.AsyncFunctionDef)
        or not _valid_entrypoint(compute_functions[0])
        or len(fallback_functions) != 1
        or not _valid_entrypoint(fallback_functions[0])
    ):
        raise ValueError(
            "reward proposal requires compute_score and compute_fallback_score entrypoints"
        )
    validate_python_safety(tree)
    judge_calls = [
        node
        for node in ast.walk(compute_functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "llm_judge"
    ]
    for judge_call in judge_calls:
        if (
            judge_call.keywords
            or len(judge_call.args) != 2
            or not any(
                isinstance(node, ast.Await) and node.value is judge_call
                for node in ast.walk(compute_functions[0])
            )
        ):
            raise ValueError("reward proposal Judge calls must await the injected two-argument capability")
    if any(
        isinstance(node, ast.Name) and node.id == "llm_judge"
        for node in ast.walk(fallback_functions[0])
    ):
        raise ValueError("reward fallback cannot depend on llm_judge")


def _safe_module_statement(node: ast.stmt) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef)):
        return True
    if isinstance(node, ast.Expr):
        return isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    if isinstance(node, ast.Assign):
        return all(isinstance(target, ast.Name) for target in node.targets) and _literal(
            node.value
        )
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.value is not None and _literal(
            node.value
        )
    return False


def _literal(node: ast.expr) -> bool:
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError):
        return False
    return True


def _valid_entrypoint(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return (
        tuple(argument.arg for argument in function.args.args) == _REWARD_ARGUMENTS
        and not function.decorator_list
        and function.args.vararg is None
        and function.args.kwarg is None
        and not function.args.defaults
        and not function.args.kw_defaults
    )


def _bounds(value: object, label: str) -> tuple[float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) not in {int, float} or not math.isfinite(item) for item in value)
        or value[0] >= value[1]
    ):
        raise ValueError(f"{label} requires finite ascending score_bounds")
    return float(value[0]), float(value[1])
