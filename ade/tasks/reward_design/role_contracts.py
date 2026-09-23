"""GRPO RFT task-role contracts."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
import math

from ade.core.agent import AgentRole
from ade.core.validation import DeliveryViolation, ValidationReport
from ade.tasks.contracts import (
    AnalysisReport,
    ArtifactDelivery,
    PlanSummary,
    PlanningDecision,
    RunSummary,
    SupportingArtifact,
)
from ade.tasks.role_contract import (
    BoundRoleContract,
    RoleOutputError,
    StagedAnalyzerContract,
    analysis_report_violations,
    decode_analysis_report,
    decode_executable_artifact,
    decode_plan_summary,
    decode_planning_decision,
    decode_run_summary,
    experiment_design_violations,
    non_empty_text,
    planning_base_violations,
    summary_violations,
)
from ade.tasks.safety_policy import python_safety_violations

_TASK_ID = "reward_design"
_DOMAIN = "grpo_rft"
_ANALYSIS_SECTIONS = (
    "## Analysis Scope and Evidence",
    "## Direct Inspection",
    "## Reward and Rollout Diagnosis",
    "## Training Dynamics and Validation Response",
    "## Findings",
    "## Contradictions and Uncertainty",
    "## Recommendations",
)


def role_contracts() -> tuple[BoundRoleContract, ...]:
    return (
        BoundRoleContract(
            _TASK_ID,
            _DOMAIN,
            AgentRole.COORDINATOR,
            "ade-coordinate-reward-design",
            "decision.json",
            "planning_decision",
            decode_planning_decision,
            _validate_planning,
            (("plan.md", "planning_decision_report"),),
        ),
        BoundRoleContract(
            _TASK_ID,
            _DOMAIN,
            AgentRole.ARTIFACT_BUILDER,
            "ade-build-reward-design-artifact",
            "reward.py",
            "reward_design_proposal",
            _decode_reward_artifact,
            _validate_artifact,
            (("design.md", "design_report"),),
        ),
        StagedAnalyzerContract(
            _TASK_ID,
            _DOMAIN,
            AgentRole.ANALYZER,
            "ade-analyze-reward-design-trial",
            _validate_analysis,
        ),
        BoundRoleContract(
            _TASK_ID,
            _DOMAIN,
            AgentRole.PLAN_SUMMARIZER,
            "ade-summarize-reward-design-plan",
            "MEMORY.md",
            "plan_memory",
            decode_plan_summary,
            _validate_plan_summary,
        ),
        BoundRoleContract(
            _TASK_ID,
            _DOMAIN,
            AgentRole.RUN_SUMMARIZER,
            "ade-summarize-reward-design-run",
            "MEMORY.md",
            "run_memory",
            decode_run_summary,
            _validate_run_summary,
        ),
    )


def _decode_reward_artifact(path, declaration) -> ArtifactDelivery:
    output = decode_executable_artifact(path, declaration)
    try:
        tree = ast.parse(output.content.decode("utf-8"), filename="reward.py")
    except SyntaxError as error:
        raise RoleOutputError(DeliveryViolation(
            "invalid_python", str(error), "output/reward.py", repairable=True,
        )) from error
    reference = json.loads(
        (path.parents[1] / "input" / "reference" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_ids = list(reference.get("source_artifact_ref_ids", ()))
    has_judge_call = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "llm_judge"
        for node in ast.walk(tree)
    )
    has_group_credit = any(
        isinstance(node, ast.FunctionDef) and node.name == "assign_group_credit"
        for node in tree.body
    )
    group_metadata = (
        {
            "group_credit_entrypoint": "reward.py:assign_group_credit",
            "group_credit_schema_version": "ade.group_credit.v1",
        }
        if has_group_credit
        else {}
    )
    return replace(
        output,
        metadata={
            "entrypoint": "reward.py:compute_score",
            "fallback_entrypoint": "reward.py:compute_fallback_score",
            "return_schema": "ade.reward_result.v2",
            "supported_inputs": [
                "question_prompt",
                "response_content",
                "extracted_answer",
                "outcome_score",
                "response_length_tokens",
                "max_response_length_tokens",
            ],
            "algorithm": "grpo",
            "score_bounds": [0.0, 1.0],
            "process_evidence": {
                "status": "required" if has_judge_call else "not_configured",
                "adapter_id": "math_process_evidence.v1" if has_judge_call else None,
                "schema_version": "ade.process_evidence.v1" if has_judge_call else None,
                "dimensions": [
                    "conclusion_support",
                    "derivation_soundness",
                    "substantive_relevance",
                ] if has_judge_call else [],
            },
            "process_failure_policy": "outcome_only",
            "judge_fallback": {"score_source": "outcome", "typed_row_only": True},
            "primary_reference_artifact_ref_id": reference.get(
                "primary_reference_artifact_ref_id"
            ),
            "exploit_probes": ["harness_independent_admission_probe"],
            "lineage": source_ids,
            "missing_input_policy": "bounded_zero_default",
            "non_finite_policy": "bounded_zero_default",
            **group_metadata,
        },
    )


def _validate_planning(output: PlanningDecision) -> ValidationReport:
    violations = planning_base_violations(output)
    if set(output.design) != {"comparisons"}:
        violations.append(
            _item(
                "invalid_reward_design",
                "Reward design contains only comparator bindings",
                "decision.json#/design",
            )
        )
    return ValidationReport(tuple(violations))


def _validate_artifact(output: ArtifactDelivery) -> ValidationReport:
    violations: list[DeliveryViolation] = experiment_design_violations(output)
    try:
        source = output.content.decode("utf-8")
        tree = ast.parse(source, filename=output.path)
    except (UnicodeDecodeError, SyntaxError) as error:
        return ValidationReport(
            (_item("invalid_reward_python", str(error), "reward.py"),)
        )
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {
            "compute_score",
            "compute_fallback_score",
            "assign_group_credit",
        }
    ]
    by_name = {node.name: node for node in functions}
    if (
        not {"compute_score", "compute_fallback_score"}.issubset(by_name)
        or not isinstance(by_name.get("compute_score"), ast.AsyncFunctionDef)
        or any(
            [arg.arg for arg in by_name[name].args.args]
            != [
                "question_prompt",
                "response_content",
                "extracted_answer",
                "outcome_score",
                "response_length_tokens",
                "max_response_length_tokens",
            ]
            for name in ("compute_score", "compute_fallback_score")
            if name in by_name
        )
    ):
        violations.append(
            _item(
                "invalid_reward_entrypoint",
                "require six-parameter normal and fallback entrypoints",
                "reward.py",
            )
        )
    group_function = by_name.get("assign_group_credit")
    if group_function is not None and (
        not isinstance(group_function, ast.FunctionDef)
        or [argument.arg for argument in group_function.args.args] != ["group_input"]
        or group_function.args.vararg is not None
        or group_function.args.kwarg is not None
        or group_function.args.defaults
        or group_function.args.kw_defaults
        or group_function.decorator_list
    ):
        violations.append(
            _item(
                "invalid_group_credit_entrypoint",
                "assign_group_credit must be synchronous with one group_input argument",
                "reward.py",
            )
        )
    judge_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "llm_judge"
    ]
    if judge_calls and not all(
        any(isinstance(node, ast.Await) and node.value is call for node in ast.walk(tree))
        for call in judge_calls
    ):
        violations.append(
            _item(
                "invalid_process_evaluator",
                "every llm_judge call must be awaited",
                "reward.py",
            )
        )
    if any(call.keywords or len(call.args) != 2 for call in judge_calls):
        violations.append(
            _item(
                "invalid_process_evaluator",
                "llm_judge must use the fixed two-argument capability",
                "reward.py",
            )
        )
    fallback = by_name.get("compute_fallback_score")
    if fallback is not None and any(
        isinstance(node, ast.Name) and node.id == "llm_judge"
        for node in ast.walk(fallback)
    ):
        violations.append(
            _item(
                "invalid_process_fallback",
                "compute_fallback_score cannot depend on llm_judge",
                "reward.py",
            )
        )
    if group_function is not None and any(
        isinstance(node, ast.Name) and node.id == "llm_judge"
        for node in ast.walk(group_function)
    ):
        violations.append(
            _item(
                "invalid_group_credit_entrypoint",
                "assign_group_credit cannot depend on llm_judge",
                "reward.py",
            )
        )
    for kind, message in python_safety_violations(tree):
        code = {
            "import": "forbidden_reward_import",
            "call": "forbidden_reward_call",
        }.get(kind, "unsafe_reward_python")
        violations.append(_item(code, message, "reward.py"))
    metadata = output.metadata
    required = {
        "entrypoint": "reward.py:compute_score",
        "fallback_entrypoint": "reward.py:compute_fallback_score",
        "return_schema": "ade.reward_result.v2",
        "supported_inputs": [
            "question_prompt",
            "response_content",
            "extracted_answer",
            "outcome_score",
            "response_length_tokens",
            "max_response_length_tokens",
        ],
        "algorithm": "grpo",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            violations.append(
                _item(
                    f"invalid_{key}",
                    f"{key} must be {expected}",
                    f"delivery.json#/artifacts/0/{key}",
                )
            )
    if not _finite_bounds(metadata.get("score_bounds")):
        violations.append(
            _item(
                "invalid_score_bounds",
                "score_bounds must be finite and ascending",
                "delivery.json#/artifacts/0/score_bounds",
            )
        )
    if metadata.get("score_bounds") != [0.0, 1.0]:
        violations.append(
            _item(
                "invalid_score_bounds",
                "score_bounds must be [0.0, 1.0]",
                "delivery.json#/artifacts/0/score_bounds",
            )
        )
    expected_process_evidence = {
        "status": "required" if judge_calls else "not_configured",
        "adapter_id": "math_process_evidence.v1" if judge_calls else None,
        "schema_version": "ade.process_evidence.v1" if judge_calls else None,
        "dimensions": [
            "conclusion_support",
            "derivation_soundness",
            "substantive_relevance",
        ] if judge_calls else [],
    }
    if metadata.get("process_evidence") != expected_process_evidence:
        violations.append(
            _item(
                "invalid_process_evidence",
                "process_evidence must match the task-owned fixed adapter",
                "delivery.json#/artifacts/0/process_evidence",
            )
        )
    if metadata.get("process_failure_policy") != "outcome_only":
        violations.append(
            _item(
                "invalid_process_failure_policy",
                "process_failure_policy must use row-local outcome_only fallback",
                "delivery.json#/artifacts/0/process_failure_policy",
            )
        )
    fallback_metadata = metadata.get("judge_fallback")
    if fallback_metadata != {"score_source": "outcome", "typed_row_only": True}:
        violations.append(
            _item(
                "invalid_process_fallback",
                "judge_fallback must be restricted to typed row unavailability",
                "delivery.json#/artifacts/0/judge_fallback",
            )
        )
    reference_id = metadata.get("primary_reference_artifact_ref_id")
    if not non_empty_text(reference_id):
        violations.append(
            _item(
                "missing_reference_artifact",
                "primary_reference_artifact_ref_id is required",
                "delivery.json#/artifacts/0/primary_reference_artifact_ref_id",
            )
        )
    for key in ("exploit_probes", "lineage"):
        values = metadata.get(key)
        if (
            not isinstance(values, list)
            or not values
            or not all(non_empty_text(item) for item in values)
        ):
            violations.append(
                _item(
                    f"invalid_{key}",
                    f"{key} must be a non-empty string list",
                    f"delivery.json#/artifacts/0/{key}",
                )
            )
    for key in ("missing_input_policy", "non_finite_policy"):
        if not non_empty_text(metadata.get(key)):
            violations.append(
                _item(
                    f"missing_{key}",
                    f"{key} is required",
                    f"delivery.json#/artifacts/0/{key}",
                )
            )
    return ValidationReport(tuple(violations))

def _validate_analysis(output: AnalysisReport) -> ValidationReport:
    return ValidationReport(tuple(analysis_report_violations(output, _ANALYSIS_SECTIONS)))


def _validate_plan_summary(output: PlanSummary) -> ValidationReport:
    return ValidationReport(tuple(summary_violations(output)))


def _validate_run_summary(output: RunSummary) -> ValidationReport:
    return ValidationReport(tuple(summary_violations(output)))


def _finite_bounds(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(
            type(item) in {int, float} and math.isfinite(item)
            for item in value
        )
        and value[0] < value[1]
    )


def _item(code: str, message: str, path: str) -> DeliveryViolation:
    return DeliveryViolation(code, message, path, repairable=True)
