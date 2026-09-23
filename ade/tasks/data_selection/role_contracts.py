"""Long-CoT SFT task-role contracts."""

from __future__ import annotations

import ast

from ade.core.agent import AgentRole
from ade.core.validation import DeliveryViolation, ValidationReport
from ade.tasks.contracts import (
    AnalysisReport,
    ArtifactDelivery,
    PlanSummary,
    PlanningDecision,
    RunSummary,
)
from ade.tasks.role_contract import (
    BoundRoleContract,
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
    repairable_violation,
    string_list,
    summary_violations,
)

_TASK_ID = "data_selection"
_DOMAIN = "long_cot_sft"
_ANALYSIS_SECTIONS = (
    "## Analysis Scope and Evidence",
    "## Direct Inspection",
    "## Data Selection Diagnosis",
    "## Training and Validation Response",
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
            "ade-coordinate-data-selection",
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
            "ade-build-data-selection-artifact",
            "selection.py",
            "data_selection_proposal",
            decode_executable_artifact,
            _validate_artifact,
            (("design.md", "design_report"),),
        ),
        StagedAnalyzerContract(
            _TASK_ID,
            _DOMAIN,
            AgentRole.ANALYZER,
            "ade-analyze-data-selection-trial",
            _validate_analysis,
        ),
        BoundRoleContract(
            _TASK_ID,
            _DOMAIN,
            AgentRole.PLAN_SUMMARIZER,
            "ade-summarize-data-selection-plan",
            "MEMORY.md",
            "plan_memory",
            decode_plan_summary,
            _validate_plan_summary,
        ),
        BoundRoleContract(
            _TASK_ID,
            _DOMAIN,
            AgentRole.RUN_SUMMARIZER,
            "ade-summarize-data-selection-run",
            "MEMORY.md",
            "run_memory",
            decode_run_summary,
            _validate_run_summary,
        ),
    )


def _validate_planning(output: PlanningDecision) -> ValidationReport:
    violations = planning_base_violations(output)
    if set(output.design) != {"comparisons"}:
        violations.append(
            repairable_violation(
                "invalid_data_selection_design",
                "Data Selection design requires exactly comparisons",
                "decision.json#/design",
            )
        )
    return ValidationReport(tuple(violations))


def _validate_artifact(output: ArtifactDelivery) -> ValidationReport:
    violations: list[DeliveryViolation] = experiment_design_violations(output)
    try:
        source = output.content.decode("utf-8")
        tree = ast.parse(source, filename="selection.py")
    except (UnicodeDecodeError, SyntaxError) as error:
        violations.append(
            repairable_violation(
                "invalid_selection_python",
                str(error),
                "selection.py",
            )
        )
        return ValidationReport(tuple(violations))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "select_trajectories"
    ]
    if (
        len(functions) != 1
        or [argument.arg for argument in functions[0].args.args]
        != ["candidate_inventory", "select_size", "judge"]
    ):
        violations.append(
            repairable_violation(
                "invalid_selection_entrypoint",
                "selection.py requires async select_trajectories(candidate_inventory, select_size, judge)",
                "selection.py",
            )
        )
    return ValidationReport(tuple(violations))


def _validate_analysis(output: AnalysisReport) -> ValidationReport:
    return ValidationReport(tuple(analysis_report_violations(output, _ANALYSIS_SECTIONS)))


def _validate_plan_summary(output: PlanSummary) -> ValidationReport:
    return ValidationReport(tuple(summary_violations(output)))


def _validate_run_summary(output: RunSummary) -> ValidationReport:
    return ValidationReport(tuple(summary_violations(output)))
    experiment_design_violations,
