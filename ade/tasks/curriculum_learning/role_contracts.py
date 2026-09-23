"""Agent role contracts for the Curriculum Learning vertical task."""

from __future__ import annotations

from ade.core.agent import AgentRole
from ade.core.validation import DeliveryViolation, ValidationReport
from ade.tasks.contracts import (
    AnalysisReport,
    ArtifactDelivery,
    PlanSummary,
    PlanningDecision,
    RunSummary,
)
from ade.tasks.curriculum_learning.schedule_contract import validate_curriculum_source
from ade.tasks.role_contract import (
    BoundRoleContract,
    StagedAnalyzerContract,
    analysis_report_violations,
    decode_executable_artifact,
    decode_plan_summary,
    decode_planning_decision,
    decode_run_summary,
    experiment_design_violations,
    planning_base_violations,
    repairable_violation,
    summary_violations,
)


_TASK_ID = "curriculum_learning"
_DOMAIN = "grpo_rft"
_ANALYSIS_SECTIONS = (
    "## Analysis Scope and Evidence",
    "## Direct Inspection",
    "## Curriculum Schedule Diagnosis",
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
            "ade-coordinate-curriculum-learning",
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
            "ade-build-curriculum-learning-artifact",
            "curriculum.py",
            "curriculum_learning_proposal",
            decode_executable_artifact,
            _validate_artifact,
            (("design.md", "design_report"),),
        ),
        StagedAnalyzerContract(
            _TASK_ID,
            _DOMAIN,
            AgentRole.ANALYZER,
            "ade-analyze-curriculum-learning-trial",
            _validate_analysis,
        ),
        BoundRoleContract(
            _TASK_ID,
            _DOMAIN,
            AgentRole.PLAN_SUMMARIZER,
            "ade-summarize-curriculum-learning-plan",
            "MEMORY.md",
            "plan_memory",
            decode_plan_summary,
            _validate_plan_summary,
        ),
        BoundRoleContract(
            _TASK_ID,
            _DOMAIN,
            AgentRole.RUN_SUMMARIZER,
            "ade-summarize-curriculum-learning-run",
            "MEMORY.md",
            "run_memory",
            decode_run_summary,
            _validate_run_summary,
        ),
    )


def _validate_planning(output: PlanningDecision) -> ValidationReport:
    violations = planning_base_violations(output)
    judge_use = output.design.get("judge_enrichment")
    if judge_use is not None and type(judge_use) is not bool:
        violations.append(
            repairable_violation(
                "invalid_curriculum_judge_plan",
                "design.judge_enrichment must declare a boolean use decision",
                "decision.json#/design/judge_enrichment",
            )
        )
    return ValidationReport(tuple(violations))


def _validate_artifact(output: ArtifactDelivery) -> ValidationReport:
    violations: list[DeliveryViolation] = experiment_design_violations(output)
    try:
        validate_curriculum_source(output.content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        violations.append(
            repairable_violation(
                "invalid_curriculum_python",
                str(error),
                "curriculum.py",
            )
        )
    return ValidationReport(tuple(violations))


def _validate_analysis(output: AnalysisReport) -> ValidationReport:
    return ValidationReport(tuple(analysis_report_violations(output, _ANALYSIS_SECTIONS)))


def _validate_plan_summary(output: PlanSummary) -> ValidationReport:
    return ValidationReport(tuple(summary_violations(output)))


def _validate_run_summary(output: RunSummary) -> ValidationReport:
    return ValidationReport(tuple(summary_violations(output)))
