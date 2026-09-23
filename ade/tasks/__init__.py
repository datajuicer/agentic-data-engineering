"""Typed Task Plugin contracts and registry."""

from ade.tasks.contracts import (
    AgentContextFile,
    AgentInputRequest,
    AgentInputSpec,
    AgentRoleContract,
    AnalysisFinding,
    AnalysisReport,
    ArtifactDelivery,
    ArtifactSpec,
    EngineCommandRequest,
    PlanSummary,
    PlanningDecision,
    RankingSpec,
    RunSummary,
    TrialResult,
)
from ade.tasks.plugin import TaskPlugin
from ade.tasks.registry import TaskRegistry, default_task_registry

__all__ = [
    "AgentContextFile",
    "AgentInputRequest",
    "AgentInputSpec",
    "AgentRoleContract",
    "AnalysisFinding",
    "AnalysisReport",
    "ArtifactDelivery",
    "ArtifactSpec",
    "EngineCommandRequest",
    "PlanSummary",
    "PlanningDecision",
    "RankingSpec",
    "RunSummary",
    "TaskPlugin",
    "TaskRegistry",
    "TrialResult",
    "default_task_registry",
]
