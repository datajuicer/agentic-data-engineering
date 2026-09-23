"""Accepted executor results consumed by the Reducer."""

from dataclasses import dataclass
from enum import StrEnum

from ade.core.agent import AgentSession
from ade.core.artifacts import ArtifactRef
from ade.core.bootstrap import BaseEvaluationRecord, BootstrapStatus
from ade.core.failures import FailureState
from ade.core.operator import OperatorEvaluationRecord, OperatorEvaluationStatus
from ade.core.plan import PlanState
from ade.core.snapshot import SnapshotRef
from ade.core.run import ActiveAgentCallRef, EngineCommandRef, ReviewCommandRef


class OutcomeKind(StrEnum):
    AGENT_DELIVERED = "agent_delivered"
    ENGINE_COMPLETED = "engine_completed"
    REVIEW_COMPLETED = "review_completed"
    ACTION_FAILED = "action_failed"
    RUN_COMPLETED = "run_completed"


@dataclass(frozen=True)
class OperatorEvaluationScheduledOutcome:
    run_id: str
    basis_revision: int
    record: OperatorEvaluationRecord


@dataclass(frozen=True)
class OperatorEvaluationTerminalOutcome:
    run_id: str
    basis_revision: int
    target_id: str
    command_id: str
    logical_command_id: str
    attempt_id: str
    attempt_index: int
    receipt_id: str
    status: OperatorEvaluationStatus
    result_ref: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class BootstrapBaseScheduledOutcome:
    run_id: str
    basis_revision: int
    commands: tuple[EngineCommandRef, ...]


@dataclass(frozen=True)
class BootstrapBaseAttemptSubmittedOutcome:
    run_id: str
    basis_revision: int
    previous_command_id: str
    command: EngineCommandRef


@dataclass(frozen=True)
class BootstrapBaseAttemptRetryPendingOutcome:
    run_id: str
    basis_revision: int
    command_id: str
    logical_command_id: str
    attempt_id: str
    attempt_index: int
    receipt_id: str
    failure_kind: str
    message: str
    observed_at: float = 0.0


@dataclass(frozen=True)
class BootstrapBaseTerminalOutcome:
    run_id: str
    basis_revision: int
    command_status: dict[str, str]
    record: BaseEvaluationRecord
    evidence_ref: ArtifactRef | None = None


@dataclass(frozen=True)
class OperatorEvaluationAttemptSubmittedOutcome:
    run_id: str
    basis_revision: int
    target_id: str
    previous_attempt_index: int
    command: EngineCommandRef


@dataclass(frozen=True)
class OperatorEvaluationAttemptRetryPendingOutcome:
    run_id: str
    basis_revision: int
    target_id: str
    command_id: str
    logical_command_id: str
    attempt_id: str
    attempt_index: int
    receipt_id: str
    failure_kind: str
    message: str
    observed_at: float = 0.0


@dataclass(frozen=True)
class BootstrapP000RegisteredOutcome:
    run_id: str
    basis_revision: int
    coordinator_id: str
    plan_id: str
    trial_id: str
    artifact_ref: ArtifactRef
    artifact_path: str


@dataclass(frozen=True)
class BootstrapStageAdvancedOutcome:
    run_id: str
    basis_revision: int
    from_status: BootstrapStatus
    to_status: BootstrapStatus


@dataclass(frozen=True)
class BootstrapCompletedOutcome:
    run_id: str
    basis_revision: int
    coordinator_id: str
    plan_id: str
    trial_id: str


@dataclass(frozen=True)
class BootstrapFailedOutcome:
    run_id: str
    basis_revision: int
    reason: str
    command_status: dict[str, str]


@dataclass(frozen=True)
class RunResumedOutcome:
    run_id: str
    basis_revision: int
    automatic_recovery_max_attempts: int | None = None


@dataclass(frozen=True)
class LocalJudgeReplacedOutcome:
    run_id: str
    basis_revision: int
    launch_id: str
    state_path: str
    host: str
    gateway_url: str
    node_id: str
    service_state: str = "ready"


@dataclass(frozen=True)
class RunPauseRequestedOutcome:
    run_id: str
    basis_revision: int
    reason: str


@dataclass(frozen=True)
class RunCancelledOutcome:
    run_id: str
    basis_revision: int
    reason: str


@dataclass(frozen=True)
class CoordinatorFinishRequestedOutcome:
    run_id: str
    coordinator_id: str
    basis_revision: int
    requested_plan_limit: int | None
    reason: str


@dataclass(frozen=True)
class CoordinatorCancelRequestedOutcome:
    run_id: str
    coordinator_id: str
    basis_revision: int
    reason: str


@dataclass(frozen=True)
class CoordinatorFinishedEarlyOutcome:
    run_id: str
    coordinator_id: str
    basis_revision: int


@dataclass(frozen=True)
class CoordinatorCancelledOutcome:
    run_id: str
    coordinator_id: str
    basis_revision: int


@dataclass(frozen=True)
class CoordinatorPlanningCancelledOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    reservation_id: str
    basis_revision: int
    cancellation_ref: ArtifactRef


@dataclass(frozen=True)
class CoordinatorTrialCancelledOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    cancellation_ref: ArtifactRef
    trial_snapshot_ref: SnapshotRef
    plan_snapshot_ref: SnapshotRef
    run_snapshot_ref: SnapshotRef


@dataclass(frozen=True)
class CoordinatorPlanCancelledOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    basis_revision: int
    cancellation_ref: ArtifactRef


@dataclass(frozen=True)
class RunFinishRequestedOutcome:
    run_id: str
    basis_revision: int
    requested_plan_limit: int
    reason: str


@dataclass(frozen=True)
class RunFinishedEarlyOutcome:
    run_id: str
    basis_revision: int


@dataclass(frozen=True)
class Outcome:
    action_id: str
    run_id: str
    kind: OutcomeKind
    subject_id: str
    basis_revision: int
    accepted_refs: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True)
class PlanningDecisionOutcome:
    action_id: str
    run_id: str
    subject_id: str
    basis_revision: int
    call_id: str
    plan: PlanState
    decision_ref: ArtifactRef
    supporting_refs: tuple[ArtifactRef, ...] = ()
    snapshot_ref: SnapshotRef | None = None
    agent_session: AgentSession | None = None


@dataclass(frozen=True)
class PlanningSlotReservedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    reservation_id: str
    sequence: int
    basis_revision: int


@dataclass(frozen=True)
class PlanCatalogUpdatedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    basis_revision: int


@dataclass(frozen=True)
class TrialProposedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int


@dataclass(frozen=True)
class AgentCallSubmittedOutcome:
    run_id: str
    coordinator_id: str | None
    plan_id: str | None
    trial_id: str
    basis_revision: int
    role: str
    call_ref: ActiveAgentCallRef


@dataclass(frozen=True)
class CoordinatorCallSubmittedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    basis_revision: int
    call_ref: ActiveAgentCallRef


@dataclass(frozen=True)
class AgentRetrySubmittedOutcome:
    run_id: str
    subject_id: str
    basis_revision: int
    call_ref: ActiveAgentCallRef
    previous_attempt_id: str
    retry_reason: str


@dataclass(frozen=True)
class AgentRetryHeldOutcome:
    run_id: str
    subject_id: str
    basis_revision: int
    call_id: str
    previous_attempt_id: str
    retry_reason: str


@dataclass(frozen=True)
class BuilderProposalCompletedOutcome:
    run_id: str
    basis_revision: int
    call_id: str
    attempt_id: str
    delivery_ref: str


@dataclass(frozen=True)
class BuilderReflectionCompletedOutcome:
    run_id: str
    basis_revision: int
    call_id: str
    attempt_id: str
    reflection_index: int
    delivery_ref: str


@dataclass(frozen=True)
class BuilderRealizationStartedOutcome:
    run_id: str
    basis_revision: int
    call_id: str
    reflection_index: int
    delivery_ref: str


@dataclass(frozen=True)
class BuilderRealizationCompletedOutcome:
    run_id: str
    basis_revision: int
    call_id: str
    reflection_index: int
    delivery_ref: str
    realization_ref: ArtifactRef


@dataclass(frozen=True)
class BuilderReflectionSubmittedOutcome:
    run_id: str
    basis_revision: int
    call_ref: ActiveAgentCallRef
    previous_delivery_ref: str
    previous_realization_ref: ArtifactRef


@dataclass(frozen=True)
class BuilderRealizationFinalizedOutcome:
    run_id: str
    basis_revision: int
    call_id: str
    reflection_index: int
    delivery_ref: str
    realization_ref: ArtifactRef


@dataclass(frozen=True)
class CoordinatorFailedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    basis_revision: int
    call_id: str
    failure_ref: ArtifactRef
    agent_session: AgentSession | None = None


@dataclass(frozen=True)
class BaselineTrialRegisteredOutcome:
    run_id: str
    coordinator_id: str
    trial_id: str
    plan_id: str
    basis_revision: int
    artifact_ref: ArtifactRef


@dataclass(frozen=True)
class ArtifactAcceptedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    call_id: str
    artifact_ref: ArtifactRef
    supporting_refs: tuple[ArtifactRef, ...] = ()
    agent_session: AgentSession | None = None


@dataclass(frozen=True)
class BuilderFailedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    call_id: str
    failure_ref: ArtifactRef
    snapshot_ref: SnapshotRef
    objective_comparison_ref: ArtifactRef
    agent_session: AgentSession | None = None


@dataclass(frozen=True)
class EngineQueuedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    command_id: str
    command_kind: str
    submitted_at: float
    liveness_deadline: float
    logical_command_id: str = ""
    attempt_id: str = "attempt-001"
    attempt_index: int = 1


@dataclass(frozen=True)
class EngineAttemptRetryPendingOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    command_id: str
    logical_command_id: str
    attempt_id: str
    attempt_index: int
    receipt_id: str
    failure_ref: ArtifactRef
    failure_kind: str
    message: str
    observed_at: float = 0.0


@dataclass(frozen=True)
class EngineCompletedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    command_id: str
    receipt_id: str
    result_refs: tuple[str, ...]
    package_ref: ArtifactRef
    authorized_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class EngineFailedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    command_id: str
    receipt_id: str
    failure_ref: ArtifactRef
    snapshot_ref: SnapshotRef | None
    failure_kind: str
    objective_comparison_ref: ArtifactRef | None = None
    result_refs: tuple[str, ...] = ()
    package_ref: ArtifactRef | None = None
    authorized_unit_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisAcceptedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    call_id: str
    analysis_ref: ArtifactRef
    findings_ref: ArtifactRef
    evidence_ref: ArtifactRef
    review_coverage_ref: ArtifactRef
    snapshot_ref: SnapshotRef
    objective_comparison_ref: ArtifactRef
    offline_score: float | None
    offline_secondary_score: float | None = None
    agent_session: AgentSession | None = None


@dataclass(frozen=True)
class AnalysisReviewSubmittedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    call_id: str
    design_ref: ArtifactRef
    command_ref: ReviewCommandRef
    agent_session: AgentSession | None = None


@dataclass(frozen=True)
class AnalysisReviewCompletedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    command_id: str
    receipt_id: str
    packet_ref: ArtifactRef
    coverage_ref: ArtifactRef


@dataclass(frozen=True)
class AnalysisReviewAttemptRetryPendingOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    command_id: str
    logical_command_id: str
    attempt_id: str
    attempt_index: int
    receipt_id: str
    failure_ref: ArtifactRef
    failure_kind: str
    message: str
    observed_at: float = 0.0


@dataclass(frozen=True)
class AnalysisReviewAttemptSubmittedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    previous_command_id: str
    command_ref: ReviewCommandRef


@dataclass(frozen=True)
class AnalysisFailedOutcome:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    call_id: str
    failure_ref: ArtifactRef
    snapshot_ref: SnapshotRef
    objective_comparison_ref: ArtifactRef
    offline_score: float | None
    offline_secondary_score: float | None = None
    agent_session: AgentSession | None = None


@dataclass(frozen=True)
class SummaryAcceptedOutcome:
    run_id: str
    coordinator_id: str | None
    subject_id: str
    basis_revision: int
    call_id: str
    summary_kind: str
    summary_ref: ArtifactRef
    evidence_ref: ArtifactRef
    snapshot_ref: SnapshotRef
    trial_id: str
    trial_coordinator_id: str
    trial_plan_id: str
    agent_session: AgentSession | None = None


@dataclass(frozen=True)
class RunCompletedOutcome:
    run_id: str
    basis_revision: int


@dataclass(frozen=True)
class RunPausedOutcome:
    run_id: str
    basis_revision: int
    reason: str


@dataclass(frozen=True)
class RunSuspendedOutcome:
    run_id: str
    subject_ref: str
    basis_revision: int
    failure: FailureState
    agent_session: AgentSession | None = None
    call_id: str | None = None
