"""Trial state."""

from dataclasses import dataclass
from enum import StrEnum


class TrialPhase(StrEnum):
    CREATED = "created"
    BUILDING_ARTIFACT = "building_artifact"
    ARTIFACT_READY = "artifact_ready"
    ENGINE_RUNNING = "engine_running"
    EVIDENCE_READY = "evidence_ready"
    ANALYSIS_DESIGNING = "analysis_designing"
    REVIEW_RUNNING = "review_running"
    REVIEW_READY = "review_ready"
    ANALYZING = "analyzing"
    ANALYSIS_READY = "analysis_ready"
    PLAN_SUMMARIZING = "plan_summarizing"
    PLAN_SUMMARY_READY = "plan_summary_ready"
    RUN_SUMMARIZING = "run_summarizing"
    ARCHIVED = "archived"


class TrialKind(StrEnum):
    SEARCH = "search"
    BOOTSTRAP_BASELINE = "bootstrap_baseline"


class TrialOutcome(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrialArchiveStatus(StrEnum):
    PENDING = "pending"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class TrialState:
    trial_id: str
    coordinator_id: str
    plan_id: str
    phase: TrialPhase = TrialPhase.CREATED
    kind: TrialKind = TrialKind.SEARCH
    command_id: str | None = None
    logical_command_id: str | None = None
    engine_attempt_id: str | None = None
    engine_attempt_index: int = 0
    engine_retry_pending: bool = False
    engine_attempt_failure_ref_ids: tuple[str, ...] = ()
    artifact_ref_id: str | None = None
    artifact_supporting_ref_ids: tuple[str, ...] = ()
    result_refs: tuple[str, ...] = ()
    package_ref_id: str | None = None
    authorized_unit_ids: tuple[str, ...] = ()
    analysis_status: str | None = None
    analysis_design_ref_id: str | None = None
    analysis_review_command_id: str | None = None
    analysis_review_logical_command_id: str | None = None
    analysis_review_attempt_id: str | None = None
    analysis_review_attempt_index: int = 0
    analysis_review_retry_pending: bool = False
    analysis_review_attempt_failure_ref_ids: tuple[str, ...] = ()
    analysis_review_packet_ref_id: str | None = None
    analysis_ref_id: str | None = None
    analysis_evidence_ref_id: str | None = None
    analysis_review_coverage_ref_id: str | None = None
    analysis_failure_ref_id: str | None = None
    objective_comparison_ref_id: str | None = None
    outcome: TrialOutcome = TrialOutcome.PENDING
    archive_status: TrialArchiveStatus = TrialArchiveStatus.PENDING
    failure_kind: str | None = None
    offline_score: float | None = None
    offline_secondary_score: float | None = None
    source_artifact_ref_ids: tuple[str, ...] = ()
    analysis_packet_ref_id: str | None = None
    plan_snapshot_ref_id: str | None = None
    run_snapshot_ref_id: str | None = None
    trial_record_ref: str | None = None
    plan_memory_basis: str | None = None
    plan_memory_result: str | None = None
    run_memory_basis: str | None = None
    run_memory_result: str | None = None

    def __post_init__(self) -> None:
        if not self.trial_id or not self.coordinator_id or not self.plan_id:
            raise ValueError("Trial identity is required")
        if self.engine_attempt_index < 0:
            raise ValueError("Engine attempt index cannot be negative")
        if self.engine_retry_pending and self.command_id is not None:
            raise ValueError("retry-pending Engine command cannot remain active")
        if self.analysis_review_attempt_index < 0:
            raise ValueError("Review attempt index cannot be negative")
        if (
            self.analysis_review_retry_pending
            and self.analysis_review_command_id is not None
        ):
            raise ValueError("retry-pending Review command cannot remain active")
        if self.archive_status is TrialArchiveStatus.ARCHIVED:
            if self.phase is not TrialPhase.ARCHIVED:
                raise ValueError("archived Trial must use archived lifecycle status")
            if self.outcome is TrialOutcome.PENDING:
                raise ValueError("archived Trial requires a final outcome")
            if self.plan_snapshot_ref_id is None or self.run_snapshot_ref_id is None:
                raise ValueError("archived Trial requires Plan and Run snapshots")
        if self.outcome is TrialOutcome.FAILED and not self.failure_kind:
            raise ValueError("failed Trial requires failure_kind")
