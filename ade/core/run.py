"""Authoritative ADE RunState."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
import hashlib
import re
from ade.core.actions import Action, ActionKind
from ade.core.agent import (
    AgentRole,
    AgentSession,
    AgentSessionStatus,
    agent_owner_subject_ref,
    validate_agent_target,
)
from ade.core.artifacts import ArtifactRef
from ade.core.bootstrap import BootstrapState
from ade.core.catalog import ActivePlanIntent, PlanCatalogState, SourceEligibleResult
from ade.core.coordinator import (
    CoordinatorControlStatus,
    CoordinatorKind,
    CoordinatorState,
)
from ade.core.failures import FailureState
from ade.core.insight import InsightGraph, InsightNode
from ade.core.plan import (
    PlanKind,
    PlanRelation,
    PlanRelationKind,
    PlanState,
    PlanStatus,
)
from ade.core.operator import OperatorEvaluationRecord, OperatorEvaluationStatus
from ade.core.ranking import RankingEntry, RankingState
from ade.core.scope import PlanKey, subject_ref as canonical_subject_ref
from ade.core.scheduling import PlanningSlot
from ade.core.snapshot import SnapshotKind, SnapshotRef
from ade.core.trial import (
    TrialArchiveStatus,
    TrialKind,
    TrialOutcome,
    TrialState,
    TrialPhase,
)


class RunStatus(StrEnum):
    CREATED = "created"
    BOOTSTRAPPING = "bootstrapping"
    RUNNING = "running"
    RECOVERING = "recovering"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchOutcome(StrEnum):
    PENDING = "pending"
    VALID_RESULT = "valid_result"
    NO_VALID_PLAN = "no_valid_plan"
    NO_VALID_TRIAL = "no_valid_trial"


class CompletionKind(StrEnum):
    FULL_BUDGET = "full_budget"
    OPERATOR_EARLY_FINISH = "operator_early_finish"


@dataclass(frozen=True)
class RunFinishRequest:
    requested_plan_limit: int
    reason: str
    requested_revision: int

    def __post_init__(self) -> None:
        if (
            self.requested_plan_limit < 0
            or not self.reason.strip()
            or self.requested_revision < 0
        ):
            raise ValueError("Run finish request is invalid")


class FactClass(StrEnum):
    AUDIT = "audit"
    OPERATIONAL = "operational"
    SCIENTIFIC = "scientific"


class ForkCause(StrEnum):
    INVALID_ACCEPTED_FACT = "invalid_accepted_fact"
    UNFENCEABLE_EXECUTION = "unfenceable_execution"
    CONTINUE_CANCELLED = "continue_cancelled"


class RunSeedBoundaryKind(StrEnum):
    BOOTSTRAP = "bootstrap"
    ACCEPTED_FRONTIER = "accepted_frontier"


@dataclass(frozen=True)
class AutomaticRecoveryPolicy:
    max_attempts: int = 3
    dependency_readiness_timeout_seconds: int = 1800

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("automatic recovery max_attempts must be positive")
        if self.dependency_readiness_timeout_seconds < 1:
            raise ValueError(
                "automatic recovery dependency readiness timeout must be positive"
            )


@dataclass(frozen=True)
class RecoveryState:
    subject_ref: str
    logical_work_ref: str
    failed_attempt_id: str
    failed_attempt_index: int
    failure_kind: str
    attempts_used: int
    max_attempts: int
    next_attempt_index: int
    started_at: float
    readiness_deadline: float
    fence_ref: str
    lease_ref: str

    def __post_init__(self) -> None:
        if any(
            not value
            for value in (
                self.subject_ref,
                self.logical_work_ref,
                self.failed_attempt_id,
                self.failure_kind,
                self.fence_ref,
                self.lease_ref,
            )
        ):
            raise ValueError("automatic recovery identity and fence are required")
        if (
            self.failed_attempt_index < 1
            or self.attempts_used < 1
            or self.max_attempts < self.attempts_used
            or self.next_attempt_index != self.failed_attempt_index + 1
            or self.next_attempt_index > self.max_attempts
            or self.started_at < 0
            or self.readiness_deadline <= self.started_at
        ):
            raise ValueError("automatic recovery counters or deadline are invalid")


@dataclass(frozen=True)
class ForkLineage:
    lineage_root_run_id: str
    generation: int
    source_run_id: str
    source_revision: int
    source_revision_ref: str
    cause: ForkCause
    cause_ref: str
    replay_boundary_revision: int
    replay_boundary_ref: str

    def __post_init__(self) -> None:
        if (
            not self.lineage_root_run_id
            or self.generation < 1
            or not self.source_run_id
            or self.source_revision < 0
            or not self.source_revision_ref
            or not self.cause_ref
            or self.replay_boundary_revision < 0
            or not self.replay_boundary_ref
        ):
            raise ValueError("typed fork lineage is incomplete")
        if self.source_revision_ref != (
            f"{self.source_run_id}@rev-{self.source_revision:06d}"
        ):
            raise ValueError("fork source_revision_ref is not canonical")
        if self.replay_boundary_ref != (
            f"{self.source_run_id}@rev-{self.replay_boundary_revision:06d}"
        ):
            raise ValueError("fork replay_boundary_ref is not canonical")
        if self.replay_boundary_revision > self.source_revision:
            raise ValueError("fork replay boundary cannot follow source evidence")


@dataclass(frozen=True)
class RunSeedProvenance:
    source_run_id: str
    source_revision: int
    source_revision_ref: str
    boundary_kind: RunSeedBoundaryKind
    requested_frontier: dict[str, str] = field(default_factory=dict)
    resolved_frontier: dict[str, str] = field(default_factory=dict)
    discarded_nonterminal_scopes: tuple[str, ...] = ()
    source_deployment_id: str | None = None

    def __post_init__(self) -> None:
        if not self.source_run_id or self.source_revision < 0:
            raise ValueError("Run Seed source identity is incomplete")
        if self.source_revision_ref != (
            f"{self.source_run_id}@rev-{self.source_revision:06d}"
        ):
            raise ValueError("Run Seed source_revision_ref is not canonical")
        if self.boundary_kind is RunSeedBoundaryKind.BOOTSTRAP:
            if self.requested_frontier or self.resolved_frontier:
                raise ValueError("bootstrap Run Seed cannot carry a Plan frontier")
        elif (
            not self.requested_frontier
            or self.requested_frontier != self.resolved_frontier
        ):
            raise ValueError(
                "accepted-frontier Run Seed requires exact requested/resolved equality"
            )
        for coordinator_id, plan_id in self.requested_frontier.items():
            if re.fullmatch(r"c\d{3,}", coordinator_id) is None or re.fullmatch(
                r"p\d{3,}", plan_id
            ) is None:
                raise ValueError("Run Seed frontier identities are not canonical")
        if any(
            not scope.startswith(f"{self.source_run_id}/")
            for scope in self.discarded_nonterminal_scopes
        ):
            raise ValueError("discarded Run Seed scopes must belong to the source Run")


@dataclass(frozen=True)
class TransitionRecord:
    transition_id: str
    kind: str
    fact_class: FactClass
    scope: dict[str, str]
    subject_ref: str
    logical_work_ref: str
    accepted_fact_refs: tuple[str, ...]
    origin_refs: tuple[str, ...]
    from_revision: int | None
    to_revision: int

    def __post_init__(self) -> None:
        if not self.transition_id or not self.kind or not self.subject_ref:
            raise ValueError("transition identity is required")
        run_id = self.scope.get("run_id")
        if not run_id or self.scope != _scope_from_subject_ref(
            run_id, self.subject_ref
        ):
            raise ValueError("transition scope must match SubjectRef")
        if not self.logical_work_ref:
            raise ValueError("transition logical_work_ref is required")
        if self.fact_class is FactClass.SCIENTIFIC:
            if not self.accepted_fact_refs:
                raise ValueError(
                    "scientific transition requires accepted fact refs"
                )
        elif self.accepted_fact_refs:
            raise ValueError(
                "audit/operational transition cannot accept scientific facts"
            )
        if self.to_revision < 0 or (
            self.from_revision is not None
            and self.to_revision != self.from_revision + 1
        ):
            raise ValueError("transition revision boundary is invalid")

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        kind: str,
        subject_ref: str,
        from_revision: int | None,
        to_revision: int,
        fact_class: FactClass | str | None = None,
        logical_work_ref: str | None = None,
        accepted_fact_refs: tuple[str, ...] = (),
        origin_refs: tuple[str, ...] = (),
    ) -> "TransitionRecord":
        scope = _scope_from_subject_ref(run_id, subject_ref)
        resolved_fact_class = FactClass(
            fact_class or fact_class_for_transition(kind)
        )
        logical_ref = logical_work_ref or subject_ref
        identity = (
            f"{run_id}\0{kind}\0{subject_ref}\0{from_revision}\0{to_revision}"
        )
        return cls(
            transition_id=f"transition-{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
            kind=kind,
            fact_class=resolved_fact_class,
            scope=scope,
            subject_ref=subject_ref,
            logical_work_ref=logical_ref,
            accepted_fact_refs=tuple(accepted_fact_refs),
            origin_refs=tuple(origin_refs),
            from_revision=from_revision,
            to_revision=to_revision,
        )

def fact_class_for_transition(kind: str) -> FactClass:
    if kind in {
        "bootstrap_base_completed",
        "bootstrap_p000_registered",
        "plan_accepted",
        "artifact_accepted",
        "builder_failed",
        "engine_completed",
        "engine_failed",
        "analysis_review_completed",
        "analysis_accepted",
        "analysis_defaulted",
        "summary_accepted",
        "operator_evaluation_terminal",
        "baseline_trial_registered",
        "coordinator_failed",
        "run_seeded",
    }:
        return FactClass.SCIENTIFIC
    if kind.endswith("_late") or kind.endswith("_rejected"):
        return FactClass.AUDIT
    return FactClass.OPERATIONAL


def _scope_from_subject_ref(run_id: str, subject_ref: str) -> dict[str, str]:
    parts = subject_ref.split("/")
    if not parts or parts[0] != run_id or len(parts) > 4:
        raise ValueError("transition subject_ref must start with its Run ID")
    labels = ("run_id", "coordinator_id", "plan_id", "trial_id")
    return {label: part for label, part in zip(labels, parts, strict=False)}


@dataclass(frozen=True)
class MemoryState:
    collection_basis: str
    run_head: str
    plan_heads: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.collection_basis or not self.run_head:
            raise ValueError("Memory collection basis and Run head are required")
        match = re.fullmatch(r"(?P<run>[^/]+)/RM\d{3,}", self.run_head)
        if match is None:
            raise ValueError("Run Memory head must be <run>/RMNNN")
        run_id = match.group("run")
        for key, value in self.plan_heads.items():
            if re.fullmatch(
                rf"{re.escape(run_id)}/c\d{{3,}}/p\d{{3,}}",
                key,
            ) is None or value != f"{key}/{value.rsplit('/', 1)[-1]}":
                raise ValueError(
                    "Plan Memory identity must use <run>/cNNN/pNNN/PMNNN"
                )
            if re.fullmatch(r"PM\d{3,}", value.rsplit("/", 1)[-1]) is None:
                raise ValueError(
                    "Plan Memory identity must use <run>/cNNN/pNNN/PMNNN"
                )


@dataclass(frozen=True)
class ActiveAgentCallRef:
    run_id: str
    session_id: str
    call_id: str
    attempt_id: str
    role: str
    subject_id: str
    basis_revision: int
    action_id: str
    target_subject_ref: str
    action_fields: dict[str, object]
    coordinator_id: str | None
    plan_id: str | None
    trial_id: str | None
    retry_index: int
    max_retries: int
    status: str
    submitted_at: float
    last_heartbeat_at: float
    resume_handle: str | None
    lease_generation: int
    memory_view_id: str
    run_memory_basis: str
    plan_memory_basis: str | None = None
    reflection_index: int = 0
    max_reflections: int = 0
    round_status: str = "agent_running"
    current_delivery_ref: str | None = None
    current_realization_ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        if any(
            not value
            for value in (
                self.session_id,
                self.run_id,
                self.call_id,
                self.attempt_id,
                self.role,
                self.subject_id,
                self.action_id,
                self.target_subject_ref,
                self.status,
                self.memory_view_id,
                self.run_memory_basis,
            )
        ):
            raise ValueError("active Agent Call identity and Memory basis are required")
        validate_agent_target(
            run_id=self.run_id,
            role=AgentRole(self.role),
            coordinator_id=self.coordinator_id,
            plan_id=self.plan_id,
            target_subject_ref=self.target_subject_ref,
        )
        if (
            self.basis_revision < 0
            or self.retry_index < 0
            or self.max_retries < 0
            or self.lease_generation < 0
            or self.reflection_index < 0
            or self.max_reflections < 0
        ):
            raise ValueError("active Agent Call counters must be non-negative")
        if self.reflection_index > self.max_reflections:
            raise ValueError("active Agent Call reflection budget is exceeded")
        if self.round_status not in {
            "agent_running",
            "delivery_ready",
            "realization_running",
            "realization_ready",
            "finalized",
        }:
            raise ValueError("active Agent Call round status is invalid")
        if AgentRole(self.role) is not AgentRole.ARTIFACT_BUILDER and (
            self.reflection_index != 0
            or self.max_reflections != 0
            or self.current_delivery_ref is not None
            or self.current_realization_ref is not None
        ):
            raise ValueError("only Artifact Builder Calls may carry reflection state")

    @property
    def owner_subject_ref(self) -> str:
        return agent_owner_subject_ref(
            run_id=self.run_id,
            role=AgentRole(self.role),
            coordinator_id=self.coordinator_id,
            plan_id=self.plan_id,
        )


@dataclass(frozen=True)
class ResolvedTask:
    task_id: str
    plugin_id: str
    config_ref: str | None = None
    config: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioState:
    max_plans: int
    max_trials: int
    min_trials_per_plan: int = 1
    max_trials_per_plan: int | None = None
    no_improvement_patience: int = 2

    def __post_init__(self) -> None:
        if self.max_plans < 1 or self.max_trials < 0:
            raise ValueError("portfolio limits must be positive, except max_trials")
        if self.max_trials == 0:
            object.__setattr__(self, "min_trials_per_plan", 0)
            object.__setattr__(self, "max_trials_per_plan", 0)
            return
        if self.min_trials_per_plan < 1:
            raise ValueError("min_trials_per_plan must be positive")
        if self.max_trials_per_plan is None:
            object.__setattr__(self, "max_trials_per_plan", self.max_trials)
        assert self.max_trials_per_plan is not None
        if self.max_trials_per_plan < self.min_trials_per_plan:
            raise ValueError("max_trials_per_plan cannot be below the minimum")
        if self.max_trials_per_plan > self.max_trials:
            raise ValueError("max_trials_per_plan cannot exceed total Trial budget")
        if self.no_improvement_patience < 1:
            raise ValueError("no_improvement_patience must be positive")


@dataclass(frozen=True)
class EngineCommandRef:
    command_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    kind: str
    logical_command_id: str = ""
    attempt_id: str = "attempt-001"
    attempt_index: int = 1
    status: str = "submitted"
    submitted_at: float | None = None
    claimed_at: float | None = None
    last_heartbeat_at: float | None = None
    liveness_deadline: float | None = None


@dataclass(frozen=True)
class ReviewCommandRef:
    command_id: str
    logical_command_id: str
    attempt_id: str
    attempt_index: int
    coordinator_id: str
    plan_id: str
    trial_id: str
    status: str = "submitted"

    def __post_init__(self) -> None:
        if (
            not self.command_id
            or not self.logical_command_id
            or not self.attempt_id
            or self.attempt_index < 1
        ):
            raise ValueError("active Review Attempt identity is required")


@dataclass(frozen=True)
class RunState:
    run_id: str
    revision: int
    status: RunStatus
    task: ResolvedTask
    portfolio: PortfolioState
    insight_graph: InsightGraph
    ranking: RankingState
    coordinators: tuple[CoordinatorState, ...] = ()
    plans: tuple[PlanState, ...] = ()
    trials: tuple[TrialState, ...] = ()
    pending_actions: tuple[Action, ...] = ()
    active_agent_calls: tuple[ActiveAgentCallRef, ...] = ()
    active_engine_commands: tuple[EngineCommandRef, ...] = ()
    active_review_commands: tuple[ReviewCommandRef, ...] = ()
    planning_queue: tuple[PlanningSlot, ...] = ()
    rm_merge_queue: tuple[str, ...] = ()
    accepted_plan_refs: tuple[ArtifactRef, ...] = ()
    accepted_evidence_refs: tuple[ArtifactRef, ...] = ()
    accepted_experiment_outcome_refs: tuple[ArtifactRef, ...] = ()
    accepted_finding_refs: tuple[ArtifactRef, ...] = ()
    accepted_summary_refs: tuple[ArtifactRef, ...] = ()
    agent_sessions: tuple[AgentSession, ...] = ()
    accepted_snapshot_refs: tuple[SnapshotRef, ...] = ()
    latest_run_snapshot_ref_id: str | None = None
    memory: MemoryState | None = None
    research_outcome: ResearchOutcome = ResearchOutcome.PENDING
    completion_kind: CompletionKind | None = None
    finish_request: RunFinishRequest | None = None
    last_transition: TransitionRecord | None = None
    pause_requested: bool = False
    pause_reason: str | None = None
    failure: FailureState | None = None
    forked_from: ForkLineage | None = None
    seeded_from: RunSeedProvenance | None = None
    plan_catalog: PlanCatalogState = PlanCatalogState()
    bootstrap: BootstrapState = BootstrapState()
    operator_evaluations: tuple[OperatorEvaluationRecord, ...] = ()
    analysis_policy: dict[str, object] | None = None
    run_resources: dict[str, object] | None = None
    schema_version: int = 3
    automatic_recovery: AutomaticRecoveryPolicy = AutomaticRecoveryPolicy()
    recovery: RecoveryState | None = None
    continuation_status: RunStatus | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if self.schema_version != 3:
            raise ValueError("RunState schema_version must be 3")
        if self.forked_from is not None and self.seeded_from is not None:
            raise ValueError("RunState cannot be both forked and seeded")
        search_coordinators = tuple(
            item
            for item in self.coordinators
            if item.kind is CoordinatorKind.SEARCH
        )
        if search_coordinators and sum(
            item.original_plan_limit for item in search_coordinators
        ) != self.portfolio.max_plans:
            raise ValueError(
                "Search Coordinator original limits must equal Run max_plans"
            )
        terminal_coordinator_ids = {
            item.coordinator_id
            for item in search_coordinators
            if item.control_status
            in {
                CoordinatorControlStatus.FINISHED_EARLY,
                CoordinatorControlStatus.CANCELLED,
            }
        }
        if terminal_coordinator_ids:
            has_terminal_scope = (
                any(
                    item.coordinator_id in terminal_coordinator_ids
                    for item in self.planning_queue
                )
                or any(
                    item.coordinator_id in terminal_coordinator_ids
                    and item.status is PlanStatus.ACTIVE
                    for item in self.plans
                )
                or any(
                    item.coordinator_id in terminal_coordinator_ids
                    for item in self.active_agent_calls
                )
                or any(
                    item.coordinator_id in terminal_coordinator_ids
                    for item in self.active_engine_commands
                )
                or any(
                    item.coordinator_id in terminal_coordinator_ids
                    for item in self.active_review_commands
                )
                or any(
                    entry.split("/")[1] in terminal_coordinator_ids
                    for entry in self.rm_merge_queue
                )
                or any(
                    item.coordinator_id in terminal_coordinator_ids
                    and item.status
                    not in {
                        OperatorEvaluationStatus.COMPLETED,
                        OperatorEvaluationStatus.FAILED,
                        OperatorEvaluationStatus.CANCELLED,
                        OperatorEvaluationStatus.NOT_APPLICABLE,
                    }
                    for item in self.operator_evaluations
                )
            )
            if has_terminal_scope:
                raise ValueError("terminal Coordinator cannot retain active scope")
        if self.status is RunStatus.COMPLETED:
            if self.completion_kind is None:
                raise ValueError("completed Run requires completion kind")
        elif self.completion_kind is not None:
            raise ValueError("non-completed Run cannot retain completion kind")
        if self.memory is None:
            object.__setattr__(
                self,
                "memory",
                MemoryState("collection/none", f"{self.run_id}/RM000"),
            )
        assert self.memory is not None
        if not self.memory.run_head.startswith(f"{self.run_id}/RM"):
            raise ValueError("RunState v3 requires a qualified Run Memory head")
        for plan_ref, memory_ref in self.memory.plan_heads.items():
            if not plan_ref.startswith(f"{self.run_id}/") or not memory_ref.startswith(
                f"{plan_ref}/PM"
            ):
                raise ValueError(
                    "RunState v3 requires qualified Plan Memory heads"
                )
        for plan in self.plans:
            plan_ref = canonical_subject_ref(
                self.run_id, plan.coordinator_id, plan.plan_id
            )
            if plan.plan_memory_head is not None and not plan.plan_memory_head.startswith(
                f"{plan_ref}/PM"
            ):
                raise ValueError(
                    "RunState v3 requires qualified PlanState Memory heads"
                )
        for trial in self.trials:
            trial_ref = canonical_subject_ref(
                self.run_id,
                trial.coordinator_id,
                trial.plan_id,
                trial.trial_id,
            )
            plan_ref = trial_ref.rsplit("/", 1)[0]
            if (
                trial.trial_record_ref is not None
                and trial.trial_record_ref != f"{trial_ref}/record"
            ):
                raise ValueError(
                    "RunState v3 requires a qualified Trial Record ref"
                )
            if any(
                value is not None and not value.startswith(f"{plan_ref}/PM")
                for value in (
                    trial.plan_memory_basis,
                    trial.plan_memory_result,
                )
            ):
                raise ValueError(
                    "RunState v3 requires qualified Trial Plan Memory refs"
                )
            if any(
                value is not None and not value.startswith(f"{self.run_id}/RM")
                for value in (
                    trial.run_memory_basis,
                    trial.run_memory_result,
                )
            ):
                raise ValueError(
                    "RunState v3 requires qualified Trial Run Memory refs"
                )
        if any(
            not entry.subject_id.startswith(f"{self.run_id}/")
            for entry in self.ranking.entries
        ):
            raise ValueError(
                "RunState v3 requires qualified Ranking SubjectRefs"
            )
        if self.revision < 0:
            raise ValueError("revision must be non-negative")
        if self.last_transition is None:
            kind = "run_created" if self.revision == 0 else "state_committed"
            object.__setattr__(
                self,
                "last_transition",
                TransitionRecord.create(
                    run_id=self.run_id,
                    kind=kind,
                    subject_ref=self.run_id,
                    from_revision=None if self.revision == 0 else self.revision - 1,
                    to_revision=self.revision,
                ),
            )
        assert self.last_transition is not None
        if self.last_transition.to_revision != self.revision:
            raise ValueError("last_transition must describe the current revision")
        session_ids = tuple(session.session_id for session in self.agent_sessions)
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("Agent Session IDs must be unique")
        snapshot_ids = tuple(ref.snapshot_id for ref in self.accepted_snapshot_refs)
        if len(snapshot_ids) != len(set(snapshot_ids)):
            raise ValueError("Snapshot IDs must be unique")
        if (
            self.latest_run_snapshot_ref_id is not None
            and self.latest_run_snapshot_ref_id not in snapshot_ids
        ):
            raise ValueError("latest Run snapshot must be accepted")
        reservation_ids = tuple(item.reservation_id for item in self.planning_queue)
        if len(reservation_ids) != len(set(reservation_ids)):
            raise ValueError("planning reservation IDs must be unique")
        if tuple(item.sequence for item in self.planning_queue) != tuple(
            sorted(item.sequence for item in self.planning_queue)
        ):
            raise ValueError("planning queue must preserve FIFO sequence")
        if len(self.rm_merge_queue) != len(set(self.rm_merge_queue)):
            raise ValueError("RM merge queue Trial SubjectRefs must be unique")
        if any(not item.startswith(f"{self.run_id}/") for item in self.rm_merge_queue):
            raise ValueError("RM merge queue requires full Trial SubjectRefs")
        if self.status is RunStatus.RECOVERING:
            if self.recovery is None or self.continuation_status not in {
                RunStatus.BOOTSTRAPPING,
                RunStatus.RUNNING,
            }:
                raise ValueError("recovering Run requires durable recovery state")
        elif self.recovery is not None:
            raise ValueError("only a recovering Run may retain recovery state")
        if self.status in {RunStatus.PAUSED, RunStatus.SUSPENDED}:
            if self.continuation_status not in {
                RunStatus.BOOTSTRAPPING,
                RunStatus.RUNNING,
            }:
                raise ValueError("paused/suspended Run requires continuation_status")
        elif self.status is not RunStatus.RECOVERING:
            if self.pause_requested and self.status in {
                RunStatus.BOOTSTRAPPING,
                RunStatus.RUNNING,
            }:
                if self.continuation_status is not self.status:
                    raise ValueError(
                        "pause-requested Run must retain its active continuation"
                    )
            elif self.continuation_status is not None:
                raise ValueError(
                    "active or terminal Run cannot retain continuation_status"
                )
        if self.status is RunStatus.PAUSED:
            if not self.pause_reason:
                raise ValueError("paused Run requires a reason")

    def next_revision(self, **changes: object) -> "RunState":
        if "revision" in changes:
            raise ValueError("revision is managed by RunState")
        transition_kind = str(changes.pop("transition_kind", "state_committed"))
        subject_ref = str(changes.pop("transition_subject_ref", self.run_id))
        transition_fact_class = changes.pop("transition_fact_class", None)
        transition_logical_work_ref = changes.pop(
            "transition_logical_work_ref", None
        )
        transition_accepted_fact_refs = tuple(
            changes.pop("transition_accepted_fact_refs", ())
        )
        transition_origin_refs = tuple(
            changes.pop("transition_origin_refs", ())
        )
        next_revision = self.revision + 1
        return replace(
            self,
            revision=next_revision,
            last_transition=TransitionRecord.create(
                run_id=self.run_id,
                kind=transition_kind,
                subject_ref=subject_ref,
                from_revision=self.revision,
                to_revision=next_revision,
                fact_class=transition_fact_class,
                logical_work_ref=transition_logical_work_ref,
                accepted_fact_refs=transition_accepted_fact_refs,
                origin_refs=transition_origin_refs,
            ),
            **changes,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["bootstrap"] = self.bootstrap.to_dict()
        scoped_collections = {
            "coordinators": lambda item: canonical_subject_ref(
                self.run_id, item.coordinator_id
            ),
            "plans": lambda item: canonical_subject_ref(
                self.run_id, item.coordinator_id, item.plan_id
            ),
            "trials": lambda item: canonical_subject_ref(
                self.run_id,
                item.coordinator_id,
                item.plan_id,
                item.trial_id,
            ),
            "active_engine_commands": lambda item: canonical_subject_ref(
                self.run_id,
                item.coordinator_id,
                item.plan_id,
                item.trial_id,
            ),
            "active_review_commands": lambda item: canonical_subject_ref(
                self.run_id,
                item.coordinator_id,
                item.plan_id,
                item.trial_id,
            ),
            "planning_queue": lambda item: canonical_subject_ref(
                self.run_id, item.coordinator_id, item.plan_id
            ),
            "operator_evaluations": lambda item: canonical_subject_ref(
                self.run_id,
                item.coordinator_id or "c000",
                item.plan_id or "p000",
                item.trial_id or "base",
            ),
        }
        for name, resolver in scoped_collections.items():
            encoded = payload[name]
            source = getattr(self, name)
            for item, value in zip(source, encoded, strict=True):
                ref = resolver(item)
                value["scope"] = _scope_from_subject_ref(self.run_id, ref)
                value["subject_ref"] = ref
        for call, value in zip(
            self.active_agent_calls,
            payload["active_agent_calls"],
            strict=True,
        ):
            value["owner_scope"] = _scope_from_subject_ref(
                self.run_id, call.owner_subject_ref
            )
            value["owner_subject_ref"] = call.owner_subject_ref
            value["target_scope"] = _scope_from_subject_ref(
                self.run_id, call.target_subject_ref
            )
        for session, value in zip(
            self.agent_sessions,
            payload["agent_sessions"],
            strict=True,
        ):
            ref = canonical_subject_ref(
                self.run_id,
                session.coordinator_id,
                session.plan_id,
                session.trial_id,
            )
            value["scope"] = _scope_from_subject_ref(self.run_id, ref)
            value["subject_ref"] = ref
        catalog = payload["plan_catalog"]
        for item in catalog["active_intents"]:
            ref = canonical_subject_ref(
                self.run_id,
                item["coordinator_id"],
                item["plan_id"],
            )
            item["scope"] = _scope_from_subject_ref(self.run_id, ref)
            item["subject_ref"] = ref
        for item in catalog["source_eligible_results"]:
            ref = canonical_subject_ref(
                self.run_id,
                item["coordinator_id"],
                item["plan_id"],
            )
            item["scope"] = _scope_from_subject_ref(self.run_id, ref)
            item["subject_ref"] = ref
            item["representative_trial_ref"] = canonical_subject_ref(
                self.run_id,
                item["coordinator_id"],
                item["plan_id"],
                item["representative_trial_id"],
            )
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "RunState":
        if value.get("schema_version") != 3:
            raise ValueError("only RunState schema_version 3 is supported")
        if "completion_kind" not in value or "finish_request" not in value:
            raise ValueError("RunState schema_version 3 control fields are required")
        run_id = str(value["run_id"])
        scoped_collections = {
            "coordinators": lambda item: canonical_subject_ref(
                run_id, str(item["coordinator_id"])
            ),
            "plans": lambda item: canonical_subject_ref(
                run_id,
                str(item["coordinator_id"]),
                str(item["plan_id"]),
            ),
            "trials": lambda item: canonical_subject_ref(
                run_id,
                str(item["coordinator_id"]),
                str(item["plan_id"]),
                str(item["trial_id"]),
            ),
            "active_engine_commands": lambda item: canonical_subject_ref(
                run_id,
                str(item["coordinator_id"]),
                str(item["plan_id"]),
                str(item["trial_id"]),
            ),
            "active_review_commands": lambda item: canonical_subject_ref(
                run_id,
                str(item["coordinator_id"]),
                str(item["plan_id"]),
                str(item["trial_id"]),
            ),
            "planning_queue": lambda item: canonical_subject_ref(
                run_id,
                str(item["coordinator_id"]),
                str(item["plan_id"]),
            ),
            "operator_evaluations": lambda item: canonical_subject_ref(
                run_id,
                str(item.get("coordinator_id") or "c000"),
                str(item.get("plan_id") or "p000"),
                str(item.get("trial_id") or "base"),
            ),
        }
        for name, resolver in scoped_collections.items():
            for item in value.get(name, ()):
                expected = resolver(item)
                if item.get("subject_ref") != expected or item.get("scope") != _scope_from_subject_ref(run_id, expected):
                    raise ValueError(
                        f"RunState v3 {name} requires canonical scope and SubjectRef"
                    )
        for item in value.get("active_agent_calls", ()):
            owner_ref = agent_owner_subject_ref(
                run_id=run_id,
                role=AgentRole(str(item["role"])),
                coordinator_id=item.get("coordinator_id"),
                plan_id=item.get("plan_id"),
            )
            target_ref = str(item["target_subject_ref"])
            validate_agent_target(
                run_id=run_id,
                role=AgentRole(str(item["role"])),
                coordinator_id=item.get("coordinator_id"),
                plan_id=item.get("plan_id"),
                target_subject_ref=target_ref,
            )
            if (
                item.get("owner_subject_ref") != owner_ref
                or item.get("owner_scope")
                != _scope_from_subject_ref(run_id, owner_ref)
                or item.get("target_scope")
                != _scope_from_subject_ref(run_id, target_ref)
            ):
                raise ValueError(
                    "RunState v3 active Agent Calls require canonical owner and target identities"
                )
        for item in value.get("agent_sessions", ()):
            expected = canonical_subject_ref(
                run_id,
                item.get("coordinator_id"),
                item.get("plan_id"),
                item.get("trial_id"),
            )
            if item.get("subject_ref") != expected or item.get("scope") != _scope_from_subject_ref(run_id, expected):
                raise ValueError(
                    "RunState v3 agent_sessions require canonical scope and SubjectRef"
                )
        catalog_value = dict(value.get("plan_catalog", {}))
        for item in catalog_value.get("active_intents", ()):
            expected = canonical_subject_ref(
                run_id,
                str(item["coordinator_id"]),
                str(item["plan_id"]),
            )
            if (
                item.get("subject_ref") != expected
                or item.get("scope")
                != _scope_from_subject_ref(run_id, expected)
            ):
                raise ValueError(
                    "RunState v3 Plan Catalog intents require full SubjectRef"
                )
        for item in catalog_value.get("source_eligible_results", ()):
            expected = canonical_subject_ref(
                run_id,
                str(item["coordinator_id"]),
                str(item["plan_id"]),
            )
            trial_ref = canonical_subject_ref(
                run_id,
                str(item["coordinator_id"]),
                str(item["plan_id"]),
                str(item["representative_trial_id"]),
            )
            if (
                item.get("subject_ref") != expected
                or item.get("scope")
                != _scope_from_subject_ref(run_id, expected)
                or item.get("representative_trial_ref") != trial_ref
            ):
                raise ValueError(
                    "RunState v3 Plan Catalog results require full SubjectRefs"
                )
        task = value["task"]
        portfolio = value["portfolio"]
        graph = value["insight_graph"]
        ranking = value["ranking"]
        assert isinstance(task, dict)
        assert isinstance(portfolio, dict)
        assert isinstance(graph, dict)
        assert isinstance(ranking, dict)
        bootstrap = value.get("bootstrap", {})
        memory = value.get("memory")
        transition = value.get("last_transition")
        assert isinstance(bootstrap, dict)
        if not isinstance(memory, dict):
            raise ValueError("RunState v3 requires Memory state")
        assert transition is None or isinstance(transition, dict)
        return cls(
            run_id=str(value["run_id"]),
            revision=int(value["revision"]),
            status=RunStatus(str(value["status"])),
            task=ResolvedTask(**task),
            portfolio=PortfolioState(**portfolio),
            coordinators=tuple(
                CoordinatorState(
                    coordinator_id=str(item["coordinator_id"]),
                    kind=CoordinatorKind(str(item["kind"])),
                    original_plan_limit=int(item["original_plan_limit"]),
                    effective_plan_limit=int(item["effective_plan_limit"]),
                    control_status=CoordinatorControlStatus(
                        str(item["control_status"])
                    ),
                    requested_plan_limit=(
                        int(item["requested_plan_limit"])
                        if item.get("requested_plan_limit") is not None
                        else None
                    ),
                    control_reason=item.get("control_reason"),
                    control_requested_revision=(
                        int(item["control_requested_revision"])
                        if item.get("control_requested_revision") is not None
                        else None
                    ),
                )
                for item in value["coordinators"]
            ),
            plans=tuple(
                PlanState(
                    plan_id=str(item["plan_id"]),
                    coordinator_id=str(item["coordinator_id"]),
                    kind=PlanKind(str(item["kind"])),
                    status=PlanStatus(str(item["status"])),
                    basis_revision=int(item["basis_revision"]),
                    decision_ref_id=item.get("decision_ref_id"),
                    decision_report_ref_id=item.get("decision_report_ref_id"),
                    relation=(
                        PlanRelation(
                            kind=PlanRelationKind(str(item["relation"]["kind"])),
                            related_plan_keys=tuple(
                                PlanKey(
                                    run_id=str(key["run_id"]),
                                    coordinator_id=str(key["coordinator_id"]),
                                    plan_id=str(key["plan_id"]),
                                )
                                for key in item["relation"].get(
                                    "related_plan_keys", ()
                                )
                            ),
                            seed_artifact_ref_ids=tuple(
                                item["relation"].get(
                                    "seed_artifact_ref_ids", ()
                                )
                            ),
                        )
                        if item.get("relation") is not None
                        else None
                    ),
                    best_trial_id=item.get("best_trial_id"),
                    best_artifact_ref_id=item.get("best_artifact_ref_id"),
                    best_score=(
                        float(item["best_score"])
                        if item.get("best_score") is not None
                        else None
                    ),
                    best_secondary_score=(
                        float(item["best_secondary_score"])
                        if item.get("best_secondary_score") is not None
                        else None
                    ),
                    no_improvement_count=int(
                        item.get("no_improvement_count", 0)
                    ),
                    latest_snapshot_ref_id=item.get("latest_snapshot_ref_id"),
                    reservation_id=item.get("reservation_id"),
                    reservation_sequence=(
                        int(item["reservation_sequence"])
                        if item.get("reservation_sequence") is not None
                        else None
                    ),
                    planning_basis=item.get("planning_basis"),
                    hypothesis_comparator=item.get("hypothesis_comparator"),
                    portfolio_comparator=item.get("portfolio_comparator"),
                    plan_memory_head=item.get("plan_memory_head"),
                )
                for item in value.get("plans", ())
            ),
            trials=tuple(
                TrialState(
                    trial_id=str(item["trial_id"]),
                    coordinator_id=str(item["coordinator_id"]),
                    plan_id=str(item["plan_id"]),
                    phase=TrialPhase(str(item["phase"])),
                    kind=TrialKind(str(item["kind"])),
                    command_id=item.get("command_id"),
                    logical_command_id=item.get("logical_command_id"),
                    engine_attempt_id=item.get("engine_attempt_id"),
                    engine_attempt_index=int(item.get("engine_attempt_index", 0)),
                    engine_retry_pending=bool(
                        item.get("engine_retry_pending", False)
                    ),
                    engine_attempt_failure_ref_ids=tuple(
                        item.get("engine_attempt_failure_ref_ids", ())
                    ),
                    artifact_ref_id=item.get("artifact_ref_id"),
                    artifact_supporting_ref_ids=tuple(
                        item.get("artifact_supporting_ref_ids", ())
                    ),
                    result_refs=tuple(item.get("result_refs", ())),
                    package_ref_id=item.get("package_ref_id"),
                    authorized_unit_ids=tuple(item.get("authorized_unit_ids", ())),
                    analysis_status=item.get("analysis_status"),
                    analysis_design_ref_id=item.get("analysis_design_ref_id"),
                    analysis_review_command_id=item.get("analysis_review_command_id"),
                    analysis_review_logical_command_id=item.get(
                        "analysis_review_logical_command_id"
                    ),
                    analysis_review_attempt_id=item.get(
                        "analysis_review_attempt_id"
                    ),
                    analysis_review_attempt_index=int(
                        item["analysis_review_attempt_index"]
                    ),
                    analysis_review_retry_pending=bool(
                        item["analysis_review_retry_pending"]
                    ),
                    analysis_review_attempt_failure_ref_ids=tuple(
                        item["analysis_review_attempt_failure_ref_ids"]
                    ),
                    analysis_review_packet_ref_id=item.get("analysis_review_packet_ref_id"),
                    analysis_ref_id=item.get("analysis_ref_id"),
                    analysis_evidence_ref_id=item.get("analysis_evidence_ref_id"),
                    analysis_review_coverage_ref_id=item.get("analysis_review_coverage_ref_id"),
                    analysis_failure_ref_id=item.get("analysis_failure_ref_id"),
                    objective_comparison_ref_id=item.get(
                        "objective_comparison_ref_id"
                    ),
                    outcome=TrialOutcome(str(item.get("outcome", "pending"))),
                    archive_status=TrialArchiveStatus(
                        str(item.get("archive_status", "pending"))
                    ),
                    failure_kind=item.get("failure_kind"),
                    offline_score=(
                        float(item["offline_score"])
                        if item.get("offline_score") is not None
                        else None
                    ),
                    offline_secondary_score=(
                        float(item["offline_secondary_score"])
                        if item.get("offline_secondary_score") is not None
                        else None
                    ),
                    source_artifact_ref_ids=tuple(
                        item.get("source_artifact_ref_ids", ())
                    ),
                    analysis_packet_ref_id=item.get("analysis_packet_ref_id"),
                    plan_snapshot_ref_id=item.get("plan_snapshot_ref_id"),
                    run_snapshot_ref_id=item.get("run_snapshot_ref_id"),
                    trial_record_ref=item.get("trial_record_ref"),
                    plan_memory_basis=item.get("plan_memory_basis"),
                    plan_memory_result=item.get("plan_memory_result"),
                    run_memory_basis=item.get("run_memory_basis"),
                    run_memory_result=item.get("run_memory_result"),
                )
                for item in value.get("trials", ())
            ),
            insight_graph=InsightGraph(
                revision=int(graph["revision"]),
                nodes=tuple(
                    InsightNode(
                        insight_id=str(item["insight_id"]),
                        statement=str(item["statement"]),
                        evidence_ref_ids=tuple(item["evidence_ref_ids"]),
                    )
                    for item in graph.get("nodes", ())
                ),
            ),
            ranking=RankingState(
                metric_id=str(ranking["metric_id"]),
                direction=str(ranking.get("direction", "maximize")),
                entries=tuple(
                    RankingEntry(
                        subject_id=str(item["subject_id"]),
                        score=float(item["score"]),
                        level=str(item.get("level", "trial_level")),
                        evaluation_profile=str(
                            item.get("evaluation_profile", "offline")
                        ),
                        secondary_score=(
                            float(item["secondary_score"])
                            if item.get("secondary_score") is not None
                            else None
                        ),
                        accepted_revision=int(item.get("accepted_revision", 0)),
                        artifact_ref_id=item.get("artifact_ref_id"),
                        artifact_digest=item.get("artifact_digest"),
                        source_eligible=bool(item.get("source_eligible", False)),
                        representative_trial_id=item.get(
                            "representative_trial_id"
                        ),
                        operator_status=str(
                            item.get("operator_status", "not_scheduled")
                        ),
                        operator_result_ref=item.get("operator_result_ref"),
                    )
                    for item in ranking.get("entries", ())
                ),
                revision=int(ranking.get("revision", 0)),
            ),
            plan_catalog=PlanCatalogState(
                revision=int(
                    dict(value.get("plan_catalog", {})).get("revision", 0)
                ),
                active_intents=tuple(
                    ActivePlanIntent(
                        coordinator_id=str(item["coordinator_id"]),
                        plan_id=str(item["plan_id"]),
                        accepted_revision=int(item["accepted_revision"]),
                        decision_ref_id=str(item["decision_ref_id"]),
                        decision_report_ref_id=item.get(
                            "decision_report_ref_id"
                        ),
                        relation_kind=str(item["relation_kind"]),
                        related_plan_keys=tuple(
                            str(key) for key in item.get("related_plan_keys", ())
                        ),
                    )
                    for item in dict(value.get("plan_catalog", {})).get(
                        "active_intents", ()
                    )
                ),
                source_eligible_results=tuple(
                    SourceEligibleResult(
                        coordinator_id=str(item["coordinator_id"]),
                        plan_id=str(item["plan_id"]),
                        representative_trial_id=str(
                            item["representative_trial_id"]
                        ),
                        score=float(item["score"]),
                        evaluation_profile=str(item["evaluation_profile"]),
                        artifact_ref_id=str(item["artifact_ref_id"]),
                        artifact_digest=str(item["artifact_digest"]),
                        rm_version=str(item["rm_version"]),
                        archived_revision=int(item["archived_revision"]),
                        secondary_score=(
                            float(item["secondary_score"])
                            if item.get("secondary_score") is not None
                            else None
                        ),
                    )
                    for item in dict(value.get("plan_catalog", {})).get(
                        "source_eligible_results", ()
                    )
                ),
            ),
            pending_actions=tuple(
                Action(
                    action_id=str(item["action_id"]),
                    action_fields=dict(item.get("action_fields", {})),
                    run_id=str(item["run_id"]),
                    kind=ActionKind(str(item["kind"])),
                    subject_id=str(item["subject_id"]),
                    basis_revision=int(item["basis_revision"]),
                )
                for item in value.get("pending_actions", ())
            ),
            active_agent_calls=tuple(
                ActiveAgentCallRef(
                    run_id=str(item["run_id"]),
                    session_id=str(item["session_id"]),
                    call_id=str(item["call_id"]),
                    attempt_id=str(item["attempt_id"]),
                    role=str(item["role"]),
                    subject_id=str(item["subject_id"]),
                    basis_revision=int(item["basis_revision"]),
                    action_id=str(item["action_id"]),
                    target_subject_ref=str(item["target_subject_ref"]),
                    action_fields=dict(item.get("action_fields", {})),
                    coordinator_id=item.get("coordinator_id"),
                    plan_id=item.get("plan_id"),
                    trial_id=item.get("trial_id"),
                    retry_index=int(item["retry_index"]),
                    max_retries=int(item["max_retries"]),
                    status=str(item["status"]),
                    submitted_at=float(item["submitted_at"]),
                    last_heartbeat_at=float(item["last_heartbeat_at"]),
                    resume_handle=item.get("resume_handle"),
                    lease_generation=int(item["lease_generation"]),
                    memory_view_id=str(item["memory_view_id"]),
                    run_memory_basis=str(item["run_memory_basis"]),
                    plan_memory_basis=item.get("plan_memory_basis"),
                    reflection_index=int(item.get("reflection_index", 0)),
                    max_reflections=int(item.get("max_reflections", 0)),
                    round_status=str(item.get("round_status", "agent_running")),
                    current_delivery_ref=item.get("current_delivery_ref"),
                    current_realization_ref=(
                        ArtifactRef.from_dict(item["current_realization_ref"])
                        if isinstance(item.get("current_realization_ref"), dict)
                        else None
                    ),
                )
                for item in value.get("active_agent_calls", ())
            ),
            active_engine_commands=tuple(
                EngineCommandRef(
                    command_id=str(item["command_id"]),
                    logical_command_id=str(
                        item["logical_command_id"]
                        if "logical_command_id" in item
                        else item["command_id"]
                    ),
                    attempt_id=str(item.get("attempt_id") or "attempt-001"),
                    attempt_index=int(item.get("attempt_index") or 1),
                    coordinator_id=str(item["coordinator_id"]),
                    plan_id=str(item["plan_id"]),
                    trial_id=str(item["trial_id"]),
                    kind=str(item["kind"]),
                    status=str(item.get("status", "submitted")),
                    submitted_at=(
                        float(item["submitted_at"])
                        if item.get("submitted_at") is not None
                        else None
                    ),
                    claimed_at=(
                        float(item["claimed_at"])
                        if item.get("claimed_at") is not None
                        else None
                    ),
                    last_heartbeat_at=(
                        float(item["last_heartbeat_at"])
                        if item.get("last_heartbeat_at") is not None
                        else None
                    ),
                    liveness_deadline=(
                        float(item["liveness_deadline"])
                        if item.get("liveness_deadline") is not None
                        else None
                    ),
                )
                for item in value.get("active_engine_commands", ())
            ),
            active_review_commands=tuple(
                ReviewCommandRef(
                    **{
                        key: member
                        for key, member in item.items()
                        if key not in {"scope", "subject_ref"}
                    }
                )
                for item in value.get("active_review_commands", ())
            ),
            planning_queue=tuple(
                PlanningSlot(
                    reservation_id=str(item["reservation_id"]),
                    coordinator_id=str(item["coordinator_id"]),
                    plan_id=str(item["plan_id"]),
                    sequence=int(item["sequence"]),
                    status=str(item.get("status", "queued")),
                )
                for item in value.get("planning_queue", ())
            ),
            rm_merge_queue=tuple(
                str(item) for item in value.get("rm_merge_queue", ())
            ),
            accepted_plan_refs=tuple(
                ArtifactRef.from_dict(item)
                for item in value.get("accepted_plan_refs", ())
            ),
            accepted_evidence_refs=tuple(ArtifactRef.from_dict(item) for item in value.get("accepted_evidence_refs", ())),
            accepted_experiment_outcome_refs=tuple(
                ArtifactRef.from_dict(item)
                for item in value.get("accepted_experiment_outcome_refs", ())
            ),
            accepted_finding_refs=tuple(ArtifactRef.from_dict(item) for item in value.get("accepted_finding_refs", ())),
            accepted_summary_refs=tuple(ArtifactRef.from_dict(item) for item in value.get("accepted_summary_refs", ())),
            agent_sessions=tuple(
                AgentSession(
                    session_id=str(item["session_id"]),
                    run_id=str(item["run_id"]),
                    role=AgentRole(str(item["role"])),
                    subject_id=str(item["subject_id"]),
                    status=AgentSessionStatus(str(item.get("status", "active"))),
                    coordinator_id=item.get("coordinator_id"),
                    plan_id=item.get("plan_id"),
                    trial_id=item.get("trial_id"),
                    resume_handle=item.get("resume_handle"),
                    call_ids=tuple(item.get("call_ids", ())),
                )
                for item in value.get("agent_sessions", ())
            ),
            accepted_snapshot_refs=tuple(
                SnapshotRef(
                    snapshot_id=str(item["snapshot_id"]),
                    kind=SnapshotKind(str(item["kind"])),
                    run_id=str(item["run_id"]),
                    revision=int(item["revision"]),
                    manifest_sha256=str(item["manifest_sha256"]),
                    root=str(item["root"]),
                    coordinator_id=item.get("coordinator_id"),
                    plan_id=item.get("plan_id"),
                    trial_id=item.get("trial_id"),
                )
                for item in value.get("accepted_snapshot_refs", ())
            ),
            latest_run_snapshot_ref_id=value.get("latest_run_snapshot_ref_id"),
            memory=MemoryState(
                collection_basis=str(
                    memory.get("collection_basis", "collection/none")
                ),
                run_head=str(memory["run_head"]),
                plan_heads={
                    str(key): str(item)
                    for key, item in dict(memory.get("plan_heads", {})).items()
                },
            ),
            research_outcome=ResearchOutcome(
                str(value.get("research_outcome", "pending"))
            ),
            completion_kind=(
                CompletionKind(str(value["completion_kind"]))
                if value["completion_kind"] is not None
                else None
            ),
            finish_request=(
                RunFinishRequest(**dict(value["finish_request"]))
                if value["finish_request"] is not None
                else None
            ),
            last_transition=(
                TransitionRecord(
                    transition_id=str(transition["transition_id"]),
                    kind=str(transition["kind"]),
                    fact_class=FactClass(str(transition["fact_class"])),
                    scope={str(key): str(item) for key, item in dict(transition["scope"]).items()},
                    subject_ref=str(transition["subject_ref"]),
                    logical_work_ref=str(transition["logical_work_ref"]),
                    accepted_fact_refs=tuple(transition.get("accepted_fact_refs", ())),
                    origin_refs=tuple(transition.get("origin_refs", ())),
                    from_revision=(
                        int(transition["from_revision"])
                        if transition.get("from_revision") is not None
                        else None
                    ),
                    to_revision=int(transition["to_revision"]),
                )
                if transition is not None
                else None
            ),
            pause_requested=bool(value.get("pause_requested", False)),
            pause_reason=value.get("pause_reason"),
            continuation_status=(
                RunStatus(str(value["continuation_status"]))
                if value.get("continuation_status") is not None
                else None
            ),
            automatic_recovery=AutomaticRecoveryPolicy(
                **dict(value["automatic_recovery"])
            ),
            recovery=(
                RecoveryState(**dict(value["recovery"]))
                if value.get("recovery") is not None
                else None
            ),
            failure=FailureState(**value["failure"]) if value.get("failure") else None,
            forked_from=(
                ForkLineage(
                    lineage_root_run_id=str(
                        value["forked_from"]["lineage_root_run_id"]
                    ),
                    generation=int(value["forked_from"]["generation"]),
                    source_run_id=str(value["forked_from"]["source_run_id"]),
                    source_revision=int(value["forked_from"]["source_revision"]),
                    source_revision_ref=str(
                        value["forked_from"]["source_revision_ref"]
                    ),
                    cause=ForkCause(str(value["forked_from"]["cause"])),
                    cause_ref=str(value["forked_from"]["cause_ref"]),
                    replay_boundary_revision=int(
                        value["forked_from"]["replay_boundary_revision"]
                    ),
                    replay_boundary_ref=str(
                        value["forked_from"]["replay_boundary_ref"]
                    ),
                )
                if value.get("forked_from") is not None
                else None
            ),
            seeded_from=(
                RunSeedProvenance(
                    source_run_id=str(value["seeded_from"]["source_run_id"]),
                    source_revision=int(value["seeded_from"]["source_revision"]),
                    source_revision_ref=str(
                        value["seeded_from"]["source_revision_ref"]
                    ),
                    source_deployment_id=value["seeded_from"].get(
                        "source_deployment_id"
                    ),
                    boundary_kind=RunSeedBoundaryKind(
                        str(value["seeded_from"]["boundary_kind"])
                    ),
                    requested_frontier={
                        str(key): str(item)
                        for key, item in dict(
                            value["seeded_from"].get("requested_frontier", {})
                        ).items()
                    },
                    resolved_frontier={
                        str(key): str(item)
                        for key, item in dict(
                            value["seeded_from"].get("resolved_frontier", {})
                        ).items()
                    },
                    discarded_nonterminal_scopes=tuple(
                        str(item)
                        for item in value["seeded_from"].get(
                            "discarded_nonterminal_scopes", ()
                        )
                    ),
                )
                if value.get("seeded_from") is not None
                else None
            ),
            bootstrap=BootstrapState.from_dict(bootstrap),
            operator_evaluations=tuple(
                OperatorEvaluationRecord(
                    target_id=str(item["target_id"]),
                    target_kind=str(item["target_kind"]),
                    artifact_ref_id=item.get("artifact_ref_id"),
                    artifact_digest=item.get("artifact_digest"),
                    profile_digest=str(item["profile_digest"]),
                    status=OperatorEvaluationStatus(str(item["status"])),
                    coordinator_id=item.get("coordinator_id"),
                    plan_id=item.get("plan_id"),
                    trial_id=item.get("trial_id"),
                    command_id=item.get("command_id"),
                    logical_command_id=item.get("logical_command_id"),
                    attempt_id=item.get("attempt_id"),
                    attempt_index=int(item.get("attempt_index", 0)),
                    retry_pending=bool(item.get("retry_pending", False)),
                    failed_attempt_receipt_ids=tuple(
                        item.get("failed_attempt_receipt_ids", ())
                    ),
                    receipt_id=item.get("receipt_id"),
                    result_ref=item.get("result_ref"),
                    error=item.get("error"),
                )
                for item in value.get("operator_evaluations", ())
            ),
            analysis_policy=(
                dict(value["analysis_policy"])
                if isinstance(value.get("analysis_policy"), dict)
                else None
            ),
            run_resources=(
                dict(value["run_resources"])
                if isinstance(value.get("run_resources"), dict)
                else None
            ),
        )
