"""Agent Call identity and delivery contracts."""

from dataclasses import dataclass
from enum import StrEnum

from ade.core.artifacts import ArtifactRef
from ade.core.scope import subject_ref


class AgentRole(StrEnum):
    COORDINATOR = "coordinator"
    ARTIFACT_BUILDER = "artifact_builder"
    ANALYZER = "analyzer"
    PLAN_SUMMARIZER = "plan_summarizer"
    RUN_SUMMARIZER = "run_summarizer"


class AttemptKind(StrEnum):
    INITIAL = "initial"
    RETRY = "retry"


class AgentCallScope(StrEnum):
    RUN = "run"
    COORDINATOR = "coordinator"
    PLAN = "plan"
    TRIAL = "trial"


class AgentSessionStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"


def agent_owner_subject_ref(
    *,
    run_id: str,
    role: AgentRole,
    coordinator_id: str | None,
    plan_id: str | None,
) -> str:
    if role is AgentRole.RUN_SUMMARIZER:
        return subject_ref(run_id)
    if role is AgentRole.COORDINATOR:
        return subject_ref(run_id, coordinator_id)
    return subject_ref(run_id, coordinator_id, plan_id)


def validate_agent_target(
    *,
    run_id: str,
    role: AgentRole,
    coordinator_id: str | None,
    plan_id: str | None,
    target_subject_ref: str,
) -> str:
    owner_ref = agent_owner_subject_ref(
        run_id=run_id,
        role=role,
        coordinator_id=coordinator_id,
        plan_id=plan_id,
    )
    target_parts = target_subject_ref.split("/")
    expected_size = 3 if role is AgentRole.COORDINATOR else 4
    owner_parts = owner_ref.split("/")
    if (
        len(target_parts) != expected_size
        or target_parts[: len(owner_parts)] != owner_parts
        or any(not part for part in target_parts)
    ):
        raise ValueError(f"{role.value} target SubjectRef does not belong to its Session owner")
    return owner_ref


@dataclass(frozen=True)
class AgentSession:
    session_id: str
    run_id: str
    role: AgentRole
    subject_id: str
    status: AgentSessionStatus = AgentSessionStatus.ACTIVE
    coordinator_id: str | None = None
    plan_id: str | None = None
    trial_id: str | None = None
    resume_handle: str | None = None
    call_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.session_id or not self.run_id or not self.subject_id:
            raise ValueError("Agent Session identity is required")
        if self.role is AgentRole.RUN_SUMMARIZER:
            if self.subject_id != self.run_id or any(
                value is not None
                for value in (self.coordinator_id, self.plan_id, self.trial_id)
            ):
                raise ValueError("Run Summarizer requires only Run scope")
        elif self.role is AgentRole.COORDINATOR:
            if self.coordinator_id != self.subject_id or any(
                value is not None for value in (self.plan_id, self.trial_id)
            ):
                raise ValueError("Coordinator requires only Coordinator scope")
        elif self.role is AgentRole.PLAN_SUMMARIZER:
            if (
                self.coordinator_id is None
                or self.plan_id != self.subject_id
                or self.trial_id is not None
            ):
                raise ValueError("Plan Summarizer requires Plan scope")
        elif self.role in {AgentRole.ARTIFACT_BUILDER, AgentRole.ANALYZER}:
            if (
                self.coordinator_id is None
                or self.plan_id != self.subject_id
                or self.trial_id is not None
            ):
                raise ValueError(f"{self.role.value} Session requires Plan owner scope")

    @property
    def scope(self) -> AgentCallScope:
        if self.role is AgentRole.RUN_SUMMARIZER:
            return AgentCallScope.RUN
        if self.role is AgentRole.COORDINATOR:
            return AgentCallScope.COORDINATOR
        if self.role is AgentRole.PLAN_SUMMARIZER:
            return AgentCallScope.PLAN
        return AgentCallScope.PLAN

    @property
    def owner_subject_ref(self) -> str:
        return agent_owner_subject_ref(
            run_id=self.run_id,
            role=self.role,
            coordinator_id=self.coordinator_id,
            plan_id=self.plan_id,
        )


@dataclass(frozen=True)
class AgentCall:
    call_id: str
    run_id: str
    role: AgentRole
    skill_id: str
    subject_id: str
    basis_revision: int
    max_retries: int
    target_subject_ref: str
    session_id: str | None = None
    scope: AgentCallScope = AgentCallScope.RUN
    coordinator_id: str | None = None
    plan_id: str | None = None
    trial_id: str | None = None
    reflection_index: int = 0

    def __post_init__(self) -> None:
        if (
            not self.call_id
            or not self.run_id
            or not self.subject_id
            or not self.target_subject_ref
        ):
            raise ValueError("Agent Call identity is required")
        validate_agent_target(
            run_id=self.run_id,
            role=self.role,
            coordinator_id=self.coordinator_id,
            plan_id=self.plan_id,
            target_subject_ref=self.target_subject_ref,
        )
        if self.session_id is not None and not self.session_id:
            raise ValueError("session_id cannot be empty")
        if (
            self.basis_revision < 0
            or self.max_retries < 0
            or self.reflection_index < 0
        ):
            raise ValueError("basis_revision and max_retries must be non-negative")
        if self.role is not AgentRole.ARTIFACT_BUILDER and self.reflection_index:
            raise ValueError("only Artifact Builder Calls may use reflection rounds")
        if self.scope is AgentCallScope.RUN:
            if any(
                item is not None
                for item in (self.coordinator_id, self.plan_id, self.trial_id)
            ):
                raise ValueError("run scope cannot identify a Coordinator, Plan, or Trial")
        elif self.scope is AgentCallScope.COORDINATOR:
            if self.coordinator_id is None or any(
                item is not None for item in (self.plan_id, self.trial_id)
            ):
                raise ValueError("coordinator scope requires only coordinator_id")
        elif self.scope is AgentCallScope.PLAN:
            if (
                self.coordinator_id is None
                or self.plan_id is None
                or self.trial_id is not None
            ):
                raise ValueError("plan scope requires coordinator_id and plan_id")
        elif self.scope is AgentCallScope.TRIAL:
            if any(
                item is None
                for item in (self.coordinator_id, self.plan_id, self.trial_id)
            ):
                raise ValueError(
                    "trial scope requires coordinator_id, plan_id, and trial_id"
                )

    @property
    def owner_subject_ref(self) -> str:
        return agent_owner_subject_ref(
            run_id=self.run_id,
            role=self.role,
            coordinator_id=self.coordinator_id,
            plan_id=self.plan_id,
        )


@dataclass(frozen=True)
class AgentCallAttempt:
    attempt_id: str
    call_id: str
    number: int
    kind: AttemptKind
    workspace_uri: str
    reflection_index: int = 0

    def __post_init__(self) -> None:
        if self.number < 0 or self.reflection_index < 0:
            raise ValueError("attempt counters must be non-negative")


@dataclass(frozen=True)
class AgentDelivery:
    call_id: str
    attempt_id: str
    artifact_refs: tuple[ArtifactRef, ...] = ()
