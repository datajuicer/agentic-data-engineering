"""Plan state."""

from dataclasses import dataclass
from enum import StrEnum

from ade.core.scope import PlanKey


class PlanStatus(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class PlanKind(StrEnum):
    BOOTSTRAP = "bootstrap"
    SEARCH = "search"


class PlanRelationKind(StrEnum):
    NEW_DIRECTION = "new_direction"
    REVISIT = "revisit"
    CONTRADICTION = "contradiction"
    COMBINE = "combine"


@dataclass(frozen=True)
class PlanRelation:
    kind: PlanRelationKind
    related_plan_keys: tuple[PlanKey, ...] = ()
    seed_artifact_ref_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.related_plan_keys)) != len(self.related_plan_keys):
            raise ValueError("Plan relation keys must be unique")
        if any(not value for value in self.seed_artifact_ref_ids):
            raise ValueError("Plan relation seed artifact IDs must be non-empty")
        if self.kind is PlanRelationKind.NEW_DIRECTION:
            if self.related_plan_keys:
                raise ValueError("new_direction cannot reference historical Plans")
            if len(self.seed_artifact_ref_ids) != 1:
                raise ValueError("new_direction requires one baseline seed")
        elif self.kind in {
            PlanRelationKind.REVISIT,
            PlanRelationKind.CONTRADICTION,
        }:
            if len(self.related_plan_keys) != 1:
                raise ValueError(f"{self.kind.value} requires exactly one related Plan")
            if len(self.seed_artifact_ref_ids) != 1:
                raise ValueError(f"{self.kind.value} requires one seed artifact")
        elif self.kind is PlanRelationKind.COMBINE:
            if len(self.related_plan_keys) < 2:
                raise ValueError("combine requires at least two related Plans")
            if len(self.seed_artifact_ref_ids) != len(self.related_plan_keys):
                raise ValueError("combine requires one seed per related Plan")


@dataclass(frozen=True)
class PlanState:
    plan_id: str
    coordinator_id: str
    kind: PlanKind = PlanKind.SEARCH
    status: PlanStatus = PlanStatus.PROPOSED
    basis_revision: int = 0
    decision_ref_id: str | None = None
    decision_report_ref_id: str | None = None
    relation: PlanRelation | None = None
    best_trial_id: str | None = None
    best_artifact_ref_id: str | None = None
    best_score: float | None = None
    best_secondary_score: float | None = None
    no_improvement_count: int = 0
    latest_snapshot_ref_id: str | None = None
    reservation_id: str | None = None
    reservation_sequence: int | None = None
    planning_basis: dict[str, object] | None = None
    hypothesis_comparator: dict[str, object] | None = None
    portfolio_comparator: dict[str, object] | None = None
    plan_memory_head: str | None = None

    def __post_init__(self) -> None:
        if not self.plan_id or not self.coordinator_id:
            raise ValueError("Plan identity is required")
        if self.basis_revision < 0 or self.no_improvement_count < 0:
            raise ValueError("Plan revisions and counters must be non-negative")
        if self.reservation_sequence is not None and self.reservation_sequence < 0:
            raise ValueError("Plan reservation sequence must be non-negative")
        if (self.reservation_id is None) != (self.reservation_sequence is None):
            raise ValueError("Plan reservation identity must be set together")
        best_fields = (
            self.best_trial_id,
            self.best_artifact_ref_id,
            self.best_score,
        )
        if any(value is None for value in best_fields) and any(
            value is not None for value in best_fields
        ):
            raise ValueError("Plan best fields must be set together")
