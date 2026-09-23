"""Coordinator state."""

from dataclasses import dataclass
from enum import StrEnum


class CoordinatorKind(StrEnum):
    BOOTSTRAP = "bootstrap"
    SEARCH = "search"


class CoordinatorControlStatus(StrEnum):
    ACTIVE = "active"
    FINISH_REQUESTED = "finish_requested"
    CANCEL_REQUESTED = "cancel_requested"
    FINISHED_EARLY = "finished_early"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CoordinatorState:
    coordinator_id: str
    kind: CoordinatorKind
    original_plan_limit: int
    effective_plan_limit: int
    control_status: CoordinatorControlStatus = CoordinatorControlStatus.ACTIVE
    requested_plan_limit: int | None = None
    control_reason: str | None = None
    control_requested_revision: int | None = None

    def __post_init__(self) -> None:
        if not self.coordinator_id:
            raise ValueError("Coordinator identity is required")
        if self.kind is CoordinatorKind.BOOTSTRAP:
            if self.original_plan_limit != 0 or self.effective_plan_limit != 0:
                raise ValueError("Bootstrap Coordinator cannot own Search Plan budget")
            if self.control_status is not CoordinatorControlStatus.ACTIVE:
                raise ValueError("Bootstrap Coordinator cannot be operator-controlled")
        elif (
            self.original_plan_limit < 1
            or self.effective_plan_limit < 0
            or self.effective_plan_limit > self.original_plan_limit
        ):
            raise ValueError("Search Coordinator Plan limits are invalid")
        requested = self.control_status in {
            CoordinatorControlStatus.FINISH_REQUESTED,
            CoordinatorControlStatus.CANCEL_REQUESTED,
            CoordinatorControlStatus.FINISHED_EARLY,
            CoordinatorControlStatus.CANCELLED,
        }
        if requested:
            if (
                self.requested_plan_limit is None
                or self.requested_plan_limit < 0
                or self.requested_plan_limit > self.original_plan_limit
                or self.effective_plan_limit < self.requested_plan_limit
                or not (self.control_reason or "").strip()
                or self.control_requested_revision is None
                or self.control_requested_revision < 0
            ):
                raise ValueError("controlled Coordinator requires request facts")
        elif any(
            value is not None
            for value in (
                self.requested_plan_limit,
                self.control_reason,
                self.control_requested_revision,
            )
        ):
            raise ValueError("active Coordinator cannot retain control request facts")
        elif self.effective_plan_limit != self.original_plan_limit:
            raise ValueError("active Coordinator must retain its original Plan limit")
