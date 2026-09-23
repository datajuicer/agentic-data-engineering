"""Scheduling contracts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningReservation:
    call_id: str
    basis_revision: int
    sequence: int


@dataclass(frozen=True)
class PlanningSlot:
    reservation_id: str
    coordinator_id: str
    plan_id: str
    sequence: int
    status: str = "queued"

    def __post_init__(self) -> None:
        if not self.reservation_id or not self.coordinator_id or not self.plan_id:
            raise ValueError("planning slot identity is required")
        if self.sequence < 0 or self.status not in {
            "queued",
            "planning",
            "catalog_pending",
        }:
            raise ValueError("planning slot state is invalid")
