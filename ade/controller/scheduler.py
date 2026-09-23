"""Capacity reservations and completion-order admission."""

from __future__ import annotations

from enum import StrEnum
from threading import Lock

from ade.core.scheduling import PlanningReservation


class PlanningReservationBook:
    def __init__(self, *, max_active: int, planning_budget: int) -> None:
        if max_active < 1 or planning_budget < 0:
            raise ValueError("reservation capacity must be positive and budget non-negative")
        self.max_active = max_active
        self._remaining_budget = planning_budget
        self._active: dict[str, PlanningReservation] = {}
        self._sequence = 0
        self._lock = Lock()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def remaining_budget(self) -> int:
        with self._lock:
            return self._remaining_budget

    def reserve(self, call_id: str, *, basis_revision: int) -> PlanningReservation | None:
        with self._lock:
            if call_id in self._active:
                return self._active[call_id]
            if len(self._active) >= self.max_active or self._remaining_budget == 0:
                return None
            reservation = PlanningReservation(call_id, basis_revision, self._sequence)
            self._sequence += 1
            self._remaining_budget -= 1
            self._active[call_id] = reservation
            return reservation

    def complete(self, call_id: str) -> PlanningReservation:
        with self._lock:
            try:
                return self._active.pop(call_id)
            except KeyError as error:
                raise KeyError(f"unknown active planning reservation: {call_id}") from error


class AdmissionDecision(StrEnum):
    ADMIT = "admit"
    RETRY = "retry"


class CompletionAdmission:
    def decide(
        self,
        *,
        basis_revision: int,
        current_revision: int,
        conflicts: tuple[str, ...],
    ) -> AdmissionDecision:
        if basis_revision > current_revision:
            raise ValueError("delivery basis revision is in the future")
        if basis_revision < current_revision and conflicts:
            return AdmissionDecision.RETRY
        return AdmissionDecision.ADMIT
