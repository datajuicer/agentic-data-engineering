"""Pure Coordinator control and effective-budget projections."""

from __future__ import annotations

from ade.core.coordinator import CoordinatorKind
from ade.core.plan import PlanKind
from ade.core.run import RunState


def allocated_plan_slots(state: RunState, coordinator_id: str) -> int:
    plans = {
        plan.reservation_id or f"plan:{plan.plan_id}"
        for plan in state.plans
        if plan.kind is PlanKind.SEARCH
        and plan.coordinator_id == coordinator_id
    }
    queued = {
        slot.reservation_id
        for slot in state.planning_queue
        if slot.coordinator_id == coordinator_id
    }
    return len(plans | queued)


def effective_max_plans(state: RunState) -> int:
    return sum(
        item.effective_plan_limit
        for item in state.coordinators
        if item.kind is CoordinatorKind.SEARCH
    )
