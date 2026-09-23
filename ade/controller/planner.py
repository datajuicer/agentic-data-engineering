"""Pure deterministic action planning."""

from ade.core.actions import Action, ActionKind
from ade.core.run import RunState, RunStatus
from ade.core.bootstrap import BootstrapStatus
from ade.core.coordinator import CoordinatorControlStatus, CoordinatorKind
from ade.core.coordinator_control import allocated_plan_slots, effective_max_plans
from ade.core.plan import PlanKind, PlanStatus
from ade.core.trial import TrialKind
from ade.core.outcomes import PlanningSlotReservedOutcome
from ade.core.scope import CoordinatorKey, PlanKey, agent_action_id


class Planner:
    def next_reservation(
        self,
        state: RunState,
    ) -> PlanningSlotReservedOutcome | None:
        if state.status is not RunStatus.RUNNING:
            return None
        if state.bootstrap.enabled and state.bootstrap.status is not BootstrapStatus.COMPLETED:
            return None
        search_plans = tuple(
            plan for plan in state.plans if plan.kind is PlanKind.SEARCH
        )
        search_coordinators = tuple(
            item
            for item in state.coordinators
            if item.kind is CoordinatorKind.SEARCH
        )
        if not search_coordinators:
            return None
        if len(search_plans) + len(state.planning_queue) >= effective_max_plans(state):
            return None
        if sum(trial.kind is TrialKind.SEARCH for trial in state.trials) >= state.portfolio.max_trials:
            return None
        reserved = {item.coordinator_id for item in state.planning_queue}
        busy = {
            plan.coordinator_id
            for plan in search_plans
            if plan.status in {PlanStatus.PROPOSED, PlanStatus.ACTIVE}
        }
        coordinators = sorted(
            (
                item.coordinator_id
                for item in search_coordinators
                if item.control_status
                in {
                    CoordinatorControlStatus.ACTIVE,
                    CoordinatorControlStatus.FINISH_REQUESTED,
                }
                and item.coordinator_id not in reserved
                and item.coordinator_id not in busy
                and allocated_plan_slots(state, item.coordinator_id)
                < item.effective_plan_limit
            )
        )
        if not coordinators:
            return None
        counts = {
            coordinator_id: allocated_plan_slots(state, coordinator_id)
            for coordinator_id in coordinators
        }
        coordinator_id = min(coordinators, key=lambda value: (counts[value], value))
        plan_numbers = [
            int(plan.plan_id[1:])
            for plan in search_plans
            if plan.coordinator_id == coordinator_id
            and plan.plan_id.startswith("p")
            and plan.plan_id[1:].isdigit()
        ]
        sequence = max(
            (
                *(item.sequence for item in state.planning_queue),
                *(
                    plan.reservation_sequence
                    for plan in search_plans
                    if plan.reservation_sequence is not None
                ),
            ),
            default=-1,
        ) + 1
        return PlanningSlotReservedOutcome(
            run_id=state.run_id,
            coordinator_id=coordinator_id,
            plan_id=f"p{max(plan_numbers, default=0) + 1:03d}",
            reservation_id=f"planning-{state.run_id}-{sequence:06d}",
            sequence=sequence,
            basis_revision=state.revision,
        )

    def next_actions(self, state: RunState) -> tuple[Action, ...]:
        if state.status is not RunStatus.RUNNING:
            return ()
        if state.bootstrap.enabled and state.bootstrap.status is not BootstrapStatus.COMPLETED:
            return ()
        if any(
            call.role == "coordinator" for call in state.active_agent_calls
        ):
            return ()
        if not state.planning_queue or state.planning_queue[0].status != "queued":
            return ()
        reservation = state.planning_queue[0]
        return (
            Action(
                action_id=agent_action_id(
                    owner_subject_ref=CoordinatorKey(
                        state.run_id, reservation.coordinator_id
                    ).subject_ref,
                    target_subject_ref=PlanKey(
                        state.run_id,
                        reservation.coordinator_id,
                        reservation.plan_id,
                    ).subject_ref,
                    role_segment="coordinator",
                    basis_revision=state.revision,
                ),
                run_id=state.run_id,
                kind=ActionKind.CALL_AGENT,
                subject_id=reservation.coordinator_id,
                basis_revision=state.revision,
            ),
        )
