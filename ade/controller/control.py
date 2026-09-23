"""Single-writer control loop."""

from ade.controller.executor import Executor
from ade.controller.planner import Planner
from ade.controller.reducer import Reducer
from ade.controller.state import StateCoordinator
from ade.core.run import RunState
from ade.core.outcomes import PlanCatalogUpdatedOutcome
from ade.controller.ports import RunRepository


class ControlLoop:
    def __init__(
        self,
        *,
        repository: RunRepository,
        planner: Planner,
        executor: Executor,
        reducer: Reducer,
    ) -> None:
        self.repository = repository
        self.planner = planner
        self.executor = executor
        self.reducer = reducer
        self.state = StateCoordinator(repository, reducer)

    def tick(self, run_id: str) -> RunState:
        state = self.repository.load(run_id)
        active = tuple(
            call for call in state.active_agent_calls
            if call.role == "coordinator"
        )
        if active:
            if len(active) != 1:
                raise ValueError("N=1 Control requires one active Coordinator Call")
            outcome = self.executor.collect(active[0])
            if outcome is None:
                return state
            return self.state.apply(
                run_id,
                outcome,
                event_type={
                    "AgentRetryHeldOutcome": "agent_retry_held_for_pause",
                    "AgentRetrySubmittedOutcome": "agent_retry_submitted",
                    "CoordinatorFailedOutcome": "coordinator_failed",
                    "RunSuspendedOutcome": "run_suspended",
                }.get(type(outcome).__name__, "plan_accepted"),
            )
        if (
            state.planning_queue
            and state.planning_queue[0].status == "catalog_pending"
        ):
            pending = state.planning_queue[0]
            return self.state.apply(
                run_id,
                PlanCatalogUpdatedOutcome(
                    run_id=run_id,
                    coordinator_id=pending.coordinator_id,
                    plan_id=pending.plan_id,
                    basis_revision=state.revision,
                ),
                event_type="plan_catalog_updated",
            )
        actions = self.planner.next_actions(state)
        if not actions:
            reservation = self.planner.next_reservation(state)
            if reservation is None:
                return state
            return self.state.apply(
                run_id,
                reservation,
                event_type="planning_slot_reserved",
            )
        submitted = self.executor.prepare(actions[0])
        return self.state.apply(
            run_id,
            submitted,
            event_type="coordinator_submitted",
        )
