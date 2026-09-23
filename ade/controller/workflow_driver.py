"""Advance one durable boundary of the minimal trial workflow."""

from __future__ import annotations

from collections.abc import Mapping
import json
import time
from typing import Callable

from ade.agent_runtime.runtime import AcceptedCall, RejectedCall
from ade.agent_runtime.service import AgentCallService
from ade.controller.control import ControlLoop
from ade.controller.operator_evaluation import TrialOperatorEvaluationDriver
from ade.controller.ports import RunRepository
from ade.controller.reducer import InvalidOutcomeError
from ade.controller.state import StateCoordinator
from ade.core.coordinator import CoordinatorControlStatus, CoordinatorKind
from ade.core.coordinator_control import allocated_plan_slots, effective_max_plans
from ade.controller.trial_lifecycle import TrialLifecycle
from ade.controller.bootstrap import BootstrapWorkflowDriver
from ade.core.agent import AgentRole
from ade.core.artifacts import ArtifactRef
from ade.core.bootstrap import BootstrapStatus
from ade.core.failures import FailureState
from ade.core.outcomes import (
    AgentCallSubmittedOutcome,
    AgentRetryHeldOutcome,
    AgentRetrySubmittedOutcome,
    BuilderReflectionSubmittedOutcome,
    CoordinatorFinishedEarlyOutcome,
    CoordinatorCancelledOutcome,
    CoordinatorPlanCancelledOutcome,
    CoordinatorPlanningCancelledOutcome,
    LocalJudgeReplacedOutcome,
    RunCompletedOutcome,
    RunFinishedEarlyOutcome,
    RunPausedOutcome,
    RunSuspendedOutcome,
    TrialProposedOutcome,
)
from ade.core.plan import PlanKind, PlanStatus
from ade.core.run import RunState, RunStatus
from ade.core.reconcile import Advanced, ReconcileResult, Terminal, Waiting
from ade.core.scope import TrialKey, agent_action_id
from ade.core.trial import TrialArchiveStatus, TrialKind, TrialPhase
from ade.local_rubric_judge.lifecycle import (
    LocalJudgeBinding,
    RunResourceAdmissionError,
)
from ade.harness.run_cleanup import cleanup_cancelled_coordinator
from ade.tasks.contracts import (
    AnalysisAdmissionError,
    ArtifactCompilationError,
    ArtifactRealizationIntegrityError,
    SummaryAdmissionError,
)


class WorkflowDriver:
    def __init__(
        self,
        *,
        repository: RunRepository,
        calls: AgentCallService,
        control: ControlLoop,
        trials: TrialLifecycle,
        engine_inputs: Mapping[str, dict[str, object]],
        bootstrap: BootstrapWorkflowDriver | None = None,
        operator_evaluation: TrialOperatorEvaluationDriver | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.repository = repository
        self.calls = calls
        self.control = control
        self.trials = trials
        self.engine_inputs = dict(engine_inputs)
        self.bootstrap = bootstrap
        self.operator_evaluation = operator_evaluation
        self.clock = clock
        self.state = StateCoordinator(repository)

    def reconcile_once(self, run_id: str) -> ReconcileResult:
        before = self.repository.load(run_id)
        if before.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            return Terminal(before.revision, before.status)
        try:
            after = self._advance_once(run_id)
        except InvalidOutcomeError as error:
            if not str(error).startswith("stale outcome basis "):
                raise
            # An operator transition (for example, pause) may commit between
            # the initial load and an external receipt collection. Observe
            # that newer state instead of killing the supervisor.
            after = self.repository.load(run_id)
        delta = after.revision - before.revision
        if delta not in {0, 1}:
            raise RuntimeError(
                "one reconciliation must commit zero or one transition"
            )
        if delta == 1:
            transition = after.last_transition
            return Advanced(
                from_revision=before.revision,
                to_revision=after.revision,
                transition=transition.kind,
                subject_ref=transition.subject_ref,
            )
        reason, wakeup_ref = self._waiting_reason(after)
        return Waiting(after.revision, reason, wakeup_ref)

    def _advance_once(self, run_id: str) -> RunState:
        state = self.repository.load(run_id)
        if state.pause_requested:
            paused = self._settle_requested_pause(run_id, state)
            if paused is not None:
                return paused
            if not any(
                call.status != "retry_pending"
                for call in state.active_agent_calls
            ):
                return state
        cancelled = self._settle_cancelled_coordinator(run_id, state)
        if cancelled is not None:
            return cancelled
        recovery_ready = True
        if state.status is RunStatus.RECOVERING:
            recovery_basis_revision = state.revision
            state, recovery_ready = self._ensure_recovery_dependencies(state)
            if state.revision != recovery_basis_revision:
                return state
            if state.status is not RunStatus.RECOVERING:
                return state
            if recovery_ready:
                retried = self._submit_recovery_attempt(state)
                if retried.revision != state.revision:
                    return retried
        if self.operator_evaluation is not None:
            operator_state = (
                self.operator_evaluation.reconcile(
                    run_id, allow_submit=False
                )
                if state.pause_requested
                or state.status in {RunStatus.PAUSED, RunStatus.SUSPENDED}
                or (
                    state.status is RunStatus.RECOVERING
                    and not recovery_ready
                )
                else self.operator_evaluation.reconcile(run_id)
            )
            if operator_state.revision != state.revision:
                return operator_state
            state = operator_state
        if state.status in {RunStatus.PAUSED, RunStatus.SUSPENDED}:
            return state
        if state.bootstrap.enabled and state.bootstrap.status is not BootstrapStatus.COMPLETED:
            if state.status is RunStatus.RECOVERING and not recovery_ready:
                return state
            if self.bootstrap is None:
                raise RuntimeError("Agent Search bootstrap driver is not configured")
            return self.bootstrap.tick(run_id)
        if state.status not in {RunStatus.RUNNING, RunStatus.RECOVERING}:
            return state
        if state.status is RunStatus.RUNNING:
            settled = self._settle_finished_coordinator(run_id, state)
            if settled is not None:
                return settled
            planning = self.control.tick(run_id)
            if planning.revision != state.revision:
                return planning
            state = planning
            self._resubmit_external_work(state)
        search_trials = tuple(
            item for item in state.trials if item.kind is TrialKind.SEARCH
        )
        search_plans = tuple(
            plan for plan in state.plans if plan.kind is PlanKind.SEARCH
        )
        active_plans = tuple(
            plan for plan in search_plans if plan.status is PlanStatus.ACTIVE
        )
        for plan in active_plans:
            plan_trial_count = sum(
                trial.coordinator_id == plan.coordinator_id
                and trial.plan_id == plan.plan_id
                for trial in search_trials
            )
            plan_has_unfinished = any(
                trial.coordinator_id == plan.coordinator_id
                and trial.plan_id == plan.plan_id
                and trial.phase is not TrialPhase.ARCHIVED
                for trial in search_trials
            )
            if (
                not plan_has_unfinished
                and plan_trial_count < state.portfolio.max_trials_per_plan
                and len(search_trials) < state.portfolio.max_trials
            ):
                return self.state.apply(
                    run_id,
                    TrialProposedOutcome(
                        run_id=run_id,
                        coordinator_id=plan.coordinator_id,
                        plan_id=plan.plan_id,
                        trial_id=f"t{plan_trial_count + 1:03d}",
                        basis_revision=state.revision,
                    ),
                    event_type="trial_proposed",
                )
        trial = next(
            (
                item
                for item in search_trials
                if item.phase is not TrialPhase.ARCHIVED
                and not self._is_recovery_target(state, item)
                and self._trial_is_runnable(state, item)
            ),
            None,
        )
        if trial is not None:
            trial_key = TrialKey(
                state.run_id,
                trial.coordinator_id,
                trial.plan_id,
                trial.trial_id,
            )
            if trial.phase is TrialPhase.CREATED:
                active = self.calls.prepare(
                    run_id=state.run_id,
                    role=AgentRole.ARTIFACT_BUILDER,
                    subject_id=trial.trial_id,
                    basis_revision=state.revision,
                    action_id=agent_action_id(
                        owner_subject_ref=trial_key.plan.subject_ref,
                        target_subject_ref=trial_key.subject_ref,
                        role_segment="artifact-builder",
                        basis_revision=state.revision,
                    ),
                    target_subject_ref=trial_key.subject_ref,
                    coordinator_id=trial.coordinator_id,
                    plan_id=trial.plan_id,
                    action_fields={
                        "source_artifact_ref_ids": list(
                            trial.source_artifact_ref_ids
                        ),
                    },
                )
                return self._submit_trial_call(state, trial, active)
            if trial.phase is TrialPhase.BUILDING_ARTIFACT:
                return self._advance_builder_round(state, trial, trial_key)
            if trial.phase is TrialPhase.ARTIFACT_READY:
                try:
                    engine_input = self.engine_inputs[state.task.task_id]
                except KeyError as error:
                    raise ValueError(
                        f"missing Engine input for task {state.task.task_id}"
                    ) from error
                self.trials.submit_engine(run_id, trial_key, dict(engine_input))
                return self.repository.load(run_id)
            if trial.phase is TrialPhase.ENGINE_RUNNING:
                if trial.engine_retry_pending:
                    try:
                        engine_input = self.engine_inputs[state.task.task_id]
                    except KeyError as error:
                        raise ValueError(
                            f"missing Engine input for task {state.task.task_id}"
                        ) from error
                    self.trials.submit_engine(
                        run_id, trial_key, dict(engine_input)
                    )
                    return self.repository.load(run_id)
                if not trial.command_id:
                    raise ValueError("queued Trial has no command_id")
                self.trials.queue.submit(
                    self.repository.load_engine_command(
                        state.run_id,
                        trial.coordinator_id,
                        trial.plan_id,
                        trial.trial_id,
                        trial.command_id,
                    )
                )
                self.trials.queue.expire_stale()
                if not self.trials.queue.has_receipt(trial.command_id):
                    return state
                receipt = self.trials.queue.load_receipt(trial.command_id)
                if (
                    state.status is RunStatus.RECOVERING
                    and receipt.status.value == "failed"
                    and receipt.retryable
                ):
                    return state
                self.trials.collect_engine(run_id, trial.command_id)
                return self.repository.load(run_id)
            if trial.phase is TrialPhase.EVIDENCE_READY:
                active = self.calls.prepare(
                    run_id=state.run_id,
                    role=AgentRole.ANALYZER,
                    subject_id=trial.trial_id,
                    basis_revision=state.revision,
                    action_id=agent_action_id(
                        owner_subject_ref=trial_key.plan.subject_ref,
                        target_subject_ref=trial_key.subject_ref,
                        role_segment="analyzer-review-design",
                        basis_revision=state.revision,
                    ),
                    target_subject_ref=trial_key.subject_ref,
                    coordinator_id=trial.coordinator_id,
                    plan_id=trial.plan_id,
                    action_fields={"stage": "review_design"},
                )
                return self._submit_trial_call(state, trial, active)
            if trial.phase is TrialPhase.ANALYSIS_DESIGNING:
                result = self._execute_active(state, trial, AgentRole.ANALYZER)
                if result is None:
                    return state
                if isinstance(result, RejectedCall):
                    retried = self._retry_agent(state, result, result.validation)
                    if retried is not None:
                        return retried
                    return self.trials.fail_analysis(
                        run_id, trial_key, result, self._rejection_reason(result)
                    )
                try:
                    return self.trials.submit_analysis_review(
                        run_id, trial_key, result
                    )
                except AnalysisAdmissionError as error:
                    retried = self._retry_agent(state, result, error.report)
                    if retried is not None:
                        return retried
                    return self.trials.fail_analysis(
                        run_id,
                        trial_key,
                        result,
                        "; ".join(item.message for item in error.report.violations),
                    )
            if trial.phase is TrialPhase.REVIEW_RUNNING:
                if trial.analysis_review_retry_pending:
                    return self.trials.retry_analysis_review(run_id, trial_key)
                if not trial.analysis_review_command_id:
                    raise ValueError("Review-running Trial has no command_id")
                if self.trials.review_queue is None:
                    raise ValueError("Review-running Trial has no Review queue")
                command = self.repository.load_review_command(
                    state.run_id,
                    trial.coordinator_id,
                    trial.plan_id,
                    trial.trial_id,
                    trial.analysis_review_command_id,
                )
                self.trials.review_queue.submit(command)
                if not self.trials.review_queue.has_receipt(command.command_id):
                    return state
                if (
                    state.status is RunStatus.RECOVERING
                    and self.trials.analysis_review_receipt_requires_recovery(
                        run_id, command.command_id
                    )
                ):
                    return state
                return self.trials.collect_analysis_review(run_id, command.command_id)
            if trial.phase is TrialPhase.REVIEW_READY:
                active = self.calls.prepare(
                    run_id=state.run_id,
                    role=AgentRole.ANALYZER,
                    subject_id=trial.trial_id,
                    basis_revision=state.revision,
                    action_id=agent_action_id(
                        owner_subject_ref=trial_key.plan.subject_ref,
                        target_subject_ref=trial_key.subject_ref,
                        role_segment="analyzer-synthesis",
                        basis_revision=state.revision,
                    ),
                    target_subject_ref=trial_key.subject_ref,
                    coordinator_id=trial.coordinator_id,
                    plan_id=trial.plan_id,
                    action_fields={"stage": "synthesis"},
                )
                return self._submit_trial_call(state, trial, active)
            if trial.phase is TrialPhase.ANALYZING:
                result = self._execute_active(state, trial, AgentRole.ANALYZER)
                if result is None:
                    return state
                if isinstance(result, RejectedCall):
                    retried = self._retry_agent(state, result, result.validation)
                    if retried is not None:
                        return retried
                    return self.trials.fail_analysis(
                        run_id,
                        trial_key,
                        result,
                        self._rejection_reason(result),
                    )
                try:
                    return self.trials.accept_analysis(run_id, trial_key, result)
                except AnalysisAdmissionError as error:
                    retried = self._retry_agent(state, result, error.report)
                    if retried is not None:
                        return retried
                    return self.trials.fail_analysis(
                        run_id,
                        trial_key,
                        result,
                        "; ".join(item.message for item in error.report.violations),
                    )
            if trial.phase is TrialPhase.ANALYSIS_READY:
                active = self.calls.prepare(
                    run_id=state.run_id,
                    role=AgentRole.PLAN_SUMMARIZER,
                    subject_id=trial.plan_id,
                    basis_revision=state.revision,
                    action_id=agent_action_id(
                        owner_subject_ref=trial_key.plan.subject_ref,
                        target_subject_ref=trial_key.subject_ref,
                        role_segment="plan-summarizer",
                        basis_revision=state.revision,
                    ),
                    target_subject_ref=trial_key.subject_ref,
                    coordinator_id=trial.coordinator_id,
                )
                return self._submit_trial_call(state, trial, active)
            if trial.phase is TrialPhase.PLAN_SUMMARIZING:
                delivery = self._execute_active(
                    state, trial, AgentRole.PLAN_SUMMARIZER
                )
                if delivery is None:
                    return state
                if isinstance(delivery, RejectedCall):
                    retried = self._retry_agent(state, delivery, delivery.validation)
                    if retried is not None:
                        return retried
                    return self.trials.default_summary(
                        run_id, delivery, self._rejection_reason(delivery)
                    )
                return self._accept_summary(run_id, delivery)
            if trial.phase is TrialPhase.PLAN_SUMMARY_READY:
                active = self.calls.prepare(
                    run_id=state.run_id,
                    role=AgentRole.RUN_SUMMARIZER,
                    subject_id=state.run_id,
                    basis_revision=state.revision,
                    action_id=agent_action_id(
                        owner_subject_ref=state.run_id,
                        target_subject_ref=trial_key.subject_ref,
                        role_segment="run-summarizer",
                        basis_revision=state.revision,
                    ),
                    target_subject_ref=trial_key.subject_ref,
                )
                return self._submit_trial_call(state, trial, active)
            if trial.phase is TrialPhase.RUN_SUMMARIZING:
                delivery = self._execute_active(
                    state, trial, AgentRole.RUN_SUMMARIZER
                )
                if delivery is None:
                    return state
                if isinstance(delivery, RejectedCall):
                    retried = self._retry_agent(state, delivery, delivery.validation)
                    if retried is not None:
                        return retried
                    return self.trials.default_summary(
                        run_id, delivery, self._rejection_reason(delivery)
                    )
                return self._accept_summary(run_id, delivery)
            raise ValueError(f"unsupported Trial phase: {trial.phase}")
        if (
            len(search_plans) == effective_max_plans(state)
            and all(
                plan.decision_ref_id is None
                or sum(
                    trial.coordinator_id == plan.coordinator_id
                    and trial.plan_id == plan.plan_id
                    for trial in search_trials
                )
                == state.portfolio.max_trials_per_plan
                for plan in search_plans
            )
            and all(
                plan.status in {
                    PlanStatus.COMPLETED,
                    PlanStatus.FAILED,
                    PlanStatus.REJECTED,
                    PlanStatus.CANCELLED,
                }
                for plan in search_plans
            )
            and all(item.phase is TrialPhase.ARCHIVED for item in search_trials)
            and not state.planning_queue
            and not state.rm_merge_queue
            and not state.active_agent_calls
            and not state.active_engine_commands
            and not state.active_review_commands
        ):
            return self._complete(run_id, state)
        return state

    def _ensure_recovery_dependencies(
        self,
        state: RunState,
    ) -> tuple[RunState, bool]:
        recovery = state.recovery
        if recovery is None:
            raise RuntimeError("recovering Run lacks durable recovery state")
        if recovery.failure_kind == "harness_review_packet_invalid":
            return state, True
        if state.run_resources is None:
            return state, True
        admission = getattr(self.repository, "run_resource_admission", None)
        if admission is None:
            return self._suspend_recovery(
                state,
                code="automatic_recovery_admission_unavailable",
                message="GPU Run has no resource admission service",
            ), False
        resources = state.run_resources
        ray = resources["ray_cluster"]
        judge = resources["local_judge"]
        try:
            ready = admission.admit(
                run_id=state.run_id,
                binding=LocalJudgeBinding.from_dict(
                    {
                        "cluster_id": ray["cluster_id"],
                        "ray_address": ray["address"],
                        "gpu_count": judge["gpu_count"],
                        "model_path": judge["model_path"],
                        "model_digest": judge["model_digest"],
                        "gateway_port": judge["gateway_port"],
                        "protocol": judge["protocol"],
                    }
                ),

            )
        except RunResourceAdmissionError as error:
            unsafe = error.code in {
                "fork_binding_changed",
                "cluster_already_leased",
                "persisted_lease_conflict",
                "ray_judge_capacity_missing",
                "run_binding_changed",
            }
            if unsafe or self.clock() >= recovery.readiness_deadline:
                return self._suspend_recovery(
                    state,
                    code=f"automatic_recovery_{error.code}",
                    message=str(error),
                ), False
            return state, False
        if ready.state_path != judge.get("state_path"):
            return self._suspend_recovery(
                state,
                code="automatic_recovery_binding_mismatch",
                message=(
                    "Local Judge attachment path differs from the frozen Run binding"
                ),
            ), False
        if ready.launch_id != judge.get("launch_id"):
            return self.state.apply(
                state.run_id,
                LocalJudgeReplacedOutcome(
                    run_id=state.run_id,
                    basis_revision=state.revision,
                    launch_id=ready.launch_id,
                    state_path=ready.state_path,
                    service_state=ready.status,
                    host=ready.host,
                    gateway_url=ready.gateway_url,
                    node_id=ready.node_id,
                ),
                event_type="local_judge_replaced",
            ), False
        return state, True

    def _submit_recovery_attempt(self, state: RunState) -> RunState:
        recovery = state.recovery
        if recovery is None:
            raise RuntimeError("recovering Run lacks durable recovery state")
        if self.operator_evaluation is not None and any(
            item.retry_pending
            and item.logical_command_id == recovery.logical_work_ref
            for item in state.operator_evaluations
        ):
            return self.operator_evaluation.reconcile(state.run_id)
        if any(
            status == "retry_pending"
            for status in state.bootstrap.command_status.values()
        ):
            if self.bootstrap is None:
                raise RuntimeError("Base recovery requires the bootstrap driver")
            return self.bootstrap.tick(state.run_id)
        review_trial = next(
            (
                item
                for item in state.trials
                if item.analysis_review_retry_pending
                and TrialKey(
                    state.run_id,
                    item.coordinator_id,
                    item.plan_id,
                    item.trial_id,
                ).subject_ref
                == recovery.subject_ref
                and item.analysis_review_logical_command_id
                == recovery.logical_work_ref
            ),
            None,
        )
        if review_trial is not None:
            return self.trials.retry_analysis_review(
                state.run_id,
                TrialKey(
                    state.run_id,
                    review_trial.coordinator_id,
                    review_trial.plan_id,
                    review_trial.trial_id,
                ),
            )
        trial = next(
            (
                item
                for item in state.trials
                if item.engine_retry_pending
                and TrialKey(
                    state.run_id,
                    item.coordinator_id,
                    item.plan_id,
                    item.trial_id,
                ).subject_ref
                == recovery.subject_ref
                and item.logical_command_id == recovery.logical_work_ref
            ),
            None,
        )
        if trial is None:
            raise RuntimeError(
                "automatic recovery target has no retry-pending logical work"
            )
        try:
            engine_input = self.engine_inputs[state.task.task_id]
        except KeyError as error:
            raise ValueError(
                f"missing Engine input for task {state.task.task_id}"
            ) from error
        self.trials.submit_engine(
            state.run_id,
            TrialKey(
                state.run_id,
                trial.coordinator_id,
                trial.plan_id,
                trial.trial_id,
            ),
            dict(engine_input),
        )
        return self.repository.load(state.run_id)

    def _suspend_recovery(
        self,
        state: RunState,
        *,
        code: str,
        message: str,
    ) -> RunState:
        assert state.recovery is not None
        return self.state.apply(
            state.run_id,
            RunSuspendedOutcome(
                run_id=state.run_id,
                subject_ref=state.recovery.subject_ref,
                basis_revision=state.revision,
                failure=FailureState(code, message, retryable=True),
            ),
            event_type="run_suspended",
        )

    @staticmethod
    def _is_recovery_target(state: RunState, trial) -> bool:
        return bool(
            state.status is RunStatus.RECOVERING
            and state.recovery is not None
            and TrialKey(
                state.run_id,
                trial.coordinator_id,
                trial.plan_id,
                trial.trial_id,
            ).subject_ref
            == state.recovery.subject_ref
        )

    def _settle_requested_pause(
        self,
        run_id: str,
        state: RunState,
    ) -> RunState | None:
        if state.active_engine_commands:
            command = state.active_engine_commands[0]
            self.trials.queue.interrupt(command.command_id)
            if command.command_id in state.bootstrap.command_ids:
                if self.bootstrap is None:
                    raise RuntimeError("Agent Search bootstrap driver is not configured")
                return self.bootstrap.tick(run_id)
            if any(
                trial.command_id == command.command_id
                for trial in state.trials
            ):
                self.trials.collect_engine(run_id, command.command_id)
                return self.repository.load(run_id)
            return None
        submitted_reviews = tuple(
            command
            for command in state.active_review_commands
            if command.status != "pending_submit"
        )
        if submitted_reviews:
            command_ref = submitted_reviews[0]
            if self.trials.review_queue is None:
                raise ValueError("Active Review command has no Review queue")
            command = self.repository.load_review_command(
                state.run_id,
                command_ref.coordinator_id,
                command_ref.plan_id,
                command_ref.trial_id,
                command_ref.command_id,
            )
            self.trials.review_queue.submit(command)
            if self.trials.review_queue.has_receipt(command.command_id):
                self.trials.collect_analysis_review(run_id, command.command_id)
                return self.repository.load(run_id)
            return None
        active_agents = tuple(
            call
            for call in state.active_agent_calls
            if call.status != "retry_pending"
            and (
                call.role != AgentRole.ARTIFACT_BUILDER.value
                or call.round_status == "agent_running"
            )
        )
        if active_agents:
            active = active_agents[0]
            if not self.calls.has_terminal(active):
                return None
            if active.role == AgentRole.COORDINATOR.value:
                return self.control.tick(run_id)
            trial = next(
                (
                    item
                    for item in state.trials
                    if item.coordinator_id == active.coordinator_id
                    and item.plan_id == active.plan_id
                    and item.trial_id == active.trial_id
                ),
                None,
            )
            if trial is None:
                raise ValueError("Terminal Agent Call has no active Trial")
            return self._collect_terminal_agent_for_pause(state, trial, active)
        return self.state.apply(
            run_id,
            RunPausedOutcome(
                run_id=run_id,
                basis_revision=state.revision,
                reason=state.pause_reason or "operator_requested",
            ),
            event_type="run_paused",
        )

    def _collect_terminal_agent_for_pause(self, state, trial, active):
        """Collect one terminal Agent delivery without opening new work."""
        role = AgentRole(active.role)
        delivery = self._execute_active(state, trial, role)
        if delivery is None:
            return state
        trial_key = TrialKey(
            state.run_id,
            trial.coordinator_id,
            trial.plan_id,
            trial.trial_id,
        )
        if isinstance(delivery, RejectedCall):
            retried = self._retry_agent(state, delivery, delivery.validation)
            if retried is not None:
                return retried
            if role is AgentRole.ARTIFACT_BUILDER:
                return self.trials.fail_builder(
                    state.run_id, trial_key, delivery
                )
            if role is AgentRole.ANALYZER:
                return self.trials.fail_analysis(
                    state.run_id,
                    trial_key,
                    delivery,
                    self._rejection_reason(delivery),
                )
            if role in {AgentRole.PLAN_SUMMARIZER, AgentRole.RUN_SUMMARIZER}:
                return self.trials.default_summary(
                    state.run_id,
                    delivery,
                    self._rejection_reason(delivery),
                )
            raise ValueError(f"Unsupported Agent role during pause: {role.value}")
        if role is AgentRole.ARTIFACT_BUILDER:
            return self.trials.record_builder_delivery(
                state.run_id, trial_key, delivery
            )
        if role is AgentRole.ANALYZER:
            if trial.phase is TrialPhase.ANALYSIS_DESIGNING:
                try:
                    return self.trials.submit_analysis_review(
                        state.run_id, trial_key, delivery
                    )
                except AnalysisAdmissionError as error:
                    retried = self._retry_agent(state, delivery, error.report)
                    if retried is not None:
                        return retried
                    return self.trials.fail_analysis(
                        state.run_id,
                        trial_key,
                        delivery,
                        "; ".join(item.message for item in error.report.violations),
                    )
            if trial.phase is TrialPhase.ANALYZING:
                try:
                    return self.trials.accept_analysis(
                        state.run_id, trial_key, delivery
                    )
                except AnalysisAdmissionError as error:
                    retried = self._retry_agent(state, delivery, error.report)
                    if retried is not None:
                        return retried
                    return self.trials.fail_analysis(
                        state.run_id,
                        trial_key,
                        delivery,
                        "; ".join(item.message for item in error.report.violations),
                    )
        if role in {AgentRole.PLAN_SUMMARIZER, AgentRole.RUN_SUMMARIZER}:
            return self._accept_summary(state.run_id, delivery)
        raise ValueError(
            f"Unsupported terminal Agent phase during pause: {role.value}/{trial.phase.value}"
        )

    @staticmethod
    def _waiting_reason(state: RunState) -> tuple[str, str | None]:
        if state.status in {
            RunStatus.PAUSED,
            RunStatus.RECOVERING,
            RunStatus.SUSPENDED,
        }:
            return state.status.value, None
        if state.active_agent_calls:
            call = state.active_agent_calls[0]
            return "agent_delivery", call.attempt_id
        if state.active_engine_commands:
            command = state.active_engine_commands[0]
            return "engine_receipt", command.command_id
        if state.active_review_commands:
            command = state.active_review_commands[0]
            return "analysis_review_receipt", command.command_id
        if state.bootstrap.enabled and state.bootstrap.status is not BootstrapStatus.COMPLETED:
            return "bootstrap_progress", None
        return "no_new_fact", None

    def _trial_is_runnable(self, state: RunState, trial) -> bool:
        if trial.phase in {
            TrialPhase.CREATED,
            TrialPhase.ARTIFACT_READY,
            TrialPhase.EVIDENCE_READY,
            TrialPhase.REVIEW_READY,
            TrialPhase.ANALYSIS_READY,
        }:
            return True
        if trial.phase is TrialPhase.PLAN_SUMMARY_READY:
            trial_key = TrialKey(
                state.run_id,
                trial.coordinator_id,
                trial.plan_id,
                trial.trial_id,
            ).subject_ref
            return bool(
                state.rm_merge_queue
                and state.rm_merge_queue[0] == trial_key
            ) and not any(
                call.role == AgentRole.RUN_SUMMARIZER.value
                for call in state.active_agent_calls
            )
        if trial.phase is TrialPhase.ENGINE_RUNNING:
            if trial.engine_retry_pending:
                return not state.pause_requested
            return bool(
                trial.command_id
                and self.trials.queue.has_receipt(trial.command_id)
            )
        if trial.phase is TrialPhase.REVIEW_RUNNING:
            return bool(
                trial.analysis_review_retry_pending
                or (
                    trial.analysis_review_command_id
                    and self.trials.review_queue is not None
                    and self.trials.review_queue.has_receipt(
                        trial.analysis_review_command_id
                    )
                )
            )
        role = {
            TrialPhase.BUILDING_ARTIFACT: AgentRole.ARTIFACT_BUILDER,
            TrialPhase.ANALYZING: AgentRole.ANALYZER,
            TrialPhase.ANALYSIS_DESIGNING: AgentRole.ANALYZER,
            TrialPhase.PLAN_SUMMARIZING: AgentRole.PLAN_SUMMARIZER,
            TrialPhase.RUN_SUMMARIZING: AgentRole.RUN_SUMMARIZER,
        }.get(trial.phase)
        if role is None:
            return False
        try:
            active = self._active_call(state, trial, role)
        except ValueError:
            return False
        return (
            self.calls.execution_mode == "inline"
            or self.calls.has_terminal(active)
        )

    def _resubmit_external_work(self, state: RunState) -> None:
        """Keep every Coordinator's durable external work dispatchable.

        Trial selection is intentionally scoped below: a waiting Engine or
        Review command must not occupy the single control reconciliation lane
        while another Coordinator has a ready Trial.  Re-submission is
        idempotent and makes that change safe across a control-process restart
        or a worker restart that recovered a claimed command.
        """
        for trial in state.trials:
            if trial.phase is TrialPhase.ENGINE_RUNNING and trial.command_id:
                self.trials.queue.submit(
                    self.repository.load_engine_command(
                        state.run_id,
                        trial.coordinator_id,
                        trial.plan_id,
                        trial.trial_id,
                        trial.command_id,
                    )
                )
            elif (
                trial.phase is TrialPhase.REVIEW_RUNNING
                and trial.analysis_review_command_id
                and self.trials.review_queue is not None
            ):
                self.trials.review_queue.submit(
                    self.repository.load_review_command(
                        state.run_id,
                        trial.coordinator_id,
                        trial.plan_id,
                        trial.trial_id,
                        trial.analysis_review_command_id,
                    )
                )

    def _complete(self, run_id: str, state: RunState) -> RunState:
        early = any(
            item.kind is CoordinatorKind.SEARCH
            and item.control_status is not CoordinatorControlStatus.ACTIVE
            for item in state.coordinators
        )
        return self.state.apply(
            run_id,
            (
                RunFinishedEarlyOutcome(
                    run_id=run_id,
                    basis_revision=state.revision,
                )
                if early
                else RunCompletedOutcome(
                    run_id=run_id,
                    basis_revision=state.revision,
                )
            ),
            event_type="run_finished_early" if early else "run_completed",
        )

    def _settle_finished_coordinator(
        self,
        run_id: str,
        state: RunState,
    ) -> RunState | None:
        for coordinator in state.coordinators:
            if (
                coordinator.kind is not CoordinatorKind.SEARCH
                or coordinator.control_status
                is not CoordinatorControlStatus.FINISH_REQUESTED
                or allocated_plan_slots(state, coordinator.coordinator_id)
                != coordinator.effective_plan_limit
            ):
                continue
            coordinator_id = coordinator.coordinator_id
            prefix = f"{state.run_id}/{coordinator_id}/"
            if (
                any(
                    item.coordinator_id == coordinator_id
                    for item in state.planning_queue
                )
                or any(
                    item.coordinator_id == coordinator_id
                    or item.target_subject_ref.startswith(prefix)
                    for item in state.active_agent_calls
                )
                or any(
                    item.coordinator_id == coordinator_id
                    for item in state.active_engine_commands
                )
                or any(
                    item.coordinator_id == coordinator_id
                    for item in state.active_review_commands
                )
                or any(item.startswith(prefix) for item in state.rm_merge_queue)
                or any(
                    item.coordinator_id == coordinator_id
                    and item.status.value == "pending"
                    for item in state.operator_evaluations
                )
                or any(
                    item.coordinator_id == coordinator_id
                    and item.status
                    not in {
                        PlanStatus.COMPLETED,
                        PlanStatus.FAILED,
                        PlanStatus.REJECTED,
                        PlanStatus.CANCELLED,
                    }
                    for item in state.plans
                )
                or any(
                    item.coordinator_id == coordinator_id
                    and item.phase is not TrialPhase.ARCHIVED
                    for item in state.trials
                )
            ):
                continue
            return self.state.apply(
                run_id,
                CoordinatorFinishedEarlyOutcome(
                    run_id=run_id,
                    coordinator_id=coordinator_id,
                    basis_revision=state.revision,
                ),
                event_type="coordinator_finished_early",
            )
        return None

    def _settle_cancelled_coordinator(
        self,
        run_id: str,
        state: RunState,
    ) -> RunState | None:
        for coordinator in state.coordinators:
            if (
                coordinator.kind is not CoordinatorKind.SEARCH
                or coordinator.control_status
                is not CoordinatorControlStatus.CANCEL_REQUESTED
            ):
                continue
            coordinator_id = coordinator.coordinator_id
            cleanup = cleanup_cancelled_coordinator(
                state=state,
                coordinator_id=coordinator_id,
                repository=self.repository,
                engine_queue=self.trials.queue,
                review_queue=self.trials.review_queue,
                calls=self.calls,
            )
            if cleanup["status"] != "complete":
                return state
            reservation = next(
                (
                    item
                    for item in state.planning_queue
                    if item.coordinator_id == coordinator_id
                ),
                None,
            )
            if reservation is not None:
                cancellation_ref = self._coordinator_cancellation_ref(
                    state,
                    coordinator_id=coordinator_id,
                    plan_id=reservation.plan_id,
                    trial_id=None,
                    target_kind="planning_reservation",
                    reason=coordinator.control_reason,
                )
                return self.state.apply(
                    run_id,
                    CoordinatorPlanningCancelledOutcome(
                        run_id=run_id,
                        coordinator_id=coordinator_id,
                        plan_id=reservation.plan_id,
                        reservation_id=reservation.reservation_id,
                        basis_revision=state.revision,
                        cancellation_ref=cancellation_ref,
                    ),
                    event_type="coordinator_planning_cancelled",
                )
            trial = next(
                (
                    item
                    for item in state.trials
                    if item.coordinator_id == coordinator_id
                    and item.phase is not TrialPhase.ARCHIVED
                    and TrialKey(
                        state.run_id,
                        item.coordinator_id,
                        item.plan_id,
                        item.trial_id,
                    ).subject_ref
                    not in state.rm_merge_queue
                ),
                None,
            )
            if trial is not None:
                return self.trials.cancel_trial(
                    run_id,
                    TrialKey(
                        state.run_id,
                        trial.coordinator_id,
                        trial.plan_id,
                        trial.trial_id,
                    ),
                    reason=coordinator.control_reason or "operator_requested",
                )
            plan = next(
                (
                    item
                    for item in state.plans
                    if item.coordinator_id == coordinator_id
                    and (
                        item.status is not PlanStatus.CANCELLED
                        or any(
                            record.coordinator_id == coordinator_id
                            and record.plan_id == item.plan_id
                            and record.status.value == "pending"
                            for record in state.operator_evaluations
                        )
                    )
                    and not any(
                        entry.startswith(
                            f"{state.run_id}/{coordinator_id}/{item.plan_id}/"
                        )
                        for entry in state.rm_merge_queue
                    )
                    and not any(
                        candidate.coordinator_id == coordinator_id
                        and candidate.plan_id == item.plan_id
                        and candidate.phase is not TrialPhase.ARCHIVED
                        for candidate in state.trials
                    )
                ),
                None,
            )
            if plan is not None:
                cancellation_ref = self._coordinator_cancellation_ref(
                    state,
                    coordinator_id=coordinator_id,
                    plan_id=plan.plan_id,
                    trial_id=None,
                    target_kind="plan",
                    reason=coordinator.control_reason,
                )
                return self.state.apply(
                    run_id,
                    CoordinatorPlanCancelledOutcome(
                        run_id=run_id,
                        coordinator_id=coordinator_id,
                        plan_id=plan.plan_id,
                        basis_revision=state.revision,
                        cancellation_ref=cancellation_ref,
                    ),
                    event_type="coordinator_plan_cancelled",
                )
            prefix = f"{state.run_id}/{coordinator_id}/"
            if (
                any(item.coordinator_id == coordinator_id for item in state.planning_queue)
                or any(
                    item.coordinator_id == coordinator_id
                    or item.target_subject_ref.startswith(prefix)
                    for item in state.active_agent_calls
                )
                or any(
                    item.coordinator_id == coordinator_id
                    for item in state.active_engine_commands
                )
                or any(
                    item.coordinator_id == coordinator_id
                    for item in state.active_review_commands
                )
                or any(item.startswith(prefix) for item in state.rm_merge_queue)
                or any(
                    item.coordinator_id == coordinator_id
                    and item.status.value == "pending"
                    for item in state.operator_evaluations
                )
            ):
                continue
            return self.state.apply(
                run_id,
                CoordinatorCancelledOutcome(
                    run_id=run_id,
                    coordinator_id=coordinator_id,
                    basis_revision=state.revision,
                ),
                event_type="coordinator_cancelled",
            )
        return None

    def _coordinator_cancellation_ref(
        self,
        state: RunState,
        *,
        coordinator_id: str,
        plan_id: str,
        trial_id: str | None,
        target_kind: str,
        reason: str | None,
    ):
        payload = (
            json.dumps(
                {
                    "schema_version": "ade.coordinator_cancellation.v1",
                    "run_id": state.run_id,
                    "coordinator_id": coordinator_id,
                    "plan_id": plan_id,
                    "trial_id": trial_id,
                    "target_kind": target_kind,
                    "basis_revision": state.revision,
                    "reason": reason or "operator_requested",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        return self.repository.put_artifact(
            state.run_id, "coordinator_cancellation", payload
        )

    def _submit_trial_call(self, state, trial, active) -> RunState:
        return self.state.apply(
            state.run_id,
            AgentCallSubmittedOutcome(
                run_id=state.run_id,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                trial_id=trial.trial_id,
                basis_revision=state.revision,
                role=active.role,
                call_ref=active,
            ),
            event_type={
                "artifact_builder": "builder_submitted",
                "analyzer": (
                    "analysis_design_submitted"
                    if active.action_fields.get("stage") == "review_design"
                    else "analysis_synthesis_submitted"
                ),
                "plan_summarizer": "plan_summarizer_submitted",
                "run_summarizer": "run_summarizer_submitted",
            }[active.role],
        )

    def _advance_builder_round(self, state, trial, trial_key: TrialKey) -> RunState:
        active = self._active_call(state, trial, AgentRole.ARTIFACT_BUILDER)
        if active.round_status == "agent_running":
            delivery = self._execute_active(
                state, trial, AgentRole.ARTIFACT_BUILDER
            )
            if delivery is None:
                return state
            if isinstance(delivery, RejectedCall):
                retried = self._retry_agent(state, delivery, delivery.validation)
                if retried is not None:
                    return retried
                return self.trials.fail_builder(
                    state.run_id, trial_key, delivery
                )
            return self.trials.record_builder_delivery(
                state.run_id, trial_key, delivery
            )
        if active.round_status == "delivery_ready":
            return self.trials.start_builder_realization(
                state.run_id, active.call_id
            )
        if active.round_status == "realization_running":
            delivery = self._execute_active(
                state, trial, AgentRole.ARTIFACT_BUILDER
            )
            if delivery is None:
                return state
            if isinstance(delivery, RejectedCall):
                retried = self._retry_agent(state, delivery, delivery.validation)
                if retried is not None:
                    return retried
                return self.trials.fail_builder(
                    state.run_id, trial_key, delivery
                )
            if not isinstance(delivery, AcceptedCall):
                raise ValueError(
                    "accepted Builder delivery is unavailable for realization"
                )
            try:
                self.trials.complete_builder_realization(
                    state.run_id, trial_key, delivery
                )
            except ArtifactRealizationIntegrityError as error:
                integrity = error.report.violations[0]
                return self.state.apply(
                    state.run_id,
                    RunSuspendedOutcome(
                        run_id=state.run_id,
                        subject_ref=active.target_subject_ref,
                        basis_revision=state.revision,
                        failure=FailureState(
                            integrity.code,
                            integrity.message,
                            retryable=False,
                        ),
                    ),
                    event_type="run_suspended",
                )
            except ArtifactCompilationError as error:
                retried = self._retry_agent(state, delivery, error.report)
                if retried is not None:
                    return retried
                return self.trials.fail_builder(
                    state.run_id,
                    trial_key,
                    delivery,
                    validation=error.report,
                )
            return self.repository.load(state.run_id)
        if active.round_status == "realization_ready":
            assert active.current_realization_ref is not None
            realization_content = self.repository.read_artifact(
                state.run_id, active.current_realization_ref
            )
            realization = json.loads(realization_content)
            # The realization is complete, so the original Builder Attempt is
            # already terminal. Revalidate its persisted output directly;
            # inspection can miss the delivery after the realization boundary
            # has consumed the Agent Port receipt.
            delivery = self.calls.revalidate_active(active)
            if delivery is None:
                return state
            if isinstance(delivery, RejectedCall):
                retried = self._retry_agent(state, delivery, delivery.validation)
                if retried is not None:
                    return retried
                return self.trials.fail_builder(
                    state.run_id, trial_key, delivery
                )
            if not isinstance(delivery, AcceptedCall):
                raise ValueError(
                    "accepted Builder delivery is unavailable for reflection"
                )
            decisions = [
                line.removeprefix("Reflection decision: ").strip()
                for artifact in delivery.output.supporting_artifacts
                if artifact.path == "design.md"
                for line in artifact.content.decode("utf-8").splitlines()
                if line.startswith("Reflection decision: ")
            ]
            decision = decisions[0] if len(decisions) == 1 else None
            should_finalize = (
                realization.get("realization_status") == "unverified"
                or active.reflection_index >= active.max_reflections
                or (
                    active.reflection_index > 0
                    and decision == "finalize"
                )
            )
            if should_finalize:
                return self.trials.finalize_builder_realization(
                    state.run_id, active.call_id
                )
            reflected = self.calls.prepare_reflection(
                active,
                self._builder_reflection_files(
                    active,
                    delivery,
                    realization_content,
                ),
            )
            return self.state.apply(
                state.run_id,
                BuilderReflectionSubmittedOutcome(
                    state.run_id,
                    state.revision,
                    reflected,
                    str(active.current_delivery_ref),
                    active.current_realization_ref,
                ),
                event_type="builder_reflection_submitted",
            )
        if active.round_status == "finalized":
            delivery = self._execute_active(
                state, trial, AgentRole.ARTIFACT_BUILDER
            )
            if isinstance(delivery, RejectedCall):
                retried = self._retry_agent(state, delivery, delivery.validation)
                if retried is not None:
                    return retried
                return self.trials.fail_builder(
                    state.run_id, trial_key, delivery
                )
            if not isinstance(delivery, AcceptedCall):
                raise ValueError(
                    "final Builder delivery is unavailable for artifact acceptance"
                )
            self.trials.accept_finalized_artifact(
                state.run_id, trial_key, delivery
            )
            return self.repository.load(state.run_id)
        raise ValueError(f"unsupported Builder round status: {active.round_status}")

    def _builder_reflection_files(self, active, delivery, realization_content: bytes):
        remaining = active.max_reflections - active.reflection_index - 1
        round_content = (
            json.dumps(
                {
                    "schema_version": "ade.builder_reflection_round.v1",
                    "reflection_index": active.reflection_index + 1,
                    "remaining_reflections": remaining,
                    "basis_revision": active.basis_revision,
                    "prior_delivery_ref": active.current_delivery_ref,
                    "prior_realization_ref": active.current_realization_ref.artifact_id,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        files = {
            "reflection/round.json": (
                round_content,
                f"{active.current_realization_ref.uri}#round",
            ),
            "reflection/builder-realization.json": (
                realization_content,
                active.current_realization_ref.uri,
            ),
        }
        realization = json.loads(realization_content)
        supporting = tuple(
            ArtifactRef.from_dict(item)
            for item in realization.get("supporting_refs", ())
        )
        reports = tuple(
            ref
            for ref in supporting
            if ref.kind
            in {
                "curriculum_realization",
                "data_selection_realization",
                "reward_compliance_report",
            }
        )
        if len(reports) != 1:
            raise ValueError("Builder reflection requires one realization report")
        report_ref = reports[0]
        report_content = self.repository.read_artifact(active.run_id, report_ref)
        report = json.loads(report_content)
        files["reflection/realization-report.json"] = (
            report_content,
            report_ref.uri,
        )
        if report_ref.kind == "data_selection_realization":
            files.update(
                {
                    "reflection/selection-result.json": (
                        self._json_bytes(report["selection"]),
                        f"{report_ref.uri}#selection",
                    ),
                    "reflection/selected-examples.jsonl": (
                        self._jsonl_bytes(report["selected_rows"]),
                        f"{report_ref.uri}#selected_rows",
                    ),
                    "reflection/judge-evidence.jsonl": (
                        self._jsonl_bytes(report.get("judge_evidence", ())),
                        f"{report_ref.uri}#judge_evidence",
                    ),
                    "reflection/judge-manifest.json": (
                        self._json_bytes(report["judge_binding"]),
                        f"{report_ref.uri}#judge_binding",
                    ),
                }
            )
        else:
            judge_manifest = {
                key: report.get(key)
                for key in (
                    "judge_binding",
                    "judge_status",
                    "judge_completed_count",
                    "judge_unavailable_count",
                )
            }
            files.update(
                {
                    "reflection/source-groups.jsonl": (
                        self._jsonl_bytes(report.get("source_records", ())),
                        f"{report_ref.uri}#source_records",
                    ),
                    "reflection/group-results.jsonl": (
                        self._jsonl_bytes(report.get("records", ())),
                        f"{report_ref.uri}#records",
                    ),
                    "reflection/judge-results.jsonl": (
                        self._jsonl_bytes(report.get("judge_results", ())),
                        f"{report_ref.uri}#judge_results",
                    ),
                    "reflection/judge-manifest.json": (
                        self._json_bytes(judge_manifest),
                        f"{report_ref.uri}#judge_manifest",
                    ),
                }
            )
        output = delivery.workspace / "output"
        for source in output.rglob("*"):
            if source.is_file() and not source.is_symlink():
                relative = source.relative_to(output).as_posix()
                files[f"reflection/prior-delivery/{relative}"] = (
                    source.read_bytes(),
                    f"{delivery.attempt.workspace_uri}#output/{relative}",
                )
        return files

    @staticmethod
    def _json_bytes(value: object) -> bytes:
        return (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()

    @staticmethod
    def _jsonl_bytes(rows) -> bytes:
        return b"".join(
            (
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode()
            for row in rows
        )

    @staticmethod
    def _active_call(state, trial, role: AgentRole):
        matches = tuple(
            item
            for item in state.active_agent_calls
            if item.role == role.value
            and (
                role is AgentRole.RUN_SUMMARIZER
                or (
                    item.coordinator_id == trial.coordinator_id
                    and item.plan_id == trial.plan_id
                )
            )
            and (
                item.trial_id == trial.trial_id
                or (
                    role is AgentRole.PLAN_SUMMARIZER
                    and item.plan_id == trial.plan_id
                )
                or role is AgentRole.RUN_SUMMARIZER
            )
        )
        if len(matches) != 1:
            raise ValueError(
                f"Trial phase requires one active {role.value} Call"
            )
        return matches[0]

    def _execute_active(self, state, trial, role: AgentRole):
        active = self._active_call(state, trial, role)
        if self.calls.is_expired(active):
            return self.calls.expire_active(active)
        return self.calls.execute_active(active)

    def _accept_summary(
        self,
        run_id: str,
        delivery: AcceptedCall,
    ) -> RunState:
        try:
            return self.trials.accept_summary(run_id, delivery)
        except SummaryAdmissionError as error:
            state = self.repository.load(run_id)
            retried = self._retry_agent(state, delivery, error.report)
            if retried is not None:
                return retried
            return self.trials.default_summary(
                run_id,
                delivery,
                "; ".join(item.message for item in error.report.violations),
            )

    def _retry_agent(self, state, previous, report) -> RunState | None:
        active = next(
            (
                item for item in state.active_agent_calls
                if item.call_id == previous.call.call_id
            ),
            None,
        )
        if (
            active is not None
            and active.status == "retry_pending"
            and state.pause_requested
        ):
            return state
        if (
            active is None
            or not report.violations
            or any(not item.repairable for item in report.violations)
        ):
            return None
        if active.retry_index >= active.max_retries:
            if any(
                item.code in {"agent_backend_failed", "heartbeat_timeout"}
                for item in report.violations
            ):
                if state.status is RunStatus.RECOVERING:
                    return state
                return self._suspend_for_agent(previous, active)
            return None
        if state.pause_requested:
            return self.state.apply(
                state.run_id,
                AgentRetryHeldOutcome(
                    run_id=state.run_id,
                    subject_id=active.subject_id,
                    basis_revision=state.revision,
                    call_id=active.call_id,
                    previous_attempt_id=active.attempt_id,
                    retry_reason=report.violations[0].code,
                ),
                event_type="agent_retry_held_for_pause",
            )
        retried = self.calls.prepare_retry(active, previous, report)
        return self.state.apply(
            state.run_id,
            AgentRetrySubmittedOutcome(
                run_id=state.run_id,
                subject_id=active.subject_id,
                basis_revision=state.revision,
                call_ref=retried,
                previous_attempt_id=active.attempt_id,
                retry_reason=report.violations[0].code,
            ),
            event_type="agent_retry_submitted",
        )

    def _suspend_for_agent(self, result: RejectedCall, active) -> RunState:
        details = self._rejection_reason(result)
        state = self.repository.load(result.call.run_id)
        return self.state.apply(
            result.call.run_id,
            RunSuspendedOutcome(
                run_id=result.call.run_id,
                subject_ref=active.target_subject_ref,
                basis_revision=state.revision,
                failure=FailureState(
                    f"{result.call.role.value}_exhausted",
                    details,
                    retryable=True,
                ),
                agent_session=result.session,
                call_id=result.call.call_id,
            ),
            event_type="run_suspended_for_agent",
        )

    @staticmethod
    def _accepted(
        role: AgentRole,
        result: AcceptedCall | RejectedCall,
    ) -> AcceptedCall:
        if isinstance(result, RejectedCall):
            details = "; ".join(item.message for item in result.validation.violations)
            raise RuntimeError(f"{role.value} delivery rejected: {details}")
        if not isinstance(result, AcceptedCall):
            raise TypeError(f"{role.value} call returned an unknown result")
        return result

    @staticmethod
    def _rejection_reason(result: RejectedCall) -> str:
        details = "; ".join(item.message for item in result.validation.violations)
        return details or "Analyzer delivery exhausted its retry budget"
