"""The only RunState transition implementation."""

from dataclasses import replace

from ade.core.agent import AgentRole, AgentSession, AgentSessionStatus
from ade.core.coordinator import CoordinatorControlStatus, CoordinatorKind
from ade.core.coordinator_control import allocated_plan_slots, effective_max_plans
from ade.core.outcomes import (
    AnalysisAcceptedOutcome,
    AnalysisFailedOutcome,
    AnalysisReviewAttemptRetryPendingOutcome,
    AnalysisReviewAttemptSubmittedOutcome,
    AnalysisReviewCompletedOutcome,
    AnalysisReviewSubmittedOutcome,
    AgentCallSubmittedOutcome,
    AgentRetryHeldOutcome,
    AgentRetrySubmittedOutcome,
    CoordinatorCallSubmittedOutcome,
    CoordinatorCancelRequestedOutcome,
    CoordinatorCancelledOutcome,
    CoordinatorPlanCancelledOutcome,
    CoordinatorPlanningCancelledOutcome,
    CoordinatorTrialCancelledOutcome,
    CoordinatorFailedOutcome,
    CoordinatorFinishedEarlyOutcome,
    CoordinatorFinishRequestedOutcome,
    ArtifactAcceptedOutcome,
    BaselineTrialRegisteredOutcome,
    BootstrapBaseAttemptRetryPendingOutcome,
    BootstrapBaseAttemptSubmittedOutcome,
    BootstrapBaseScheduledOutcome,
    BootstrapBaseTerminalOutcome,
    BootstrapCompletedOutcome,
    BootstrapFailedOutcome,
    BootstrapP000RegisteredOutcome,
    BootstrapStageAdvancedOutcome,
    BuilderProposalCompletedOutcome,
    BuilderRealizationCompletedOutcome,
    BuilderRealizationFinalizedOutcome,
    BuilderRealizationStartedOutcome,
    BuilderReflectionCompletedOutcome,
    BuilderReflectionSubmittedOutcome,
    BuilderFailedOutcome,
    EngineCompletedOutcome,
    EngineFailedOutcome,
    EngineAttemptRetryPendingOutcome,
    EngineQueuedOutcome,
    LocalJudgeReplacedOutcome,
    OperatorEvaluationAttemptRetryPendingOutcome,
    OperatorEvaluationAttemptSubmittedOutcome,
    OperatorEvaluationScheduledOutcome,
    OperatorEvaluationTerminalOutcome,
    PlanCatalogUpdatedOutcome,
    PlanningDecisionOutcome,
    PlanningSlotReservedOutcome,
    RunCompletedOutcome,
    RunCancelledOutcome,
    RunFinishedEarlyOutcome,
    RunFinishRequestedOutcome,
    RunPauseRequestedOutcome,
    RunPausedOutcome,
    RunResumedOutcome,
    RunSuspendedOutcome,
    SummaryAcceptedOutcome,
    TrialProposedOutcome,
)
from ade.core.failures import FailureState
from ade.core.bootstrap import BootstrapStatus
from ade.core.catalog import ActivePlanIntent, SourceEligibleResult
from ade.core.lifecycle import PlanStopPolicy, PlanTerminalDecision, TrialMetric
from ade.core.plan import PlanKind, PlanRelationKind, PlanState, PlanStatus
from ade.core.operator import OperatorEvaluationRecord, OperatorEvaluationStatus
from ade.core.ranking import RankingEntry, score_pair_sort_key
from ade.core.run import (
    CompletionKind,
    EngineCommandRef,
    RecoveryState,
    ResearchOutcome,
    RunState,
    RunFinishRequest,
    RunStatus,
)
from ade.core.snapshot import SnapshotKind
from ade.core.scope import PlanKey, subject_ref
from ade.core.scheduling import PlanningSlot
from ade.core.trial import (
    TrialArchiveStatus,
    TrialKind,
    TrialOutcome,
    TrialState,
    TrialPhase,
)


class InvalidOutcomeError(RuntimeError):
    pass


class Reducer:
    def apply(
        self,
        state: RunState,
        outcome: (
            PlanningDecisionOutcome
            | PlanCatalogUpdatedOutcome
            | PlanningSlotReservedOutcome
            | TrialProposedOutcome
            | AgentCallSubmittedOutcome
            | AgentRetryHeldOutcome
            | AgentRetrySubmittedOutcome
            | CoordinatorCallSubmittedOutcome
            | CoordinatorCancelRequestedOutcome
            | CoordinatorCancelledOutcome
            | CoordinatorPlanCancelledOutcome
            | CoordinatorPlanningCancelledOutcome
            | CoordinatorTrialCancelledOutcome
            | CoordinatorFailedOutcome
            | CoordinatorFinishedEarlyOutcome
            | CoordinatorFinishRequestedOutcome
            | BaselineTrialRegisteredOutcome
            | BootstrapBaseAttemptRetryPendingOutcome
            | BootstrapBaseAttemptSubmittedOutcome
            | BootstrapBaseScheduledOutcome
            | BootstrapBaseTerminalOutcome
            | BootstrapCompletedOutcome
            | BootstrapFailedOutcome
            | BootstrapP000RegisteredOutcome
            | BootstrapStageAdvancedOutcome
            | ArtifactAcceptedOutcome
            | BuilderProposalCompletedOutcome
            | BuilderRealizationCompletedOutcome
            | BuilderRealizationFinalizedOutcome
            | BuilderRealizationStartedOutcome
            | BuilderReflectionCompletedOutcome
            | BuilderReflectionSubmittedOutcome
            | BuilderFailedOutcome
            | EngineQueuedOutcome
            | EngineAttemptRetryPendingOutcome
            | EngineCompletedOutcome
            | EngineFailedOutcome
            | AnalysisAcceptedOutcome
            | AnalysisReviewSubmittedOutcome
            | AnalysisReviewCompletedOutcome
            | AnalysisFailedOutcome
            | SummaryAcceptedOutcome
            | RunCompletedOutcome
            | RunCancelledOutcome
            | RunFinishedEarlyOutcome
            | RunFinishRequestedOutcome
            | RunPauseRequestedOutcome
            | RunPausedOutcome
            | RunResumedOutcome
            | LocalJudgeReplacedOutcome
            | RunSuspendedOutcome
            | OperatorEvaluationAttemptRetryPendingOutcome
            | OperatorEvaluationAttemptSubmittedOutcome
            | OperatorEvaluationScheduledOutcome
            | OperatorEvaluationTerminalOutcome
        ),
    ) -> RunState:
        if outcome.run_id != state.run_id:
            raise InvalidOutcomeError("outcome belongs to another run")
        if outcome.basis_revision != state.revision:
            raise InvalidOutcomeError(
                f"stale outcome basis {outcome.basis_revision}, current revision {state.revision}"
            )
        if isinstance(outcome, BootstrapBaseScheduledOutcome):
            self._require_bootstrap_status(state, BootstrapStatus.PENDING)
            command_ids = tuple(
                command.command_id for command in outcome.commands
            )
            if not command_ids or len(command_ids) != len(set(command_ids)):
                raise InvalidOutcomeError(
                    "Base evaluation requires unique Commands"
                )
            if any(
                command.coordinator_id != "c000"
                or command.plan_id != "p000"
                or command.kind != "evaluate"
                or not command.logical_command_id
                or command.attempt_id != "attempt-001"
                or command.attempt_index != 1
                for command in outcome.commands
            ):
                raise InvalidOutcomeError(
                    "Base evaluation requires scoped Evaluate Commands"
                )
            return state.next_revision(
                bootstrap=replace(
                    state.bootstrap,
                    status=BootstrapStatus.BASE_EVALUATING,
                    command_ids=command_ids,
                    command_status={
                        command_id: "submitted"
                        for command_id in command_ids
                    },
                    command_logical_ids={
                        command.command_id: (
                            command.logical_command_id or command.command_id
                        )
                        for command in outcome.commands
                    },
                    command_attempt_ids={
                        command.command_id: command.attempt_id
                        for command in outcome.commands
                    },
                    command_attempt_indices={
                        command.command_id: command.attempt_index
                        for command in outcome.commands
                    },
                    metadata={"base_receipt_accepted": False},
                ),
                active_engine_commands=(
                    *state.active_engine_commands,
                    *outcome.commands,
                ),
            )
        if isinstance(outcome, BootstrapBaseAttemptRetryPendingOutcome):
            self._require_bootstrap_status(
                state, BootstrapStatus.BASE_EVALUATING
            )
            matches = tuple(
                item
                for item in state.active_engine_commands
                if item.command_id == outcome.command_id
            )
            if len(matches) != 1:
                raise InvalidOutcomeError(
                    "Base retry-pending Receipt is not active"
                )
            active = matches[0]
            if (
                active.logical_command_id != outcome.logical_command_id
                or active.attempt_id != outcome.attempt_id
                or active.attempt_index != outcome.attempt_index
                or state.bootstrap.command_logical_ids.get(outcome.command_id)
                != outcome.logical_command_id
                or state.bootstrap.command_attempt_ids.get(outcome.command_id)
                != outcome.attempt_id
                or state.bootstrap.command_attempt_indices.get(outcome.command_id)
                != outcome.attempt_index
            ):
                raise InvalidOutcomeError(
                    "Base retry-pending Receipt crossed the active Attempt fence"
                )
            operator_pause = state.pause_requested
            recovery_changes = self._retryable_attempt_state(
                state,
                subject_ref=subject_ref(state.run_id, "c000", "p000"),
                logical_work_ref=outcome.logical_command_id,
                attempt_id=outcome.attempt_id,
                attempt_index=outcome.attempt_index,
                failure_kind=outcome.failure_kind,
                message=outcome.message,
                receipt_id=outcome.receipt_id,
                lease_ref=(
                    f"{state.run_id}/c000/offline_validation/"
                    f"{outcome.logical_command_id}"
                ),
                observed_at=outcome.observed_at,
                operator_pause=operator_pause,
            )
            return state.next_revision(
                **recovery_changes,
                bootstrap=replace(
                    state.bootstrap,
                    command_status={
                        **state.bootstrap.command_status,
                        outcome.command_id: "retry_pending",
                    },
                ),
                active_engine_commands=tuple(
                    item
                    for item in state.active_engine_commands
                    if item.command_id != outcome.command_id
                ),
            )
        if isinstance(outcome, BootstrapBaseAttemptSubmittedOutcome):
            self._require_bootstrap_status(
                state, BootstrapStatus.BASE_EVALUATING
            )
            previous = outcome.previous_command_id
            command = outcome.command
            if state.bootstrap.command_status.get(previous) != "retry_pending":
                raise InvalidOutcomeError(
                    "Base retry requires a retry-pending previous Attempt"
                )
            previous_logical = state.bootstrap.command_logical_ids.get(previous)
            previous_index = state.bootstrap.command_attempt_indices.get(previous)
            if (
                not previous_logical
                or previous_index is None
                or command.logical_command_id != previous_logical
                or command.attempt_index != previous_index + 1
                or command.attempt_id
                != f"attempt-{command.attempt_index:03d}"
                or command.command_id in state.bootstrap.command_status
                or command.command_id in {
                    item.command_id for item in state.active_engine_commands
                }
            ):
                raise InvalidOutcomeError(
                    "Base retry does not create the next fenced Attempt"
                )
            current_ids = tuple(
                command.command_id if item == previous else item
                for item in state.bootstrap.command_ids
            )
            if previous not in state.bootstrap.command_ids:
                raise InvalidOutcomeError(
                    "Base retry previous Attempt is not current"
                )
            return state.next_revision(
                **self._finish_automatic_recovery(
                    state, outcome.command.attempt_index
                ),
                bootstrap=replace(
                    state.bootstrap,
                    command_ids=current_ids,
                    command_status={
                        **state.bootstrap.command_status,
                        command.command_id: "submitted",
                    },
                    command_logical_ids={
                        **state.bootstrap.command_logical_ids,
                        command.command_id: command.logical_command_id,
                    },
                    command_attempt_ids={
                        **state.bootstrap.command_attempt_ids,
                        command.command_id: command.attempt_id,
                    },
                    command_attempt_indices={
                        **state.bootstrap.command_attempt_indices,
                        command.command_id: command.attempt_index,
                    },
                ),
                active_engine_commands=(
                    *state.active_engine_commands,
                    command,
                ),
            )
        if isinstance(outcome, BootstrapBaseTerminalOutcome):
            self._require_bootstrap_status(
                state, BootstrapStatus.BASE_EVALUATING
            )
            if set(outcome.command_status) != set(state.bootstrap.command_ids):
                raise InvalidOutcomeError(
                    "Base terminal status does not cover submitted Commands"
                )
            evidence = outcome.evidence_ref
            if (
                outcome.record.offline_evidence_ref_id
                != (evidence.artifact_id if evidence is not None else None)
            ):
                raise InvalidOutcomeError(
                    "Base evaluation evidence identity is inconsistent"
                )
            entries = state.ranking.entries
            if outcome.record.offline_score is not None and evidence is not None:
                entries = tuple(
                    item
                    for item in entries
                    if not (
                        item.level == "reference"
                        and item.subject_id
                        == f"{state.run_id}/c000/p000/base"
                    )
                ) + (
                    RankingEntry(
                        subject_id=f"{state.run_id}/c000/p000/base",
                        score=outcome.record.offline_score,
                        level="reference",
                        secondary_score=outcome.record.offline_secondary_score,
                        accepted_revision=state.revision + 1,
                        artifact_ref_id=evidence.artifact_id,
                        artifact_digest=evidence.digest,
                    ),
                )
            return state.next_revision(
                bootstrap=replace(
                    state.bootstrap,
                    status=BootstrapStatus.BASE_COMPLETED,
                    command_status={
                        **state.bootstrap.command_status,
                        **outcome.command_status,
                    },
                    base_evaluation=outcome.record,
                    metadata={
                        "base_receipt_accepted": True,
                        "base_evaluation_status": outcome.record.status.value,
                    },
                ),
                accepted_evidence_refs=(
                    *state.accepted_evidence_refs,
                    *((evidence,) if evidence is not None else ()),
                ),
                active_engine_commands=tuple(
                    command
                    for command in state.active_engine_commands
                    if command.command_id not in outcome.command_status
                ),
                ranking=replace(
                    state.ranking,
                    entries=entries,
                    revision=state.revision + 1,
                ),
            )
        if isinstance(outcome, BootstrapP000RegisteredOutcome):
            self._require_bootstrap_status(state, BootstrapStatus.BASE_COMPLETED)
            registered = self._register_baseline_trial(
                state,
                coordinator_id=outcome.coordinator_id,
                plan_id=outcome.plan_id,
                trial_id=outcome.trial_id,
                artifact_ref=outcome.artifact_ref,
            )
            return replace(
                registered,
                bootstrap=replace(
                    state.bootstrap,
                    status=BootstrapStatus.P000_TRAINING,
                    metadata={
                        **state.bootstrap.metadata,
                        "p000_artifact_digest": outcome.artifact_ref.digest,
                        "p000_artifact_kind": outcome.artifact_ref.kind,
                        "p000_artifact_path": outcome.artifact_path,
                    },
                ),
            )
        if isinstance(outcome, BootstrapStageAdvancedOutcome):
            self._require_bootstrap_status(state, outcome.from_status)
            allowed = {
                BootstrapStatus.P000_TRAINING: BootstrapStatus.P000_ANALYZING,
                BootstrapStatus.P000_ANALYZING: BootstrapStatus.P000_PLAN_SUMMARIZING,
                BootstrapStatus.P000_PLAN_SUMMARIZING: BootstrapStatus.P000_RUN_SUMMARIZING,
            }
            if allowed.get(outcome.from_status) is not outcome.to_status:
                raise InvalidOutcomeError("invalid Bootstrap stage transition")
            return state.next_revision(
                bootstrap=replace(state.bootstrap, status=outcome.to_status)
            )
        if isinstance(outcome, BootstrapCompletedOutcome):
            self._require_bootstrap_status(
                state, BootstrapStatus.P000_RUN_SUMMARIZING
            )
            trial = self._trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            )
            if (
                trial.kind is not TrialKind.BOOTSTRAP_BASELINE
                or trial.phase is not TrialPhase.ARCHIVED
            ):
                raise InvalidOutcomeError(
                    "Bootstrap cannot complete before P000 archive"
                )
            return state.next_revision(
                status=(
                    RunStatus.COMPLETED
                    if state.bootstrap.stop_after_baseline
                    else RunStatus.RUNNING
                ),
                completion_kind=(
                    CompletionKind.FULL_BUDGET
                    if state.bootstrap.stop_after_baseline
                    else None
                ),
                bootstrap=replace(
                    state.bootstrap,
                    status=BootstrapStatus.COMPLETED,
                    baseline_revision=state.revision + 1,
                    error=None,
                ),
            )
        if isinstance(outcome, BootstrapFailedOutcome):
            if not state.bootstrap.enabled or state.bootstrap.status in {
                BootstrapStatus.COMPLETED,
                BootstrapStatus.FAILED,
            }:
                raise InvalidOutcomeError("Bootstrap is not fail-able")
            continuation_status = (
                state.continuation_status
                if state.status is RunStatus.RECOVERING
                else state.status
            )
            return state.next_revision(
                status=RunStatus.SUSPENDED,
                continuation_status=continuation_status,
                recovery=None,
                bootstrap=replace(
                    state.bootstrap,
                    status=BootstrapStatus.FAILED,
                    command_status=dict(outcome.command_status),
                    error=outcome.reason,
                ),
                failure=FailureState(
                    "bootstrap_integrity_failure",
                    outcome.reason,
                    retryable=True,
                ),
            )
        if isinstance(outcome, RunResumedOutcome):
            completed_failed_engine_trial = any(
                trial.kind is TrialKind.SEARCH
                and trial.outcome is TrialOutcome.FAILED
                and trial.failure_kind == "engine_failed"
                and trial.artifact_ref_id is not None
                and trial.phase is TrialPhase.ARCHIVED
                and trial.archive_status is TrialArchiveStatus.ARCHIVED
                for trial in state.trials
            )
            if state.status not in {RunStatus.PAUSED, RunStatus.SUSPENDED} and not (
                state.pause_requested
                and state.status in {RunStatus.BOOTSTRAPPING, RunStatus.RUNNING}
            ) and not (
                state.status is RunStatus.COMPLETED
                and completed_failed_engine_trial
            ):
                raise InvalidOutcomeError(
                    f"cannot resume run in status {state.status.value}"
                )
            requested_recovery_max = (
                int(outcome.automatic_recovery_max_attempts)
                if outcome.automatic_recovery_max_attempts is not None
                else state.automatic_recovery.max_attempts
            )
            gpu_memory_attempts = [
                trial.engine_attempt_index
                for trial in state.trials
                if trial.engine_retry_pending
                and trial.engine_attempt_index > 0
            ]
            if (
                state.status is RunStatus.SUSPENDED
                and state.failure is not None
                and state.failure.code
                == "automatic_recovery_exhausted:gpu_memory_unavailable"
                and gpu_memory_attempts
                and requested_recovery_max <= max(gpu_memory_attempts)
            ):
                raise InvalidOutcomeError(
                    "GPU-memory recovery resume requires a larger "
                    "automatic_recovery.max_attempts budget"
                )
            resumed = state.next_revision(
                status=(state.continuation_status or RunStatus.RUNNING),
                pause_requested=False,
                pause_reason=None,
                continuation_status=None,
                failure=None,
                completion_kind=None,
                automatic_recovery=(
                    replace(
                        state.automatic_recovery,
                        max_attempts=max(
                            state.automatic_recovery.max_attempts,
                            outcome.automatic_recovery_max_attempts,
                        ),
                    )
                    if outcome.automatic_recovery_max_attempts is not None
                    else state.automatic_recovery
                ),
            )
            resumed = self._rewind_failed_bootstrap(state, resumed)
            resumed = self._rewind_failed_search_engine_trial(state, resumed)
            return self._rewind_suspended_agent_work(state, resumed)
        if isinstance(outcome, LocalJudgeReplacedOutcome):
            if state.run_resources is None:
                raise InvalidOutcomeError(
                    "cannot replace Local Judge for a CPU-only Run"
                )
            if not outcome.launch_id or not outcome.state_path:
                raise InvalidOutcomeError(
                    "Local Judge replacement identity is incomplete"
                )
            local_judge = dict(state.run_resources["local_judge"])
            if outcome.state_path != local_judge.get("state_path"):
                raise InvalidOutcomeError(
                    "Local Judge replacement changed the stable state binding"
                )
            if outcome.service_state != "ready":
                raise InvalidOutcomeError(
                    "Local Judge replacement is not ready"
                )
            local_judge.update(
                {
                    "launch_id": outcome.launch_id,
                    "host": outcome.host,
                    "gateway_url": outcome.gateway_url,
                    "node_id": outcome.node_id,
                    "state_path": outcome.state_path,
                    "service_state": outcome.service_state,
                    "readiness": outcome.service_state == "ready",
                }
            )
            return state.next_revision(
                run_resources={
                    **state.run_resources,
                    "local_judge": local_judge,
                }
            )
        if isinstance(outcome, RunPauseRequestedOutcome):
            if state.status not in {
                RunStatus.RUNNING,
                RunStatus.BOOTSTRAPPING,
                RunStatus.RECOVERING,
            }:
                raise InvalidOutcomeError(
                    f"cannot request pause in status {state.status.value}"
                )
            if state.pause_requested:
                raise InvalidOutcomeError("Run pause is already requested")
            reason = outcome.reason.strip()
            if not reason:
                raise InvalidOutcomeError("operator pause requires a reason")
            return state.next_revision(
                pause_requested=True,
                pause_reason=reason,
                continuation_status=(
                    state.continuation_status
                    if state.status is RunStatus.RECOVERING
                    else state.status
                ),
            )
        if isinstance(outcome, CoordinatorFinishRequestedOutcome):
            return self._request_coordinator_finish(state, outcome)
        if isinstance(outcome, CoordinatorCancelRequestedOutcome):
            return self._request_coordinator_cancel(state, outcome)
        if isinstance(outcome, RunFinishRequestedOutcome):
            return self._request_run_finish(state, outcome)
        if isinstance(outcome, CoordinatorFinishedEarlyOutcome):
            return self._finish_coordinator(
                state,
                outcome.coordinator_id,
                expected=CoordinatorControlStatus.FINISH_REQUESTED,
                terminal=CoordinatorControlStatus.FINISHED_EARLY,
            )
        if isinstance(outcome, CoordinatorCancelledOutcome):
            return self._finish_coordinator(
                state,
                outcome.coordinator_id,
                expected=CoordinatorControlStatus.CANCEL_REQUESTED,
                terminal=CoordinatorControlStatus.CANCELLED,
            )
        if isinstance(outcome, CoordinatorPlanningCancelledOutcome):
            return self._cancel_coordinator_planning(state, outcome)
        if isinstance(outcome, CoordinatorTrialCancelledOutcome):
            return self._cancel_coordinator_trial(state, outcome)
        if isinstance(outcome, CoordinatorPlanCancelledOutcome):
            return self._cancel_coordinator_plan(state, outcome)
        if isinstance(outcome, RunCancelledOutcome):
            return self.cancel(state, reason=outcome.reason)
        if isinstance(outcome, OperatorEvaluationScheduledOutcome):
            record = outcome.record
            if any(
                item.target_id == record.target_id
                for item in state.operator_evaluations
            ):
                raise InvalidOutcomeError("duplicate operator evaluation target")
            active = state.active_engine_commands
            if record.status is OperatorEvaluationStatus.PENDING:
                active = (*active, self._operator_command_ref(record))
            entries = state.ranking.entries
            if record.status in {
                OperatorEvaluationStatus.COMPLETED,
                OperatorEvaluationStatus.FAILED,
                OperatorEvaluationStatus.CANCELLED,
            }:
                entries = tuple(
                    replace(
                        entry,
                        operator_status=record.status.value,
                        operator_result_ref=record.result_ref,
                    )
                    if self._operator_ranking_match(entry, record)
                    else entry
                    for entry in entries
                )
            return state.next_revision(
                operator_evaluations=(*state.operator_evaluations, record),
                active_engine_commands=active,
                ranking=replace(
                    state.ranking,
                    entries=entries,
                    revision=state.revision,
                ),
            )
        if isinstance(outcome, OperatorEvaluationAttemptRetryPendingOutcome):
            matches = tuple(
                item
                for item in state.operator_evaluations
                if item.target_id == outcome.target_id
            )
            if len(matches) != 1:
                raise InvalidOutcomeError("unknown operator evaluation target")
            record = matches[0]
            active_matches = tuple(
                item
                for item in state.active_engine_commands
                if item.command_id == outcome.command_id
            )
            if (
                record.status is not OperatorEvaluationStatus.PENDING
                or record.retry_pending
                or len(active_matches) != 1
                or record.command_id != outcome.command_id
                or record.logical_command_id != outcome.logical_command_id
                or record.attempt_id != outcome.attempt_id
                or record.attempt_index != outcome.attempt_index
                or active_matches[0].logical_command_id
                != outcome.logical_command_id
                or active_matches[0].attempt_id != outcome.attempt_id
                or active_matches[0].attempt_index != outcome.attempt_index
            ):
                raise InvalidOutcomeError(
                    "operator retry-pending Receipt crossed the active Attempt fence"
                )
            pending = replace(
                record,
                command_id=None,
                attempt_id=outcome.attempt_id,
                retry_pending=True,
                failed_attempt_receipt_ids=(
                    *record.failed_attempt_receipt_ids,
                    outcome.receipt_id,
                ),
                error=outcome.message,
            )
            operator_pause = state.pause_requested
            recovery_changes = self._retryable_attempt_state(
                state,
                subject_ref=subject_ref(
                    state.run_id,
                    record.coordinator_id,
                    record.plan_id,
                    record.trial_id,
                ),
                logical_work_ref=outcome.logical_command_id,
                attempt_id=outcome.attempt_id,
                attempt_index=outcome.attempt_index,
                failure_kind=outcome.failure_kind,
                message=outcome.message,
                receipt_id=outcome.receipt_id,
                lease_ref=(
                    f"{state.run_id}/{record.coordinator_id or 'c000'}/"
                    f"operator_test/{outcome.logical_command_id}"
                ),
                observed_at=outcome.observed_at,
                operator_pause=operator_pause,
            )
            return state.next_revision(
                **recovery_changes,
                operator_evaluations=tuple(
                    pending if item.target_id == pending.target_id else item
                    for item in state.operator_evaluations
                ),
                active_engine_commands=tuple(
                    item
                    for item in state.active_engine_commands
                    if item.command_id != outcome.command_id
                ),
            )
        if isinstance(outcome, OperatorEvaluationAttemptSubmittedOutcome):
            matches = tuple(
                item
                for item in state.operator_evaluations
                if item.target_id == outcome.target_id
            )
            if len(matches) != 1:
                raise InvalidOutcomeError("unknown operator evaluation target")
            previous = matches[0]
            command = outcome.command
            if (
                previous.status is not OperatorEvaluationStatus.PENDING
                or not previous.retry_pending
                or previous.attempt_index != outcome.previous_attempt_index
                or command.logical_command_id != previous.logical_command_id
                or command.attempt_index != previous.attempt_index + 1
                or command.attempt_id
                != f"attempt-{command.attempt_index:03d}"
                or any(
                    item.command_id == command.command_id
                    for item in state.active_engine_commands
                )
            ):
                raise InvalidOutcomeError(
                    "operator retry does not create the next fenced Attempt"
                )
            submitted = replace(
                previous,
                command_id=command.command_id,
                attempt_id=command.attempt_id,
                attempt_index=command.attempt_index,
                retry_pending=False,
                error=None,
            )
            return state.next_revision(
                **self._finish_automatic_recovery(
                    state, outcome.command.attempt_index
                ),
                operator_evaluations=tuple(
                    submitted if item.target_id == submitted.target_id else item
                    for item in state.operator_evaluations
                ),
                active_engine_commands=(
                    *state.active_engine_commands,
                    command,
                ),
            )
        if isinstance(outcome, OperatorEvaluationTerminalOutcome):
            matches = tuple(
                item
                for item in state.operator_evaluations
                if item.target_id == outcome.target_id
            )
            if (
                len(matches) != 1
                or matches[0].command_id != outcome.command_id
                or matches[0].logical_command_id != outcome.logical_command_id
                or matches[0].attempt_id != outcome.attempt_id
                or matches[0].attempt_index != outcome.attempt_index
            ):
                raise InvalidOutcomeError("unknown operator evaluation target")
            active_matches = tuple(
                item
                for item in state.active_engine_commands
                if item.command_id == outcome.command_id
            )
            if (
                len(active_matches) != 1
                or active_matches[0].logical_command_id
                != outcome.logical_command_id
                or active_matches[0].attempt_id != outcome.attempt_id
                or active_matches[0].attempt_index != outcome.attempt_index
            ):
                raise InvalidOutcomeError(
                    "operator terminal Receipt crossed the active Attempt fence"
                )
            if matches[0].status is not OperatorEvaluationStatus.PENDING:
                raise InvalidOutcomeError("operator evaluation is already terminal")
            terminal = replace(
                matches[0],
                status=outcome.status,
                retry_pending=False,
                receipt_id=outcome.receipt_id,
                result_ref=outcome.result_ref,
                error=outcome.error,
            )
            entries = tuple(
                replace(
                    entry,
                    operator_status=outcome.status.value,
                    operator_result_ref=outcome.result_ref,
                )
                if self._operator_ranking_match(entry, terminal)
                else entry
                for entry in state.ranking.entries
            )
            return state.next_revision(
                operator_evaluations=tuple(
                    terminal if item.target_id == terminal.target_id else item
                    for item in state.operator_evaluations
                ),
                active_engine_commands=tuple(
                    item
                    for item in state.active_engine_commands
                    if item.command_id != outcome.command_id
                ),
                ranking=replace(
                    state.ranking,
                    entries=entries,
                    revision=state.revision,
                ),
            )
        if isinstance(outcome, PlanningDecisionOutcome):
            return self._apply_planning(state, outcome)
        if isinstance(outcome, PlanCatalogUpdatedOutcome):
            return self._apply_plan_catalog_update(state, outcome)
        if isinstance(outcome, PlanningSlotReservedOutcome):
            coordinator = self._search_coordinator(state, outcome.coordinator_id)
            if coordinator.control_status not in {
                CoordinatorControlStatus.ACTIVE,
                CoordinatorControlStatus.FINISH_REQUESTED,
            }:
                raise InvalidOutcomeError("Coordinator is not accepting reservations")
            if (
                allocated_plan_slots(state, outcome.coordinator_id)
                >= coordinator.effective_plan_limit
            ):
                raise InvalidOutcomeError("Coordinator effective Plan capacity exhausted")
            expected_sequence = max(
                (
                    *(
                        item.sequence for item in state.planning_queue
                    ),
                    *(
                        plan.reservation_sequence
                        for plan in state.plans
                        if plan.reservation_sequence is not None
                    ),
                ),
                default=-1,
            ) + 1
            if outcome.sequence != expected_sequence:
                raise InvalidOutcomeError("planning reservation sequence is not FIFO")
            if any(
                item.coordinator_id == outcome.coordinator_id
                for item in state.planning_queue
            ):
                raise InvalidOutcomeError("Coordinator already has a planning reservation")
            return state.next_revision(
                planning_queue=(
                    *state.planning_queue,
                    PlanningSlot(
                        reservation_id=outcome.reservation_id,
                        coordinator_id=outcome.coordinator_id,
                        plan_id=outcome.plan_id,
                        sequence=outcome.sequence,
                    ),
                )
            )
        if isinstance(outcome, TrialProposedOutcome):
            matching_plans = tuple(
                plan
                for plan in state.plans
                if plan.coordinator_id == outcome.coordinator_id
                and plan.plan_id == outcome.plan_id
                and plan.status is PlanStatus.ACTIVE
            )
            if len(matching_plans) != 1:
                raise InvalidOutcomeError(
                    "unknown active plan: "
                    f"{outcome.coordinator_id}/{outcome.plan_id}"
                )
            if any(
                trial.coordinator_id == outcome.coordinator_id
                and trial.plan_id == outcome.plan_id
                and trial.trial_id == outcome.trial_id
                for trial in state.trials
            ):
                raise InvalidOutcomeError("duplicate Trial scope")
            if self._search_trial_count(state) >= state.portfolio.max_trials:
                raise InvalidOutcomeError("trial capacity exhausted")
            plan = matching_plans[0]
            previous_trials = tuple(
                trial
                for trial in state.trials
                if trial.coordinator_id == outcome.coordinator_id
                and trial.plan_id == outcome.plan_id
            )
            if previous_trials:
                previous = previous_trials[-1]
                if previous.archive_status is not TrialArchiveStatus.ARCHIVED:
                    raise InvalidOutcomeError(
                        "next Trial requires the previous archived Trial"
                    )
                if previous.artifact_ref_id is not None:
                    source_artifact_ref_ids = (previous.artifact_ref_id,)
                else:
                    if plan.relation is None:
                        raise InvalidOutcomeError(
                            "Trial fallback requires a frozen Plan relation"
                        )
                    source_artifact_ref_ids = plan.relation.seed_artifact_ref_ids
            else:
                if plan.relation is None:
                    raise InvalidOutcomeError("first Trial requires a frozen Plan relation")
                source_artifact_ref_ids = plan.relation.seed_artifact_ref_ids
            return state.next_revision(
                trials=(
                    *state.trials,
                    TrialState(
                        trial_id=outcome.trial_id,
                        coordinator_id=outcome.coordinator_id,
                        plan_id=outcome.plan_id,
                        source_artifact_ref_ids=source_artifact_ref_ids,
                        trial_record_ref=(
                            f"{state.run_id}/{outcome.coordinator_id}/"
                            f"{outcome.plan_id}/{outcome.trial_id}/record"
                        ),
                        plan_memory_basis=plan.plan_memory_head,
                        run_memory_basis=state.memory.run_head,
                    ),
                )
            )
        if isinstance(outcome, CoordinatorCallSubmittedOutcome):
            call = outcome.call_ref
            if call.role != AgentRole.COORDINATOR.value:
                raise InvalidOutcomeError("Coordinator submission role mismatch")
            if call.coordinator_id != outcome.coordinator_id:
                raise InvalidOutcomeError("Coordinator submission scope mismatch")
            if call.target_subject_ref != subject_ref(
                state.run_id,
                outcome.coordinator_id,
                outcome.plan_id,
            ):
                raise InvalidOutcomeError(
                    "Coordinator Call target does not match reserved Plan SubjectRef"
                )
            if any(item.call_id == call.call_id for item in state.active_agent_calls):
                raise InvalidOutcomeError("duplicate active Agent Call")
            if not state.planning_queue:
                raise InvalidOutcomeError("Coordinator Call requires a planning reservation")
            reservation = state.planning_queue[0]
            if (
                reservation.status != "queued"
                or reservation.coordinator_id != outcome.coordinator_id
                or reservation.plan_id != outcome.plan_id
            ):
                raise InvalidOutcomeError("Coordinator Call is not the FIFO queue head")
            existing_session = next(
                (
                    item for item in state.agent_sessions
                    if item.session_id == call.session_id
                ),
                None,
            )
            session = AgentSession(
                session_id=call.session_id,
                run_id=state.run_id,
                role=AgentRole.COORDINATOR,
                subject_id=outcome.coordinator_id,
                coordinator_id=outcome.coordinator_id,
                resume_handle=call.resume_handle,
                call_ids=(
                    *((existing_session.call_ids) if existing_session else ()),
                    call.call_id,
                ),
            )
            return self._record_agent_session(
                state.next_revision(
                    active_agent_calls=(*state.active_agent_calls, call),
                    planning_queue=(
                        replace(reservation, status="planning"),
                        *state.planning_queue[1:],
                    ),
                ),
                session,
            )
        if isinstance(outcome, AgentRetryHeldOutcome):
            matches = tuple(
                item
                for item in state.active_agent_calls
                if item.call_id == outcome.call_id
            )
            if len(matches) != 1:
                raise InvalidOutcomeError(
                    "held Agent retry requires one active Call"
                )
            previous = matches[0]
            if (
                not state.pause_requested
                or previous.attempt_id != outcome.previous_attempt_id
                or previous.status == "retry_pending"
                or previous.retry_index >= previous.max_retries
            ):
                raise InvalidOutcomeError(
                    "held Agent retry does not preserve the pause fence"
                )
            return state.next_revision(
                active_agent_calls=tuple(
                    replace(item, status="retry_pending")
                    if item.call_id == outcome.call_id
                    else item
                    for item in state.active_agent_calls
                )
            )
        if isinstance(outcome, AgentRetrySubmittedOutcome):
            call = outcome.call_ref
            matches = tuple(
                item for item in state.active_agent_calls
                if item.call_id == call.call_id
            )
            if len(matches) != 1:
                raise InvalidOutcomeError("Agent retry requires one active Call")
            previous = matches[0]
            if (
                previous.attempt_id != outcome.previous_attempt_id
                or previous.status not in {"submitted", "running", "retry_pending"}
                or call.retry_index != previous.retry_index + 1
                or call.retry_index > call.max_retries
                or call.session_id != previous.session_id
                or call.basis_revision != previous.basis_revision
                or call.memory_view_id != previous.memory_view_id
                or call.reflection_index != previous.reflection_index
                or call.max_reflections != previous.max_reflections
            ):
                raise InvalidOutcomeError("Agent retry does not preserve Call fence")
            return state.next_revision(
                active_agent_calls=tuple(
                    call if item.call_id == call.call_id else item
                    for item in state.active_agent_calls
                )
            )
        if isinstance(outcome, BuilderProposalCompletedOutcome):
            return self._complete_builder_delivery(
                state,
                call_id=outcome.call_id,
                attempt_id=outcome.attempt_id,
                reflection_index=0,
                delivery_ref=outcome.delivery_ref,
                expected_event="proposal",
            )
        if isinstance(outcome, BuilderReflectionCompletedOutcome):
            return self._complete_builder_delivery(
                state,
                call_id=outcome.call_id,
                attempt_id=outcome.attempt_id,
                reflection_index=outcome.reflection_index,
                delivery_ref=outcome.delivery_ref,
                expected_event="reflection",
            )
        if isinstance(outcome, BuilderRealizationStartedOutcome):
            active = self._builder_round_call(
                state, outcome.call_id, outcome.reflection_index
            )
            if (
                active.round_status != "delivery_ready"
                or active.current_delivery_ref != outcome.delivery_ref
                or active.current_realization_ref is not None
            ):
                raise InvalidOutcomeError(
                    "Builder realization start does not match ready delivery"
                )
            return state.next_revision(
                active_agent_calls=tuple(
                    replace(item, round_status="realization_running")
                    if item.call_id == outcome.call_id
                    else item
                    for item in state.active_agent_calls
                )
            )
        if isinstance(outcome, BuilderRealizationCompletedOutcome):
            active = self._builder_round_call(
                state, outcome.call_id, outcome.reflection_index
            )
            if (
                active.round_status != "realization_running"
                or active.current_delivery_ref != outcome.delivery_ref
                or active.current_realization_ref is not None
            ):
                raise InvalidOutcomeError(
                    "Builder realization completion does not match running round"
                )
            return state.next_revision(
                active_agent_calls=tuple(
                    replace(
                        item,
                        round_status="realization_ready",
                        current_realization_ref=outcome.realization_ref,
                    )
                    if item.call_id == outcome.call_id
                    else item
                    for item in state.active_agent_calls
                )
            )
        if isinstance(outcome, BuilderReflectionSubmittedOutcome):
            call = outcome.call_ref
            previous = self._builder_round_call(
                state, call.call_id, call.reflection_index - 1
            )
            if (
                previous.round_status != "realization_ready"
                or previous.current_delivery_ref != outcome.previous_delivery_ref
                or previous.current_realization_ref
                != outcome.previous_realization_ref
                or call.reflection_index != previous.reflection_index + 1
                or call.reflection_index > previous.max_reflections
                or call.retry_index != 0
                or call.round_status != "agent_running"
                or call.current_delivery_ref is not None
                or call.current_realization_ref is not None
                or call.session_id != previous.session_id
                or call.basis_revision != previous.basis_revision
                or call.memory_view_id != previous.memory_view_id
                or call.target_subject_ref != previous.target_subject_ref
            ):
                raise InvalidOutcomeError(
                    "Builder reflection does not preserve the completed round fence"
                )
            return state.next_revision(
                active_agent_calls=tuple(
                    call if item.call_id == call.call_id else item
                    for item in state.active_agent_calls
                )
            )
        if isinstance(outcome, BuilderRealizationFinalizedOutcome):
            active = self._builder_round_call(
                state, outcome.call_id, outcome.reflection_index
            )
            if (
                active.round_status != "realization_ready"
                or active.current_delivery_ref != outcome.delivery_ref
                or active.current_realization_ref != outcome.realization_ref
            ):
                raise InvalidOutcomeError(
                    "Builder finalization does not match ready realization"
                )
            return state.next_revision(
                active_agent_calls=tuple(
                    replace(item, round_status="finalized")
                    if item.call_id == outcome.call_id
                    else item
                    for item in state.active_agent_calls
                )
            )
        if isinstance(outcome, CoordinatorFailedOutcome):
            if not state.planning_queue:
                raise InvalidOutcomeError("failed Coordinator has no reservation")
            reservation = state.planning_queue[0]
            if (
                reservation.coordinator_id != outcome.coordinator_id
                or reservation.plan_id != outcome.plan_id
            ):
                raise InvalidOutcomeError("failed Coordinator reservation mismatch")
            if any(
                plan.coordinator_id == outcome.coordinator_id
                and plan.plan_id == outcome.plan_id
                for plan in state.plans
            ):
                raise InvalidOutcomeError("duplicate failed Plan slot")
            failed = state.next_revision(
                plans=(
                    *state.plans,
                    PlanState(
                        plan_id=outcome.plan_id,
                        coordinator_id=outcome.coordinator_id,
                        kind=PlanKind.SEARCH,
                        status=PlanStatus.FAILED,
                        basis_revision=state.revision,
                        reservation_id=reservation.reservation_id,
                        reservation_sequence=reservation.sequence,
                    ),
                ),
                accepted_finding_refs=(
                    *state.accepted_finding_refs,
                    outcome.failure_ref,
                ),
                planning_queue=state.planning_queue[1:],
            )
            return self._finish_agent_call(
                self._record_agent_session(failed, outcome.agent_session),
                outcome.call_id,
            )
        if isinstance(outcome, AgentCallSubmittedOutcome):
            if outcome.call_ref.target_subject_ref != subject_ref(
                state.run_id,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            ):
                raise InvalidOutcomeError(
                    "Agent Call target does not match Trial SubjectRef"
                )
            transitions = {
                "artifact_builder": (
                    TrialPhase.CREATED,
                    TrialPhase.BUILDING_ARTIFACT,
                ),
                "plan_summarizer": (
                    TrialPhase.ANALYSIS_READY,
                    TrialPhase.PLAN_SUMMARIZING,
                ),
                "run_summarizer": (
                    TrialPhase.PLAN_SUMMARY_READY,
                    TrialPhase.RUN_SUMMARIZING,
                ),
            }
            if outcome.role == "analyzer":
                stage = outcome.call_ref.action_fields.get("stage")
                transitions["analyzer"] = {
                    "review_design": (
                        TrialPhase.EVIDENCE_READY,
                        TrialPhase.ANALYSIS_DESIGNING,
                    ),
                    "synthesis": (
                        TrialPhase.REVIEW_READY,
                        TrialPhase.ANALYZING,
                    ),
                }.get(stage)
                if transitions["analyzer"] is None:
                    raise InvalidOutcomeError("Analyzer Call requires a valid stage")
            try:
                current_phase, next_phase = transitions[outcome.role]
            except KeyError as error:
                raise InvalidOutcomeError(
                    f"unsupported Trial Agent role: {outcome.role}"
                ) from error
            if any(
                item.call_id == outcome.call_ref.call_id
                for item in state.active_agent_calls
            ):
                raise InvalidOutcomeError("duplicate active Agent Call")
            updated = self._replace_trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
                allowed={current_phase},
                phase=next_phase,
                **(
                    {"run_memory_basis": state.memory.run_head}
                    if outcome.role == "run_summarizer"
                    else {}
                ),
            )
            updated = replace(
                updated,
                active_agent_calls=(
                    *updated.active_agent_calls,
                    outcome.call_ref,
                ),
            )
            role = AgentRole(outcome.role)
            session_subject = outcome.call_ref.subject_id
            session_trial_id = outcome.call_ref.trial_id
            if role in {AgentRole.ARTIFACT_BUILDER, AgentRole.ANALYZER}:
                session_subject = outcome.call_ref.plan_id
                session_trial_id = None
            existing_session = next(
                (
                    item for item in state.agent_sessions
                    if item.session_id == outcome.call_ref.session_id
                ),
                None,
            )
            session = AgentSession(
                session_id=outcome.call_ref.session_id,
                run_id=state.run_id,
                role=role,
                subject_id=session_subject,
                coordinator_id=outcome.call_ref.coordinator_id,
                plan_id=outcome.call_ref.plan_id,
                trial_id=session_trial_id,
                resume_handle=outcome.call_ref.resume_handle,
                call_ids=(
                    *((existing_session.call_ids) if existing_session else ()),
                    outcome.call_ref.call_id,
                ),
            )
            return self._record_agent_session(updated, session)
        if isinstance(outcome, AnalysisReviewSubmittedOutcome):
            command_ref = (
                replace(outcome.command_ref, status="pending_submit")
                if state.pause_requested
                else outcome.command_ref
            )
            updated = self._replace_trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
                allowed={TrialPhase.ANALYSIS_DESIGNING},
                phase=TrialPhase.REVIEW_RUNNING,
                analysis_design_ref_id=outcome.design_ref.artifact_id,
                analysis_review_command_id=command_ref.command_id,
                analysis_review_logical_command_id=(
                    command_ref.logical_command_id
                ),
                analysis_review_attempt_id=command_ref.attempt_id,
                analysis_review_attempt_index=command_ref.attempt_index,
                analysis_review_retry_pending=False,
            )
            accepted = replace(
                updated,
                accepted_evidence_refs=(
                    *updated.accepted_evidence_refs,
                    outcome.design_ref,
                ),
                active_review_commands=(
                    *updated.active_review_commands,
                    command_ref,
                ),
            )
            return self._finish_agent_call(
                self._record_agent_session(accepted, outcome.agent_session),
                outcome.call_id,
            )
        if isinstance(outcome, AnalysisReviewAttemptRetryPendingOutcome):
            trial = self._trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            )
            active = tuple(
                item
                for item in state.active_review_commands
                if item.command_id == outcome.command_id
            )
            if (
                trial.phase is not TrialPhase.REVIEW_RUNNING
                or trial.analysis_review_command_id != outcome.command_id
                or trial.analysis_review_logical_command_id
                != outcome.logical_command_id
                or trial.analysis_review_attempt_id != outcome.attempt_id
                or trial.analysis_review_attempt_index != outcome.attempt_index
                or len(active) != 1
                or active[0].logical_command_id != outcome.logical_command_id
                or active[0].attempt_id != outcome.attempt_id
                or active[0].attempt_index != outcome.attempt_index
            ):
                raise InvalidOutcomeError(
                    "retry-pending Receipt is not the current Review Attempt"
                )
            updated = self._replace_trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
                allowed={TrialPhase.REVIEW_RUNNING},
                analysis_review_command_id=None,
                analysis_review_retry_pending=True,
                analysis_review_attempt_failure_ref_ids=(
                    *trial.analysis_review_attempt_failure_ref_ids,
                    outcome.failure_ref.artifact_id,
                ),
            )
            recovery_changes = self._retryable_attempt_state(
                state,
                subject_ref=subject_ref(
                    state.run_id,
                    outcome.coordinator_id,
                    outcome.plan_id,
                    outcome.trial_id,
                ),
                logical_work_ref=outcome.logical_command_id,
                attempt_id=outcome.attempt_id,
                attempt_index=outcome.attempt_index,
                failure_kind=outcome.failure_kind,
                message=outcome.message,
                receipt_id=outcome.receipt_id,
                lease_ref=(
                    f"review:{state.run_id}/{outcome.coordinator_id}/"
                    f"{outcome.plan_id}/{outcome.trial_id}"
                ),
                observed_at=outcome.observed_at,
                operator_pause=False,
            )
            return replace(
                updated,
                **recovery_changes,
                active_review_commands=tuple(
                    item
                    for item in updated.active_review_commands
                    if item.command_id != outcome.command_id
                ),
            )
        if isinstance(outcome, AnalysisReviewAttemptSubmittedOutcome):
            trial = self._trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            )
            command = outcome.command_ref
            if (
                trial.phase is not TrialPhase.REVIEW_RUNNING
                or not trial.analysis_review_retry_pending
                or trial.analysis_review_command_id is not None
                or trial.analysis_review_logical_command_id
                != command.logical_command_id
                or trial.analysis_review_attempt_index
                != outcome.command_ref.attempt_index - 1
            ):
                raise InvalidOutcomeError("Review retry state is inconsistent")
            updated = self._replace_trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
                allowed={TrialPhase.REVIEW_RUNNING},
                analysis_review_command_id=command.command_id,
                analysis_review_attempt_id=command.attempt_id,
                analysis_review_attempt_index=command.attempt_index,
                analysis_review_retry_pending=False,
            )
            return replace(
                updated,
                **self._finish_automatic_recovery(
                    state, command.attempt_index
                ),
                active_review_commands=(
                    *updated.active_review_commands,
                    command,
                ),
            )
        if isinstance(outcome, AnalysisReviewCompletedOutcome):
            trial = self._trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            )
            active = tuple(
                item
                for item in state.active_review_commands
                if item.command_id == outcome.command_id
            )
            if (
                trial.analysis_review_command_id != outcome.command_id
                or len(active) != 1
            ):
                raise InvalidOutcomeError(
                    "Review Receipt is not the current active Attempt"
                )
            updated = self._replace_trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
                allowed={TrialPhase.REVIEW_RUNNING},
                phase=TrialPhase.REVIEW_READY,
                analysis_review_packet_ref_id=outcome.packet_ref.artifact_id,
                analysis_review_coverage_ref_id=outcome.coverage_ref.artifact_id,
            )
            return replace(
                updated,
                active_review_commands=tuple(
                    item
                    for item in updated.active_review_commands
                    if item.command_id != outcome.command_id
                ),
                accepted_evidence_refs=(
                    *updated.accepted_evidence_refs,
                    outcome.packet_ref,
                    outcome.coverage_ref,
                ),
            )
        if isinstance(outcome, BaselineTrialRegisteredOutcome):
            return self._register_baseline_trial(
                state,
                coordinator_id=outcome.coordinator_id,
                plan_id=outcome.plan_id,
                trial_id=outcome.trial_id,
                artifact_ref=outcome.artifact_ref,
            )
        if isinstance(outcome, ArtifactAcceptedOutcome):
            updated = self._replace_trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
                allowed={TrialPhase.BUILDING_ARTIFACT},
                phase=TrialPhase.ARTIFACT_READY,
                artifact_ref_id=outcome.artifact_ref.artifact_id,
                artifact_supporting_ref_ids=tuple(
                    ref.artifact_id for ref in outcome.supporting_refs
                ),
            )
            accepted = replace(
                updated,
                accepted_evidence_refs=(
                    *updated.accepted_evidence_refs,
                    outcome.artifact_ref,
                    *outcome.supporting_refs,
                ),
            )
            session = outcome.agent_session
            return self._finish_agent_call(
                self._record_agent_session(accepted, session), outcome.call_id
            )
        if isinstance(outcome, BuilderFailedOutcome):
            updated = self._replace_trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
                allowed={TrialPhase.BUILDING_ARTIFACT},
                phase=TrialPhase.ANALYSIS_READY,
                analysis_status="failed",
                analysis_failure_ref_id=outcome.failure_ref.artifact_id,
                objective_comparison_ref_id=(
                    outcome.objective_comparison_ref.artifact_id
                ),
                outcome=TrialOutcome.FAILED,
                archive_status=TrialArchiveStatus.PENDING,
                failure_kind="builder_validation_failed",
                analysis_packet_ref_id=outcome.snapshot_ref.snapshot_id,
            )
            accepted = replace(
                updated,
                accepted_finding_refs=(
                    *updated.accepted_finding_refs,
                    outcome.failure_ref,
                ),
                accepted_snapshot_refs=(
                    *updated.accepted_snapshot_refs,
                    outcome.snapshot_ref,
                ),
                accepted_evidence_refs=(
                    *updated.accepted_evidence_refs,
                    outcome.objective_comparison_ref,
                ),
            )
            recorded = self._record_trial_result(
                accepted,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            )
            session = outcome.agent_session
            return self._finish_agent_call(
                self._record_agent_session(recorded, session), outcome.call_id
            )
        if isinstance(outcome, EngineQueuedOutcome):
            trial = self._trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            )
            logical_command_id = (
                outcome.logical_command_id or outcome.command_id
            )
            retry = (
                trial.phase is TrialPhase.ENGINE_RUNNING
                and trial.engine_retry_pending
            )
            if retry:
                if trial.logical_command_id != logical_command_id:
                    raise InvalidOutcomeError(
                        "Engine retry changed logical Command identity"
                    )
                if outcome.attempt_index != trial.engine_attempt_index + 1:
                    raise InvalidOutcomeError(
                        "Engine retry Attempt index is not consecutive"
                    )
            elif outcome.attempt_index != 1:
                raise InvalidOutcomeError("initial Engine Attempt index must be one")
            updated = self._replace_trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
                allowed=(
                    {TrialPhase.ENGINE_RUNNING}
                    if retry
                    else {TrialPhase.ARTIFACT_READY}
                ),
                phase=TrialPhase.ENGINE_RUNNING,
                command_id=outcome.command_id,
                logical_command_id=logical_command_id,
                engine_attempt_id=outcome.attempt_id,
                engine_attempt_index=outcome.attempt_index,
                engine_retry_pending=False,
            )
            return replace(
                updated,
                **self._finish_automatic_recovery(state, outcome.attempt_index),
                active_engine_commands=(
                    *updated.active_engine_commands,
                    self._command_ref(outcome),
                ),
            )
        if isinstance(outcome, EngineAttemptRetryPendingOutcome):
            trial = self._trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            )
            active = tuple(
                item
                for item in state.active_engine_commands
                if item.command_id == outcome.command_id
            )
            if (
                trial.phase is not TrialPhase.ENGINE_RUNNING
                or trial.command_id != outcome.command_id
                or trial.logical_command_id != outcome.logical_command_id
                or trial.engine_attempt_id != outcome.attempt_id
                or trial.engine_attempt_index != outcome.attempt_index
                or len(active) != 1
            ):
                raise InvalidOutcomeError(
                    "retry-pending Receipt is not the current Engine Attempt"
                )
            updated = self._replace_trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
                allowed={TrialPhase.ENGINE_RUNNING},
                command_id=None,
                engine_retry_pending=True,
                engine_attempt_failure_ref_ids=(
                    *trial.engine_attempt_failure_ref_ids,
                    outcome.failure_ref.artifact_id,
                ),
            )
            operator_pause = (
                state.pause_requested
                and outcome.failure_kind == "operator_interrupted"
            )
            recovery_changes = self._retryable_attempt_state(
                state,
                subject_ref=subject_ref(
                    state.run_id,
                    outcome.coordinator_id,
                    outcome.plan_id,
                    outcome.trial_id,
                ),
                logical_work_ref=outcome.logical_command_id,
                attempt_id=outcome.attempt_id,
                attempt_index=outcome.attempt_index,
                failure_kind=outcome.failure_kind,
                message=outcome.message,
                receipt_id=outcome.receipt_id,
                lease_ref=(
                    f"{state.run_id}/{outcome.coordinator_id}/training"
                ),
                observed_at=outcome.observed_at,
                operator_pause=operator_pause,
            )
            return replace(
                updated,
                **recovery_changes,
                pause_requested=state.pause_requested,
                active_engine_commands=tuple(
                    item
                    for item in updated.active_engine_commands
                    if item.command_id != outcome.command_id
                ),
            )
        if isinstance(outcome, EngineCompletedOutcome):
            self._require_active_engine_attempt(state, outcome)
            updated = self._replace_trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
                allowed={TrialPhase.ENGINE_RUNNING},
                phase=TrialPhase.EVIDENCE_READY,
                result_refs=outcome.result_refs,
                package_ref_id=outcome.package_ref.artifact_id,
                authorized_unit_ids=outcome.authorized_unit_ids,
                outcome=TrialOutcome.SUCCEEDED,
            )
            accepted = replace(
                updated,
                active_engine_commands=tuple(
                    item
                    for item in updated.active_engine_commands
                    if item.command_id != outcome.command_id
                ),
                accepted_evidence_refs=(
                    *updated.accepted_evidence_refs,
                    outcome.package_ref,
                ),
            )
            return self._advance_bootstrap_after_engine(state, accepted, outcome)
        if isinstance(outcome, EngineFailedOutcome):
            self._require_active_engine_attempt(state, outcome)
            if outcome.package_ref is not None:
                updated = self._replace_trial(
                    state,
                    outcome.coordinator_id,
                    outcome.plan_id,
                    outcome.trial_id,
                    allowed={TrialPhase.ENGINE_RUNNING},
                    phase=TrialPhase.EVIDENCE_READY,
                    result_refs=outcome.result_refs,
                    package_ref_id=outcome.package_ref.artifact_id,
                    authorized_unit_ids=outcome.authorized_unit_ids,
                    outcome=TrialOutcome.FAILED,
                    failure_kind=outcome.failure_kind,
                )
                accepted = replace(
                    updated,
                    active_engine_commands=tuple(
                        item for item in updated.active_engine_commands
                        if item.command_id != outcome.command_id
                    ),
                    accepted_evidence_refs=(
                        *updated.accepted_evidence_refs,
                        outcome.package_ref,
                    ),
                    accepted_finding_refs=(
                        *updated.accepted_finding_refs,
                        outcome.failure_ref,
                    ),
                )
                return self._advance_bootstrap_after_engine(
                    state, accepted, outcome
                )
            if outcome.snapshot_ref is None:
                raise InvalidOutcomeError(
                    "Engine failure without evidence requires a default analysis snapshot"
                )
            updated = self._replace_trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
                allowed={TrialPhase.ENGINE_RUNNING},
                phase=TrialPhase.ANALYSIS_READY,
                analysis_status="defaulted",
                analysis_failure_ref_id=outcome.failure_ref.artifact_id,
                objective_comparison_ref_id=(
                    outcome.objective_comparison_ref.artifact_id
                    if outcome.objective_comparison_ref is not None
                    else None
                ),
                outcome=TrialOutcome.FAILED,
                failure_kind=outcome.failure_kind,
                analysis_packet_ref_id=outcome.snapshot_ref.snapshot_id,
            )
            accepted = replace(
                updated,
                active_engine_commands=tuple(
                    item for item in updated.active_engine_commands
                    if item.command_id != outcome.command_id
                ),
                accepted_finding_refs=(
                    *updated.accepted_finding_refs,
                    outcome.failure_ref,
                ),
                accepted_snapshot_refs=(
                    *updated.accepted_snapshot_refs,
                    outcome.snapshot_ref,
                ),
                accepted_evidence_refs=(
                    *updated.accepted_evidence_refs,
                    *(
                        (outcome.objective_comparison_ref,)
                        if outcome.objective_comparison_ref is not None
                        else ()
                    ),
                ),
            )
            recorded = self._record_trial_result(
                accepted,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            )
            return self._advance_bootstrap_after_engine(
                state, recorded, outcome
            )
        if isinstance(outcome, AnalysisAcceptedOutcome):
            trial = self._trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            )
            updated = self._replace_trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
                allowed={TrialPhase.ANALYZING},
                phase=TrialPhase.ANALYSIS_READY,
                analysis_status="accepted",
                analysis_ref_id=outcome.analysis_ref.artifact_id,
                objective_comparison_ref_id=(
                    outcome.objective_comparison_ref.artifact_id
                ),
                analysis_evidence_ref_id=outcome.evidence_ref.artifact_id,
                analysis_review_coverage_ref_id=outcome.review_coverage_ref.artifact_id,
                outcome=trial.outcome,
                archive_status=TrialArchiveStatus.PENDING,
                offline_score=outcome.offline_score,
                offline_secondary_score=outcome.offline_secondary_score,
                analysis_packet_ref_id=outcome.snapshot_ref.snapshot_id,
            )
            accepted = replace(
                updated,
                accepted_finding_refs=(
                    *updated.accepted_finding_refs,
                    outcome.analysis_ref,
                    outcome.findings_ref,
                    outcome.evidence_ref,
                    outcome.review_coverage_ref,
                ),
                accepted_evidence_refs=(
                    *updated.accepted_evidence_refs,
                    outcome.objective_comparison_ref,
                ),
                accepted_snapshot_refs=(
                    *updated.accepted_snapshot_refs,
                    outcome.snapshot_ref,
                ),
            )
            recorded = self._record_trial_result(
                accepted,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            )
            session = outcome.agent_session
            return self._finish_agent_call(
                self._record_agent_session(recorded, session), outcome.call_id
            )
        if isinstance(outcome, AnalysisFailedOutcome):
            trial = self._trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            )
            updated = self._replace_trial(
                state,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
                allowed={
                    TrialPhase.ANALYSIS_DESIGNING,
                    TrialPhase.REVIEW_RUNNING,
                    TrialPhase.REVIEW_READY,
                    TrialPhase.ANALYZING,
                    TrialPhase.EVIDENCE_READY,
                },
                phase=TrialPhase.ANALYSIS_READY,
                analysis_status="defaulted",
                analysis_failure_ref_id=outcome.failure_ref.artifact_id,
                objective_comparison_ref_id=(
                    outcome.objective_comparison_ref.artifact_id
                ),
                outcome=trial.outcome,
                archive_status=trial.archive_status,
                failure_kind=trial.failure_kind,
                offline_score=outcome.offline_score,
                offline_secondary_score=outcome.offline_secondary_score,
                analysis_packet_ref_id=outcome.snapshot_ref.snapshot_id,
            )
            accepted = replace(
                updated,
                active_review_commands=tuple(
                    item
                    for item in updated.active_review_commands
                    if not (
                        item.coordinator_id == outcome.coordinator_id
                        and item.plan_id == outcome.plan_id
                        and item.trial_id == outcome.trial_id
                        and (
                            trial.analysis_review_command_id is None
                            or item.command_id == trial.analysis_review_command_id
                        )
                    )
                ),
                accepted_finding_refs=(*updated.accepted_finding_refs, outcome.failure_ref),
                accepted_evidence_refs=(
                    *updated.accepted_evidence_refs,
                    outcome.objective_comparison_ref,
                ),
                accepted_snapshot_refs=(
                    *updated.accepted_snapshot_refs,
                    outcome.snapshot_ref,
                ),
            )
            recorded = self._record_trial_result(
                accepted,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            )
            session = outcome.agent_session
            return self._finish_agent_call(
                self._record_agent_session(recorded, session), outcome.call_id
            )
        if isinstance(outcome, SummaryAcceptedOutcome):
            if outcome.summary_kind == "plan_summary":
                plan = self._plan(state, outcome.coordinator_id, outcome.subject_id)
                if (
                    outcome.trial_coordinator_id != plan.coordinator_id
                    or outcome.trial_plan_id != plan.plan_id
                ):
                    raise InvalidOutcomeError("Plan summary Trial scope mismatch")
                if (
                    outcome.snapshot_ref.kind is not SnapshotKind.PLAN
                    or outcome.snapshot_ref.coordinator_id != plan.coordinator_id
                    or outcome.snapshot_ref.plan_id != plan.plan_id
                ):
                    raise InvalidOutcomeError("Plan summary snapshot identity mismatch")
                next_plan_memory = self._next_memory_id(
                    plan.plan_memory_head, "PM"
                )
                qualified_plan_memory = (
                    f"{state.run_id}/{plan.coordinator_id}/"
                    f"{plan.plan_id}/{next_plan_memory}"
                )
                updated = self._replace_trial(
                    state,
                    plan.coordinator_id,
                    plan.plan_id,
                    outcome.trial_id,
                    allowed={TrialPhase.PLAN_SUMMARIZING},
                    phase=TrialPhase.PLAN_SUMMARY_READY,
                    plan_snapshot_ref_id=outcome.snapshot_ref.snapshot_id,
                    plan_memory_result=qualified_plan_memory,
                )
                accepted = replace(
                    updated,
                    plans=tuple(
                        replace(
                            item,
                            latest_snapshot_ref_id=outcome.snapshot_ref.snapshot_id,
                            plan_memory_head=qualified_plan_memory,
                        )
                        if item.coordinator_id == plan.coordinator_id
                        and item.plan_id == plan.plan_id
                        else item
                        for item in updated.plans
                    ),
                    accepted_summary_refs=(
                        *updated.accepted_summary_refs,
                        outcome.summary_ref,
                        outcome.evidence_ref,
                    ),
                    accepted_snapshot_refs=(
                        *updated.accepted_snapshot_refs,
                        outcome.snapshot_ref,
                    ),
                    memory=replace(
                        updated.memory,
                        plan_heads={
                            **updated.memory.plan_heads,
                            f"{state.run_id}/{plan.coordinator_id}/"
                            f"{plan.plan_id}": qualified_plan_memory,
                        },
                    ),
                    rm_merge_queue=(
                        *updated.rm_merge_queue,
                        f"{state.run_id}/{plan.coordinator_id}/"
                        f"{plan.plan_id}/{outcome.trial_id}",
                    ),
                )
                return self._finish_agent_call(
                    self._record_agent_session(
                        accepted,
                        outcome.agent_session,
                    ),
                    outcome.call_id,
                )
            if outcome.summary_kind == "run_summary":
                if outcome.coordinator_id is not None or outcome.subject_id != state.run_id:
                    raise InvalidOutcomeError("run summary subject does not match run")
                if outcome.snapshot_ref.kind is not SnapshotKind.RUN:
                    raise InvalidOutcomeError("Run summary requires a Run snapshot")
                trial = self._trial(
                    state,
                    outcome.trial_coordinator_id,
                    outcome.trial_plan_id,
                    outcome.trial_id,
                )
                trial_key = (
                    f"{state.run_id}/{trial.coordinator_id}/"
                    f"{trial.plan_id}/{trial.trial_id}"
                )
                if not state.rm_merge_queue or state.rm_merge_queue[0] != trial_key:
                    raise InvalidOutcomeError("Run summary is not the RM FIFO queue head")
                plan = self._plan(state, trial.coordinator_id, trial.plan_id)
                if plan.latest_snapshot_ref_id is None:
                    raise InvalidOutcomeError("Run summary requires an accepted Plan snapshot")
                progress = self._plan_progress(state, plan.coordinator_id, plan.plan_id)
                plan_status = {
                    PlanTerminalDecision.CONTINUE: PlanStatus.ACTIVE,
                    PlanTerminalDecision.COMPLETED: PlanStatus.COMPLETED,
                    PlanTerminalDecision.FAILED: PlanStatus.FAILED,
                }[progress.decision]
                if plan.kind is PlanKind.BOOTSTRAP:
                    plan_status = PlanStatus.COMPLETED
                next_run_memory_id = self._next_memory_id(
                    state.memory.run_head, "RM"
                )
                next_run_memory = f"{state.run_id}/{next_run_memory_id}"
                updated = self._replace_trial(
                    state,
                    trial.coordinator_id,
                    trial.plan_id,
                    trial.trial_id,
                    allowed={TrialPhase.RUN_SUMMARIZING},
                    phase=TrialPhase.ARCHIVED,
                    archive_status=TrialArchiveStatus.ARCHIVED,
                    run_snapshot_ref_id=outcome.snapshot_ref.snapshot_id,
                    run_memory_result=next_run_memory,
                )
                accepted = replace(
                    updated,
                    plans=tuple(
                        replace(
                            item,
                            status=plan_status,
                            best_trial_id=progress.best_trial_id,
                            best_artifact_ref_id=progress.best_artifact_ref_id,
                            best_score=progress.best_score,
                            best_secondary_score=progress.best_secondary_score,
                            no_improvement_count=progress.no_improvement_count,
                        )
                        if item.coordinator_id == plan.coordinator_id
                        and item.plan_id == plan.plan_id
                        else item
                        for item in updated.plans
                    ),
                    accepted_summary_refs=(
                        *updated.accepted_summary_refs,
                        outcome.summary_ref,
                        outcome.evidence_ref,
                    ),
                    accepted_snapshot_refs=(
                        *updated.accepted_snapshot_refs,
                        outcome.snapshot_ref,
                    ),
                    latest_run_snapshot_ref_id=outcome.snapshot_ref.snapshot_id,
                    memory=replace(
                        updated.memory,
                        run_head=next_run_memory,
                    ),
                    rm_merge_queue=updated.rm_merge_queue[1:],
                )
                accepted = self._record_agent_session(
                    accepted,
                    outcome.agent_session,
                )
                accepted = self._mark_source_eligible(
                    accepted,
                    trial.coordinator_id,
                    trial.plan_id,
                    trial.trial_id,
                    plan_status,
                )
                if plan_status in {PlanStatus.COMPLETED, PlanStatus.FAILED}:
                    accepted = self._close_plan_summarizer_session(
                        accepted,
                        plan.coordinator_id,
                        plan.plan_id,
                    )
                return self._finish_agent_call(accepted, outcome.call_id)
            else:
                raise InvalidOutcomeError(f"unknown summary plan: {outcome.subject_id}")
        if isinstance(outcome, RunCompletedOutcome):
            self._validate_run_completion(state)
            completed = state.next_revision(
                status=RunStatus.COMPLETED,
                research_outcome=self._research_outcome(state),
                completion_kind=CompletionKind.FULL_BUDGET,
            )
            return replace(
                completed,
                agent_sessions=tuple(
                    replace(session, status=AgentSessionStatus.CLOSED)
                    for session in completed.agent_sessions
                ),
            )
        if isinstance(outcome, RunFinishedEarlyOutcome):
            self._validate_run_completion(state, early=True)
            completed = state.next_revision(
                status=RunStatus.COMPLETED,
                research_outcome=self._research_outcome(state),
                completion_kind=CompletionKind.OPERATOR_EARLY_FINISH,
            )
            return replace(
                completed,
                agent_sessions=tuple(
                    replace(session, status=AgentSessionStatus.CLOSED)
                    for session in completed.agent_sessions
                ),
            )
        if isinstance(outcome, RunPausedOutcome):
            if not state.pause_requested:
                raise InvalidOutcomeError(
                    "operator pause was not requested"
                )
            if (
                any(
                    call.status != "retry_pending"
                    and (
                        call.role != AgentRole.ARTIFACT_BUILDER.value
                        or call.round_status == "agent_running"
                    )
                    for call in state.active_agent_calls
                )
                or state.active_engine_commands
                or any(
                    command.status != "pending_submit"
                    for command in state.active_review_commands
                )
            ):
                raise InvalidOutcomeError(
                    "operator pause requires quiesced external work"
                )
            return state.next_revision(
                status=RunStatus.PAUSED,
                pause_requested=False,
                pause_reason=outcome.reason,
                recovery=None,
            )
        if isinstance(outcome, RunSuspendedOutcome):
            continuation_status = (
                state.continuation_status
                if state.status is RunStatus.RECOVERING
                else state.status
            )
            suspended = state.next_revision(
                status=RunStatus.SUSPENDED,
                continuation_status=continuation_status,
                recovery=None,
                failure=outcome.failure,
            )
            suspended = self._record_agent_session(
                suspended,
                outcome.agent_session,
            )
            return (
                self._finish_agent_call(suspended, outcome.call_id)
                if outcome.call_id is not None
                else suspended
            )
        raise TypeError(f"unsupported outcome type: {type(outcome).__name__}")

    def _request_coordinator_finish(
        self,
        state: RunState,
        outcome: CoordinatorFinishRequestedOutcome,
    ) -> RunState:
        self._validate_coordinator_control_run(state, allow_recovering=False)
        coordinator = self._search_coordinator(state, outcome.coordinator_id)
        if coordinator.control_status is not CoordinatorControlStatus.ACTIVE:
            raise InvalidOutcomeError("Coordinator is not active")
        allocated = allocated_plan_slots(state, coordinator.coordinator_id)
        requested = (
            allocated
            if outcome.requested_plan_limit is None
            else outcome.requested_plan_limit
        )
        if requested < 0 or requested > coordinator.original_plan_limit:
            raise InvalidOutcomeError("Coordinator finish target is out of range")
        reason = outcome.reason.strip()
        if not reason:
            raise InvalidOutcomeError("Coordinator finish requires a reason")
        updated = replace(
            coordinator,
            control_status=CoordinatorControlStatus.FINISH_REQUESTED,
            requested_plan_limit=requested,
            effective_plan_limit=max(requested, allocated),
            control_reason=reason,
            control_requested_revision=state.revision + 1,
        )
        return state.next_revision(
            coordinators=tuple(
                updated if item.coordinator_id == updated.coordinator_id else item
                for item in state.coordinators
            )
        )

    def _request_coordinator_cancel(
        self,
        state: RunState,
        outcome: CoordinatorCancelRequestedOutcome,
    ) -> RunState:
        self._validate_coordinator_control_run(state, allow_recovering=True)
        coordinator = self._search_coordinator(state, outcome.coordinator_id)
        if coordinator.control_status not in {
            CoordinatorControlStatus.ACTIVE,
            CoordinatorControlStatus.FINISH_REQUESTED,
        }:
            raise InvalidOutcomeError("Coordinator cannot be cancelled")
        reason = outcome.reason.strip()
        if not reason:
            raise InvalidOutcomeError("Coordinator cancel requires a reason")
        allocated = allocated_plan_slots(state, coordinator.coordinator_id)
        updated = replace(
            coordinator,
            control_status=CoordinatorControlStatus.CANCEL_REQUESTED,
            requested_plan_limit=allocated,
            effective_plan_limit=allocated,
            control_reason=reason,
            control_requested_revision=state.revision + 1,
        )
        return state.next_revision(
            coordinators=tuple(
                updated if item.coordinator_id == updated.coordinator_id else item
                for item in state.coordinators
            )
        )

    def _request_run_finish(
        self,
        state: RunState,
        outcome: RunFinishRequestedOutcome,
    ) -> RunState:
        self._validate_coordinator_control_run(state, allow_recovering=False)
        reason = outcome.reason.strip()
        if not reason:
            raise InvalidOutcomeError("Run finish requires a reason")
        updated = []
        for coordinator in state.coordinators:
            if coordinator.kind is not CoordinatorKind.SEARCH or coordinator.control_status in {
                CoordinatorControlStatus.FINISHED_EARLY,
                CoordinatorControlStatus.CANCELLED,
            }:
                updated.append(coordinator)
                continue
            if coordinator.control_status is CoordinatorControlStatus.CANCEL_REQUESTED:
                updated.append(coordinator)
                continue
            if (
                outcome.requested_plan_limit < 0
                or outcome.requested_plan_limit > coordinator.original_plan_limit
            ):
                raise InvalidOutcomeError("Run finish target is out of range")
            allocated = allocated_plan_slots(state, coordinator.coordinator_id)
            updated.append(
                replace(
                    coordinator,
                    control_status=CoordinatorControlStatus.FINISH_REQUESTED,
                    requested_plan_limit=outcome.requested_plan_limit,
                    effective_plan_limit=max(
                        outcome.requested_plan_limit, allocated
                    ),
                    control_reason=reason,
                    control_requested_revision=state.revision + 1,
                )
            )
        return state.next_revision(
            coordinators=tuple(updated),
            finish_request=RunFinishRequest(
                requested_plan_limit=outcome.requested_plan_limit,
                reason=reason,
                requested_revision=state.revision + 1,
            ),
        )

    def _cancel_coordinator_planning(
        self,
        state: RunState,
        outcome: CoordinatorPlanningCancelledOutcome,
    ) -> RunState:
        coordinator = self._search_coordinator(state, outcome.coordinator_id)
        if coordinator.control_status is not CoordinatorControlStatus.CANCEL_REQUESTED:
            raise InvalidOutcomeError("Coordinator cancellation was not requested")
        matches = tuple(
            item
            for item in state.planning_queue
            if item.coordinator_id == outcome.coordinator_id
            and item.plan_id == outcome.plan_id
            and item.reservation_id == outcome.reservation_id
        )
        if len(matches) != 1:
            raise InvalidOutcomeError("cancelled planning reservation is not active")
        reservation = matches[0]
        plans = tuple(
            item
            for item in state.plans
            if item.coordinator_id == outcome.coordinator_id
            and item.plan_id == outcome.plan_id
        )
        if len(plans) > 1 or (plans and plans[0].status is not PlanStatus.PROPOSED):
            raise InvalidOutcomeError("planning cancellation found an invalid Plan")
        cancelled_plan = (
            replace(plans[0], status=PlanStatus.CANCELLED)
            if plans
            else PlanState(
                plan_id=outcome.plan_id,
                coordinator_id=outcome.coordinator_id,
                kind=PlanKind.SEARCH,
                status=PlanStatus.CANCELLED,
                basis_revision=state.revision,
                reservation_id=reservation.reservation_id,
                reservation_sequence=reservation.sequence,
            )
        )
        retained_plans = tuple(
            item
            for item in state.plans
            if not (
                item.coordinator_id == outcome.coordinator_id
                and item.plan_id == outcome.plan_id
            )
        )
        return state.next_revision(
            plans=(*retained_plans, cancelled_plan),
            planning_queue=tuple(item for item in state.planning_queue if item != reservation),
            active_agent_calls=tuple(
                item
                for item in state.active_agent_calls
                if not (
                    item.coordinator_id == outcome.coordinator_id
                    and (
                        item.plan_id == outcome.plan_id
                        or item.target_subject_ref
                        == subject_ref(
                            state.run_id,
                            outcome.coordinator_id,
                            outcome.plan_id,
                        )
                    )
                )
            ),
            plan_catalog=replace(
                state.plan_catalog,
                revision=state.revision + 1,
                active_intents=tuple(
                    item
                    for item in state.plan_catalog.active_intents
                    if not (
                        item.coordinator_id == outcome.coordinator_id
                        and item.plan_id == outcome.plan_id
                    )
                ),
            ),
        )

    def _cancel_coordinator_trial(
        self,
        state: RunState,
        outcome: CoordinatorTrialCancelledOutcome,
    ) -> RunState:
        coordinator = self._search_coordinator(state, outcome.coordinator_id)
        if coordinator.control_status is not CoordinatorControlStatus.CANCEL_REQUESTED:
            raise InvalidOutcomeError("Coordinator cancellation was not requested")
        trial = self._trial(
            state,
            outcome.coordinator_id,
            outcome.plan_id,
            outcome.trial_id,
        )
        if trial.phase is TrialPhase.ARCHIVED:
            raise InvalidOutcomeError("cancelled Trial is already archived")
        trial_ref = subject_ref(
            state.run_id,
            outcome.coordinator_id,
            outcome.plan_id,
            outcome.trial_id,
        )
        if trial_ref in state.rm_merge_queue:
            raise InvalidOutcomeError("committed RM merge cannot be cancelled")
        expected_snapshots = (
            (
                outcome.trial_snapshot_ref,
                SnapshotKind.TRIAL,
                outcome.coordinator_id,
                outcome.plan_id,
                outcome.trial_id,
            ),
            (
                outcome.plan_snapshot_ref,
                SnapshotKind.PLAN,
                outcome.coordinator_id,
                outcome.plan_id,
                None,
            ),
            (outcome.run_snapshot_ref, SnapshotKind.RUN, None, None, None),
        )
        if any(
            ref.run_id != state.run_id
            or ref.kind is not kind
            or ref.coordinator_id != coordinator_id
            or ref.plan_id != plan_id
            or ref.trial_id != trial_id
            for ref, kind, coordinator_id, plan_id, trial_id in expected_snapshots
        ):
            raise InvalidOutcomeError("cancellation snapshot identity mismatch")
        updated = self._replace_trial(
            state,
            outcome.coordinator_id,
            outcome.plan_id,
            outcome.trial_id,
            allowed=set(TrialPhase) - {TrialPhase.ARCHIVED},
            phase=TrialPhase.ARCHIVED,
            command_id=None,
            engine_retry_pending=False,
            analysis_review_command_id=None,
            analysis_review_retry_pending=False,
            analysis_status="cancelled",
            outcome=TrialOutcome.CANCELLED,
            archive_status=TrialArchiveStatus.ARCHIVED,
            failure_kind=None,
            analysis_packet_ref_id=outcome.trial_snapshot_ref.snapshot_id,
            plan_snapshot_ref_id=outcome.plan_snapshot_ref.snapshot_id,
            run_snapshot_ref_id=outcome.run_snapshot_ref.snapshot_id,
        )
        recovery_changes = {}
        if state.recovery is not None and state.recovery.subject_ref.startswith(
            f"{state.run_id}/{outcome.coordinator_id}/"
        ):
            recovery_changes = {
                "status": state.continuation_status or RunStatus.RUNNING,
                "recovery": None,
                "continuation_status": None,
            }
        return replace(
            updated,
            **recovery_changes,
            plans=tuple(
                replace(
                    item,
                    status=PlanStatus.CANCELLED,
                    latest_snapshot_ref_id=outcome.plan_snapshot_ref.snapshot_id,
                )
                if item.coordinator_id == outcome.coordinator_id
                and item.plan_id == outcome.plan_id
                else item
                for item in updated.plans
            ),
            active_agent_calls=tuple(
                item
                for item in updated.active_agent_calls
                if not (
                    item.coordinator_id == outcome.coordinator_id
                    and item.plan_id == outcome.plan_id
                    and (
                        item.trial_id == outcome.trial_id
                        or item.target_subject_ref == trial_ref
                    )
                )
            ),
            active_engine_commands=tuple(
                item
                for item in updated.active_engine_commands
                if not (
                    item.coordinator_id == outcome.coordinator_id
                    and item.plan_id == outcome.plan_id
                    and item.trial_id == outcome.trial_id
                )
            ),
            active_review_commands=tuple(
                item
                for item in updated.active_review_commands
                if not (
                    item.coordinator_id == outcome.coordinator_id
                    and item.plan_id == outcome.plan_id
                    and item.trial_id == outcome.trial_id
                )
            ),
            operator_evaluations=tuple(
                replace(
                    item,
                    status=OperatorEvaluationStatus.CANCELLED,
                    command_id=None,
                    attempt_id=None,
                    retry_pending=False,
                    error="coordinator_cancelled",
                )
                if item.coordinator_id == outcome.coordinator_id
                and item.plan_id == outcome.plan_id
                and item.trial_id == outcome.trial_id
                and item.status is OperatorEvaluationStatus.PENDING
                else item
                for item in updated.operator_evaluations
            ),
            ranking=replace(
                updated.ranking,
                entries=tuple(
                    item for item in updated.ranking.entries if item.subject_id != trial_ref
                ),
                revision=state.revision,
            ),
            plan_catalog=replace(
                updated.plan_catalog,
                revision=state.revision + 1,
                active_intents=tuple(
                    item
                    for item in updated.plan_catalog.active_intents
                    if not (
                        item.coordinator_id == outcome.coordinator_id
                        and item.plan_id == outcome.plan_id
                    )
                ),
            ),
            accepted_snapshot_refs=(
                *updated.accepted_snapshot_refs,
                outcome.trial_snapshot_ref,
                outcome.plan_snapshot_ref,
                outcome.run_snapshot_ref,
            ),
            latest_run_snapshot_ref_id=outcome.run_snapshot_ref.snapshot_id,
        )

    def _cancel_coordinator_plan(
        self,
        state: RunState,
        outcome: CoordinatorPlanCancelledOutcome,
    ) -> RunState:
        coordinator = self._search_coordinator(state, outcome.coordinator_id)
        if coordinator.control_status is not CoordinatorControlStatus.CANCEL_REQUESTED:
            raise InvalidOutcomeError("Coordinator cancellation was not requested")
        plan = self._plan(state, outcome.coordinator_id, outcome.plan_id)
        if any(
            item.coordinator_id == outcome.coordinator_id
            and item.plan_id == outcome.plan_id
            and item.phase is not TrialPhase.ARCHIVED
            for item in state.trials
        ) or any(
            item.startswith(f"{state.run_id}/{outcome.coordinator_id}/{outcome.plan_id}/")
            for item in state.rm_merge_queue
        ):
            raise InvalidOutcomeError("Plan cancellation requires settled Trials")
        pending_operator_evaluations = tuple(
            item
            for item in state.operator_evaluations
            if item.coordinator_id == outcome.coordinator_id
            and item.plan_id == outcome.plan_id
            and item.status is OperatorEvaluationStatus.PENDING
        )
        pending_operator_commands = {
            item.command_id
            for item in pending_operator_evaluations
            if item.command_id is not None
        }
        if plan.status is PlanStatus.CANCELLED and not pending_operator_evaluations:
            raise InvalidOutcomeError("Plan is already cancelled")
        return state.next_revision(
            plans=tuple(
                replace(item, status=PlanStatus.CANCELLED)
                if item.coordinator_id == outcome.coordinator_id
                and item.plan_id == outcome.plan_id
                else item
                for item in state.plans
            ),
            active_engine_commands=tuple(
                item
                for item in state.active_engine_commands
                if item.command_id not in pending_operator_commands
            ),
            operator_evaluations=tuple(
                replace(
                    item,
                    status=OperatorEvaluationStatus.CANCELLED,
                    command_id=None,
                    attempt_id=None,
                    retry_pending=False,
                    error="coordinator_cancelled",
                )
                if item.coordinator_id == outcome.coordinator_id
                and item.plan_id == outcome.plan_id
                and item.status is OperatorEvaluationStatus.PENDING
                else item
                for item in state.operator_evaluations
            ),
            plan_catalog=replace(
                state.plan_catalog,
                revision=state.revision + 1,
                active_intents=tuple(
                    item
                    for item in state.plan_catalog.active_intents
                    if not (
                        item.coordinator_id == outcome.coordinator_id
                        and item.plan_id == outcome.plan_id
                    )
                ),
            ),
        )

    def _finish_coordinator(
        self,
        state: RunState,
        coordinator_id: str,
        *,
        expected: CoordinatorControlStatus,
        terminal: CoordinatorControlStatus,
    ) -> RunState:
        coordinator = self._search_coordinator(state, coordinator_id)
        if coordinator.control_status is not expected:
            raise InvalidOutcomeError("Coordinator control state cannot settle")
        if allocated_plan_slots(state, coordinator_id) != coordinator.effective_plan_limit:
            raise InvalidOutcomeError("Coordinator effective Plan limit is not allocated")
        prefix = f"{state.run_id}/{coordinator_id}/"
        if (
            any(item.coordinator_id == coordinator_id for item in state.planning_queue)
            or any(
                call.coordinator_id == coordinator_id
                or call.target_subject_ref.startswith(prefix)
                for call in state.active_agent_calls
            )
            or any(
                command.coordinator_id == coordinator_id
                for command in state.active_engine_commands
            )
            or any(
                command.coordinator_id == coordinator_id
                for command in state.active_review_commands
            )
            or any(item.startswith(prefix) for item in state.rm_merge_queue)
        ):
            raise InvalidOutcomeError("Coordinator still owns active work")
        plans = tuple(
            item for item in state.plans if item.coordinator_id == coordinator_id
        )
        if any(
            item.status
            not in {
                PlanStatus.COMPLETED,
                PlanStatus.FAILED,
                PlanStatus.REJECTED,
                PlanStatus.CANCELLED,
            }
            for item in plans
        ):
            raise InvalidOutcomeError("Coordinator still owns a nonterminal Plan")
        if any(
            trial.archive_status is not TrialArchiveStatus.ARCHIVED
            for trial in state.trials
            if trial.coordinator_id == coordinator_id
        ):
            raise InvalidOutcomeError("Coordinator still owns an unarchived Trial")
        replacement = replace(coordinator, control_status=terminal)
        return state.next_revision(
            coordinators=tuple(
                replacement if item.coordinator_id == coordinator_id else item
                for item in state.coordinators
            ),
            agent_sessions=tuple(
                replace(session, status=AgentSessionStatus.CLOSED)
                if session.coordinator_id == coordinator_id
                else session
                for session in state.agent_sessions
            ),
        )

    @staticmethod
    def _validate_coordinator_control_run(
        state: RunState,
        *,
        allow_recovering: bool,
    ) -> None:
        allowed = {RunStatus.RUNNING}
        if allow_recovering:
            allowed.add(RunStatus.RECOVERING)
        if state.status not in allowed or state.pause_requested:
            raise InvalidOutcomeError("Run is not accepting Coordinator control")
        if state.bootstrap.enabled and state.bootstrap.status is not BootstrapStatus.COMPLETED:
            raise InvalidOutcomeError("Coordinator control requires completed Bootstrap")

    @staticmethod
    def _search_coordinator(state: RunState, coordinator_id: str):
        coordinator = next(
            (
                item
                for item in state.coordinators
                if item.coordinator_id == coordinator_id
            ),
            None,
        )
        if coordinator is None or coordinator.kind is not CoordinatorKind.SEARCH:
            raise InvalidOutcomeError("unknown Search Coordinator")
        return coordinator

    @staticmethod
    def _research_outcome(state: RunState) -> ResearchOutcome:
        search_trials = tuple(
            trial for trial in state.trials if trial.kind is TrialKind.SEARCH
        )
        if any(trial.offline_score is not None for trial in search_trials):
            return ResearchOutcome.VALID_RESULT
        if any(
            plan.kind is PlanKind.SEARCH and plan.decision_ref_id is not None
            for plan in state.plans
        ):
            return ResearchOutcome.NO_VALID_TRIAL
        return ResearchOutcome.NO_VALID_PLAN

    def _validate_run_completion(self, state: RunState, *, early: bool = False) -> None:
        if state.status is not RunStatus.RUNNING:
            raise InvalidOutcomeError("only a running Run can complete")
        search_plans = tuple(
            plan for plan in state.plans if plan.kind is PlanKind.SEARCH
        )
        search_trials = tuple(
            trial for trial in state.trials if trial.kind is TrialKind.SEARCH
        )
        if len(search_plans) != effective_max_plans(state):
            raise InvalidOutcomeError(
                "run completion requires allocated Plan budget"
            )
        controlled = any(
            item.kind is CoordinatorKind.SEARCH
            and item.control_status is not CoordinatorControlStatus.ACTIVE
            for item in state.coordinators
        )
        if early != controlled:
            raise InvalidOutcomeError("run completion kind does not match Coordinator control")
        if any(
            item.kind is CoordinatorKind.SEARCH
            and item.control_status
            in {
                CoordinatorControlStatus.FINISH_REQUESTED,
                CoordinatorControlStatus.CANCEL_REQUESTED,
            }
            for item in state.coordinators
        ):
            raise InvalidOutcomeError("run completion requires terminal Coordinators")
        for plan in search_plans:
            if plan.decision_ref_id is None or plan.status is PlanStatus.CANCELLED:
                continue
            trial_count = sum(
                trial.coordinator_id == plan.coordinator_id
                and trial.plan_id == plan.plan_id
                for trial in search_trials
            )
            if trial_count != state.portfolio.max_trials_per_plan:
                raise InvalidOutcomeError(
                    "run completion requires each accepted Plan's configured "
                    "Trial count"
                )
        if state.planning_queue:
            raise InvalidOutcomeError(
                "run completion requires an empty planning queue"
            )
        if state.rm_merge_queue:
            raise InvalidOutcomeError(
                "run completion requires an empty RM merge queue"
            )
        if (
            state.active_agent_calls
            or state.active_engine_commands
            or state.active_review_commands
        ):
            raise InvalidOutcomeError(
                "run completion requires no active external work"
            )
        if any(
            plan.status not in {
                PlanStatus.COMPLETED,
                PlanStatus.FAILED,
                PlanStatus.REJECTED,
                PlanStatus.CANCELLED,
            }
            for plan in state.plans
        ):
            raise InvalidOutcomeError(
                "run completion requires terminal Plans"
            )
        if any(
            trial.phase is not TrialPhase.ARCHIVED
            or trial.archive_status is not TrialArchiveStatus.ARCHIVED
            for trial in state.trials
        ):
            raise InvalidOutcomeError(
                "run completion requires archived Trials"
            )
        summary_kinds = {ref.kind for ref in state.accepted_summary_refs}
        summarized_trials = tuple(
            trial
            for trial in search_trials
            if trial.outcome is not TrialOutcome.CANCELLED
        )
        if summarized_trials and not {"plan_summary", "run_summary"}.issubset(
            summary_kinds
        ):
            raise InvalidOutcomeError(
                "run completion requires Plan and Run summaries"
            )
        required_operator_targets: set[str] = set()
        if state.bootstrap.enabled:
            if state.bootstrap.status is not BootstrapStatus.COMPLETED:
                raise InvalidOutcomeError(
                    "run completion requires completed Bootstrap"
                )
            if state.bootstrap.base_evaluation is None:
                raise InvalidOutcomeError(
                    "run completion requires a Base evaluation record"
                )
            required_operator_targets.add(
                subject_ref(state.run_id, "c000", "p000", "base")
            )
            required_operator_targets.update(
                subject_ref(
                    state.run_id,
                    trial.coordinator_id,
                    trial.plan_id,
                    trial.trial_id,
                )
                for trial in state.trials
                if trial.outcome is TrialOutcome.SUCCEEDED
            )
        operator_by_target = {
            item.target_id: item for item in state.operator_evaluations
        }
        if len(operator_by_target) != len(state.operator_evaluations):
            raise InvalidOutcomeError(
                "run completion rejects duplicate operator targets"
            )
        missing = required_operator_targets - set(operator_by_target)
        if missing:
            raise InvalidOutcomeError(
                "run completion requires operator targets: "
                + ", ".join(sorted(missing))
            )
        if any(
            operator_by_target[target].status
            not in {
                OperatorEvaluationStatus.COMPLETED,
                OperatorEvaluationStatus.FAILED,
                OperatorEvaluationStatus.CANCELLED,
                OperatorEvaluationStatus.NOT_APPLICABLE,
            }
            for target in required_operator_targets
        ):
            raise InvalidOutcomeError(
                "run completion requires terminal operator evaluations"
            )

    def _register_baseline_trial(
        self,
        state: RunState,
        *,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        artifact_ref,
    ) -> RunState:
        if any(
            trial.coordinator_id == coordinator_id
            and trial.plan_id == plan_id
            and trial.trial_id == trial_id
            for trial in state.trials
        ):
            raise InvalidOutcomeError("duplicate Trial scope")
        return state.next_revision(
            trials=(
                *state.trials,
                TrialState(
                    trial_id=trial_id,
                    coordinator_id=coordinator_id,
                    plan_id=plan_id,
                    kind=TrialKind.BOOTSTRAP_BASELINE,
                    phase=TrialPhase.ARTIFACT_READY,
                    artifact_ref_id=artifact_ref.artifact_id,
                    trial_record_ref=(
                        f"{state.run_id}/{coordinator_id}/"
                        f"{plan_id}/{trial_id}/record"
                    ),
                    plan_memory_basis=self._plan(
                        state, coordinator_id, plan_id
                    ).plan_memory_head,
                    run_memory_basis=state.memory.run_head,
                ),
            ),
            accepted_evidence_refs=(
                *state.accepted_evidence_refs,
                artifact_ref,
            ),
        )

    def _advance_bootstrap_after_engine(
        self,
        state: RunState,
        updated: RunState,
        outcome: EngineCompletedOutcome | EngineFailedOutcome,
    ) -> RunState:
        trial = self._trial(
            state,
            outcome.coordinator_id,
            outcome.plan_id,
            outcome.trial_id,
        )
        if trial.kind is not TrialKind.BOOTSTRAP_BASELINE:
            return updated
        self._require_bootstrap_status(
            state, BootstrapStatus.P000_TRAINING
        )
        return replace(
            updated,
            bootstrap=replace(
                updated.bootstrap,
                status=BootstrapStatus.P000_ANALYZING,
            ),
        )

    def _require_active_engine_attempt(
        self,
        state: RunState,
        outcome: EngineCompletedOutcome | EngineFailedOutcome,
    ) -> None:
        trial = self._trial(
            state,
            outcome.coordinator_id,
            outcome.plan_id,
            outcome.trial_id,
        )
        active = tuple(
            command
            for command in state.active_engine_commands
            if command.command_id == outcome.command_id
        )
        if (
            trial.phase is not TrialPhase.ENGINE_RUNNING
            or trial.command_id != outcome.command_id
            or len(active) != 1
        ):
            raise InvalidOutcomeError(
                "Engine outcome is not for the current active Attempt"
            )

    @staticmethod
    def _require_bootstrap_status(
        state: RunState,
        expected: BootstrapStatus,
    ) -> None:
        if not state.bootstrap.enabled or state.bootstrap.status is not expected:
            raise InvalidOutcomeError(
                "Bootstrap status mismatch: "
                f"expected {expected.value}, found {state.bootstrap.status.value}"
            )

    @staticmethod
    def _plan(state: RunState, coordinator_id: str | None, plan_id: str):
        matches = [
            plan for plan in state.plans
            if plan.coordinator_id == coordinator_id and plan.plan_id == plan_id
        ]
        if len(matches) != 1:
            raise InvalidOutcomeError(f"unknown summary plan: {plan_id}")
        return matches[0]

    @staticmethod
    def _trial(
        state: RunState,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
    ):
        matches = [
            trial
            for trial in state.trials
            if trial.coordinator_id == coordinator_id
            and trial.plan_id == plan_id
            and trial.trial_id == trial_id
        ]
        if len(matches) != 1:
            raise InvalidOutcomeError(
                "unknown or ambiguous Trial: "
                f"{coordinator_id}/{plan_id}/{trial_id}"
            )
        return matches[0]

    @staticmethod
    def _plan_progress(state: RunState, coordinator_id: str, plan_id: str):
        trials = tuple(
            trial
            for trial in state.trials
            if trial.coordinator_id == coordinator_id
            and trial.plan_id == plan_id
            and trial.analysis_packet_ref_id is not None
        )
        metrics = tuple(
            TrialMetric(
                trial.trial_id,
                trial.artifact_ref_id or f"unavailable-{trial.trial_id}",
                trial.offline_score,
                trial.offline_secondary_score,
            )
            for trial in trials
        )
        return PlanStopPolicy(
            min_trials=state.portfolio.max_trials_per_plan,
            max_trials=state.portfolio.max_trials_per_plan,
            no_improvement_patience=state.portfolio.max_trials_per_plan + 1,
            direction=state.ranking.direction,
        ).evaluate(metrics)

    @staticmethod
    def _operator_command_ref(
        record: OperatorEvaluationRecord,
    ) -> EngineCommandRef:
        assert record.command_id is not None
        assert record.logical_command_id is not None
        assert record.attempt_id is not None
        return EngineCommandRef(
            command_id=record.command_id,
            logical_command_id=record.logical_command_id,
            attempt_id=record.attempt_id,
            attempt_index=record.attempt_index,
            coordinator_id=record.coordinator_id or "c000",
            plan_id=record.plan_id or "p000",
            trial_id=record.trial_id or "base",
            kind="evaluate",
        )

    @staticmethod
    def _operator_ranking_match(
        entry: RankingEntry,
        record: OperatorEvaluationRecord,
    ) -> bool:
        if record.target_kind == "base":
            return (
                entry.level == "reference"
                and entry.subject_id == record.target_id
            )
        return (
            entry.level == "trial_level"
            and entry.subject_id == record.target_id
        )

    def _record_trial_result(
        self,
        state: RunState,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
    ) -> RunState:
        trial = self._trial(state, coordinator_id, plan_id, trial_id)
        progress = self._plan_progress(state, coordinator_id, plan_id)
        ranking_entries = list(state.ranking.entries)
        subject_id = f"{state.run_id}/{coordinator_id}/{plan_id}/{trial_id}"
        artifact_ref = next(
            (
                ref
                for ref in state.accepted_evidence_refs
                if ref.artifact_id == trial.artifact_ref_id
            ),
            None,
        )
        if trial.offline_score is not None:
            operator = next(
                (
                    item
                    for item in state.operator_evaluations
                    if item.target_id == subject_id
                ),
                None,
            )
            trial_entry = RankingEntry(
                subject_id,
                trial.offline_score,
                level="trial_level",
                secondary_score=trial.offline_secondary_score,
                accepted_revision=state.revision,
                artifact_ref_id=trial.artifact_ref_id,
                artifact_digest=(artifact_ref.digest if artifact_ref else None),
                operator_status=(
                    operator.status.value
                    if operator is not None
                    else "not_scheduled"
                ),
                operator_result_ref=(
                    operator.result_ref if operator is not None else None
                ),
            )
            ranking_entries = [
                item
                for item in ranking_entries
                if not (
                    item.level == "trial_level"
                    and item.subject_id == subject_id
                )
            ]
            ranking_entries.append(trial_entry)
        ranking_entries = [
            item
            for item in ranking_entries
            if not (
                item.level == "plan_level"
                and item.subject_id
                == f"{state.run_id}/{coordinator_id}/{plan_id}"
            )
        ]
        if progress.best_trial_id is not None and progress.best_score is not None:
            best_trial = self._trial(
                state,
                coordinator_id,
                plan_id,
                progress.best_trial_id,
            )
            best_ref = next(
                (
                    ref
                    for ref in state.accepted_evidence_refs
                    if ref.artifact_id == best_trial.artifact_ref_id
                ),
                None,
            )
            ranking_entries.append(
                RankingEntry(
                    f"{state.run_id}/{coordinator_id}/{plan_id}",
                    progress.best_score,
                    level="plan_level",
                    secondary_score=progress.best_secondary_score,
                    accepted_revision=state.revision,
                    artifact_ref_id=progress.best_artifact_ref_id,
                    artifact_digest=(best_ref.digest if best_ref else None),
                    representative_trial_id=progress.best_trial_id,
                )
            )
        ranking_entries.sort(
            key=lambda item: (
                {"reference": 0, "trial_level": 1, "plan_level": 2}[item.level],
                *score_pair_sort_key(
                    item.score,
                    item.secondary_score,
                    direction=state.ranking.direction,
                ),
                -item.accepted_revision,
                item.subject_id,
            )
        )
        return replace(
            state,
            plans=tuple(
                replace(
                    plan,
                    best_trial_id=progress.best_trial_id,
                    best_artifact_ref_id=progress.best_artifact_ref_id,
                    best_score=progress.best_score,
                    best_secondary_score=progress.best_secondary_score,
                    no_improvement_count=progress.no_improvement_count,
                )
                if plan.coordinator_id == coordinator_id and plan.plan_id == plan_id
                else plan
                for plan in state.plans
            ),
            ranking=replace(
                state.ranking,
                entries=tuple(ranking_entries),
                revision=state.revision,
            ),
        )

    def _mark_source_eligible(
        self,
        state: RunState,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        plan_status: PlanStatus,
    ) -> RunState:
        trial = self._trial(state, coordinator_id, plan_id, trial_id)
        subject_id = f"{state.run_id}/{coordinator_id}/{plan_id}/{trial_id}"
        entries = tuple(
            replace(item, source_eligible=True)
            if item.level == "trial_level"
            and item.subject_id == subject_id
            and trial.artifact_ref_id is not None
            else item
            for item in state.ranking.entries
        )
        eligible_trials = tuple(
            item
            for item in entries
            if item.level == "trial_level"
            and item.subject_id.startswith(
                f"{state.run_id}/{coordinator_id}/{plan_id}/"
            )
            and item.source_eligible
        )
        if eligible_trials:
            best = sorted(
                eligible_trials,
                key=lambda item: (
                    *score_pair_sort_key(
                        item.score,
                        item.secondary_score,
                        direction=state.ranking.direction,
                    ),
                    -item.accepted_revision,
                    item.subject_id,
                ),
            )[0]
            entries = tuple(
                replace(
                    item,
                    source_eligible=True,
                    representative_trial_id=best.subject_id.rsplit("/", 1)[-1],
                    artifact_ref_id=best.artifact_ref_id,
                    artifact_digest=best.artifact_digest,
                )
                if item.level == "plan_level"
                and item.subject_id
                == f"{state.run_id}/{coordinator_id}/{plan_id}"
                else item
                for item in entries
            )
            source = SourceEligibleResult(
                coordinator_id=coordinator_id,
                plan_id=plan_id,
                representative_trial_id=best.subject_id.rsplit("/", 1)[-1],
                score=best.score,
                evaluation_profile=best.evaluation_profile,
                artifact_ref_id=str(best.artifact_ref_id),
                artifact_digest=str(best.artifact_digest),
                rm_version=state.memory.run_head,
                archived_revision=state.revision,
                secondary_score=best.secondary_score,
            )
            sources = tuple(
                item
                for item in state.plan_catalog.source_eligible_results
                if not (
                    item.coordinator_id == coordinator_id
                    and item.plan_id == plan_id
                )
            ) + (source,)
        else:
            sources = state.plan_catalog.source_eligible_results
        terminal = plan_status in {PlanStatus.COMPLETED, PlanStatus.FAILED}
        return replace(
            state,
            ranking=replace(
                state.ranking,
                entries=entries,
                revision=state.revision,
            ),
            plan_catalog=replace(
                state.plan_catalog,
                revision=state.revision,
                active_intents=tuple(
                    item
                    for item in state.plan_catalog.active_intents
                    if not (
                        terminal
                        and item.coordinator_id == coordinator_id
                        and item.plan_id == plan_id
                    )
                ),
                source_eligible_results=sources,
            ),
        )

    def _apply_planning(
        self,
        state: RunState,
        outcome: PlanningDecisionOutcome,
    ) -> RunState:
        if not state.planning_queue:
            raise InvalidOutcomeError("accepted Plan has no planning reservation")
        reservation = state.planning_queue[0]
        if (
            reservation.status != "planning"
            or reservation.coordinator_id != outcome.plan.coordinator_id
            or reservation.plan_id != outcome.plan.plan_id
        ):
            raise InvalidOutcomeError("accepted Plan reservation mismatch")
        snapshot = outcome.snapshot_ref
        if (
            snapshot is None
            or snapshot.kind is not SnapshotKind.PLAN
            or snapshot.run_id != state.run_id
            or snapshot.coordinator_id != outcome.plan.coordinator_id
            or snapshot.plan_id != outcome.plan.plan_id
            or snapshot.trial_id is not None
            or snapshot.revision != state.revision + 1
        ):
            raise InvalidOutcomeError(
                "accepted Plan requires its opening Plan snapshot"
            )
        if (
            outcome.decision_ref.kind != "planning_decision"
            or outcome.plan.decision_ref_id != outcome.decision_ref.artifact_id
        ):
            raise InvalidOutcomeError(
                "plan must identify its accepted PlanningDecision artifact"
            )
        if (
            len(outcome.supporting_refs) != 1
            or outcome.supporting_refs[0].kind != "planning_decision_report"
            or outcome.plan.decision_report_ref_id
            != outcome.supporting_refs[0].artifact_id
        ):
            raise InvalidOutcomeError(
                "plan must identify its accepted PlanningDecision report"
            )
        if (
            not isinstance(outcome.plan.hypothesis_comparator, dict)
            or not isinstance(outcome.plan.portfolio_comparator, dict)
            or outcome.plan.portfolio_comparator.get("ranking_revision")
            != (outcome.plan.planning_basis or {}).get("ranking_revision")
        ):
            raise InvalidOutcomeError(
                "accepted Search Plan requires immutable comparator bindings"
            )
        coordinator = self._search_coordinator(
            state, outcome.plan.coordinator_id
        )
        if coordinator.control_status not in {
            CoordinatorControlStatus.ACTIVE,
            CoordinatorControlStatus.FINISH_REQUESTED,
        }:
            raise InvalidOutcomeError("Coordinator is not accepting Plans")
        if any(
            plan.coordinator_id == outcome.plan.coordinator_id
            and plan.plan_id == outcome.plan.plan_id
            for plan in state.plans
        ):
            raise InvalidOutcomeError("duplicate Plan scope")
        if self._search_plan_count(state) >= effective_max_plans(state):
            raise InvalidOutcomeError("plan capacity exhausted")
        self._validate_plan_relation(state, outcome.plan)
        accepted = state.next_revision(
            status=RunStatus.RUNNING,
            plans=(
                *state.plans,
                replace(
                    outcome.plan,
                    status=PlanStatus.PROPOSED,
                    latest_snapshot_ref_id=snapshot.snapshot_id,
                    plan_memory_head=(
                        f"{state.run_id}/{outcome.plan.coordinator_id}/"
                        f"{outcome.plan.plan_id}/PM000"
                    ),
                    reservation_id=reservation.reservation_id,
                    reservation_sequence=reservation.sequence,
                ),
            ),
            accepted_plan_refs=(
                *state.accepted_plan_refs,
                outcome.decision_ref,
                *outcome.supporting_refs,
            ),
            accepted_snapshot_refs=(
                *state.accepted_snapshot_refs,
                snapshot,
            ),
            memory=replace(
                state.memory,
                plan_heads={
                    **state.memory.plan_heads,
                    f"{state.run_id}/{outcome.plan.coordinator_id}/"
                    f"{outcome.plan.plan_id}": (
                        f"{state.run_id}/{outcome.plan.coordinator_id}/"
                        f"{outcome.plan.plan_id}/PM000"
                    ),
                },
            ),
            planning_queue=(
                replace(reservation, status="catalog_pending"),
                *state.planning_queue[1:],
            ),
        )
        return self._finish_agent_call(
            self._record_agent_session(accepted, outcome.agent_session),
            outcome.call_id,
        )

    def _apply_plan_catalog_update(
        self,
        state: RunState,
        outcome: PlanCatalogUpdatedOutcome,
    ) -> RunState:
        if not state.planning_queue:
            raise InvalidOutcomeError("Plan Catalog update has no reservation")
        reservation = state.planning_queue[0]
        if (
            reservation.status != "catalog_pending"
            or reservation.coordinator_id != outcome.coordinator_id
            or reservation.plan_id != outcome.plan_id
        ):
            raise InvalidOutcomeError("Plan Catalog update is not the FIFO head")
        matches = tuple(
            plan
            for plan in state.plans
            if plan.coordinator_id == outcome.coordinator_id
            and plan.plan_id == outcome.plan_id
        )
        if len(matches) != 1 or matches[0].status is not PlanStatus.PROPOSED:
            raise InvalidOutcomeError("Plan Catalog update requires one proposed Plan")
        plan = matches[0]
        if (
            plan.decision_ref_id is None
            or plan.decision_report_ref_id is None
            or plan.relation is None
        ):
            raise InvalidOutcomeError(
                "Plan Catalog update requires an accepted direction report"
            )
        if any(
            item.coordinator_id == plan.coordinator_id
            and item.plan_id == plan.plan_id
            for item in state.plan_catalog.active_intents
        ):
            raise InvalidOutcomeError("Plan Catalog direction is already active")
        return state.next_revision(
            plans=tuple(
                replace(item, status=PlanStatus.ACTIVE)
                if item.coordinator_id == plan.coordinator_id
                and item.plan_id == plan.plan_id
                else item
                for item in state.plans
            ),
            planning_queue=state.planning_queue[1:],
            plan_catalog=replace(
                state.plan_catalog,
                revision=state.revision + 1,
                active_intents=(
                    *state.plan_catalog.active_intents,
                    ActivePlanIntent(
                        coordinator_id=plan.coordinator_id,
                        plan_id=plan.plan_id,
                        accepted_revision=state.revision,
                        decision_ref_id=plan.decision_ref_id,
                        decision_report_ref_id=plan.decision_report_ref_id,
                        relation_kind=plan.relation.kind.value,
                        related_plan_keys=tuple(
                            key.subject_ref
                            for key in plan.relation.related_plan_keys
                        ),
                    ),
                ),
            ),
        )

    @staticmethod
    def _finish_agent_call(state: RunState, call_id: str) -> RunState:
        matches = tuple(
            item for item in state.active_agent_calls if item.call_id == call_id
        )
        if len(matches) != 1:
            raise InvalidOutcomeError(
                f"accepted Agent outcome requires one active Call: {call_id}"
            )
        return replace(
            state,
            active_agent_calls=tuple(
                item for item in state.active_agent_calls
                if item.call_id != call_id
            ),
        )

    @staticmethod
    def _builder_round_call(
        state: RunState,
        call_id: str,
        reflection_index: int,
    ):
        matches = tuple(
            item
            for item in state.active_agent_calls
            if item.call_id == call_id
            and item.role == AgentRole.ARTIFACT_BUILDER.value
            and item.reflection_index == reflection_index
        )
        if len(matches) != 1:
            raise InvalidOutcomeError(
                "Builder round requires one exact active Call"
            )
        return matches[0]

    def _complete_builder_delivery(
        self,
        state: RunState,
        *,
        call_id: str,
        attempt_id: str,
        reflection_index: int,
        delivery_ref: str,
        expected_event: str,
    ) -> RunState:
        active = self._builder_round_call(state, call_id, reflection_index)
        if (
            active.round_status != "agent_running"
            or active.attempt_id != attempt_id
            or not delivery_ref
            or (expected_event == "proposal" and reflection_index != 0)
            or (expected_event == "reflection" and reflection_index == 0)
        ):
            raise InvalidOutcomeError(
                f"Builder {expected_event} completion does not match active round"
            )
        return state.next_revision(
            active_agent_calls=tuple(
                replace(
                    item,
                    round_status="delivery_ready",
                    current_delivery_ref=delivery_ref,
                    current_realization_ref=None,
                )
                if item.call_id == call_id
                else item
                for item in state.active_agent_calls
            )
        )

    @staticmethod
    def _next_memory_id(current: str | None, prefix: str) -> str:
        if current is None:
            raise InvalidOutcomeError(f"{prefix} head is unavailable")
        short = current.rsplit("/", 1)[-1]
        if not short.startswith(prefix) or not short[len(prefix):].isdigit():
            raise InvalidOutcomeError(f"invalid {prefix} head: {current}")
        return f"{prefix}{int(short[len(prefix):]) + 1:03d}"

    @staticmethod
    def _record_agent_session(
        state: RunState,
        session: AgentSession | None,
    ) -> RunState:
        if session is None:
            return state
        if session.run_id != state.run_id:
            raise InvalidOutcomeError("Agent Session belongs to another Run")
        if len(session.call_ids) != len(set(session.call_ids)):
            raise InvalidOutcomeError("Agent Session Call IDs must be unique")
        matches = tuple(
            item
            for item in state.agent_sessions
            if item.session_id == session.session_id
        )
        if not matches:
            return replace(
                state,
                agent_sessions=(*state.agent_sessions, session),
            )
        if len(matches) != 1:
            raise InvalidOutcomeError("duplicate Agent Session ID")
        current = matches[0]
        identity = (
            "run_id",
            "role",
            "subject_id",
            "coordinator_id",
            "plan_id",
            "trial_id",
        )
        if any(
            getattr(current, field) != getattr(session, field)
            for field in identity
        ):
            raise InvalidOutcomeError("Agent Session identity changed")
        if not set(current.call_ids).issubset(session.call_ids):
            raise InvalidOutcomeError("Agent Session lost accepted Call IDs")
        if (
            current.status is AgentSessionStatus.CLOSED
            and session.status is AgentSessionStatus.ACTIVE
        ):
            raise InvalidOutcomeError("closed Agent Session cannot become active")
        return replace(
            state,
            agent_sessions=tuple(
                session if item.session_id == session.session_id else item
                for item in state.agent_sessions
            ),
        )

    @staticmethod
    def _close_plan_summarizer_session(
        state: RunState,
        coordinator_id: str,
        plan_id: str,
    ) -> RunState:
        return replace(
            state,
            agent_sessions=tuple(
                replace(session, status=AgentSessionStatus.CLOSED)
                if session.coordinator_id == coordinator_id
                and session.plan_id == plan_id
                else session
                for session in state.agent_sessions
            ),
        )

    @staticmethod
    def _validate_plan_relation(state: RunState, plan) -> None:
        relation = plan.relation
        if relation is None:
            raise InvalidOutcomeError("search Plan requires a frozen relation")
        if relation.kind is PlanRelationKind.NEW_DIRECTION:
            source_keys = (PlanKey(state.run_id, "c000", "p000"),)
        else:
            source_keys = relation.related_plan_keys
        sources = []
        for key in source_keys:
            if key.run_id != state.run_id:
                raise InvalidOutcomeError("Plan relation source belongs to another Run")
            matches = [
                item
                for item in state.plans
                if item.coordinator_id == key.coordinator_id
                and item.plan_id == key.plan_id
            ]
            if len(matches) != 1:
                raise InvalidOutcomeError(
                    "unknown Plan relation source: "
                    f"{key.coordinator_id}/{key.plan_id}"
                )
            source = matches[0]
            if (
                source.status is not PlanStatus.COMPLETED
                or source.best_artifact_ref_id is None
            ):
                raise InvalidOutcomeError(
                    "Plan relation sources must be completed with a best artifact"
                )
            sources.append(source)
        frozen = tuple(source.best_artifact_ref_id for source in sources)
        if frozen != relation.seed_artifact_ref_ids:
            raise InvalidOutcomeError(
                "Plan relation seed artifacts do not match frozen source best artifacts"
            )

    @staticmethod
    def _retryable_attempt_state(
        state: RunState,
        *,
        subject_ref: str,
        logical_work_ref: str,
        attempt_id: str,
        attempt_index: int,
        failure_kind: str,
        message: str,
        receipt_id: str,
        lease_ref: str,
        observed_at: float,
        operator_pause: bool,
    ) -> dict[str, object]:
        if state.status not in {RunStatus.BOOTSTRAPPING, RunStatus.RUNNING}:
            raise InvalidOutcomeError(
                "retryable Attempt must fail from bootstrapping or running"
            )
        if operator_pause:
            return {
                "status": state.status,
                "continuation_status": state.continuation_status,
                "recovery": None,
                "failure": state.failure,
            }
        policy = state.automatic_recovery
        if attempt_index >= policy.max_attempts:
            return {
                "status": RunStatus.SUSPENDED,
                "continuation_status": state.status,
                "recovery": None,
                "failure": FailureState(
                    code=f"automatic_recovery_exhausted:{failure_kind}",
                    message=message,
                    retryable=True,
                ),
            }
        return {
            "status": RunStatus.RECOVERING,
            "continuation_status": state.status,
            "recovery": RecoveryState(
                subject_ref=subject_ref,
                logical_work_ref=logical_work_ref,
                failed_attempt_id=attempt_id,
                failed_attempt_index=attempt_index,
                failure_kind=failure_kind,
                attempts_used=attempt_index,
                max_attempts=policy.max_attempts,
                next_attempt_index=attempt_index + 1,
                started_at=observed_at,
                readiness_deadline=(
                    observed_at
                    + policy.dependency_readiness_timeout_seconds
                ),
                fence_ref=f"receipt:{receipt_id}",
                lease_ref=lease_ref,
            ),
            "failure": None,
        }

    @staticmethod
    def _finish_automatic_recovery(
        state: RunState,
        submitted_attempt_index: int,
    ) -> dict[str, object]:
        if state.status is not RunStatus.RECOVERING:
            return {}
        if state.recovery is None or state.continuation_status not in {
            RunStatus.BOOTSTRAPPING,
            RunStatus.RUNNING,
        }:
            raise InvalidOutcomeError("automatic recovery state is incomplete")
        if submitted_attempt_index != state.recovery.next_attempt_index:
            raise InvalidOutcomeError(
                "automatic recovery submitted the wrong physical Attempt index"
            )
        return {
            "status": state.continuation_status,
            "continuation_status": None,
            "recovery": None,
            "failure": None,
        }

    @staticmethod
    def _rewind_failed_bootstrap(
        suspended: RunState,
        resumed: RunState,
    ) -> RunState:
        if suspended.bootstrap.status is not BootstrapStatus.FAILED:
            return resumed
        failed_command_ids = {
            command_id
            for command_id, status in suspended.bootstrap.command_status.items()
            if status == "failed"
        }
        rewound = replace(
            resumed,
            bootstrap=replace(
                resumed.bootstrap,
                status=(
                    BootstrapStatus.BASE_EVALUATING
                    if failed_command_ids
                    else BootstrapStatus.P000_TRAINING
                ),
                command_status={
                    command_id: (
                        "retry_pending"
                        if command_id in failed_command_ids
                        else status
                    )
                    for command_id, status in resumed.bootstrap.command_status.items()
                },
                error=None,
            ),
            active_engine_commands=tuple(
                command
                for command in resumed.active_engine_commands
                if command.command_id not in failed_command_ids
            ),
        )
        if failed_command_ids:
            return rewound

        reopened_agent_sessions = tuple(
            replace(session, status=AgentSessionStatus.ACTIVE)
            if session.status is AgentSessionStatus.CLOSED
            and session.role
            in {
                AgentRole.ANALYZER,
                AgentRole.PLAN_SUMMARIZER,
                AgentRole.RUN_SUMMARIZER,
            }
            else session
            for session in rewound.agent_sessions
        )
        completed_p000 = next(
            (
                trial
                for trial in suspended.trials
                if trial.kind is TrialKind.BOOTSTRAP_BASELINE
                and trial.outcome is TrialOutcome.SUCCEEDED
                and trial.phase is TrialPhase.EVIDENCE_READY
            ),
            None,
        )
        analysis_p000 = next(
            (
                trial
                for trial in suspended.trials
                if trial.kind is TrialKind.BOOTSTRAP_BASELINE
                and trial.outcome is TrialOutcome.SUCCEEDED
                and trial.phase
                in {
                    TrialPhase.ANALYSIS_DESIGNING,
                    TrialPhase.REVIEW_RUNNING,
                    TrialPhase.REVIEW_READY,
                    TrialPhase.ANALYZING,
                    TrialPhase.ANALYSIS_READY,
                }
            ),
            None,
        )
        if analysis_p000 is not None:
            return replace(
                rewound,
                agent_sessions=reopened_agent_sessions,
                bootstrap=replace(
                    rewound.bootstrap,
                    status=BootstrapStatus.P000_ANALYZING,
                ),
            )
        if completed_p000 is not None:
            return replace(
                rewound,
                agent_sessions=reopened_agent_sessions,
                bootstrap=replace(
                    rewound.bootstrap,
                    status=BootstrapStatus.P000_ANALYZING,
                ),
            )

        archived_p000 = next(
            (
                trial
                for trial in suspended.trials
                if trial.kind is TrialKind.BOOTSTRAP_BASELINE
                and trial.outcome is TrialOutcome.SUCCEEDED
                and trial.phase is TrialPhase.ARCHIVED
                and trial.archive_status is TrialArchiveStatus.ARCHIVED
            ),
            None,
        )
        if archived_p000 is not None:
            return replace(
                rewound,
                agent_sessions=reopened_agent_sessions,
                bootstrap=replace(
                    rewound.bootstrap,
                    status=BootstrapStatus.P000_RUN_SUMMARIZING,
                ),
            )

        summary_p000 = next(
            (
                trial
                for trial in suspended.trials
                if trial.kind is TrialKind.BOOTSTRAP_BASELINE
                and trial.outcome is TrialOutcome.SUCCEEDED
                and trial.phase is TrialPhase.PLAN_SUMMARIZING
            ),
            None,
        )
        if summary_p000 is not None:
            return replace(
                rewound,
                agent_sessions=reopened_agent_sessions,
                bootstrap=replace(
                    rewound.bootstrap,
                    status=BootstrapStatus.P000_PLAN_SUMMARIZING,
                ),
            )

        # Packaging can fail after a successful Engine receipt but before the
        # EngineCompleted transition is committed.  In that case the P000
        # Trial remains engine_running with its accepted receipt still
        # current; resume must retry reconciliation of that receipt instead
        # of requiring a second failed Engine Attempt.
        pending_p000 = next(
            (
                trial
                for trial in suspended.trials
                if trial.kind is TrialKind.BOOTSTRAP_BASELINE
                and trial.phase is TrialPhase.ENGINE_RUNNING
                and trial.command_id is not None
            ),
            None,
        )
        if pending_p000 is not None:
            return replace(
                rewound,
                agent_sessions=reopened_agent_sessions,
                bootstrap=replace(
                    rewound.bootstrap,
                    status=BootstrapStatus.P000_TRAINING,
                ),
            )

        # A source repair can be requested after Analyzer has archived a
        # failed P000 trial.  Preserve the failure evidence, but reopen the
        # logical Engine workload for a new physical Attempt; otherwise
        # resume would only replay already-completed analysis and suspend the
        # Run again without exercising the repaired worker.
        failed_p000 = next(
            (
                trial
                for trial in suspended.trials
                if trial.kind is TrialKind.BOOTSTRAP_BASELINE
                and trial.outcome is TrialOutcome.FAILED
                and trial.failure_kind == "engine_failed"
            ),
            None,
        )
        if failed_p000 is None or failed_p000.artifact_ref_id is None:
            raise InvalidOutcomeError(
                "failed Bootstrap cannot resume without a failed Engine Attempt"
            )
        reopened = replace(
            failed_p000,
            phase=TrialPhase.ENGINE_RUNNING,
            command_id=None,
            engine_retry_pending=True,
            result_refs=(),
            package_ref_id=None,
            authorized_unit_ids=(),
            analysis_status=None,
            analysis_design_ref_id=None,
            analysis_review_command_id=None,
            analysis_review_logical_command_id=None,
            analysis_review_attempt_id=None,
            analysis_review_attempt_index=0,
            analysis_review_retry_pending=False,
            analysis_review_packet_ref_id=None,
            analysis_ref_id=None,
            analysis_evidence_ref_id=None,
            analysis_review_coverage_ref_id=None,
            analysis_failure_ref_id=None,
            outcome=TrialOutcome.PENDING,
            archive_status=TrialArchiveStatus.PENDING,
            failure_kind=None,
            offline_score=None,
            analysis_packet_ref_id=None,
            plan_snapshot_ref_id=None,
            run_snapshot_ref_id=None,
            plan_memory_result=None,
            run_memory_result=None,
        )
        return replace(
            rewound,
            agent_sessions=reopened_agent_sessions,
            trials=tuple(
                reopened
                if trial.trial_id == failed_p000.trial_id
                and trial.coordinator_id == failed_p000.coordinator_id
                and trial.plan_id == failed_p000.plan_id
                else trial
                for trial in rewound.trials
            ),
            plans=tuple(
                replace(
                    plan,
                    status=PlanStatus.ACTIVE,
                    best_trial_id=None,
                    best_artifact_ref_id=None,
                    best_score=None,
                    no_improvement_count=0,
                    latest_snapshot_ref_id=None,
                )
                if plan.plan_id == failed_p000.plan_id
                and plan.coordinator_id == failed_p000.coordinator_id
                else plan
                for plan in rewound.plans
            ),
        )

    @staticmethod
    def _rewind_failed_search_engine_trial(
        suspended: RunState,
        resumed: RunState,
    ) -> RunState:
        """Reopen an archived Search Trial whose Engine Attempt failed.

        A retryable Engine failure must be repaired by re-running the frozen
        logical workload, not by accepting the subsequent failure analysis as
        scientific completion. This path is intentionally limited to an
        archived failed Search Trial with an accepted artifact.
        """
        failed = next(
            (
                trial
                for trial in suspended.trials
                if trial.kind is TrialKind.SEARCH
                and trial.outcome is TrialOutcome.FAILED
                and trial.failure_kind == "engine_failed"
                and trial.artifact_ref_id is not None
                and trial.phase is TrialPhase.ARCHIVED
                and trial.archive_status is TrialArchiveStatus.ARCHIVED
            ),
            None,
        )
        if failed is None:
            return resumed
        reopened = replace(
            failed,
            phase=TrialPhase.ENGINE_RUNNING,
            command_id=None,
            engine_retry_pending=True,
            result_refs=(),
            package_ref_id=None,
            authorized_unit_ids=(),
            analysis_status=None,
            analysis_design_ref_id=None,
            analysis_review_command_id=None,
            analysis_review_logical_command_id=None,
            analysis_review_attempt_id=None,
            analysis_review_attempt_index=0,
            analysis_review_retry_pending=False,
            analysis_review_packet_ref_id=None,
            analysis_ref_id=None,
            analysis_evidence_ref_id=None,
            analysis_review_coverage_ref_id=None,
            analysis_failure_ref_id=None,
            outcome=TrialOutcome.PENDING,
            archive_status=TrialArchiveStatus.PENDING,
            failure_kind=None,
            offline_score=None,
            analysis_packet_ref_id=None,
            plan_snapshot_ref_id=None,
            run_snapshot_ref_id=None,
            plan_memory_result=None,
            run_memory_result=None,
        )
        return replace(
            resumed,
            trials=tuple(
                reopened
                if trial.trial_id == failed.trial_id
                and trial.coordinator_id == failed.coordinator_id
                and trial.plan_id == failed.plan_id
                else trial
                for trial in resumed.trials
            ),
            plans=tuple(
                replace(
                    plan,
                    status=PlanStatus.ACTIVE,
                    best_trial_id=None,
                    best_artifact_ref_id=None,
                    best_score=None,
                    no_improvement_count=0,
                    latest_snapshot_ref_id=None,
                )
                if plan.plan_id == failed.plan_id
                and plan.coordinator_id == failed.coordinator_id
                else plan
                for plan in resumed.plans
            ),
        )

    @staticmethod
    def _rewind_suspended_agent_work(
        suspended: RunState,
        resumed: RunState,
    ) -> RunState:
        failure = suspended.failure
        if failure is None:
            return resumed
        if failure.code == "comparator_binding_integrity_failure":
            role = "coordinator"
        elif failure.code.endswith("_exhausted"):
            role = failure.code.removesuffix("_exhausted")
        else:
            return resumed
        if role not in {
            "coordinator",
            "artifact_builder",
            "analyzer",
            "plan_summarizer",
            "run_summarizer",
        }:
            return resumed
        if role == "coordinator":
            matches = tuple(
                item
                for item in resumed.planning_queue
                if item.status == "planning"
            )
            if len(matches) != 1:
                raise InvalidOutcomeError(
                    "Coordinator suspension lacks one planning reservation"
                )
            target = matches[0]
            return replace(
                resumed,
                planning_queue=tuple(
                    replace(item, status="queued") if item is target else item
                    for item in resumed.planning_queue
                ),
            )
        rewind = {
            "artifact_builder": {
                TrialPhase.BUILDING_ARTIFACT: TrialPhase.CREATED,
            },
            "analyzer": {
                TrialPhase.ANALYSIS_DESIGNING: TrialPhase.EVIDENCE_READY,
                TrialPhase.ANALYZING: TrialPhase.REVIEW_READY,
            },
            "plan_summarizer": {
                TrialPhase.PLAN_SUMMARIZING: TrialPhase.ANALYSIS_READY,
            },
            "run_summarizer": {
                TrialPhase.RUN_SUMMARIZING: TrialPhase.PLAN_SUMMARY_READY,
            },
        }[role]
        matches = tuple(
            trial for trial in resumed.trials if trial.phase in rewind
        )
        if len(matches) != 1:
            raise InvalidOutcomeError(
                f"{role} suspension lacks one resumable Trial"
            )
        target = matches[0]
        target_phase = rewind[target.phase]
        return replace(
            resumed,
            trials=tuple(
                replace(item, phase=target_phase)
                if item.coordinator_id == target.coordinator_id
                and item.plan_id == target.plan_id
                and item.trial_id == target.trial_id
                else item
                for item in resumed.trials
            ),
        )

    def _replace_trial(
        self,
        state: RunState,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        *,
        allowed: set[TrialPhase],
        **changes: object,
    ) -> RunState:
        matches = [
            trial
            for trial in state.trials
            if trial.coordinator_id == coordinator_id
            and trial.plan_id == plan_id
            and trial.trial_id == trial_id
        ]
        if len(matches) != 1:
            raise InvalidOutcomeError(f"unknown or duplicate trial_id: {trial_id}")
        current = matches[0]
        if current.phase not in allowed:
            raise InvalidOutcomeError(
                f"trial {trial_id} cannot transition from {current.phase}"
            )
        trials = tuple(
            replace(trial, **changes)
            if trial.coordinator_id == coordinator_id
            and trial.plan_id == plan_id
            and trial.trial_id == trial_id
            else trial
            for trial in state.trials
        )
        return state.next_revision(trials=trials)

    @staticmethod
    def _command_ref(outcome: EngineQueuedOutcome):
        from ade.core.run import EngineCommandRef

        return EngineCommandRef(
            command_id=outcome.command_id,
            logical_command_id=(
                outcome.logical_command_id or outcome.command_id
            ),
            attempt_id=outcome.attempt_id,
            attempt_index=outcome.attempt_index,
            coordinator_id=outcome.coordinator_id,
            plan_id=outcome.plan_id,
            trial_id=outcome.trial_id,
            kind=outcome.command_kind,
            submitted_at=outcome.submitted_at,
            last_heartbeat_at=outcome.submitted_at,
            liveness_deadline=outcome.liveness_deadline,
        )

    @staticmethod
    def _search_trial_count(state: RunState) -> int:
        return sum(trial.kind is TrialKind.SEARCH for trial in state.trials)

    @staticmethod
    def _search_plan_count(state: RunState) -> int:
        return sum(plan.kind is PlanKind.SEARCH for plan in state.plans)

    def cancel(self, state: RunState, *, reason: str) -> RunState:
        if state.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            raise InvalidOutcomeError(f"cannot cancel run in status {state.status}")
        return state.next_revision(
            status=RunStatus.CANCELLED,
            continuation_status=None,
            recovery=None,
            failure=FailureState("operator_cancelled", reason),
        )
