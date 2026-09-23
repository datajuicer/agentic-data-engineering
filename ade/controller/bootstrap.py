"""Restart-safe execution of the Agent Search bootstrap contract."""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Mapping

from ade.agent_runtime.runtime import RejectedCall
from ade.agent_runtime.service import AgentCallService
from ade.controller.ports import EngineCommandPort, EngineObjectPort, RunRepository
from ade.controller.operator_evaluation import TrialOperatorEvaluationDriver
from ade.controller.state import StateCoordinator
from ade.controller.trial_lifecycle import TrialLifecycle
from ade.core.agent import AgentRole
from ade.core.failures import FailureState
from ade.core.bootstrap import (
    BaseEvaluationRecord,
    BaseEvaluationStatus,
    BootstrapStatus,
)
from ade.core.engine import EngineReceiptStatus, EvaluateCommand
from ade.core.outcomes import (
    AgentCallSubmittedOutcome,
    AgentRetryHeldOutcome,
    AgentRetrySubmittedOutcome,
    BootstrapBaseAttemptRetryPendingOutcome,
    BootstrapBaseAttemptSubmittedOutcome,
    BootstrapBaseScheduledOutcome,
    BootstrapBaseTerminalOutcome,
    BootstrapCompletedOutcome,
    BootstrapFailedOutcome,
    BootstrapP000RegisteredOutcome,
    BootstrapStageAdvancedOutcome,
    RunSuspendedOutcome,
)
from ade.core.run import EngineCommandRef, RunState, RunStatus
from ade.core.scope import TrialKey, agent_action_id
from ade.core.trial import TrialKind, TrialOutcome, TrialPhase
from ade.tasks.contracts import AnalysisAdmissionError, SummaryAdmissionError


class BaseEvaluationDriver:
    """Submit immutable reference-model evaluations idempotently."""

    def __init__(
        self,
        *,
        queue: EngineCommandPort,
        objects: EngineObjectPort,
        contract: Mapping[str, object],
    ) -> None:
        self.queue = queue
        self.objects = objects
        self.contract = contract

    def commands(self, run_id: str) -> tuple[EvaluateCommand, ...]:
        base = self._mapping(self.contract["base"])
        profiles = self._mapping(base["evaluation_profiles"])
        if "offline" not in profiles:
            raise ValueError(
                "bootstrap base evaluation requires an offline profile"
            )
        base_model_protocol = self._mapping(base["model_protocol"])
        base_thinking_protocol = self._mapping(base_model_protocol["thinking"])
        base_reasoning_parser = base_thinking_protocol["reasoning_parser"]
        result: list[EvaluateCommand] = []
        requested_profiles = [("offline_validation", "offline")]
        if "online" in profiles:
            requested_profiles.append(("online_validation", "online"))
        for purpose, profile_name in requested_profiles:
            profile = self._mapping(profiles[profile_name])
            trial_id = f"base-{profile_name}"
            logical_command_id = (
                f"{run_id}-c000-p000-{trial_id}-evaluate-{profile_name}"
            )
            attempt_id = "attempt-001"
            command_id = f"{logical_command_id}-{attempt_id}"
            input_ref = f"engine://inputs/{command_id}.json"
            request = {
                **dict(self._mapping(profile["request"])),
                "model_protocol": dict(base_model_protocol),
                "reasoning_parser": (
                    ""
                    if base_reasoning_parser == "none"
                    else base_reasoning_parser
                ),
                "thinking_budget": -1,
                "model": base["model"],
                "checkpoint": base["model"],
                "coordinator_resource_owner": f"{run_id}/c000",
                "ade_run_id": run_id,
                "ade_coordinator_id": "c000",
                "ade_plan_id": "p000",
                "ade_trial_id": trial_id,
                "evaluation_subject_kind": "base_model",
                "ade_engine_command_id": command_id,
                "ade_workload": purpose,
            }
            if purpose == "online_validation":
                request["artifact_position"] = {"unit": "rl_step", "value": 0}
            self.objects.put_json(
                input_ref,
                {
                    "schema_version": 1,
                    "evaluation": {
                        "purpose": purpose,
                        "request": request,
                    },
                },
            )
            result.append(
                EvaluateCommand(
                    command_id=command_id,
                    run_id=run_id,
                    coordinator_id="c000",
                    plan_id="p000",
                    trial_id=trial_id,
                    input_ref=input_ref,
                    output_uri=f"engine://outputs/{command_id}",
                    logical_command_id=logical_command_id,
                    attempt_id=attempt_id,
                    attempt_index=1,
                )
            )
        return tuple(result)

    @staticmethod
    def _mapping(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            raise ValueError("bootstrap contract field must be a mapping")
        return value


class BootstrapWorkflowDriver:
    """Advance bootstrap boundaries and leave Trial execution to TrialLifecycle."""

    def __init__(
        self,
        *,
        repository: RunRepository,
        queue: EngineCommandPort,
        objects: EngineObjectPort,
        contract: Mapping[str, object],
        engine_inputs: Mapping[str, dict[str, object]],
        trials: TrialLifecycle,
        calls: AgentCallService,
        operator_evaluation: TrialOperatorEvaluationDriver | None = None,
    ) -> None:
        self.repository = repository
        self.queue = queue
        self.objects = objects
        self.contract = dict(contract)
        self.engine_inputs = engine_inputs
        self.trials = trials
        self.calls = calls
        self.operator_evaluation = operator_evaluation
        self.state = StateCoordinator(repository)
        self.base = BaseEvaluationDriver(queue=queue, objects=objects, contract=contract)

    def tick(self, run_id: str) -> RunState:
        state = self.repository.load(run_id)
        bootstrap = state.bootstrap
        if not bootstrap.enabled or bootstrap.status is BootstrapStatus.COMPLETED:
            return state
        try:
            if bootstrap.status is BootstrapStatus.PENDING:
                commands = self.base.commands(run_id)
                for command in commands:
                    self.repository.store_engine_command(command)
                committed = self.state.apply(
                    run_id,
                    BootstrapBaseScheduledOutcome(
                        run_id=run_id,
                        basis_revision=state.revision,
                        commands=tuple(
                            EngineCommandRef(
                                command_id=command.command_id,
                                logical_command_id=command.logical_command_id,
                                attempt_id=command.attempt_id,
                                attempt_index=command.attempt_index,
                                coordinator_id=command.coordinator_id,
                                plan_id=command.plan_id,
                                trial_id=command.trial_id,
                                kind=command.kind,
                            )
                            for command in commands
                        ),
                    ),
                    event_type="bootstrap_base_submitted",
                )
                for command in commands:
                    try:
                        self.queue.submit(command)
                    except OSError:
                        return committed
                return committed
            if bootstrap.status is BootstrapStatus.BASE_EVALUATING:
                return self._collect_base(state)
            if bootstrap.status is BootstrapStatus.BASE_COMPLETED:
                return self._register_p000(state)
            if bootstrap.status is BootstrapStatus.P000_TRAINING:
                return self._advance_p000_training(state)
            if bootstrap.status is BootstrapStatus.P000_ANALYZING:
                return self._advance_p000_analysis(state)
            if bootstrap.status is BootstrapStatus.P000_PLAN_SUMMARIZING:
                return self._advance_p000_plan_summary(state)
            if bootstrap.status is BootstrapStatus.P000_RUN_SUMMARIZING:
                return self._advance_p000_run_summary(state)
            return state
        except Exception as error:
            failed_state = self.repository.load(run_id)
            failed_bootstrap = failed_state.bootstrap
            command_status = dict(failed_bootstrap.command_status)
            for command_id in failed_bootstrap.command_ids:
                if self.queue.has_receipt(command_id):
                    command_status[command_id] = self.queue.load_receipt(
                        command_id
                    ).status.value
            return self.state.apply(
                run_id,
                BootstrapFailedOutcome(
                    run_id=run_id,
                    basis_revision=failed_state.revision,
                    reason=str(error),
                    command_status=command_status,
                ),
                event_type="bootstrap_failed",
            )

    def _collect_base(self, state: RunState) -> RunState:
        retry_pending = next(
            (
                command_id
                for command_id in state.bootstrap.command_ids
                if state.bootstrap.command_status.get(command_id)
                == "retry_pending"
            ),
            None,
        )
        if retry_pending is not None:
            return self._retry_base(state, retry_pending)
        for command_id in state.bootstrap.command_ids:
            if not self.queue.has_receipt(command_id):
                try:
                    self.queue.submit(
                    self.repository.load_engine_command(
                        state.run_id,
                        "c000",
                        "p000",
                        next(
                            command.trial_id
                            for command in state.active_engine_commands
                            if command.command_id == command_id
                        ),
                        command_id,
                    )
                )
                except OSError:
                    return state
        available = tuple(
            self.queue.load_receipt(command_id)
            for command_id in state.bootstrap.command_ids
            if self.queue.has_receipt(command_id)
        )
        if len(available) != len(state.bootstrap.command_ids):
            return state
        for receipt in available:
            active = next(
                (
                    command
                    for command in state.active_engine_commands
                    if command.command_id == receipt.command_id
                ),
                None,
            )
            if (
                active is None
                or receipt.logical_command_id != active.logical_command_id
                or receipt.attempt_id != active.attempt_id
                or receipt.attempt_index != active.attempt_index
            ):
                raise ValueError(
                    "Base evaluation Receipt is not the current active Attempt"
                )
            self.repository.store_engine_receipt(receipt)
            if receipt.status is EngineReceiptStatus.FAILED and receipt.retryable:
                return self.state.apply(
                    state.run_id,
                    BootstrapBaseAttemptRetryPendingOutcome(
                        run_id=state.run_id,
                        basis_revision=state.revision,
                        command_id=receipt.command_id,
                        logical_command_id=receipt.logical_command_id,
                        attempt_id=receipt.attempt_id,
                        attempt_index=receipt.attempt_index,
                        receipt_id=receipt.receipt_id,
                        failure_kind=(
                            receipt.failure_kind or "engine_attempt_failed"
                        ),
                        message=receipt.error or "Base evaluation Attempt failed",
                    ),
                    event_type=(
                        "bootstrap_base_attempt_retry_pending"
                        if state.pause_requested
                        else "automatic_recovery_started"
                    ),
                )
        result_refs = tuple(ref for receipt in available for ref in receipt.output_refs)
        offline_receipt = next(
            receipt for receipt in available if receipt.trial_id == "base-offline"
        )
        offline_result_refs = (
            offline_receipt.output_refs[:1]
            if offline_receipt.status is EngineReceiptStatus.SUCCEEDED
            else ()
        )
        online_receipt = next(
            (receipt for receipt in available if receipt.trial_id == "base-online"),
            None,
        )
        if (
            online_receipt is not None
            and online_receipt.status is not EngineReceiptStatus.SUCCEEDED
        ):
            raise ValueError(
                "step-0 online evaluation must succeed before RFT training starts"
            )
        online_result_ref = (
            online_receipt.output_refs[0]
            if online_receipt is not None and online_receipt.output_refs
            else None
        )
        offline_evidence_ref = None
        offline_score = None
        offline_secondary_score = None
        if offline_result_refs:
            result_payload = self.objects.read_json(offline_result_refs[0])
            score = result_payload.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                candidate = float(score)
                if math.isfinite(candidate):
                    offline_score = candidate
            secondary_score = result_payload.get("secondary_score")
            if isinstance(secondary_score, (int, float)) and not isinstance(
                secondary_score, bool
            ):
                secondary_candidate = float(secondary_score)
                if math.isfinite(secondary_candidate):
                    offline_secondary_score = secondary_candidate
            offline_evidence_ref = self.repository.put_artifact(
                state.run_id,
                "base_offline_evaluation",
                self.objects.read_bytes(offline_result_refs[0]),
            )
        profile_status = {
            receipt.trial_id.removeprefix("base-"): receipt.status.value
            for receipt in available
        }
        record = BaseEvaluationRecord(
            status=(
                BaseEvaluationStatus.COMPLETE
                if all(
                    receipt.status is EngineReceiptStatus.SUCCEEDED
                    for receipt in available
                )
                else BaseEvaluationStatus.COMPLETED_DEGRADED
            ),
            profile_status=profile_status,
            receipt_ids=tuple(receipt.receipt_id for receipt in available),
            result_refs=result_refs,
            online_result_ref=online_result_ref,
            offline_evidence_ref_id=(
                offline_evidence_ref.artifact_id if offline_evidence_ref else None
            ),
            offline_score=offline_score,
            offline_secondary_score=offline_secondary_score,
        )
        return self.state.apply(
            state.run_id,
            BootstrapBaseTerminalOutcome(
                run_id=state.run_id,
                basis_revision=state.revision,
                command_status={
                    receipt.command_id: receipt.status.value
                    for receipt in available
                },
                record=record,
                evidence_ref=offline_evidence_ref,
            ),
            event_type="bootstrap_base_completed",
        )

    def _retry_base(self, state: RunState, previous_command_id: str) -> RunState:
        previous = self.repository.load_engine_command(
            state.run_id,
            "c000",
            "p000",
            self._base_trial_id(state, previous_command_id),
            previous_command_id,
        )
        attempt_index = previous.attempt_index + 1
        attempt_id = f"attempt-{attempt_index:03d}"
        logical_command_id = previous.logical_command_id or previous.command_id
        command_id = f"{logical_command_id}-{attempt_id}"
        input_ref = f"engine://inputs/{command_id}.json"
        payload = copy.deepcopy(self.objects.read_json(previous.input_ref))
        request = self._mapping(self._mapping(payload["evaluation"])["request"])
        request["ade_engine_command_id"] = command_id
        command = EvaluateCommand(
            command_id=command_id,
            run_id=previous.run_id,
            coordinator_id=previous.coordinator_id,
            plan_id=previous.plan_id,
            trial_id=previous.trial_id,
            input_ref=input_ref,
            output_uri=f"engine://outputs/{command_id}",
            logical_command_id=logical_command_id,
            attempt_id=attempt_id,
            attempt_index=attempt_index,
        )
        self.objects.put_json(input_ref, payload)
        self.repository.store_engine_command(command)
        committed = self.state.apply(
            state.run_id,
            BootstrapBaseAttemptSubmittedOutcome(
                run_id=state.run_id,
                basis_revision=state.revision,
                previous_command_id=previous_command_id,
                command=EngineCommandRef(
                    command_id=command.command_id,
                    logical_command_id=command.logical_command_id,
                    attempt_id=command.attempt_id,
                    attempt_index=command.attempt_index,
                    coordinator_id=command.coordinator_id,
                    plan_id=command.plan_id,
                    trial_id=command.trial_id,
                    kind=command.kind,
                ),
            ),
            event_type=(
                "automatic_recovery_attempt_submitted"
                if state.status is RunStatus.RECOVERING
                else "bootstrap_base_attempt_submitted"
            ),
        )
        try:
            self.queue.submit(command)
        except OSError:
            pass
        return committed

    @staticmethod
    def _base_trial_id(state: RunState, command_id: str) -> str:
        active = next(
            (
                command.trial_id
                for command in state.active_engine_commands
                if command.command_id == command_id
            ),
            None,
        )
        if active is not None:
            return active
        logical = state.bootstrap.command_logical_ids.get(command_id, command_id)
        for profile in ("offline", "online"):
            if f"-base-{profile}-" in logical:
                return f"base-{profile}"
        raise ValueError("Base evaluation Command has no known profile identity")

    def _register_p000(self, state: RunState) -> RunState:
        p000 = self._mapping(self.contract["p000"])
        artifact = self._mapping(p000["artifact"])
        trial_id = self._required_text(p000, "trial_id")
        coordinator_id = self._required_text(p000, "coordinator_id")
        plan_id = self._required_text(p000, "plan_id")
        trial_key = TrialKey(state.run_id, coordinator_id, plan_id, trial_id)
        kind = self._required_text(artifact, "kind")
        path = self._required_text(artifact, "path")
        content = self._required_text(artifact, "content").encode()
        digest = self._required_text(artifact, "digest")
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("p000 artifact digest mismatch")
        if any(
            trial.coordinator_id == coordinator_id
            and trial.plan_id == plan_id
            and trial.trial_id == trial_id
            for trial in state.trials
        ):
            raise ValueError("p000 Trial is already registered")
        artifact_ref = self.repository.put_artifact(state.run_id, kind, content)
        report = self.trials.tasks.get(state.task.task_id).validate_artifact(
            artifact_ref
        )
        if not report.ok:
            raise ValueError(report.violations[0].message)
        return self.state.apply(
            state.run_id,
            BootstrapP000RegisteredOutcome(
                run_id=state.run_id,
                basis_revision=state.revision,
                coordinator_id=trial_key.coordinator_id,
                plan_id=trial_key.plan_id,
                trial_id=trial_key.trial_id,
                artifact_ref=artifact_ref,
                artifact_path=path,
            ),
            event_type="bootstrap_p000_registered",
        )

    def _advance_p000_training(self, state: RunState) -> RunState:
        trial = self._p000_trial(state)
        if trial.phase is TrialPhase.ARTIFACT_READY:
            engine_config = self.engine_inputs.get(state.task.task_id)
            if engine_config is None:
                raise ValueError(f"missing Engine input for task {state.task.task_id}")
            command_id = (
                f"{state.run_id}-{trial.coordinator_id}-{trial.plan_id}-"
                f"{trial.trial_id}-"
                f"{self.trials.tasks.get(state.task.task_id).engine_command_kind}"
            )
            command = self.trials.submit_engine(
                state.run_id,
                self._trial_key(state, trial),
                dict(engine_config),
                command_id=command_id,
            )
            return self.repository.load(command.run_id)
        if trial.phase is TrialPhase.ENGINE_RUNNING:
            if trial.engine_retry_pending:
                engine_config = self.engine_inputs.get(state.task.task_id)
                if engine_config is None:
                    raise ValueError(
                        f"missing Engine input for task {state.task.task_id}"
                    )
                self.trials.submit_engine(
                    state.run_id,
                    self._trial_key(state, trial),
                    dict(engine_config),
                )
                return self.repository.load(state.run_id)
            if not trial.command_id:
                raise ValueError("p000 queued Trial has no command_id")
            self.queue.submit(
                self.repository.load_engine_command(
                    state.run_id,
                    trial.coordinator_id,
                    trial.plan_id,
                    trial.trial_id,
                    trial.command_id,
                )
            )
            self.queue.expire_stale()
            if not self.queue.has_receipt(trial.command_id):
                return state
            self.trials.collect_engine(state.run_id, trial.command_id)
            return self.repository.load(state.run_id)
        if trial.phase is TrialPhase.EVIDENCE_READY:
            return self.state.apply(
                state.run_id,
                BootstrapStageAdvancedOutcome(
                    run_id=state.run_id,
                    basis_revision=state.revision,
                    from_status=BootstrapStatus.P000_TRAINING,
                    to_status=BootstrapStatus.P000_ANALYZING,
                ),
                event_type="bootstrap_p000_training_completed",
            )
        if trial.phase is TrialPhase.ARCHIVED:
            return self._finish_p000(state, trial)
        raise ValueError(f"unsupported p000 Trial phase: {trial.phase}")

    def _advance_p000_analysis(self, state: RunState) -> RunState:
        trial = self._p000_trial(state)
        if trial.phase is TrialPhase.ARCHIVED:
            return self._finish_p000(state, trial)
        if trial.phase is TrialPhase.ANALYSIS_READY:
            return self.state.apply(
                state.run_id,
                BootstrapStageAdvancedOutcome(
                    run_id=state.run_id,
                    basis_revision=state.revision,
                    from_status=BootstrapStatus.P000_ANALYZING,
                    to_status=BootstrapStatus.P000_PLAN_SUMMARIZING,
                ),
                event_type="bootstrap_p000_analysis_completed",
            )
        if trial.phase is TrialPhase.EVIDENCE_READY:
            active = self.calls.prepare(
                run_id=state.run_id,
                role=AgentRole.ANALYZER,
                subject_id=trial.trial_id,
                basis_revision=state.revision,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                action_id=agent_action_id(
                    owner_subject_ref=self._trial_key(state, trial).plan.subject_ref,
                    target_subject_ref=self._trial_key(state, trial).subject_ref,
                    role_segment="analyzer-review-design",
                    basis_revision=state.revision,
                ),
                target_subject_ref=self._trial_key(state, trial).subject_ref,
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
                    state.run_id,
                    self._trial_key(state, trial),
                    result,
                    self._rejection_reason(result),
                )
            try:
                return self.trials.submit_analysis_review(
                    state.run_id, self._trial_key(state, trial), result
                )
            except AnalysisAdmissionError as error:
                retried = self._retry_agent(state, result, error.report)
                if retried is not None:
                    return retried
                return self.trials.fail_analysis(
                    state.run_id,
                    self._trial_key(state, trial),
                    result,
                    "; ".join(item.message for item in error.report.violations),
                )
        if trial.phase is TrialPhase.REVIEW_RUNNING:
            if trial.analysis_review_retry_pending:
                return self.trials.retry_analysis_review(
                    state.run_id, self._trial_key(state, trial)
                )
            command_id = trial.analysis_review_command_id
            if not command_id or self.trials.review_queue is None:
                raise ValueError("p000 Review queue/command is unavailable")
            command = self.repository.load_review_command(
                state.run_id,
                trial.coordinator_id,
                trial.plan_id,
                trial.trial_id,
                command_id,
            )
            self.trials.review_queue.submit(command)
            if not self.trials.review_queue.has_receipt(command_id):
                return state
            return self.trials.collect_analysis_review(state.run_id, command_id)
        if trial.phase is TrialPhase.REVIEW_READY:
            active = self.calls.prepare(
                run_id=state.run_id,
                role=AgentRole.ANALYZER,
                subject_id=trial.trial_id,
                basis_revision=state.revision,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                action_id=agent_action_id(
                    owner_subject_ref=self._trial_key(state, trial).plan.subject_ref,
                    target_subject_ref=self._trial_key(state, trial).subject_ref,
                    role_segment="analyzer-synthesis",
                    basis_revision=state.revision,
                ),
                target_subject_ref=self._trial_key(state, trial).subject_ref,
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
                    state.run_id,
                    self._trial_key(state, trial),
                    result,
                    self._rejection_reason(result),
                )
            try:
                return self.trials.accept_analysis(
                    state.run_id, self._trial_key(state, trial), result
                )
            except AnalysisAdmissionError as error:
                retried = self._retry_agent(state, result, error.report)
                if retried is not None:
                    return retried
                return self.trials.fail_analysis(
                    state.run_id,
                    self._trial_key(state, trial),
                    result,
                    "; ".join(item.message for item in error.report.violations),
                )
        raise ValueError(f"p000 is not ready for analysis: {trial.phase}")

    def _advance_p000_plan_summary(self, state: RunState) -> RunState:
        trial = self._p000_trial(state)
        if trial.phase is TrialPhase.PLAN_SUMMARY_READY:
            return self.state.apply(
                state.run_id,
                BootstrapStageAdvancedOutcome(
                    run_id=state.run_id,
                    basis_revision=state.revision,
                    from_status=BootstrapStatus.P000_PLAN_SUMMARIZING,
                    to_status=BootstrapStatus.P000_RUN_SUMMARIZING,
                ),
                event_type="bootstrap_p000_plan_summarized",
            )
        if trial.phase is TrialPhase.ANALYSIS_READY:
            active = self.calls.prepare(
                run_id=state.run_id,
                role=AgentRole.PLAN_SUMMARIZER,
                subject_id=trial.plan_id,
                basis_revision=state.revision,
                coordinator_id=trial.coordinator_id,
                action_id=agent_action_id(
                    owner_subject_ref=self._trial_key(state, trial).plan.subject_ref,
                    target_subject_ref=self._trial_key(state, trial).subject_ref,
                    role_segment="plan-summarizer",
                    basis_revision=state.revision,
                ),
                target_subject_ref=self._trial_key(state, trial).subject_ref,
            )
            return self._submit_trial_call(state, trial, active)
        if trial.phase is not TrialPhase.PLAN_SUMMARIZING:
            raise ValueError(f"p000 is not ready for Plan summary: {trial.phase}")
        delivery = self._execute_active(state, trial, AgentRole.PLAN_SUMMARIZER)
        if delivery is None:
            return state
        if isinstance(delivery, RejectedCall):
            retried = self._retry_agent(state, delivery, delivery.validation)
            if retried is not None:
                return retried
            return self.trials.default_summary(
                state.run_id, delivery, self._rejection_reason(delivery)
            )
        return self._accept_summary(state.run_id, delivery)

    def _advance_p000_run_summary(self, state: RunState) -> RunState:
        trial = self._p000_trial(state)
        if trial.phase is TrialPhase.ARCHIVED:
            return self._finish_p000(state, trial)
        if trial.phase is TrialPhase.PLAN_SUMMARY_READY:
            active = self.calls.prepare(
                run_id=state.run_id,
                role=AgentRole.RUN_SUMMARIZER,
                subject_id=state.run_id,
                basis_revision=state.revision,
                action_id=agent_action_id(
                    owner_subject_ref=state.run_id,
                    target_subject_ref=self._trial_key(state, trial).subject_ref,
                    role_segment="run-summarizer",
                    basis_revision=state.revision,
                ),
                target_subject_ref=self._trial_key(state, trial).subject_ref,
            )
            return self._submit_trial_call(state, trial, active)
        if trial.phase is not TrialPhase.RUN_SUMMARIZING:
            raise ValueError(f"p000 is not ready for Run summary: {trial.phase}")
        delivery = self._execute_active(state, trial, AgentRole.RUN_SUMMARIZER)
        if delivery is None:
            return state
        if isinstance(delivery, RejectedCall):
            retried = self._retry_agent(state, delivery, delivery.validation)
            if retried is not None:
                return retried
            return self.trials.default_summary(
                state.run_id, delivery, self._rejection_reason(delivery)
            )
        return self._accept_summary(state.run_id, delivery)

    def _accept_summary(self, run_id: str, delivery):
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
                "analyzer": (
                    "analysis_design_submitted"
                    if active.action_fields.get("stage") == "review_design"
                    else "analysis_synthesis_submitted"
                ),
                "plan_summarizer": "plan_summarizer_submitted",
                "run_summarizer": "run_summarizer_submitted",
            }[active.role],
        )

    @staticmethod
    def _active_call(state, trial, role: AgentRole):
        matches = tuple(
            item
            for item in state.active_agent_calls
            if item.role == role.value
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
            raise ValueError(f"p000 requires one active {role.value} Call")
        return matches[0]

    def _execute_active(self, state, trial, role: AgentRole):
        active = self._active_call(state, trial, role)
        if self.calls.is_expired(active):
            return self.calls.expire_active(active)
        return self.calls.execute_active(active)

    def _retry_agent(self, state, previous, report) -> RunState | None:
        active = next(
            (
                item for item in state.active_agent_calls
                if item.call_id == previous.call.call_id
            ),
            None,
        )
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
                return self.state.apply(
                    state.run_id,
                        RunSuspendedOutcome(
                            run_id=state.run_id,
                            subject_ref=active.target_subject_ref,
                        basis_revision=state.revision,
                        failure=FailureState(
                            f"{previous.call.role.value}_exhausted",
                            "; ".join(
                                item.message for item in report.violations
                            ),
                            retryable=True,
                        ),
                        agent_session=previous.session,
                        call_id=previous.call.call_id,
                    ),
                    event_type="run_suspended_for_agent",
                )
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

    def _finish_p000(self, state: RunState, trial) -> RunState:
        if trial.phase is not TrialPhase.ARCHIVED:
            raise ValueError("p000 cannot complete before Trial archive")
        if trial.outcome is not TrialOutcome.SUCCEEDED:
            return self.state.apply(
                state.run_id,
                BootstrapFailedOutcome(
                    run_id=state.run_id,
                    basis_revision=state.revision,
                    reason=(
                        "p000 bootstrap Trial did not succeed: "
                        f"{trial.failure_kind or trial.outcome.value}"
                    ),
                    command_status=dict(state.bootstrap.command_status),
                ),
                event_type="bootstrap_failed",
            )
        if not state.bootstrap.stop_after_baseline and not any(
            item.evaluation_profile == "offline"
            and (
                item.level == "reference"
                or (item.level == "trial_level" and item.source_eligible)
            )
            for item in state.ranking.entries
        ):
            raise ValueError(
                "bootstrap cannot enter Search: frozen offline Ranking has no "
                "reference or source-eligible Trial with a finite primary score"
            )
        return self.state.apply(
            state.run_id,
            BootstrapCompletedOutcome(
                run_id=state.run_id,
                basis_revision=state.revision,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                trial_id=trial.trial_id,
            ),
            event_type="bootstrap_completed",
        )

    def _p000_trial(self, state: RunState):
        p000 = self._mapping(self.contract["p000"])
        coordinator_id = self._required_text(p000, "coordinator_id")
        plan_id = self._required_text(p000, "plan_id")
        trial_id = self._required_text(p000, "trial_id")
        matches = [
            trial
            for trial in state.trials
            if trial.coordinator_id == coordinator_id
            and trial.plan_id == plan_id
            and trial.trial_id == trial_id
        ]
        if len(matches) != 1:
            raise ValueError("p000 bootstrap Trial is unavailable")
        if matches[0].kind is not TrialKind.BOOTSTRAP_BASELINE:
            raise ValueError("p000 Trial has the wrong kind")
        return matches[0]

    @staticmethod
    def _trial_key(state: RunState, trial) -> TrialKey:
        return TrialKey(
            state.run_id,
            trial.coordinator_id,
            trial.plan_id,
            trial.trial_id,
        )

    @staticmethod
    def _mapping(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            raise ValueError("bootstrap contract field must be a mapping")
        return value

    @staticmethod
    def _required_text(value: Mapping[str, object], field: str) -> str:
        result = value.get(field)
        if not isinstance(result, str) or not result:
            raise ValueError(f"p000 {field} is required")
        return result

    @staticmethod
    def _rejection_reason(delivery: RejectedCall) -> str:
        messages = [item.message for item in delivery.validation.violations]
        return "; ".join(messages) if messages else "Analyzer delivery rejected"
