"""Non-blocking human-only evaluation for Base and every completed Trial."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

from ade.controller.ports import EngineCommandPort, EngineObjectPort, RunRepository
from ade.controller.state import StateCoordinator
from ade.core.engine import EngineReceiptStatus, EvaluateCommand
from ade.core.operator import OperatorEvaluationRecord, OperatorEvaluationStatus
from ade.core.outcomes import (
    OperatorEvaluationAttemptRetryPendingOutcome,
    OperatorEvaluationAttemptSubmittedOutcome,
    OperatorEvaluationScheduledOutcome,
    OperatorEvaluationTerminalOutcome,
)
from ade.core.run import EngineCommandRef, RunState, RunStatus
from ade.core.scope import subject_ref
from ade.core.trial import TrialKind, TrialOutcome, TrialPhase


class TrialOperatorEvaluationDriver:
    def __init__(
        self,
        *,
        repository: RunRepository,
        queue: EngineCommandPort,
        objects: EngineObjectPort,
        request: Mapping[str, object],
        base_model: str,
        base_model_protocol: Mapping[str, object],
    ) -> None:
        self.repository = repository
        self.queue = queue
        self.objects = objects
        self.request = copy.deepcopy(dict(request))
        self.base_model = base_model
        self.base_model_protocol = copy.deepcopy(dict(base_model_protocol))
        self.profile_digest = hashlib.sha256(
            json.dumps(
                self.request,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.base_profile_digest = hashlib.sha256(
            json.dumps(
                self._request_for_subject("base_model"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.state = StateCoordinator(repository)

    def reconcile(self, run_id: str, *, allow_submit: bool = True) -> RunState:
        state = self.repository.load(run_id)
        pending_items = tuple(
            item
            for item in state.operator_evaluations
            if item.status is OperatorEvaluationStatus.PENDING
        )
        pending = next(
            (
                item for item in pending_items
                if item.command_id is not None
                and self.queue.has_receipt(item.command_id)
            ),
            None,
        )
        if pending is not None:
            assert pending.command_id is not None
            receipt = self.queue.load_receipt(pending.command_id)
            if (
                receipt.logical_command_id != pending.logical_command_id
                or receipt.attempt_id != pending.attempt_id
                or receipt.attempt_index != pending.attempt_index
            ):
                raise ValueError(
                    "Operator evaluation Receipt is not the current active Attempt"
                )
            if (
                state.status is RunStatus.RECOVERING
                and receipt.status is EngineReceiptStatus.FAILED
                and receipt.retryable
            ):
                return state
            self.repository.store_engine_receipt(receipt)
            if receipt.status is EngineReceiptStatus.FAILED and receipt.retryable:
                return self.state.apply(
                    run_id,
                    OperatorEvaluationAttemptRetryPendingOutcome(
                        run_id=run_id,
                        basis_revision=state.revision,
                        target_id=pending.target_id,
                        command_id=receipt.command_id,
                        logical_command_id=receipt.logical_command_id,
                        attempt_id=receipt.attempt_id,
                        attempt_index=receipt.attempt_index,
                        receipt_id=receipt.receipt_id,
                        failure_kind=(
                            receipt.failure_kind or "engine_attempt_failed"
                        ),
                        message=(
                            receipt.error or "Operator evaluation Attempt failed"
                        ),
                    ),
                    event_type=(
                        "operator_evaluation_attempt_retry_pending"
                        if state.pause_requested
                        else "automatic_recovery_started"
                    ),
                )
            succeeded = receipt.status is EngineReceiptStatus.SUCCEEDED
            return self.state.apply(
                run_id,
                OperatorEvaluationTerminalOutcome(
                    run_id=run_id,
                    basis_revision=state.revision,
                    target_id=pending.target_id,
                    command_id=pending.command_id,
                    logical_command_id=receipt.logical_command_id,
                    attempt_id=receipt.attempt_id,
                    attempt_index=receipt.attempt_index,
                    receipt_id=receipt.receipt_id,
                    status=(
                        OperatorEvaluationStatus.COMPLETED
                        if succeeded
                        else OperatorEvaluationStatus.FAILED
                    ),
                    result_ref=(receipt.output_refs[0] if succeeded and receipt.output_refs else None),
                    error=None if succeeded else receipt.error or "operator evaluation failed",
                ),
                event_type="operator_evaluation_terminal",
            )

        retry_pending = next(
            (item for item in pending_items if item.retry_pending),
            None,
        )
        if retry_pending is not None:
            return (
                self._retry(state, retry_pending)
                if allow_submit
                else state
            )

        if not allow_submit:
            return state

        known = {item.target_id for item in state.operator_evaluations}
        base_target = subject_ref(run_id, "c000", "p000", "base")
        if state.bootstrap.base_evaluation is not None and base_target not in known:
            base = state.bootstrap.base_evaluation
            evidence = self._accepted_ref(state, base.offline_evidence_ref_id)
            logical_command_id = f"{run_id}-c000-p000-base-operator-test"
            return self._schedule(
                state,
                OperatorEvaluationRecord(
                    target_id=base_target,
                    target_kind="base",
                    artifact_ref_id=(evidence.artifact_id if evidence else None),
                    artifact_digest=(evidence.digest if evidence else None),
                    profile_digest=self.base_profile_digest,
                    status=OperatorEvaluationStatus.PENDING,
                    command_id=f"{logical_command_id}-attempt-001",
                    logical_command_id=logical_command_id,
                    attempt_id="attempt-001",
                    attempt_index=1,
                ),
                checkpoint=self.base_model,
            )

        for item in pending_items:
            assert item.command_id is not None
            try:
                self.queue.submit(
                    self.repository.load_engine_command(
                        state.run_id,
                        item.coordinator_id or "c000",
                        item.plan_id or "p000",
                        item.trial_id or "base",
                        item.command_id,
                    )
                )
            except OSError:
                pass

        eligible_phases = {
            TrialPhase.EVIDENCE_READY,
            TrialPhase.ANALYSIS_DESIGNING,
            TrialPhase.REVIEW_RUNNING,
            TrialPhase.REVIEW_READY,
            TrialPhase.ANALYZING,
            TrialPhase.ANALYSIS_READY,
            TrialPhase.PLAN_SUMMARIZING,
            TrialPhase.PLAN_SUMMARY_READY,
            TrialPhase.RUN_SUMMARIZING,
            TrialPhase.ARCHIVED,
        }
        for trial in sorted(
            (
                item for item in state.trials
                if item.outcome is TrialOutcome.SUCCEEDED
                and item.phase in eligible_phases
            ),
            key=lambda item: (item.coordinator_id, item.plan_id, item.trial_id),
        ):
            target_id = subject_ref(
                run_id,
                trial.coordinator_id,
                trial.plan_id,
                trial.trial_id,
            )
            if target_id in known:
                continue
            artifact = self._accepted_ref(state, trial.artifact_ref_id)
            checkpoint = self._checkpoint(trial)
            if checkpoint is None:
                return self.state.apply(
                    run_id,
                    OperatorEvaluationScheduledOutcome(
                        run_id=run_id,
                        basis_revision=state.revision,
                        record=OperatorEvaluationRecord(
                            target_id=target_id,
                            target_kind="trial",
                            coordinator_id=trial.coordinator_id,
                            plan_id=trial.plan_id,
                            trial_id=trial.trial_id,
                            artifact_ref_id=trial.artifact_ref_id,
                            artifact_digest=(artifact.digest if artifact else None),
                            profile_digest=self.profile_digest,
                            status=OperatorEvaluationStatus.NOT_APPLICABLE,
                        ),
                    ),
                    event_type="operator_evaluation_not_applicable",
                )
            logical_command_id = (
                f"{run_id}-{trial.coordinator_id}-{trial.plan_id}-"
                f"{trial.trial_id}-operator-test"
            )
            return self._schedule(
                state,
                OperatorEvaluationRecord(
                    target_id=target_id,
                    target_kind="trial",
                    coordinator_id=trial.coordinator_id,
                    plan_id=trial.plan_id,
                    trial_id=trial.trial_id,
                    artifact_ref_id=trial.artifact_ref_id,
                    artifact_digest=(artifact.digest if artifact else None),
                    profile_digest=self.profile_digest,
                    status=OperatorEvaluationStatus.PENDING,
                    command_id=f"{logical_command_id}-attempt-001",
                    logical_command_id=logical_command_id,
                    attempt_id="attempt-001",
                    attempt_index=1,
                ),
                checkpoint=checkpoint,
            )
        return state

    def _schedule(
        self,
        state: RunState,
        record: OperatorEvaluationRecord,
        *,
        checkpoint: str,
    ) -> RunState:
        assert record.command_id is not None
        coordinator_id = record.coordinator_id or "c000"
        plan_id = record.plan_id or "p000"
        trial_id = record.trial_id or "base"
        input_ref = f"engine://inputs/{record.command_id}.json"
        command = EvaluateCommand(
            command_id=record.command_id,
            run_id=state.run_id,
            coordinator_id=coordinator_id,
            plan_id=plan_id,
            trial_id=trial_id,
            input_ref=input_ref,
            output_uri=f"engine://outputs/{record.command_id}",
            logical_command_id=record.logical_command_id or record.command_id,
            attempt_id=record.attempt_id or "attempt-001",
            attempt_index=record.attempt_index,
        )
        self._write_input(
            state,
            command=command,
            checkpoint=checkpoint,
            subject_kind=self._subject_kind(state, record),
        )
        self.repository.store_engine_command(command)
        committed = self.state.apply(
            state.run_id,
            OperatorEvaluationScheduledOutcome(
                run_id=state.run_id,
                basis_revision=state.revision,
                record=record,
            ),
            event_type="operator_evaluation_scheduled",
        )
        try:
            self.queue.submit(command)
        except OSError:
            pass
        return committed

    def _write_input(
        self,
        state: RunState,
        *,
        command: EvaluateCommand,
        checkpoint: str,
        subject_kind: str,
    ) -> None:
        request = self._request_for_subject(subject_kind)
        self.objects.put_json(
            command.input_ref,
            {
                "schema_version": 1,
                "evaluation": {
                    "purpose": "operator_test",
                    "request": {
                        **request,
                        "checkpoint": checkpoint,
                        "model": checkpoint,
                        "coordinator_resource_owner": (
                            f"{state.run_id}/{command.coordinator_id}"
                        ),
                        "ade_run_id": state.run_id,
                        "ade_coordinator_id": command.coordinator_id,
                        "ade_plan_id": command.plan_id,
                        "ade_trial_id": command.trial_id,
                        "evaluation_subject_kind": subject_kind,
                        "ade_engine_command_id": command.command_id,
                        "ade_workload": "operator_evaluation",
                    },
                },
            },
        )

    def _request_for_subject(self, subject_kind: str) -> dict[str, object]:
        request = copy.deepcopy(self.request)
        if subject_kind == "base_model":
            request.update(
                {
                    "model_protocol": copy.deepcopy(self.base_model_protocol),
                    "reasoning_parser": "",
                    "thinking_budget": -1,
                }
            )
        return request

    @staticmethod
    def _command_ref(command: EvaluateCommand) -> EngineCommandRef:
        return EngineCommandRef(
            command_id=command.command_id,
            logical_command_id=command.logical_command_id,
            attempt_id=command.attempt_id,
            attempt_index=command.attempt_index,
            coordinator_id=command.coordinator_id,
            plan_id=command.plan_id,
            trial_id=command.trial_id,
            kind=command.kind,
        )

    def _retry(
        self,
        state: RunState,
        record: OperatorEvaluationRecord,
    ) -> RunState:
        assert record.logical_command_id is not None
        if record.target_kind == "base":
            checkpoint = self.base_model
        else:
            trial = next(
                (
                    item
                    for item in state.trials
                    if item.coordinator_id == record.coordinator_id
                    and item.plan_id == record.plan_id
                    and item.trial_id == record.trial_id
                ),
                None,
            )
            if trial is None:
                raise ValueError("Operator evaluation retry target is missing")
            checkpoint = self._checkpoint(trial)
            if checkpoint is None:
                raise ValueError(
                    "Operator evaluation retry checkpoint is unavailable"
                )
        attempt_index = record.attempt_index + 1
        attempt_id = f"attempt-{attempt_index:03d}"
        command_id = f"{record.logical_command_id}-{attempt_id}"
        input_ref = f"engine://inputs/{command_id}.json"
        coordinator_id = record.coordinator_id or "c000"
        plan_id = record.plan_id or "p000"
        trial_id = record.trial_id or "base"
        command = EvaluateCommand(
            command_id=command_id,
            run_id=state.run_id,
            coordinator_id=coordinator_id,
            plan_id=plan_id,
            trial_id=trial_id,
            input_ref=input_ref,
            output_uri=f"engine://outputs/{command_id}",
            logical_command_id=record.logical_command_id,
            attempt_id=attempt_id,
            attempt_index=attempt_index,
        )
        self._write_input(
            state,
            command=command,
            checkpoint=checkpoint,
            subject_kind=self._subject_kind(state, record),
        )
        self.repository.store_engine_command(command)
        committed = self.state.apply(
            state.run_id,
            OperatorEvaluationAttemptSubmittedOutcome(
                run_id=state.run_id,
                basis_revision=state.revision,
                target_id=record.target_id,
                previous_attempt_index=record.attempt_index,
                command=self._command_ref(command),
            ),
            event_type=(
                "automatic_recovery_attempt_submitted"
                if state.status is RunStatus.RECOVERING
                else "operator_evaluation_attempt_submitted"
            ),
        )
        try:
            self.queue.submit(command)
        except OSError:
            pass
        return committed

    def _checkpoint(self, trial) -> str | None:
        result_refs = tuple(
            ref for ref in trial.result_refs if ref.endswith("/result.json")
        )
        if len(result_refs) != 1:
            return None
        result = self.objects.read_json(result_refs[0])
        checkpoint = result.get("checkpoint_ref") or result.get("model_ref")
        return checkpoint if isinstance(checkpoint, str) and checkpoint else None

    @staticmethod
    def _subject_kind(state: RunState, record: OperatorEvaluationRecord) -> str:
        if record.target_kind == "base":
            return "base_model"
        trial = next(
            (
                item
                for item in state.trials
                if item.coordinator_id == record.coordinator_id
                and item.plan_id == record.plan_id
                and item.trial_id == record.trial_id
            ),
            None,
        )
        if trial is None:
            raise ValueError("Operator evaluation subject Trial is missing")
        return (
            "p000_baseline"
            if trial.kind is TrialKind.BOOTSTRAP_BASELINE
            else "search_trial"
        )

    @staticmethod
    def _accepted_ref(state: RunState, artifact_id: str | None):
        return next(
            (
                item
                for item in state.accepted_evidence_refs
                if item.artifact_id == artifact_id
            ),
            None,
        )
