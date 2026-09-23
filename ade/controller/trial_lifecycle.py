"""Typed orchestration for one accepted Trial."""

from __future__ import annotations

import hashlib
import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from ade.agent_runtime.experiment_package import (
    EngineExperimentPackageBuilder,
    attach_curriculum_realization,
    encode_experiment_package,
)
from ade.agent_runtime.runtime import AcceptedCall, RejectedCall
from ade.agent_runtime.analysis_validation import (
    observed_analysis_failure,
    validate_analysis_delivery,
)
from ade.controller.reducer import Reducer
from ade.controller.state import StateCoordinator
from ade.controller.ports import (
    EngineCommandPort,
    EngineObjectPort,
    ReviewCommandPort,
    RunRepository,
    SnapshotPort,
)
from ade.core.agent import AgentRole
from ade.core.engine import EngineReceipt, EngineReceiptStatus
from ade.core.lifecycle import PlanTerminalDecision
from ade.core.plan import PlanKind, PlanStatus
from ade.core.ranking import score_pair_sort_key
from ade.core.scope import PlanKey, TrialKey
from ade.core.snapshot import SnapshotFile, SnapshotKind, SnapshotRef
from ade.core.trial import TrialArchiveStatus, TrialKind, TrialOutcome, TrialPhase
from ade.core.outcomes import (
    AnalysisAcceptedOutcome,
    AnalysisFailedOutcome,
    AnalysisReviewAttemptRetryPendingOutcome,
    AnalysisReviewAttemptSubmittedOutcome,
    AnalysisReviewCompletedOutcome,
    AnalysisReviewSubmittedOutcome,
    ArtifactAcceptedOutcome,
    BaselineTrialRegisteredOutcome,
    BuilderProposalCompletedOutcome,
    BuilderRealizationCompletedOutcome,
    BuilderRealizationFinalizedOutcome,
    BuilderRealizationStartedOutcome,
    BuilderReflectionCompletedOutcome,
    BuilderFailedOutcome,
    EngineAttemptRetryPendingOutcome,
    EngineCompletedOutcome,
    EngineFailedOutcome,
    EngineQueuedOutcome,
    CoordinatorTrialCancelledOutcome,
    SummaryAcceptedOutcome,
)
from ade.core.validation import DeliveryViolation, ValidationReport
from ade.core.artifacts import ArtifactRef
from ade.core.run import ReviewCommandRef, RunStatus
from ade.memory.records import objective_comparison_record_path
from ade.review_labor.planning import (
    compile_review_command,
    coverage_from_packet,
    validate_review_packet,
)
from ade.review_labor.protocol import ReviewCommand
from ade.tasks.contracts import (
    ArtifactCompilationError,
    ArtifactCompilationRequest,
    EngineArtifactBindingRequest,
    EngineCommandRequest,
    SummaryAdmissionError,
    AnalysisAdmissionError,
)
from ade.tasks.contracts import (
    AnalysisReport,
    AnalysisReviewPlan,
    ArtifactDelivery,
    PlanSummary,
    RunSummary,
)
from ade.tasks.registry import TaskRegistry
from ade.tasks.plugin import ArtifactAcceptanceContext


def engine_binding_key(command_id: str, artifact_id: str) -> str:
    identity = f"{command_id}\0{artifact_id}".encode()
    return hashlib.sha256(identity).hexdigest()[:24]


class TrialLifecycle:
    def __init__(
        self,
        *,
        repository: RunRepository,
        tasks: TaskRegistry,
        queue: EngineCommandPort,
        engine_io: EngineObjectPort,
        snapshots: SnapshotPort | None = None,
        reward_harness_judge_factory=None,
        selection_harness_judge_factory=None,
        review_queue: ReviewCommandPort | None = None,
    ) -> None:
        self.repository = repository
        self.tasks = tasks
        self.queue = queue
        self.engine_io = engine_io
        self.snapshots = snapshots or getattr(repository, "snapshots", None)
        self.reward_harness_judge_factory = reward_harness_judge_factory
        self.selection_harness_judge_factory = selection_harness_judge_factory
        self.review_queue = review_queue
        self.reducer = Reducer()
        self.state = StateCoordinator(repository, self.reducer)

    def cancel_trial(
        self,
        run_id: str,
        trial_key: TrialKey,
        *,
        reason: str,
    ):
        state = self.repository.load(run_id)
        trial = self._trial(state, trial_key)
        payload = (
            json.dumps(
                {
                    "schema_version": "ade.trial_cancellation.v1",
                    "run_id": run_id,
                    "coordinator_id": trial.coordinator_id,
                    "plan_id": trial.plan_id,
                    "trial_id": trial.trial_id,
                    "basis_revision": state.revision,
                    "phase": trial.phase.value,
                    "reason": reason,
                    "outcome": TrialOutcome.CANCELLED.value,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        cancellation_ref = self.repository.put_artifact(
            run_id, "trial_cancellation", payload
        )
        files = (
            SnapshotFile.generated(
                "cancellation.json",
                payload,
                source_id=cancellation_ref.artifact_id,
            ),
        )
        trial_snapshot = self.snapshots.materialize(
            kind=SnapshotKind.TRIAL,
            run_id=run_id,
            coordinator_id=trial.coordinator_id,
            plan_id=trial.plan_id,
            trial_id=trial.trial_id,
            revision=state.revision,
            files=files,
        )
        plan_snapshot = self.snapshots.materialize(
            kind=SnapshotKind.PLAN,
            run_id=run_id,
            coordinator_id=trial.coordinator_id,
            plan_id=trial.plan_id,
            revision=state.revision,
            files=files,
        )
        run_snapshot = self.snapshots.materialize(
            kind=SnapshotKind.RUN,
            run_id=run_id,
            revision=state.revision,
            files=files,
        )
        return self.state.apply(
            run_id,
            CoordinatorTrialCancelledOutcome(
                run_id=run_id,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                trial_id=trial.trial_id,
                basis_revision=state.revision,
                cancellation_ref=cancellation_ref,
                trial_snapshot_ref=trial_snapshot,
                plan_snapshot_ref=plan_snapshot,
                run_snapshot_ref=run_snapshot,
            ),
            event_type="coordinator_trial_cancelled",
        )

    def register_baseline_artifact(
        self,
        run_id: str,
        *,
        trial: TrialKey,
        kind: str,
        content: bytes,
    ):
        if trial.run_id != run_id:
            raise ValueError("baseline Trial belongs to another run")
        state = self.repository.load(run_id)
        ref = self.repository.put_artifact(run_id, kind, content)
        report = self.tasks.get(state.task.task_id).validate_artifact(ref)
        if not report.ok:
            raise ValueError(report.violations[0].message)
        self.state.apply(
            run_id,
            BaselineTrialRegisteredOutcome(
                run_id=run_id,
                coordinator_id=trial.coordinator_id,
                trial_id=trial.trial_id,
                plan_id=trial.plan_id,
                basis_revision=state.revision,
                artifact_ref=ref,
            ),
            event_type="baseline_trial_registered",
        )
        return ref

    def fail_builder(
        self,
        run_id: str,
        trial_key: TrialKey,
        delivery: AcceptedCall | RejectedCall,
        *,
        validation: ValidationReport | None = None,
    ):
        state = self.repository.load(run_id)
        if trial_key.run_id != run_id or (
            delivery.call.run_id,
            delivery.call.coordinator_id,
            delivery.call.plan_id,
            delivery.call.subject_id,
        ) != (
            trial_key.run_id,
            trial_key.coordinator_id,
            trial_key.plan_id,
            trial_key.trial_id,
        ):
            raise ValueError("Builder failure identity does not match Trial")
        if delivery.call.role is not AgentRole.ARTIFACT_BUILDER:
            raise ValueError("Builder failure requires Artifact Builder role")
        self._validate_active_call(state, delivery)
        failure_validation = validation or delivery.validation
        payload = {
            "schema_version": "1",
            "coordinator_id": trial_key.coordinator_id,
            "plan_id": trial_key.plan_id,
            "trial_id": trial_key.trial_id,
            "call_id": delivery.call.call_id,
            "attempt_id": delivery.attempt.attempt_id,
            "reason": "Artifact Builder exhausted its retry budget",
            "violations": [
                {
                    "code": item.code,
                    "message": item.message,
                    "path": item.path,
                    "repairable": item.repairable,
                }
                for item in failure_validation.violations
            ],
            "last_workspace_uri": delivery.attempt.workspace_uri,
        }
        ref = self.repository.put_artifact(
            run_id,
            "builder_failure",
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        snapshot_ref, objective_comparison_ref, _, _ = self._analysis_packet(
            state,
            trial_key,
            files=(
                SnapshotFile.generated(
                    "failure.json",
                    self.repository.read_artifact(run_id, ref),
                    source_id=ref.artifact_id,
                ),
            ),
            evidence_status="unavailable",
            outcome="failed",
        )
        return self.state.apply(
            run_id,
            BuilderFailedOutcome(
                run_id=run_id,
                coordinator_id=trial_key.coordinator_id,
                plan_id=trial_key.plan_id,
                trial_id=trial_key.trial_id,
                basis_revision=state.revision,
                call_id=delivery.call.call_id,
                failure_ref=ref,
                snapshot_ref=snapshot_ref,
                objective_comparison_ref=objective_comparison_ref,
                agent_session=delivery.session,
            ),
            event_type="builder_failed",
        )

    def _materialize_builder_realization(
        self,
        run_id: str,
        trial_key: TrialKey,
        delivery: AcceptedCall,
    ):
        state = self.repository.load(run_id)
        if trial_key.run_id != run_id or (
            delivery.call.run_id,
            delivery.call.coordinator_id,
            delivery.call.plan_id,
            delivery.call.subject_id,
        ) != (
            trial_key.run_id,
            trial_key.coordinator_id,
            trial_key.plan_id,
            trial_key.trial_id,
        ):
            raise ValueError("artifact delivery identity does not match Trial")
        self._validate_active_call(state, delivery)
        if delivery.call.role is not AgentRole.ARTIFACT_BUILDER:
            raise ValueError("artifact delivery requires Artifact Builder role")
        if not isinstance(delivery.output, ArtifactDelivery):
            raise ValueError("artifact delivery requires typed ArtifactDelivery output")
        trial = self._trial(state, trial_key)
        plan = self._plan(state, trial_key.plan)
        decision_ref = self._accepted_ref_by_id(
            state.accepted_plan_refs,
            plan.decision_ref_id,
            "PlanningDecision",
        )
        try:
            planning_decision = json.loads(
                self.repository.read_artifact(run_id, decision_ref)
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("accepted PlanningDecision artifact is invalid") from error
        if not isinstance(planning_decision, dict):
            raise ValueError("accepted PlanningDecision must be a JSON object")
        plugin = self.tasks.get(state.task.task_id)
        compiled = plugin.compile_artifact(
            ArtifactCompilationRequest(
                delivery=delivery.output,
                task_config=state.task.config,
                planning_decision=planning_decision,
            )
        )
        duplicate = next(
            (
                prior
                for prior in state.trials
                if prior.artifact_ref_id is not None
                and prior.outcome is TrialOutcome.SUCCEEDED
                and self.repository.read_artifact(
                    run_id,
                    self._artifact_by_id(state, prior.artifact_ref_id),
                )
                == compiled.content
            ),
            None,
        )
        if duplicate is not None:
            raise ArtifactCompilationError(
                ValidationReport(
                    (
                        DeliveryViolation(
                            "duplicate_trial_artifact",
                            (
                                "compiled artifact is byte-identical to the accepted "
                                f"artifact from {duplicate.coordinator_id}/"
                                f"{duplicate.plan_id}/{duplicate.trial_id}; implement "
                                "the proposed experimental change before resubmitting"
                            ),
                            compiled.path,
                            repairable=True,
                        ),
                    )
                )
            )
        realization_report_ref = plugin.validate_artifact_acceptance(
            ArtifactAcceptanceContext(
                run_id=run_id,
                trial_key=trial_key,
                state=state,
                trial=trial,
                delivery=delivery,
                compiled=compiled,
                planning_decision=planning_decision,
                lifecycle=self,
            )
        )
        ref = self.repository.put_artifact(
            run_id,
            compiled.kind,
            compiled.content,
        )
        delivered_supporting_refs = tuple(
            self.repository.put_artifact(
                run_id,
                artifact.kind,
                artifact.content,
            )
            for artifact in delivery.output.supporting_artifacts
        )
        supporting_refs = (
            *delivered_supporting_refs,
            *((realization_report_ref,) if realization_report_ref is not None else ()),
        )
        manifest_content = (
            json.dumps(
                {
                    "schema_version": "1",
                    "abi_version": "1",
                    "run_id": run_id,
                    "coordinator_id": trial_key.coordinator_id,
                    "plan_id": trial_key.plan_id,
                    "trial_id": trial_key.trial_id,
                    "task_id": state.task.task_id,
                    "admission_revision": state.revision,
                    "validation_version": "1",
                    "lineage": list(self._trial(state, trial_key).source_artifact_ref_ids),
                    "files": [
                        {
                            "path": compiled.path,
                            "kind": ref.kind,
                            "artifact_id": ref.artifact_id,
                            "sha256": ref.digest,
                            "size_bytes": ref.size_bytes,
                        }
                        for ref in (ref,)
                    ]
                    + [
                        {
                            "path": artifact.path,
                            "kind": stored.kind,
                            "artifact_id": stored.artifact_id,
                            "sha256": stored.digest,
                            "size_bytes": stored.size_bytes,
                        }
                        for artifact, stored in zip(
                            delivery.output.supporting_artifacts,
                            delivered_supporting_refs,
                            strict=True,
                        )
                    ]
                    + (
                        [
                            {
                                "path": (
                                    "reward-compliance-report.json"
                                    if realization_report_ref.kind
                                    == "reward_compliance_report"
                                    else "curriculum-realization.json"
                                    if realization_report_ref.kind
                                    == "curriculum_realization"
                                    else "selection-realization.json"
                                ),
                                "kind": realization_report_ref.kind,
                                "artifact_id": realization_report_ref.artifact_id,
                                "sha256": realization_report_ref.digest,
                                "size_bytes": realization_report_ref.size_bytes,
                            }
                        ]
                        if realization_report_ref is not None
                        else []
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        manifest_ref = self.repository.put_artifact(
            run_id,
            "experiment_artifact_manifest",
            manifest_content,
        )
        supporting_refs = (*supporting_refs, manifest_ref)
        report = plugin.validate_artifact(ref)
        if not report.ok:
            raise ValueError(report.violations[0].message)
        realization_status = "verified"
        realization_reason = None
        if realization_report_ref is not None:
            try:
                realization_report = json.loads(
                    self.repository.read_artifact(run_id, realization_report_ref)
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("artifact realization report is invalid") from error
            if isinstance(realization_report, dict) and realization_report.get(
                "realization_status"
            ) is not None:
                realization_status = str(realization_report["realization_status"])
                realization_reason = realization_report.get("reason")
        realization_content = (
            json.dumps(
                {
                    "schema_version": "ade.builder_realization.v1",
                    "reflection_index": delivery.call.reflection_index,
                    "delivery_ref": delivery.attempt.workspace_uri,
                    "realization_status": realization_status,
                    "reason": realization_reason,
                    "artifact_ref": asdict(ref),
                    "supporting_refs": [asdict(item) for item in supporting_refs],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        realization_ref = self.repository.put_artifact(
            run_id, "builder_realization", realization_content
        )
        return ref, supporting_refs, realization_ref

    def record_builder_delivery(
        self,
        run_id: str,
        trial_key: TrialKey,
        delivery: AcceptedCall,
    ):
        state = self.repository.load(run_id)
        self._validate_active_call(state, delivery)
        active = next(
            item
            for item in state.active_agent_calls
            if item.call_id == delivery.call.call_id
        )
        if active.round_status != "agent_running":
            raise ValueError("Builder delivery round is not running")
        outcome_type = (
            BuilderProposalCompletedOutcome
            if active.reflection_index == 0
            else BuilderReflectionCompletedOutcome
        )
        common = {
            "run_id": run_id,
            "basis_revision": state.revision,
            "call_id": active.call_id,
            "attempt_id": active.attempt_id,
            "delivery_ref": delivery.attempt.workspace_uri,
        }
        outcome = (
            outcome_type(**common)
            if active.reflection_index == 0
            else outcome_type(
                **common,
                reflection_index=active.reflection_index,
            )
        )
        return self.state.apply(
            run_id,
            outcome,
            event_type=(
                "builder_proposal_completed"
                if active.reflection_index == 0
                else "builder_reflection_completed"
            ),
        )

    def start_builder_realization(
        self,
        run_id: str,
        call_id: str,
    ):
        state = self.repository.load(run_id)
        active = next(
            item for item in state.active_agent_calls if item.call_id == call_id
        )
        if active.current_delivery_ref is None:
            raise ValueError("Builder realization requires a delivery ref")
        return self.state.apply(
            run_id,
            BuilderRealizationStartedOutcome(
                run_id,
                state.revision,
                call_id,
                active.reflection_index,
                active.current_delivery_ref,
            ),
            event_type="builder_realization_started",
        )

    def complete_builder_realization(
        self,
        run_id: str,
        trial_key: TrialKey,
        delivery: AcceptedCall,
    ):
        state = self.repository.load(run_id)
        active = next(
            item
            for item in state.active_agent_calls
            if item.call_id == delivery.call.call_id
        )
        if active.round_status != "realization_running":
            raise ValueError("Builder realization round is not running")
        _ref, _supporting_refs, realization_ref = (
            self._materialize_builder_realization(run_id, trial_key, delivery)
        )
        assert active.current_delivery_ref is not None
        self.state.apply(
            run_id,
            BuilderRealizationCompletedOutcome(
                run_id,
                state.revision,
                active.call_id,
                active.reflection_index,
                active.current_delivery_ref,
                realization_ref,
            ),
            event_type="builder_realization_completed",
        )
        return realization_ref

    def finalize_builder_realization(self, run_id: str, call_id: str):
        state = self.repository.load(run_id)
        active = next(
            item for item in state.active_agent_calls if item.call_id == call_id
        )
        if (
            active.current_delivery_ref is None
            or active.current_realization_ref is None
        ):
            raise ValueError("Builder finalization requires ready refs")
        return self.state.apply(
            run_id,
            BuilderRealizationFinalizedOutcome(
                run_id,
                state.revision,
                call_id,
                active.reflection_index,
                active.current_delivery_ref,
                active.current_realization_ref,
            ),
            event_type="builder_realization_finalized",
        )

    def accept_finalized_artifact(
        self,
        run_id: str,
        trial_key: TrialKey,
        delivery: AcceptedCall,
    ):
        state = self.repository.load(run_id)
        active = next(
            item
            for item in state.active_agent_calls
            if item.call_id == delivery.call.call_id
        )
        if active.round_status != "finalized" or active.current_realization_ref is None:
            raise ValueError("Builder realization is not finalized")
        realization = json.loads(
            self.repository.read_artifact(
                run_id, active.current_realization_ref
            )
        )
        ref = ArtifactRef.from_dict(realization["artifact_ref"])
        supporting_refs = tuple(
            ArtifactRef.from_dict(item)
            for item in realization.get("supporting_refs", ())
        )
        self.state.apply(
            run_id,
            ArtifactAcceptedOutcome(
                run_id=run_id,
                coordinator_id=trial_key.coordinator_id,
                plan_id=trial_key.plan_id,
                trial_id=trial_key.trial_id,
                basis_revision=state.revision,
                call_id=delivery.call.call_id,
                artifact_ref=ref,
                supporting_refs=(*supporting_refs, active.current_realization_ref),
                agent_session=delivery.session,
            ),
            event_type="artifact_accepted",
        )
        return ref

    def admit_artifact(
        self,
        run_id: str,
        trial_key: TrialKey,
        delivery: AcceptedCall,
    ):
        state = self.repository.load(run_id)
        ref, supporting_refs, _realization_ref = (
            self._materialize_builder_realization(run_id, trial_key, delivery)
        )
        self.state.apply(
            run_id,
            ArtifactAcceptedOutcome(
                run_id=run_id,
                coordinator_id=trial_key.coordinator_id,
                plan_id=trial_key.plan_id,
                trial_id=trial_key.trial_id,
                basis_revision=state.revision,
                call_id=delivery.call.call_id,
                artifact_ref=ref,
                supporting_refs=supporting_refs,
                agent_session=delivery.session,
            ),
            event_type="artifact_accepted",
        )
        return ref

    def submit_engine(
        self,
        run_id: str,
        trial_key: TrialKey,
        config: dict[str, object],
        *,
        command_id: str | None = None,
    ):
        if trial_key.run_id != run_id:
            raise ValueError("Trial belongs to another run")
        state = self.repository.load(run_id)
        trial = self._trial(state, trial_key)
        if not trial.artifact_ref_id:
            raise ValueError("Trial artifact must be accepted before Engine submission")
        plugin = self.tasks.get(state.task.task_id)
        logical_command_id = command_id or trial.logical_command_id or (
            f"{run_id}-{trial_key.coordinator_id}-{trial_key.plan_id}-"
            f"{trial_key.trial_id}-{plugin.engine_command_kind}"
        )
        retry = trial.engine_retry_pending
        recovered_output_refs = (
            self._completed_retry_output_refs(state, trial)
            if retry
            else ()
        )
        attempt_index = trial.engine_attempt_index + 1 if retry else 1
        attempt_id = f"attempt-{attempt_index:03d}"
        command_id = f"{logical_command_id}-{attempt_id}"
        input_ref = f"engine://inputs/{command_id}.json"
        output_uri = f"engine://outputs/{command_id}"
        artifact = self._artifact_by_id(state, trial.artifact_ref_id)
        plan = self._plan(state, trial_key.plan)
        planning_decision: dict[str, object] = {}
        if plan.kind is not PlanKind.BOOTSTRAP:
            decision_ref = self._accepted_ref_by_id(
                state.accepted_plan_refs,
                plan.decision_ref_id,
                "PlanningDecision",
            )
            try:
                decoded_decision = json.loads(
                    self.repository.read_artifact(run_id, decision_ref)
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("accepted PlanningDecision artifact is invalid") from error
            if not isinstance(decoded_decision, dict):
                raise ValueError("accepted PlanningDecision must be a JSON object")
            planning_decision = decoded_decision
        config = copy.deepcopy(config)
        config = plugin.prepare_engine_config(config, state)
        layout = getattr(self.repository, "layout", None)
        if layout is not None:
            config["usage_run_dir"] = str(layout.run_dir(run_id).resolve())
        if state.forked_from is not None:
            config["fork_lineage"] = asdict(state.forked_from)
        binding_key = engine_binding_key(command_id, artifact.artifact_id)
        final_realization: dict[str, object] = {}
        if (
            state.task.task_id in {"data_selection", "curriculum_learning"}
            and plan.kind is not PlanKind.BOOTSTRAP
        ):
            expected_realization_kind = (
                "data_selection_realization"
                if state.task.task_id == "data_selection"
                else "curriculum_realization"
            )
            realization_refs = tuple(
                self._artifact_by_id(state, artifact_id)
                for artifact_id in trial.artifact_supporting_ref_ids
                if self._artifact_by_id(state, artifact_id).kind
                == expected_realization_kind
            )
            if len(realization_refs) != 1:
                raise ValueError(
                    f"{state.task.task_id} training requires one final realization"
                )
            decoded_realization = json.loads(
                self.repository.read_artifact(run_id, realization_refs[0])
            )
            if not isinstance(decoded_realization, dict):
                raise ValueError(
                    f"{state.task.task_id} final realization is invalid"
                )
            final_realization = decoded_realization
        binding = plugin.bind_engine_artifact(
            EngineArtifactBindingRequest(
                binding_uri=f"engine://bindings/{binding_key}",
                compiled_kind=artifact.kind,
                compiled_content=self.repository.read_artifact(run_id, artifact),
                task_config=state.task.config,
                engine_config=config,
                planning_decision=planning_decision,
                final_realization=final_realization,
                is_baseline=plan.kind is PlanKind.BOOTSTRAP,
            )
        )
        for payload in binding.objects:
            self.engine_io.put_bytes(payload.uri, payload.content)
        self.engine_io.put_json(
            input_ref,
            dict(binding.input_payload),
        )
        command = plugin.build_engine_command(
            EngineCommandRequest(
                command_id=command_id,
                run_id=run_id,
                coordinator_id=trial_key.coordinator_id,
                plan_id=trial_key.plan_id,
                trial_id=trial_key.trial_id,
                input_ref=input_ref,
                output_uri=output_uri,
                logical_command_id=logical_command_id,
                attempt_id=attempt_id,
                attempt_index=attempt_index,
            )
        )
        self.repository.store_engine_command(command)
        submitted_at = time.time()
        self.state.apply(
            run_id,
            EngineQueuedOutcome(
                run_id=run_id,
                coordinator_id=trial_key.coordinator_id,
                plan_id=trial_key.plan_id,
                trial_id=trial_key.trial_id,
                basis_revision=state.revision,
                command_id=command.command_id,
                logical_command_id=logical_command_id,
                attempt_id=attempt_id,
                attempt_index=attempt_index,
                command_kind=command.kind,
                submitted_at=submitted_at,
                liveness_deadline=(
                    submitted_at + self.queue.claim_timeout_seconds
                ),
            ),
            event_type=(
                "automatic_recovery_attempt_submitted"
                if state.status is RunStatus.RECOVERING
                else "engine_attempt_submitted"
                if retry
                else "engine_submitted"
            ),
        )
        self.queue.submit(command)
        if recovered_output_refs:
            self.queue.publish_receipt(
                EngineReceipt(
                    receipt_id=f"receipt-{command.command_id}-completed-output-reused",
                    command_id=command.command_id,
                    run_id=command.run_id,
                    coordinator_id=command.coordinator_id,
                    plan_id=command.plan_id,
                    trial_id=command.trial_id,
                    status=EngineReceiptStatus.SUCCEEDED,
                    output_refs=recovered_output_refs,
                    logical_command_id=(
                        command.logical_command_id or command.command_id
                    ),
                    attempt_id=command.attempt_id,
                    attempt_index=command.attempt_index,
                )
            )
        return command

    def _completed_retry_output_refs(self, state, trial) -> tuple[str, ...]:
        for attempt_index in range(trial.engine_attempt_index, 0, -1):
            previous_command_id = (
                f"{trial.logical_command_id}-attempt-{attempt_index:03d}"
            )
            result_ref = f"engine://outputs/{previous_command_id}/result.json"
            manifest_ref = (
                f"engine://outputs/{previous_command_id}/raw/manifest.json"
            )
            try:
                self.engine_io.read_json(result_ref)
                package = EngineExperimentPackageBuilder(self.engine_io).build(
                    manifest_ref,
                    result_ref=result_ref,
                    task_id=state.task.task_id,
                    evidence_specs=self.tasks.get(
                        state.task.task_id
                    ).analyzer_evidence_specs,
                )
            except FileNotFoundError:
                continue
            manifest = json.loads(package["experiment/manifest.json"])
            if manifest.get("trial_status") not in {
                "completed",
                "completed_with_failures",
            }:
                continue
            identity = tuple(
                manifest[field]
                for field in ("run_id", "coordinator_id", "plan_id", "trial_id")
            )
            expected = (
                state.run_id,
                trial.coordinator_id,
                trial.plan_id,
                trial.trial_id,
            )
            if identity != expected:
                raise ValueError(
                    "completed retry output identity does not match Trial"
                )
            return result_ref, manifest_ref
        return ()

    def collect_engine(self, run_id: str, command_id: str) -> dict[str, bytes]:
        state = self.repository.load(run_id)
        receipt = self.queue.load_receipt(command_id)
        if receipt.run_id != run_id:
            raise ValueError("Engine receipt belongs to another run")
        trial_key = TrialKey(
            run_id,
            receipt.coordinator_id,
            receipt.plan_id,
            receipt.trial_id,
        )
        trial = self._trial(state, trial_key)
        active = tuple(
            item
            for item in state.active_engine_commands
            if item.command_id == command_id
        )
        if (
            trial.phase is not TrialPhase.ENGINE_RUNNING
            or trial.command_id != command_id
            or len(active) != 1
            or active[0].logical_command_id
            != (receipt.logical_command_id or receipt.command_id)
            or active[0].attempt_id != receipt.attempt_id
            or active[0].attempt_index != receipt.attempt_index
        ):
            raise ValueError("Engine receipt is not the current active Attempt")
        self.repository.store_engine_receipt(receipt)
        plugin = self.tasks.get(state.task.task_id)
        result = plugin.normalize_engine_receipt(receipt)
        if not result.succeeded:
            reason = result.error or "Engine command failed"
            failure_kind = receipt.failure_kind or (
                "worker_lost" if reason.startswith("worker_lost:")
                else "engine_failed"
            )
            failure_ref = self.repository.put_artifact(
                run_id,
                "engine_failure",
                (json.dumps({
                    "schema_version": "1",
                    "command_id": command_id,
                    "logical_command_id": receipt.logical_command_id,
                    "attempt_id": receipt.attempt_id,
                    "attempt_index": receipt.attempt_index,
                    "receipt_id": receipt.receipt_id,
                    "execution_status": failure_kind,
                    "retryable": receipt.retryable,
                    "error": reason,
                }, indent=2, sort_keys=True) + "\n").encode(),
            )
            if receipt.retryable:
                self.state.apply(
                    run_id,
                    EngineAttemptRetryPendingOutcome(
                        run_id=run_id,
                        coordinator_id=result.coordinator_id,
                        plan_id=result.plan_id,
                        trial_id=result.trial_id,
                        basis_revision=state.revision,
                        command_id=command_id,
                        logical_command_id=(
                            receipt.logical_command_id or command_id
                        ),
                        attempt_id=receipt.attempt_id,
                        attempt_index=receipt.attempt_index,
                        receipt_id=receipt.receipt_id,
                        failure_ref=failure_ref,
                        failure_kind=failure_kind,
                        message=reason,
                    ),
                    event_type=(
                        "engine_attempt_retry_pending"
                        if state.pause_requested
                        else "automatic_recovery_started"
                    ),
                )
                return {}
            package_ref = None
            unit_ids: tuple[str, ...] = ()
            snapshot_ref = None
            objective_comparison_ref = None
            if len(result.output_refs) >= 2:
                _package, package_ref, unit_ids = self._engine_package(
                    run_id,
                    result,
                )
            else:
                (
                    snapshot_ref,
                    objective_comparison_ref,
                    _,
                    _,
                ) = self._analysis_packet(
                    state,
                    trial_key,
                    files=(SnapshotFile.generated(
                        "failure.json",
                        self.repository.read_artifact(run_id, failure_ref),
                        source_id=failure_ref.artifact_id,
                    ),),
                    evidence_status="unavailable",
                    outcome="failed",
                )
            self.state.apply(
                run_id,
                EngineFailedOutcome(
                    run_id=run_id,
                    coordinator_id=result.coordinator_id,
                    plan_id=result.plan_id,
                    trial_id=result.trial_id,
                    basis_revision=state.revision,
                    command_id=command_id,
                    receipt_id=receipt.receipt_id,
                    failure_ref=failure_ref,
                    snapshot_ref=snapshot_ref,
                    failure_kind=failure_kind,
                    objective_comparison_ref=objective_comparison_ref,
                    result_refs=result.output_refs,
                    package_ref=package_ref,
                    authorized_unit_ids=unit_ids,
                ),
                event_type="engine_failed",
            )
            return {}
        if len(result.output_refs) < 2:
            raise ValueError("Engine receipt requires result and raw manifest refs")
        package, package_ref, unit_ids = self._engine_package(run_id, result)
        assert package_ref is not None
        self.state.apply(
            run_id,
            EngineCompletedOutcome(
                run_id=run_id,
                coordinator_id=result.coordinator_id,
                plan_id=result.plan_id,
                trial_id=result.trial_id,
                basis_revision=state.revision,
                command_id=command_id,
                receipt_id=receipt.receipt_id,
                result_refs=result.output_refs,
                package_ref=package_ref,
                authorized_unit_ids=unit_ids,
            ),
            event_type="engine_completed",
        )
        return package

    def _engine_package(self, run_id: str, result):
        state = self.repository.load(run_id)
        plugin = self.tasks.get(state.task.task_id)
        trial_key = TrialKey(
            run_id, result.coordinator_id, result.plan_id, result.trial_id
        )
        trial = self._trial(state, trial_key)
        package = EngineExperimentPackageBuilder(self.engine_io).build(
            result.output_refs[1],
            result_ref=result.output_refs[0],
            task_id=plugin.task_id,
            evidence_specs=plugin.analyzer_evidence_specs,
        )
        package = self._attach_final_realization(
            state, trial, package
        )
        manifest_content = package["experiment/manifest.json"]
        package_manifest = json.loads(manifest_content)
        package_identity = tuple(
            package_manifest[field]
            for field in ("run_id", "coordinator_id", "plan_id", "trial_id")
        )
        result_identity = (
            run_id,
            result.coordinator_id,
            result.plan_id,
            result.trial_id,
        )
        if package_identity != result_identity:
            raise ValueError("Engine Experiment Package identity does not match receipt")
        unit_ids = tuple(
            str(item)
            for item in (
                package_manifest.get("unit_ids")
                or [
                    artifact.get("id")
                    for artifact in package_manifest.get("artifacts", ())
                    if isinstance(artifact, dict)
                ]
            )
            if item
        )
        package_ref = self.repository.put_artifact(
            run_id,
            "experiment_package",
            encode_experiment_package(package),
        )
        return package, package_ref, unit_ids

    def _attach_final_realization(self, state, trial, package):
        report_refs = tuple(
            ref
            for artifact_id in trial.artifact_supporting_ref_ids
            for ref in (self._artifact_by_id(state, artifact_id),)
            if ref.kind
            in {
                "data_selection_realization",
                "reward_compliance_report",
                "curriculum_realization",
            }
        )
        if trial.kind is TrialKind.BOOTSTRAP_BASELINE and not report_refs:
            if state.task.task_id != "curriculum_learning":
                return package
            if not trial.command_id:
                raise ValueError(
                    "Curriculum baseline final binding command is unavailable"
                )
            command = self.repository.load_engine_command(
                state.run_id,
                trial.coordinator_id,
                trial.plan_id,
                trial.trial_id,
                trial.command_id,
            )
            bound_input = self.engine_io.read_json(command.input_ref)
            rft = bound_input.get("rft")
            realization_ref = (
                rft.get("curriculum_realization_ref")
                if isinstance(rft, dict)
                else None
            )
            if not isinstance(realization_ref, str) or not realization_ref:
                raise ValueError(
                    "Curriculum baseline final realization is unavailable"
                )
            return attach_curriculum_realization(
                package, self.engine_io.read_bytes(realization_ref)
            )
        if len(report_refs) != 1:
            raise ValueError("Search Trial requires one final realization report")
        report_ref = report_refs[0]
        report_content = self.repository.read_artifact(state.run_id, report_ref)
        report = json.loads(report_content)
        if not isinstance(report, dict):
            raise ValueError("final realization report is invalid")
        files: dict[str, bytes] = {
            "experiment/realization/final-realization.json": report_content,
        }
        if report_ref.kind == "data_selection_realization":
            if not trial.command_id:
                raise ValueError("Data Selection final binding command is unavailable")
            command = self.repository.load_engine_command(
                state.run_id,
                trial.coordinator_id,
                trial.plan_id,
                trial.trial_id,
                trial.command_id,
            )
            bound_input = self.engine_io.read_json(command.input_ref)
            sft = bound_input.get("sft")
            dataset = sft.get("dataset") if isinstance(sft, dict) else None
            if not isinstance(dataset, dict):
                raise ValueError("Data Selection final training binding is unavailable")
            required = {
                "selection-result.json": "selection_ref",
                "training-binding.json": "training_binding_ref",
            }
            for filename, key in required.items():
                ref = dataset.get(key)
                if not isinstance(ref, str) or not ref:
                    raise ValueError(f"Data Selection final {key} is unavailable")
                files[f"experiment/realization/{filename}"] = (
                    self.engine_io.read_bytes(ref)
                )
            rows = report.get("selected_rows")
            if not isinstance(rows, list):
                raise ValueError("Data Selection selected rows are unavailable")
            files["experiment/realization/selected-examples.jsonl"] = (
                self._jsonl_content(rows)
            )
        elif report_ref.kind == "curriculum_realization":
            return attach_curriculum_realization(package, report_content)
        else:
            files.update(
                {
                    "experiment/realization/source-groups.jsonl": (
                        self._jsonl_content(report.get("source_records"))
                    ),
                    "experiment/realization/group-results.jsonl": (
                        self._jsonl_content(report.get("records"))
                    ),
                    "experiment/realization/judge-manifest.json": (
                        json.dumps(
                            {
                                key: report.get(key)
                                for key in (
                                    "judge_binding",
                                    "judge_status",
                                    "judge_completed_count",
                                    "judge_unavailable_count",
                                )
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode(),
                }
            )
        manifest = json.loads(package["experiment/manifest.json"])
        manifest["realization"] = {
            "schema_version": "1",
            "status": report.get("realization_status"),
            "reason": report.get("reason"),
            "files": sorted(files),
        }
        updated = dict(package)
        updated.update(files)
        updated["experiment/manifest.json"] = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        return updated

    @staticmethod
    def _jsonl_content(rows) -> bytes:
        if not isinstance(rows, list):
            raise ValueError("final realization JSONL rows are unavailable")
        return b"".join(
            (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
            for row in rows
        )

    def accept_analysis(
        self,
        run_id: str,
        trial_key: TrialKey,
        delivery: AcceptedCall,
    ):
        state = self.repository.load(run_id)
        if trial_key.run_id != run_id or (
            delivery.call.run_id,
            delivery.call.coordinator_id,
            delivery.call.plan_id,
            delivery.call.subject_id,
        ) != (
            trial_key.run_id,
            trial_key.coordinator_id,
            trial_key.plan_id,
            trial_key.trial_id,
        ):
            raise ValueError("analysis delivery identity does not match Trial")
        if delivery.call.role is not AgentRole.ANALYZER:
            raise ValueError("analysis delivery requires Analyzer role")
        self._validate_active_call(state, delivery)
        if not isinstance(delivery.output, AnalysisReport):
            raise ValueError("analysis delivery requires AnalysisReport")
        trial = self._trial(state, trial_key)
        report = validate_analysis_delivery(
            delivery.output,
            delivery.workspace,
            task_id=state.task.task_id,
            authorized_ids=trial.authorized_unit_ids,
        )

        if not report.ok:
            raise AnalysisAdmissionError(report)
        analysis_ref = self.repository.put_artifact(
            run_id,
            "trial_analysis",
            delivery.output.content,
        )
        findings_ref = self.repository.put_artifact(
            run_id,
            "trial_findings",
            delivery.output.findings_content,
        )
        evidence_ref = self.repository.put_artifact(
            run_id,
            "trial_analysis_evidence",
            delivery.output.evidence_content,
        )
        coverage_ref = self.repository.put_artifact(
            run_id,
            "trial_analysis_review_coverage",
            delivery.output.review_coverage_content,
        )
        (
            snapshot_ref,
            objective_comparison_ref,
            offline_score,
            offline_secondary_score,
        ) = self._analysis_packet(
            state,
            trial_key,
            files=(
                SnapshotFile.generated(
                    "analysis.md",
                    delivery.output.content,
                    source_id=analysis_ref.artifact_id,
                ),
                SnapshotFile.generated(
                    "findings.md",
                    delivery.output.findings_content,
                    source_id=findings_ref.artifact_id,
                ),
                SnapshotFile.generated(
                    "evidence.json",
                    delivery.output.evidence_content,
                    source_id=evidence_ref.artifact_id,
                ),
                SnapshotFile.generated(
                    "review-coverage.json",
                    delivery.output.review_coverage_content,
                    source_id=coverage_ref.artifact_id,
                ),
            ),
            evidence_status="complete",
            outcome="succeeded",
        )
        return self.state.apply(
            run_id,
            AnalysisAcceptedOutcome(
                run_id=run_id,
                coordinator_id=trial_key.coordinator_id,
                plan_id=trial_key.plan_id,
                trial_id=trial_key.trial_id,
                basis_revision=state.revision,
                call_id=delivery.call.call_id,
                analysis_ref=analysis_ref,
                findings_ref=findings_ref,
                evidence_ref=evidence_ref,
                review_coverage_ref=coverage_ref,
                snapshot_ref=snapshot_ref,
                objective_comparison_ref=objective_comparison_ref,
                offline_score=offline_score,
                offline_secondary_score=offline_secondary_score,
                agent_session=delivery.session,
            ),
            event_type="analysis_accepted",
        )

    def submit_analysis_review(
        self,
        run_id: str,
        trial_key: TrialKey,
        delivery: AcceptedCall,
    ):
        state = self.repository.load(run_id)
        if trial_key.run_id != run_id or delivery.call.role is not AgentRole.ANALYZER:
            raise ValueError("Review design delivery does not match Analyzer Trial")
        self._validate_active_call(state, delivery)
        if not isinstance(delivery.output, AnalysisReviewPlan):
            raise ValueError("Review design delivery requires AnalysisReviewPlan")
        trial = self._trial(state, trial_key)
        logical_command_id = (
            f"review-{run_id}-{trial.coordinator_id}-{trial.plan_id}-"
            f"{trial.trial_id}"
        )
        attempt_index = 1
        while True:
            attempt_id = f"attempt-{attempt_index:03d}"
            command_id = f"{logical_command_id}-{attempt_id}"
            try:
                self.repository.load_review_command(
                    run_id,
                    trial.coordinator_id,
                    trial.plan_id,
                    trial.trial_id,
                    command_id,
                )
            except ValueError as error:
                if not str(error).startswith(
                    "persisted Review command is missing:"
                ):
                    raise
                break
            attempt_index += 1
        try:
            command = compile_review_command(
                plan=delivery.output,
                attempt=delivery.workspace,
                command_id=command_id,
                logical_command_id=logical_command_id,
                attempt_id=attempt_id,
                attempt_index=attempt_index,
                run_id=run_id,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                trial_id=trial.trial_id,
                basis_revision=state.revision,
            )
        except ValueError as error:
            raise AnalysisAdmissionError(
                ValidationReport(
                    (
                        DeliveryViolation(
                            "invalid_analysis_review_plan",
                            str(error),
                            path="review-plan.json",
                            repairable=True,
                        ),
                    )
                )
            ) from error
        self.repository.store_review_command(command)
        design_ref = self.repository.put_artifact(
            run_id, "analysis_review_plan", delivery.output.content
        )
        return self.state.apply(
            run_id,
            AnalysisReviewSubmittedOutcome(
                run_id=run_id,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                trial_id=trial.trial_id,
                basis_revision=state.revision,
                call_id=delivery.call.call_id,
                design_ref=design_ref,
                command_ref=ReviewCommandRef(
                    command_id=command.command_id,
                    logical_command_id=command.logical_command_id,
                    attempt_id=command.attempt_id,
                    attempt_index=command.attempt_index,
                    coordinator_id=trial.coordinator_id,
                    plan_id=trial.plan_id,
                    trial_id=trial.trial_id,
                ),
                agent_session=delivery.session,
            ),
            event_type="analysis_review_submitted",
        )

    def collect_analysis_review(self, run_id: str, command_id: str):
        if self.review_queue is None:
            raise ValueError("Review queue is not configured")
        state = self.repository.load(run_id)
        refs = tuple(
            item for item in state.active_review_commands if item.command_id == command_id
        )
        if len(refs) != 1:
            raise ValueError("Review command is not active")
        ref = refs[0]
        command = self.repository.load_review_command(
            run_id, ref.coordinator_id, ref.plan_id, ref.trial_id, command_id
        )
        try:
            receipt = self.review_queue.load_receipt(command_id)
            if (
                receipt.command_id != command.command_id
                or receipt.logical_command_id != command.logical_command_id
                or receipt.attempt_id != command.attempt_id
                or receipt.attempt_index != command.attempt_index
                or receipt.run_id != command.run_id
                or receipt.coordinator_id != command.coordinator_id
                or receipt.plan_id != command.plan_id
                or receipt.trial_id != command.trial_id
            ):
                raise ValueError("Review Receipt crossed the active Attempt fence")
            packet = dict(receipt.packet)
            validate_review_packet(command, packet)
            coverage = coverage_from_packet(command, packet)
            if (
                coverage.get("command_id") != command.command_id
                or coverage.get("passed") is not True
                or not isinstance(coverage.get("pools"), dict)
            ):
                raise ValueError("Review coverage is malformed")
        except (KeyError, TypeError, ValueError) as error:
            failure_ref = self.repository.put_artifact(
                run_id,
                "analysis_review_attempt_failure",
                (
                    json.dumps(
                        {
                            "schema_version": 1,
                            "command_id": command.command_id,
                            "logical_command_id": command.logical_command_id,
                            "attempt_id": command.attempt_id,
                            "attempt_index": command.attempt_index,
                            "failure_kind": "harness_review_packet_invalid",
                            "error": str(error),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode(),
            )
            return self.state.apply(
                run_id,
                AnalysisReviewAttemptRetryPendingOutcome(
                    run_id=run_id,
                    coordinator_id=ref.coordinator_id,
                    plan_id=ref.plan_id,
                    trial_id=ref.trial_id,
                    basis_revision=state.revision,
                    command_id=command.command_id,
                    logical_command_id=command.logical_command_id,
                    attempt_id=command.attempt_id,
                    attempt_index=command.attempt_index,
                    receipt_id=f"receipt-{command.command_id}",
                    failure_ref=failure_ref,
                    failure_kind="harness_review_packet_invalid",
                    message=str(error),
                ),
                event_type="automatic_recovery_started",
            )
        if receipt.error:
            packet["worker_error"] = receipt.error
        packet_content = (json.dumps(packet, indent=2, sort_keys=True) + "\n").encode()
        coverage_content = (json.dumps(coverage, indent=2, sort_keys=True) + "\n").encode()
        packet_ref = self.repository.put_artifact(
            run_id, "analysis_review_packet", packet_content
        )
        coverage_ref = self.repository.put_artifact(
            run_id, "analysis_review_coverage", coverage_content
        )
        usage = packet.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        usage_status = str(usage.get("usage_status") or "unavailable")
        self.repository.append_usage(
            {
                "event_id": f"review-labor-command:{command_id}",
                "category": "analyzer_review",
                "component": "review_labor",
                "provider": "review_labor",
                "role": AgentRole.ANALYZER.value,
                "status": receipt.status.value,
                "run_id": run_id,
                "coordinator_id": ref.coordinator_id,
                "plan_id": ref.plan_id,
                "trial_id": ref.trial_id,
                "command_id": command_id,
                "basis_revision": command.basis_revision,
                "transition_intent": "analysis_review",
                "usage_status": usage_status,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cached_tokens": usage.get("cached_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "attempts": int(usage.get("attempts") or 0),
                "retries": int(usage.get("retries") or 0),
                "requests": int(usage.get("requests") or 0),
            }
        )
        return self.state.apply(
            run_id,
            AnalysisReviewCompletedOutcome(
                run_id=run_id,
                coordinator_id=ref.coordinator_id,
                plan_id=ref.plan_id,
                trial_id=ref.trial_id,
                basis_revision=state.revision,
                command_id=command_id,
                receipt_id=receipt.receipt_id,
                packet_ref=packet_ref,
                coverage_ref=coverage_ref,
            ),
            event_type="analysis_review_completed",
        )

    def analysis_review_receipt_requires_recovery(
        self,
        run_id: str,
        command_id: str,
    ) -> bool:
        """Report whether collecting this Receipt would start recovery.

        A Run has one durable RecoveryState.  The workflow uses this read-only
        probe to defer a second malformed Review Receipt until the current
        recovery has submitted its replacement Attempt.
        """
        if self.review_queue is None:
            raise ValueError("Review queue is not configured")
        state = self.repository.load(run_id)
        refs = tuple(
            item
            for item in state.active_review_commands
            if item.command_id == command_id
        )
        if len(refs) != 1:
            raise ValueError("Review command is not active")
        ref = refs[0]
        command = self.repository.load_review_command(
            run_id,
            ref.coordinator_id,
            ref.plan_id,
            ref.trial_id,
            command_id,
        )
        try:
            receipt = self.review_queue.load_receipt(command_id)
            if (
                receipt.command_id != command.command_id
                or receipt.logical_command_id != command.logical_command_id
                or receipt.attempt_id != command.attempt_id
                or receipt.attempt_index != command.attempt_index
                or receipt.run_id != command.run_id
                or receipt.coordinator_id != command.coordinator_id
                or receipt.plan_id != command.plan_id
                or receipt.trial_id != command.trial_id
            ):
                raise ValueError("Review Receipt crossed the active Attempt fence")
            packet = dict(receipt.packet)
            validate_review_packet(command, packet)
            coverage = coverage_from_packet(command, packet)
            if (
                coverage.get("command_id") != command.command_id
                or coverage.get("passed") is not True
                or not isinstance(coverage.get("pools"), dict)
            ):
                raise ValueError("Review coverage is malformed")
        except (KeyError, TypeError, ValueError):
            return True
        return False

    def retry_analysis_review(self, run_id: str, trial_key: TrialKey):
        if self.review_queue is None:
            raise ValueError("Review queue is not configured")
        state = self.repository.load(run_id)
        trial = self._trial(state, trial_key)
        if (
            trial.phase is not TrialPhase.REVIEW_RUNNING
            or not trial.analysis_review_retry_pending
            or trial.analysis_review_logical_command_id is None
            or trial.analysis_review_attempt_index < 1
        ):
            raise ValueError("Trial has no retry-pending Review Attempt")
        previous_index = trial.analysis_review_attempt_index
        previous_id = f"attempt-{previous_index:03d}"
        previous_command_id = (
            f"{trial.analysis_review_logical_command_id}-{previous_id}"
        )
        previous = self.repository.load_review_command(
            run_id,
            trial.coordinator_id,
            trial.plan_id,
            trial.trial_id,
            previous_command_id,
        )
        attempt_index = previous_index + 1
        attempt_id = f"attempt-{attempt_index:03d}"
        command = ReviewCommand(
            command_id=f"{previous.logical_command_id}-{attempt_id}",
            logical_command_id=previous.logical_command_id,
            attempt_id=attempt_id,
            attempt_index=attempt_index,
            run_id=previous.run_id,
            coordinator_id=previous.coordinator_id,
            plan_id=previous.plan_id,
            trial_id=previous.trial_id,
            basis_revision=state.revision,
            batches=previous.batches,
        )
        self.repository.store_review_command(command)
        committed = self.state.apply(
            run_id,
            AnalysisReviewAttemptSubmittedOutcome(
                run_id=run_id,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                trial_id=trial.trial_id,
                basis_revision=state.revision,
                previous_command_id=previous_command_id,
                command_ref=ReviewCommandRef(
                    command_id=command.command_id,
                    logical_command_id=command.logical_command_id,
                    attempt_id=command.attempt_id,
                    attempt_index=command.attempt_index,
                    coordinator_id=command.coordinator_id,
                    plan_id=command.plan_id,
                    trial_id=command.trial_id,
                ),
            ),
            event_type="automatic_recovery_attempt_submitted",
        )
        self.review_queue.submit(command)
        return committed

    def fail_analysis(
        self,
        run_id: str,
        trial_key: TrialKey,
        delivery,
        reason: str,
    ):
        if trial_key.run_id != run_id:
            raise ValueError("Trial belongs to another run")
        state = self.repository.load(run_id)
        trial = self._trial(state, trial_key)
        observed = observed_analysis_failure(delivery.workspace)
        payload = {
            "schema_version": "1",
            "coordinator_id": trial_key.coordinator_id,
            "plan_id": trial_key.plan_id,
            "trial_id": trial_key.trial_id,
            "call_id": delivery.call.call_id,
            "attempt_id": delivery.attempt.attempt_id,
            "reason": reason,
            "violations": [
                {
                    "code": item.code,
                    "message": item.message,
                    "path": item.path,
                    "repairable": item.repairable,
                }
                for item in delivery.validation.violations
            ],
            **observed,
            "last_workspace_uri": delivery.attempt.workspace_uri,
        }
        ref = self.repository.put_artifact(
            run_id,
            "analysis_failure",
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        (
            snapshot_ref,
            objective_comparison_ref,
            offline_score,
            offline_secondary_score,
        ) = self._analysis_packet(
            state,
            trial_key,
            files=(
                SnapshotFile.generated(
                    "failure.json",
                    self.repository.read_artifact(run_id, ref),
                    source_id=ref.artifact_id,
                ),
            ),
            evidence_status="unavailable",
            outcome=trial.outcome.value,
        )
        return self.state.apply(
            run_id,
            AnalysisFailedOutcome(
                run_id=run_id,
                coordinator_id=trial_key.coordinator_id,
                plan_id=trial_key.plan_id,
                trial_id=trial_key.trial_id,
                basis_revision=state.revision,
                call_id=delivery.call.call_id,
                failure_ref=ref,
                snapshot_ref=snapshot_ref,
                objective_comparison_ref=objective_comparison_ref,
                offline_score=offline_score,
                offline_secondary_score=offline_secondary_score,
                agent_session=delivery.session,
            ),
            event_type="analysis_defaulted",
        )

    def _analysis_packet(
        self,
        state,
        trial_key: TrialKey,
        *,
        files: tuple[SnapshotFile, ...],
        evidence_status: str,
        outcome: str,
    ):
        if self.snapshots is None:
            raise ValueError("Trial Analysis Packet store is not configured")
        trial = self._trial(state, trial_key)
        (
            result_content,
            offline_score,
            offline_secondary_score,
            result_source_id,
        ) = self._offline_result(trial)
        objective_content = self._objective_comparison(
            state,
            trial_key,
            offline_score,
            offline_secondary_score,
        )
        objective_ref = self.repository.put_artifact(
            state.run_id,
            "objective_comparison",
            objective_content,
        )
        state_content = (
            json.dumps(
                {
                    "schema_version": "ade.trial_analysis_packet_state.v1",
                    "run_id": state.run_id,
                    "coordinator_id": trial.coordinator_id,
                    "plan_id": trial.plan_id,
                    "trial_id": trial.trial_id,
                    "basis_revision": state.revision,
                    "outcome": outcome,
                    "evidence_status": evidence_status,
                    "metric_id": state.ranking.metric_id,
                    "offline_score": offline_score,
                    "offline_secondary_score": offline_secondary_score,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        snapshot_ref = self.snapshots.materialize(
            kind=SnapshotKind.TRIAL,
            run_id=state.run_id,
            coordinator_id=trial.coordinator_id,
            plan_id=trial.plan_id,
            trial_id=trial.trial_id,
            revision=state.revision,
            files=(
                SnapshotFile.generated(
                    "offline-eval-result.json",
                    result_content,
                    source_id=result_source_id,
                ),
                SnapshotFile.generated(
                    "state.json",
                    state_content,
                    source_id=f"harness-state-r{state.revision}",
                ),
                SnapshotFile.generated(
                    "objective-comparison.json",
                    objective_content,
                    source_id=objective_ref.artifact_id,
                ),
                *files,
            ),
        )
        return (
            snapshot_ref,
            objective_ref,
            offline_score,
            offline_secondary_score,
        )

    def _objective_comparison(
        self,
        state,
        trial_key: TrialKey,
        score: float | None,
        secondary_score: float | None,
    ) -> bytes:
        plan = next(
            item
            for item in state.plans
            if item.coordinator_id == trial_key.coordinator_id
            and item.plan_id == trial_key.plan_id
        )
        profile = "offline"
        direction = state.ranking.direction
        candidate = (score, secondary_score)

        def comparable(binding, primary, secondary) -> bool:
            return (
                isinstance(binding, dict)
                and binding.get("evaluation_profile") == profile
                and isinstance(score, (int, float))
                and math.isfinite(score)
                and (secondary_score is None or math.isfinite(secondary_score))
                and isinstance(primary, (int, float))
                and math.isfinite(primary)
                and (secondary is None or isinstance(secondary, (int, float)))
                and (secondary is None or math.isfinite(secondary))
            )

        def metric_relation(candidate_value, reference_value) -> str:
            if (
                type(candidate_value) not in {int, float}
                or type(reference_value) not in {int, float}
                or not math.isfinite(candidate_value)
                or not math.isfinite(reference_value)
            ):
                return "not_comparable"
            if candidate_value == reference_value:
                return "tied"
            improved = (
                candidate_value > reference_value
                if direction == "maximize"
                else candidate_value < reference_value
            )
            return "improved" if improved else "regressed"

        hypothesis = plan.hypothesis_comparator
        hypothesis_primary = (
            hypothesis.get("primary", {}).get("value")
            if isinstance(hypothesis, dict)
            and isinstance(hypothesis.get("primary"), dict)
            else None
        )
        hypothesis_secondary = (
            hypothesis.get("secondary", {}).get("value")
            if isinstance(hypothesis, dict)
            and isinstance(hypothesis.get("secondary"), dict)
            else None
        )
        hypothesis_result = "not_comparable"
        hypothesis_metric_relations = {
            "primary": "not_comparable",
            "secondary": "not_comparable",
        }
        if (
            isinstance(hypothesis, dict)
            and hypothesis.get("evaluation_profile") == profile
        ):
            hypothesis_metric_relations = {
                "primary": metric_relation(score, hypothesis_primary),
                "secondary": metric_relation(
                    secondary_score, hypothesis_secondary
                ),
            }
        if comparable(hypothesis, hypothesis_primary, hypothesis_secondary):
            candidate_key = score_pair_sort_key(*candidate, direction=direction)
            reference_key = score_pair_sort_key(
                hypothesis_primary,
                hypothesis_secondary,
                direction=direction,
            )
            hypothesis_result = (
                "improved"
                if candidate_key < reference_key
                else "tied"
                if candidate_key == reference_key
                else "regressed"
            )

        portfolio = plan.portfolio_comparator
        target = (
            portfolio.get("target") if isinstance(portfolio, dict) else None
        )
        target_primary = (
            target.get("primary", {}).get("value")
            if isinstance(target, dict)
            and isinstance(target.get("primary"), dict)
            else None
        )
        target_secondary = (
            target.get("secondary", {}).get("value")
            if isinstance(target, dict)
            and isinstance(target.get("secondary"), dict)
            else None
        )
        portfolio_result = "not_comparable"
        if (
            isinstance(portfolio, dict)
            and portfolio.get("direction") == direction
            and isinstance(target, dict)
            and comparable(portfolio, target_primary, target_secondary)
        ):
            portfolio_result = (
                "new_best"
                if score_pair_sort_key(*candidate, direction=direction)
                < score_pair_sort_key(
                    target_primary,
                    target_secondary,
                    direction=direction,
                )
                else "not_new_best"
            )

        payload = {
            "schema_version": "1",
            "trial": (
                f"{state.run_id}/{trial_key.coordinator_id}/"
                f"{trial_key.plan_id}/{trial_key.trial_id}"
            ),
            "evaluation_profile": profile,
            "ranking_values": {
                "ranking_score": score,
                "secondary_score": secondary_score,
            },
            "hypothesis_comparator": {
                "subject_id": (
                    hypothesis.get("subject_id")
                    if isinstance(hypothesis, dict)
                    else None
                ),
                "result": hypothesis_result,
                "metric_relations": hypothesis_metric_relations,
            },
            "portfolio_comparator": {
                "subject_id": (
                    portfolio.get("subject_id")
                    if isinstance(portfolio, dict)
                    else None
                ),
                "ranking_revision": (
                    portfolio.get("ranking_revision")
                    if isinstance(portfolio, dict)
                    else None
                ),
                "result": portfolio_result,
            },
        }
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()

    def _offline_result(self, trial):
        if not trial.result_refs:
            unavailable = b'{"schema_version":"1","status":"unavailable"}\n'
            return unavailable, None, None, "engine-result-unavailable"
        source_ref = trial.result_refs[0]
        content = self.engine_io.read_bytes(source_ref)
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return content, None, None, source_ref
        score = None
        secondary_score = None
        if isinstance(value, dict):
            offline = value.get("offline_validation")
            candidate = (
                offline.get("ranking_score")
                if isinstance(offline, dict)
                else value.get("ranking_score", value.get("score"))
            )
            if type(candidate) in {int, float} and math.isfinite(candidate):
                score = float(candidate)
            secondary_candidate = (
                offline.get("secondary_score")
                if isinstance(offline, dict)
                else value.get("secondary_score")
            )
            if (
                type(secondary_candidate) in {int, float}
                and math.isfinite(secondary_candidate)
            ):
                secondary_score = float(secondary_candidate)
        return content, score, secondary_score, source_ref

    def _latest_trial_snapshot_refs(
        self,
        state,
        *,
        coordinator_id: str,
        plan_id: str,
    ) -> tuple[SnapshotRef, ...]:
        refs = tuple(
            ref
            for ref in state.accepted_snapshot_refs
            if ref.kind is SnapshotKind.TRIAL
            and ref.coordinator_id == coordinator_id
            and ref.plan_id == plan_id
        )
        latest_by_trial: dict[str | None, SnapshotRef] = {}
        for ref in refs:
            current = latest_by_trial.get(ref.trial_id)
            if current is None or ref.revision > current.revision:
                latest_by_trial[ref.trial_id] = ref
        return tuple(
            ref for ref in refs if latest_by_trial.get(ref.trial_id) is ref
        )

    def _plan_summary_sources(self, state, trial) -> dict[str, str]:
        memories = getattr(self.repository, "memory_versions", None)
        records = getattr(self.repository, "trial_records", None)
        if memories is None or records is None:
            raise ValueError("Plan Summary requires Memory and Trial Record stores")
        plan = self._plan(
            state,
            PlanKey(state.run_id, trial.coordinator_id, trial.plan_id),
        )
        parent = memories.read_plan_version(
            run_id=state.run_id,
            coordinator_id=trial.coordinator_id,
            plan_id=trial.plan_id,
            memory_id=trial.plan_memory_basis.rsplit("/", 1)[-1],
        )
        record = records.read_files(
            run_id=state.run_id,
            coordinator_id=trial.coordinator_id,
            plan_id=trial.plan_id,
            trial_id=trial.trial_id,
        )
        sources = {
            str(trial.plan_memory_basis): hashlib.sha256(
                parent["manifest.json"]
            ).hexdigest(),
            str(trial.trial_record_ref): hashlib.sha256(
                record["manifest.json"]
            ).hexdigest(),
        }
        if plan.kind is PlanKind.SEARCH:
            plan_ref = self._accepted_ref_by_id(
                state.accepted_plan_refs,
                plan.decision_report_ref_id,
                "immutable Plan",
            )
            sources[plan_ref.uri] = plan_ref.digest
        return sources

    @staticmethod
    def _trial_summary_snapshot(state, trial) -> tuple[SnapshotRef, ...]:
        matches = tuple(
            ref
            for ref in state.accepted_snapshot_refs
            if ref.kind is SnapshotKind.TRIAL
            and ref.snapshot_id == trial.analysis_packet_ref_id
        )
        if len(matches) != 1:
            raise ValueError("current Trial has no accepted analysis snapshot")
        return matches

    def _run_summary_sources(self, state, trial, delivery) -> dict[str, str]:
        memories = getattr(self.repository, "memory_versions", None)
        if memories is None:
            raise ValueError("Run Summary requires Memory store")
        parent_id = state.memory.run_head
        parent = memories.read_run_version(
            run_id=state.run_id,
            memory_id=parent_id.rsplit("/", 1)[-1],
        )
        plan_id = str(trial.plan_memory_result)
        plan_memory = memories.read_plan_version(
            run_id=state.run_id,
            coordinator_id=trial.coordinator_id,
            plan_id=trial.plan_id,
            memory_id=plan_id.rsplit("/", 1)[-1],
        )
        queue_head = state.rm_merge_queue[0]
        update = (
            "# Plan Memory Update\n\n"
            f"- Queue entry: `{queue_head}`\n"
            f"- Plan: `{state.run_id}/{trial.coordinator_id}/{trial.plan_id}`\n"
            f"- Trial: `{trial.trial_record_ref}`\n"
            f"- Plan Memory: `{trial.plan_memory_result}`\n"
            f"- Run Memory parent: `{state.memory.run_head}`\n"
            f"- Basis revision: `{delivery.call.basis_revision}`\n"
        ).encode()
        input_root = delivery.workspace / "input"
        facts_content = (input_root / "run/manifest.json").read_bytes()
        facts = json.loads(facts_content)
        if (
            facts.get("identity") != f"{state.run_id}@rev-{delivery.call.basis_revision}"
            or facts.get("basis_revision") != delivery.call.basis_revision
        ):
            raise ValueError("Run Summary facts basis is invalid")
        for path, digest in facts.get("files", {}).items():
            content = (input_root / "run" / path).read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("Run Summary facts content is invalid")
        return {
            parent_id: hashlib.sha256(parent["manifest.json"]).hexdigest(),
            plan_id: hashlib.sha256(plan_memory["manifest.json"]).hexdigest(),
            f"{queue_head}#rm-merge": hashlib.sha256(update).hexdigest(),
            str(facts["identity"]): hashlib.sha256(facts_content).hexdigest(),
        }

    @staticmethod
    def _plan_summary_snapshot(state, trial) -> tuple[SnapshotRef, ...]:
        matches = tuple(
            ref
            for ref in state.accepted_snapshot_refs
            if ref.kind is SnapshotKind.PLAN
            and ref.snapshot_id == trial.plan_snapshot_ref_id
        )
        if len(matches) != 1:
            raise ValueError("queue-head Trial has no accepted Plan snapshot")
        return matches

    def accept_summary(
        self,
        run_id: str,
        delivery: AcceptedCall,
    ):
        state = self.repository.load(run_id)
        if delivery.call.run_id != run_id:
            raise ValueError("summary delivery belongs to another run")
        self._validate_active_call(state, delivery)
        if delivery.call.role is AgentRole.PLAN_SUMMARIZER:
            if not isinstance(delivery.output, PlanSummary):
                raise ValueError("Plan Summarizer requires typed PlanSummary output")
            summary_kind = "plan_summary"
        elif delivery.call.role is AgentRole.RUN_SUMMARIZER:
            if not isinstance(delivery.output, RunSummary):
                raise ValueError("Run Summarizer requires typed RunSummary output")
            summary_kind = "run_summary"
        else:
            raise ValueError("summary delivery requires a summarizer role")
        summary = delivery.output
        if summary_kind == "plan_summary":
            trial = self._pending_summary_trial(
                state,
                TrialPhase.PLAN_SUMMARIZING,
                coordinator_id=delivery.call.coordinator_id,
                plan_id=summary.subject_id,
            )
            sources = self._trial_summary_snapshot(state, trial)
            expected = self._plan_summary_sources(state, trial)
        else:
            trial = self._pending_summary_trial(state, TrialPhase.RUN_SUMMARIZING)
            sources = self._plan_summary_snapshot(state, trial)
            expected = self._run_summary_sources(state, trial, delivery)
        received = dict(summary.source_manifest_digests)
        if received != expected:
            raise SummaryAdmissionError(
                ValidationReport(
                    (
                        DeliveryViolation(
                            code="unauthorized_summary_source",
                            message=(
                                "summary sources do not equal frozen accepted inputs: "
                                f"expected {sorted(expected)}, received {sorted(received)}"
                            ),
                            path="evidence.json#/sources",
                            repairable=True,
                        ),
                    )
                )
            )
        if summary_kind == "plan_summary" and trial.kind is TrialKind.SEARCH:
            self._validate_plan_conclusions(state, trial, summary.content)
        elif summary_kind == "run_summary" and trial.kind is TrialKind.SEARCH:
            self._validate_run_conclusions(state, trial, summary.content)
        ref = self.repository.put_artifact(run_id, summary_kind, summary.content)
        evidence_ref = self.repository.put_artifact(
            run_id,
            f"{summary_kind}_evidence",
            summary.evidence_content,
        )
        snapshot_ref = self._summary_snapshot(
            state,
            trial,
            summary_kind=summary_kind,
            summary_ref=ref,
            evidence_ref=evidence_ref,
            sources=sources,
        )
        return self.state.apply(
            run_id,
            SummaryAcceptedOutcome(
                run_id=run_id,
                coordinator_id=delivery.call.coordinator_id,
                subject_id=summary.subject_id,
                basis_revision=state.revision,
                call_id=delivery.call.call_id,
                summary_kind=summary_kind,
                summary_ref=ref,
                evidence_ref=evidence_ref,
                snapshot_ref=snapshot_ref,
                trial_id=trial.trial_id,
                trial_coordinator_id=trial.coordinator_id,
                trial_plan_id=trial.plan_id,
                agent_session=delivery.session,
            ),
            event_type="summary_accepted",
        )

    def default_summary(self, run_id: str, delivery, reason: str):
        state = self.repository.load(run_id)
        self._validate_active_call(state, delivery)
        role = delivery.call.role
        if role is AgentRole.PLAN_SUMMARIZER:
            summary_kind = "plan_summary"
            trial = self._pending_summary_trial(
                state,
                TrialPhase.PLAN_SUMMARIZING,
                coordinator_id=delivery.call.coordinator_id,
                plan_id=delivery.call.plan_id,
            )
            sources = self._trial_summary_snapshot(state, trial)
            bound_sources = self._plan_summary_sources(state, trial)
            subject_id = trial.plan_id
        elif role is AgentRole.RUN_SUMMARIZER:
            summary_kind = "run_summary"
            trial = self._pending_summary_trial(state, TrialPhase.RUN_SUMMARIZING)
            sources = self._plan_summary_snapshot(state, trial)
            bound_sources = self._run_summary_sources(state, trial, delivery)
            subject_id = state.run_id
        else:
            raise ValueError("default summary requires a Summarizer Call")
        memories = getattr(self.repository, "memory_versions", None)
        if memories is None:
            raise ValueError("default Summary requires Memory store")
        if summary_kind == "plan_summary":
            parent_content = memories.read_plan_version(
                run_id=state.run_id,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                memory_id=trial.plan_memory_basis.rsplit("/", 1)[-1],
            )["MEMORY.md"]
            if trial.kind is TrialKind.SEARCH:
                expected_realization, expected_portfolio = (
                    self._expected_trial_conclusions(state, trial)
                )
                conclusion_content = (
                    "\n\n## Trial Conclusions\n\n"
                    f"Realization status: {expected_realization}\n"
                    "Hypothesis result: inconclusive\n"
                    f"Portfolio result: {expected_portfolio}\n"
                ).encode()
            else:
                conclusion_content = b""
        else:
            parent_content = memories.read_run_version(
                run_id=state.run_id,
                memory_id=state.memory.run_head.rsplit("/", 1)[-1],
            )["MEMORY.md"]
            if trial.kind is TrialKind.SEARCH:
                plan_memory = memories.read_plan_version(
                    run_id=state.run_id,
                    coordinator_id=trial.coordinator_id,
                    plan_id=trial.plan_id,
                    memory_id=trial.plan_memory_result.rsplit("/", 1)[-1],
                )["MEMORY.md"]
                conclusions = self._decode_conclusion_lines(plan_memory)
                conclusion_content = (
                    "\n\n## Trial Conclusions\n\n"
                    f"Realization status: {conclusions['realization']}\n"
                    f"Hypothesis result: {conclusions['hypothesis']}\n"
                    f"Portfolio result: {conclusions['portfolio']}\n"
                ).encode()
            else:
                conclusion_content = b""
        summary_content = (
            parent_content.rstrip()
            + b"\n\n## Harness Default Update\n\n"
            + f"No Agent-authored findings were admitted. Reason: {reason}\n".encode()
            + conclusion_content
        )
        evidence_content = (
            json.dumps(
                {
                    "schema_version": "1",
                    "status": "defaulted",
                    "reason": reason,
                    "sources": [
                        {"snapshot_id": source_id, "manifest_sha256": digest}
                        for source_id, digest in sorted(bound_sources.items())
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        ref = self.repository.put_artifact(run_id, summary_kind, summary_content)
        evidence_ref = self.repository.put_artifact(
            run_id, f"{summary_kind}_evidence", evidence_content
        )
        snapshot_ref = self._summary_snapshot(
            state,
            trial,
            summary_kind=summary_kind,
            summary_ref=ref,
            evidence_ref=evidence_ref,
            sources=sources,
        )
        return self.state.apply(
            run_id,
            SummaryAcceptedOutcome(
                run_id=run_id,
                coordinator_id=(
                    trial.coordinator_id if summary_kind == "plan_summary" else None
                ),
                subject_id=subject_id,
                basis_revision=state.revision,
                call_id=delivery.call.call_id,
                summary_kind=summary_kind,
                summary_ref=ref,
                evidence_ref=evidence_ref,
                snapshot_ref=snapshot_ref,
                trial_id=trial.trial_id,
                trial_coordinator_id=trial.coordinator_id,
                trial_plan_id=trial.plan_id,
                agent_session=delivery.session,
            ),
            event_type=f"{summary_kind}_defaulted",
        )

    def _validate_plan_conclusions(self, state, trial, content: bytes) -> None:
        conclusions = self._decode_conclusion_lines(content)
        expected_realization, expected_portfolio = (
            self._expected_trial_conclusions(state, trial)
        )
        violations = []
        if conclusions["realization"] != expected_realization:
            violations.append(
                DeliveryViolation(
                    "realization_conclusion_mismatch",
                    "Plan Summary realization status does not match final realization",
                    "MEMORY.md#Realization status",
                    repairable=True,
                )
            )
        if conclusions["portfolio"] != expected_portfolio:
            violations.append(
                DeliveryViolation(
                    "portfolio_conclusion_mismatch",
                    "Plan Summary portfolio result does not match objective comparison",
                    "MEMORY.md#Portfolio result",
                    repairable=True,
                )
            )
        if violations:
            raise SummaryAdmissionError(ValidationReport(tuple(violations)))

    def _validate_run_conclusions(self, state, trial, content: bytes) -> None:
        memories = getattr(self.repository, "memory_versions", None)
        if memories is None or trial.plan_memory_result is None:
            raise ValueError("Run Summary requires accepted Plan Memory")
        plan_content = memories.read_plan_version(
            run_id=state.run_id,
            coordinator_id=trial.coordinator_id,
            plan_id=trial.plan_id,
            memory_id=trial.plan_memory_result.rsplit("/", 1)[-1],
        )["MEMORY.md"]
        if self._decode_conclusion_lines(content) != self._decode_conclusion_lines(
            plan_content
        ):
            raise SummaryAdmissionError(
                ValidationReport(
                    (
                        DeliveryViolation(
                            "run_conclusion_mismatch",
                            "Run Summary conclusions do not preserve accepted Plan Memory",
                            "MEMORY.md#Trial Conclusions",
                            repairable=True,
                        ),
                    )
                )
            )

    def _expected_trial_conclusions(self, state, trial) -> tuple[str, str]:
        records = getattr(self.repository, "trial_records", None)
        if records is None:
            raise ValueError("Trial conclusions require Trial Record store")
        files = records.read_files(
            run_id=state.run_id,
            coordinator_id=trial.coordinator_id,
            plan_id=trial.plan_id,
            trial_id=trial.trial_id,
        )
        try:
            objective_path = objective_comparison_record_path(
                trial.engine_attempt_index
            )
            realization = json.loads(
                files["realization/final-realization.json"]
            )["realization_status"]
            portfolio = json.loads(
                files[objective_path]
            )["portfolio_comparator"]["result"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Trial conclusion evidence is incomplete") from error
        if realization not in {"verified", "deviated", "unverified"}:
            raise ValueError("final realization status is invalid")
        if portfolio not in {"new_best", "not_new_best", "not_comparable"}:
            raise ValueError("objective portfolio result is invalid")
        return str(realization), str(portfolio)

    @staticmethod
    def _decode_conclusion_lines(content: bytes) -> dict[str, str]:
        try:
            lines = content.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise SummaryAdmissionError(
                ValidationReport(
                    (
                        DeliveryViolation(
                            "invalid_conclusions",
                            "Summary conclusions must be UTF-8",
                            "MEMORY.md#Trial Conclusions",
                            repairable=True,
                        ),
                    )
                )
            ) from error
        section_indices = [
            index
            for index, line in enumerate(lines)
            if line.strip() == "## Trial Conclusions"
        ]
        if section_indices:
            lines = lines[section_indices[-1] + 1 :]
            next_section = next(
                (
                    index
                    for index, line in enumerate(lines)
                    if line.startswith("## ")
                ),
                len(lines),
            )
            lines = lines[:next_section]
        fields = {
            "realization": (
                "Realization status: ",
                {"verified", "deviated", "unverified"},
            ),
            "hypothesis": (
                "Hypothesis result: ",
                {"supported", "rejected", "inconclusive"},
            ),
            "portfolio": (
                "Portfolio result: ",
                {"new_best", "not_new_best", "not_comparable"},
            ),
        }
        result = {}
        for name, (prefix, allowed) in fields.items():
            values = [
                line.removeprefix(prefix)
                for line in lines
                if line.startswith(prefix)
            ]
            if len(values) != 1 or values[0] not in allowed:
                raise SummaryAdmissionError(
                    ValidationReport(
                        (
                            DeliveryViolation(
                                "invalid_trial_conclusions",
                                f"Summary requires exactly one valid {prefix.strip()}",
                                f"MEMORY.md#{prefix.strip()}",
                                repairable=True,
                            ),
                        )
                    )
                )
            result[name] = values[0]
        return result

    def _summary_snapshot(
        self,
        state,
        trial,
        *,
        summary_kind: str,
        summary_ref,
        evidence_ref,
        sources: tuple[SnapshotRef, ...],
    ) -> SnapshotRef:
        if self.snapshots is None:
            raise ValueError("summary snapshot store is not configured")
        state_payload = self._summary_state_payload(
            state,
            trial,
            summary_kind=summary_kind,
        )
        files = [
            SnapshotFile.generated(
                "MEMORY.md", self.repository.read_artifact(state.run_id, summary_ref),
                source_id=summary_ref.artifact_id,
            ),
            SnapshotFile.generated(
                "evidence.json", self.repository.read_artifact(state.run_id, evidence_ref),
                source_id=evidence_ref.artifact_id,
            ),
            SnapshotFile.generated(
                "state.json",
                (json.dumps(state_payload, indent=2, sort_keys=True) + "\n").encode(),
                source_id=f"harness-state-r{state.revision}",
            ),
        ]
        for source in sources:
            prefix = (
                f"trials/{source.trial_id}"
                if summary_kind == "plan_summary"
                else f"coordinators/{source.coordinator_id}/plans/{source.plan_id}"
            )
            if summary_kind == "plan_summary":
                latest_revision = max(
                    item.revision
                    for item in sources
                    if item.trial_id == source.trial_id
                )
                if source.revision != latest_revision:
                    prefix = f"{prefix}/revisions/r{source.revision:03d}"
            files.extend(self._inherited_snapshot_files(source, prefix))
        if summary_kind == "plan_summary":
            snapshot_revision = state.revision
            existing_snapshot_ids = {
                ref.snapshot_id for ref in state.accepted_snapshot_refs
            }
            while (
                f"plan-{trial.coordinator_id}-{trial.plan_id}-r"
                f"{snapshot_revision:03d}"
            ) in existing_snapshot_ids or self.repository.layout.plan_snapshot_dir(
                state.run_id,
                trial.coordinator_id,
                trial.plan_id,
                snapshot_revision,
            ).exists():
                snapshot_revision += 1
            return self.snapshots.materialize(
                kind=SnapshotKind.PLAN,
                run_id=state.run_id,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                revision=snapshot_revision,
                files=tuple(files),
            )
        snapshot_revision = state.revision
        existing_snapshot_ids = {
            ref.snapshot_id for ref in state.accepted_snapshot_refs
        }
        while (
            f"run-r{snapshot_revision:03d}" in existing_snapshot_ids
            or self.repository.layout.run_snapshot_dir(
                state.run_id, snapshot_revision
            ).exists()
        ):
            snapshot_revision += 1
        return self.snapshots.materialize(
            kind=SnapshotKind.RUN,
            run_id=state.run_id,
            revision=snapshot_revision,
            files=tuple(files),
        )

    def _summary_state_payload(self, state, trial, *, summary_kind: str):
        plan = self._plan(
            state,
            PlanKey(state.run_id, trial.coordinator_id, trial.plan_id),
        )
        progress = self.reducer._plan_progress(
            state,
            trial.coordinator_id,
            trial.plan_id,
        )
        projected_status = plan.status
        if summary_kind == "run_summary":
            projected_status = {
                PlanTerminalDecision.CONTINUE: PlanStatus.ACTIVE,
                PlanTerminalDecision.COMPLETED: PlanStatus.COMPLETED,
                PlanTerminalDecision.FAILED: PlanStatus.FAILED,
            }[progress.decision]
            if plan.kind is PlanKind.BOOTSTRAP:
                projected_status = PlanStatus.COMPLETED
        search_plans = sum(item.kind is PlanKind.SEARCH for item in state.plans)
        search_trials = sum(item.kind is TrialKind.SEARCH for item in state.trials)
        plan_trials = sum(
            item.coordinator_id == plan.coordinator_id
            and item.plan_id == plan.plan_id
            for item in state.trials
        )
        budget = {
            "max_search_plans": state.portfolio.max_plans,
            "allocated_search_plans": search_plans,
            "remaining_search_plans": max(0, state.portfolio.max_plans - search_plans),
            "max_search_trials": state.portfolio.max_trials,
            "allocated_search_trials": search_trials,
            "remaining_search_trials": max(
                0,
                state.portfolio.max_trials - search_trials,
            ),
            "min_trials_per_plan": state.portfolio.min_trials_per_plan,
            "max_trials_per_plan": state.portfolio.max_trials_per_plan,
            "no_improvement_patience": state.portfolio.no_improvement_patience,
            "allocated_plan_trials": plan_trials,
            "remaining_plan_trials": max(
                0,
                state.portfolio.max_trials_per_plan - plan_trials,
            ),
        }
        ranking = {
            "metric_id": state.ranking.metric_id,
            "direction": state.ranking.direction,
            "entries": [
                {
                    "subject_id": item.subject_id,
                    "score": item.score,
                    "secondary_score": item.secondary_score,
                }
                for item in state.ranking.entries
            ],
        }
        trial_payload = {
            "coordinator_id": trial.coordinator_id,
            "plan_id": trial.plan_id,
            "trial_id": trial.trial_id,
            "outcome": trial.outcome.value,
            "offline_score": trial.offline_score,
            "offline_secondary_score": trial.offline_secondary_score,
            "archive_status": (
                TrialArchiveStatus.PENDING.value
                if summary_kind == "plan_summary"
                else TrialArchiveStatus.ARCHIVED.value
            ),
        }
        plan_payload = {
            "coordinator_id": plan.coordinator_id,
            "plan_id": plan.plan_id,
            "kind": plan.kind.value,
            "status": projected_status.value,
            "planning_basis": plan.planning_basis,
            "hypothesis_comparator": plan.hypothesis_comparator,
            "portfolio_comparator": plan.portfolio_comparator,
            "best_trial_id": progress.best_trial_id,
            "best_artifact_ref_id": progress.best_artifact_ref_id,
            "best_score": progress.best_score,
            "best_secondary_score": progress.best_secondary_score,
            "no_improvement_count": progress.no_improvement_count,
        }
        common = {
            "schema_version": "ade.summary_snapshot_state.v2",
            "basis_revision": state.revision,
            "run": {
                "run_id": state.run_id,
                "status": state.status.value,
            },
            "trial": trial_payload,
            "ranking": ranking,
            "budget": budget,
        }
        if summary_kind == "plan_summary":
            return {**common, "plan": plan_payload}
        plans = []
        for item in state.plans:
            if item.coordinator_id == plan.coordinator_id and item.plan_id == plan.plan_id:
                plans.append(plan_payload)
            else:
                plans.append(
                    {
                        "coordinator_id": item.coordinator_id,
                        "plan_id": item.plan_id,
                        "kind": item.kind.value,
                        "status": item.status.value,
                        "planning_basis": item.planning_basis,
                        "hypothesis_comparator": item.hypothesis_comparator,
                        "portfolio_comparator": item.portfolio_comparator,
                        "best_trial_id": item.best_trial_id,
                        "best_artifact_ref_id": item.best_artifact_ref_id,
                        "best_score": item.best_score,
                        "best_secondary_score": item.best_secondary_score,
                        "no_improvement_count": item.no_improvement_count,
                    }
                )
        return {**common, "plans": plans}

    @staticmethod
    def _inherited_snapshot_files(
        source: SnapshotRef,
        prefix: str,
    ) -> tuple[SnapshotFile, ...]:
        root = Path(source.root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        paths = ("manifest.json", *(
            str(item["path"])
            for item in manifest["files"]
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        ))
        return tuple(
            SnapshotFile.inherited(
                f"{prefix}/{path}",
                root / path,
                source_id=source.snapshot_id,
            )
            for path in paths
        )

    @staticmethod
    def _pending_summary_trial(
        state,
        phase: TrialPhase,
        *,
        coordinator_id: str | None = None,
        plan_id: str | None = None,
    ):
        matches = [
            trial for trial in state.trials
            if trial.phase is phase
            and (coordinator_id is None or trial.coordinator_id == coordinator_id)
            and (plan_id is None or trial.plan_id == plan_id)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one Trial in {phase.value}, found {len(matches)}")
        return matches[0]

    @staticmethod
    def _validate_active_call(state, delivery) -> None:
        matches = tuple(
            item
            for item in state.active_agent_calls
            if item.call_id == delivery.call.call_id
        )
        if len(matches) != 1:
            raise ValueError("Agent delivery has no unique active Call")
        active = matches[0]
        if (
            active.run_id != delivery.call.run_id
            or active.session_id != delivery.call.session_id
            or active.attempt_id != delivery.attempt.attempt_id
            or active.basis_revision != delivery.call.basis_revision
            or active.role != delivery.call.role.value
        ):
            raise ValueError("Agent delivery does not match active Call fence")

    @staticmethod
    def _trial(state, key: TrialKey):
        matches = [
            trial
            for trial in state.trials
            if trial.coordinator_id == key.coordinator_id
            and trial.plan_id == key.plan_id
            and trial.trial_id == key.trial_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "unknown or duplicate Trial scope: "
                f"{key.coordinator_id}/{key.plan_id}/{key.trial_id}"
            )
        return matches[0]

    @staticmethod
    def _plan(state, key: PlanKey):
        matches = [
            plan
            for plan in state.plans
            if plan.coordinator_id == key.coordinator_id
            and plan.plan_id == key.plan_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "unknown or duplicate Plan scope: "
                f"{key.coordinator_id}/{key.plan_id}"
            )
        return matches[0]

    @staticmethod
    def _accepted_ref_by_id(refs, artifact_id, label):
        if not artifact_id:
            raise ValueError(f"{label} reference is required")
        matches = [ref for ref in refs if ref.artifact_id == artifact_id]
        if not matches:
            raise ValueError(f"{label} reference is not accepted: {artifact_id}")
        if any(ref != matches[0] for ref in matches[1:]):
            raise ValueError(f"{label} accepted references conflict: {artifact_id}")
        return matches[0]

    @staticmethod
    def _artifact_by_id(state, artifact_id):
        matches = tuple(dict.fromkeys(
            ref
            for ref in (*state.accepted_evidence_refs, *state.accepted_finding_refs)
            if ref.artifact_id == artifact_id
        ))
        if len(matches) != 1:
            raise ValueError(f"artifact ref is not admitted in RunState: {artifact_id}")
        return matches[0]
