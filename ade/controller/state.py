"""The single accepted-outcome commit path for RunState."""

from __future__ import annotations

from dataclasses import replace
import json
import time
from typing import Any

from ade.controller.ports import RunRepository
from ade.controller.reducer import Reducer
from ade.core.outcomes import (
    AnalysisAcceptedOutcome,
    AnalysisFailedOutcome,
    AnalysisReviewAttemptRetryPendingOutcome,
    AnalysisReviewCompletedOutcome,
    AnalysisReviewSubmittedOutcome,
    ArtifactAcceptedOutcome,
    BaselineTrialRegisteredOutcome,
    BootstrapP000RegisteredOutcome,
    CoordinatorCancelRequestedOutcome,
    CoordinatorTrialCancelledOutcome,
    CoordinatorFinishRequestedOutcome,
    BuilderFailedOutcome,
    EngineCompletedOutcome,
    EngineFailedOutcome,
    EngineAttemptRetryPendingOutcome,
    BootstrapBaseAttemptRetryPendingOutcome,
    OperatorEvaluationAttemptRetryPendingOutcome,
    PlanningDecisionOutcome,
    RunCancelledOutcome,
    RunFinishRequestedOutcome,
    RunPauseRequestedOutcome,
    SummaryAcceptedOutcome,
    TrialProposedOutcome,
)
from ade.core.coordinator import CoordinatorControlStatus
from ade.core.coordinator_control import allocated_plan_slots
from ade.core.artifacts import ArtifactRef
from ade.core.run import FactClass, RunState, TransitionRecord, fact_class_for_transition
from ade.core.scope import subject_ref
from ade.core.snapshot import SnapshotRef
from ade.memory.records import objective_comparison_record_path


class StateCoordinator:
    def __init__(self, repository: RunRepository, reducer: Reducer | None = None) -> None:
        self.repository = repository
        self.reducer = reducer or Reducer()

    def apply(self, run_id: str, outcome: Any, *, event_type: str) -> RunState:
        state = self.repository.load(run_id)
        if isinstance(
            outcome,
            (
                EngineAttemptRetryPendingOutcome,
                BootstrapBaseAttemptRetryPendingOutcome,
                OperatorEvaluationAttemptRetryPendingOutcome,
                AnalysisReviewAttemptRetryPendingOutcome,
            ),
        ) and outcome.observed_at <= 0:
            outcome = replace(outcome, observed_at=time.time())
        updated = self.reducer.apply(state, outcome)
        subject = self._subject_ref(outcome, state.run_id)
        fact_class = fact_class_for_transition(event_type)
        accepted_refs, origin_refs = self._transition_refs(
            outcome,
            fact_class,
            event_type=event_type,
            subject=subject,
        )
        return self._commit_transition(
            state,
            updated,
            event_type=event_type,
            subject_ref=subject,
            fact_class=fact_class,
            logical_work_ref=self._logical_work_ref(state, outcome, subject),
            accepted_fact_refs=accepted_refs,
            origin_refs=origin_refs,
            accepted_outcome=outcome,
        )

    def _commit_transition(
        self,
        state: RunState,
        updated: RunState,
        *,
        event_type: str,
        subject_ref: str | None = None,
        fact_class: FactClass = FactClass.OPERATIONAL,
        logical_work_ref: str | None = None,
        accepted_fact_refs: tuple[str, ...] = (),
        origin_refs: tuple[str, ...] = (),
        accepted_outcome: Any | None = None,
    ) -> RunState:
        if updated.run_id != state.run_id:
            raise ValueError("transition cannot change Run identity")
        if updated.revision != state.revision + 1:
            raise ValueError("transition must advance exactly one revision")
        updated = replace(
            updated,
            last_transition=TransitionRecord.create(
                run_id=state.run_id,
                kind=event_type,
                subject_ref=subject_ref or state.run_id,
                from_revision=state.revision,
                to_revision=updated.revision,
                fact_class=fact_class,
                logical_work_ref=logical_work_ref,
                accepted_fact_refs=accepted_fact_refs,
                origin_refs=origin_refs,
            ),
        )
        self._publish_control_owned_files(
            state,
            updated,
            event_type,
            subject_ref,
            outcome=accepted_outcome,
        )
        committed = self.repository.commit(
            updated,
            expected_revision=state.revision,
            event_type=event_type,
        )
        admission = getattr(self.repository, "run_resource_admission", None)
        if committed.status.value in {"completed", "failed", "cancelled"} and admission is not None:
            admission.terminal(committed.run_id)
        return committed

    def cancel(self, run_id: str, *, reason: str) -> RunState:
        state = self.repository.load(run_id)
        return self.apply(
            run_id,
            RunCancelledOutcome(
                run_id=run_id,
                basis_revision=state.revision,
                reason=reason,
            ),
            event_type="run_cancelled",
        )

    def request_pause(self, run_id: str, *, reason: str) -> RunState:
        state = self.repository.load(run_id)
        return self.apply(
            run_id,
            RunPauseRequestedOutcome(
                run_id=run_id,
                basis_revision=state.revision,
                reason=reason,
            ),
            event_type="run_pause_requested",
        )

    def request_coordinator_finish(
        self,
        run_id: str,
        *,
        coordinator_id: str,
        requested_plan_limit: int | None,
        reason: str,
    ) -> RunState:
        state = self.repository.load(run_id)
        coordinator = next(
            (
                item
                for item in state.coordinators
                if item.coordinator_id == coordinator_id
            ),
            None,
        )
        if coordinator is None:
            raise ValueError("unknown Coordinator")
        target = (
            allocated_plan_slots(state, coordinator_id)
            if requested_plan_limit is None
            else requested_plan_limit
        )
        normalized_reason = reason.strip()
        if coordinator.control_status in {
            CoordinatorControlStatus.FINISH_REQUESTED,
            CoordinatorControlStatus.FINISHED_EARLY,
        }:
            if (
                coordinator.requested_plan_limit == target
                and coordinator.control_reason == normalized_reason
            ):
                return state
            raise ValueError("Coordinator already has a different finish intent")
        return self.apply(
            run_id,
            CoordinatorFinishRequestedOutcome(
                run_id=run_id,
                coordinator_id=coordinator_id,
                basis_revision=state.revision,
                requested_plan_limit=target,
                reason=normalized_reason,
            ),
            event_type="coordinator_finish_requested",
        )

    def request_coordinator_cancel(
        self,
        run_id: str,
        *,
        coordinator_id: str,
        reason: str,
    ) -> RunState:
        state = self.repository.load(run_id)
        coordinator = next(
            (
                item
                for item in state.coordinators
                if item.coordinator_id == coordinator_id
            ),
            None,
        )
        if coordinator is None:
            raise ValueError("unknown Coordinator")
        normalized_reason = reason.strip()
        if coordinator.control_status in {
            CoordinatorControlStatus.CANCEL_REQUESTED,
            CoordinatorControlStatus.CANCELLED,
        }:
            if coordinator.control_reason == normalized_reason:
                return state
            raise ValueError("Coordinator already has a different cancel intent")
        return self.apply(
            run_id,
            CoordinatorCancelRequestedOutcome(
                run_id=run_id,
                coordinator_id=coordinator_id,
                basis_revision=state.revision,
                reason=normalized_reason,
            ),
            event_type="coordinator_cancel_requested",
        )

    def request_run_finish(
        self,
        run_id: str,
        *,
        requested_plan_limit: int,
        reason: str,
    ) -> RunState:
        state = self.repository.load(run_id)
        normalized_reason = reason.strip()
        if state.finish_request is not None:
            if (
                state.finish_request.requested_plan_limit
                == requested_plan_limit
                and state.finish_request.reason == normalized_reason
            ):
                return state
            raise ValueError("Run already has a different finish intent")
        return self.apply(
            run_id,
            RunFinishRequestedOutcome(
                run_id=run_id,
                basis_revision=state.revision,
                requested_plan_limit=requested_plan_limit,
                reason=normalized_reason,
            ),
            event_type="run_finish_requested",
        )

    @staticmethod
    def _subject_ref(outcome: Any, fallback: str) -> str:
        explicit = getattr(outcome, "subject_ref", None)
        if explicit is not None:
            if explicit != fallback and not explicit.startswith(f"{fallback}/"):
                raise ValueError("outcome SubjectRef belongs to another Run")
            return explicit
        if isinstance(outcome, SummaryAcceptedOutcome):
            return subject_ref(
                fallback,
                outcome.trial_coordinator_id,
                outcome.trial_plan_id,
                outcome.trial_id,
            )
        coordinator_id = getattr(outcome, "coordinator_id", None) or None
        plan_id = getattr(outcome, "plan_id", None) or None
        trial_id = getattr(outcome, "trial_id", None) or None
        plan = getattr(outcome, "plan", None)
        if plan is not None:
            coordinator_id = getattr(plan, "coordinator_id", coordinator_id)
            plan_id = getattr(plan, "plan_id", plan_id)
        record = getattr(outcome, "record", None)
        if record is not None:
            coordinator_id = getattr(record, "coordinator_id", coordinator_id)
            plan_id = getattr(record, "plan_id", plan_id)
            trial_id = getattr(record, "trial_id", trial_id)
        return subject_ref(fallback, coordinator_id, plan_id, trial_id)

    @staticmethod
    def _logical_work_ref(
        state: RunState,
        outcome: Any,
        subject: str,
    ) -> str:
        command_id = getattr(outcome, "command_id", None)
        if command_id:
            active = next(
                (
                    item
                    for item in state.active_engine_commands
                    if item.command_id == command_id
                ),
                None,
            )
            if active is not None:
                return active.logical_command_id
            active_review = next(
                (
                    item
                    for item in state.active_review_commands
                    if item.command_id == command_id
                ),
                None,
            )
            if active_review is not None:
                return active_review.logical_command_id
        command_ref = getattr(outcome, "command_ref", None)
        if command_ref is not None and getattr(
            command_ref, "logical_command_id", None
        ):
            return str(command_ref.logical_command_id)
        for name in ("logical_command_id", "command_id", "call_id", "action_id"):
            value = getattr(outcome, name, None)
            if value:
                if name == "logical_command_id":
                    return str(value)
                return f"{name.removesuffix('_id')}:{value}"
        call = getattr(outcome, "call_ref", None)
        if call is not None and getattr(call, "call_id", None):
            return f"call:{call.call_id}"
        return subject

    @classmethod
    def _transition_refs(
        cls,
        outcome: Any,
        fact_class: FactClass,
        *,
        event_type: str,
        subject: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        accepted: list[str] = []
        origins: list[str] = []
        for name, value in vars(outcome).items():
            values = value if isinstance(value, tuple) else (value,)
            for item in values:
                if isinstance(item, ArtifactRef):
                    if fact_class is FactClass.SCIENTIFIC:
                        accepted.append(item.artifact_id)
                elif isinstance(item, SnapshotRef):
                    if fact_class is FactClass.SCIENTIFIC:
                        accepted.append(f"snapshot:{item.snapshot_id}")
                elif isinstance(item, str) and item:
                    if name == "receipt_id":
                        origins.append(f"receipt:{item}")
                    elif name in {"command_id", "previous_command_id"}:
                        origins.append(f"command:{item}")
                    elif name in {"call_id", "previous_attempt_id"}:
                        origins.append(f"delivery:{item}")
                    elif fact_class is FactClass.SCIENTIFIC and name in {
                        "result_ref",
                        "result_refs",
                        "artifact_path",
                    }:
                        accepted.append(item)
        if fact_class is FactClass.SCIENTIFIC and not accepted:
            accepted.append(f"state:{subject}/{event_type}")
        return tuple(dict.fromkeys(accepted)), tuple(dict.fromkeys(origins))

    def _publish_control_owned_files(
        self,
        state: RunState,
        updated: RunState,
        event_type: str,
        subject_ref: str | None,
        outcome: Any | None,
    ) -> None:
        # Generic/manual transitions have no accepted external payload to publish.
        if outcome is None:
            return
        records = getattr(self.repository, "trial_records", None)
        memories = getattr(self.repository, "memory_versions", None)
        transition_id = updated.last_transition.transition_id

        if isinstance(outcome, PlanningDecisionOutcome) and memories is not None:
            plan = next(
                item
                for item in updated.plans
                if item.coordinator_id == outcome.plan.coordinator_id
                and item.plan_id == outcome.plan.plan_id
            )
            memory_id = plan.plan_memory_head.rsplit("/", 1)[-1]
            memories.create_plan_version(
                run_id=state.run_id,
                coordinator_id=plan.coordinator_id,
                plan_id=plan.plan_id,
                memory_id=memory_id,
                parent_memory_id=None,
                created_by=transition_id,
                created_revision=updated.revision,
                new_sources=(state.memory.run_head,),
                included_sources=(state.memory.run_head,),
                memory_md=f"# Plan Memory {memory_id}\n\nInitial Plan memory.\n",
                plan_md=self.repository.read_artifact(
                    state.run_id, outcome.supporting_refs[0]
                ).decode("utf-8", errors="replace"),
            )

        if isinstance(
            outcome,
            (
                TrialProposedOutcome,
                BaselineTrialRegisteredOutcome,
                BootstrapP000RegisteredOutcome,
            ),
        ) and records is not None:
            trial = self._trial(updated, outcome)
            records.create(
                run_id=state.run_id,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                trial_id=trial.trial_id,
                created_revision=updated.revision,
                plan_memory_basis=trial.plan_memory_basis,
                run_memory_basis=trial.run_memory_basis,
            )
            if isinstance(outcome, TrialProposedOutcome):
                plan = next(
                    item
                    for item in updated.plans
                    if item.coordinator_id == trial.coordinator_id
                    and item.plan_id == trial.plan_id
                )
                comparator_content = (
                    json.dumps(
                        {
                            "schema_version": "1",
                            "trial": (
                                f"{state.run_id}/{trial.coordinator_id}/"
                                f"{trial.plan_id}/{trial.trial_id}"
                            ),
                            "hypothesis_comparator": plan.hypothesis_comparator,
                            "portfolio_comparator": plan.portfolio_comparator,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode()
                records.admit(
                    run_id=state.run_id,
                    coordinator_id=trial.coordinator_id,
                    plan_id=trial.plan_id,
                    trial_id=trial.trial_id,
                    transition_id=transition_id,
                    source_ref=self._subject_ref(outcome, state.run_id),
                    files={
                        "comparisons/planning-comparators.json": (
                            comparator_content
                        )
                    },
                )
            if isinstance(
                outcome,
                (BaselineTrialRegisteredOutcome, BootstrapP000RegisteredOutcome),
            ):
                self._admit_refs(
                    updated,
                    outcome,
                    {"artifact/baseline.bin": outcome.artifact_ref},
                )

        def analysis_record_path(base: str) -> str:
            trial = self._trial(updated, outcome)
            attempt_index = trial.analysis_review_attempt_index
            if attempt_index > 1:
                return f"analysis/attempt-{attempt_index:03d}/{base}"
            return f"analysis/{base}"

        objective_record_path = "comparisons/objective-comparison.json"
        if isinstance(
            outcome,
            (
                BuilderFailedOutcome,
                EngineFailedOutcome,
                AnalysisAcceptedOutcome,
                AnalysisFailedOutcome,
            ),
        ):
            trial = self._trial(updated, outcome)
            objective_record_path = objective_comparison_record_path(
                trial.engine_attempt_index
            )

        admissions: dict[type, dict[str, str]] = {
            ArtifactAcceptedOutcome: {"artifact/accepted.bin": "artifact_ref"},
            BuilderFailedOutcome: {
                "failure/builder.md": "failure_ref",
                objective_record_path: "objective_comparison_ref",
            },
            EngineCompletedOutcome: {"engine/outcome-package.bin": "package_ref"},
            EngineFailedOutcome: {"failure/engine.md": "failure_ref"},
            AnalysisFailedOutcome: {"failure/analysis.md": "failure_ref"},
            CoordinatorTrialCancelledOutcome: {
                "failure/operator-cancellation.json": "cancellation_ref"
            },
        }
        if isinstance(outcome, AnalysisAcceptedOutcome):
            admissions[type(outcome)] = {
                analysis_record_path("analysis.md"): "analysis_ref",
                analysis_record_path("findings.md"): "findings_ref",
                analysis_record_path("evidence.bin"): "evidence_ref",
                analysis_record_path("review-coverage.bin"): "review_coverage_ref",
                objective_record_path: "objective_comparison_ref",
            }
        elif isinstance(outcome, AnalysisReviewSubmittedOutcome):
            admissions[type(outcome)] = {
                analysis_record_path("review-plan.json"): "design_ref",
            }
        elif isinstance(outcome, AnalysisReviewCompletedOutcome):
            admissions[type(outcome)] = {
                analysis_record_path("review-packet.json"): "packet_ref",
                analysis_record_path("review-coverage.json"): "coverage_ref",
            }
        for outcome_type, paths in admissions.items():
            if isinstance(outcome, outcome_type):
                self._admit_refs(
                    updated,
                    outcome,
                    {path: getattr(outcome, field) for path, field in paths.items()},
                )

        if isinstance(outcome, AnalysisFailedOutcome):
            self._admit_refs(
                updated,
                outcome,
                {
                    objective_record_path: outcome.objective_comparison_ref
                },
            )

        if (
            isinstance(outcome, EngineFailedOutcome)
            and outcome.objective_comparison_ref is not None
        ):
            self._admit_refs(
                updated,
                outcome,
                {
                    objective_record_path: outcome.objective_comparison_ref
                },
            )

        if isinstance(outcome, BuilderFailedOutcome):
            trial = self._trial(updated, outcome)
            records.admit(
                run_id=state.run_id,
                coordinator_id=trial.coordinator_id,
                plan_id=trial.plan_id,
                trial_id=trial.trial_id,
                transition_id=transition_id,
                source_ref=self._subject_ref(outcome, state.run_id),
                files={
                    "realization/final-realization.json": (
                        json.dumps(
                            {
                                "schema_version": "1",
                                "realization_status": "unverified",
                                "reason": "builder_failed",
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode()
                },
            )

        if isinstance(outcome, ArtifactAcceptedOutcome):
            realization_refs = tuple(
                ref
                for ref in outcome.supporting_refs
                if ref.kind
                in {
                    "data_selection_realization",
                    "reward_compliance_report",
                    "curriculum_realization",
                }
            )
            if len(realization_refs) != 1:
                raise ValueError(
                    "accepted Search artifact requires one final realization report"
                )
            self._admit_refs(
                updated,
                outcome,
                {"realization/final-realization.json": realization_refs[0]},
            )

        if isinstance(outcome, SummaryAcceptedOutcome) and memories is not None:
            summary = self.repository.read_artifact(
                state.run_id, outcome.summary_ref
            ).decode("utf-8", errors="replace")
            trial = self._trial(updated, outcome)
            if outcome.summary_kind == "plan_summary":
                memory_id = trial.plan_memory_result.rsplit("/", 1)[-1]
                parent = self._trial(state, outcome).plan_memory_basis
                parent_files = memories.read_plan_version(
                    run_id=state.run_id,
                    coordinator_id=trial.coordinator_id,
                    plan_id=trial.plan_id,
                    memory_id=parent.rsplit("/", 1)[-1],
                )
                parent_manifest = json.loads(parent_files["manifest.json"])
                record_files = self.repository.trial_records.read_files(
                    run_id=state.run_id,
                    coordinator_id=trial.coordinator_id,
                    plan_id=trial.plan_id,
                    trial_id=trial.trial_id,
                )
                cumulative_files = {
                    path: content
                    for path, content in parent_files.items()
                    if path.startswith(("outcomes/", "findings/"))
                }
                outcome_payload = {
                    "schema_version": "1",
                    "trial": trial.trial_record_ref,
                    "outcome": trial.outcome.value,
                    "metric_id": updated.ranking.metric_id,
                    "offline_score": trial.offline_score,
                    "offline_secondary_score": trial.offline_secondary_score,
                    "analysis_status": trial.analysis_status,
                    "failure_kind": trial.failure_kind,
                }
                outcome_root = f"outcomes/{trial.trial_id}"
                cumulative_files[f"{outcome_root}/outcome.json"] = (
                    json.dumps(outcome_payload, indent=2, sort_keys=True) + "\n"
                ).encode()
                if trial.failure_kind or trial.analysis_failure_ref_id:
                    cumulative_files[f"{outcome_root}/failure.json"] = (
                        json.dumps(
                            {
                                "schema_version": "1",
                                "failure_kind": trial.failure_kind,
                                "analysis_failure_ref_id": trial.analysis_failure_ref_id,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode()
                semantic_names = {
                    "findings.md": "findings.md",
                    "evidence.bin": "evidence.json",
                    "review-coverage.bin": "review-coverage.json",
                }
                for source_name, target_name in semantic_names.items():
                    candidates = sorted(
                        path
                        for path in record_files
                        if path.startswith("analysis/")
                        and path.rsplit("/", 1)[-1] == source_name
                    )
                    if candidates:
                        cumulative_files[
                            f"findings/{trial.trial_id}/{target_name}"
                        ] = record_files[candidates[-1]]
                included_sources = tuple(
                    dict.fromkeys(
                        (*parent_manifest["included_sources"], trial.trial_record_ref)
                    )
                )
                memories.create_plan_version(
                    run_id=state.run_id,
                    coordinator_id=trial.coordinator_id,
                    plan_id=trial.plan_id,
                    memory_id=memory_id,
                    parent_memory_id=parent,
                    created_by=transition_id,
                    created_revision=updated.revision,
                    new_sources=(trial.trial_record_ref,),
                    included_sources=included_sources,
                    memory_md=summary,
                    plan_md=parent_files["plan.md"].decode("utf-8"),
                    files=cumulative_files,
                )
            elif outcome.summary_kind == "run_summary":
                parent_id = state.memory.run_head
                parent_files = memories.read_run_version(
                    run_id=state.run_id,
                    memory_id=parent_id.rsplit("/", 1)[-1],
                )
                parent_manifest = json.loads(parent_files["manifest.json"])
                plan_memory_id = trial.plan_memory_result
                plan_files = memories.read_plan_version(
                    run_id=state.run_id,
                    coordinator_id=trial.coordinator_id,
                    plan_id=trial.plan_id,
                    memory_id=plan_memory_id.rsplit("/", 1)[-1],
                )
                queue_entry = (
                    f"{state.run_id}/{trial.coordinator_id}/"
                    f"{trial.plan_id}/{trial.trial_id}"
                )
                update = (
                    "# Plan Memory Update\n\n"
                    f"- Queue entry: `{queue_entry}`\n"
                    f"- Plan: `{state.run_id}/{trial.coordinator_id}/{trial.plan_id}`\n"
                    f"- Trial: `{trial.trial_record_ref}`\n"
                    f"- Plan Memory: `{plan_memory_id}`\n"
                    f"- Run Memory parent: `{parent_id}`\n"
                    f"- Basis revision: `{state.revision}`\n"
                ).encode()
                root = (
                    f"{trial.coordinator_id}/{trial.plan_id}/{trial.trial_id}"
                )
                cumulative_files = {
                    path: content
                    for path, content in parent_files.items()
                    if path.startswith(("outcomes/", "findings/"))
                }
                cumulative_files[f"outcomes/{root}/PLAN_UPDATE.md"] = update
                cumulative_files[f"outcomes/{root}/run-facts.json"] = (
                    json.dumps(
                        {
                            "schema_version": "1",
                            "basis_revision": state.revision,
                            "plan_catalog_revision": state.plan_catalog.revision,
                            "ranking_revision": state.ranking.revision,
                            "metric_id": state.ranking.metric_id,
                            "queue_entry": queue_entry,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode()
                cumulative_files[f"findings/{root}/PLAN_MEMORY.md"] = plan_files[
                    "MEMORY.md"
                ]
                included_sources = tuple(
                    dict.fromkeys(
                        (*parent_manifest["included_sources"], plan_memory_id)
                    )
                )
                memories.create_run_version(
                    run_id=state.run_id,
                    memory_id=trial.run_memory_result.rsplit("/", 1)[-1],
                    parent_memory_id=parent_id,
                    created_by=transition_id,
                    created_revision=updated.revision,
                    new_sources=(plan_memory_id,),
                    included_sources=included_sources,
                    memory_md=summary,
                    files=cumulative_files,
                )

    def _admit_refs(
        self,
        state: RunState,
        outcome: Any,
        refs: dict[str, Any],
    ) -> None:
        records = getattr(self.repository, "trial_records", None)
        if records is None:
            return
        trial = self._trial(state, outcome)
        records.admit(
            run_id=state.run_id,
            coordinator_id=trial.coordinator_id,
            plan_id=trial.plan_id,
            trial_id=trial.trial_id,
            transition_id=state.last_transition.transition_id,
            source_ref=self._subject_ref(outcome, state.run_id),
            files={
                path: self.repository.read_artifact(state.run_id, ref)
                for path, ref in refs.items()
            },
        )

    @staticmethod
    def _trial(state: RunState, outcome: Any):
        coordinator_id = getattr(
            outcome, "trial_coordinator_id", getattr(outcome, "coordinator_id", None)
        )
        plan_id = getattr(
            outcome, "trial_plan_id", getattr(outcome, "plan_id", None)
        )
        trial_id = getattr(outcome, "trial_id", None)
        matches = tuple(
            trial
            for trial in state.trials
            if trial.coordinator_id == coordinator_id
            and trial.plan_id == plan_id
            and trial.trial_id == trial_id
        )
        if len(matches) != 1:
            raise ValueError("accepted outcome Trial scope is unavailable")
        return matches[0]
