"""Controller adapter for one typed Coordinator Agent Call."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math

from ade.agent_runtime.runtime import AcceptedCall, RejectedCall
from ade.agent_runtime.service import AgentCallService
from ade.controller.ports import RunRepository
from ade.core.actions import Action, ActionKind
from ade.core.agent import AgentRole
from ade.core.coordinator import CoordinatorKind
from ade.core.failures import FailureState
from ade.core.outcomes import (
    AgentRetryHeldOutcome,
    AgentRetrySubmittedOutcome,
    CoordinatorCallSubmittedOutcome,
    CoordinatorFailedOutcome,
    PlanningDecisionOutcome,
    RunSuspendedOutcome,
)
from ade.core.run import ActiveAgentCallRef
from ade.core.plan import PlanRelation, PlanRelationKind, PlanStatus
from ade.core.scope import PlanKey
from ade.core.snapshot import SnapshotFile, SnapshotKind
from ade.core.validation import DeliveryViolation, ValidationReport
from ade.tasks.contracts import PlanningDecision
from ade.tasks.registry import TaskRegistry


class CoordinatorAgentPort:
    def __init__(
        self,
        *,
        calls: AgentCallService,
        repository: RunRepository,
        tasks: TaskRegistry,
    ) -> None:
        self.calls = calls
        self.repository = repository
        self.tasks = tasks

    def prepare(self, action: Action) -> CoordinatorCallSubmittedOutcome:
        if action.kind is not ActionKind.CALL_AGENT:
            raise ValueError("Coordinator AgentPort requires CALL_AGENT action")
        state = self.repository.load(action.run_id)
        if action.basis_revision != state.revision:
            raise ValueError("Coordinator submission basis is stale")
        target_plan_id = self._next_plan_id(state, action.subject_id)
        active = self.calls.prepare(
            run_id=action.run_id,
            role=AgentRole.COORDINATOR,
            subject_id=action.subject_id,
            basis_revision=action.basis_revision,
            action_id=action.action_id,
            target_subject_ref=PlanKey(
                action.run_id, action.subject_id, target_plan_id
            ).subject_ref,
            action_fields={
                "target_plan_id": target_plan_id,
                "planning_basis": {
                    "run_memory_id": state.memory.run_head,
                    "ranking_revision": state.ranking.revision,
                    "plan_catalog_revision": state.plan_catalog.revision,
                    "run_revision": state.revision,
                },
            },
        )
        return CoordinatorCallSubmittedOutcome(
            run_id=action.run_id,
            coordinator_id=action.subject_id,
            plan_id=target_plan_id,
            basis_revision=state.revision,
            call_ref=active,
        )

    def collect(
        self,
        active: ActiveAgentCallRef,
    ) -> (
        PlanningDecisionOutcome
        | AgentRetryHeldOutcome
        | AgentRetrySubmittedOutcome
        | CoordinatorFailedOutcome
        | RunSuspendedOutcome
        | None
    ):
        if active.role != AgentRole.COORDINATOR.value:
            raise ValueError("active Call is not a Coordinator Call")
        action = Action(
            action_id=active.action_id,
            run_id=active.run_id,
            kind=ActionKind.CALL_AGENT,
            subject_id=active.subject_id,
            basis_revision=active.basis_revision,
        )
        state = self.repository.load(action.run_id)
        if not any(
            coordinator.coordinator_id == action.subject_id
            and coordinator.kind is CoordinatorKind.SEARCH
            for coordinator in state.coordinators
        ):
            raise ValueError("Coordinator action subject must be a search Coordinator")
        target_plan_id = self._next_plan_id(state, action.subject_id)
        result = (
            self.calls.expire_active(active)
            if self.calls.is_expired(active)
            else self.calls.execute_active(active)
        )
        if result is None:
            return None
        if isinstance(result, RejectedCall):
            details = "; ".join(item.message for item in result.validation.violations)
            return self._retry_or_fail(
                active,
                result,
                state,
                target_plan_id,
                result.validation,
                details or "Coordinator retry budget exhausted",
            )
        if not isinstance(result, AcceptedCall) or not isinstance(
            result.output,
            PlanningDecision,
        ):
            raise TypeError("Coordinator call did not return PlanningDecision")
        try:
            if result.output.plan_id != target_plan_id:
                raise ValueError(
                    "Coordinator PlanningDecision subject does not match Runtime target"
                )
            relation = self._resolve_relation(state, result.output)
        except ValueError as error:
            report = ValidationReport(
                (
                    DeliveryViolation(
                        code="invalid_plan_relation",
                        message=str(error),
                        path="decision.json#/related_plan_keys",
                        repairable=True,
                    ),
                )
            )
            return self._retry_or_fail(
                active,
                result,
                state,
                target_plan_id,
                report,
                str(error),
            )
        try:
            hypothesis_comparator, portfolio_comparator = (
                self._resolve_comparator_bindings(state, active, result, relation)
            )
        except ValueError as error:
            return RunSuspendedOutcome(
                run_id=state.run_id,
                subject_ref=active.target_subject_ref,
                basis_revision=state.revision,
                failure=FailureState(
                    "comparator_binding_integrity_failure",
                    str(error),
                    retryable=True,
                ),
                agent_session=result.session,
                call_id=result.call.call_id,
            )
        plugin = self.tasks.get(state.task.task_id)
        decision_ref = self.repository.put_artifact(
            action.run_id,
            "planning_decision",
            result.output.content,
        )
        supporting_refs = tuple(
            self.repository.put_artifact(
                action.run_id,
                artifact.kind,
                artifact.content,
            )
            for artifact in result.output.supporting_artifacts
        )
        outcome = plugin.planning_outcome(
            action,
            result.output,
            decision_ref,
            result.call.call_id,
        )
        outcome = replace(
            outcome,
            basis_revision=state.revision,
            plan=replace(
                outcome.plan,
                relation=relation,
                basis_revision=active.basis_revision,
                planning_basis=dict(active.action_fields["planning_basis"]),
                hypothesis_comparator=hypothesis_comparator,
                portfolio_comparator=portfolio_comparator,
            ),
        )
        if supporting_refs:
            outcome = PlanningDecisionOutcome(
                action_id=outcome.action_id,
                run_id=outcome.run_id,
                subject_id=outcome.subject_id,
                basis_revision=outcome.basis_revision,
                call_id=outcome.call_id,
                plan=replace(
                    outcome.plan,
                    decision_report_ref_id=supporting_refs[0].artifact_id,
                ),
                decision_ref=outcome.decision_ref,
                supporting_refs=supporting_refs,
            )
        snapshot_files = [
            SnapshotFile.generated(
                "decision.json",
                result.output.content,
                source_id=decision_ref.artifact_id,
            ),
            SnapshotFile.generated(
                "state.json",
                (
                    json.dumps(
                        {
                            "schema_version": "1",
                            "run_id": state.run_id,
                            "coordinator_id": outcome.plan.coordinator_id,
                            "plan_id": outcome.plan.plan_id,
                            "status": PlanStatus.PROPOSED.value,
                            "basis_revision": state.revision,
                            "relation": (
                                asdict(outcome.plan.relation)
                                if outcome.plan.relation is not None
                                else None
                            ),
                            "hypothesis_comparator": outcome.plan.hypothesis_comparator,
                            "portfolio_comparator": outcome.plan.portfolio_comparator,
                            "best_trial_id": None,
                            "best_artifact_ref_id": None,
                            "best_score": None,
                            "no_improvement_count": 0,
                            "accepted_trials": [],
                            "remaining_plan_trial_budget": (
                                state.portfolio.max_trials_per_plan
                            ),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode(),
                source_id=f"run-state-r{state.revision:03d}",
            ),
        ]
        snapshot_files.extend(
            SnapshotFile.generated(
                artifact.path,
                artifact.content,
                source_id=stored.artifact_id,
            )
            for artifact, stored in zip(
                result.output.supporting_artifacts,
                supporting_refs,
                strict=True,
            )
        )
        snapshot = self.repository.snapshots.materialize(
            kind=SnapshotKind.PLAN,
            run_id=state.run_id,
            coordinator_id=outcome.plan.coordinator_id,
            plan_id=outcome.plan.plan_id,
            revision=state.revision + 1,
            files=tuple(snapshot_files),
        )
        return replace(
            outcome,
            snapshot_ref=snapshot,
            agent_session=result.session,
        )

    def _retry_or_fail(
        self,
        active: ActiveAgentCallRef,
        result: AcceptedCall | RejectedCall,
        state,
        target_plan_id: str,
        report: ValidationReport,
        reason: str,
    ) -> (
        AgentRetryHeldOutcome
        | AgentRetrySubmittedOutcome
        | CoordinatorFailedOutcome
        | RunSuspendedOutcome
    ):
        if (
            active.retry_index < active.max_retries
            and report.violations
            and all(item.repairable for item in report.violations)
        ):
            if state.pause_requested:
                return AgentRetryHeldOutcome(
                    run_id=state.run_id,
                    subject_id=active.subject_id,
                    basis_revision=state.revision,
                    call_id=active.call_id,
                    previous_attempt_id=active.attempt_id,
                    retry_reason=report.violations[0].code,
                )
            retried = self.calls.prepare_retry(active, result, report)
            return AgentRetrySubmittedOutcome(
                run_id=state.run_id,
                subject_id=active.subject_id,
                basis_revision=state.revision,
                call_ref=retried,
                previous_attempt_id=active.attempt_id,
                retry_reason=report.violations[0].code,
            )
        if any(
            item.code in {"agent_backend_failed", "heartbeat_timeout"}
            for item in report.violations
        ):
            return RunSuspendedOutcome(
                run_id=state.run_id,
                subject_ref=active.target_subject_ref,
                basis_revision=state.revision,
                failure=FailureState(
                    "coordinator_exhausted",
                    reason,
                    retryable=True,
                ),
                agent_session=result.session,
                call_id=result.call.call_id,
            )
        failure_ref = self.repository.put_artifact(
            state.run_id,
            "coordinator_failure",
            (
                json.dumps(
                    {
                        "schema_version": "1",
                        "coordinator_id": active.subject_id,
                        "plan_id": target_plan_id,
                        "call_id": result.call.call_id,
                        "attempt_id": result.attempt.attempt_id,
                        "reason": reason,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
        )
        return CoordinatorFailedOutcome(
            run_id=state.run_id,
            coordinator_id=active.subject_id,
            plan_id=target_plan_id,
            basis_revision=state.revision,
            call_id=result.call.call_id,
            failure_ref=failure_ref,
            agent_session=result.session,
        )

    @staticmethod
    def _next_plan_id(state, coordinator_id: str) -> str:
        numbers = []
        for plan in state.plans:
            if plan.coordinator_id != coordinator_id:
                continue
            if not plan.plan_id.startswith("p") or not plan.plan_id[1:].isdigit():
                raise ValueError("Runtime-owned Plan IDs must use pNNN")
            numbers.append(int(plan.plan_id[1:]))
        return f"p{max(numbers, default=0) + 1:03d}"

    @staticmethod
    def _resolve_relation(state, output: PlanningDecision) -> PlanRelation:
        kind = PlanRelationKind(output.relation_kind)
        if kind is PlanRelationKind.NEW_DIRECTION:
            related_keys: tuple[PlanKey, ...] = ()
            source_keys = (PlanKey(state.run_id, "c000", "p000"),)
        else:
            related_keys = tuple(
                PlanKey(*value.split("/"))
                for value in output.related_plan_keys
            )
            if any(key.run_id != state.run_id for key in related_keys):
                raise ValueError("Plan relation source belongs to another Run")
            source_keys = related_keys
        source_artifact_ids = []
        for key in source_keys:
            matches = [
                plan
                for plan in state.plans
                if plan.coordinator_id == key.coordinator_id
                and plan.plan_id == key.plan_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    "unknown Plan relation source: "
                    f"{key.subject_ref}"
                )
            source = matches[0]
            eligible = tuple(
                item
                for item in state.plan_catalog.source_eligible_results
                if item.coordinator_id == key.coordinator_id
                and item.plan_id == key.plan_id
            )
            if len(eligible) != 1:
                if (
                    key.coordinator_id == "c000"
                    and key.plan_id == "p000"
                    and source.status is PlanStatus.COMPLETED
                    and source.best_artifact_ref_id is not None
                ):
                    source_artifact_ids.append(source.best_artifact_ref_id)
                    continue
                raise ValueError(
                    f"Plan relation source is not Source-Eligible: "
                    f"{key.subject_ref}"
                )
            source_artifact_ids.append(eligible[0].artifact_ref_id)
        return PlanRelation(
            kind=kind,
            related_plan_keys=related_keys,
            seed_artifact_ref_ids=tuple(source_artifact_ids),
        )

    @staticmethod
    def _resolve_comparator_bindings(
        state,
        active: ActiveAgentCallRef,
        result: AcceptedCall,
        relation: PlanRelation,
    ) -> tuple[dict[str, object], dict[str, object]]:
        input_root = result.workspace / "input" / "subject"
        try:
            ranking = json.loads(
                (input_root / "ranking.json").read_text(encoding="utf-8")
            )
            target = json.loads(
                (input_root / "portfolio-target.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("frozen comparator input is unavailable") from error
        comparisons = result.output.design.get("comparisons")
        if not isinstance(comparisons, dict):
            raise ValueError("PlanningDecision comparisons are unavailable")
        hypothesis = comparisons.get("hypothesis")
        if not isinstance(hypothesis, dict):
            raise ValueError("PlanningDecision hypothesis expectation is invalid")
        ranking_revision = target.get("ranking_revision")
        planning_basis = active.action_fields.get("planning_basis")
        if (
            not isinstance(ranking, dict)
            or not isinstance(target, dict)
            or not isinstance(planning_basis, dict)
            or ranking.get("revision") != ranking_revision
            or planning_basis.get("ranking_revision") != ranking_revision
        ):
            raise ValueError("frozen comparator ranking revision is inconsistent")
        entries = ranking.get("entries")
        if not isinstance(entries, list):
            raise ValueError("frozen Ranking entries are unavailable")

        primary_source = (
            relation.related_plan_keys[0]
            if relation.related_plan_keys
            else PlanKey(state.run_id, "c000", "p000")
        )
        source_results = [
            item
            for item in state.plan_catalog.source_eligible_results
            if item.coordinator_id == primary_source.coordinator_id
            and item.plan_id == primary_source.plan_id
        ]
        if len(source_results) == 1:
            representative_trial_id = source_results[0].representative_trial_id
        else:
            source_plans = [
                item
                for item in state.plans
                if item.coordinator_id == primary_source.coordinator_id
                and item.plan_id == primary_source.plan_id
                and item.status is PlanStatus.COMPLETED
                and item.best_artifact_ref_id is not None
                and item.best_trial_id is not None
            ]
            if len(source_plans) != 1:
                raise ValueError(
                    "Harness cannot resolve one completed primary relation source"
                )
            representative_trial_id = source_plans[0].best_trial_id
        hypothesis_subject = (
            f"{primary_source.subject_ref}/{representative_trial_id}"
        )
        matches = [
            item
            for item in entries
            if isinstance(item, dict)
            and item.get("subject_id") == hypothesis_subject
            and item.get("evaluation_profile") == target.get("evaluation_profile")
            and item.get("level") == "trial_level"
            and item.get("source_eligible") is True
            and isinstance(item.get("score"), (int, float))
            and not isinstance(item.get("score"), bool)
            and math.isfinite(item["score"])
            and (
                item.get("secondary_score") is None
                or (
                    isinstance(item.get("secondary_score"), (int, float))
                    and not isinstance(item.get("secondary_score"), bool)
                    and math.isfinite(item["secondary_score"])
                )
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                "Harness cannot bind the primary relation source to one profile-matched source-eligible Trial"
            )
        hypothesis_entry = matches[0]
        global_best = target.get("global_best")
        expected_portfolio_subject = (
            global_best.get("subject_id")
            if isinstance(global_best, dict)
            else None
        )
        hypothesis_binding = {
            "subject_id": hypothesis_entry["subject_id"],
            "level": hypothesis_entry["level"],
            "artifact_ref_id": hypothesis_entry.get("artifact_ref_id"),
            "artifact_digest": hypothesis_entry.get("artifact_digest"),
            "evaluation_profile": hypothesis_entry["evaluation_profile"],
            "ranking_revision": ranking_revision,
            "primary": {
                "metric_id": ranking.get("metric_id"),
                "value": hypothesis_entry.get("score"),
            },
            "secondary": {
                "metric_id": "offline.secondary_score",
                "value": hypothesis_entry.get("secondary_score"),
            },
            "reason": hypothesis["reason"],
            "expected_observation": dict(hypothesis["expected_observation"]),
        }
        portfolio_binding = {
            "subject_id": expected_portfolio_subject,
            "ranking_revision": ranking_revision,
            "evaluation_profile": target.get("evaluation_profile"),
            "direction": target.get("direction"),
            "target": global_best,
        }
        return hypothesis_binding, portfolio_binding
