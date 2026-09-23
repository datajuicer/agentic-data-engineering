"""Exact-revision Run fork initialization."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import uuid

from ade.core.artifacts import ArtifactRef
from ade.core.bootstrap import BootstrapStatus
from ade.core.coordinator import CoordinatorControlStatus
from ade.core.operator import OperatorEvaluationStatus
from ade.core.plan import PlanKind, PlanRelation, PlanStatus
from ade.core.run import (
    ForkLineage,
    FactClass,
    ForkCause,
    ResearchOutcome,
    RunState,
    RunStatus,
    TransitionRecord,
)
from ade.core.scope import PlanKey
from ade.core.snapshot import SnapshotRef
from ade.core.trial import TrialArchiveStatus, TrialPhase
from ade.controller.ports import RunRepository
from ade.controller.run_reference_materialization import (
    RunReferenceMaterializer,
    read_tree,
    rewrite_run_ref,
    rewrite_run_refs,
)


_SELECTOR = re.compile(r"^(?P<run>[A-Za-z0-9][A-Za-z0-9._-]*)@rev-(?P<rev>\d{6,})$")
_ATTEMPT_EVIDENCE = re.compile(
    r"^attempt:(?P<logical>[A-Za-z0-9][A-Za-z0-9._:-]*)/"
    r"(?P<attempt>attempt-\d{3,})@writer:"
    r"(?P<writer>[A-Za-z0-9][A-Za-z0-9._:-]*)$"
)
_WAITING_PHASES = {
    TrialPhase.BUILDING_ARTIFACT: TrialPhase.CREATED,
    TrialPhase.ENGINE_RUNNING: TrialPhase.ARTIFACT_READY,
    TrialPhase.ANALYSIS_DESIGNING: TrialPhase.EVIDENCE_READY,
    TrialPhase.REVIEW_RUNNING: TrialPhase.EVIDENCE_READY,
    TrialPhase.ANALYZING: TrialPhase.REVIEW_READY,
    TrialPhase.PLAN_SUMMARIZING: TrialPhase.ANALYSIS_READY,
    TrialPhase.RUN_SUMMARIZING: TrialPhase.PLAN_SUMMARY_READY,
}
class RunForkService:
    def __init__(
        self,
        repository: RunRepository,
        *,
        source_repository: RunRepository | None = None,
    ) -> None:
        self.repository = repository
        self.source_repository = source_repository or repository
        self.materializer = RunReferenceMaterializer(
            repository,
            source_repository=self.source_repository,
        )

    @staticmethod
    def parse_selector(selector: str) -> tuple[str, int]:
        match = _SELECTOR.fullmatch(selector)
        if match is None:
            raise ValueError("fork source must use exact run_id@rev-000000 selector")
        return match.group("run"), int(match.group("rev"))

    def fork(self, *, cause: ForkCause | str, evidence_ref: str) -> RunState:
        resolved_cause = ForkCause(cause)
        source_run_id, source_revision = self.parse_selector(evidence_ref)
        evidence = self.repository.load_revision(source_run_id, source_revision)
        replay_revision, cause_ref = self._resolve_replay_boundary(
            evidence,
            cause=resolved_cause,
        )
        source = self.repository.load_revision(source_run_id, replay_revision)
        self._validate_forkable(source)
        lineage_root = (
            evidence.forked_from.lineage_root_run_id
            if evidence.forked_from is not None
            else source_run_id
        )
        generation = (
            evidence.forked_from.generation + 1
            if evidence.forked_from is not None
            else 1
        )
        new_run_id = (
            f"{lineage_root}-f{generation}-r{replay_revision:06d}-"
            f"{uuid.uuid4().hex[:12]}"
        )

        artifact_pairs, rewritten = self.materializer.copy_artifacts(
            source,
            new_run_id=new_run_id,
        )
        accepted_transition_ids = {
            self.repository.load_revision(
                source_run_id, revision
            ).last_transition.transition_id
            for revision in range(replay_revision + 1)
        }
        inherited_files, snapshots = self.materializer.inherited_files(
            source,
            new_run_id=new_run_id,
            accepted_transition_ids=accepted_transition_ids,
        )
        lineage = ForkLineage(
            lineage_root_run_id=lineage_root,
            generation=generation,
            source_run_id=source_run_id,
            source_revision=source_revision,
            source_revision_ref=(
                f"{source_run_id}@rev-{source_revision:06d}"
            ),
            cause=resolved_cause,
            cause_ref=cause_ref,
            replay_boundary_revision=replay_revision,
            replay_boundary_ref=(
                f"{source_run_id}@rev-{replay_revision:06d}"
            ),
        )
        forked = self._normalize_state(
            source,
            new_run_id=new_run_id,
            lineage=lineage,
            rewritten=rewritten,
            snapshots=snapshots,
        )
        config_files = read_tree(
            self.repository.layout.run_dir(source_run_id) / "config"
        )
        return self.repository.create(
            forked,
            initial_artifacts=artifact_pairs,
            initial_config_files=config_files,
            inherited_files=inherited_files,
            initialize_memory=False,
        )

    def _resolve_replay_boundary(
        self,
        evidence: RunState,
        *,
        cause: ForkCause,
    ) -> tuple[int, str]:
        transition = evidence.last_transition
        if cause is ForkCause.INVALID_ACCEPTED_FACT:
            if (
                transition.fact_class is not FactClass.SCIENTIFIC
                or not transition.accepted_fact_refs
            ):
                raise ValueError(
                    "--invalidate must reference a scientific AcceptedBoundary"
                )
            logical_work_ref = transition.logical_work_ref
            matching = []
            for revision in range(evidence.revision + 1):
                candidate = self.repository.load_revision(
                    evidence.run_id, revision
                ).last_transition
                if candidate.logical_work_ref == logical_work_ref:
                    matching.append(candidate)
            if not matching:
                raise ValueError("AcceptedBoundary has no logical-work history")
            from_revision = matching[0].from_revision
            replay_revision = 0 if from_revision is None else from_revision
            while replay_revision > 0:
                boundary = self.repository.load_revision(
                    evidence.run_id, replay_revision
                )
                if not boundary.planning_queue and not any(
                    plan.status is PlanStatus.PROPOSED
                    for plan in boundary.plans
                ):
                    break
                replay_revision -= 1
            return replay_revision, (
                f"{evidence.run_id}@rev-{evidence.revision:06d}#"
                f"{transition.transition_id}"
            )
        if cause is ForkCause.UNFENCEABLE_EXECUTION:
            if (
                evidence.status is not RunStatus.SUSPENDED
                or evidence.failure is None
                or evidence.failure.code != "unfenceable_execution"
                or not evidence.failure.evidence_ref
            ):
                raise ValueError(
                    "unfenceable fork requires suspended typed Attempt evidence"
                )
            attempt_ref = evidence.failure.evidence_ref
            match = _ATTEMPT_EVIDENCE.fullmatch(attempt_ref)
            if match is None:
                raise ValueError(
                    "unfenceable Attempt evidence must identify logical work, "
                    "physical Attempt, and backend writer"
                )
            logical_work_ref = match.group("logical")
            attempt_id = match.group("attempt")
            if (
                transition.fact_class is not FactClass.OPERATIONAL
                or transition.logical_work_ref != logical_work_ref
                or attempt_ref not in transition.origin_refs
            ):
                raise ValueError(
                    "unfenceable suspension transition does not cite its typed Attempt"
                )
            active_matches = sum(
                item.logical_command_id == logical_work_ref
                and item.attempt_id == attempt_id
                for item in (
                    *evidence.active_engine_commands,
                    *evidence.active_review_commands,
                )
            ) + sum(
                f"call:{item.call_id}" == logical_work_ref
                and item.attempt_id == attempt_id
                for item in evidence.active_agent_calls
            )
            if active_matches != 1:
                raise ValueError(
                    "unfenceable evidence must match exactly one active physical Attempt"
                )
            replay_revision = self._pre_dispatch_boundary(
                evidence.run_id,
                through_revision=evidence.revision,
                logical_work_ref=logical_work_ref,
            )
            return replay_revision, attempt_ref
        if (
            evidence.status is not RunStatus.CANCELLED
            or transition.kind != "run_cancelled"
        ):
            raise ValueError(
                "continue_cancelled must reference the terminal cancel transition"
            )
        return evidence.revision, (
            f"{evidence.run_id}@rev-{evidence.revision:06d}#"
            f"{transition.transition_id}"
        )

    def _pre_dispatch_boundary(
        self,
        run_id: str,
        *,
        through_revision: int,
        logical_work_ref: str,
    ) -> int:
        for revision in range(through_revision + 1):
            transition = self.repository.load_revision(
                run_id, revision
            ).last_transition
            if transition.logical_work_ref != logical_work_ref:
                continue
            if transition.from_revision is None:
                return 0
            return transition.from_revision
        raise ValueError("unfenceable Attempt has no dispatch history")

    @staticmethod
    def _validate_forkable(source: RunState) -> None:
        if source.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
            raise ValueError(f"source revision is not forkable: {source.status.value}")
        if source.planning_queue or any(
            plan.status is PlanStatus.PROPOSED for plan in source.plans
        ):
            raise ValueError(
                "fork source must precede planning reservation or follow Plan Catalog update"
            )
        search_plans = tuple(p for p in source.plans if p.kind is PlanKind.SEARCH)
        search_trials = tuple(t for t in source.trials if t.kind.value == "search")
        unfinished = any(
            t.archive_status is not TrialArchiveStatus.ARCHIVED
            for t in search_trials
        )
        remaining = (
            len(search_plans) < source.portfolio.max_plans
            and len(search_trials) < source.portfolio.max_trials
        )
        if not unfinished and not remaining:
            raise ValueError("source revision has no remaining workflow or budget")

    @staticmethod
    def _normalize_state(
        source: RunState,
        *,
        new_run_id: str,
        lineage: ForkLineage,
        rewritten: dict[str, ArtifactRef],
        snapshots: tuple[SnapshotRef, ...],
    ) -> RunState:
        def refs(field: str) -> tuple[ArtifactRef, ...]:
            return tuple(rewritten[ref.artifact_id] for ref in getattr(source, field))

        plans = tuple(
            replace(
                plan,
                relation=(
                    replace(
                        plan.relation,
                        related_plan_keys=tuple(
                            PlanKey(new_run_id, key.coordinator_id, key.plan_id)
                            for key in plan.relation.related_plan_keys
                        ),
                    )
                    if isinstance(plan.relation, PlanRelation)
                    else None
                ),
                reservation_id=None,
                reservation_sequence=None,
                planning_basis=(
                    rewrite_run_refs(
                        plan.planning_basis, source.run_id, new_run_id
                    )
                    if plan.planning_basis is not None
                    else None
                ),
                hypothesis_comparator=(
                    rewrite_run_refs(
                        plan.hypothesis_comparator, source.run_id, new_run_id
                    )
                    if plan.hypothesis_comparator is not None
                    else None
                ),
                portfolio_comparator=(
                    rewrite_run_refs(
                        plan.portfolio_comparator, source.run_id, new_run_id
                    )
                    if plan.portfolio_comparator is not None
                    else None
                ),
                plan_memory_head=rewrite_run_ref(
                    plan.plan_memory_head, source.run_id, new_run_id
                ),
            )
            for plan in source.plans
        )
        reset_review_design_refs = {
            trial.analysis_design_ref_id
            for trial in source.trials
            if trial.phase is TrialPhase.REVIEW_RUNNING
            and trial.analysis_design_ref_id is not None
        }
        trials = tuple(
            replace(
                trial,
                phase=_WAITING_PHASES.get(trial.phase, trial.phase),
                command_id=(
                    None
                    if trial.phase is TrialPhase.ENGINE_RUNNING
                    else trial.command_id
                ),
                logical_command_id=(
                    None
                    if trial.phase is TrialPhase.ENGINE_RUNNING
                    else trial.logical_command_id
                ),
                engine_attempt_id=(
                    None
                    if trial.phase is TrialPhase.ENGINE_RUNNING
                    else trial.engine_attempt_id
                ),
                engine_attempt_index=(
                    0
                    if trial.phase is TrialPhase.ENGINE_RUNNING
                    else trial.engine_attempt_index
                ),
                engine_retry_pending=(
                    False
                    if trial.phase is TrialPhase.ENGINE_RUNNING
                    else trial.engine_retry_pending
                ),
                analysis_design_ref_id=(
                    None
                    if trial.phase is TrialPhase.REVIEW_RUNNING
                    else trial.analysis_design_ref_id
                ),
                analysis_review_command_id=(
                    None
                    if trial.phase is TrialPhase.REVIEW_RUNNING
                    else trial.analysis_review_command_id
                ),
                analysis_review_logical_command_id=(
                    None
                    if trial.phase is TrialPhase.REVIEW_RUNNING
                    else trial.analysis_review_logical_command_id
                ),
                analysis_review_attempt_id=(
                    None
                    if trial.phase is TrialPhase.REVIEW_RUNNING
                    else trial.analysis_review_attempt_id
                ),
                analysis_review_attempt_index=(
                    0
                    if trial.phase is TrialPhase.REVIEW_RUNNING
                    else trial.analysis_review_attempt_index
                ),
                analysis_review_retry_pending=(
                    False
                    if trial.phase is TrialPhase.REVIEW_RUNNING
                    else trial.analysis_review_retry_pending
                ),
                trial_record_ref=rewrite_run_ref(
                    trial.trial_record_ref, source.run_id, new_run_id
                ),
                plan_memory_basis=rewrite_run_ref(
                    trial.plan_memory_basis, source.run_id, new_run_id
                ),
                plan_memory_result=rewrite_run_ref(
                    trial.plan_memory_result, source.run_id, new_run_id
                ),
                run_memory_basis=rewrite_run_ref(
                    trial.run_memory_basis, source.run_id, new_run_id
                ),
                run_memory_result=rewrite_run_ref(
                    trial.run_memory_result, source.run_id, new_run_id
                ),
            )
            for trial in source.trials
        )
        transition = TransitionRecord.create(
            run_id=new_run_id,
            kind="run_forked",
            subject_ref=new_run_id,
            from_revision=None,
            to_revision=0,
        )
        run_resources = source.run_resources
        if run_resources is not None:
            local_judge = dict(run_resources["local_judge"])
            state_path = Path(str(local_judge["state_path"]))
            local_judge["state_path"] = str(
                state_path.with_name(f"{new_run_id}.json")
            )
            service_path = state_path.with_name("service.json")
            if service_path.is_file():
                service = json.loads(service_path.read_text(encoding="utf-8"))
                if service.get("status") != "ready":
                    raise ValueError("fork requires a ready Local Judge service")
                handle = service.get("service_handle")
                if not isinstance(handle, dict) or not handle.get("launch_id"):
                    raise ValueError("ready Local Judge service lacks launch identity")
                local_judge["launch_id"] = str(handle["launch_id"])
            run_resources = {**run_resources, "local_judge": local_judge}
        bootstrap = source.bootstrap
        if bootstrap.status in {
            BootstrapStatus.PENDING,
            BootstrapStatus.BASE_EVALUATING,
        }:
            bootstrap = replace(
                bootstrap,
                status=BootstrapStatus.PENDING,
                command_ids=(),
                command_status={},
                command_logical_ids={},
                command_attempt_ids={},
                command_attempt_indices={},
                base_evaluation=None,
                error=None,
                metadata={},
            )
        return replace(
            source,
            run_id=new_run_id,
            revision=0,
            status=(
                RunStatus.BOOTSTRAPPING
                if bootstrap.enabled
                and bootstrap.status is not BootstrapStatus.COMPLETED
                else RunStatus.RUNNING
            ),
            task=replace(source.task, config_ref=source.task.config_ref),
            plans=plans,
            trials=trials,
            pending_actions=(),
            active_agent_calls=(),
            active_engine_commands=(),
            active_review_commands=(),
            agent_sessions=(),
            accepted_plan_refs=refs("accepted_plan_refs"),
            accepted_evidence_refs=tuple(
                ref
                for ref in refs("accepted_evidence_refs")
                if ref.artifact_id not in reset_review_design_refs
            ),
            accepted_experiment_outcome_refs=refs(
                "accepted_experiment_outcome_refs"
            ),
            accepted_finding_refs=refs("accepted_finding_refs"),
            accepted_summary_refs=refs("accepted_summary_refs"),
            accepted_snapshot_refs=snapshots,
            research_outcome=ResearchOutcome.PENDING,
            completion_kind=None,
            finish_request=None,
            coordinators=tuple(
                replace(
                    coordinator,
                    control_status=CoordinatorControlStatus.ACTIVE,
                    effective_plan_limit=coordinator.original_plan_limit,
                    requested_plan_limit=None,
                    control_reason=None,
                    control_requested_revision=None,
                )
                for coordinator in source.coordinators
            ),
            last_transition=transition,
            pause_requested=False,
            pause_reason=None,
            continuation_status=None,
            recovery=None,
            failure=None,
            forked_from=lineage,
            seeded_from=None,
            bootstrap=bootstrap,
            operator_evaluations=tuple(
                replace(
                    item,
                    target_id=rewrite_run_ref(
                        item.target_id, source.run_id, new_run_id
                    ),
                )
                for item in source.operator_evaluations
                if item.status is not OperatorEvaluationStatus.PENDING
            ),
            memory=replace(
                source.memory,
                run_head=rewrite_run_ref(
                    source.memory.run_head, source.run_id, new_run_id
                ),
                plan_heads={
                    rewrite_run_ref(
                        key, source.run_id, new_run_id
                    ): rewrite_run_ref(
                        value, source.run_id, new_run_id
                    )
                    for key, value in source.memory.plan_heads.items()
                },
            ),
            ranking=replace(
                source.ranking,
                entries=tuple(
                    replace(
                        item,
                        subject_id=rewrite_run_ref(
                            item.subject_id, source.run_id, new_run_id
                        ),
                    )
                    for item in source.ranking.entries
                ),
            ),
            plan_catalog=replace(
                source.plan_catalog,
                active_intents=tuple(
                    replace(
                        item,
                        related_plan_keys=tuple(
                            rewrite_run_ref(
                                ref, source.run_id, new_run_id
                            )
                            for ref in item.related_plan_keys
                        ),
                    )
                    for item in source.plan_catalog.active_intents
                ),
            ),
            run_resources=run_resources,
        )
