"""Import a completed Run's accepted bootstrap or Plan frontier."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import copy
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import shutil
import re
from typing import Any, Iterable
from zipfile import BadZipFile, ZipFile

from ade.controller.run_reference_materialization import (
    ARTIFACT_FIELDS,
    RunReferenceMaterializer,
    rewrite_run_ref,
)
from ade.core.bootstrap import BootstrapStatus
from ade.core.coordinator import CoordinatorKind
from ade.core.plan import PlanKind, PlanStatus
from ade.core.operator import OperatorEvaluationStatus
from ade.core.run import (
    FactClass,
    RunSeedBoundaryKind,
    RunSeedProvenance,
    ResearchOutcome,
    RunState,
    RunStatus,
    TransitionRecord,
)
from ade.core.snapshot import SnapshotKind
from ade.core.trial import TrialArchiveStatus, TrialKind, TrialPhase
from ade.memory.repository import FileRunRepository
from ade.memory.repository import RunNotFoundError
from ade.engine.telemetry.seed_import import SeedTrackingImporter
from ade.engine.storage.object_store import FileEngineObjectStore


_BOOTSTRAP_SCIENTIFIC_CONTRACT_FIELDS = (
    "seed",
    "task",
    "model",
    "dataset",
    "training",
    "evaluation",
    "task_mutation_contract",
    "operator_evaluation",
)

_TERMINAL_PLAN_STATUSES = frozenset(
    {
        PlanStatus.COMPLETED,
        PlanStatus.FAILED,
        PlanStatus.REJECTED,
        PlanStatus.CANCELLED,
    }
)


@dataclass(frozen=True)
class RunSeedRequest:
    source_run_id: str
    boundary_kind: RunSeedBoundaryKind
    frontier: dict[str, str]

    @classmethod
    def parse(
        cls,
        source_run_id: str,
        frontier_values: Iterable[str],
    ) -> "RunSeedRequest":
        source_run_id = source_run_id.strip()
        values = tuple(value.strip() for value in frontier_values if value.strip())
        if not source_run_id:
            raise ValueError("Run Seed source Run ID is required")
        if not values:
            raise ValueError("Run Seed frontier is required")
        if values == ("bootstrap",):
            return cls(source_run_id, RunSeedBoundaryKind.BOOTSTRAP, {})
        if "bootstrap" in values:
            raise ValueError("bootstrap cannot be mixed with Coordinator frontiers")
        frontier: dict[str, str] = {}
        for value in values:
            match = re.fullmatch(r"(c\d{3,})=(p\d{3,})", value)
            if match is None:
                raise ValueError(
                    "Run Seed frontier must be bootstrap or cNNN=pNNN"
                )
            coordinator_id, plan_id = match.groups()
            if coordinator_id in frontier:
                raise ValueError(
                    f"Run Seed frontier repeats Coordinator {coordinator_id}"
                )
            frontier[coordinator_id] = plan_id
        return cls(
            source_run_id,
            RunSeedBoundaryKind.ACCEPTED_FRONTIER,
            frontier,
        )


@dataclass(frozen=True)
class RunSeedResolution:
    request: RunSeedRequest
    source_tip: RunState
    source: RunState
    terminal_revisions: dict[str, int]
    resolved_frontier: dict[str, str]
    discarded_nonterminal_scopes: tuple[str, ...]

    @property
    def provenance(self) -> RunSeedProvenance:
        return RunSeedProvenance(
            source_run_id=self.source.run_id,
            source_revision=self.source.revision,
            source_revision_ref=(
                f"{self.source.run_id}@rev-{self.source.revision:06d}"
            ),
            source_deployment_id=None,
            boundary_kind=self.request.boundary_kind,
            requested_frontier=dict(self.request.frontier),
            resolved_frontier=dict(self.resolved_frontier),
            discarded_nonterminal_scopes=self.discarded_nonterminal_scopes,
        )


def resolve_run_seed_source(
    project_root: str | Path,
    reference_run_id: str,
) -> tuple[FileRunRepository, FileEngineObjectStore, str]:
    """Resolve a Run Seed source across deployment-scoped repositories."""
    deployments_root = Path(project_root).resolve() / "runs" / "deployments"
    matches: list[tuple[Path, Path, str]] = []
    if deployments_root.is_dir():
        for deployment in sorted(deployments_root.iterdir()):
            if not deployment.is_dir() or deployment.is_symlink():
                continue
            control_root = deployment / "control"
            run_json = control_root / reference_run_id / "run.json"
            objects_root = deployment / "objects"
            if run_json.is_file() and not run_json.is_symlink() and objects_root.is_dir():
                matches.append((control_root, objects_root, deployment.name))
    if not matches:
        raise RunNotFoundError(reference_run_id)
    if len(matches) != 1:
        deployments = ", ".join(item[2] for item in matches)
        raise ValueError(
            f"Run Seed source is ambiguous across deployments: {deployments}"
        )
    control_root, objects_root, deployment_id = matches[0]
    return (
        FileRunRepository(control_root),
        FileEngineObjectStore(objects_root),
        deployment_id,
    )


class RunSeedImporter:
    """Validate and materialize one accepted source boundary.

    The source remains immutable. Accepted files are copied through the same
    repository path used by forks; a later reflink optimization can replace
    the byte copy without changing this contract.
    """

    def __init__(
        self,
        repository: FileRunRepository,
        objects: FileEngineObjectStore | None = None,
        *,
        source_repository: FileRunRepository | None = None,
        source_objects: FileEngineObjectStore | None = None,
        source_deployment_id: str | None = None,
    ) -> None:
        self.repository = repository
        self.objects = objects
        self.source_repository = source_repository or repository
        self.source_objects = source_objects or objects
        self.source_deployment_id = source_deployment_id

    def resolve(
        self,
        request: RunSeedRequest,
        *,
        target_resolved: dict[str, object],
    ) -> RunSeedResolution:
        """Resolve and admit an exact immutable source boundary."""
        tip = self.source_repository.load(request.source_run_id)
        if tip.status is not RunStatus.COMPLETED:
            raise ValueError("Run Seed source must be a completed Run")
        if tip.bootstrap.status is not BootstrapStatus.COMPLETED:
            raise ValueError("Run Seed source bootstrap is not completed")

        if request.boundary_kind is RunSeedBoundaryKind.BOOTSTRAP:
            revision = tip.bootstrap.baseline_revision
            if revision is None:
                raise ValueError("Run Seed source has no completed bootstrap boundary")
            source = self.source_repository.load_revision(tip.run_id, revision)
            if source.last_transition.kind != "bootstrap_completed":
                raise ValueError(
                    "Run Seed bootstrap revision is not the bootstrap_completed boundary"
                )
            terminal_revisions: dict[str, int] = {}
            resolved_frontier: dict[str, str] = {}
            discarded: tuple[str, ...] = ()
        else:
            terminal_revisions = self._terminal_revisions(tip, request.frontier)
            revision = max(terminal_revisions.values())
            source = self.source_repository.load_revision(tip.run_id, revision)
            resolved_frontier, discarded = self._resolve_frontier(source)
            if resolved_frontier != request.frontier:
                raise ValueError(
                    "Run Seed requested frontier does not equal the anchor's "
                    f"contiguous closed frontier: requested={request.frontier}, "
                    f"resolved={resolved_frontier}"
                )
            self._reject_merged_nonterminal_work(source, discarded)

        self._validate_boundary_evidence(
            source,
            discarded_scopes=discarded,
        )
        self._validate_scientific_contract(
            source.run_id,
            target_resolved,
            boundary_kind=request.boundary_kind,
        )
        return RunSeedResolution(
            request=request,
            source_tip=tip,
            source=source,
            terminal_revisions=terminal_revisions,
            resolved_frontier=resolved_frontier,
            discarded_nonterminal_scopes=discarded,
        )

    def _validate_boundary_evidence(
        self,
        source: RunState,
        *,
        discarded_scopes: tuple[str, ...],
    ) -> None:
        discarded = set(discarded_scopes)
        selected_trials = tuple(
            trial
            for trial in source.trials
            if f"{source.run_id}/{trial.coordinator_id}/{trial.plan_id}"
            not in discarded
        )
        selected_operators = tuple(
            record
            for record in source.operator_evaluations
            if record.coordinator_id in {None, "c000"}
            or record.plan_id is None
            or f"{source.run_id}/{record.coordinator_id}/{record.plan_id}"
            not in discarded
        )
        p000 = self._p000(source)
        if (
            p000.phase is not TrialPhase.ARCHIVED
            or p000.archive_status is not TrialArchiveStatus.ARCHIVED
        ):
            raise ValueError("Run Seed source P000 is not archived")
        if self.source_objects is not None:
            missing_p000_refs = tuple(
                ref for ref in p000.result_refs if not self.source_objects.exists(ref)
            )
            if missing_p000_refs:
                raise ValueError(
                    "Run Seed source P000 result is unavailable: "
                    + ", ".join(missing_p000_refs)
                )
        expected_operator_targets = {
            f"{source.run_id}/c000/p000/base",
            f"{source.run_id}/{p000.coordinator_id}/{p000.plan_id}/{p000.trial_id}",
        }
        operator_records = {
            record.target_id: record for record in selected_operators
        }
        if not expected_operator_targets.issubset(operator_records):
            raise ValueError(
                "Run Seed source requires terminal Base and P000 operator records"
            )
        terminal_operator_statuses = {
            OperatorEvaluationStatus.COMPLETED,
            OperatorEvaluationStatus.FAILED,
            OperatorEvaluationStatus.CANCELLED,
            OperatorEvaluationStatus.NOT_APPLICABLE,
        }
        for target_id in expected_operator_targets:
            record = operator_records[target_id]
            if record.status not in terminal_operator_statuses:
                raise ValueError(
                    f"Run Seed source operator evaluation is not terminal: {target_id}"
                )
            if (
                record.status is OperatorEvaluationStatus.COMPLETED
                and record.result_ref is None
            ):
                raise ValueError(
                    "completed Run Seed operator evaluation has no result: "
                    f"{target_id}"
                )
        if source.task.task_id in {"reward_design", "curriculum_learning"}:
            base = source.bootstrap.base_evaluation
            if (
                base is None
                or base.profile_status.get("online") != "succeeded"
                or not base.online_result_ref
            ):
                raise ValueError(
                    "RFT Run Seed requires a successful step-0 online evaluation"
                )
            if self.source_objects is not None and not self.source_objects.exists(
                base.online_result_ref
            ):
                raise ValueError("RFT Run Seed step-0 online result is unavailable")
        if self.source_objects is None:
            return
        referenced_results = tuple(
            dict.fromkeys(
                (
                    *(source.bootstrap.base_evaluation.result_refs if source.bootstrap.base_evaluation else ()),
                    *(ref for trial in selected_trials for ref in trial.result_refs),
                    *(
                        record.result_ref
                        for record in selected_operators
                        if record.result_ref is not None
                        and record.status in terminal_operator_statuses
                    ),
                )
            )
        )
        missing_results = tuple(
            ref for ref in referenced_results if not self.source_objects.exists(ref)
        )
        if missing_results:
            raise ValueError(
                "Run Seed Engine result is unavailable: "
                + ", ".join(missing_results)
            )
        for source_ref in referenced_results:
            try:
                payload = self.source_objects.read_json(source_ref)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            units = payload.get("units")
            if isinstance(units, list):
                missing_units = tuple(
                    str(unit["uri"])
                    for unit in units
                    if isinstance(unit, dict)
                    and isinstance(unit.get("uri"), str)
                    and not self.source_objects.exists(str(unit["uri"]))
                )
                if missing_units:
                    raise ValueError(
                        "Run Seed evaluation payload is unavailable: "
                        + ", ".join(missing_units)
                    )
            manifest_path = payload.get("trial_artifact_manifest_path")
            if isinstance(manifest_path, str):
                try:
                    manifest_payload = (
                        self.source_objects.read_json(manifest_path)
                        if manifest_path.startswith("engine://")
                        else json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                    )
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        "Run Seed Trial artifact manifest is unavailable: "
                        f"{manifest_path}"
                    ) from error
                if not isinstance(manifest_payload, dict):
                    raise ValueError(
                        "Run Seed Trial artifact manifest is invalid: "
                        f"{manifest_path}"
                    )
        for record in selected_operators:
            if (
                record.status is OperatorEvaluationStatus.COMPLETED
                and record.result_ref is not None
                and record.result_ref.endswith("/result.json")
            ):
                raw_ref = (
                    f"{record.result_ref[:-len('/result.json')]}"
                    "/raw/units/operator_test.json"
                )
                if not self.source_objects.exists(raw_ref):
                    raise ValueError(
                        f"Run Seed operator payload is unavailable: {raw_ref}"
                    )

    def _terminal_revisions(
        self,
        tip: RunState,
        requested: dict[str, str],
    ) -> dict[str, int]:
        coordinator_ids = {
            coordinator.coordinator_id
            for coordinator in tip.coordinators
            if coordinator.kind is CoordinatorKind.SEARCH
        }
        if set(requested) != coordinator_ids:
            missing = sorted(coordinator_ids - set(requested))
            extra = sorted(set(requested) - coordinator_ids)
            raise ValueError(
                "Run Seed frontier must specify every Search Coordinator exactly "
                f"once; missing={missing}, extra={extra}"
            )
        unresolved = dict(requested)
        revisions: dict[str, int] = {}
        for revision in range(tip.revision + 1):
            state = self.source_repository.load_revision(tip.run_id, revision)
            plans = {
                (plan.coordinator_id, plan.plan_id): plan
                for plan in state.plans
                if plan.kind is PlanKind.SEARCH
            }
            for coordinator_id, plan_id in tuple(unresolved.items()):
                plan = plans.get((coordinator_id, plan_id))
                if plan is not None and plan.status in _TERMINAL_PLAN_STATUSES:
                    revisions[coordinator_id] = revision
                    del unresolved[coordinator_id]
            if not unresolved:
                break
        if unresolved:
            requested_refs = ", ".join(
                f"{coordinator_id}/{plan_id}"
                for coordinator_id, plan_id in sorted(unresolved.items())
            )
            raise ValueError(
                f"Run Seed requested Plan is not terminal: {requested_refs}"
            )
        return revisions

    @staticmethod
    def _resolve_frontier(
        source: RunState,
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        resolved: dict[str, str] = {}
        discarded: list[str] = []
        coordinator_ids = sorted(
            coordinator.coordinator_id
            for coordinator in source.coordinators
            if coordinator.kind is CoordinatorKind.SEARCH
        )
        for coordinator_id in coordinator_ids:
            plans = tuple(
                plan
                for plan in source.plans
                if plan.kind is PlanKind.SEARCH
                and plan.coordinator_id == coordinator_id
            )
            terminal_numbers = {
                int(plan.plan_id[1:])
                for plan in plans
                if plan.status in _TERMINAL_PLAN_STATUSES
            }
            frontier = 0
            while frontier + 1 in terminal_numbers:
                frontier += 1
            if any(number > frontier for number in terminal_numbers):
                raise ValueError(
                    f"Run Seed source has a non-contiguous terminal frontier for {coordinator_id}"
                )
            if frontier:
                resolved[coordinator_id] = f"p{frontier:03d}"
            discarded.extend(
                f"{source.run_id}/{plan.coordinator_id}/{plan.plan_id}"
                for plan in plans
                if plan.status not in _TERMINAL_PLAN_STATUSES
            )
        return resolved, tuple(sorted(discarded))

    @staticmethod
    def _reject_merged_nonterminal_work(
        source: RunState,
        discarded_scopes: tuple[str, ...],
    ) -> None:
        discarded = set(discarded_scopes)
        for trial in source.trials:
            plan_scope = f"{source.run_id}/{trial.coordinator_id}/{trial.plan_id}"
            if plan_scope not in discarded:
                continue
            if (
                trial.archive_status is TrialArchiveStatus.ARCHIVED
                or trial.run_memory_result is not None
            ):
                raise ValueError(
                    "Run Seed cannot discard non-terminal Plan work that is "
                    f"archived or merged into Run Memory: {plan_scope}"
                )
        for result in source.plan_catalog.source_eligible_results:
            plan_scope = (
                f"{source.run_id}/{result.coordinator_id}/{result.plan_id}"
            )
            if plan_scope in discarded:
                raise ValueError(
                    "Run Seed cannot discard a non-terminal Plan already present "
                    f"in the accepted Plan Catalog: {plan_scope}"
                )

    def materialize(
        self,
        resolution: RunSeedResolution,
        target: RunState,
    ) -> tuple[RunState, tuple[tuple[Any, bytes], ...], tuple[tuple[str, bytes], ...]]:
        source = self.project(resolution)
        self._validate_target_budget(
            source,
            target_plan_budget=target.portfolio.max_plans,
            target_plan_limits={
                coordinator.coordinator_id: coordinator.effective_plan_limit
                for coordinator in target.coordinators
                if coordinator.kind is CoordinatorKind.SEARCH
            },
        )
        self._materialize_trial_artifact_references(source)
        helper = RunReferenceMaterializer(
            self.repository,
            source_repository=self.source_repository,
        )
        artifacts, rewritten = helper.copy_artifacts(
            source, new_run_id=target.run_id
        )
        accepted_transition_ids = {
            self.source_repository.load_revision(source.run_id, revision).last_transition.transition_id
            for revision in range(source.revision + 1)
        }
        inherited, snapshots = helper.inherited_files(
            source,
            new_run_id=target.run_id,
            accepted_transition_ids=accepted_transition_ids,
        )
        inherited = inherited + SeedTrackingImporter(
            source_repository=self.source_repository,
            target_repository=self.repository,
            source_deployment_id=self.source_deployment_id,
            source_objects=self.source_objects,
        ).staged_files(source)

        def artifact_refs(field: str):
            return tuple(rewritten[item.artifact_id] for item in getattr(source, field))

        plans = tuple(
            replace(
                self._rewrite_object(plan, source.run_id, target.run_id),
                basis_revision=0,
                reservation_id=None,
                reservation_sequence=None,
            )
            for plan in source.plans
        )
        memory = replace(
            source.memory,
            run_head=rewrite_run_ref(
                source.memory.run_head, source.run_id, target.run_id
            ),
            plan_heads={
                rewrite_run_ref(key, source.run_id, target.run_id):
                rewrite_run_ref(value, source.run_id, target.run_id)
                for key, value in source.memory.plan_heads.items()
            },
        )
        ranking = replace(
            source.ranking,
            revision=0,
            entries=tuple(
                replace(
                    item,
                    subject_id=rewrite_run_ref(
                        item.subject_id, source.run_id, target.run_id
                    ),
                    accepted_revision=0,
                    artifact_ref_id=item.artifact_ref_id,
                )
                for item in source.ranking.entries
            ),
        )
        accepted_fact_refs = tuple(
            ref.uri
            for field in (
                "accepted_plan_refs",
                "accepted_evidence_refs",
                "accepted_experiment_outcome_refs",
                "accepted_finding_refs",
                "accepted_summary_refs",
            )
            for ref in artifact_refs(field)
        )
        if not accepted_fact_refs:
            raise ValueError("Run Seed source has no accepted scientific artifacts")
        transition = TransitionRecord.create(
            run_id=target.run_id,
            kind="run_seeded",
            subject_ref=target.run_id,
            from_revision=None,
            to_revision=0,
            fact_class=FactClass.SCIENTIFIC,
            accepted_fact_refs=accepted_fact_refs,
            origin_refs=(f"{source.run_id}@rev-{source.revision:06d}",),
        )
        online_result_ref = None
        imported_result_refs: dict[str, str] = {}
        if source.bootstrap.base_evaluation is not None:
            base = source.bootstrap.base_evaluation
            if self.source_objects is not None and self.objects is not None:
                engine_refs = tuple(
                    dict.fromkeys(
                        (
                            *base.result_refs,
                            *(ref for trial in source.trials for ref in trial.result_refs),
                            *(
                                record.result_ref
                                for record in source.operator_evaluations
                                if record.result_ref is not None
                            ),
                        )
                    )
                )
                imported_result_refs = self._materialize_engine_results(
                    target.run_id,
                    engine_refs,
                    operator_result_refs=tuple(
                        record.result_ref
                        for record in source.operator_evaluations
                        if record.result_ref is not None
                    ),
                )
            online_result_ref = (
                imported_result_refs.get(base.online_result_ref, base.online_result_ref)
                if base.online_result_ref
                else None
            )
        trials = tuple(
            replace(
                self._rewrite_object(trial, source.run_id, target.run_id),
                result_refs=tuple(
                    imported_result_refs.get(
                        ref, rewrite_run_ref(ref, source.run_id, target.run_id)
                    )
                    for ref in trial.result_refs
                ),
            )
            for trial in source.trials
        )
        operator_evaluations = tuple(
            replace(
                self._rewrite_object(record, source.run_id, target.run_id),
                result_ref=(
                    imported_result_refs.get(
                        record.result_ref,
                        rewrite_run_ref(
                            record.result_ref, source.run_id, target.run_id
                        ),
                    )
                    if record.result_ref is not None
                    else None
                ),
            )
            for record in source.operator_evaluations
        )
        ranking = replace(
            ranking,
            entries=tuple(
                replace(
                    entry,
                    operator_result_ref=(
                        imported_result_refs.get(
                            entry.operator_result_ref,
                            rewrite_run_ref(
                                entry.operator_result_ref,
                                source.run_id,
                                target.run_id,
                            ),
                        )
                        if entry.operator_result_ref is not None
                        else None
                    ),
                )
                for entry in ranking.entries
            ),
        )
        bootstrap = replace(
            target.bootstrap,
            status=BootstrapStatus.COMPLETED,
            reference_enabled=True,
            reference_run_id=source.run_id,
            reference_revision=source.bootstrap.baseline_revision,
            command_ids=(),
            command_status={},
            command_logical_ids={},
            command_attempt_ids={},
            command_attempt_indices={},
            base_evaluation=(
                replace(
                    source.bootstrap.base_evaluation,
                    result_refs=tuple(
                        imported_result_refs.get(
                            ref,
                            rewrite_run_ref(
                                ref, source.run_id, target.run_id
                            ),
                        )
                        for ref in source.bootstrap.base_evaluation.result_refs
                    ),
                    online_result_ref=online_result_ref,
                    offline_evidence_ref_id=source.bootstrap.base_evaluation.offline_evidence_ref_id,
                )
                if source.bootstrap.base_evaluation is not None
                else None
            ),
            baseline_revision=0,
            metadata={
                **self._rewrite_object(
                    source.bootstrap.metadata, source.run_id, target.run_id
                ),
                "imported_from": {
                    "run_id": source.run_id,
                    "revision": source.revision,
                    "boundary_kind": resolution.request.boundary_kind.value,
                    **(
                        {"deployment_id": self.source_deployment_id}
                        if self.source_deployment_id is not None
                        else {}
                    ),
                },
            },
        )
        imported = replace(
            target,
            status=RunStatus.RUNNING,
            revision=0,
            coordinators=target.coordinators,
            plans=plans,
            trials=trials,
            accepted_plan_refs=artifact_refs("accepted_plan_refs"),
            accepted_evidence_refs=artifact_refs("accepted_evidence_refs"),
            accepted_experiment_outcome_refs=artifact_refs("accepted_experiment_outcome_refs"),
            accepted_finding_refs=artifact_refs("accepted_finding_refs"),
            accepted_summary_refs=artifact_refs("accepted_summary_refs"),
            accepted_snapshot_refs=snapshots,
            latest_run_snapshot_ref_id=source.latest_run_snapshot_ref_id,
            memory=memory,
            ranking=ranking,
            insight_graph=replace(source.insight_graph, revision=0),
            plan_catalog=replace(
                self._rewrite_object(
                    source.plan_catalog, source.run_id, target.run_id
                ),
                revision=0,
                active_intents=(),
                source_eligible_results=tuple(
                    replace(item, archived_revision=0)
                    for item in self._rewrite_object(
                        source.plan_catalog.source_eligible_results,
                        source.run_id,
                        target.run_id,
                    )
                ),
            ),
            operator_evaluations=operator_evaluations,
            bootstrap=bootstrap,
            last_transition=transition,
            active_agent_calls=(),
            active_engine_commands=(),
            active_review_commands=(),
            pending_actions=(),
            planning_queue=(),
            rm_merge_queue=(),
            agent_sessions=(),
            failure=None,
            pause_requested=False,
            pause_reason=None,
            continuation_status=None,
            recovery=None,
            completion_kind=None,
            finish_request=None,
            research_outcome=ResearchOutcome.PENDING,
            forked_from=None,
            seeded_from=replace(
                resolution.provenance,
                source_deployment_id=self.source_deployment_id,
            ),
        )
        return imported, artifacts, inherited

    def project(self, resolution: RunSeedResolution) -> RunState:
        source = resolution.source
        discarded = set(resolution.discarded_nonterminal_scopes)
        selected_plan_keys = {
            (plan.coordinator_id, plan.plan_id)
            for plan in source.plans
            if plan.kind is PlanKind.BOOTSTRAP
            or plan.status in _TERMINAL_PLAN_STATUSES
        }
        plans = tuple(
            plan
            for plan in source.plans
            if (plan.coordinator_id, plan.plan_id) in selected_plan_keys
        )
        trials = tuple(
            trial
            for trial in source.trials
            if (trial.coordinator_id, trial.plan_id) in selected_plan_keys
        )
        selected_prefixes = {
            f"{source.run_id}/{coordinator_id}/{plan_id}"
            for coordinator_id, plan_id in selected_plan_keys
        }

        excluded_uris: set[str] = set()
        for revision in range(source.revision + 1):
            transition = self.source_repository.load_revision(
                source.run_id, revision
            ).last_transition
            if any(
                transition.subject_ref == scope
                or transition.subject_ref.startswith(f"{scope}/")
                for scope in discarded
            ):
                excluded_uris.update(transition.accepted_fact_refs)
        excluded_ids: set[str] = set()
        for plan in source.plans:
            scope = f"{source.run_id}/{plan.coordinator_id}/{plan.plan_id}"
            if scope not in discarded:
                continue
            excluded_ids.update(
                value
                for value in (
                    plan.decision_ref_id,
                    plan.decision_report_ref_id,
                    plan.best_artifact_ref_id,
                )
                if value is not None
            )
            if plan.relation is not None:
                excluded_ids.update(plan.relation.seed_artifact_ref_ids)
        for trial in source.trials:
            scope = f"{source.run_id}/{trial.coordinator_id}/{trial.plan_id}"
            if scope not in discarded:
                continue
            for field in fields(trial):
                value = getattr(trial, field.name)
                if field.name.endswith("_ref_id") and isinstance(value, str):
                    excluded_ids.add(value)
                elif field.name.endswith("_ref_ids") and isinstance(value, tuple):
                    excluded_ids.update(str(item) for item in value)

        def accepted(field: str):
            return tuple(
                ref
                for ref in getattr(source, field)
                if ref.artifact_id not in excluded_ids and ref.uri not in excluded_uris
            )

        snapshots = tuple(
            ref
            for ref in source.accepted_snapshot_refs
            if ref.kind is SnapshotKind.RUN
            or ref.coordinator_id == "c000"
            or (
                ref.coordinator_id is not None
                and ref.plan_id is not None
                and f"{source.run_id}/{ref.coordinator_id}/{ref.plan_id}"
                in selected_prefixes
            )
        )
        operators = tuple(
            record
            for record in source.operator_evaluations
            if record.coordinator_id in {None, "c000"}
            or (
                record.coordinator_id is not None
                and record.plan_id is not None
                and f"{source.run_id}/{record.coordinator_id}/{record.plan_id}"
                in selected_prefixes
            )
        )
        ranking = replace(
            source.ranking,
            entries=tuple(
                entry
                for entry in source.ranking.entries
                if entry.subject_id.startswith(f"{source.run_id}/c000/")
                or any(
                    entry.subject_id == prefix
                    or entry.subject_id.startswith(f"{prefix}/")
                    for prefix in selected_prefixes
                )
            ),
        )
        catalog = replace(
            source.plan_catalog,
            active_intents=(),
            source_eligible_results=tuple(
                item
                for item in source.plan_catalog.source_eligible_results
                if (item.coordinator_id, item.plan_id) in selected_plan_keys
            ),
        )
        selected_memory_prefixes = (
            f"{source.run_id}/{coordinator_id}/{plan_id}"
            for coordinator_id, plan_id in selected_plan_keys
        )
        memory_prefixes = set(selected_memory_prefixes)
        memory = replace(
            source.memory,
            plan_heads={
                key: value
                for key, value in source.memory.plan_heads.items()
                if key in memory_prefixes
            },
        )
        latest_run_snapshot = source.latest_run_snapshot_ref_id
        if latest_run_snapshot not in {item.snapshot_id for item in snapshots}:
            latest_run_snapshot = next(
                (
                    item.snapshot_id
                    for item in reversed(snapshots)
                    if item.kind is SnapshotKind.RUN
                ),
                None,
            )
        return replace(
            source,
            task=replace(source.task, config_ref=None),
            plans=plans,
            trials=trials,
            accepted_plan_refs=accepted("accepted_plan_refs"),
            accepted_evidence_refs=accepted("accepted_evidence_refs"),
            accepted_experiment_outcome_refs=accepted(
                "accepted_experiment_outcome_refs"
            ),
            accepted_finding_refs=accepted("accepted_finding_refs"),
            accepted_summary_refs=accepted("accepted_summary_refs"),
            accepted_snapshot_refs=snapshots,
            latest_run_snapshot_ref_id=latest_run_snapshot,
            operator_evaluations=operators,
            ranking=ranking,
            plan_catalog=catalog,
            memory=memory,
            pending_actions=(),
            active_agent_calls=(),
            active_engine_commands=(),
            active_review_commands=(),
            planning_queue=(),
            rm_merge_queue=(),
            agent_sessions=(),
        )

    def inspect(
        self,
        resolution: RunSeedResolution,
        *,
        target_plan_budget: int,
    ) -> dict[str, object]:
        source = self.project(resolution)
        self._validate_durable_projection(source)
        self._materialize_trial_artifact_references(source, copy=False)
        search_plans = tuple(
            plan for plan in source.plans if plan.kind is PlanKind.SEARCH
        )
        search_trials = tuple(
            trial for trial in source.trials if trial.kind is TrialKind.SEARCH
        )
        search_coordinator_ids = tuple(
            sorted({plan.coordinator_id for plan in search_plans})
        )
        if search_coordinator_ids and (
            target_plan_budget % len(search_coordinator_ids)
        ):
            raise ValueError(
                "Run Seed target Plan budget must divide across Search Coordinators"
            )
        if search_coordinator_ids:
            per_coordinator_limit = (
                target_plan_budget // len(search_coordinator_ids)
            )
            self._validate_target_budget(
                source,
                target_plan_budget=target_plan_budget,
                target_plan_limits={
                    coordinator_id: per_coordinator_limit
                    for coordinator_id in search_coordinator_ids
                },
            )
        elif target_plan_budget < 1:
            raise ValueError(
                "Run Seed target Plan budget must leave at least one unimported slot"
            )
        return {
            "source_run_id": source.run_id,
            "source_deployment_id": self.source_deployment_id,
            "source_terminal_status": resolution.source_tip.status.value,
            "source_terminal_revision": resolution.source_tip.revision,
            "boundary_kind": resolution.request.boundary_kind.value,
            "requested_terminal_revisions": dict(
                resolution.terminal_revisions
            ),
            "anchor_revision": source.revision,
            "requested_frontier": dict(resolution.request.frontier),
            "resolved_frontier": dict(resolution.resolved_frontier),
            "discarded_nonterminal_scopes": list(
                resolution.discarded_nonterminal_scopes
            ),
            "imported_plan_slots": len(search_plans),
            "imported_trials": len(search_trials),
            "plan_statuses": {
                f"{plan.coordinator_id}/{plan.plan_id}": plan.status.value
                for plan in search_plans
            },
            "target_plan_budget": target_plan_budget,
            "target_remaining_plan_budget": (
                target_plan_budget - len(search_plans)
            ),
            "missing_durable_refs": [],
        }

    @staticmethod
    def _validate_target_budget(
        source: RunState,
        *,
        target_plan_budget: int,
        target_plan_limits: dict[str, int],
    ) -> None:
        imported_coordinator_ids = {
            plan.coordinator_id
            for plan in source.plans
            if plan.kind is PlanKind.SEARCH
        }
        if not imported_coordinator_ids.issubset(target_plan_limits):
            raise ValueError(
                "Run Seed source and target Search Coordinator sets differ"
            )
        imported_counts = {
            coordinator_id: sum(
                plan.kind is PlanKind.SEARCH
                and plan.coordinator_id == coordinator_id
                for plan in source.plans
            )
            for coordinator_id in target_plan_limits
        }
        over_limit = {
            coordinator_id: {
                "imported": imported_counts[coordinator_id],
                "limit": target_plan_limits[coordinator_id],
            }
            for coordinator_id in imported_counts
            if imported_counts[coordinator_id] > target_plan_limits[coordinator_id]
        }
        if over_limit:
            raise ValueError(
                "Run Seed imported Plan slots exceed target Coordinator Plan "
                f"budget: {over_limit}"
            )
        if sum(imported_counts.values()) >= target_plan_budget:
            raise ValueError(
                "Run Seed target Plan budget must leave at least one unimported slot"
            )

    def _validate_durable_projection(self, source: RunState) -> None:
        run_root = self.source_repository.layout.run_dir(source.run_id)
        for field in ARTIFACT_FIELDS:
            for ref in getattr(source, field):
                self.source_repository.read_artifact(source.run_id, ref)
        for ref in source.accepted_snapshot_refs:
            root = Path(ref.root).resolve()
            if (
                not root.is_relative_to(run_root)
                or not root.is_dir()
                or not (root / "manifest.json").is_file()
            ):
                raise ValueError(
                    f"Run Seed accepted snapshot is unavailable: {ref.snapshot_id}"
                )
        run_memory = (
            run_root
            / "memory"
            / "run"
            / "versions"
            / source.memory.run_head.rsplit("/", 1)[-1]
        )
        if not run_memory.is_dir():
            raise ValueError(
                f"Run Seed Run Memory head is unavailable: {source.memory.run_head}"
            )
        for plan_ref, memory_ref in source.memory.plan_heads.items():
            parts = plan_ref.split("/")
            plan_memory = (
                run_root
                / "memory"
                / "plans"
                / parts[1]
                / parts[2]
                / "versions"
                / memory_ref.rsplit("/", 1)[-1]
            )
            if not plan_memory.is_dir():
                raise ValueError(
                    f"Run Seed Plan Memory head is unavailable: {memory_ref}"
                )

    def _materialize_trial_artifact_references(
        self,
        source: RunState,
        *,
        copy: bool = True,
    ) -> None:
        """Validate and optionally copy package-owned durable Trial files."""
        package_refs = {
            ref.artifact_id: ref
            for field in (
                "accepted_plan_refs",
                "accepted_evidence_refs",
                "accepted_experiment_outcome_refs",
                "accepted_finding_refs",
                "accepted_summary_refs",
            )
            for ref in getattr(source, field)
            if ref.kind == "experiment_package"
        }
        prefix = "engine://trial-artifacts/"
        source_root = (
            self.source_repository.layout.runs_root.parent
            / "engine-work"
            / "trial_artifacts"
        ).resolve()
        target_root = (
            (
                self.repository.layout.runs_root.parent
                / "engine-work"
                / "trial_artifacts"
            ).resolve()
            if copy
            else None
        )
        for trial in source.trials:
            if trial.package_ref_id is None:
                continue
            package_ref = package_refs.get(trial.package_ref_id)
            if package_ref is None:
                raise ValueError(
                    f"Run Seed experiment package is unavailable: {trial.package_ref_id}"
                )
            try:
                package = self.source_repository.read_artifact(
                    source.run_id, package_ref
                )
                with ZipFile(BytesIO(package), "r") as archive:
                    manifest = json.loads(
                        archive.read("experiment/manifest.json")
                    )
            except (
                BadZipFile,
                KeyError,
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as error:
                raise ValueError("Run Seed experiment package is invalid") from error
            artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
            if not isinstance(artifacts, list):
                raise ValueError("Run Seed Trial artifacts are unavailable")
            self._copy_durable_trial_artifacts(
                artifacts,
                prefix=prefix,
                source_root=source_root,
                target_root=target_root,
            )

    @staticmethod
    def _copy_durable_trial_artifacts(
        artifacts: list[object],
        *,
        prefix: str,
        source_root: Path,
        target_root: Path | None,
    ) -> None:
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            storage = artifact.get("storage")
            if not isinstance(storage, dict) or storage.get("mode") != "durable_reference":
                continue
            uri = str(storage.get("uri") or "")
            relative = PurePosixPath(uri.removeprefix(prefix))
            if (
                not uri.startswith(prefix)
                or relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) < 6
            ):
                raise ValueError("Run Seed Trial artifact path is invalid")
            source_path = source_root.joinpath(*relative.parts)
            expected_size = storage.get("size_bytes")
            if (
                not source_path.is_file()
                or source_path.is_symlink()
                or not source_path.resolve().is_relative_to(source_root)
                or type(expected_size) is not int
                or source_path.stat().st_size != expected_size
                or artifact.get("size_bytes") != expected_size
                or artifact.get("sha256") != storage.get("sha256")
            ):
                raise ValueError("Run Seed Trial artifact binding is invalid")
            if target_root is None:
                continue
            target_path = target_root.joinpath(*relative.parts)
            if target_path.exists():
                if (
                    not target_path.is_file()
                    or target_path.is_symlink()
                    or not target_path.resolve().is_relative_to(target_root)
                    or target_path.stat().st_size != expected_size
                ):
                    raise ValueError("materialized Run Seed Trial artifact is invalid")
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)

    def publish_tracking(
        self,
        source: RunState,
        target: RunState,
        *,
        target_tracking: dict[str, object],
    ) -> tuple[dict[str, object], ...]:
        return SeedTrackingImporter(
            source_repository=self.source_repository,
            target_repository=self.repository,
            source_deployment_id=self.source_deployment_id,
            source_objects=self.source_objects,
        ).publish(source, target, target_tracking=target_tracking)

    def _materialize_engine_results(
        self,
        target_run_id: str,
        source_refs: tuple[str, ...],
        *,
        operator_result_refs: tuple[str, ...],
    ) -> dict[str, str]:
        if self.source_objects is None or self.objects is None:
            return {}
        imported = {
            source_ref: (
                f"engine://imports/{target_run_id}/seed-import/"
                f"accepted-results/{index:03d}-{Path(source_ref).name}"
            )
            for index, source_ref in enumerate(source_refs)
        }
        external_json: dict[str, dict[str, Any]] = {}
        dependent_refs: list[str] = []
        for source_ref in source_refs:
            try:
                payload = self.source_objects.read_json(source_ref)
            except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            units = payload.get("units")
            if isinstance(units, list):
                dependent_refs.extend(
                    str(unit["uri"])
                    for unit in units
                    if isinstance(unit, dict)
                    and isinstance(unit.get("uri"), str)
                    and self.source_objects.exists(str(unit["uri"]))
                )
            manifest_path = payload.get("trial_artifact_manifest_path")
            if isinstance(manifest_path, str) and (
                self.source_objects.exists(manifest_path) if manifest_path.startswith("engine://")
                else Path(manifest_path).is_file()
            ):
                try:
                    manifest_payload = (
                        self.source_objects.read_json(manifest_path)
                        if manifest_path.startswith("engine://")
                        else json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    manifest_payload = None
                if isinstance(manifest_payload, dict):
                    imported[manifest_path] = (
                        f"engine://imports/{target_run_id}/seed-import/"
                        f"trial-artifacts/{Path(manifest_path).name}"
                    )
                    external_json[manifest_path] = manifest_payload
        for source_ref in operator_result_refs:
            if not source_ref.endswith("/result.json"):
                continue
            raw_ref = (
                f"{source_ref[:-len('/result.json')]}/raw/units/operator_test.json"
            )
            if self.source_objects.exists(raw_ref):
                target_result_ref = imported[source_ref]
                imported[raw_ref] = (
                    f"{target_result_ref[:-len('/result.json')]}"
                    "/raw/units/operator_test.json"
                )
        for source_ref in dict.fromkeys(dependent_refs):
            if source_ref not in imported:
                imported[source_ref] = (
                    f"engine://imports/{target_run_id}/seed-import/accepted-results/"
                    f"{len(imported):03d}-{Path(source_ref).name}"
                )
        for source_ref, target_ref in imported.items():
            if source_ref in external_json:
                self.objects.put_json(
                    target_ref,
                    self._rewrite_engine_refs(external_json[source_ref], imported),
                )
                continue
            try:
                payload = self.source_objects.read_json(source_ref)
            except (TypeError, ValueError, json.JSONDecodeError):
                self.objects.put_bytes(
                    target_ref,
                    self.source_objects.read_bytes(source_ref),
                )
            else:
                self.objects.put_json(
                    target_ref,
                    self._rewrite_engine_refs(payload, imported),
                )
        return imported

    @classmethod
    def _rewrite_engine_refs(cls, value: Any, imported: dict[str, str]):
        if isinstance(value, str):
            return imported.get(value, value)
        if isinstance(value, list):
            return [cls._rewrite_engine_refs(item, imported) for item in value]
        if isinstance(value, dict):
            return {
                key: cls._rewrite_engine_refs(item, imported)
                for key, item in value.items()
            }
        return value

    def _p000(self, source: RunState):
        matches = tuple(
            trial for trial in source.trials
            if trial.kind is TrialKind.BOOTSTRAP_BASELINE
            and trial.plan_id == "p000"
        )
        if len(matches) != 1:
            raise ValueError("Run Seed source must contain exactly one P000 Trial")
        return matches[0]

    def _rewrite_object(cls, value: Any, source_run_id: str, target_run_id: str):
        if isinstance(value, str):
            return rewrite_run_ref(value, source_run_id, target_run_id)
        if is_dataclass(value):
            return replace(
                value,
                **{
                    field.name: cls._rewrite_object(
                        getattr(value, field.name), source_run_id, target_run_id
                    )
                    for field in fields(value)
                },
            )
        if isinstance(value, tuple):
            return tuple(cls._rewrite_object(item, source_run_id, target_run_id) for item in value)
        if isinstance(value, list):
            return [cls._rewrite_object(item, source_run_id, target_run_id) for item in value]
        if isinstance(value, dict):
            return {
                key: cls._rewrite_object(item, source_run_id, target_run_id)
                for key, item in value.items()
            }
        return value

    def _validate_scientific_contract(
        self,
        reference_run_id: str,
        target_resolved: dict[str, object],
        *,
        boundary_kind: RunSeedBoundaryKind,
    ) -> None:
        path = self.source_repository.layout.run_dir(reference_run_id) / "config" / "resolved.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("Run Seed source resolved config is unavailable")
        source = json.loads(path.read_text(encoding="utf-8"))
        if boundary_kind is RunSeedBoundaryKind.ACCEPTED_FRONTIER:
            source_normalized = _frontier_scientific_contract(source)
            target_normalized = _frontier_scientific_contract(target_resolved)
            if source_normalized != target_normalized:
                differing = sorted(
                    key
                    for key in set(source_normalized) | set(target_normalized)
                    if source_normalized.get(key) != target_normalized.get(key)
                )
                raise ValueError(
                    "Run Seed accepted frontier is scientifically incompatible at "
                    + ", ".join(f"resolved.{key}" for key in differing)
                )
            return
        source_task = source.get("task")
        target_task = target_resolved.get("task")
        reward_design_bootstrap = (
            isinstance(source_task, dict)
            and isinstance(target_task, dict)
            and source_task.get("type") == "reward_design"
            and target_task.get("type") == "reward_design"
        )
        for key in _BOOTSTRAP_SCIENTIFIC_CONTRACT_FIELDS:
            source_value = source.get(key)
            target_value = target_resolved.get(key)
            if key == "training" and reward_design_bootstrap:
                source_value = _without_bootstrap_group_credit(source_value)
                target_value = _without_bootstrap_group_credit(target_value)
            if key == "task" and reward_design_bootstrap:
                source_value = _without_bootstrap_task_group_credit(source_value)
                target_value = _without_bootstrap_task_group_credit(target_value)
            if source_value != target_value:
                raise ValueError(
                    f"Run Seed bootstrap is incompatible at resolved.{key}"
                )


def _without_bootstrap_group_credit(value: object) -> object:
    if not isinstance(value, dict):
        return value
    normalized = copy.deepcopy(value)
    if isinstance(normalized.get("rft"), dict):
        rft = normalized["rft"]
        rft.pop("group_credit", None)
        rft.pop("semantic_evidence_interval", None)
    return normalized


def _without_bootstrap_task_group_credit(value: object) -> object:
    """Ignore only the target Group-Credit controls in a reward task view."""
    if not isinstance(value, dict):
        return value
    normalized = copy.deepcopy(value)
    config = normalized.get("config")
    if isinstance(config, dict) and isinstance(config.get("rft"), dict):
        rft = config["rft"]
        rft.pop("group_credit", None)
        rft.pop("semantic_evidence_interval", None)
    return normalized


def _frontier_scientific_contract(value: dict[str, object]) -> dict[str, object]:
    """Remove only the operational and budget differences allowed for a frontier."""
    normalized = copy.deepcopy(value)
    for key in (
        "experiment_id",
        "config_digest",
        "source_manifest",
        "deployment",
        "run_resources",
        "runtime",
        "tracking",
    ):
        normalized.pop(key, None)
    search = normalized.get("search")
    if isinstance(search, dict):
        for key in ("plans_per_coordinator", "max_plans", "max_search_trials"):
            search.pop(key, None)
    engine = normalized.get("engine")
    if isinstance(engine, dict):
        for key in (
            "command_transport",
            "claim_timeout_seconds",
            "heartbeat_timeout_seconds",
            "automatic_recovery",
        ):
            engine.pop(key, None)
    agent = normalized.get("agent")
    if isinstance(agent, dict):
        for key in (
            "max_retries",
            "heartbeat_timeout_seconds",
            "sandbox_mode",
            "approval_policy",
        ):
            agent.pop(key, None)
    bootstrap = normalized.get("bootstrap")
    if isinstance(bootstrap, dict):
        reference = bootstrap.get("reference")
        if isinstance(reference, dict):
            reference.pop("enabled", None)
    return normalized
