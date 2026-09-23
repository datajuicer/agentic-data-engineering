"""Operator-facing Harness operations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ade.controller.state import StateCoordinator
from ade.core.coordinator_control import allocated_plan_slots, effective_max_plans
from ade.controller.run_seed import RunSeedImporter, RunSeedRequest
from ade.core.bootstrap import BootstrapState, BootstrapStatus
from ade.core.coordinator import CoordinatorKind, CoordinatorState
from ade.core.insight import InsightGraph
from ade.core.plan import PlanKind, PlanState, PlanStatus
from ade.core.ranking import RankingState
from ade.core.run import AutomaticRecoveryPolicy, MemoryState, RunState, RunStatus
from ade.harness.config import ConfigCompiler
from ade.harness.experiment_config import ResolvedExperimentConfig
from ade.memory.repository import FileRunRepository
from ade.local_rubric_judge.lifecycle import LocalJudgeBinding
from ade.engine.storage.atomic import write_json_atomic
from ade.engine.storage.object_store import FileEngineObjectStore
from ade.review_labor.command_queue import FileReviewCommandQueue
from ade.harness.run_cleanup import cleanup_cancelled_run


class Harness:
    def __init__(
        self,
        repository: FileRunRepository,
        compiler: ConfigCompiler,
        engine_objects: FileEngineObjectStore | None = None,
        run_seed_source_resolver=None,
    ) -> None:
        self.repository = repository
        self.compiler = compiler
        self.engine_objects = engine_objects
        self.run_seed_source_resolver = run_seed_source_resolver
        self.state = StateCoordinator(repository)

    def create(self, payload: dict[str, Any]) -> RunState:
        return self._create(self.compiler.compile(payload))

    def create_experiment(
        self,
        experiment: ResolvedExperimentConfig,
        *,
        initial_state_reference: str | None = None,
        initial_state_frontier: tuple[str, ...] = (),
    ) -> RunState:
        content = experiment.encode()
        ref = self.repository.describe_artifact(
            experiment.run.run_id,
            "resolved_experiment_config",
            content,
        )
        return self._create(
            experiment.run,
            config_ref_id=ref.artifact_id,
            extra_artifacts=((ref, content),),
            bootstrap=experiment.bootstrap,
            config_publication=experiment.publication_files(),
            coordinator_count=int(
                experiment.control["search"]["coordinator_count"]
            ),
            automatic_recovery=dict(
                experiment.control["engine"]["automatic_recovery"]
            ),
            initial_state_reference=initial_state_reference,
            initial_state_frontier=initial_state_frontier,
            resolved=experiment.resolved,
        )

    def _create(
        self,
        config,
        *,
        config_ref_id: str | None = None,
        extra_artifacts=(),
        bootstrap: dict[str, object] | None = None,
        config_publication: tuple[tuple[str, bytes], ...] = (),
        coordinator_count: int = 1,
        automatic_recovery: dict[str, object] | None = None,
        initial_state_reference: str | None = None,
        initial_state_frontier: tuple[str, ...] = (),
        resolved: dict[str, object] | None = None,
    ) -> RunState:
        if coordinator_count < 1:
            raise ValueError("coordinator_count must be positive")
        task = replace(config.task, config_ref=config_ref_id)
        bootstrap_state = BootstrapState.from_dict(
            bootstrap or {"enabled": False, "status": "completed"}
        )
        reference_enabled = bool(bootstrap_state.reference_enabled)
        if reference_enabled and not initial_state_reference:
            raise ValueError(
                "bootstrap.reference.enabled requires --initial-state-reference"
            )
        if reference_enabled and not initial_state_frontier:
            raise ValueError(
                "bootstrap.reference.enabled requires --initial-state-frontier"
            )
        if not reference_enabled and (
            initial_state_reference or initial_state_frontier
        ):
            raise ValueError(
                "initial-state arguments require bootstrap.reference.enabled=true"
            )
        if initial_state_reference == config.run_id:
            raise ValueError("Run Seed source and target Run IDs must differ")
        seed_resolution = None
        source_repository = self.repository
        source_objects = self.engine_objects
        source_deployment_id = None
        if (
            initial_state_reference
            and self.run_seed_source_resolver is not None
        ):
            source_repository, source_objects, source_deployment_id = (
                self.run_seed_source_resolver(initial_state_reference)
            )
        seed_importer = RunSeedImporter(
            self.repository,
            self.engine_objects,
            source_repository=source_repository,
            source_objects=source_objects,
            source_deployment_id=source_deployment_id,
        )
        if initial_state_reference:
            seed_resolution = seed_importer.resolve(
                RunSeedRequest.parse(
                    initial_state_reference, initial_state_frontier
                ),
                target_resolved=resolved or {},
            )
            seed_importer.inspect(
                seed_resolution,
                target_plan_budget=config.portfolio.max_plans,
            )
        if coordinator_count < 1 or config.portfolio.max_plans % coordinator_count:
            raise ValueError("Run Plan budget must divide across Search Coordinators")
        plans_per_coordinator = config.portfolio.max_plans // coordinator_count
        coordinators = (
            (
                CoordinatorState("c000", CoordinatorKind.BOOTSTRAP, 0, 0),
                *(
                    CoordinatorState(
                        f"c{index:03d}",
                        CoordinatorKind.SEARCH,
                        plans_per_coordinator,
                        plans_per_coordinator,
                    )
                    for index in range(1, coordinator_count + 1)
                ),
            )
            if bootstrap_state.enabled
            else tuple(
                CoordinatorState(
                    f"c{index:03d}",
                    CoordinatorKind.SEARCH,
                    plans_per_coordinator,
                    plans_per_coordinator,
                )
                for index in range(1, coordinator_count + 1)
            )
        )
        plans = (
            (
                PlanState(
                    plan_id="p000",
                    coordinator_id="c000",
                    kind=PlanKind.BOOTSTRAP,
                    status=PlanStatus.ACTIVE,
                    plan_memory_head=f"{config.run_id}/c000/p000/PM000",
                ),
            )
            if bootstrap_state.enabled
            else ()
        )
        resolved_resources = None
        admission = getattr(self.repository, "run_resource_admission", None)
        if config.run_resources is not None:
            if admission is None:
                raise ValueError("GPU-capable Run requires a Run resource admission service")
            ray = config.run_resources["ray_cluster"]
            judge = config.run_resources["local_judge"]
            binding = LocalJudgeBinding.from_dict(
                {
                    "cluster_id": ray["cluster_id"],
                    "ray_address": ray["address"],
                    "gpu_count": judge["gpu_count"],
                    "model_path": judge["model_path"],
                    "model_digest": judge["model_digest"],
                    "gateway_port": judge["gateway_port"],
                    "protocol": judge["protocol"],
                }
            )
            ready = admission.admit(
                run_id=config.run_id,
                binding=binding,
            )
            resolved_resources = {
                **config.run_resources,
                "local_judge": {
                    **judge,
                    "launch_id": ready.launch_id,
                    "host": ready.host,
                    "gateway_url": ready.gateway_url,
                    "node_id": ready.node_id,
                    "service_state": ready.status,
                    "readiness": True,
                    "state_path": ready.state_path,
                },
            }
        try:
            initial_state = RunState(
                    run_id=config.run_id,
                    revision=0,
                    status=(
                        RunStatus.BOOTSTRAPPING
                        if bootstrap_state.enabled
                        else RunStatus.RUNNING
                    ),
                    task=task,
                    portfolio=config.portfolio,
                    insight_graph=InsightGraph(revision=0),
                    ranking=RankingState(metric_id="offline.ranking_score"),
                    coordinators=coordinators,
                    plans=plans,
                    memory=MemoryState(
                        "collection/none",
                        f"{config.run_id}/RM000",
                        {
                            f"{config.run_id}/c000/p000":
                            f"{config.run_id}/c000/p000/PM000"
                        }
                        if bootstrap_state.enabled
                        else {},
                    ),
                    bootstrap=bootstrap_state,
                    analysis_policy=config.analysis_policy,
                    run_resources=resolved_resources,
                    automatic_recovery=AutomaticRecoveryPolicy(
                        **dict(automatic_recovery or {})
                    ),
                )
            initial_artifacts = tuple(extra_artifacts)
            inherited_files = ()
            if seed_resolution is not None:
                initial_state, imported_artifacts, inherited_files = (
                    seed_importer.materialize(seed_resolution, initial_state)
                )
                initial_artifacts = initial_artifacts + tuple(imported_artifacts)
            created = self.repository.create(
                initial_state,
                initial_artifacts=initial_artifacts,
                initial_config_files=config_publication,
                inherited_files=inherited_files,
                initialize_memory=(seed_resolution is None or initial_state.bootstrap.status is BootstrapStatus.P000_ANALYZING),
            )
            if seed_resolution is not None:
                reference_source = seed_importer.project(seed_resolution)
                target_tracking = dict((resolved or {}).get("tracking", {}))
                tracking_results = seed_importer.publish_tracking(
                    reference_source,
                    created,
                    target_tracking=target_tracking,
                )
                write_json_atomic(
                    self.repository.layout.run_dir(created.run_id)
                    / "tracking"
                    / "seed-import.json",
                    {
                        "source_run_id": reference_source.run_id,
                        "source_revision": reference_source.revision,
                        **(
                            {"source_deployment_id": source_deployment_id}
                            if source_deployment_id is not None
                            else {}
                        ),
                        "boundary_kind": (
                            seed_resolution.request.boundary_kind.value
                        ),
                        "requested_frontier": dict(
                            seed_resolution.request.frontier
                        ),
                        "resolved_frontier": dict(
                            seed_resolution.resolved_frontier
                        ),
                        "evaluation_results": list(tracking_results),
                        "wandb_history_status": (
                            "pending_retry"
                            if bool(target_tracking.get("enabled"))
                            and target_tracking.get("mode") == "online"
                            else "disabled"
                        ),
                        "wandb_history_runs": [],
                    },
                )
            return created
        except Exception:
            if resolved_resources is not None:
                admission.terminal(config.run_id)
            raise

    def status(self, run_id: str) -> dict[str, object]:
        state = self.repository.load(run_id)
        payload = {
            "run_id": state.run_id,
            "revision": state.revision,
            "status": state.status.value,
            "task_id": state.task.task_id,
            "completion_kind": (
                state.completion_kind.value
                if state.completion_kind is not None
                else None
            ),
            "plan_budget": {
                "original": state.portfolio.max_plans,
                "requested_per_coordinator": (
                    state.finish_request.requested_plan_limit
                    if state.finish_request is not None
                    else None
                ),
                "effective": effective_max_plans(state),
                "actual": sum(
                    plan.kind is PlanKind.SEARCH for plan in state.plans
                ),
            },
            "coordinators": [
                {
                    "coordinator_id": item.coordinator_id,
                    "kind": item.kind.value,
                    "control_status": item.control_status.value,
                    "original_plan_limit": item.original_plan_limit,
                    "requested_plan_limit": item.requested_plan_limit,
                    "effective_plan_limit": item.effective_plan_limit,
                    "allocated_plan_slots": allocated_plan_slots(
                        state, item.coordinator_id
                    ),
                    "control_reason": item.control_reason,
                    "control_requested_revision": (
                        item.control_requested_revision
                    ),
                }
                for item in state.coordinators
            ],
        }
        review_progress = {}
        review_queue = FileReviewCommandQueue(
            self.repository.runs_root.parent / "queue" / "review"
        )
        for trial in state.trials:
            command_id = trial.analysis_review_command_id
            if trial.phase.value != "review_running" or not command_id:
                continue
            try:
                review_progress[command_id] = review_queue.load_progress(command_id)
            except (FileNotFoundError, ValueError, TypeError):
                continue
        if review_progress:
            payload["analysis_review_progress"] = review_progress
        return payload

    def inspect(self, run_id: str) -> dict[str, object]:
        return self.repository.load(run_id).to_dict()

    def cancel(self, run_id: str, *, reason: str) -> RunState:
        current = self.repository.load(run_id)
        admission = getattr(self.repository, "run_resource_admission", None)
        updated = (
            current
            if current.status is RunStatus.CANCELLED
            else self.state.cancel(run_id, reason=reason)
        )
        if current.status is RunStatus.CANCELLED and admission is not None:
            admission.terminal(run_id)
        run_dir = self.repository.runs_root / run_id
        if (run_dir / "reports" / "worker-provenance.json").is_file():
            run_resources = updated.run_resources or {}
            ray_cluster = (
                run_resources.get("ray_cluster", {})
                if isinstance(run_resources, dict)
                else {}
            )
            artifact_staging = (
                run_resources.get("artifact_staging", {})
                if isinstance(run_resources, dict)
                else {}
            )
            cleanup_cancelled_run(
                run_id=run_id,
                run_dir=run_dir,
                queue_root=self.repository.runs_root.parent / "queue",
                review_root=self.repository.runs_root.parent / "review",
                ray_address=(
                    str(ray_cluster.get("address"))
                    if isinstance(ray_cluster, dict) and ray_cluster.get("address")
                    else None
                ),
                artifact_cache_dirs=(
                    (str(artifact_staging["cache_dir"]),)
                    if isinstance(artifact_staging, dict)
                    and artifact_staging.get("enabled")
                    and artifact_staging.get("cache_dir")
                    else ()
                ),
            )
        return updated

    def request_pause(self, run_id: str, *, reason: str) -> RunState:
        return self.state.request_pause(run_id, reason=reason)

    def request_coordinator_finish(
        self,
        run_id: str,
        *,
        coordinator_id: str,
        after_plans: int | None,
        reason: str,
    ) -> RunState:
        return self.state.request_coordinator_finish(
            run_id,
            coordinator_id=coordinator_id,
            requested_plan_limit=after_plans,
            reason=reason,
        )

    def request_run_finish(
        self,
        run_id: str,
        *,
        plans_per_coordinator: int,
        reason: str,
    ) -> RunState:
        return self.state.request_run_finish(
            run_id,
            requested_plan_limit=plans_per_coordinator,
            reason=reason,
        )
