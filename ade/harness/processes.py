"""Long-running process loops for production Control and Engine entrypoints."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import re
import threading
import time
from typing import Protocol

from ade.core.bootstrap import BootstrapStatus
from ade.controller.state import StateCoordinator
from ade.core.outcomes import LocalJudgeReplacedOutcome, RunResumedOutcome
from ade.core.reconcile import ReconcileResult, Waiting
from ade.core.run import RunState, RunStatus
from ade.core.trial import TrialArchiveStatus, TrialKind, TrialOutcome, TrialPhase
from ade.engine.worker import EngineWorker
from ade.harness.experiment_config import (
    ExperimentConfigCompiler,
    ResolvedExperimentConfig,
)
from ade.harness.wiring import (
    AgentProcess,
    ControlProcess,
    build_agent_process,
    build_control_process,
    build_engine_process,
    build_review_process,
)
from ade.memory.repository import FileRunRepository, RunNotFoundError
from ade.local_rubric_judge.lifecycle import LocalJudgeBinding, RunResourceAdmission
from ade.local_rubric_judge.launcher import ProductionJudgeLauncher
from ade.tasks.registry import default_task_registry

_TERMINAL_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
}
_CONTROL_STOP_STATUSES = {
    *_TERMINAL_STATUSES,
    RunStatus.PAUSED,
    RunStatus.SUSPENDED,
}


class StateRepository(Protocol):
    def load(self, run_id: str) -> RunState: ...

    def materialize_results(self, state: RunState) -> None: ...


class Workflow(Protocol):
    def reconcile_once(self, run_id: str) -> ReconcileResult: ...


class ControlProcessRunner:
    def __init__(
        self,
        *,
        workflow: Workflow,
        repository: StateRepository,
        poll_interval: float = 10.0,
        wait: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.workflow = workflow
        self.repository = repository
        self.poll_interval = poll_interval
        self.wait = wait

    def run(self, run_id: str) -> RunState:
        state = self.repository.load(run_id)
        while not _control_should_stop(state):
            result = self.workflow.reconcile_once(run_id)
            state = self.repository.load(run_id)
            if isinstance(result, Waiting) and not _control_should_stop(state):
                self.wait(self.poll_interval)
        return state


class EngineProcessRunner:
    def __init__(
        self,
        *,
        worker: EngineWorker,
        poll_interval: float = 10.0,
        wait: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.worker = worker
        self.poll_interval = poll_interval
        self.wait = wait

    def run(
        self,
        *,
        stop_requested: Callable[[], bool] = lambda: False,
    ) -> tuple[str, ...]:
        coordinator_id = getattr(self.worker, "coordinator_id", None)

        def expire_stale():
            if coordinator_id is None:
                return self.worker.queue.expire_stale()
            return self.worker.queue.expire_stale(coordinator_id=coordinator_id)

        recovered = list(expire_stale())
        while not stop_requested():
            recovered.extend(expire_stale())
            result: list[object | None] = []
            errors: list[BaseException] = []

            def execute_once() -> None:
                try:
                    result.append(self.worker.run_once())
                except BaseException as error:
                    errors.append(error)

            execution = threading.Thread(
                target=execute_once,
                name="ade-engine-command-execution",
                daemon=True,
            )
            execution.start()
            execution.join(0)
            while execution.is_alive() and not stop_requested():
                expired = expire_stale()
                recovered.extend(expired)
                active_command_id = getattr(self.worker, "active_command_id", None)
                if expired and active_command_id in expired:
                    return tuple(dict.fromkeys(recovered))
                self.wait(self.poll_interval)
            execution.join(0)
            if errors:
                raise errors[0]
            if not result or result[0] is None:
                self.wait(self.poll_interval)
        return tuple(dict.fromkeys(recovered))


class ReviewProcessRunner:
    def __init__(self, *, worker, poll_interval: float = 10.0, wait=time.sleep) -> None:
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.worker = worker
        self.poll_interval = poll_interval
        self.wait = wait

    def run(self, *, stop_requested: Callable[[], bool] = lambda: False) -> None:
        while not stop_requested():
            if self.worker.run_once() is None:
                self.wait(self.poll_interval)


class AgentProcessRunner:
    def __init__(
        self,
        *,
        process: AgentProcess,
        poll_interval: float = 10.0,
        wait: Callable[[float], None] = time.sleep,
        coordinator_id: str | None = None,
        global_only: bool = False,
    ) -> None:
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.process = process
        self.poll_interval = poll_interval
        self.wait = wait
        self.coordinator_id = coordinator_id
        self.global_only = global_only
        if coordinator_id is not None and global_only:
            raise ValueError("coordinator_id and global_only are mutually exclusive")

    def run(self, run_id: str) -> RunState:
        while True:
            state = self.process.repository.load(run_id)
            if state.status in _CONTROL_STOP_STATUSES - {RunStatus.SUSPENDED}:
                return state
            if self.run_once(run_id) is None:
                if state.status is RunStatus.SUSPENDED:
                    return state
                self.wait(self.poll_interval)

    def run_once(self, run_id: str) -> str | None:
        state = self.process.repository.load(run_id)
        pending = tuple(
            call
            for call in state.active_agent_calls
            if getattr(call, "status", "submitted") != "retry_pending"
            if (
                (self.global_only and call.coordinator_id is None)
                or (
                    not self.global_only
                    and (
                        self.coordinator_id is None
                        or call.coordinator_id == self.coordinator_id
                    )
                )
            )
            if not self.process.calls.has_terminal(call)
        )
        if not pending:
            return None
        active = pending[0]
        try:
            self.process.calls.dispatch_active(active)
        except Exception as error:
            self.process.calls.record_dispatch_failure(active, error)
        return active.attempt_id


def run_control_process(
    *,
    experiment_config: str | Path,
    deployment_config: str | Path,
    run_id: str,
    project_root: str | Path,
    runs_root: str | Path,
    queue_root: str | Path,
    object_root: str | Path,
    poll_interval: float = 10.0,
    resume: bool = False,
) -> RunState:
    project = Path(project_root).resolve()
    config_path = Path(experiment_config)
    if not config_path.is_absolute():
        config_path = project / config_path
    experiment = ExperimentConfigCompiler(
        default_task_registry(),
        project_root=project,
    ).compile_file(
        config_path,
        deployment_config=deployment_config,
        run_id=run_id,
    )
    process = build_control_process(
        project_root=project,
        runs_root=runs_root,
        queue_root=queue_root,
        object_root=object_root,
        engine_inputs=experiment.engine_inputs,
        bootstrap_contract=experiment.bootstrap,
        agent_config=experiment.control["agent"],
        artifact_builder_config=experiment.control["artifact_builder"],
        engine_config=experiment.control["engine"],
        run_resource_admission=_run_resource_admission(experiment, runs_root, project),
    )
    if resume:
        _refuse_live_suspended_resume(process.repository, run_id, runs_root, queue_root)
    _create_or_resume(process, experiment, resume=resume)
    return ControlProcessRunner(
        workflow=process.workflow,
        repository=process.repository,
        poll_interval=poll_interval,
    ).run(run_id)


def _refuse_live_suspended_resume(repository, run_id, runs_root, queue_root):
    """Direct Control resume cannot bypass the Supervisor drain boundary."""
    import os
    from ade.harness.supervisor import RunSupervisor

    if repository.load(run_id).status is not RunStatus.SUSPENDED:
        return
    queue = Path(queue_root).resolve()
    roots = tuple(str(path) for path in (
        Path(runs_root).resolve(), queue, queue.parent / "objects",
        queue.parent / "engine-work", queue.parent / "review",
        queue.parent / "review-work",
    ))
    live = RunSupervisor._deployment_worker_pids(roots)
    if any(pid != os.getpid() for pid in live) or RunSupervisor._has_processing_engine_work(queue, run_id):
        raise RuntimeError(
            "suspended Run still has live deployment processes or unfinished Engine "
            "claims; wait for drain or explicitly fence orphaned work before resume"
        )


def initialize_control_run(
    *,
    experiment: ResolvedExperimentConfig,
    project_root: str | Path,
    runs_root: str | Path,
    queue_root: str | Path,
    object_root: str | Path,
    resume: bool = False,
    initial_state_reference: str | None = None,
    initial_state_frontier: tuple[str, ...] = (),
) -> RunState:
    """Create or resume a Run before a supervisor creates Run-local logs."""
    process = build_control_process(
        project_root=project_root,
        runs_root=runs_root,
        queue_root=queue_root,
        object_root=object_root,
        engine_inputs=experiment.engine_inputs,
        bootstrap_contract=experiment.bootstrap,
        agent_config=experiment.control["agent"],
        artifact_builder_config=experiment.control["artifact_builder"],
        engine_config=experiment.control["engine"],
        run_resource_admission=_run_resource_admission(
            experiment, runs_root, project_root
        ),
    )
    return _create_or_resume(
        process,
        experiment,
        resume=resume,
        initial_state_reference=initial_state_reference,
        initial_state_frontier=initial_state_frontier,
    )


def run_control_step(
    *,
    experiment_config: str | Path,
    deployment_config: str | Path,
    run_id: str,
    project_root: str | Path,
    runs_root: str | Path,
    queue_root: str | Path,
    object_root: str | Path,
    resume: bool = False,
) -> tuple[RunState, ReconcileResult]:
    project = Path(project_root).resolve()
    config_path = Path(experiment_config)
    if not config_path.is_absolute():
        config_path = project / config_path
    experiment = ExperimentConfigCompiler(
        default_task_registry(), project_root=project
    ).compile_file(
        config_path,
        deployment_config=deployment_config,
        run_id=run_id,
    )
    process = build_control_process(
        project_root=project,
        runs_root=runs_root,
        queue_root=queue_root,
        object_root=object_root,
        engine_inputs=experiment.engine_inputs,
        bootstrap_contract=experiment.bootstrap,
        agent_config=experiment.control["agent"],
        artifact_builder_config=experiment.control["artifact_builder"],
        engine_config=experiment.control["engine"],
        run_resource_admission=_run_resource_admission(experiment, runs_root, project),
    )
    if resume:
        _refuse_live_suspended_resume(process.repository, run_id, runs_root, queue_root)
    _create_or_resume(process, experiment, resume=resume)
    result = process.workflow.reconcile_once(run_id)
    return process.repository.load(run_id), result


def run_agent_process(
    *,
    experiment_config: str | Path,
    deployment_config: str | Path,
    run_id: str,
    project_root: str | Path,
    runs_root: str | Path,
    poll_interval: float = 10.0,
    coordinator_id: str | None = None,
    global_only: bool = False,
) -> RunState:
    project = Path(project_root).resolve()
    config_path = Path(experiment_config)
    if not config_path.is_absolute():
        config_path = project / config_path
    experiment = ExperimentConfigCompiler(
        default_task_registry(),
        project_root=project,
    ).compile_file(
        config_path,
        deployment_config=deployment_config,
        run_id=run_id,
    )
    process = build_agent_process(
        project_root=project,
        runs_root=runs_root,
        agent_config=experiment.control["agent"],
        artifact_builder_config=experiment.control["artifact_builder"],
    )
    while True:
        try:
            state = process.repository.load(run_id)
            break
        except RunNotFoundError:
            time.sleep(poll_interval)
    expected = process.repository.describe_artifact(
        run_id,
        "resolved_experiment_config",
        experiment.encode(),
    )
    if state.task.config_ref != expected.artifact_id:
        raise ValueError("existing run uses a different resolved experiment config")
    return AgentProcessRunner(
        process=process,
        poll_interval=poll_interval,
        coordinator_id=coordinator_id,
        global_only=global_only,
    ).run(run_id)


def run_agent_one(
    *,
    experiment_config: str | Path,
    deployment_config: str | Path,
    run_id: str,
    project_root: str | Path,
    runs_root: str | Path,
) -> tuple[RunState, str | None]:
    project = Path(project_root).resolve()
    config_path = Path(experiment_config)
    if not config_path.is_absolute():
        config_path = project / config_path
    experiment = ExperimentConfigCompiler(
        default_task_registry(), project_root=project
    ).compile_file(
        config_path,
        deployment_config=deployment_config,
        run_id=run_id,
    )
    process = build_agent_process(
        project_root=project,
        runs_root=runs_root,
        agent_config=experiment.control["agent"],
        artifact_builder_config=experiment.control["artifact_builder"],
    )
    state = process.repository.load(run_id)
    expected = process.repository.describe_artifact(
        run_id, "resolved_experiment_config", experiment.encode()
    )
    if state.task.config_ref != expected.artifact_id:
        raise ValueError("existing run uses a different resolved experiment config")
    attempt_id = AgentProcessRunner(process=process).run_once(run_id)
    return process.repository.load(run_id), attempt_id


def revalidate_agent_one(
    *,
    experiment_config: str | Path,
    deployment_config: str | Path,
    run_id: str,
    project_root: str | Path,
    runs_root: str | Path,
) -> tuple[RunState, str]:
    project = Path(project_root).resolve()
    config_path = Path(experiment_config)
    if not config_path.is_absolute():
        config_path = project / config_path
    experiment = ExperimentConfigCompiler(
        default_task_registry(), project_root=project
    ).compile_file(
        config_path,
        deployment_config=deployment_config,
        run_id=run_id,
    )
    process = build_agent_process(
        project_root=project,
        runs_root=runs_root,
        agent_config=experiment.control["agent"],
        artifact_builder_config=experiment.control["artifact_builder"],
    )
    state = process.repository.load(run_id)
    expected = process.repository.describe_artifact(
        run_id, "resolved_experiment_config", experiment.encode()
    )
    if state.task.config_ref != expected.artifact_id:
        raise ValueError("existing run uses a different resolved experiment config")
    if not state.active_agent_calls:
        raise ValueError("run has no active Agent Call to revalidate")
    active = state.active_agent_calls[0]
    result = process.calls.revalidate_active(active)
    return process.repository.load(run_id), result.attempt.attempt_id


def run_engine_process(
    *,
    queue_root: str | Path,
    object_root: str | Path,
    work_root: str | Path,
    poll_interval: float = 10.0,
    claim_timeout_seconds: float = 1800.0,
    heartbeat_timeout_seconds: float = 1800.0,
    coordinator_id: str | None = None,
) -> None:
    import ray

    ray.init(address=os.environ["RAY_ADDRESS"], namespace="ade")
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    process = build_engine_process(
        queue_root=queue_root,
        object_root=object_root,
        work_root=work_root,
        engine_config={
            "claim_timeout_seconds": claim_timeout_seconds,
            "heartbeat_timeout_seconds": heartbeat_timeout_seconds,
        },
        coordinator_id=coordinator_id,
    )
    EngineProcessRunner(
        worker=process.worker,
        poll_interval=poll_interval,
    ).run()


def run_engine_one(
    *,
    queue_root: str | Path,
    object_root: str | Path,
    work_root: str | Path,
    claim_timeout_seconds: float,
    heartbeat_timeout_seconds: float,
) -> str | None:
    import ray

    ray.init(address=os.environ["RAY_ADDRESS"], namespace="ade")
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    process = build_engine_process(
        queue_root=queue_root,
        object_root=object_root,
        work_root=work_root,
        engine_config={
            "claim_timeout_seconds": claim_timeout_seconds,
            "heartbeat_timeout_seconds": heartbeat_timeout_seconds,
        },
    )
    receipt = process.worker.run_once()
    return receipt.command_id if receipt is not None else None


def run_review_process(
    *,
    experiment_config: str | Path,
    deployment_config: str | Path,
    run_id: str,
    project_root: str | Path,
    runs_root: str | Path,
    queue_root: str | Path,
    work_root: str | Path,
    poll_interval: float = 10.0,
    coordinator_id: str | None = None,
) -> None:
    process = _review_process(
        experiment_config=experiment_config,
        deployment_config=deployment_config,
        run_id=run_id,
        project_root=project_root,
        runs_root=runs_root,
        queue_root=queue_root,
        work_root=work_root,
        coordinator_id=coordinator_id,
    )
    try:
        ReviewProcessRunner(worker=process.worker, poll_interval=poll_interval).run()
    finally:
        process.close()


def run_review_one(
    *,
    experiment_config: str | Path,
    deployment_config: str | Path,
    run_id: str,
    project_root: str | Path,
    runs_root: str | Path,
    queue_root: str | Path,
    work_root: str | Path,
) -> str | None:
    process = _review_process(
        experiment_config=experiment_config,
        deployment_config=deployment_config,
        run_id=run_id,
        project_root=project_root,
        runs_root=runs_root,
        queue_root=queue_root,
        work_root=work_root,
    )
    try:
        receipt = process.worker.run_once()
        return receipt.command_id if receipt is not None else None
    finally:
        process.close()


def _review_process(
    *,
    experiment_config: str | Path,
    deployment_config: str | Path,
    run_id: str,
    project_root: str | Path,
    runs_root: str | Path,
    queue_root: str | Path,
    work_root: str | Path,
    coordinator_id: str | None = None,
):
    project = Path(project_root).resolve()
    config_path = Path(experiment_config)
    if not config_path.is_absolute():
        config_path = project / config_path
    experiment = ExperimentConfigCompiler(
        default_task_registry(), project_root=project
    ).compile_file(
        config_path,
        deployment_config=deployment_config,
        run_id=run_id,
    )
    analysis = experiment.resolved.get("analysis")
    if not isinstance(analysis, dict) or not isinstance(analysis.get("provider"), dict):
        raise ValueError("resolved experiment has no Review provider")
    state = FileRunRepository(runs_root).load(run_id)
    if state.run_resources is None:
        raise ValueError("Run has no Local Analyzer resource")
    local_judge = state.run_resources["local_judge"]
    return build_review_process(
        project_root=project,
        queue_root=queue_root,
        work_root=work_root,
        provider_config=analysis["provider"],
        local_judge_config=local_judge,
        coordinator_id=coordinator_id,
    )


def _create_or_resume(
    process: ControlProcess,
    experiment: ResolvedExperimentConfig,
    *,
    resume: bool = False,
    initial_state_reference: str | None = None,
    initial_state_frontier: tuple[str, ...] = (),
) -> RunState:
    run_id = experiment.run.run_id
    if resume and (initial_state_reference or initial_state_frontier):
        raise ValueError("initial-state arguments are only valid for a new Run")
    try:
        state = process.repository.load(run_id)
    except RunNotFoundError:
        if resume:
            raise ValueError("cannot resume a Run that does not exist")
        return process.harness.create_experiment(
            experiment,
            initial_state_reference=initial_state_reference,
            initial_state_frontier=initial_state_frontier,
        )
    if not resume:
        raise ValueError("Run already exists; pass --resume to continue")
    expected = process.repository.describe_artifact(
        run_id,
        "resolved_experiment_config",
        experiment.encode(),
    )
    if state.task.config_ref != expected.artifact_id and not (
        state.status in {RunStatus.SUSPENDED, RunStatus.RUNNING}
        and _only_recovery_budget_changed(
            process.repository,
            run_id,
            state.task.config_ref,
            experiment.encode(),
        )
    ):
        raise ValueError("existing run uses a different resolved experiment config")
    process.repository.materialize_results(state)
    state = _admit_resumed_run_resources(process, state, experiment)
    failed_engine_trial = any(
        trial.kind is TrialKind.SEARCH
        and trial.outcome is TrialOutcome.FAILED
        and trial.failure_kind == "engine_failed"
        and trial.artifact_ref_id is not None
        and trial.phase is TrialPhase.ARCHIVED
        and trial.archive_status is TrialArchiveStatus.ARCHIVED
        for trial in state.trials
    )
    if (
        state.status in {RunStatus.PAUSED, RunStatus.SUSPENDED}
        or state.pause_requested
        or (state.status is RunStatus.COMPLETED and failed_engine_trial)
    ):
        return StateCoordinator(process.repository).apply(
            run_id,
            RunResumedOutcome(
                run_id=run_id,
                basis_revision=state.revision,
                automatic_recovery_max_attempts=int(
                    experiment.control["engine"]["automatic_recovery"]["max_attempts"]
                ),
            ),
            event_type="run_resumed",
        )
    return state


def _only_recovery_budget_changed(
    repository, run_id: str, config_ref: str, current_content: bytes
) -> bool:
    """Permit a suspended Run to receive only a larger recovery budget."""
    candidates = repository.layout.run_dir(run_id).joinpath(
        "artifacts", "resolved_experiment_config"
    ).glob("*")
    previous_content = next(
        (
            path.read_bytes()
            for path in candidates
            if f"resolved_experiment_config-{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
            == config_ref
        ),
        None,
    )
    if previous_content is None:
        return False
    try:
        previous = json.loads(previous_content)
        current = json.loads(current_content)
        previous_budget = previous["engine"]["automatic_recovery"][
            "max_attempts"
        ]
        current_budget = current["engine"]["automatic_recovery"][
            "max_attempts"
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    previous["engine"]["automatic_recovery"]["max_attempts"] = current_budget
    # This digest is derived from the resolved config itself and therefore
    # changes when the recovery budget changes.
    previous["config_digest"] = current.get("config_digest")
    previous["source_manifest"] = current.get("source_manifest")
    return current_budget >= previous_budget and previous == current


def _admit_resumed_run_resources(
    process: ControlProcess,
    state: RunState,
    experiment: ResolvedExperimentConfig,
) -> RunState:
    if state.run_resources is None:
        return state
    resources = experiment.run.run_resources
    if resources is None:
        raise ValueError("persisted GPU Run lacks resolved resource configuration")
    admission = getattr(process.repository, "run_resource_admission", None)
    if admission is None:
        raise ValueError("GPU-capable Run requires a Run resource admission service")
    ray = resources["ray_cluster"]
    judge = resources["local_judge"]
    supersedes_run_id = None
    if state.forked_from is not None:
        source = process.repository.load(state.forked_from.source_run_id)
        if source.status not in {
            RunStatus.PAUSED,
            RunStatus.SUSPENDED,
            RunStatus.CANCELLED,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
        }:
            raise ValueError(
                "fork resource handoff requires the source Run at a safe boundary"
            )
        supersedes_run_id = source.run_id
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

        supersedes_run_id=supersedes_run_id,
    )
    persisted = state.run_resources
    if persisted is None:
        raise ValueError("resumed GPU Run lacks persisted resource binding")
    expected_judge = persisted["local_judge"]
    if ready.state_path != expected_judge.get("state_path"):
        admission.terminal(state.run_id)
        raise ValueError(
            "resumed Local Judge state binding does not match persisted Run state"
        )
    if ready.launch_id == expected_judge.get("launch_id"):
        return state
    return StateCoordinator(process.repository).apply(
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
    )


def _run_resource_admission(
    experiment: ResolvedExperimentConfig,
    runs_root: str | Path,
    project_root: str | Path,
) -> RunResourceAdmission | None:
    resources = experiment.run.run_resources
    if resources is None:
        return None
    return run_resource_admission_from_resources(
        resources,
        runs_root=runs_root,
        project_root=project_root,
    )


def run_resource_admission_from_resources(
    resources: dict[str, Any],
    *,
    runs_root: str | Path,
    project_root: str | Path,
) -> RunResourceAdmission:
    """Rebuild deployment admission from a Run's persisted resource facts."""
    judge = resources["local_judge"]
    cluster_id = str(resources["ray_cluster"]["cluster_id"])
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", cluster_id) is None:
        raise ValueError("deployment cluster_id is not a safe runtime scope")
    shared_root = Path(runs_root).resolve().parent
    deployment_root = (
        shared_root
        if shared_root.name == cluster_id
        else shared_root / "deployments" / cluster_id
    )
    launcher = ProductionJudgeLauncher(
        deployment_root / "local-judge",
        authorization_env=str(judge["authorization_env"]),
        readiness_seconds=float(judge["timeout_policy"]["readiness_seconds"]),
        request_timeout_seconds=float(judge["timeout_policy"]["request_timeout_seconds"]),
        vllm=dict(judge["vllm"]),
        generation=dict(judge["generation"]),
    )
    return RunResourceAdmission(deployment_root / "run-services", launcher)


def control_stop_reason(state: RunState) -> str:
    if state.status in _CONTROL_STOP_STATUSES:
        return state.status.value
    if state.bootstrap.status is BootstrapStatus.FAILED:
        return "bootstrap-failed"
    raise ValueError("Control returned before reaching a stop condition")


def _control_should_stop(state: RunState) -> bool:
    if state.status in _CONTROL_STOP_STATUSES:
        return True
    if state.bootstrap.status is BootstrapStatus.FAILED:
        return True
    return False
