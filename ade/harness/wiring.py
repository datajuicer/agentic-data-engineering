"""Production composition roots for the Control and Engine processes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import os

from ade.agent_runtime.backend import CodexBackend
from ade.agent_runtime.context import RoleContextPackager
from ade.agent_runtime.runtime import AgentRuntime
from ade.agent_runtime.service import AgentCallService
from ade.agent_runtime.skills import SkillResolver
from ade.agent_runtime.workspace import WorkspaceManager
from ade.core.dotenv import read_dotenv_value
from ade.controller.agent_port import CoordinatorAgentPort
from ade.controller.bootstrap import BootstrapWorkflowDriver
from ade.controller.run_seed import resolve_run_seed_source
from ade.controller.control import ControlLoop
from ade.controller.executor import Executor
from ade.controller.operator_evaluation import TrialOperatorEvaluationDriver
from ade.controller.planner import Planner
from ade.controller.reducer import Reducer
from ade.controller.trial_lifecycle import TrialLifecycle
from ade.controller.workflow_driver import WorkflowDriver
from ade.tasks.evaluation.backend import VllmEvaluationBackend
from ade.engine.backends.llamafactory import LlamaFactorySFTBackend
from ade.engine.backends.verl import VerlRFTBackend
from ade.engine.command_queue import FileCommandQueue
from ade.engine.evaluation_dispatcher import EngineEvaluationDispatcher
from ade.engine.judge_dispatcher import SelectionJudgeBatchDispatcher
from ade.tasks.evaluation.handler import EvaluationExecutor
from ade.tasks.reward_design.handler import RFTExecutor
from ade.tasks.data_selection.handler import SFTExecutor
from ade.engine.storage.object_store import FileEngineObjectStore
from ade.engine.trial_artifacts import TrialArtifactPublisher
from ade.engine.worker import EngineWorker
from ade.engine.execution.gpu import release_inactive_command_leases
from ade.harness.config import ConfigCompiler
from ade.harness.results_report import ResultsReportProjector
from ade.harness.service import Harness
from ade.memory.repository import FileRunRepository
from ade.tasks.registry import TaskRegistry, default_task_registry
from ade.local_rubric_judge import HttpRubricJobGateway, LocalRubricJudgeClient
from ade.review_labor import FileReviewCommandQueue
from ade.review_labor.local import LocalAnalyzerTransport
from ade.review_labor.service import ReviewLaborService
from ade.review_labor.analyzer_service import LocalAnalyzerReviewService
from ade.review_labor.worker import ReviewLaborBatchProcessor, ReviewWorker
from ade.tasks.reward_design.harness_judge import RewardHarnessJudge


@dataclass(frozen=True)
class EngineProcess:
    worker: EngineWorker
    queue: FileCommandQueue
    objects: FileEngineObjectStore


@dataclass(frozen=True)
class ControlProcess:
    harness: Harness
    trials: TrialLifecycle
    agents: AgentRuntime
    contexts: RoleContextPackager
    agent_calls: AgentCallService
    control: ControlLoop
    workflow: WorkflowDriver
    repository: FileRunRepository
    tasks: TaskRegistry


@dataclass(frozen=True)
class AgentProcess:
    calls: AgentCallService
    repository: FileRunRepository


@dataclass(frozen=True)
class ReviewProcess:
    worker: ReviewWorker
    queue: FileReviewCommandQueue
    close: Callable[[], None]


def _rft_training_progress_marker(
    runtime_root: Path,
    command,
) -> str | None:
    rollout_root = (
        runtime_root
        / command.command_id
        / "engine_audit"
        / "reward_rollouts_raw"
    )
    steps = [
        int(path.stem)
        for path in rollout_root.glob("*/*.jsonl")
        if path.stem.isdigit()
    ]
    return f"reward_rollout_step:{max(steps)}" if steps else None


def _evaluation_progress_marker(
    objects: FileEngineObjectStore,
    command,
) -> str | None:
    payload = objects.read_json(command.input_ref)
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        return None
    request = evaluation.get("request")
    if not isinstance(request, dict) or not request.get("run_dir"):
        return None
    log_root = Path(str(request["run_dir"])) / "logs" / "eval"
    files = [path for path in log_root.rglob("*") if path.is_file()]
    if not files:
        return None
    stats = [path.stat() for path in files]
    return (
        f"evaluation_logs:files={len(stats)}:"
        f"bytes={sum(item.st_size for item in stats)}:"
        f"mtime_ns={max(item.st_mtime_ns for item in stats)}"
    )


def build_engine_process(
    *,
    queue_root: str | Path,
    object_root: str | Path,
    work_root: str | Path,
    engine_config: Mapping[str, object] | None = None,
    coordinator_id: str | None = None,
) -> EngineProcess:
    resolved_engine = dict(engine_config or {})
    queue = FileCommandQueue(
        queue_root,
        claim_timeout_seconds=float(resolved_engine.get("claim_timeout_seconds", 1800)),
        heartbeat_timeout_seconds=float(
            resolved_engine.get("heartbeat_timeout_seconds", 1800)
        ),
    )
    # A detached allocator outlives a worker that died inside training or
    # evaluation.  Reclaim only leases whose exact command ID is no longer in
    # this queue's processing set; active commands remain untouched.
    if coordinator_id is not None:
        release_inactive_command_leases(set(queue.processing_command_ids()))
    objects = FileEngineObjectStore(object_root)
    work = Path(work_root).resolve()
    artifacts = TrialArtifactPublisher(work / "trial_artifacts")
    evaluation = EvaluationExecutor(io=objects, backend=VllmEvaluationBackend())
    evaluations = EngineEvaluationDispatcher(
        queue=queue,
        io=objects,
        execute=evaluation.execute,
        heartbeat_interval_seconds=float(
            resolved_engine.get("heartbeat_interval_seconds", 30.0)
        ),
    )
    sft = SFTExecutor(
        io=objects,
        backend=LlamaFactorySFTBackend(),
        evaluations=evaluations,
        artifacts=artifacts,
    )
    rft = RFTExecutor(
        io=objects,
        backend=VerlRFTBackend(work / "runtime"),
        evaluations=evaluations,
        artifacts=artifacts,
    )
    worker = EngineWorker(
        queue=queue,
        handlers={
            "evaluate": evaluation.execute,
            "train_sft": sft.execute,
            "train_rft": rft.execute,
        },
        coordinator_id=coordinator_id,
        progress_probes={
            "evaluate": lambda command: _evaluation_progress_marker(
                objects,
                command,
            ),
            "train_rft": lambda command: _rft_training_progress_marker(
                work / "runtime",
                command,
            )
        },
    )
    return EngineProcess(worker=worker, queue=queue, objects=objects)


def _build_agent_runtime(
    *,
    project_root: str | Path,
    runs_root: str | Path,
    agent_config: Mapping[str, object],
) -> AgentRuntime:
    """Use the same backend and workspace configuration in every Agent host."""
    resolved = dict(agent_config)
    backend_name = str(resolved.get("backend", "codex"))
    if backend_name != "codex":
        raise ValueError(f"unsupported Agent backend: {backend_name}")
    return AgentRuntime(
        skills=SkillResolver(Path(project_root).resolve() / ".agents" / "skills"),
        workspaces=WorkspaceManager.for_runs_root(runs_root),
        backend=CodexBackend(
            model=str(resolved.get("model", "gpt-5.6-sol")),
            reasoning_effort=str(resolved.get("reasoning_effort", "xhigh")),
            sandbox_mode=str(resolved.get("sandbox_mode", "danger-full-access")),
            approval_policy=str(resolved.get("approval_policy", "never")),
        ),
    )


def build_control_process(
    *,
    project_root: str | Path,
    runs_root: str | Path,
    queue_root: str | Path,
    object_root: str | Path,
    engine_inputs: Mapping[str, dict[str, object]],
    bootstrap_contract: Mapping[str, object] | None = None,
    agent_config: Mapping[str, object] | None = None,
    artifact_builder_config: Mapping[str, object] | None = None,
    engine_config: Mapping[str, object] | None = None,
    run_resource_admission=None,
    review_queue_root: str | Path | None = None,
) -> ControlProcess:
    project = Path(project_root).resolve()
    objects = FileEngineObjectStore(object_root)
    repository = FileRunRepository(
        runs_root,
        run_resource_admission=run_resource_admission,
        results_projector=ResultsReportProjector(objects),
    )
    tasks = default_task_registry()
    harness = Harness(
        repository,
        ConfigCompiler(tasks),
        engine_objects=objects,
        run_seed_source_resolver=lambda reference_run_id: resolve_run_seed_source(
            project, reference_run_id
        ),
    )
    resolved_agent = dict(agent_config or {})
    resolved_artifact_builder = dict(artifact_builder_config or {})
    agents = _build_agent_runtime(
        project_root=project, runs_root=runs_root, agent_config=resolved_agent
    )
    resolved_engine = dict(engine_config or {})
    queue = FileCommandQueue(
        queue_root,
        claim_timeout_seconds=float(resolved_engine.get("claim_timeout_seconds", 1800)),
        heartbeat_timeout_seconds=float(
            resolved_engine.get("heartbeat_timeout_seconds", 1800)
        ),
    )
    review_queue = FileReviewCommandQueue(
        review_queue_root
        if review_queue_root is not None
        else Path(queue_root).resolve().parent / "review"
    )
    trials = TrialLifecycle(
        repository=repository,
        tasks=tasks,
        queue=queue,
        engine_io=objects,
        reward_harness_judge_factory=_reward_harness_judge,
        selection_harness_judge_factory=_selection_harness_judge,
        review_queue=review_queue,
    )
    contexts = RoleContextPackager(repository=repository, tasks=tasks)
    max_agent_retries = int(resolved_agent.get("max_retries", 3))
    heartbeat_timeout_seconds = float(
        resolved_agent.get("heartbeat_timeout_seconds", 900)
    )
    agent_calls = AgentCallService(
        runtime=agents,
        contexts=contexts,
        tasks=tasks,
        max_retries=max_agent_retries,
        max_reflections=int(resolved_artifact_builder.get("max_reflections", 2)),
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        execution_mode="external",
    )

    control = ControlLoop(
        repository=repository,
        planner=Planner(),
        executor=Executor(
            agent_port=CoordinatorAgentPort(
                calls=agent_calls,
                repository=repository,
                tasks=tasks,
            )
        ),
        reducer=Reducer(),
    )
    operator_evaluation = _operator_evaluation_driver(
        repository=repository,
        queue=queue,
        objects=objects,
        bootstrap_contract=bootstrap_contract,
    )
    workflow = WorkflowDriver(
        repository=repository,
        calls=agent_calls,
        control=control,
        trials=trials,
        engine_inputs=engine_inputs,
        bootstrap=(
            BootstrapWorkflowDriver(
                repository=repository,
                queue=queue,
                objects=objects,
                contract=bootstrap_contract,
                engine_inputs=engine_inputs,
                trials=trials,
                calls=agent_calls,
                operator_evaluation=operator_evaluation,
            )
            if bootstrap_contract is not None
            else None
        ),
        operator_evaluation=operator_evaluation,
    )
    return ControlProcess(
        harness=harness,
        trials=trials,
        agents=agents,
        contexts=contexts,
        agent_calls=agent_calls,
        control=control,
        workflow=workflow,
        repository=repository,
        tasks=tasks,
    )


def build_agent_process(
    *,
    project_root: str | Path,
    runs_root: str | Path,
    agent_config: Mapping[str, object] | None = None,
    artifact_builder_config: Mapping[str, object] | None = None,
) -> AgentProcess:
    """Compose the independent Agent worker side of the Agent Port."""
    project = Path(project_root).resolve()
    repository = FileRunRepository(runs_root)
    tasks = default_task_registry()
    resolved = dict(agent_config or {})
    resolved_artifact_builder = dict(artifact_builder_config or {})
    runtime = _build_agent_runtime(
        project_root=project, runs_root=runs_root, agent_config=resolved
    )
    calls = AgentCallService(
        runtime=runtime,
        contexts=RoleContextPackager(repository=repository, tasks=tasks),
        tasks=tasks,
        max_retries=int(resolved.get("max_retries", 3)),
        max_reflections=int(resolved_artifact_builder.get("max_reflections", 2)),
        heartbeat_timeout_seconds=float(resolved.get("heartbeat_timeout_seconds", 900)),
        execution_mode="inline",
    )
    return AgentProcess(calls=calls, repository=repository)


def build_review_process(
    *,
    project_root: str | Path,
    queue_root: str | Path,
    work_root: str | Path,
    provider_config: Mapping[str, object],
    local_judge_config: Mapping[str, object],
    coordinator_id: str | None = None,
) -> ReviewProcess:
    """Compose the independent Review worker without an MCP/Agent boundary."""
    work = Path(work_root).resolve()
    queue = FileReviewCommandQueue(queue_root, coordinator_id=coordinator_id)
    provider = LocalAnalyzerTransport.from_config(
        provider=provider_config,
        local_judge=local_judge_config,
    )
    jobs = LocalAnalyzerReviewService(
        work / "rubric-jobs.json",
        provider,
        auto_run=True,
        max_concurrency=provider.max_concurrent_requests,
        batch_timeout_seconds=provider.batch_timeout_seconds,
    )
    output_root = work / "artifacts"

    def rubric_job_only(**_kwargs):
        raise RuntimeError("production Analyzer Review requires the Rubric Job path")

    service = ReviewLaborService(
        reviewer=rubric_job_only,
        model=provider.model,
        output_root=output_root,
        workspace_root=work,
        rubric_jobs=jobs,
    )
    return ReviewProcess(
        worker=ReviewWorker(
            queue,
            ReviewLaborBatchProcessor(
                service,
                workspace_root=work,
                output_root=output_root,
            ),
        ),
        queue=queue,
        close=provider.close,
    )


def _reward_harness_judge(state) -> RewardHarnessJudge:
    if state.run_resources is None:
        raise ValueError("Run has no Local Judge resource binding")
    judge = state.run_resources["local_judge"]
    authorization_env = str(judge["authorization_env"])
    authorization = os.environ.get(authorization_env) or read_dotenv_value(
        authorization_env
    )
    if not authorization:
        raise ValueError("Local Judge authorization environment is unavailable")
    gateway = HttpRubricJobGateway(
        str(judge["gateway_url"]),
        authorization=authorization,
        timeout_seconds=float(judge["timeout_policy"]["request_timeout_seconds"]),
    )
    return RewardHarnessJudge(LocalRubricJudgeClient(gateway))


def _selection_harness_judge(state, context) -> SelectionJudgeBatchDispatcher:
    if state.run_resources is None:
        raise ValueError("Run has no Local Judge resource binding")
    judge = state.run_resources["local_judge"]
    authorization_env = str(judge["authorization_env"])
    authorization = os.environ.get(authorization_env) or read_dotenv_value(
        authorization_env
    )
    if not authorization:
        raise ValueError("Local Judge authorization environment is unavailable")
    gateway = HttpRubricJobGateway(
        str(judge["gateway_url"]),
        authorization=authorization,
        timeout_seconds=float(judge["timeout_policy"]["request_timeout_seconds"]),
    )
    vllm = judge.get("vllm") if isinstance(judge.get("vllm"), dict) else {}
    return SelectionJudgeBatchDispatcher(
        LocalRubricJudgeClient(gateway),
        max_batch_size=int(vllm.get("max_num_seqs") or 1),
        submission_id=(
            f"{context.run_id}:{context.delivery.call.call_id}:"
            f"r{context.delivery.call.reflection_index:03d}:selection"
        ),
        job_metadata={
            "run_id": context.run_id,
            "scope": {
                "run_id": context.run_id,
                "trial_id": context.trial_key.trial_id,
            },
            "subject_ref": context.trial_key.subject_ref,
            "phase": "sft_selection_pretraining",
        },
    )


def _operator_evaluation_driver(
    *,
    repository: FileRunRepository,
    queue: FileCommandQueue,
    objects: FileEngineObjectStore,
    bootstrap_contract: Mapping[str, object] | None,
) -> TrialOperatorEvaluationDriver | None:
    if not bootstrap_contract or not bootstrap_contract.get("enabled"):
        return None
    base = bootstrap_contract.get("base")
    if not isinstance(base, dict):
        raise ValueError("bootstrap base contract must be a mapping")
    profiles = base.get("evaluation_profiles")
    if not isinstance(profiles, dict):
        raise ValueError("bootstrap evaluation profiles must be a mapping")
    operator = profiles.get("operator")
    if not isinstance(operator, dict) or not isinstance(operator.get("request"), dict):
        raise ValueError("bootstrap operator evaluation request is required")
    return TrialOperatorEvaluationDriver(
        repository=repository,
        queue=queue,
        objects=objects,
        request=operator["request"],
        base_model=str(base["model"]),
        base_model_protocol=base["model_protocol"],
    )
