"""CLI for the component-redesigned ADE Harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ade")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    commands = parser.add_subparsers(dest="command", required=True)
    from ade.harness.cluster import add_cluster_parser

    add_cluster_parser(commands)
    create = commands.add_parser("create")
    create.add_argument("config", type=Path)
    create.add_argument("--run-id")
    create.add_argument("--deployment", type=Path)
    for name in ("status", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("run_id")
    cancel = commands.add_parser("cancel")
    cancel.add_argument("run_id")
    cancel.add_argument("--reason", required=True)
    pause = commands.add_parser("pause")
    pause.add_argument("run_id")
    pause.add_argument("--reason", required=True)
    coordinator = commands.add_parser("coordinator")
    coordinator_commands = coordinator.add_subparsers(
        dest="coordinator_command", required=True
    )
    coordinator_finish = coordinator_commands.add_parser("finish")
    coordinator_finish.add_argument("run_id")
    coordinator_finish.add_argument("--coordinator-id", required=True)
    coordinator_finish.add_argument("--after-plans", type=int)
    coordinator_finish.add_argument("--reason", required=True)
    experiment = commands.add_parser("experiment")
    experiment_commands = experiment.add_subparsers(
        dest="experiment_command", required=True
    )
    experiment_inspect = experiment_commands.add_parser("inspect")
    experiment_inspect.add_argument("config", type=Path)
    experiment_inspect.add_argument("--deployment", type=Path, required=True)
    experiment_inspect.add_argument("--run-id", required=True)
    run = commands.add_parser("run")
    run_commands = run.add_subparsers(dest="run_command", required=True)
    run_history = run_commands.add_parser("history")
    run_history.add_argument("run_id")
    run_observe = run_commands.add_parser("observe")
    run_observe.add_argument("run_id")
    run_observe.add_argument("--queue-root", type=Path, default=Path("runs/queue"))
    run_finish = run_commands.add_parser("finish")
    run_finish.add_argument("run_id")
    run_finish.add_argument("--plans-per-coordinator", type=int, required=True)
    run_finish.add_argument("--reason", required=True)
    run_fork = run_commands.add_parser("fork")
    fork_cause = run_fork.add_mutually_exclusive_group(required=True)
    fork_cause.add_argument("--invalidate", dest="invalid_accepted_fact")
    fork_cause.add_argument("--unfenceable", dest="unfenceable_execution")
    fork_cause.add_argument("--continue-cancelled", dest="continue_cancelled")
    for name in ("start", "resume"):
        supervised = run_commands.add_parser(name)
        supervised.add_argument("experiment_config", type=Path)
        supervised.add_argument("--deployment", type=Path, required=True)
        supervised.add_argument("--run-id", required=True)
        if name == "start":
            supervised.add_argument("--initial-state-reference")
            supervised.add_argument(
                "--initial-state-frontier", action="append", default=[]
            )
        supervised.add_argument("--project-root", type=Path, default=Path.cwd())
        supervised.add_argument("--control-root", type=Path)
        supervised.add_argument("--queue-root", type=Path)
        supervised.add_argument("--object-root", type=Path)
        supervised.add_argument("--work-root", type=Path)
        supervised.add_argument("--poll-interval", type=float, default=10.0)
    run_seed = run_commands.add_parser("seed")
    run_seed_commands = run_seed.add_subparsers(
        dest="run_seed_command", required=True
    )
    run_seed_inspect = run_seed_commands.add_parser("inspect")
    run_seed_inspect.add_argument("experiment_config", type=Path)
    run_seed_inspect.add_argument("--deployment", type=Path, required=True)
    run_seed_inspect.add_argument("--source-run", required=True)
    run_seed_inspect.add_argument("--frontier", action="append", required=True)
    control = commands.add_parser("control")
    control_commands = control.add_subparsers(dest="control_command", required=True)
    control_run = control_commands.add_parser("run")
    control_run.add_argument("experiment_config", type=Path)
    control_run.add_argument("--deployment", type=Path, required=True)
    control_run.add_argument("--run-id", required=True)
    control_run.add_argument("--project-root", type=Path, default=Path.cwd())
    control_run.add_argument("--runs-root", type=Path, default=Path("runs/control"))
    control_run.add_argument("--queue-root", type=Path, default=Path("runs/queue"))
    control_run.add_argument("--object-root", type=Path, default=Path("runs/objects"))
    control_run.add_argument("--poll-interval", type=float, default=10.0)
    control_run.add_argument("--resume", action="store_true")
    control_step = control_commands.add_parser("step")
    for action in (control_step,):
        action.add_argument("experiment_config", type=Path)
        action.add_argument("--deployment", type=Path, required=True)
        action.add_argument("--run-id", required=True)
        action.add_argument("--project-root", type=Path, default=Path.cwd())
        action.add_argument("--runs-root", type=Path, default=Path("runs/control"))
        action.add_argument("--queue-root", type=Path, default=Path("runs/queue"))
        action.add_argument("--object-root", type=Path, default=Path("runs/objects"))
        action.add_argument("--resume", action="store_true")
    agent = commands.add_parser("agent")
    agent_commands = agent.add_subparsers(dest="agent_command", required=True)
    agent_worker = agent_commands.add_parser("worker")
    agent_worker.add_argument("experiment_config", type=Path)
    agent_worker.add_argument("--deployment", type=Path, required=True)
    agent_worker.add_argument("--run-id", required=True)
    agent_worker.add_argument("--project-root", type=Path, default=Path.cwd())
    agent_worker.add_argument("--runs-root", type=Path, default=Path("runs/control"))
    agent_worker.add_argument("--poll-interval", type=float, default=10.0)
    agent_worker.add_argument("--coordinator-id")
    agent_worker.add_argument("--global-only", action="store_true")
    agent_one = agent_commands.add_parser("run-one")
    agent_one.add_argument("experiment_config", type=Path)
    agent_one.add_argument("--deployment", type=Path, required=True)
    agent_one.add_argument("--run-id", required=True)
    agent_one.add_argument("--project-root", type=Path, default=Path.cwd())
    agent_one.add_argument("--runs-root", type=Path, default=Path("runs/control"))
    agent_revalidate = agent_commands.add_parser("revalidate-one")
    agent_revalidate.add_argument("experiment_config", type=Path)
    agent_revalidate.add_argument("--deployment", type=Path, required=True)
    agent_revalidate.add_argument("--run-id", required=True)
    agent_revalidate.add_argument("--project-root", type=Path, default=Path.cwd())
    agent_revalidate.add_argument("--runs-root", type=Path, default=Path("runs/control"))
    engine = commands.add_parser("engine")
    engine_commands = engine.add_subparsers(dest="engine_command", required=True)
    engine_worker = engine_commands.add_parser("worker")
    engine_worker.add_argument("--queue-root", type=Path, required=True)
    engine_worker.add_argument("--object-root", type=Path, required=True)
    engine_worker.add_argument("--work-root", type=Path, default=Path("runs/engine-work"))
    engine_worker.add_argument("--poll-interval", type=float, default=10.0)
    engine_worker.add_argument(
        "--claim-timeout-seconds", type=float, default=1800.0
    )
    engine_worker.add_argument(
        "--heartbeat-timeout-seconds", type=float, default=1800.0
    )
    engine_worker.add_argument("--coordinator-id")
    engine_one = engine_commands.add_parser("run-one")
    engine_one.add_argument("--queue-root", type=Path, required=True)
    engine_one.add_argument("--object-root", type=Path, required=True)
    engine_one.add_argument("--work-root", type=Path, default=Path("runs/engine-work"))
    engine_one.add_argument("--claim-timeout-seconds", type=float, default=1800.0)
    engine_one.add_argument("--heartbeat-timeout-seconds", type=float, default=1800.0)
    review = commands.add_parser("review")
    review_commands = review.add_subparsers(dest="review_command", required=True)
    for name in ("worker", "run-one"):
        review_action = review_commands.add_parser(name)
        review_action.add_argument("experiment_config", type=Path)
        review_action.add_argument("--deployment", type=Path, required=True)
        review_action.add_argument("--run-id", required=True)
        review_action.add_argument("--project-root", type=Path, default=Path.cwd())
        review_action.add_argument("--runs-root", type=Path, required=True)
        review_action.add_argument("--queue-root", type=Path, required=True)
        review_action.add_argument("--work-root", type=Path, required=True)
        if name == "worker":
            review_action.add_argument("--poll-interval", type=float, default=10.0)
            review_action.add_argument("--coordinator-id")
    monitor = commands.add_parser("monitor")
    monitor_commands = monitor.add_subparsers(dest="monitor_command", required=True)
    monitor_run = monitor_commands.add_parser("run")
    monitor_run.add_argument("experiment_config", type=Path)
    monitor_run.add_argument("--deployment", type=Path, required=True)
    monitor_run.add_argument("--run-id", required=True)
    monitor_run.add_argument("--project-root", type=Path, default=Path.cwd())
    monitor_run.add_argument("--runs-root", type=Path, default=Path("runs/control"))
    evaluate = commands.add_parser("evaluate")
    evaluate_commands = evaluate.add_subparsers(
        dest="evaluate_command", required=True
    )
    evaluate_create = evaluate_commands.add_parser("create")
    evaluate_create.add_argument("config", type=Path)
    evaluate_create.add_argument("--evaluation-id")
    evaluate_create.add_argument("--deployment", type=Path, required=True)
    evaluate_run = evaluate_commands.add_parser("run")
    evaluate_run.add_argument("evaluation_id")
    evaluate_run.add_argument("--unit-id")
    evaluate_status = evaluate_commands.add_parser("status")
    evaluate_status.add_argument("evaluation_id")
    evaluate_retry = evaluate_commands.add_parser("retry")
    evaluate_retry.add_argument("evaluation_id")
    evaluate_retry.add_argument("--unit-id", required=True)
    for evaluate_action in evaluate_commands.choices.values():
        evaluate_action.add_argument(
            "--evaluation-root", type=Path, default=Path("runs/evaluations")
        )
        evaluate_action.add_argument(
            "--queue-root", type=Path, default=Path("runs/queue")
        )
        evaluate_action.add_argument(
            "--object-root", type=Path, default=Path("runs/objects")
        )
    evaluate_prepare = evaluate_commands.add_parser("prepare")
    evaluate_prepare.add_argument("operation_request", type=Path)
    evaluate_supervise = evaluate_commands.add_parser("supervise")
    evaluate_supervise.add_argument("operation_request", type=Path)
    evaluate_supervise.add_argument(
        "--retry-repaired-unit",
        help="Retry one failed full-matrix unit after an executor repair, within the original budget",
    )
    pre_gpu = commands.add_parser("pre-gpu")
    pre_gpu_commands = pre_gpu.add_subparsers(
        dest="pre_gpu_command", required=True
    )
    pre_gpu_run = pre_gpu_commands.add_parser("run")
    pre_gpu_run.add_argument("experiment_config", type=Path)
    pre_gpu_run.add_argument("--deployment", type=Path, required=True)
    pre_gpu_run.add_argument("--run-id", required=True)
    pre_gpu_run.add_argument("--project-root", type=Path, default=Path.cwd())
    pre_gpu_run.add_argument("--gate-root", type=Path, required=True)
    pre_gpu_run.add_argument("--max-cycles", type=int, default=160)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from ade.harness.environment import load_project_environment

    load_project_environment(args.project_root)
    if args.command == "cluster":
        from ade.harness.cluster import run_cluster

        return run_cluster(args)
    if args.command == "experiment":
        from ade.harness.experiment_config import ExperimentConfigCompiler
        from ade.harness.runtime_roots import deployment_runtime_roots
        from ade.tasks.registry import default_task_registry

        config = args.config
        if not config.is_absolute():
            config = args.project_root / config
        compiled = ExperimentConfigCompiler(
            default_task_registry(), project_root=args.project_root
        ).compile_file(
            config,
            deployment_config=args.deployment,
            run_id=args.run_id,
        )
        roots = deployment_runtime_roots(args.project_root, compiled.resolved)
        result = {
            "experiment_config": str(config.resolve()),
            "experiment_id": compiled.experiment_id,
            "run_id": compiled.run.run_id,
            "config_digest": compiled.config_digest,
            "runtime_roots": {
                **{name: str(path) for name, path in roots.items()},
                "run": str(roots["control"] / compiled.run.run_id),
            },
            "resolved": compiled.to_dict(),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        if args.run_command == "seed":
            from ade.controller.run_seed import (
                RunSeedImporter,
                RunSeedRequest,
                resolve_run_seed_source,
            )
            from ade.harness.experiment_config import ExperimentConfigCompiler
            from ade.tasks.registry import default_task_registry

            config = args.experiment_config
            if not config.is_absolute():
                config = args.project_root / config
            experiment = ExperimentConfigCompiler(
                default_task_registry(), project_root=args.project_root
            ).compile_file(config, deployment_config=args.deployment)
            if not bool(experiment.bootstrap.get("reference_enabled")):
                raise ValueError(
                    "run seed inspect requires bootstrap.reference.enabled=true"
                )
            source_repository, source_objects, deployment_id = (
                resolve_run_seed_source(args.project_root, args.source_run)
            )
            importer = RunSeedImporter(
                source_repository,
                source_objects,
                source_repository=source_repository,
                source_objects=source_objects,
                source_deployment_id=deployment_id,
            )
            resolution = importer.resolve(
                RunSeedRequest.parse(args.source_run, args.frontier),
                target_resolved=experiment.resolved,
            )
            print(
                json.dumps(
                    importer.inspect(
                        resolution,
                        target_plan_budget=experiment.run.portfolio.max_plans,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.run_command in {"start", "resume"}:
            from ade.harness.supervisor import RunSupervisor

            return RunSupervisor(
                project_root=args.project_root,
                runs_root=args.control_root,
                queue_root=args.queue_root,
                object_root=args.object_root,
                work_root=args.work_root,
            ).run(
                experiment_config=args.experiment_config,
                deployment_config=args.deployment,
                run_id=args.run_id,
                resume=args.run_command == "resume",
                initial_state_reference=(
                    args.initial_state_reference
                    if args.run_command == "start"
                    else None
                ),
                initial_state_frontier=(
                    tuple(args.initial_state_frontier)
                    if args.run_command == "start"
                    else ()
                ),
                poll_interval=args.poll_interval,
            )
        from ade.memory.repository import FileRunRepository

        repository = FileRunRepository(args.runs_root)
        if args.run_command == "finish":
            from ade.harness.config import ConfigCompiler
            from ade.harness.service import Harness
            from ade.tasks.registry import default_task_registry

            result = Harness(
                repository,
                ConfigCompiler(default_task_registry()),
            ).request_run_finish(
                args.run_id,
                plans_per_coordinator=args.plans_per_coordinator,
                reason=args.reason,
            ).to_dict()
        elif args.run_command == "fork":
            from ade.controller.fork import RunForkService
            from ade.core.run import ForkCause

            cause, evidence_ref = next(
                (ForkCause(name), getattr(args, name))
                for name in (
                    "invalid_accepted_fact",
                    "unfenceable_execution",
                    "continue_cancelled",
                )
                if getattr(args, name) is not None
            )
            result = RunForkService(repository).fork(
                cause=cause,
                evidence_ref=evidence_ref,
            ).to_dict()
        elif args.run_command == "history":
            from ade.core.scope import subject_ref

            revisions = []
            root = repository.layout.run_dir(args.run_id) / "state" / "revisions"
            if not root.is_dir():
                repository.load(args.run_id)
            for path in sorted(root.glob("rev-*/run.json")):
                state = repository.load_revision(
                    args.run_id,
                    int(path.parent.name.removeprefix("rev-")),
                )
                revisions.append(
                    {
                        "revision": state.revision,
                        "revision_ref": f"{state.run_id}@rev-{state.revision:06d}",
                        "transition": {
                            "transition_id": state.last_transition.transition_id,
                            "kind": state.last_transition.kind,
                            "fact_class": state.last_transition.fact_class.value,
                            "scope": state.last_transition.scope,
                            "subject_ref": state.last_transition.subject_ref,
                            "logical_work_ref": (
                                state.last_transition.logical_work_ref
                            ),
                            "accepted_fact_refs": (
                                state.last_transition.accepted_fact_refs
                            ),
                            "origin_refs": state.last_transition.origin_refs,
                        },
                        "status": state.status.value,
                        "trial_phases": {
                            subject_ref(
                                state.run_id,
                                trial.coordinator_id,
                                trial.plan_id,
                                trial.trial_id,
                            ): trial.phase.value
                            for trial in state.trials
                        },
                        "recovery": (
                            None
                            if state.recovery is None
                            else {
                                **state.recovery.__dict__,
                                "continuation_status": (
                                    state.continuation_status.value
                                    if state.continuation_status is not None
                                    else None
                                ),
                            }
                        ),
                        "active_agent_calls": len(state.active_agent_calls),
                        "active_engine_commands": len(state.active_engine_commands),
                        "active_review_commands": len(state.active_review_commands),
                    }
                )
            result = {"run_id": args.run_id, "revisions": revisions}
        else:
            from ade.core.scope import subject_ref
            from ade.engine.command_queue import FileCommandQueue

            state = repository.load(args.run_id)
            queue = FileCommandQueue(args.queue_root)
            result = {
                "run_id": state.run_id,
                "revision": state.revision,
                "revision_ref": f"{state.run_id}@rev-{state.revision:06d}",
                "status": state.status.value,
                "bootstrap_status": state.bootstrap.status.value,
                "pause_requested": state.pause_requested,
                "pause_reason": state.pause_reason,
                "last_transition": (
                    state.last_transition.__dict__
                    if state.last_transition
                    else None
                ),
                "trial_phases": {
                    subject_ref(
                        state.run_id,
                        trial.coordinator_id,
                        trial.plan_id,
                        trial.trial_id,
                    ): trial.phase.value
                    for trial in state.trials
                },
                "recovery": (
                    None
                    if state.recovery is None
                    else {
                        **state.recovery.__dict__,
                        "continuation_status": (
                            state.continuation_status.value
                            if state.continuation_status is not None
                            else None
                        ),
                    }
                ),
                "active_agent_calls": [
                    {
                        "attempt_id": call.attempt_id,
                        "role": call.role,
                        "owner_subject_ref": call.owner_subject_ref,
                        "target_subject_ref": call.target_subject_ref,
                        "status": call.status,
                    }
                    for call in state.active_agent_calls
                ],
                "active_engine_commands": [
                    {
                        "command_id": command.command_id,
                        "logical_command_id": command.logical_command_id,
                        "attempt_id": command.attempt_id,
                        "attempt_index": command.attempt_index,
                        "kind": command.kind,
                        "subject_ref": subject_ref(
                            state.run_id,
                            command.coordinator_id,
                            command.plan_id,
                            command.trial_id,
                        ),
                        "status": command.status,
                        "liveness": (
                            queue.load_liveness(command.command_id)
                            if (queue.liveness / f"{command.command_id}.json").is_file()
                            else None
                        ),
                    }
                    for command in state.active_engine_commands
                ],
                "active_review_commands": [
                    {
                        "command_id": command.command_id,
                        "logical_command_id": command.logical_command_id,
                        "attempt_id": command.attempt_id,
                        "attempt_index": command.attempt_index,
                        "subject_ref": subject_ref(
                            state.run_id,
                            command.coordinator_id,
                            command.plan_id,
                            command.trial_id,
                        ),
                        "status": command.status,
                    }
                    for command in state.active_review_commands
                ],
                "accepted_artifacts": [
                    {"artifact_id": ref.artifact_id, "kind": ref.kind, "uri": ref.uri}
                    for ref in state.accepted_evidence_refs
                ],
                "paths": {
                    "run": str(repository.layout.run_dir(state.run_id)),
                    "state": str(repository.layout.run_dir(state.run_id) / "run.json"),
                    "queue": str(queue.root),
                },
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "control":
        from ade.harness.processes import (
            control_stop_reason,
            run_control_process,
            run_control_step,
        )

        if args.control_command == "step":
            state, transition = run_control_step(
                experiment_config=args.experiment_config,
                deployment_config=args.deployment,
                run_id=args.run_id,
                project_root=args.project_root,
                runs_root=args.runs_root,
                queue_root=args.queue_root,
                object_root=args.object_root,
                resume=args.resume,
            )
            print(json.dumps({
                "run_id": state.run_id,
                "revision": state.revision,
                "status": state.status.value,
                "result": type(transition).__name__,
                "transition": state.last_transition.kind if state.last_transition else None,
            }, indent=2, sort_keys=True))
            return 0
        state = run_control_process(
            experiment_config=args.experiment_config,
            deployment_config=args.deployment,
            run_id=args.run_id,
            project_root=args.project_root,
            runs_root=args.runs_root,
            queue_root=args.queue_root,
            object_root=args.object_root,
            poll_interval=args.poll_interval,
            resume=args.resume,
        )
        stop_reason = control_stop_reason(state)
        print(
            json.dumps(
                {
                    "revision": state.revision,
                    "run_id": state.run_id,
                    "status": state.status.value,
                    "bootstrap_status": state.bootstrap.status.value,
                    "stop_reason": stop_reason,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if (
            stop_reason in {"completed", "paused"}
            or stop_reason.startswith("until:")
        ) else 1
    if args.command == "engine":
        from ade.harness.processes import run_engine_one, run_engine_process

        if args.engine_command == "run-one":
            command_id = run_engine_one(
                queue_root=args.queue_root,
                object_root=args.object_root,
                work_root=args.work_root,
                claim_timeout_seconds=args.claim_timeout_seconds,
                heartbeat_timeout_seconds=args.heartbeat_timeout_seconds,
            )
            print(json.dumps({"command_id": command_id}, indent=2, sort_keys=True))
            return 0
        run_engine_process(
            queue_root=args.queue_root,
            object_root=args.object_root,
            work_root=args.work_root,
            poll_interval=args.poll_interval,
            claim_timeout_seconds=args.claim_timeout_seconds,
            heartbeat_timeout_seconds=args.heartbeat_timeout_seconds,
            coordinator_id=args.coordinator_id,
        )
        return 0
    if args.command == "review":
        from ade.harness.processes import run_review_one, run_review_process

        if args.review_command == "run-one":
            command_id = run_review_one(
                experiment_config=args.experiment_config,
                deployment_config=args.deployment,
                run_id=args.run_id,
                project_root=args.project_root,
                runs_root=args.runs_root,
                queue_root=args.queue_root,
                work_root=args.work_root,
            )
            print(json.dumps({"command_id": command_id}, indent=2, sort_keys=True))
            return 0
        run_review_process(
            experiment_config=args.experiment_config,
            deployment_config=args.deployment,
            run_id=args.run_id,
            project_root=args.project_root,
            runs_root=args.runs_root,
            queue_root=args.queue_root,
            work_root=args.work_root,
            poll_interval=args.poll_interval,
            coordinator_id=args.coordinator_id,
        )
        return 0
    if args.command == "monitor":
        from ade.harness.run_monitor import RunResourceMonitor

        return RunResourceMonitor(
            project_root=args.project_root,
            runs_root=args.runs_root,
            experiment_config=args.experiment_config,
            deployment_config=args.deployment,
            run_id=args.run_id,
        ).run()
    if args.command == "evaluate":
        from ade.engine.command_queue import FileCommandQueue
        from ade.engine.storage.object_store import FileEngineObjectStore
        from ade.harness.evaluation import (
            EvaluationConfigCompiler,
            StandaloneEvaluationService,
        )

        if args.evaluate_command in {"prepare", "supervise"}:
            from ade.harness.evaluation_supervisor import (
                GeneralizationSupervisor,
                ProductionRayInspector,
                SubprocessEngineWorkerPool,
                compile_operation_evaluation,
            )
            from ade.harness.generalization_operation import (
                GeneralizationOperationCompiler,
            )

            operation = GeneralizationOperationCompiler(
                project_root=args.project_root
            ).compile_file(args.operation_request)
            compiled = compile_operation_evaluation(
                project_root=args.project_root,
                operation=operation,
            )
            if args.evaluate_command == "prepare":
                result = {
                    **operation.summary(),
                    "config_digest": compiled.config_digest,
                    "resolved_unit_count": len(compiled.units),
                    "expected_generations": compiled.expected_generations,
                    "gpus_per_worker": max(
                        int(unit["request"]["data_parallel_shards"])
                        for unit in compiled.units.values()
                    ),
                    "required_gpus": operation.max_in_flight * max(
                        int(unit["request"]["data_parallel_shards"])
                        for unit in compiled.units.values()
                    ),
                    "checkpoint_unavailable_units": sorted(
                        unit_id
                        for unit_id, unit in compiled.units.items()
                        if not bool(unit["checkpoint_available"])
                    ),
                }
            else:
                queue = FileCommandQueue(operation.paths.queue_root)
                objects = FileEngineObjectStore(operation.paths.object_root)
                result = GeneralizationSupervisor(
                    project_root=args.project_root,
                    operation=operation,
                    config=compiled,
                    queue=queue,
                    objects=objects,
                    ray_inspector=ProductionRayInspector(),
                    worker=SubprocessEngineWorkerPool(operation),
                ).run_until_terminal(retry_repaired_unit=args.retry_repaired_unit)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        service = StandaloneEvaluationService(
            root=args.evaluation_root,
            queue=FileCommandQueue(args.queue_root),
            objects=FileEngineObjectStore(args.object_root),
        )
        if args.evaluate_command == "create":
            compiled = EvaluationConfigCompiler(
                project_root=args.project_root
            ).compile_file(
                args.config,
                deployment_config=args.deployment,
                evaluation_id=args.evaluation_id,
                evaluation_root=args.evaluation_root,
            )
            result = service.create(compiled)
        elif args.evaluate_command == "run":
            result = service.run(args.evaluation_id, unit_id=args.unit_id)
        elif args.evaluate_command == "retry":
            result = service.retry(args.evaluation_id, unit_id=args.unit_id)
        else:
            result = service.status(args.evaluation_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "agent":
        from ade.harness.processes import (
            revalidate_agent_one,
            run_agent_one,
            run_agent_process,
        )

        if args.agent_command == "revalidate-one":
            state, attempt_id = revalidate_agent_one(
                experiment_config=args.experiment_config,
                deployment_config=args.deployment,
                run_id=args.run_id,
                project_root=args.project_root,
                runs_root=args.runs_root,
            )
            print(json.dumps({
                "run_id": state.run_id,
                "revision": state.revision,
                "attempt_id": attempt_id,
                "backend_invoked": False,
            }, indent=2, sort_keys=True))
            return 0
        if args.agent_command == "run-one":
            state, attempt_id = run_agent_one(
                experiment_config=args.experiment_config,
                deployment_config=args.deployment,
                run_id=args.run_id,
                project_root=args.project_root,
                runs_root=args.runs_root,
            )
            print(json.dumps({
                "run_id": state.run_id,
                "revision": state.revision,
                "attempt_id": attempt_id,
            }, indent=2, sort_keys=True))
            return 0
        state = run_agent_process(
            experiment_config=args.experiment_config,
            deployment_config=args.deployment,
            run_id=args.run_id,
            project_root=args.project_root,
            runs_root=args.runs_root,
            poll_interval=args.poll_interval,
            coordinator_id=args.coordinator_id,
            global_only=args.global_only,
        )
        return 0 if state.status.value == "completed" else 1
    if args.command == "pre-gpu":
        from ade.harness.pre_gpu import PreGpuGateRunner

        runner = PreGpuGateRunner(
            project_root=args.project_root,
            experiment_config=args.experiment_config,
            deployment_config=args.deployment,
            run_id=args.run_id,
            gate_root=args.gate_root,
        )
        runner.create()
        state = runner.run(max_cycles=args.max_cycles)
        print(
            json.dumps(
                {
                    "run_id": state.run_id,
                    "revision": state.revision,
                    "status": state.status.value,
                    "evidence": str(runner.evidence_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    from ade.harness.config import ConfigCompiler
    from ade.harness.service import Harness
    from ade.memory.repository import FileRunRepository
    from ade.tasks.registry import default_task_registry

    repository = FileRunRepository(args.runs_root)
    if args.command == "cancel":
        from ade.harness.processes import run_resource_admission_from_resources

        current = repository.load(args.run_id)
        if current.run_resources is not None:
            repository.run_resource_admission = run_resource_admission_from_resources(
                current.run_resources,
                runs_root=args.runs_root,
                project_root=args.project_root,
            )
    harness = Harness(repository, ConfigCompiler(default_task_registry()))
    if args.command == "create":
        import yaml

        from ade.harness.experiment_config import (
            ExperimentConfigCompiler,
            generated_experiment_run_id,
        )

        payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("run config must be a mapping")
        if "experiment" in payload:
            if args.deployment is None:
                raise ValueError("experiment create requires --deployment")
            requested_run_id = args.run_id or generated_experiment_run_id(
                str(payload["experiment"])
            )
            experiment = ExperimentConfigCompiler(
                default_task_registry(),
                project_root=args.project_root,
            ).compile_file(
                args.config,
                deployment_config=args.deployment,
                run_id=requested_run_id,
            )
            result = harness.create_experiment(experiment).to_dict()
        else:
            if args.run_id is not None:
                raise ValueError("--run-id requires an experiment config")
            if args.deployment is not None:
                raise ValueError("--deployment requires an experiment config")
            result = harness.create(payload).to_dict()
    elif args.command == "status":
        result = harness.status(args.run_id)
    elif args.command == "inspect":
        result = harness.inspect(args.run_id)
    elif args.command == "pause":
        result = harness.request_pause(
            args.run_id, reason=args.reason
        ).to_dict()
    elif args.command == "coordinator":
        result = harness.request_coordinator_finish(
            args.run_id,
            coordinator_id=args.coordinator_id,
            after_plans=args.after_plans,
            reason=args.reason,
        ).to_dict()
    else:
        result = harness.cancel(args.run_id, reason=args.reason).to_dict()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
