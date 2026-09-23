#!/usr/bin/env python3
"""Run one real Judge-outage Engine Attempt and same-Run replay."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import uuid

from ade.controller.state import StateCoordinator
from ade.controller.trial_lifecycle import TrialLifecycle
from ade.core.coordinator import CoordinatorKind, CoordinatorState
from ade.core.engine import EngineReceiptStatus
from ade.core.insight import InsightGraph
from ade.core.outcomes import RunResumedOutcome
from ade.core.plan import PlanKind, PlanState, PlanStatus
from ade.core.ranking import RankingState
from ade.core.run import PortfolioState, RunState, RunStatus
from ade.core.scope import TrialKey
from ade.core.trial import TrialPhase, TrialState
from ade.engine.storage.atomic import write_json_atomic
from ade.harness.environment import load_project_environment
from ade.harness.experiment_config import ExperimentConfigCompiler
from ade.harness.processes import _admit_resumed_run_resources
from ade.harness.wiring import build_engine_process
from ade.local_rubric_judge.launcher import ProductionJudgeLauncher
from ade.local_rubric_judge.lifecycle import (
    LocalJudgeBinding,
    RunResourceAdmission,
)
from ade.memory.repository import FileRunRepository
from ade.tasks.registry import default_task_registry
from scripts.probe_engine_backend import (
    _reward_artifact,
    _scale_input,
    _validate_result,
)


def _admission(project: Path, compiled):
    resources = compiled.run.run_resources
    if resources is None:
        raise ValueError("RFT recovery probe requires deployment resources")
    ray = resources["ray_cluster"]
    judge = resources["local_judge"]
    launcher = ProductionJudgeLauncher(
        project / "runs" / "deployments" / str(ray["cluster_id"]) / "local-judge",
        authorization_env=str(judge["authorization_env"]),
        readiness_seconds=float(judge["timeout_policy"]["readiness_seconds"]),
        request_timeout_seconds=float(
            judge["timeout_policy"]["request_timeout_seconds"]
        ),
        vllm=dict(judge["vllm"]),
        generation=dict(judge["generation"]),
    )
    admission = RunResourceAdmission(
        project
        / "runs"
        / "deployments"
        / str(ray["cluster_id"])
        / "run-services",
        launcher,
    )
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
    return admission, launcher, binding


def _service_handle(admission: RunResourceAdmission) -> dict[str, object]:
    value = json.loads((admission.state_root / "service.json").read_text())
    handle = value.get("service_handle")
    if value.get("status") != "ready" or not isinstance(handle, dict):
        raise RuntimeError("Local Judge has no persisted ready service handle")
    return handle


def _create_probe_run(
    *,
    repository: FileRunRepository,
    compiled,
    ready,
    run_id: str,
):
    plugin = default_task_registry().get("reward_design")
    artifact = _reward_artifact(plugin, compiled.run.task.config)
    decision = (
        json.dumps(
            {
                "schema_version": "1",
                "relation": {
                    "kind": "new_direction",
                    "related_plans": [],
                },
                "design": {"score_bounds": [0.0, 1.0]},
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    artifact_ref = repository.describe_artifact(run_id, artifact.kind, artifact.content)
    decision_ref = repository.describe_artifact(run_id, "planning_decision", decision)
    resources = copy.deepcopy(compiled.run.run_resources)
    assert resources is not None
    resources["local_judge"] = {
        **resources["local_judge"],
        "host": ready.host,
        "gateway_url": ready.gateway_url,
        "node_id": ready.node_id,
        "launch_id": ready.launch_id,
        "service_state": ready.status,
        "readiness": True,
        "state_path": ready.state_path,
    }
    state = RunState(
        run_id=run_id,
        revision=0,
        status=RunStatus.RUNNING,
        task=compiled.run.task,
        portfolio=PortfolioState(1, 1),
        insight_graph=InsightGraph(0),
        ranking=RankingState("offline.score"),
        coordinators=(CoordinatorState("c001", CoordinatorKind.SEARCH, 1, 1),),
        plans=(
            PlanState(
                "p001",
                "c001",
                kind=PlanKind.SEARCH,
                status=PlanStatus.ACTIVE,
                decision_ref_id=decision_ref.artifact_id,
            ),
        ),
        trials=(
            TrialState(
                "t001",
                "c001",
                "p001",
                phase=TrialPhase.ARTIFACT_READY,
                artifact_ref_id=artifact_ref.artifact_id,
            ),
        ),
        accepted_plan_refs=(decision_ref,),
        accepted_evidence_refs=(artifact_ref,),
        analysis_policy=compiled.run.analysis_policy,
        run_resources=resources,
    )
    created = repository.create(
        state,
        initial_artifacts=(
            (decision_ref, decision),
            (artifact_ref, artifact.content),
        ),
    )
    # This focused probe seeds ARTIFACT_READY directly instead of replaying the
    # Agent workflow.  Create the Control-owned Trial Record that a normal
    # TrialProposedOutcome would have created before Engine evidence admission.
    trial = created.trials[0]
    repository.trial_records.create(
        run_id=run_id,
        coordinator_id=trial.coordinator_id,
        plan_id=trial.plan_id,
        trial_id=trial.trial_id,
        created_revision=created.revision,
        plan_memory_basis=trial.plan_memory_basis,
        run_memory_basis=trial.run_memory_basis,
    )
    return created


def _attempt_workspace(root: Path, command_id: str) -> Path:
    return root / "engine-work" / "runtime" / command_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument(
        "--experiment",
        default="configs/experiments/math-math-rft-reward-design-ADE-formal-n1.yaml",
    )
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    load_project_environment(project)
    experiment = Path(args.experiment)
    if not experiment.is_absolute():
        experiment = project / experiment
    run_id = args.run_id or (
        "r1-engine-recovery-"
        + time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        + f"-{uuid.uuid4().hex[:8]}"
    )
    root = project / "runs" / "recovery-validation" / run_id
    receipt_path = root / "receipts" / "engine-recovery.json"
    receipt: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment": str(experiment.resolve()),
        "started_at": time.time(),
        "status": "running",
    }
    admission = None
    repository = None
    run_created = False
    replacement_ready = False
    try:
        compiled = ExperimentConfigCompiler(
            default_task_registry(), project_root=project
        ).compile_file(
            experiment,
            deployment_config=args.deployment,
            run_id=run_id,
        )
        resources = compiled.run.run_resources
        if resources is None:
            raise ValueError("recovery probe requires resolved Run resources")
        circuit = resources["local_judge"]["circuit_policy"]
        if int(circuit["max_consecutive_unhealthy_jobs"]) != 1:
            raise ValueError(
                "recovery probe requires first job-wide Judge outage to open the circuit"
            )
        os.environ["RAY_ADDRESS"] = str(resources["ray_cluster"]["address"])

        admission, launcher, binding = _admission(project, compiled)
        ready = admission.admit(
            run_id=run_id,
            binding=binding,
        )
        receipt["initial_judge_launch_id"] = ready.launch_id
        repository = FileRunRepository(
            root / "control", run_resource_admission=admission
        )
        _create_probe_run(
            repository=repository,
            compiled=compiled,
            ready=ready,
            run_id=run_id,
        )
        run_created = True

        process = build_engine_process(
            queue_root=root / "queue",
            object_root=root / "objects",
            work_root=root / "engine-work",
            engine_config=compiled.control["engine"],
        )
        lifecycle = TrialLifecycle(
            repository=repository,
            tasks=default_task_registry(),
            queue=process.queue,
            engine_io=process.objects,
        )
        trial_key = TrialKey(run_id, "c001", "p001", "t001")
        engine_input = _scale_input(
            "rft", compiled.engine_inputs["reward_design"]
        )

        attempt_one = lifecycle.submit_engine(run_id, trial_key, engine_input)
        print(
            json.dumps(
                {
                    "phase": "attempt_001_submitted",
                    "run_id": run_id,
                    "command_id": attempt_one.command_id,
                }
            ),
            flush=True,
        )
        initial_service_handle = _service_handle(admission)
        launcher.stop(initial_service_handle)
        if not launcher.released(initial_service_handle):
            raise RuntimeError("fault injection did not release the initial Judge processes")
        stopped_health = launcher.health(initial_service_handle)
        if stopped_health.get("ready"):
            raise RuntimeError("fault injection left the initial Judge launch reachable")
        receipt["fault"] = {
            "kind": "local_judge_processes_stopped",
            "after_dispatch": True,
            "before_optimizer_update": True,
            "launch_id": str(initial_service_handle["launch_id"]),
            "verified_released": True,
        }
        print(json.dumps({"phase": "judge_unavailable"}), flush=True)

        failed_receipt = process.worker.run_once()
        if (
            failed_receipt is None
            or failed_receipt.status is not EngineReceiptStatus.FAILED
            or failed_receipt.failure_kind != "dependency_unavailable"
            or not failed_receipt.retryable
        ):
            raise RuntimeError(
                f"Attempt-001 did not terminalize as retryable Judge dependency failure: {failed_receipt}"
            )
        lifecycle.collect_engine(run_id, attempt_one.command_id)
        suspended = repository.load(run_id)
        trial = suspended.trials[0]
        if (
            suspended.status is not RunStatus.SUSPENDED
            or not trial.engine_retry_pending
            or trial.engine_attempt_index != 1
            or trial.command_id is not None
        ):
            raise RuntimeError("Attempt-001 did not produce durable Run suspension")
        failed_workspace = _attempt_workspace(root, attempt_one.command_id)
        failed_optimizer_checkpoints = tuple(
            str(path)
            for path in failed_workspace.glob("checkpoints/verl/global_step_*")
        )
        if failed_optimizer_checkpoints:
            raise RuntimeError(
                "Judge outage occurred after an optimizer checkpoint was written"
            )
        if suspended.ranking.entries or suspended.memory.run_head != "RM000":
            raise RuntimeError("failed Attempt changed Ranking or Memory")
        receipt["attempt_001"] = {
            "command_id": attempt_one.command_id,
            "logical_command_id": attempt_one.logical_command_id,
            "attempt_id": attempt_one.attempt_id,
            "receipt": failed_receipt.to_dict(),
            "suspended_revision": suspended.revision,
            "failure_ref_ids": list(trial.engine_attempt_failure_ref_ids),
            "optimizer_checkpoints": list(failed_optimizer_checkpoints),
            "ranking_entries": len(suspended.ranking.entries),
            "memory_head": suspended.memory.run_head,
        }
        print(
            json.dumps(
                {
                    "phase": "run_suspended",
                    "revision": suspended.revision,
                    "failure_kind": failed_receipt.failure_kind,
                }
            ),
            flush=True,
        )

        replaced = _admit_resumed_run_resources(
            SimpleNamespace(repository=repository), suspended, compiled
        )
        replacement_ready = True
        if replaced.run_resources is None:
            raise RuntimeError("replacement lost persisted Run resources")
        replacement_launch_id = replaced.run_resources["local_judge"]["launch_id"]
        if replacement_launch_id == ready.launch_id:
            raise RuntimeError("Judge outage did not create a replacement launch")
        resumed = StateCoordinator(repository).apply(
            run_id,
            RunResumedOutcome(run_id=run_id, basis_revision=replaced.revision),
            event_type="run_resumed",
        )
        receipt["replacement"] = {
            "launch_id": replacement_launch_id,
            "revision": replaced.revision,
            "resumed_revision": resumed.revision,
            "state_path_unchanged": (
                replaced.run_resources["local_judge"]["state_path"]
                == ready.state_path
            ),
        }
        print(
            json.dumps(
                {
                    "phase": "judge_replaced_and_run_resumed",
                    "launch_id": replacement_launch_id,
                    "revision": resumed.revision,
                }
            ),
            flush=True,
        )

        attempt_two = lifecycle.submit_engine(run_id, trial_key, engine_input)
        if (
            attempt_two.logical_command_id != attempt_one.logical_command_id
            or attempt_two.command_id == attempt_one.command_id
            or attempt_two.attempt_id != "attempt-002"
        ):
            raise RuntimeError("Attempt-002 identity did not replay the logical Command")
        print(
            json.dumps(
                {
                    "phase": "attempt_002_submitted",
                    "command_id": attempt_two.command_id,
                }
            ),
            flush=True,
        )
        succeeded_receipt = process.worker.run_once()
        if (
            succeeded_receipt is None
            or succeeded_receipt.status is not EngineReceiptStatus.SUCCEEDED
        ):
            raise RuntimeError(f"Attempt-002 failed: {succeeded_receipt}")
        lifecycle.collect_engine(run_id, attempt_two.command_id)
        accepted = repository.load(run_id)
        trial = accepted.trials[0]
        if (
            accepted.status is not RunStatus.RUNNING
            or trial.phase is not TrialPhase.EVIDENCE_READY
            or trial.engine_attempt_index != 2
            or len(trial.engine_attempt_failure_ref_ids) != 1
        ):
            raise RuntimeError("Attempt-002 success was not accepted exactly once")
        result = process.objects.read_json(f"{attempt_two.output_uri}/result.json")
        acceptance = _validate_result("rft", result)
        receipt["attempt_002"] = {
            "command_id": attempt_two.command_id,
            "logical_command_id": attempt_two.logical_command_id,
            "attempt_id": attempt_two.attempt_id,
            "receipt": succeeded_receipt.to_dict(),
            "accepted_revision": accepted.revision,
            "trial_phase": trial.phase.value,
            "acceptance": acceptance,
        }
        receipt["scientific_fence"] = {
            "failed_attempt_in_ranking": False,
            "failed_attempt_changed_memory": False,
            "accepted_engine_attempt_index": trial.engine_attempt_index,
        }
        receipt["status"] = "passed"
        print(
            json.dumps(
                {
                    "phase": "attempt_002_accepted",
                    "revision": accepted.revision,
                    "checkpoint": acceptance["checkpoint"],
                }
            ),
            flush=True,
        )
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        if admission is not None and run_created and repository is not None:
            try:
                state = repository.load(run_id)
                if state.status not in {
                    RunStatus.COMPLETED,
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                }:
                    StateCoordinator(repository).cancel(
                        run_id, reason="R1 recovery probe terminal cleanup"
                    )
            except Exception as cleanup_error:
                receipt["run_cleanup_error"] = (
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        elif admission is not None:
            try:
                admission.terminal(run_id)
            except Exception as cleanup_error:
                receipt["run_cleanup_error"] = (
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        if admission is not None and not replacement_ready:
            try:
                resources = compiled.run.run_resources
                assert resources is not None
                admission.admit(
                    run_id=run_id,
                    binding=binding,
                )
                admission.terminal(run_id)
                receipt["judge_restored_after_failure"] = True
            except Exception as restore_error:
                receipt["judge_restore_error"] = (
                    f"{type(restore_error).__name__}: {restore_error}"
                )
        receipt["finished_at"] = time.time()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "phase": "finished",
                    "status": receipt["status"],
                    "receipt": str(receipt_path),
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
