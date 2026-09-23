#!/usr/bin/env python3
"""Run one real SFT or RFT Engine backend calibration slice."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time
import uuid

from ade.tasks.reward_design.engine_binding import bind_run_local_judge
from ade.core.engine import EngineReceiptStatus
from ade.engine.storage.atomic import write_json_atomic
from ade.engine.storage.object_store import FileEngineObjectStore
from ade.harness.environment import load_project_environment
from ade.harness.experiment_config import ExperimentConfigCompiler
from ade.harness.wiring import build_engine_process
from ade.local_rubric_judge.launcher import ProductionJudgeLauncher
from ade.local_rubric_judge.lifecycle import LocalJudgeBinding, RunResourceAdmission
from ade.tasks.contracts import (
    ArtifactCompilationRequest,
    ArtifactDelivery,
    BaselineArtifactRequest,
    EngineArtifactBindingRequest,
    EngineCommandRequest,
    SupportingArtifact,
)
from ade.tasks.registry import default_task_registry


_RUBRIC = json.dumps(
    {
        "template": (
            "Judge only the logical validity of the response reasoning, independently "
            "of whether its final answer matches.\nQuestion: {{question}}\nResponse: {{response}}"
        ),
        "required_variables": ["question", "response"],
        "output_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scores_by_dimension": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "logical_validity": {"enum": [0.0, 0.25, 0.5, 0.75, 1.0]}
                    },
                    "required": ["logical_validity"],
                }
            },
            "required": ["scores_by_dimension"],
        },
        "projection": {
            "dimensions": [
                {
                    "id": "logical_validity",
                    "weight": 1.0,
                    "criterion": "The reasoning is logically valid and relevant.",
                    "score_levels": [
                        {"value": 0.0, "standard": "The reasoning is invalid."},
                        {"value": 0.25, "standard": "The reasoning is mostly invalid."},
                        {"value": 0.5, "standard": "The reasoning has material gaps."},
                        {"value": 0.75, "standard": "The reasoning is mostly valid."},
                        {"value": 1.0, "standard": "The reasoning is fully valid."},
                    ],
                }
            ]
        },
    },
    sort_keys=True,
)

_REWARD_SOURCE = (
    "component_weights = {'outcome': 0.6, 'process': 0.3, 'rule_based': 0.1}\n"
    "def _result(outcome, process, rule_based, judge_fallback):\n"
    "    components = {'outcome': outcome, 'process': process, 'rule_based': rule_based}\n"
    "    score = outcome if judge_fallback else sum(component_weights[name] * components[name] for name in ('outcome', 'process', 'rule_based'))\n"
    "    return {'score': score, 'components': components, 'component_weights': component_weights, 'judge_fallback': judge_fallback}\n"
    "def compute_fallback_score(question_prompt, response_content, extracted_answer, outcome_score, response_length_tokens, max_response_length_tokens):\n"
    "    return _result(outcome_score, None, 1.0 if extracted_answer else 0.0, True)\n"
    "async def compute_score(question_prompt, response_content, extracted_answer, outcome_score, response_length_tokens, max_response_length_tokens):\n"
    f"    process_rubric = {_RUBRIC!r}\n"
    "    process = await llm_judge(question_prompt, response_content, process_rubric)\n"
    "    return _result(outcome_score, process, 1.0 if extracted_answer else 0.0, False)\n"
)


def _reward_artifact(plugin, task_config):
    design = {
        "schema_version": "1",
        "hypothesis": "A process score supplies a real Judge-dependent calibration path.",
        "intervention": "Use one bounded logical-validity rubric.",
        "expected_observation": "One complete 64-row Judge job is consumed by one update.",
        "failure_signal": "The Judge job, reward projection, update, or export fails.",
        "controls_held_constant": ["model", "data", "rollout_n", "seed"],
        "implementation_notes": "Calibration changes only the number of updates.",
    }
    metadata = {
        "entrypoint": "reward.py:compute_score",
        "fallback_entrypoint": "reward.py:compute_fallback_score",
        "return_schema": "ade.reward_result.v1",
        "supported_inputs": [
            "question_prompt", "response_content", "extracted_answer",
            "outcome_score", "response_length_tokens", "max_response_length_tokens",
        ],
        "algorithm": "grpo",
        "score_bounds": [0.0, 1.0],
        "components": ["outcome", "process", "rule_based"],
        "component_weights": {"outcome": 0.6, "process": 0.3, "rule_based": 0.1},
        "process_evaluator": "llm_judge_mcp",
        "process_rubric": _RUBRIC,
        "process_failure_policy": "outcome_only",
        "judge_fallback": {
            "score_source": "outcome",
            "process": "unavailable",
            "rule_based": "evidence_only",
        },
        "primary_reference_artifact_ref_id": "calibration-baseline",
        "exploit_probes": ["empty-answer", "answer-copy"],
        "lineage": ["calibration/fixed-design.json"],
        "missing_input_policy": "return_zero",
        "non_finite_policy": "reject",
    }
    return plugin.compile_artifact(
        ArtifactCompilationRequest(
            delivery=ArtifactDelivery(
                schema_version="1",
                path="reward.py",
                kind="reward_design_proposal",
                content=_REWARD_SOURCE.encode(),
                metadata=metadata,
                payload=design,
                supporting_artifacts=(
                    SupportingArtifact(
                        "design.json",
                        "experiment_design",
                        (json.dumps(design, sort_keys=True) + "\n").encode(),
                    ),
                ),
            ),
            task_config=task_config,
            planning_decision={"design": {"score_bounds": [0.0, 1.0]}},
        )
    )


def _judge_admission(project: Path, compiled, run_id: str):
    resources = compiled.run.run_resources
    if resources is None:
        raise ValueError("Backend calibration requires deployment resources")
    ray_config = resources["ray_cluster"]
    judge = resources["local_judge"]
    binding = LocalJudgeBinding.from_dict(
        {
            "cluster_id": ray_config["cluster_id"],
            "ray_address": ray_config["address"],
            "gpu_count": judge["gpu_count"],
            "model_path": judge["model_path"],
            "model_digest": judge["model_digest"],
            "gateway_port": judge["gateway_port"],
            "protocol": judge["protocol"],
        }
    )
    deployment_root = project / "runs" / "deployments" / str(ray_config["cluster_id"])
    launcher = ProductionJudgeLauncher(
        deployment_root / "local-judge",
        authorization_env=str(judge["authorization_env"]),
        readiness_seconds=float(judge["timeout_policy"]["readiness_seconds"]),
        request_timeout_seconds=float(judge["timeout_policy"]["request_timeout_seconds"]),
        vllm=dict(judge["vllm"]),
        generation=dict(judge["generation"]),
    )
    admission = RunResourceAdmission(deployment_root / "run-services", launcher)
    ready = admission.admit(run_id=run_id, binding=binding)
    return admission, ready


def _scale_input(stage: str, engine_input: dict) -> dict:
    value = copy.deepcopy(engine_input)
    if stage == "sft":
        sft = value["sft"]
        steps = int(sft["steps_per_epoch"])
        sft["max_steps"] = steps
        sft["run_offline_validation"] = False
        request = sft["request"]
        request["num_train_epochs"] = 1
        request["max_steps"] = steps
        return value
    rft = value["rft"]
    rft["total_training_steps"] = 1
    rft["artifact_interval"] = 1
    rft["run_offline_validation"] = False
    verl = rft["verl_config"]
    for runtime in (verl["rft"], verl["train"]["rft"]):
        runtime["total_training_steps"] = 1
        runtime["artifact_interval"] = 1
    return value


def _validate_result(stage: str, result: dict) -> dict:
    if result.get("training_error"):
        raise RuntimeError(str(result["training_error"]))
    selected = result.get("selected_position")
    expected_unit = "epoch" if stage == "sft" else "rl_step"
    if selected != {"unit": expected_unit, "value": 1}:
        raise RuntimeError(f"calibration selected position is invalid: {selected}")
    checkpoint = result.get("model_ref") if stage == "sft" else result.get("checkpoint_ref")
    if not checkpoint or not Path(str(checkpoint)).is_dir():
        raise RuntimeError("calibration did not preserve a loadable checkpoint directory")
    online = result.get("online_validation")
    if not isinstance(online, dict) or online.get("status") != "complete":
        raise RuntimeError(f"calibration online validation is incomplete: {online}")
    payload = online.get("payload") if stage == "sft" else online.get("result")
    if not isinstance(payload, dict):
        raise RuntimeError("calibration online evaluation payload is unavailable")
    evaluation_results = payload.get("results")
    if not isinstance(evaluation_results, list) or not evaluation_results:
        raise RuntimeError("calibration online evaluation contains no dataset results")
    row_counts = []
    for evaluation_result in evaluation_results:
        metrics = (
            evaluation_result.get("metrics")
            if isinstance(evaluation_result, dict)
            else None
        )
        if not isinstance(metrics, dict) or not isinstance(metrics.get("num_total"), int):
            raise RuntimeError("calibration online evaluation is missing num_total")
        row_counts.append(metrics["num_total"])
    num_total = sum(row_counts)
    if num_total != 16:
        raise RuntimeError(f"calibration online evaluation expected 16 rows, got {num_total}")
    offline = result.get("offline_validation")
    if stage == "sft":
        if not isinstance(offline, dict) or offline.get("status") != "skipped":
            raise RuntimeError("calibration unexpectedly ran offline validation")
        offline_status = offline["status"]
    else:
        if offline is not None:
            raise RuntimeError("calibration unexpectedly ran offline validation")
        offline_status = "skipped"
    return {
        "checkpoint": str(Path(str(checkpoint)).resolve()),
        "online_score": online.get("ranking_score"),
        "online_rows": num_total,
        "offline_status": offline_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("sft", "rft"))
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--experiment")
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    load_project_environment(project)
    default_experiment = {
        "sft": "configs/experiments/openthoughts-math-sft-data-selection-ADE-formal-n1.yaml",
        "rft": "configs/experiments/math-math-rft-reward-design-ADE-formal-n1.yaml",
    }[args.stage]
    experiment = Path(args.experiment or project / default_experiment).resolve()
    run_id = args.run_id or (
        f"backend-calibration-{args.stage}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    root = project / "runs" / "backend-calibration" / run_id
    receipt_path = root / "receipts" / f"{args.stage}-backend-calibration.json"
    receipt: dict[str, object] = {
        "schema_version": 1,
        "stage": args.stage,
        "run_id": run_id,
        "experiment": str(experiment),
        "started_at": time.time(),
        "status": "running",
    }
    admission = None
    try:
        compiler = ExperimentConfigCompiler(default_task_registry(), project_root=project)
        compiled = compiler.compile_file(
            experiment,
            deployment_config=args.deployment,
            run_id=run_id,
        )
        task_id = "data_selection" if args.stage == "sft" else "reward_design"
        plugin = default_task_registry().get(task_id)
        engine_input = _scale_input(args.stage, compiled.engine_inputs[task_id])
        engine_input["usage_run_dir"] = str(root.resolve())
        admission, ready = _judge_admission(project, compiled, run_id)
        resources = copy.deepcopy(compiled.run.run_resources)
        resources["local_judge"].update(
            host=ready.host, gateway_url=ready.gateway_url, node_id=ready.node_id,
            launch_id=ready.launch_id, state_path=ready.state_path,
        )
        if args.stage == "rft":
            engine_input = bind_run_local_judge(engine_input, resources)
            artifact = _reward_artifact(plugin, compiled.run.task.config)
        else:
            engine_input["sft"]["request"]["local_judge"] = resources["local_judge"]
            artifact = plugin.build_baseline_artifact(
                BaselineArtifactRequest(
                    task_config=compiled.run.task.config,
                    seed=int(compiled.resolved["seed"]),
                )
            )
        receipt["judge_launch_id"] = ready.launch_id
        ray_address = (
            compiled.resolved.get("deployment", {})
            .get("run_resources", {})
            .get("ray_cluster", {})
            .get("address")
        )
        if ray_address:
            os.environ["RAY_ADDRESS"] = str(ray_address)
        command_id = f"{run_id}-c001-calibration-t001-train-{args.stage}"
        binding = plugin.bind_engine_artifact(
            EngineArtifactBindingRequest(
                binding_uri=f"engine://bindings/{command_id}",
                compiled_kind=artifact.kind,
                compiled_content=artifact.content,
                task_config=compiled.run.task.config,
                engine_config=engine_input,
                planning_decision={},
                is_baseline=True,
            )
        )
        process = build_engine_process(
            queue_root=root / "queue",
            object_root=root / "objects",
            work_root=root / "engine-work",
            engine_config=compiled.control["engine"],
        )
        for obj in binding.objects:
            process.objects.put_bytes(obj.uri, obj.content)
        input_ref = f"engine://inputs/{command_id}.json"
        output_uri = f"engine://outputs/{command_id}"
        process.objects.put_json(input_ref, dict(binding.input_payload))
        command = plugin.build_engine_command(
            EngineCommandRequest(
                command_id=command_id,
                run_id=run_id,
                coordinator_id="c001",
                plan_id="calibration",
                trial_id="t001",
                input_ref=input_ref,
                output_uri=output_uri,
            )
        )
        write_json_atomic(root / "resolved-engine-input.json", dict(binding.input_payload))
        process.queue.submit(command)
        print(json.dumps({"phase": "engine_started", "stage": args.stage, "run_id": run_id}), flush=True)
        engine_receipt = process.worker.run_once()
        if engine_receipt is not None:
            receipt["engine_receipt"] = engine_receipt.to_dict()
        if engine_receipt is None or engine_receipt.status is not EngineReceiptStatus.SUCCEEDED:
            raise RuntimeError(
                "Engine calibration failed: "
                + (engine_receipt.error if engine_receipt is not None else "no receipt")
            )
        result = process.objects.read_json(f"{output_uri}/result.json")
        receipt["acceptance"] = _validate_result(args.stage, result)
        receipt["result_uri"] = f"{output_uri}/result.json"
        receipt["status"] = "passed"
        print(json.dumps({"phase": "engine_passed", "stage": args.stage}), flush=True)
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if admission is not None:
            detached = admission.terminal(run_id)
            receipt["judge_detach"] = (
                None
                if detached is None
                else {"status": detached.status, "launch_id": detached.launch_id}
            )
        receipt["finished_at"] = time.time()
        try:
            import ray

            if ray.is_initialized():
                from ade.engine.execution.ray import cleanup_artifact_cache_on_gpu_nodes

                staging = (
                    compiled.resolved["deployment"]["run_resources"][
                        "artifact_staging"
                    ]
                    if "compiled" in locals()
                    else None
                )
                if isinstance(staging, dict) and staging.get("enabled"):
                    receipt["artifact_cache_cleanup"] = (
                        cleanup_artifact_cache_on_gpu_nodes(
                            cache_dir=str(staging["cache_dir"]), run_id=run_id
                        )
                    )
        except Exception as cleanup_error:
            receipt["artifact_cache_cleanup"] = {
                "status": "failed",
                "error": f"{type(cleanup_error).__name__}: {cleanup_error}",
            }
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(receipt_path, receipt)
        print(json.dumps({"phase": "finished", "status": receipt["status"], "receipt": str(receipt_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
