#!/usr/bin/env python3
"""Focused Curriculum Learning gates that do not launch external services."""

from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
from urllib.parse import urlparse
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ade.agent_runtime.experiment_package import (
    EngineExperimentPackageBuilder,
    attach_curriculum_realization,
    decode_experiment_package,
    encode_experiment_package,
)
from ade.agent_runtime.analysis_validation import validate_analysis_delivery
from ade.agent_runtime.backend import CodexBackend
from ade.agent_runtime.delivery import DeliveryGate
from ade.agent_runtime.input_package import AgentInputPackageBuilder
from ade.agent_runtime.runtime import AcceptedCall, AgentRuntime
from ade.agent_runtime.skills import SkillResolver
from ade.agent_runtime.workspace import WorkspaceManager
from ade.core.agent import AgentCall, AgentCallScope, AgentRole, AgentSession
from ade.engine.judge_dispatcher_impl import SelectionJudgeBatchDispatcher
from ade.engine.command_queue import FileCommandQueue
from ade.engine.protocol import encode_receipt
from ade.engine.storage.atomic import write_json_atomic
from ade.engine.storage.object_store import FileEngineObjectStore
from ade.engine.worker import EngineWorker
from ade.harness.environment import load_project_environment
from ade.harness.experiment_config import ExperimentConfigCompiler
from ade.harness.wiring import build_engine_process, build_review_process
from ade.local_rubric_judge.client import LocalRubricJudgeClient
from ade.local_rubric_judge.http_gateway import HttpRubricJobGateway
from ade.tasks.contracts import (
    AgentContextFile,
    AgentContextReference,
    AgentInputRequest,
    ArtifactCompilationRequest,
    ArtifactDelivery,
    EngineArtifactBindingRequest,
    EngineCommandRequest,
)
from ade.review_labor.planning import (
    compile_review_command,
    coverage_from_packet,
    validate_review_packet,
)
from ade.review_labor.protocol import ReviewReceiptStatus
from ade.tasks.curriculum_learning.fixed_pool import build_fixed_pool_from_task
from ade.tasks.curriculum_learning.materializer import materialize_scheduled_parquet
from ade.tasks.curriculum_learning.realizer import realize_curriculum
from ade.tasks.reward_design.contracts import RFTCheckpointResult, RFTTrainingResult
from ade.tasks.reward_design.handler import RFTExecutor
from ade.tasks.reward_design.rewards.baselines import baseline_math
from ade.tasks.registry import default_task_registry
from ade.core.engine import EngineReceiptStatus, TrainRFTCommand


_RUBRIC = json.dumps(
    {
        "template": "Question: {{question}}\nResponse: {{response}}",
        "required_variables": ["question", "response"],
        "output_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scores_by_dimension": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"difficulty": {"enum": [0.0, 0.5, 1.0]}},
                    "required": ["difficulty"],
                }
            },
            "required": ["scores_by_dimension"],
        },
        "projection": {
            "dimensions": [
                {
                    "id": "difficulty",
                    "criterion": "The problem requires substantial multi-step reasoning.",
                    "weight": 1.0,
                    "score_levels": [
                        {"value": 0.0, "standard": "direct or routine"},
                        {"value": 0.5, "standard": "moderate reasoning"},
                        {"value": 1.0, "standard": "substantial multi-step reasoning"},
                    ],
                }
            ]
        },
    },
    sort_keys=True,
)


def _artifact_source() -> bytes:
    return (
        "async def build_curriculum(candidate_inventory, total_steps, "
        "prompts_per_step, judge_batch):\n"
        f"    rubric = {_RUBRIC!r}\n"
        "    await judge_batch([{\"question\": candidate_inventory[0][\"question\"], "
        "\"response\": candidate_inventory[0][\"reference_answer\"], "
        "\"rubric\": rubric}])\n"
        "    ids = [row[\"problem_id\"] for row in candidate_inventory]\n"
        "    return [ids[index * prompts_per_step:(index + 1) * prompts_per_step] "
        "for index in range(total_steps)]\n"
    ).encode()


def _require_existing_gateway(url: str, timeout: float) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Local Judge gateway URL is invalid")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=min(timeout, 3.0)):
            return
    except OSError as error:
        raise RuntimeError(
            f"existing Local Judge gateway is not reachable at {parsed.hostname}:{port}; "
            "this probe will not launch it"
        ) from error


def _existing_judge(project: Path, resources: dict) -> dict:
    """Use the Ray-admitted service without starting or taking ownership of it."""
    state_path = (
        project / "runs" / "deployments"
        / resources["ray_cluster"]["cluster_id"] / "run-services" / "service.json"
    )
    service = json.loads(state_path.read_text())
    if service["status"] != "ready":
        raise RuntimeError("Curriculum probe requires an already admitted Local Judge")
    handle = service["service_handle"]
    return {
        **resources["local_judge"],
        "host": handle["host"],
        "gateway_url": handle["gateway_url"],
        "state_path": str(state_path),
    }


async def _run_cl2b(args: argparse.Namespace) -> dict[str, object]:
    project = Path(args.project_root).resolve()
    load_project_environment(project)
    deployment_path = Path(args.deployment)
    if not deployment_path.is_absolute():
        deployment_path = project / deployment_path
    deployment = yaml.safe_load(deployment_path.read_text(encoding="utf-8"))
    judge = _existing_judge(project, deployment["run_resources"])
    timeout = float(judge["timeout_policy"]["request_timeout_seconds"])
    _require_existing_gateway(str(judge["gateway_url"]), timeout)
    authorization = os.environ.get(str(judge["authorization_env"]))
    if not authorization:
        raise RuntimeError("configured Local Judge authorization is unavailable")

    experiment_path = Path(args.experiment)
    if not experiment_path.is_absolute():
        experiment_path = project / experiment_path
    run_id = args.run_id or f"cl2b-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    compiled = ExperimentConfigCompiler(
        default_task_registry(), project_root=project
    ).compile_file(
        experiment_path,
        deployment_config=args.deployment,
        run_id=run_id,
    )
    if compiled.run.task.task_id != "curriculum_learning":
        raise ValueError("CL2B requires a Curriculum Learning experiment")
    inventory, pool_stats = build_fixed_pool_from_task(compiled.run.task.config)
    plugin = default_task_registry().get("curriculum_learning")
    artifact = plugin.compile_artifact(
        ArtifactCompilationRequest(
            delivery=ArtifactDelivery(
                schema_version="1",
                path="curriculum.py",
                kind="curriculum_learning_proposal",
                content=_artifact_source(),
                metadata={},
            ),
            task_config=compiled.run.task.config,
            planning_decision={"design": {"judge_enrichment": True}},
        )
    )
    client = LocalRubricJudgeClient(
        HttpRubricJobGateway(
            str(judge["gateway_url"]),
            authorization=authorization,
            timeout_seconds=timeout,
        ),
        poll_interval_seconds=0.1,
    )
    dispatcher = SelectionJudgeBatchDispatcher(
        client,
        max_batch_size=8,
        submission_id=f"{run_id}:curriculum-realization",
        job_metadata={
            "run_id": run_id,
            "task_id": "curriculum_learning",
            "phase": "harness_validation",
            "operation": "curriculum_enrichment",
        },
    )
    try:
        realized = await realize_curriculum(
            artifact.content,
            inventory,
            total_steps=2,
            prompts_per_step=8,
            rollout_n=8,
            pool_stats=pool_stats,
            judge_batch=dispatcher.evaluate,
        )
    finally:
        await dispatcher.close()
    evidence = list(realized.judge_evidence)
    if (
        len(evidence) != 1
        or evidence[0]["evidence"].get("status") != "completed"
        or evidence[0]["evidence"].get("fallback") is not False
    ):
        raise RuntimeError("CL2B did not receive one complete non-fallback Judge result")
    parquet = materialize_scheduled_parquet(
        compiled.run.task.config["data"]["train"], realized.schedule
    )
    rows = pq.read_table(pa.BufferReader(parquet)).to_pylist()
    scheduled_ids = [
        problem_id
        for step in realized.schedule["steps"]
        for problem_id in step["problem_ids"]
    ]
    consumed_ids = [row["extra_info"]["sample_uid"] for row in rows]
    if consumed_ids != scheduled_ids:
        raise RuntimeError("CL2B scheduled parquet does not preserve the accepted schedule")
    return {
        "schema_version": "ade.curriculum_cl2b_probe.v1",
        "gate": "CL2B",
        "status": "passed",
        "run_id": run_id,
        "experiment": str(experiment_path),
        "deployment": str(deployment_path),
        "gateway_url": str(judge["gateway_url"]),
        "candidate_pool": pool_stats,
        "artifact": artifact.content.decode("utf-8"),
        "realization": {
            "schema_version": "ade.curriculum_realization.v1",
            "realization_status": "verified",
            "schedule": realized.schedule,
            "judge_binding": {
                "enabled": True,
                "owner": "curriculum_realization",
                "stats": dispatcher.stats,
            },
            "judge_evidence": evidence,
        },
        "scheduled_row_count": len(rows),
        "started_existing_service_only": True,
        "finished_at": time.time(),
    }


def _run_cl3(args: argparse.Namespace) -> dict[str, object]:
    project = Path(args.project_root).resolve()
    load_project_environment(project)
    experiment_path = Path(args.experiment)
    if not experiment_path.is_absolute():
        experiment_path = project / experiment_path
    run_id = args.run_id or (
        f"cl3-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    )
    probe_root = project / "runs" / "probes" / run_id
    compiled = ExperimentConfigCompiler(
        default_task_registry(), project_root=project
    ).compile_file(
        experiment_path,
        deployment_config=args.deployment,
        run_id=run_id,
    )
    if compiled.run.task.task_id != "curriculum_learning":
        raise ValueError("CL3 requires a Curriculum Learning experiment")
    baseline = compiled.bootstrap["p000"]["artifact"]
    binding = default_task_registry().get(
        "curriculum_learning"
    ).bind_engine_artifact(
        EngineArtifactBindingRequest(
            binding_uri=f"engine://bindings/{run_id}",
            compiled_kind=baseline["kind"],
            compiled_content=baseline["content"].encode("utf-8"),
            task_config=compiled.run.task.config,
            engine_config=compiled.engine_inputs["curriculum_learning"],
            is_baseline=True,
        )
    )
    objects = FileEngineObjectStore(probe_root / "engine-objects")
    for item in binding.objects:
        objects.put_bytes(item.uri, item.content)
    input_ref = f"engine://inputs/{run_id}.json"
    objects.put_json(input_ref, dict(binding.input_payload))
    command = TrainRFTCommand(
        command_id=f"{run_id}-train-rft",
        run_id=run_id,
        coordinator_id="c000",
        plan_id="p000",
        trial_id="p000-t000-baseline",
        input_ref=input_ref,
        output_uri=f"engine://outputs/{run_id}",
    )
    captured: dict[str, object] = {}

    class ScriptedBackend:
        def train(self, received, config, steps, evaluate_checkpoint):
            captured["command_id"] = received.command_id
            captured["config"] = config
            checkpoints = []
            for step in steps:
                checkpoint = probe_root / "scripted-checkpoints" / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                checkpoints.append(
                    RFTCheckpointResult(
                        step,
                        str(checkpoint),
                        evaluate_checkpoint(step, str(checkpoint)),
                    )
                )
            return RFTTrainingResult(tuple(checkpoints))

        def evaluate(self, received, config, checkpoint, purpose, step):
            return {
                "status": "complete",
                "score": float(step),
                "purpose": purpose,
                "checkpoint": checkpoint,
            }

    queue = FileCommandQueue(probe_root / "queue")
    queue.submit(command)
    receipt = EngineWorker(
        queue=queue,
        handlers={
            "train_rft": RFTExecutor(
                io=objects, backend=ScriptedBackend()
            ).execute
        },
    ).run_once()
    if receipt is None or receipt.status is not EngineReceiptStatus.SUCCEEDED:
        raise RuntimeError("CL3 scripted Engine command did not reach accepted receipt")
    if tuple(receipt.output_refs) != (
        f"engine://outputs/{run_id}/result.json",
        f"engine://outputs/{run_id}/raw/manifest.json",
    ) or not all(objects.exists(ref) for ref in receipt.output_refs):
        raise RuntimeError("CL3 accepted receipt does not reference complete outputs")
    config = captured.get("config")
    if config is None:
        raise RuntimeError("CL3 scripted backend did not receive localized Engine config")
    runtime = config.rft["verl_config"]["rft"]
    scheduled_path = Path(config.rft["verl_config"]["data"]["train"])
    scheduled_rows = pq.read_table(scheduled_path).to_pylist()
    schedule = objects.read_json(config.rft["curriculum_schedule_ref"])
    scheduled_ids = [
        problem_id
        for step in schedule["steps"]
        for problem_id in step["problem_ids"]
    ]
    consumed_ids = [row["extra_info"]["sample_uid"] for row in scheduled_rows]
    if consumed_ids != scheduled_ids:
        raise RuntimeError("CL3 localized training data differs from frozen schedule")
    if runtime.get("judge_enrichment") != {
        "enabled": False,
        "owner": "rft_reward",
    }:
        raise RuntimeError("CL3 reward-side Judge is not explicitly disabled")
    if config.rft.get("analysis_profile_id") != "curriculum_learning":
        raise RuntimeError("CL3 Curriculum analysis profile was not preserved")
    if config.rft.get("group_credit") != {
        "enabled": True,
        "entrypoint": "assign_group_credit",
        "schema_version": "ade.group_credit.v1",
    }:
        raise RuntimeError("CL3 identity Group Credit binding is invalid")
    reward_path = Path(runtime["reward_function_path"])
    reward_source = reward_path.read_text(encoding="utf-8")
    if (
        reward_path.read_bytes() != Path(baseline_math.__file__).read_bytes()
        or "def assign_group_credit" not in reward_source
        or '"mode": "identity"' not in reward_source
    ):
        raise RuntimeError("CL3 reward is not fixed baseline_math identity Group Credit")
    result = objects.read_json(receipt.output_refs[0])
    raw_manifest = objects.read_json(receipt.output_refs[1])
    if raw_manifest["analysis"]["profile"] != "curriculum_learning":
        raise RuntimeError("CL3 result lost the Curriculum analysis profile")
    return {
        "schema_version": "ade.curriculum_cl3_probe.v1",
        "gate": "CL3",
        "status": "passed",
        "run_id": run_id,
        "experiment": str(experiment_path),
        "engine_receipt": encode_receipt(receipt),
        "binding": {
            "artifact_ref": config.artifact_ref,
            "curriculum_schedule_ref": config.rft["curriculum_schedule_ref"],
            "scheduled_training_data_ref": config.rft[
                "scheduled_training_data_ref"
            ],
            "scheduled_row_count": len(scheduled_rows),
            "analysis_profile_id": config.rft["analysis_profile_id"],
            "group_credit": config.rft["group_credit"],
            "reward_side_judge": runtime["judge_enrichment"],
            "reward_function_path": str(reward_path),
        },
        "result": result,
        "analysis": raw_manifest["analysis"],
        "backend": "scripted",
        "gpu_consumed": False,
        "finished_at": time.time(),
    }


def _load_cl2b_realization(
    project: Path, receipt_value: str
) -> tuple[dict[str, object], dict[str, object]]:
    path = Path(receipt_value)
    if not path.is_absolute():
        path = project / path
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("CL6 requires a readable CL2B receipt") from error
    realization = receipt.get("realization") if isinstance(receipt, dict) else None
    artifact = receipt.get("artifact") if isinstance(receipt, dict) else None
    evidence = (
        realization.get("judge_evidence")
        if isinstance(realization, dict)
        else None
    )
    if (
        receipt.get("schema_version") != "ade.curriculum_cl2b_probe.v1"
        or receipt.get("gate") != "CL2B"
        or receipt.get("status") != "passed"
        or not isinstance(realization, dict)
        or realization.get("schema_version") != "ade.curriculum_realization.v1"
        or realization.get("realization_status") != "verified"
        or not isinstance(realization.get("schedule"), dict)
        or not isinstance(artifact, str)
        or not artifact.strip()
        or not isinstance(evidence, list)
        or not evidence
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("evidence"), dict)
            or item["evidence"].get("status") != "completed"
            or item["evidence"].get("fallback") is not False
            for item in evidence
        )
    ):
        raise ValueError("CL6 requires one accepted non-fallback CL2B realization")
    report = {
        "kind": "curriculum_realization",
        "schema_version": "ade.curriculum_realization.v1",
        "realization_status": "verified",
        "reason": None,
        "policy_source": artifact,
        "schedule": realization["schedule"],
        "judge_binding": realization.get("judge_binding"),
        "judge_evidence": evidence,
    }
    return receipt, report


def _ray_capacity(address: str, *, ray_executable: Path) -> dict[str, object]:
    result = subprocess.run(
        [str(ray_executable), "status", f"--address={address}"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Ray status query failed")
    import re

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?) GPU", result.stdout)
    if match is None:
        raise RuntimeError("Ray status did not report GPU usage")
    used = float(match.group(1))
    total = float(match.group(2))
    return {"used_gpus": used, "total_gpus": total, "free_gpus": total - used}


@contextmanager
def _wandb_proxy_isolation():
    names = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _direct_wandb_viewer(*, base_url: str, api_key: str) -> str:
    endpoint = base_url.rstrip("/") + "/graphql"
    authorization = base64.b64encode(f"api:{api_key}".encode()).decode("ascii")
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(
            {"query": "query GetDefaultEntity { viewer { id entity } }"}
        ).encode(),
        headers={
            "Authorization": f"Basic {authorization}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())
    viewer = (payload.get("data") or {}).get("viewer")
    if (
        payload.get("errors")
        or not isinstance(viewer, dict)
        or not str(viewer.get("id") or "").strip()
        or not str(viewer.get("entity") or "").strip()
    ):
        raise RuntimeError("authenticated W&B viewer response is incomplete")
    return str(viewer["entity"]).strip()


def _wandb_preflight() -> dict[str, object]:
    base_url = str(os.environ.get("WANDB_BASE_URL") or "https://api.wandb.ai").strip()
    api_key = str(os.environ.get("WANDB_API_KEY") or "").strip()
    identity = ""
    transport = "sdk"
    sdk_error: Exception | None = None
    with _wandb_proxy_isolation():
        try:
            import wandb

            viewer = wandb.Api(timeout=30).viewer
            identity = str(
                getattr(viewer, "entity", None)
                or getattr(viewer, "username", None)
                or ""
            ).strip()
        except Exception as error:
            sdk_error = error
        if not identity:
            if not api_key:
                error_name = type(sdk_error).__name__ if sdk_error else "empty_identity"
                raise RuntimeError(
                    f"W&B read-only identity preflight failed: {error_name}"
                ) from sdk_error
            try:
                identity = _direct_wandb_viewer(base_url=base_url, api_key=api_key)
                transport = "graphql_fallback"
            except Exception as error:
                sdk_name = type(sdk_error).__name__ if sdk_error else "empty_identity"
                raise RuntimeError(
                    "W&B read-only identity preflight failed: "
                    f"sdk={sdk_name}, direct={type(error).__name__}"
                ) from error
    return {
        "reachable": True,
        "identity_resolved": True,
        "identity_transport": transport,
    }


def _cl6_preflight(
    *,
    project: Path,
    compiled,
    cl2b_receipt: dict[str, object],
) -> dict[str, object]:
    if compiled.run.task.task_id != "curriculum_learning":
        raise ValueError("CL6 requires a Curriculum Learning experiment")
    resources = compiled.run.run_resources
    if not isinstance(resources, dict):
        raise ValueError("CL6 requires resolved deployment resources")
    ray_config = resources.get("ray_cluster")
    judge = _existing_judge(project, resources)
    if not isinstance(ray_config, dict) or not isinstance(judge, dict):
        raise ValueError("CL6 Ray/Judge deployment resources are incomplete")
    gateway_url = str(judge.get("gateway_url") or "")
    if cl2b_receipt.get("gateway_url") != gateway_url:
        raise ValueError("CL6 deployment does not match the accepted CL2B Judge binding")
    authorization_env = str(judge.get("authorization_env") or "")
    if not authorization_env or not os.environ.get(authorization_env):
        raise RuntimeError("CL6 deployment Judge authorization is unavailable")
    _require_existing_gateway(
        gateway_url,
        float(judge["timeout_policy"]["request_timeout_seconds"]),
    )
    ray_address = str(ray_config.get("address") or "")
    if ":" not in ray_address:
        raise ValueError("CL6 Ray address is invalid")
    ray_host, ray_port = ray_address.rsplit(":", 1)
    try:
        with socket.create_connection((ray_host, int(ray_port)), timeout=3.0):
            pass
    except OSError as error:
        raise RuntimeError("CL6 Ray control endpoint is unavailable") from error
    rft = compiled.engine_inputs["curriculum_learning"]["rft"]
    runtime_rft = rft["verl_config"]["rft"]
    if (
        rft.get("total_training_steps") != 2
        or runtime_rft.get("total_training_steps") != 2
        or runtime_rft.get("train_batch_size") != 8
        or runtime_rft.get("gen_batch_size") != 8
        or runtime_rft.get("rollout_n") != 8
        or runtime_rft.get("data_shuffle") is not False
        or runtime_rft.get("judge_enrichment")
        != {"enabled": False, "owner": "rft_reward"}
    ):
        raise ValueError("CL6 diagnostic RFT shape or fixed controls are invalid")
    base_model = Path(rft["verl_config"]["base_model"])
    train_source = Path(compiled.run.task.config["data"]["train"])
    judge_model = Path(str(judge.get("model_path") or ""))
    for label, path in (
        ("base model", base_model),
        ("training data", train_source),
        ("Judge model", judge_model),
    ):
        if not path.exists():
            raise RuntimeError(f"CL6 {label} path is unavailable: {path}")
    ray_executable = project / ".unified-vllm-0.19.1-verl-venv/bin/ray"
    capacity = _ray_capacity(ray_address, ray_executable=ray_executable)
    required = int(compiled.resolved["runtime"]["coordinator_capacity_gpus"])
    if capacity["free_gpus"] < required:
        raise RuntimeError(
            f"CL6 requires {required} free GPUs, observed {capacity['free_gpus']}"
        )
    if ray_config.get("exclusive_ade_run") is True and capacity["used_gpus"] != 0:
        raise RuntimeError(
            "CL6 exclusive Ray cluster currently has active GPU placement groups"
        )
    wandb_status = _wandb_preflight()
    return {
        "cluster_id": ray_config.get("cluster_id"),
        "ray_address": ray_address,
        "capacity": capacity,
        "required_free_gpus": required,
        "gateway_url": gateway_url,
        "paths": {
            "base_model": str(base_model.resolve()),
            "training_data": str(train_source.resolve()),
            "judge_model": str(judge_model.resolve()),
        },
        "wandb": wandb_status,
    }


def _validate_cl6_package(
    package: dict[str, bytes], realization: dict[str, object]
) -> dict[str, object]:
    catalog = json.loads(package["experiment/evidence-catalog.json"])
    pools = {
        item["pool_id"]: item
        for item in catalog.get("pools", ())
        if isinstance(item, dict) and item.get("pool_id")
    }
    rollout = pools.get("training_rollout")
    offline = pools.get("offline_validation")
    if not isinstance(rollout, dict) or not isinstance(offline, dict):
        raise RuntimeError("CL6 Experiment Package lacks RFT evidence pools")
    schedule = realization["schedule"]
    expected_by_step = {
        int(step["step"]): set(step["problem_ids"])
        for step in schedule["steps"]
    }
    observed_by_step: dict[int, dict[str, int]] = {}
    for record in rollout.get("records", ()):
        context = record.get("context") if isinstance(record, dict) else None
        position = record.get("position") if isinstance(record, dict) else None
        sample_uid = context.get("sample_uid") if isinstance(context, dict) else None
        if not isinstance(position, int) or not isinstance(sample_uid, str):
            raise RuntimeError("CL6 rollout evidence lacks schedule join fields")
        counts = observed_by_step.setdefault(position, {})
        counts[sample_uid] = counts.get(sample_uid, 0) + 1
    if set(observed_by_step) != set(expected_by_step):
        raise RuntimeError("CL6 rollout evidence positions differ from schedule")
    rollout_n = int(schedule["rollout_n"])
    for step, expected_ids in expected_by_step.items():
        counts = observed_by_step[step]
        if set(counts) != expected_ids or set(counts.values()) != {rollout_n}:
            raise RuntimeError(
                f"CL6 step {step} did not consume every scheduled problem group"
            )
    if int(offline.get("eligible_record_count", 0)) < 1:
        raise RuntimeError("CL6 offline validation evidence is empty")
    if any("online-validation" in path for path in package):
        raise RuntimeError("CL6 Analyzer package exposes online validation evidence")
    return {
        "training_rollout_groups": sum(len(item) for item in expected_by_step.values()),
        "training_rollout_records": int(rollout["eligible_record_count"]),
        "offline_validation_records": int(offline["eligible_record_count"]),
        "schedule_steps": sorted(expected_by_step),
    }


def _run_cl6(args: argparse.Namespace, *, execute: bool) -> dict[str, object]:
    project = Path(args.project_root).resolve()
    load_project_environment(project)
    experiment_path = Path(args.experiment)
    if not experiment_path.is_absolute():
        experiment_path = project / experiment_path
    run_id = args.run_id or (
        f"cl6-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    )
    compiled = ExperimentConfigCompiler(
        default_task_registry(), project_root=project
    ).compile_file(
        experiment_path,
        deployment_config=args.deployment,
        run_id=run_id,
    )
    cl2b_receipt, final_realization = _load_cl2b_realization(
        project, args.realization_receipt
    )
    preflight = _cl6_preflight(
        project=project,
        compiled=compiled,
        cl2b_receipt=cl2b_receipt,
    )
    if not execute:
        return {
            "schema_version": "ade.curriculum_cl6_preflight.v1",
            "gate": "CL6",
            "status": "passed",
            "run_id": run_id,
            "experiment": str(experiment_path),
            "preflight": preflight,
            "gpu_consumed": False,
            "finished_at": time.time(),
        }

    root = project / "runs" / "probes" / run_id
    plugin = default_task_registry().get("curriculum_learning")
    engine_input = json.loads(
        json.dumps(compiled.engine_inputs["curriculum_learning"])
    )
    engine_input["usage_run_dir"] = str(root.resolve())
    artifact_source = str(final_realization["policy_source"]).encode()
    logical_command_id = f"{run_id}-c001-cl6-t001-train-rft"
    attempt_index = int(args.attempt_index)
    if attempt_index < 1:
        raise ValueError("CL6 Engine attempt index must be positive")
    attempt_id = f"attempt-{attempt_index:03d}"
    command_id = f"{logical_command_id}-{attempt_id}"
    binding = plugin.bind_engine_artifact(
        EngineArtifactBindingRequest(
            binding_uri=f"engine://bindings/{command_id}",
            compiled_kind="curriculum_learning",
            compiled_content=artifact_source,
            task_config=compiled.run.task.config,
            engine_config=engine_input,
            planning_decision={"design": {"judge_enrichment": True}},
            final_realization=final_realization,
            is_baseline=False,
        )
    )
    os.environ["RAY_ADDRESS"] = str(preflight["ray_address"])
    process = build_engine_process(
        queue_root=root / "queue",
        object_root=root / "objects",
        work_root=root / "engine-work",
        engine_config=compiled.control["engine"],
    )
    for item in binding.objects:
        process.objects.put_bytes(item.uri, item.content)
    input_ref = f"engine://inputs/{command_id}.json"
    output_uri = f"engine://outputs/{command_id}"
    process.objects.put_json(input_ref, dict(binding.input_payload))
    command = plugin.build_engine_command(
        EngineCommandRequest(
            command_id=command_id,
            run_id=run_id,
            coordinator_id="c001",
            plan_id="cl6",
            trial_id="t001",
            input_ref=input_ref,
            output_uri=output_uri,
            logical_command_id=logical_command_id,
            attempt_id=attempt_id,
            attempt_index=attempt_index,
        )
    )
    process.queue.submit(command)
    engine_receipt = (
        process.queue.load_receipt(command_id)
        if process.queue.has_receipt(command_id)
        else process.worker.run_once()
    )
    if engine_receipt is None or engine_receipt.status is not EngineReceiptStatus.SUCCEEDED:
        raise RuntimeError(
            "CL6 Engine command failed: "
            + (engine_receipt.error if engine_receipt is not None else "no receipt")
        )
    result_ref, manifest_ref = engine_receipt.output_refs[:2]
    result = process.objects.read_json(result_ref)
    selected_position = result.get("selected_position")
    scheduled_steps = {
        int(step["step"])
        for step in final_realization["schedule"]["steps"]
    }


def _analyzer_action(
    *, run_id: str, stage: str, action_id: str
) -> dict[str, object]:
    owner_scope = {
        "run_id": run_id,
        "coordinator_id": "c001",
        "plan_id": "cl6",
    }
    target_scope = {**owner_scope, "trial_id": "t001"}
    return {
        "kind": "analyze_trial",
        "action_id": action_id,
        "stage": stage,
        "scope": target_scope,
        "subject_ref": f"{run_id}/c001/cl6/t001",
        "owner_scope": owner_scope,
        "owner_subject_ref": f"{run_id}/c001/cl6",
        "target_scope": target_scope,
        "target_subject_ref": f"{run_id}/c001/cl6/t001",
    }


def _analyzer_package(
    *,
    plugin,
    run_id: str,
    stage: str,
    action_id: str,
    package: dict[str, bytes],
    package_ref: str,
):
    manifest = json.loads(package["experiment/manifest.json"])
    references = []
    for artifact in manifest.get("artifacts", ()):
        storage = artifact.get("storage") if isinstance(artifact, dict) else None
        path = artifact.get("path") if isinstance(artifact, dict) else None
        if (
            isinstance(storage, dict)
            and storage.get("mode") == "durable_reference"
            and isinstance(path, str)
        ):
            references.append(
                AgentContextReference(
                    path=path,
                    source_ref=str(storage["uri"]),
                    sha256=str(storage["sha256"]),
                    size_bytes=int(storage["size_bytes"]),
                )
            )
    request = AgentInputRequest(
        role=AgentRole.ANALYZER,
        run_id=run_id,
        subject_id="t001",
        basis_revision=1,
        action=_analyzer_action(
            run_id=run_id, stage=stage, action_id=action_id
        ),
        task={"task_id": "curriculum_learning", "domain": "grpo_rft"},
        context_files=tuple(
            AgentContextFile(path, content, package_ref)
            for path, content in sorted(package.items())
        ),
        context_references=tuple(sorted(references, key=lambda item: item.path)),
    )
    contract = plugin.role_contract(AgentRole.ANALYZER)
    return AgentInputPackageBuilder().build(
        plugin.build_agent_input(request), contract
    )


def _analyzer_call(
    *, run_id: str, call_id: str, session_id: str, skill_id: str
) -> AgentCall:
    return AgentCall(
        call_id=call_id,
        run_id=run_id,
        role=AgentRole.ANALYZER,
        skill_id=skill_id,
        subject_id="t001",
        basis_revision=1,
        max_retries=3,
        target_subject_ref=f"{run_id}/c001/cl6/t001",
        session_id=session_id,
        scope=AgentCallScope.TRIAL,
        coordinator_id="c001",
        plan_id="cl6",
        trial_id="t001",
    )


def _run_cl7(args: argparse.Namespace) -> dict[str, object]:
    project = Path(args.project_root).resolve()
    load_project_environment(project)
    cl6_receipt = json.loads(Path(args.cl6_receipt).read_text(encoding="utf-8"))
    if cl6_receipt.get("status") != "passed" or cl6_receipt.get("gate") != "CL6":
        raise ValueError("CL7 requires a passing CL6 receipt")
    run_id = str(cl6_receipt["run_id"])
    package_path = Path(str(cl6_receipt["experiment_package"])).resolve()
    package = decode_experiment_package(package_path.read_bytes())
    manifest = json.loads(package["experiment/manifest.json"])
    expected_scope = {
        "run_id": run_id,
        "coordinator_id": "c001",
        "plan_id": "cl6",
        "trial_id": "t001",
    }
    if any(manifest.get(key) != value for key, value in expected_scope.items()):
        raise ValueError("CL7 Experiment Package scope does not match CL6")

    root = project / "runs" / "probes" / run_id
    plugin = default_task_registry().get("curriculum_learning")
    contract = plugin.role_contract(AgentRole.ANALYZER)
    workspaces = WorkspaceManager(
        root / "cl7-agent-calls",
        trial_artifacts_root=root / "engine-work" / "trial_artifacts",
    )
    runtime = AgentRuntime(
        skills=SkillResolver(project / ".agents" / "skills"),
        workspaces=workspaces,
        backend=CodexBackend(),
    )
    session_id = f"{run_id}-c001-cl6-analyzer"
    session = AgentSession(
        session_id=session_id,
        run_id=run_id,
        role=AgentRole.ANALYZER,
        subject_id="cl6",
        coordinator_id="c001",
        plan_id="cl6",
    )
    package_ref = str(cl6_receipt["experiment_package_ref"])
    attempt_index = int(args.attempt_index)
    if attempt_index < 1:
        raise ValueError("CL7 attempt index must be positive")
    attempt_id = f"attempt-{attempt_index:03d}"

    design_call_id = f"{run_id}-cl7-analyzer-review-design-{attempt_id}"
    design_call = _analyzer_call(
        run_id=run_id,
        call_id=design_call_id,
        session_id=session_id,
        skill_id=contract.skill_id,
    )
    workspaces.ensure_session(session, design_call)
    design_package = _analyzer_package(
        plugin=plugin,
        run_id=run_id,
        stage="review_design",
        action_id=design_call_id,
        package=package,
        package_ref=package_ref,
    )
    design_gate = DeliveryGate(allowed_paths={"review-plan.json"})
    design = (
        runtime.recover(design_call, design_package, design_gate, contract)
        if workspaces.call_dir(design_call).exists()
        else runtime.run(design_call, design_package, design_gate, contract)
    )
    if not isinstance(design, AcceptedCall):
        raise RuntimeError(
            "CL7 Analyzer review design was rejected: "
            + "; ".join(item.message for item in design.validation.violations)
        )

    review_command_id = f"{run_id}-cl7-review-{attempt_id}"
    review_command = compile_review_command(
        plan=design.output,
        attempt=design.workspace,
        command_id=review_command_id,
        logical_command_id=f"{run_id}-cl7-review",
        attempt_id=attempt_id,
        attempt_index=attempt_index,
        run_id=run_id,
        coordinator_id="c001",
        plan_id="cl6",
        trial_id="t001",
        basis_revision=1,
    )
    analysis_config = yaml.safe_load(
        (project / "configs/analysis/canonical-pre-gpu.yaml").read_text(
            encoding="utf-8"
        )
    )["analysis"]["provider"]
    deployment = yaml.safe_load(
        (project / args.deployment).read_text(encoding="utf-8")
    )
    local_judge = _existing_judge(project, deployment["run_resources"])
    review = build_review_process(
        project_root=project,
        queue_root=root / "cl7-review" / "queue",
        work_root=root / "cl7-review" / "work",
        provider_config=analysis_config,
        local_judge_config=local_judge,
        coordinator_id="c001",
    )
    try:
        review.queue.submit(review_command)
        review_receipt = (
            review.queue.load_receipt(review_command_id)
            if review.queue.has_receipt(review_command_id)
            else review.worker.run_once()
        )
    finally:
        review.close()
    if review_receipt is None or review_receipt.status not in {
        ReviewReceiptStatus.COMPLETED,
        ReviewReceiptStatus.COMPLETED_WITH_ERRORS,
    }:
        raise RuntimeError(
            "CL7 Review did not complete: "
            + (
                str(review_receipt.error or review_receipt.status.value)
                if review_receipt is not None
                else "no receipt"
            )
        )
    validate_review_packet(review_command, review_receipt.packet)
    coverage = coverage_from_packet(review_command, review_receipt.packet)
    if coverage.get("passed") is not True:
        raise RuntimeError("CL7 Review coverage was not accepted")

    synthesis_call_id = f"{run_id}-cl7-analyzer-synthesis-{attempt_id}"
    synthesis_call = _analyzer_call(
        run_id=run_id,
        call_id=synthesis_call_id,
        session_id=session_id,
        skill_id=contract.skill_id,
    )
    workspaces.ensure_session(session, synthesis_call)
    synthesis_package = _analyzer_package(
        plugin=plugin,
        run_id=run_id,
        stage="synthesis",
        action_id=synthesis_call_id,
        package=package,
        package_ref=package_ref,
    ).with_materialized_overlay(
        {
            "review/packet.json": (
                (json.dumps(review_receipt.packet, indent=2, sort_keys=True) + "\n").encode(),
                f"review://{review_command_id}/packet",
            ),
            "review/coverage.json": (
                (json.dumps(coverage, indent=2, sort_keys=True) + "\n").encode(),
                f"review://{review_command_id}/coverage",
            ),
        }
    )
    synthesis_gate = DeliveryGate(allowed_paths={"analysis.md", "findings.md"})
    synthesis = (
        runtime.recover(synthesis_call, synthesis_package, synthesis_gate, contract)
        if workspaces.call_dir(synthesis_call).exists()
        else runtime.run(synthesis_call, synthesis_package, synthesis_gate, contract)
    )
    if not isinstance(synthesis, AcceptedCall):
        raise RuntimeError(
            "CL7 Analyzer synthesis was rejected: "
            + "; ".join(item.message for item in synthesis.validation.violations)
        )
    authorized_ids = tuple(
        str(item["id"])
        for item in manifest.get("artifacts", ())
        if isinstance(item, dict) and item.get("id")
    )
    admission = validate_analysis_delivery(
        synthesis.output,
        synthesis.workspace,
        task_id="curriculum_learning",
        authorized_ids=authorized_ids,
    )
    if not admission.ok:
        synthesis = runtime.retry(
            synthesis_call,
            synthesis_package,
            synthesis_gate,
            contract,
            synthesis,
            admission,
        )
        if not isinstance(synthesis, AcceptedCall):
            raise RuntimeError("CL7 Analyzer evidence repair was rejected")
        admission = validate_analysis_delivery(
            synthesis.output,
            synthesis.workspace,
            task_id="curriculum_learning",
            authorized_ids=authorized_ids,
        )
    if not admission.ok:
        raise RuntimeError(
            "CL7 Analyzer evidence admission failed: "
            + "; ".join(item.message for item in admission.violations)
        )
    return {
        "schema_version": "ade.curriculum_cl7_probe.v1",
        "gate": "CL7",
        "status": "passed",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "attempt_index": attempt_index,
        "cl6_receipt": str(Path(args.cl6_receipt).resolve()),
        "experiment_package": str(package_path),
        "review_design_workspace": str(design.workspace),
        "review_receipt": review_receipt.to_dict(),
        "review_coverage": coverage,
        "analysis_workspace": str(synthesis.workspace),
        "analysis": synthesis.output.content.decode("utf-8"),
        "findings": synthesis.output.findings_content.decode("utf-8"),
        "gpu_consumed": False,
        "finished_at": time.time(),
    }
    if (
        not isinstance(selected_position, dict)
        or selected_position.get("unit") != "rl_step"
        or selected_position.get("value") not in scheduled_steps
    ):
        raise RuntimeError("CL6 selected checkpoint is outside the frozen schedule")
    checkpoint = Path(str(result.get("checkpoint_ref") or ""))
    if not checkpoint.is_dir():
        raise RuntimeError("CL6 did not preserve a checkpoint directory")
    offline = result.get("offline_validation")
    if not isinstance(offline, dict) or offline.get("status") != "complete":
        raise RuntimeError("CL6 offline validation did not complete")
    package = EngineExperimentPackageBuilder(process.objects).build(
        manifest_ref,
        result_ref=result_ref,
        task_id=plugin.task_id,
        evidence_specs=plugin.analyzer_evidence_specs,
    )
    realization_content = (
        json.dumps(final_realization, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode()
    package = attach_curriculum_realization(package, realization_content)
    package_acceptance = _validate_cl6_package(package, final_realization)
    package_uri = f"{output_uri}/cl6-experiment-package.zip"
    process.objects.put_bytes(package_uri, encode_experiment_package(package))
    package_path = process.objects.path_for(package_uri)
    return {
        "schema_version": "ade.curriculum_cl6_probe.v1",
        "gate": "CL6",
        "status": "passed",
        "run_id": run_id,
        "experiment": str(experiment_path),
        "preflight": preflight,
        "engine_receipt": engine_receipt.to_dict(),
        "logical_command_id": logical_command_id,
        "attempt_id": attempt_id,
        "attempt_index": attempt_index,
        "result_ref": result_ref,
        "manifest_ref": manifest_ref,
        "checkpoint": str(checkpoint.resolve()),
        "experiment_package_ref": package_uri,
        "experiment_package": str(package_path.resolve()),
        "package_acceptance": package_acceptance,
        "gpu_consumed": True,
        "finished_at": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="gate", required=True)
    cl2b = subparsers.add_parser("cl2b")
    cl2b.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    cl2b.add_argument(
        "--experiment",
        default="configs/experiments/math-math-rft-curriculum-learning-ADE-formal-n1.yaml",
    )
    cl2b.add_argument("--deployment", required=True)
    cl2b.add_argument("--run-id")
    cl2b.add_argument("--receipt")
    cl3 = subparsers.add_parser("cl3")
    cl3.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    cl3.add_argument(
        "--experiment",
        default="configs/experiments/math-math-rft-curriculum-learning-ADE-formal-n1.yaml",
    )
    cl3.add_argument("--deployment", required=True)
    cl3.add_argument("--run-id")
    cl3.add_argument("--receipt")
    for gate in ("cl6-preflight", "cl6"):
        command = subparsers.add_parser(gate)
        command.add_argument(
            "--project-root", default=str(Path(__file__).resolve().parents[1])
        )
        command.add_argument(
            "--experiment",
            default="configs/experiments/math-math-rft-curriculum-learning-ADE-formal-n1.yaml",
        )
        command.add_argument("--deployment", required=True)
        command.add_argument("--realization-receipt", required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--attempt-index", type=int, default=1)
        command.add_argument("--receipt")
    cl7 = subparsers.add_parser("cl7")
    cl7.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    cl7.add_argument("--cl6-receipt", required=True)
    cl7.add_argument("--deployment", required=True)
    cl7.add_argument("--attempt-index", type=int, default=1)
    cl7.add_argument("--receipt")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    receipt_path = Path(args.receipt).resolve() if args.receipt else None
    try:
        if args.gate == "cl2b":
            receipt = asyncio.run(_run_cl2b(args))
        elif args.gate == "cl3":
            receipt = _run_cl3(args)
        elif args.gate == "cl7":
            receipt = _run_cl7(args)
        else:
            receipt = _run_cl6(args, execute=args.gate == "cl6")
    except BaseException as error:
        if args.gate.startswith("cl6") or args.gate == "cl7":
            failure_run_id = getattr(args, "run_id", None)
            if failure_run_id is None and args.gate == "cl7":
                try:
                    failure_run_id = json.loads(
                        Path(args.cl6_receipt).read_text(encoding="utf-8")
                    )["run_id"]
                except (OSError, KeyError, json.JSONDecodeError):
                    failure_run_id = "cl7-unknown-run"
            receipt_path = receipt_path or (
                project
                / "runs"
                / "probes"
                / failure_run_id
                / f"{args.gate}-receipt.json"
            )
            write_json_atomic(
                receipt_path,
                {
                    "schema_version": "ade.curriculum_cl6_failure.v1",
                    "gate": "CL6",
                    "probe": args.gate,
                    "status": "failed",
                    "run_id": failure_run_id,
                    "experiment": getattr(args, "experiment", None),
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "gpu_consumption_status": (
                        "possible" if args.gate == "cl6" else "none"
                    ),
                    "finished_at": time.time(),
                },
            )
            print(
                json.dumps(
                    {"status": "failed", "receipt": str(receipt_path)},
                    sort_keys=True,
                ),
                flush=True,
            )
        raise
    receipt_path = receipt_path or (
        project
        / "runs"
        / "probes"
        / receipt["run_id"]
        / f"{args.gate}-receipt.json"
    )
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({"status": "passed", "receipt": str(receipt_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
