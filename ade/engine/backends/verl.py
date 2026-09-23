"""VERL implementation of the typed RFT backend."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from pathlib import Path
import re
from typing import Any

from ade.core.engine import EngineAttemptError, TrainRFTCommand
from ade.core.experiment import analysis_profile
from ade.tasks.reward_design.contracts import (
    RFTCheckpointResult,
    RFTTrainingResult,
)
from ade.engine.execution.ray import (
    GpuMemoryAdmissionError,
    run_eval_with_gpu_lease,
    select_gpu_node_for_memory_admission,
)
from ade.engine.execution.coordinator_resources import validate_coordinator_workload_request
from ade.engine.execution.gpu import (
    acquire_gpu_lease,
    release_gpu_lease,
)
from ade.engine.execution.ray import placement_group_node_ids
from ade.engine.requests import RFTInput
from ade.tasks.reward_design.verl_runtime import (
    export_verl_checkpoint,
    run_verl_rft,
)
from ade.engine.storage.atomic import write_json_atomic


class VerlRFTBackend:
    def __init__(self, work_root: str | Path) -> None:
        self.work_root = Path(work_root).resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        command: TrainRFTCommand,
        config: RFTInput,
        steps: tuple[int, ...],
        evaluate_checkpoint: Callable[[int, str], dict[str, Any]],
    ) -> RFTTrainingResult:
        if not steps:
            raise ValueError("VERL RFT requires checkpoint steps")
        verl_config = config.rft.get("verl_config")
        if not isinstance(verl_config, dict):
            raise ValueError("production RFT input requires rft.verl_config")
        total_steps = int(steps[-1])
        effective = dict(verl_config)
        train = dict(effective.get("train") or {})
        runtime_rft = dict(train.get("rft") or effective.get("rft") or {})
        artifact_interval = int(
            config.rft.get("artifact_interval")
            or runtime_rft.get("artifact_interval")
            or 4
        )
        runtime_rft.update(
            {
                "backend": "verl",
                "total_training_steps": total_steps,
                "artifact_interval": artifact_interval,
                "reward_trace_segment_id": f"segment_{total_steps:03d}",
                "resume_mode": "disable",
            }
        )
        train["backend"] = "verl"
        train["rft"] = runtime_rft
        effective["train"] = train
        effective["rft"] = runtime_rft
        effective["project_root"] = str(
            Path(effective.get("project_root") or Path.cwd()).resolve()
        )
        effective["plan_id"] = command.plan_id
        effective["coordinator_id"] = command.coordinator_id
        effective["trial_id"] = command.trial_id
        effective["run_id"] = command.run_id
        effective["engine_command_id"] = command.command_id
        effective["logical_command_id"] = command.logical_command_id
        effective["engine_attempt_id"] = command.attempt_id
        effective["engine_attempt_index"] = command.attempt_index
        if isinstance(config.raw.get("fork_lineage"), dict):
            effective["fork_lineage"] = dict(config.raw["fork_lineage"])
        if config.rft.get("trial_artifact_root"):
            effective["trial_artifact_root"] = str(
                Path(str(config.rft["trial_artifact_root"])).resolve()
            )
        run_dir = self.work_root / command.command_id
        train_gpus = int(runtime_rft.get("gpus_per_node", 8)) * int(
            runtime_rft.get("nodes", 1)
        )
        owner, workload = validate_coordinator_workload_request(
            runtime_rft, requested_gpus=train_gpus
        )
        training_lease = None
        if owner != "standalone":
            lease_owner = f"{owner}/{workload}/{command.command_id}"
            # Fail admission promptly, before the 1800s training-progress watchdog.
            admission_deadline = time.monotonic() + 120.0
            try:
                memory_admission = select_gpu_node_for_memory_admission(
                    required_gpus=train_gpus,
                    minimum_free_fraction=float(
                        runtime_rft.get("gpu_memory_utilization", 0.75)
                    ),
                    timeout_seconds=max(0.0, admission_deadline - time.monotonic()),
                )
                training_lease = acquire_gpu_lease(
                    owner=lease_owner,
                    kind="train",
                    requested_gpus=train_gpus,
                    minimum_gpus=train_gpus,
                    distributed_train=True,
                    coordinator_owner=owner,
                    workload=workload,
                    coordinator_policy=dict(
                        runtime_rft["coordinator_resource_policy"]
                    ),
                    target_node_resource=str(memory_admission["node_resource"]),
                    timeout_seconds=max(0.0, admission_deadline - time.monotonic()),
                )
            except (GpuMemoryAdmissionError, TimeoutError) as error:
                raise EngineAttemptError(
                    str(error),
                    failure_kind="gpu_memory_unavailable",
                    retryable=True,
                ) from error
            node_ids = list(placement_group_node_ids(training_lease["placement_group"]))
            for target in (runtime_rft, train["rft"]):
                target["placement_group_name"] = training_lease[
                    "placement_group_name"
                ]
                target["placement_group_id"] = training_lease[
                    "placement_group"
                ].id.hex()
                target["model_staging_node_ids"] = node_ids
                target["gpu_memory_admission"] = dict(memory_admission)
            effective["rft"] = runtime_rft
            effective["train"] = train
        try:
            training_failed = threading.Event()

            def run_training() -> dict[str, Any]:
                try:
                    return run_verl_rft(
                        effective,
                        run_dir=run_dir,
                        export_final_checkpoint=False,
                    )
                except BaseException:
                    training_failed.set()
                    raise

            with (
                ThreadPoolExecutor(max_workers=1) as training_pool,
                ThreadPoolExecutor(max_workers=2) as online_pool,
            ):
                training_future = training_pool.submit(run_training)
                checkpoint_futures = [
                    online_pool.submit(
                        self._export_checkpoint,
                        effective,
                        run_dir,
                        step,
                        evaluate_checkpoint,
                        training_failed,
                    )
                    for step in steps
                ]
                result = training_future.result()
                release_gpu_lease(training_lease)
                training_lease = None
                checkpoints = tuple(future.result() for future in checkpoint_futures)
        except Exception as error:
            failure_category = self._failure_category(run_dir)
            if failure_category in {
                "dependency_unavailable",
                "environment_compatibility",
                "gpu_memory_unavailable",
            }:
                raise EngineAttemptError(
                    str(error),
                    failure_kind=str(failure_category),
                    retryable=True,
                ) from error
            raise
        finally:
            release_gpu_lease(training_lease)
        analysis_manifest = self._build_analysis_manifest(
            result,
            run_dir=run_dir,
            rft=runtime_rft,
        )
        return RFTTrainingResult(checkpoints, str(analysis_manifest))

    @staticmethod
    def _failure_category(run_dir: Path) -> str | None:
        result_paths = sorted(run_dir.glob("verl_engine_result*.json"))
        if not result_paths:
            return None
        try:
            payload = json.loads(result_paths[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        value = payload.get("failure_category")
        return str(value) if value is not None else None

    def _export_checkpoint(
        self,
        effective: dict[str, Any],
        run_dir: Path,
        step: int,
        evaluate_checkpoint: Callable[[int, str], dict[str, Any]],
        training_failed: threading.Event,
    ) -> RFTCheckpointResult:
        exported = export_verl_checkpoint(
            effective,
            run_dir=run_dir,
            step=step,
            training_failed=training_failed,
        )
        checkpoint = str(exported.get("checkpoint_path") or "")
        if not checkpoint:
            raise RuntimeError(f"VERL RFT did not export checkpoint step {step}")
        online = evaluate_checkpoint(step, checkpoint)
        return RFTCheckpointResult(step, checkpoint, online)

    def evaluate(
        self,
        command: TrainRFTCommand,
        config: RFTInput,
        checkpoint: str,
        purpose: str,
        step: int,
    ) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_dir():
            raise RuntimeError(f"VERL checkpoint is missing: {checkpoint}")
        requests = config.rft.get("evaluation_requests")
        if not isinstance(requests, dict) or not isinstance(requests.get(purpose), dict):
            raise ValueError(f"rft.evaluation_requests.{purpose} is required")
        request = dict(requests[purpose])
        request.update(
            {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "ade_run_id": command.run_id,
                "ade_engine_command_id": command.command_id,
                "staging_consumer_id": (
                    f"{command.command_id}-{purpose}-rl-step-{step}"
                ),
                "phase": purpose,
                "result_suffix": f"{command.command_id}-{purpose}-rl-step-{step}",
                "gpu_lease_owner_suffix": (
                    f"{command.logical_command_id or command.command_id}-"
                    f"{purpose}-rl-step-{step}"
                ),
            }
        )
        if isinstance(config.raw.get("fork_lineage"), dict):
            request["fork_lineage"] = dict(config.raw["fork_lineage"])
        result = run_eval_with_gpu_lease(request)
        if "score" not in result:
            raise ValueError(f"{purpose} evaluation did not produce a ranking score")
        return result

    @staticmethod
    def _build_analysis_manifest(
        result: dict[str, Any],
        *,
        run_dir: Path,
        rft: dict[str, Any],
    ) -> Path:
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("VERL RFT result requires metrics")
        trace_path = Path(
            str(metrics.get("reward_rollout_trace_manifest_path") or "")
        )
        if not trace_path.is_file():
            raise ValueError("VERL RFT result requires reward trace manifest")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        source_files = trace.get("source_files")
        snapshots = trace.get("step_snapshots")
        if not isinstance(source_files, list) or not isinstance(snapshots, list):
            raise ValueError("VERL reward trace manifest lacks source topology")
        sources = [dict(item) for item in source_files]
        training_log = Path(str(metrics.get("training_log_path") or ""))
        if training_log.is_file():
            sources.append(
                _analysis_source(
                    "training-log",
                    "training_log",
                    training_log,
                    "text/plain",
                )
            )
        raw_shards = _raw_rollout_sources(
            run_dir / "engine_audit" / "reward_rollouts_raw",
            max_step=int(result.get("segment_step") or 0),
        )
        sources.extend(raw_shards)
        sources.append(
            _analysis_source(
                "reward-trace-manifest",
                "reward_trace_manifest",
                trace_path,
                "application/json",
            )
        )
        telemetry: dict[str, str] = {}
        for metric_key, source_id, kind, filename_key, media_type in (
            (
                "training_telemetry_path",
                "training-telemetry",
                "training_telemetry",
                "telemetry_unit_id",
                "application/x-ndjson",
            ),
            (
                "training_telemetry_summary_path",
                "training-telemetry-summary",
                "training_telemetry_summary",
                "summary_unit_id",
                "application/json",
            ),
            (
                "training_telemetry_manifest_path",
                "training-telemetry-manifest",
                "training_telemetry_manifest",
                "manifest_unit_id",
                "application/json",
            ),
        ):
            path_value = str(metrics.get(metric_key) or "")
            path = Path(path_value)
            if not path.is_file():
                continue
            sources.append(
                _analysis_source(source_id, kind, path, media_type)
            )
            telemetry[filename_key] = source_id
        steps = []
        for snapshot in snapshots:
            if not isinstance(snapshot, dict) or snapshot.get("step") is None:
                raise ValueError("VERL reward trace snapshot is invalid")
            item = {
                key: snapshot[key]
                for key in (
                    "step",
                    "rollout_training_step",
                    "policy_step_at_generation",
                    "status",
                    "prompt_group_count",
                    "responses_per_group",
                    "response_count",
                )
                if key in snapshot
            }
            item["rollout_unit_id"] = str(snapshot["rollout_source_id"])
            item["statistics_unit_id"] = str(snapshot["statistics_source_id"])
            steps.append(item)
        reward_rollouts = {
            "manifest_unit_id": "reward-trace-manifest",
        }
        if raw_shards:
            reward_rollouts["raw_shard_unit_ids"] = [
                str(item["source_id"]) for item in raw_shards
            ]
        source_ids = {str(item.get("source_id") or "") for item in sources}
        for key, source_id in (
            ("population_summary_unit_id", "reward-population-summary"),
        ):
            if source_id in source_ids:
                reward_rollouts[key] = source_id
        group_credit_enabled = _group_credit_enabled(rft)
        topology = {
            "steps": sorted(steps, key=lambda item: int(item["step"])),
            "training_telemetry": telemetry or None,
            "reward_rollouts": reward_rollouts,
            "offline_evaluation": None,
        }
        if group_credit_enabled:
            topology["semantic_evidence_steps"] = list(
                rft.get("semantic_evidence_steps") or ()
            )
            topology["semantic_evidence_unit"] = "groups"
        analysis_profile_id = str(rft.get("analysis_profile_id") or "reward_design")
        manifest = {
            "schema_version": (
                "ade.rft_analysis_sources.v2"
                if group_credit_enabled
                else "ade.rft_analysis_sources.v1"
            ),
            "sources": sources,
            "analysis": analysis_profile(
                analysis_profile_id,
                version=2 if group_credit_enabled else 1,
                topology=topology,
            ),
        }
        manifest_path = run_dir / "analysis_sources" / "manifest.json"
        write_json_atomic(manifest_path, manifest)
        return manifest_path


def _analysis_source(
    source_id: str,
    kind: str,
    path: Path,
    media_type: str,
) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "source_id": source_id,
        "kind": kind,
        "path": str(path.resolve()),
        "filename": path.name,
        "media_type": media_type,
        "visibility": "agent",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _group_credit_enabled(rft: dict[str, Any]) -> bool:
    group_credit = rft.get("group_credit")
    return isinstance(group_credit, dict) and group_credit.get("enabled") is True


def _raw_rollout_sources(raw_root: Path, *, max_step: int) -> list[dict[str, Any]]:
    if not raw_root.is_dir():
        return []
    sources: list[dict[str, Any]] = []
    for segment_dir in sorted(raw_root.glob("segment_*"), key=lambda path: path.name):
        match = re.fullmatch(r"segment_(\d+)", segment_dir.name)
        if match is None or not segment_dir.is_dir():
            continue
        segment_step = int(match.group(1))
        if max_step and segment_step > max_step:
            continue
        for shard in sorted(segment_dir.glob("*.jsonl.gz"), key=lambda path: path.name):
            shard_id = re.sub(
                r"[^A-Za-z0-9._-]",
                "-",
                shard.name.removesuffix(".jsonl.gz"),
            )
            source_id = f"raw-rollout-{segment_dir.name}-{shard_id}"
            source = _analysis_source(
                source_id,
                "reward_rollout_raw_shard",
                shard,
                "application/gzip",
            )
            source["step"] = segment_step
            sources.append(source)
    return sources
