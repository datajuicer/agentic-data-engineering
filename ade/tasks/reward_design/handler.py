"""Restart-safe RFT execution from a typed TrainRFTCommand."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol

from ade.core.engine import EngineAttemptError, TrainRFTCommand
from ade.core.ranking import score_pair_sort_key
from ade.core.experiment import analysis_profile, validate_analysis_profile
from ade.tasks.reward_design.contracts import RFTTrainingResult
from ade.engine.checkpoints.cleanup import cleanup_checkpoint_directories
from ade.engine.evaluation_dispatcher import EngineEvaluationDispatcher
from ade.engine.storage.atomic import write_json_atomic
from ade.engine.storage.object_store import FileEngineObjectStore
from ade.engine.requests import RFTInput, decode_rft_input
from ade.engine.trial_artifacts import (
    ArtifactRecord,
    TrialArtifactPublisher,
    TrialArtifactWorkspace,
    artifact_positions,
    remove_generated_wandb_symlinks,
    stage_training_telemetry_artifacts,
    write_evaluation_artifacts,
    write_rft_reward_audit,
    write_training_rollout_artifact,
)


class RFTBackend(Protocol):
    def train(
        self,
        command: TrainRFTCommand,
        config: RFTInput,
        steps: tuple[int, ...],
        evaluate_checkpoint: Callable[[int, str], dict[str, Any]],
    ) -> RFTTrainingResult: ...

    def evaluate(
        self,
        command: TrainRFTCommand,
        config: RFTInput,
        checkpoint: str,
        purpose: str,
        step: int,
    ) -> dict[str, Any]: ...


class RFTExecutor:
    def __init__(
        self,
        *,
        io: FileEngineObjectStore,
        backend: RFTBackend,
        evaluations: EngineEvaluationDispatcher | None = None,
        artifacts: TrialArtifactPublisher | None = None,
    ) -> None:
        self.io = io
        self.backend = backend
        self.evaluations = evaluations
        self.artifacts = artifacts

    def execute(self, command: TrainRFTCommand) -> tuple[str, ...]:
        if type(command) is not TrainRFTCommand:
            raise TypeError("RFTExecutor requires TrainRFTCommand")
        if not command.output_uri:
            raise ValueError("TrainRFTCommand output_uri is required")
        workspace = (
            self.artifacts.workspace(
                command.run_id,
                command.coordinator_id,
                command.plan_id,
                command.trial_id,
                attempt_id=command.attempt_id,
            )
            if self.artifacts is not None
            else None
        )
        artifact_records: list[ArtifactRecord] = []
        config = self._localize_input(
            decode_rft_input(self.io.read_json(command.input_ref)),
            command=command,
            workspace=workspace,
        )
        if workspace is not None:
            artifact_records.extend(
                write_rft_reward_audit(
                    workspace,
                    reward_path=self.io.path_for(config.artifact_ref),
                    runtime_config=config.rft["verl_config"],
                )
            )
        input_digest = hashlib.sha256(
            json.dumps(config.raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        state_uri = f"{command.output_uri}/state.json"
        state = self.io.read_json(state_uri) if self.io.exists(state_uri) else {
            "schema_version": "1",
            "command_id": command.command_id,
            "input_digest": input_digest,
            "checkpoints": {},
            "raw_units": [],
        }
        if state.get("command_id") != command.command_id:
            raise ValueError("persisted RFT state belongs to another command")
        if state.get("input_digest") != input_digest:
            raise ValueError("RFT input changed after execution started")
        self.io.write_json(state_uri, state)
        rft = config.rft
        step0_online_ref = rft.get("step0_online_evaluation_ref")
        if step0_online_ref is not None and (
            not isinstance(step0_online_ref, str) or not step0_online_ref
        ):
            raise ValueError("RFT step-0 online evaluation ref must be non-empty")
        total_steps = int(rft["total_training_steps"])
        interval = int(rft["artifact_interval"])
        steps = list(artifact_positions(total=total_steps, interval=interval))
        checkpoints = state["checkpoints"]
        if any(
            "checkpoint" not in checkpoints.get(str(step), {})
            or "online_validation" not in checkpoints.get(str(step), {})
            for step in steps
        ):
            try:
                training = self.backend.train(
                    command,
                    config,
                    tuple(steps),
                    lambda step, checkpoint: self._evaluate_checkpoint(
                        command,
                        config,
                        checkpoint,
                        "online_validation",
                        step,
                        workspace=workspace,
                        artifact_records=artifact_records,
                    ),
                )
            except Exception as error:
                if workspace is None or self.artifacts is None:
                    raise
                output_refs = self._publish_training_failure(
                    command,
                    config,
                    workspace,
                    steps=steps,
                    records=artifact_records,
                    error=error,
                )
                message = (
                    "RFT training failed; failure evidence published at "
                    + ", ".join(output_refs)
                )
                if isinstance(error, EngineAttemptError):
                    raise EngineAttemptError(
                        message,
                        failure_kind=error.failure_kind,
                        retryable=error.retryable,
                        output_refs=output_refs,
                    ) from error
                raise EngineAttemptError(
                    message,
                    failure_kind="engine_failed",
                    retryable=False,
                    output_refs=output_refs,
                ) from error
            if not isinstance(training, RFTTrainingResult):
                raise TypeError("RFT backend train must return RFTTrainingResult")
            by_step = {item.step: item for item in training.checkpoints}
            if set(by_step) != set(steps):
                raise ValueError("RFT backend must return every requested checkpoint step")
            for step in steps:
                item = by_step[step]
                checkpoint = item.checkpoint_path
                expected_name = (
                    f"rl-step-{step:03d}"
                    if workspace is not None
                    else f"checkpoint-{step}"
                )
                if Path(checkpoint).name != expected_name or not Path(
                    checkpoint
                ).is_dir():
                    raise ValueError(
                        f"RFT position {step} must produce {expected_name}"
                    )
                result = item.online_validation
                record = checkpoints.setdefault(str(step), {"step": step})
                record["checkpoint"] = checkpoint
                record["analysis_manifest_path"] = training.analysis_manifest_path
                record["online_validation"] = self._evaluation_state(result)
                self._record_raw_unit(
                    state,
                    command,
                    purpose="online_validation",
                    step=step,
                    payload=result,
                    visibility="agent",
                )
                self.io.write_json(state_uri, state)
        analysis = state.get("analysis")
        if not isinstance(analysis, dict):
            final_record = checkpoints[str(total_steps)]
            analysis = self._ingest_training_analysis(
                state,
                command,
                final_record.get("analysis_manifest_path"),
                analysis_profile_id=str(
                    rft.get("analysis_profile_id") or "reward_design"
                ),
                group_credit_enabled=(
                    isinstance(rft.get("group_credit"), dict)
                    and rft["group_credit"].get("enabled") is True
                ),
            )
            if workspace is not None and final_record.get(
                "analysis_manifest_path"
            ):
                artifact_records.extend(
                    self._stage_training_artifacts(
                        workspace,
                        final_record["analysis_manifest_path"],
                        rollout_n=self._rollout_n(rft),
                    )
                )
            state["analysis"] = analysis
            self.io.write_json(state_uri, state)
        rankable_steps = [
            step
            for step in steps
            if checkpoints[str(step)]["online_validation"].get("status")
            == "complete"
        ]
        selected_step = (
            min(
                rankable_steps,
                key=lambda step: (
                    *score_pair_sort_key(
                        float(
                            checkpoints[str(step)]["online_validation"][
                                "ranking_score"
                            ]
                        ),
                        (
                            float(
                                checkpoints[str(step)]["online_validation"][
                                    "secondary_score"
                                ]
                            )
                            if checkpoints[str(step)]["online_validation"].get(
                                "secondary_score"
                            )
                            is not None
                            else None
                        ),
                        direction="maximize",
                    ),
                    step,
                ),
            )
            if rankable_steps
            else None
        )
        selected = (
            checkpoints[str(selected_step)]
            if selected_step is not None
            else None
        )
        offline = state.get("offline_validation")
        if (
            rft.get("run_offline_validation", True)
            and offline is None
            and selected is not None
        ):
            payload = self._evaluate_checkpoint(
                command,
                config,
                selected["checkpoint"],
                "offline_validation",
                selected_step,
                workspace=workspace,
                artifact_records=artifact_records,
            )
            offline = self._evaluation_state(payload)
            state["offline_validation"] = offline
            self._record_raw_unit(
                state,
                command,
                purpose="offline_validation",
                step=selected_step,
                payload=payload,
                visibility="agent",
            )
            self.io.write_json(state_uri, state)
        if (
            rft.get("run_offline_validation", True)
            and offline is None
            and selected is None
        ):
            offline = {
                "status": "skipped",
                "ranking_score": None,
                "reason": "no_complete_online_validation",
            }
            state["offline_validation"] = offline
            if workspace is not None:
                artifact_records.append(
                    ArtifactRecord(
                        artifact_id="offline-validation",
                        category="eval_result",
                        kind="offline_validation",
                        status="skipped",
                        metadata={
                            "reason": "no_complete_online_validation"
                        },
                    )
                )
            self.io.write_json(state_uri, state)
        self._cleanup_exported_checkpoints(
            state,
            steps=steps,
            selected_step=selected_step,
            retention_top_k=int(rft.get("checkpoint_retention_top_k", -1)),
        )
        if selected_step is not None:
            analysis = self._link_evaluations(
                analysis,
                steps=steps,
                selected_step=selected_step,
                include_offline=(
                    isinstance(offline, dict)
                    and offline.get("status") == "complete"
                ),
            )
        state["analysis"] = analysis
        if workspace is None and selected_step is not None:
            self._write_analysis_readme(
                state,
                command,
                selected_step=selected_step,
                offline=offline,
            )
        self.io.write_json(state_uri, state)
        result_uri = f"{command.output_uri}/result.json"
        manifest_uri = f"{command.output_uri}/raw/manifest.json"
        trial_manifest_path = None
        if workspace is not None and self.artifacts is not None:
            remove_generated_wandb_symlinks(workspace.audit)
            for step in steps:
                checkpoint = checkpoints[str(step)]["checkpoint"]
                checkpoint_path = Path(checkpoint)
                artifact_records.append(
                    ArtifactRecord(
                        artifact_id=f"checkpoint-rl-step-{step:03d}",
                        category="checkpoint",
                        kind="hf_checkpoint",
                        status=(
                            "complete" if checkpoint_path.is_dir() else "pruned"
                        ),
                        path=(
                            checkpoint_path.relative_to(workspace.root).as_posix()
                            if checkpoint_path.is_dir()
                            else None
                        ),
                        metadata={
                            "artifact_position": {
                                "unit": "rl_step",
                                "value": step,
                            }
                        },
                    )
                )
            failed = any(
                checkpoints[str(step)]["online_validation"].get("status")
                != "complete"
                for step in steps
            ) or (
                isinstance(offline, dict)
                and offline.get("status") != "complete"
            )
            trial_manifest_path = str(
                self.artifacts.publish(
                    workspace,
                    trial_status=(
                        "completed_with_failures" if failed else "completed"
                    ),
                    selected_checkpoint_id=(
                        f"checkpoint-rl-step-{selected_step:03d}"
                        if selected_step is not None
                        else None
                    ),
                    records=tuple(artifact_records),
                    metadata={
                        "command_id": command.command_id,
                        "position_unit": "rl_step",
                        "artifact_interval": interval,
                        "planned_final_position": total_steps,
                    },
                )
            )
        self.io.put_json(
            result_uri,
            {
                "schema_version": "1",
                "command_id": command.command_id,
                "coordinator_id": command.coordinator_id,
                "plan_id": command.plan_id,
                "trial_id": command.trial_id,
                "selected_step": selected_step,
                "selected_position": (
                    {"unit": "rl_step", "value": selected_step}
                    if selected_step is not None
                    else None
                ),
                "checkpoint_ref": (
                    selected["checkpoint"] if selected is not None else None
                ),
                "online_validation": (
                    selected["online_validation"]
                    if selected is not None
                    else None
                ),
                "step0_online_evaluation_ref": step0_online_ref,
                "offline_validation": offline,
                "trial_artifact_manifest_path": trial_manifest_path,
            },
        )
        self.io.put_json(
            manifest_uri,
            {
                "schema_version": "2",
                "command_id": command.command_id,
                "coordinator_id": command.coordinator_id,
                "plan_id": command.plan_id,
                "trial_id": command.trial_id,
                "units": state["raw_units"],
                "analysis": analysis,
                "step0_online_evaluation_ref": step0_online_ref,
            },
        )
        state["status"] = "completed"
        self.io.write_json(state_uri, state)
        return result_uri, manifest_uri

    def _publish_training_failure(
        self,
        command: TrainRFTCommand,
        config: RFTInput,
        workspace: TrialArtifactWorkspace,
        *,
        steps: list[int],
        records: list[ArtifactRecord],
        error: Exception,
    ) -> tuple[str, ...]:
        failure_path = workspace.audit / "logs" / "training_failure.json"
        write_json_atomic(
            failure_path,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        records.append(
            ArtifactRecord(
                artifact_id="training-failure",
                category="audit",
                kind="training_failure",
                path=failure_path.relative_to(workspace.root).as_posix(),
            )
        )
        rankable: list[tuple[float, float | None, int, Path]] = []
        for result_path in workspace.eval_results.glob(
            "online-validation-rl-step-*.json"
        ):
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            position = payload.get("artifact_position") or {}
            step = int(position.get("value") or 0)
            checkpoint = workspace.checkpoints / f"rl-step-{step:03d}"
            if (
                payload.get("status") == "complete"
                and payload.get("ranking_score") is not None
                and checkpoint.is_dir()
            ):
                rankable.append(
                    (
                        float(payload["ranking_score"]),
                        (
                            float(payload["secondary_score"])
                            if payload.get("secondary_score") is not None
                            else None
                        ),
                        step,
                        checkpoint,
                    )
                )
        selected = (
            min(
                rankable,
                key=lambda item: (
                    *score_pair_sort_key(item[0], item[1], direction="maximize"),
                    item[2],
                ),
            )
            if rankable
            else None
        )
        if selected is not None:
            _, _, selected_step, selected_checkpoint = selected
            offline_payload = self._evaluate_checkpoint(
                command,
                config,
                str(selected_checkpoint),
                "offline_validation",
                selected_step,
                workspace=workspace,
                artifact_records=records,
            )
            offline = self._evaluation_state(offline_payload)
        else:
            selected_step = None
            selected_checkpoint = None
            offline = {
                "status": "skipped",
                "ranking_score": None,
                "reason": "no_complete_online_validation",
            }
            records.append(
                ArtifactRecord(
                    artifact_id="offline-validation",
                    category="eval_result",
                    kind="offline_validation",
                    status="skipped",
                    metadata={"reason": "no_complete_online_validation"},
                )
            )
        for step in steps:
            checkpoint = workspace.checkpoints / f"rl-step-{step:03d}"
            records.append(
                ArtifactRecord(
                    artifact_id=f"checkpoint-rl-step-{step:03d}",
                    category="checkpoint",
                    kind="hf_checkpoint",
                    status="complete" if checkpoint.is_dir() else "unavailable",
                    path=(
                        checkpoint.relative_to(workspace.root).as_posix()
                        if checkpoint.is_dir()
                        else None
                    ),
                    metadata={
                        "artifact_position": {
                            "unit": "rl_step",
                            "value": step,
                        },
                        **(
                            {}
                            if checkpoint.is_dir()
                            else {"reason": "training_terminated_before_checkpoint"}
                        ),
                    },
                )
            )
        remove_generated_wandb_symlinks(workspace.audit)
        trial_manifest = self.artifacts.publish(
            workspace,
            trial_status="failed",
            selected_checkpoint_id=(
                f"checkpoint-rl-step-{selected_step:03d}"
                if selected_step is not None
                else None
            ),
            records=tuple(records),
            metadata={
                "command_id": command.command_id,
                "position_unit": "rl_step",
                "artifact_interval": int(
                    config.rft.get("artifact_interval") or 4
                ),
                "planned_final_position": steps[-1],
                "training_error": str(error),
            },
        )
        result_uri = f"{command.output_uri}/result.json"
        manifest_uri = f"{command.output_uri}/raw/manifest.json"
        self.io.put_json(
            result_uri,
            {
                "schema_version": "1",
                "command_id": command.command_id,
                "coordinator_id": command.coordinator_id,
                "plan_id": command.plan_id,
                "trial_id": command.trial_id,
                "selected_step": selected_step,
                "selected_position": (
                    {"unit": "rl_step", "value": selected_step}
                    if selected_step is not None
                    else None
                ),
                "checkpoint_ref": (
                    str(selected_checkpoint)
                    if selected_checkpoint is not None
                    else None
                ),
                "online_validation": (
                    {
                        "status": "complete",
                        "ranking_score": selected[0],
                        "secondary_score": selected[1],
                    }
                    if selected is not None
                    else None
                ),
                "step0_online_evaluation_ref": config.rft.get(
                    "step0_online_evaluation_ref"
                ),
                "offline_validation": offline,
                "training_error": str(error),
                "trial_artifact_manifest_path": str(trial_manifest),
            },
        )
        self.io.put_json(
            manifest_uri,
            {
                "schema_version": "2",
                "command_id": command.command_id,
                "coordinator_id": command.coordinator_id,
                "plan_id": command.plan_id,
                "trial_id": command.trial_id,
                "units": [],
                "step0_online_evaluation_ref": config.rft.get(
                    "step0_online_evaluation_ref"
                ),
                "analysis": analysis_profile(
                    str(
                        config.rft.get("analysis_profile_id")
                        or "reward_design"
                    ),
                    version=1,
                    topology={
                        "steps": [],
                        "training_telemetry": None,
                        "reward_rollouts": None,
                        "offline_evaluation": None,
                    },
                ),
            },
        )
        return result_uri, manifest_uri

    @staticmethod
    def _stage_training_artifacts(
        workspace: TrialArtifactWorkspace,
        manifest_path: str,
        *,
        rollout_n: int,
    ) -> tuple[ArtifactRecord, ...]:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        sources = payload.get("sources")
        if not isinstance(sources, list):
            raise ValueError("RFT analysis source manifest lacks sources")
        records: list[ArtifactRecord] = []
        records.extend(
            stage_training_telemetry_artifacts(
                workspace,
                manifest_path,
                position_unit="rl_step",
                target_namespace="rft",
                move_sources=True,
            )
        )
        audit_kinds = {
            "rollout_statistics",
            "reward_population_summary",
            "training_log",
        }
        for source in sources:
            if not isinstance(source, dict):
                continue
            kind = str(source.get("kind") or "")
            path = Path(str(source.get("path") or ""))
            source_id = str(source.get("source_id") or "")
            if kind == "grpo_rollouts":
                records.append(
                    write_training_rollout_artifact(
                        workspace,
                        source_path=path,
                        step=int(source["step"]),
                        prompt_groups=int(source["prompt_group_count"]),
                        rollout_n=rollout_n,
                    )
                )
                continue
            if kind not in audit_kinds or not path.is_file():
                continue
            suffix = "".join(path.suffixes)
            target = workspace.audit / "metrics" / "rft" / f"{source_id}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"RFT audit artifact already exists: {target}")
            path.replace(target)
            records.append(
                ArtifactRecord(
                    artifact_id=f"audit-{source_id}",
                    category="audit",
                    kind=kind,
                    path=target.relative_to(workspace.root).as_posix(),
                    metadata=(
                        {
                            "artifact_position": {
                                "unit": "rl_step",
                                "value": int(source["step"]),
                            }
                        }
                        if source.get("step") is not None
                        else {}
                    ),
                )
            )
        return tuple(records)

    @staticmethod
    def _rollout_n(rft: dict[str, Any]) -> int:
        if rft.get("rollout_n") is not None:
            return int(rft["rollout_n"])
        verl_config = rft.get("verl_config")
        runtime = (
            verl_config.get("rft")
            if isinstance(verl_config, dict)
            and isinstance(verl_config.get("rft"), dict)
            else {}
        )
        return int(runtime.get("rollout_n", 16))

    def _evaluate_checkpoint(
        self,
        command: TrainRFTCommand,
        config: RFTInput,
        checkpoint: str,
        purpose: str,
        step: int,
        *,
        workspace: TrialArtifactWorkspace | None = None,
        artifact_records: list[ArtifactRecord] | None = None,
    ) -> dict[str, Any]:
        try:
            if self.evaluations is None:
                payload = self.backend.evaluate(
                    command,
                    config,
                    checkpoint,
                    purpose,
                    step,
                )
            else:
                requests = config.rft.get("evaluation_requests")
                if not isinstance(requests, dict) or not isinstance(
                    requests.get(purpose), dict
                ):
                    raise ValueError(
                        f"rft.evaluation_requests.{purpose} is required"
                    )
                evaluation_request = dict(requests[purpose])
                evaluation_request["evaluation_subject_kind"] = (
                    "p000_baseline"
                    if command.plan_id == "p000"
                    else "search_trial"
                )
                if purpose == "online_validation":
                    order = evaluation_request.get(
                        "evaluation_tracking_position_order"
                    )
                    if isinstance(order, list):
                        evaluation_request["evaluation_tracking_position_order"] = [
                            position for position in order if position != 0
                        ]
                if isinstance(config.raw.get("fork_lineage"), dict):
                    evaluation_request["fork_lineage"] = dict(
                        config.raw["fork_lineage"]
                    )
                dispatched = self.evaluations.evaluate_checkpoint(
                    parent=command,
                    checkpoint=checkpoint,
                    purpose=purpose,
                    step=step,
                    position_unit="rl_step",
                    request=evaluation_request,
                )
                if dispatched.status not in {"complete", "partial"}:
                    raise RuntimeError(
                        dispatched.error
                        or f"{purpose} evaluation failed at RL step {step}"
                    )
                payload = dict(dispatched.payload)
                payload["score"] = dispatched.score
                payload["evaluation_command_id"] = dispatched.command_id
                payload["status"] = dispatched.status
            status = str(payload.get("status") or "complete")
            if status not in {"complete", "partial"}:
                raise ValueError(f"unsupported evaluation status: {status}")
            score = self._score(payload) if status == "complete" else None
            payload["status"] = status
            payload["ranking_score"] = score
            if workspace is not None and artifact_records is not None:
                artifact_records.extend(
                    write_evaluation_artifacts(
                        workspace,
                        purpose=purpose,
                        position_unit="rl_step",
                        position_value=step,
                        checkpoint_artifact_id=(
                            f"checkpoint-rl-step-{step:03d}"
                        ),
                        ranking_score=score,
                        payload=payload,
                        status=status,
                    )
                )
            return payload
        except Exception as error:
            if workspace is None:
                raise
            failed = {
                "status": "failed",
                "score": None,
                "ranking_score": None,
                "error": {
                    "category": "evaluation_runtime_error",
                    "message": str(error),
                },
            }
            if artifact_records is not None:
                artifact_records.append(
                    ArtifactRecord(
                        artifact_id=(
                            f"{purpose.replace('_', '-')}-"
                            f"rl-step-{step:03d}"
                        ),
                        category="eval_result",
                        kind=purpose,
                        status="failed",
                        metadata={"error": failed["error"]},
                    )
                )
            return failed

    @staticmethod
    def _cleanup_exported_checkpoints(
        state: dict[str, Any],
        *,
        steps: list[int],
        selected_step: int | None,
        retention_top_k: int,
    ) -> None:
        if retention_top_k < -1:
            raise ValueError(
                "rft.checkpoint_retention_top_k must be -1 or non-negative"
            )
        if retention_top_k == -1 and selected_step is not None:
            state["retained_checkpoint_steps"] = list(steps)
            return
        checkpoints = state["checkpoints"]
        successful = [
            step
            for step in steps
            if checkpoints[str(step)]["online_validation"].get(
                "status",
                "complete",
            )
            == "complete"
            and checkpoints[str(step)]["online_validation"].get(
                "ranking_score",
                checkpoints[str(step)]["online_validation"].get("score"),
            )
            is not None
        ]
        ranked = sorted(
            successful,
            key=lambda step: (
                *score_pair_sort_key(
                    float(
                        checkpoints[str(step)]["online_validation"].get(
                            "ranking_score",
                            checkpoints[str(step)]["online_validation"].get("score"),
                        )
                    ),
                    (
                        float(
                            checkpoints[str(step)]["online_validation"][
                                "secondary_score"
                            ]
                        )
                        if checkpoints[str(step)]["online_validation"].get(
                            "secondary_score"
                        )
                        is not None
                        else None
                    ),
                    direction="maximize",
                ),
                step,
            ),
        )
        retained = (
            set(ranked[:retention_top_k])
            if retention_top_k >= 0
            else set(steps)
        )
        if selected_step is not None:
            retained.add(selected_step)
        else:
            retained = {steps[-1]}
        paths = [Path(checkpoints[str(step)]["checkpoint"]) for step in steps]
        roots = {path.resolve().parent for path in paths}
        if len(roots) != 1:
            raise ValueError("RFT exported checkpoints must share one directory")
        cleanup_checkpoint_directories(
            [path for step, path in zip(steps, paths) if step not in retained],
            allowed_root=roots.pop(),
            name_pattern=r"(?:checkpoint-\d+|rl-step-\d+)",
        )
        state["retained_checkpoint_steps"] = sorted(retained)

    def _localize_input(
        self,
        config: RFTInput,
        *,
        command: TrainRFTCommand,
        workspace: TrialArtifactWorkspace | None = None,
    ) -> RFTInput:
        reward = config.rft.get("reward")
        verl_config = config.rft.get("verl_config")
        if not isinstance(reward, dict) or not isinstance(verl_config, dict):
            raise ValueError("RFT input requires bound reward and rft.verl_config")
        function_ref = str(reward.get("function_ref") or "")
        group_credit = config.rft.get("group_credit")
        group_credit_enabled = (
            isinstance(group_credit, dict) and group_credit.get("enabled") is True
        )
        entrypoint_key = "score_entrypoint" if group_credit_enabled else "entrypoint"
        entrypoint = str(reward.get(entrypoint_key) or "")
        if function_ref != config.artifact_ref or entrypoint != "compute_score":
            raise ValueError(
                "RFT bound reward must be the accepted compute_score artifact"
            )
        reward_path = str(self.io.path_for(function_ref))
        if group_credit_enabled and (
            reward.get("fallback_entrypoint") != "compute_fallback_score"
            or reward.get("group_credit_entrypoint") != "assign_group_credit"
        ):
            raise ValueError("enabled RFT group credit requires the bound group artifact")

        localized_verl = copy.deepcopy(verl_config)
        scheduled_ref = config.rft.get("scheduled_training_data_ref")
        if scheduled_ref is not None:
            if not isinstance(scheduled_ref, str) or not scheduled_ref:
                raise ValueError("RFT scheduled_training_data_ref is invalid")
            data = dict(localized_verl.get("data") or {})
            data["train"] = str(self.io.path_for(scheduled_ref))
            localized_verl["data"] = data
        localized_verl["agent_task_id"] = command.run_id
        localized_verl["plan_id"] = command.plan_id
        localized_verl["trial_id"] = command.trial_id
        usage_run_dir = config.raw.get("usage_run_dir")
        if usage_run_dir:
            localized_verl["usage_run_dir"] = str(Path(str(usage_run_dir)).resolve())
        runtime_rft = dict(localized_verl.get("rft") or {})
        runtime_rft["reward_function_path"] = reward_path
        runtime_rft["reward_entrypoint"] = entrypoint
        if group_credit_enabled:
            runtime_rft["group_credit_function_path"] = reward_path
            runtime_rft["group_credit_entrypoint"] = "assign_group_credit"
        runtime_rft["coordinator_resource_owner"] = (
            f"{command.run_id}/{command.coordinator_id}"
        )
        train = dict(localized_verl.get("train") or {})
        train_rft = dict(train.get("rft") or {})
        train_rft["reward_function_path"] = reward_path
        train_rft["reward_entrypoint"] = entrypoint
        if group_credit_enabled:
            train_rft["group_credit_function_path"] = reward_path
            train_rft["group_credit_entrypoint"] = "assign_group_credit"
        train_rft["coordinator_resource_owner"] = (
            f"{command.run_id}/{command.coordinator_id}"
        )
        train["rft"] = train_rft
        localized_verl["rft"] = runtime_rft
        localized_verl["train"] = train
        if workspace is not None:
            localized_verl["trial_artifact_root"] = str(workspace.root)

        rft = copy.deepcopy(config.rft)
        rft["verl_config"] = localized_verl
        if workspace is not None:
            rft["trial_artifact_root"] = str(workspace.root)
        raw = copy.deepcopy(config.raw)
        raw["rft"] = rft
        return RFTInput(config.artifact_ref, rft, raw)

    def _record_raw_unit(
        self,
        state: dict[str, Any],
        command: TrainRFTCommand,
        *,
        purpose: str,
        step: int,
        payload: dict[str, Any],
        visibility: str,
    ) -> None:
        unit_id = f"{purpose.replace('_', '-')}-step-{step:03d}"
        if any(item["unit_id"] == unit_id for item in state["raw_units"]):
            return
        uri = f"{command.output_uri}/raw/units/{unit_id}.json"
        self.io.put_json(uri, payload)
        state["raw_units"].append(
            {
                "unit_id": unit_id,
                "kind": purpose,
                "uri": uri,
                "filename": (
                    "online-eval-result.json"
                    if purpose == "online_validation"
                    else "offline-eval-result.json"
                ),
                "media_type": "application/json",
                "visibility": visibility,
                "step": step,
            }
        )

    def _ingest_training_analysis(
        self,
        state: dict[str, Any],
        command: TrainRFTCommand,
        manifest_path: Any,
        *,
        analysis_profile_id: str = "reward_design",
        group_credit_enabled: bool = False,
    ) -> dict[str, Any]:
        if manifest_path is None:
            return analysis_profile(
                analysis_profile_id,
                version=2 if group_credit_enabled else 1,
                topology={
                    "steps": [],
                    "training_telemetry": None,
                    "reward_rollouts": None,
                    "offline_evaluation": None,
                    **(
                        {
                            "semantic_evidence_steps": [],
                            "semantic_evidence_unit": "groups",
                        }
                        if group_credit_enabled
                        else {}
                    ),
                },
            )
        path = Path(str(manifest_path)).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_schema = payload.get("schema_version")
        if source_schema not in {
            "ade.rft_analysis_sources.v1",
            "ade.rft_analysis_sources.v2",
        }:
            raise ValueError("RFT training analysis manifest is invalid")
        expected_source_schema = (
            "ade.rft_analysis_sources.v2"
            if group_credit_enabled
            else "ade.rft_analysis_sources.v1"
        )
        if source_schema != expected_source_schema:
            raise ValueError("RFT training analysis mechanism/schema mismatch")
        sources = payload.get("sources")
        analysis = payload.get("analysis")
        if not isinstance(sources, list) or not isinstance(analysis, dict):
            raise ValueError("RFT training analysis manifest requires sources and analysis")
        validate_analysis_profile(analysis)
        if analysis.get("profile") != analysis_profile_id:
            raise ValueError("RFT training analysis profile does not match task")
        expected_profile = (
            f"ade.{analysis_profile_id}_experiment.v2"
            if source_schema == "ade.rft_analysis_sources.v2"
            else f"ade.{analysis_profile_id}_experiment.v1"
        )
        if analysis.get("schema_version") != expected_profile:
            raise ValueError("RFT training analysis source/profile versions differ")
        seen: set[str] = set()
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("RFT training analysis sources must be objects")
            source_id = str(source.get("source_id") or "")
            source_path = Path(str(source.get("path") or "")).resolve()
            filename = str(source.get("filename") or "")
            media_type = str(source.get("media_type") or "")
            visibility = str(source.get("visibility") or "")
            if (
                not source_id
                or "/" in source_id
                or "\\" in source_id
                or source_id in {".", ".."}
                or not source_path.is_file()
                or not filename
                or Path(filename).name != filename
                or "\\" in filename
                or not media_type
                or visibility != "agent"
                or source_id in seen
            ):
                raise ValueError("RFT training analysis source is invalid")
            seen.add(source_id)
            content = source_path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if source.get("sha256") is not None and str(source["sha256"]) != digest:
                raise ValueError(f"RFT training analysis digest mismatch: {source_id}")
            uri = f"{command.output_uri}/raw/units/{source_id}/{filename}"
            self.io.put_bytes(uri, content)
            if any(item["unit_id"] == source_id for item in state["raw_units"]):
                continue
            unit = {
                "unit_id": source_id,
                "kind": str(source.get("kind") or ""),
                "uri": uri,
                "filename": filename,
                "media_type": media_type,
                "visibility": "agent",
                "sha256": digest,
                "size_bytes": len(content),
            }
            if source.get("step") is not None:
                unit["step"] = int(source["step"])
            state["raw_units"].append(unit)
        state["analysis_sources_root"] = str(path.parent)
        return copy.deepcopy(analysis)

    def _write_analysis_readme(
        self,
        state: dict[str, Any],
        command: TrainRFTCommand,
        *,
        selected_step: int,
        offline: dict[str, Any] | None,
    ) -> None:
        all_units = [item for item in state["raw_units"] if isinstance(item, dict)]
        units = [item for item in all_units if item.get("unit_id") != "analysis-readme"]
        inventory: list[dict[str, Any]] = []
        total_bytes = 0
        total_records = 0
        for item in units:
            uri = str(item.get("uri") or "")
            if not uri:
                continue
            path = self.io.path_for(uri)
            size_bytes = path.stat().st_size
            record_count = _record_count(path)
            total_bytes += size_bytes
            if record_count is not None:
                total_records += record_count
            inventory.append(
                {
                    "unit_id": str(item.get("unit_id") or ""),
                    "kind": str(item.get("kind") or ""),
                    "step": item.get("step"),
                    "size_bytes": size_bytes,
                    "record_count": record_count,
                    "media_type": str(item.get("media_type") or ""),
                }
            )
        lines = [
            "# Trial analysis inputs",
            "",
            f"- command: `{command.command_id}`",
            f"- trial: `{command.trial_id}`",
            f"- selected checkpoint step: `{selected_step}`",
            f"- offline score: `{None if offline is None else offline.get('score')}`",
            f"- analysis units: `{len(inventory)}`",
            f"- total stored bytes: `{total_bytes}`",
            f"- total counted records: `{total_records}`",
            "",
            "This directory contains the agent-visible training, rollout, and evaluation evidence. "
            "Operator-test evidence is excluded.",
            "",
            "| unit | kind | step | bytes | records | media type |",
            "|---|---|---:|---:|---:|---|",
        ]
        for item in inventory:
            records = "" if item["record_count"] is None else str(item["record_count"])
            step = "" if item["step"] is None else str(item["step"])
            lines.append(
                f"| `{item['unit_id']}` | `{item['kind']}` | {step} | "
                f"{item['size_bytes']} | {records} | `{item['media_type']}` |"
            )
        content = ("\n".join(lines) + "\n").encode("utf-8")
        readme_uri = f"{command.output_uri}/raw/units/analysis-readme.md"
        self.io.put_bytes(readme_uri, content)
        if not any(item.get("unit_id") == "analysis-readme" for item in all_units):
            state["raw_units"].append(
                {
                    "unit_id": "analysis-readme",
                    "kind": "analysis_readme",
                    "uri": readme_uri,
                    "filename": "README.md",
                    "media_type": "text/markdown",
                    "visibility": "agent",
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                }
            )
        root_value = state.get("analysis_sources_root")
        if not root_value:
            return
        analysis_root = Path(str(root_value)).resolve()
        analysis_root.mkdir(parents=True, exist_ok=True)
        local_readme = analysis_root / "README.md"
        if local_readme.exists() and local_readme.read_bytes() != content:
            raise ValueError("analysis README changed after it was created")
        if not local_readme.exists():
            local_readme.write_bytes(content)
    @staticmethod
    def _link_evaluations(
        analysis: dict[str, Any],
        *,
        steps: list[int],
        selected_step: int,
        include_offline: bool,
    ) -> dict[str, Any]:
        result = copy.deepcopy(analysis)
        validate_analysis_profile(result)
        if result.get("profile") not in {"reward_design", "curriculum_learning"}:
            raise ValueError("RFT analysis profile must identify an RFT task")
        topology = result["topology"]
        declared = topology.get("steps")
        if not isinstance(declared, list):
            raise ValueError("RFT analysis topology steps must be a list")
        by_step: dict[int, dict[str, Any]] = {}
        for item in declared:
            if not isinstance(item, dict) or item.get("step") is None:
                raise ValueError("RFT analysis topology step is invalid")
            by_step[int(item["step"])] = dict(item)
        for step in steps:
            item = by_step.setdefault(step, {"step": step})
            item["online_eval_unit_id"] = (
                f"online-validation-step-{step:03d}"
            )
        topology["steps"] = [by_step[step] for step in sorted(by_step)]
        topology["offline_evaluation"] = (
            {
                "selected_step": selected_step,
                "result_unit_id": f"offline-validation-step-{selected_step:03d}",
            }
            if include_offline
            else None
        )
        return result

    @staticmethod
    def _evaluation_state(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("status") in {"failed", "partial"}:
            return {
                "status": str(payload["status"]),
                "ranking_score": None,
                "result": payload,
            }
        score = RFTExecutor._score(payload)
        return {
            "status": "complete",
            "ranking_score": score,
            "secondary_score": (
                float(payload["secondary_score"])
                if payload.get("secondary_score") is not None
                else None
            ),
            "score": score,
            "result": payload,
        }

    @staticmethod
    def _score(payload: dict[str, Any]) -> float:
        try:
            score = float(payload["score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("RFT evaluation requires numeric score") from error
        if score != score or abs(score) == float("inf"):
            raise ValueError("RFT evaluation score must be finite")
        return score


def _record_count(path: Path) -> int | None:
    if not path.name.endswith((".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")):
        return None
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as handle:
        return sum(1 for line in handle if line.strip())
