"""SFT execution from a typed TrainSFTCommand."""

from __future__ import annotations

import copy
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

from ade.core.engine import EngineAttemptError, TrainSFTCommand
from ade.core.ranking import score_pair_sort_key
from ade.core.dotenv import read_dotenv_value
from ade.core.experiment import analysis_profile
from ade.engine.checkpoints.cleanup import cleanup_checkpoint_directories
from ade.tasks.data_selection.realizer import SelectionRealization, SelectionRealizer
from ade.engine.judge_dispatcher import SelectionJudgeBatchDispatcher
from ade.local_rubric_judge import HttpRubricJobGateway, LocalRubricJudgeClient
from ade.rubric_jobs.process import canonical_process_rubric
from ade.engine.evaluation_dispatcher import EngineEvaluationDispatcher
from ade.engine.storage.object_store import FileEngineObjectStore
from ade.engine.requests import SFTInput, decode_sft_input
from ade.engine.storage.atomic import write_json_atomic
from ade.engine.trial_artifacts import (
    ArtifactRecord,
    TrialArtifactPublisher,
    TrialArtifactWorkspace,
    remove_generated_wandb_symlinks,
    stage_training_telemetry_artifacts,
    write_evaluation_artifacts,
    write_sft_selection_audit,
)


class SFTBackend(Protocol):
    def train(
        self,
        command: TrainSFTCommand,
        config: SFTInput,
    ) -> dict[str, Any]: ...


def _selection_rubric_json(rubric: object) -> str:
    """Validate the shared canonical process-rubric declaration."""
    return canonical_process_rubric(rubric)


class SFTExecutor:
    def __init__(
        self,
        *,
        io: FileEngineObjectStore,
        backend: SFTBackend,
        evaluations: EngineEvaluationDispatcher | None = None,
        artifacts: TrialArtifactPublisher | None = None,
        selection_realizer: SelectionRealizer | None = None,
    ) -> None:
        self.io = io
        self.backend = backend
        self.evaluations = evaluations
        self.artifacts = artifacts
        self.selection_realizer = selection_realizer or SelectionRealizer()

    def execute(self, command: TrainSFTCommand) -> tuple[str, ...]:
        if type(command) is not TrainSFTCommand:
            raise TypeError("SFTExecutor requires TrainSFTCommand")
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
            decode_sft_input(self.io.read_json(command.input_ref)),
            command=command,
            workspace=workspace,
        )
        if workspace is not None:
            dataset = config.sft["dataset"]
            prompt_contract = config.sft["request"].get(
                "training_prompt_contract"
            )
            if not isinstance(prompt_contract, dict):
                raise ValueError(
                    "SFT artifact package requires training_prompt_contract"
                )
            prompt_contract_path = (
                workspace.audit / "training_prompt_contract.json"
            )
            write_json_atomic(prompt_contract_path, prompt_contract)
            artifact_records.append(
                ArtifactRecord(
                    artifact_id="sft-training-prompt-contract",
                    category="audit",
                    kind="training_prompt_contract",
                    path=prompt_contract_path.relative_to(
                        workspace.root
                    ).as_posix(),
                )
            )
            artifact_records.extend(
                write_sft_selection_audit(
                    workspace,
                    selection_path=self.io.path_for(
                        str(dataset.get("selection_ref") or config.artifact_ref)
                    ),
                    selected_data_path=self.io.path_for(str(dataset["data_ref"])),
                    candidate_pool_path=(
                        self.io.path_for(str(dataset["candidate_pool_ref"]))
                        if dataset.get("candidate_pool_ref")
                        else None
                    ),
                )
            )
        if not command.output_uri:
            raise ValueError("SFT output_uri is required")
        result_uri = f"{command.output_uri}/result.json"
        manifest_uri = f"{command.output_uri}/raw/manifest.json"
        metrics_uri = f"{command.output_uri}/raw/units/training-metrics.json"
        if not self.io.exists(result_uri):
            train_with_checkpoints = getattr(
                self.backend,
                "train_with_checkpoints",
                None,
            )
            requests = config.sft.get("evaluation_requests")
            try:
                if (
                    callable(train_with_checkpoints)
                    and (self.evaluations is not None or workspace is not None)
                    and isinstance(requests, dict)
    ):
                    payload = train_with_checkpoints(
                        command,
                        config,
                        lambda step, checkpoint: self._dispatch_online(
                            command,
                            config,
                            checkpoint,
                            step,
                            workspace=workspace,
                            artifact_records=artifact_records,
                        ),
                    )
                else:
                    payload = self.backend.train(command, config)
            except Exception as error:
                if workspace is None or self.artifacts is None:
                    raise
                output_refs = self._publish_training_failure(
                    command,
                    config,
                    workspace,
                    records=artifact_records,
                    error=error,
                )
                message = (
                    "SFT training failed; failure evidence published at "
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
            model_ref = str(payload.get("model_ref") or "")
            metrics = payload.get("metrics")
            if not model_ref or not isinstance(metrics, dict):
                raise ValueError("SFT backend requires model_ref and metrics")
            if workspace is not None and self.artifacts is not None:
                telemetry_manifest = metrics.get("training_telemetry_manifest_path")
                if telemetry_manifest and Path(str(telemetry_manifest)).is_file():
                    artifact_records.extend(
                        stage_training_telemetry_artifacts(
                            workspace,
                            str(telemetry_manifest),
                            position_unit="epoch",
                            target_namespace="sft",
                        )
                    )
            checkpoints = [
                str(value)
                for value in payload.get("checkpoints", ())
                if str(value)
            ]
            if not checkpoints:
                checkpoints = [model_ref]
            evaluations = self._evaluate_checkpoints(
                command,
                config,
                checkpoints,
                online=list(payload.get("online_evaluations") or ()),
                early_stopped=bool(metrics.get("early_stopped")),
                early_stop_epoch=metrics.get("early_stop_epoch"),
                workspace=workspace,
                artifact_records=artifact_records,
            )
            if (
                evaluations is not None
                and evaluations["selected_checkpoint"] is not None
            ):
                model_ref = evaluations["selected_checkpoint"]
            if evaluations is not None:
                self._cleanup_checkpoints(
                    checkpoints,
                    evaluations,
                    top_k=int(
                        config.sft.get("checkpoint_retention_top_k", -1)
                    ),
                )
            self.io.put_json(metrics_uri, metrics)
            if workspace is not None:
                metrics_path = workspace.audit / "metrics" / "training_metrics.json"
                write_json_atomic(metrics_path, metrics)
                artifact_records.append(
                    ArtifactRecord(
                        artifact_id="sft-training-metrics",
                        category="audit",
                        kind="training_metrics",
                        path=metrics_path.relative_to(workspace.root).as_posix(),
                    )
                )
            units = [
                {
                    "unit_id": "training-metrics",
                    "kind": "training_metrics",
                    "uri": metrics_uri,
                    "filename": "training-metrics.json",
                    "media_type": "application/json",
                    "visibility": "agent",
                }
            ]
            if evaluations is not None:
                for item in evaluations["online"]:
                    unit_uri = (
                        f"{command.output_uri}/raw/units/"
                        f"online-validation-step-{item['step']:03d}.json"
                    )
                    self.io.put_json(unit_uri, item["payload"])
                    units.append(
                        {
                            "unit_id": f"online-validation-step-{item['step']:03d}",
                            "kind": "online_validation",
                            "uri": unit_uri,
                            "filename": "online-eval-result.json",
                            "media_type": "application/json",
                            "visibility": "agent",
                            "step": item["step"],
                        }
                    )
                if evaluations["offline"].get("status") == "complete":
                    offline_uri = (
                        f"{command.output_uri}/raw/units/"
                        f"offline-validation-step-{evaluations['selected_step']:03d}.json"
                    )
                    self.io.put_json(
                        offline_uri,
                        evaluations["offline"]["payload"],
                    )
                    units.append(
                        {
                            "unit_id": (
                                "offline-validation-step-"
                                f"{evaluations['selected_step']:03d}"
                            ),
                            "kind": "offline_validation",
                            "uri": offline_uri,
                            "filename": "offline-eval-result.json",
                            "media_type": "application/json",
                            "visibility": "agent",
                            "step": evaluations["selected_step"],
                        }
                    )
            trial_manifest_path = None
            if workspace is not None and self.artifacts is not None:
                remove_generated_wandb_symlinks(workspace.audit)
                for checkpoint in checkpoints:
                    epoch = _checkpoint_step(
                        checkpoint,
                        fallback=checkpoints.index(checkpoint) + 1,
                    )
                    checkpoint_path = Path(checkpoint)
                    artifact_records.append(
                        ArtifactRecord(
                            artifact_id=f"checkpoint-epoch-{epoch:03d}",
                            category="checkpoint",
                            kind="hf_checkpoint",
                            status=(
                                "complete"
                                if checkpoint_path.is_dir()
                                else "pruned"
                            ),
                            path=(
                                checkpoint_path.relative_to(
                                    workspace.root
                                ).as_posix()
                                if checkpoint_path.is_dir()
                                else None
                            ),
                            metadata={
                                "artifact_position": {
                                    "unit": "epoch",
                                    "value": epoch,
                                }
                            },
                        )
                    )
                failed = evaluations is not None and (
                    any(
                        item.get("status") != "complete"
                        for item in evaluations["online"]
                    )
                    or (
                        config.sft.get("run_offline_validation", True)
                        and evaluations["offline"].get("status") != "complete"
                    )
                )
                trial_manifest_path = str(
                    self.artifacts.publish(
                        workspace,
                        trial_status=(
                            "completed_with_failures" if failed else "completed"
                        ),
                        selected_checkpoint_id=(
                            f"checkpoint-epoch-{evaluations['selected_step']:03d}"
                            if evaluations is not None
                            and evaluations["selected_step"] is not None
                            else None
                        ),
                        records=tuple(artifact_records),
                        metadata={
                            "command_id": command.command_id,
                            "position_unit": "epoch",
                            "artifact_interval": int(
                                config.sft.get("artifact_interval") or 1
                            ),
                        },
                    )
                )
            self.io.put_json(
                manifest_uri,
                {
                    "schema_version": "2",
                    "command_id": command.command_id,
                    "coordinator_id": command.coordinator_id,
                    "plan_id": command.plan_id,
                    "trial_id": command.trial_id,
                    "units": units,
                    "analysis": analysis_profile(
                        "data_selection",
                        version=1,
                        topology={
                            "selection": None,
                            "training": {
                                "metrics_unit_id": "training-metrics",
                                "telemetry_artifact_ids": [
                                    record.artifact_id
                                    for record in artifact_records
                                    if record.kind.startswith("training_telemetry")
                                ],
                            },
                            "evaluation": (
                                {
                                    "selected_step": evaluations["selected_step"],
                                    "online_unit_ids": [
                                        f"online-validation-step-{item['step']:03d}"
                                        for item in evaluations["online"]
                                    ],
                                    "offline_unit_id": (
                                        "offline-validation-step-"
                                        f"{evaluations['selected_step']:03d}"
                                        if evaluations["selected_step"]
                                        is not None
                                        and evaluations["offline"].get(
                                            "status"
                                        )
                                        == "complete"
                                        else None
                                    ),
                                }
                                if evaluations is not None
                                else None
                            ),
                        },
                    ),
                },
            )
            self.io.put_json(
                result_uri,
                {
                    "schema_version": "1",
                    "command_id": command.command_id,
                    "coordinator_id": command.coordinator_id,
                    "plan_id": command.plan_id,
                    "trial_id": command.trial_id,
                    "model_ref": model_ref,
                    "selected_position": (
                        {
                            "unit": "epoch",
                            "value": evaluations["selected_step"],
                        }
                        if evaluations is not None
                        and evaluations["selected_step"] is not None
                        else None
                    ),
                    "metrics": metrics,
                    "online_validation": (
                        evaluations["selected_online"]
                        if evaluations is not None
                        else None
                    ),
                    "offline_validation": (
                        evaluations["offline"]
                        if evaluations is not None
                        else None
                    ),
                    "trial_artifact_manifest_path": trial_manifest_path,
                },
            )
        if not self.io.exists(manifest_uri):
            raise RuntimeError("SFT result exists without its raw manifest")
        return result_uri, manifest_uri

    def _publish_training_failure(
        self,
        command: TrainSFTCommand,
        config: SFTInput,
        workspace: TrialArtifactWorkspace,
        *,
        records: list[ArtifactRecord],
        error: Exception,
    ) -> tuple[str, ...]:
        request = config.sft.get("request")
        telemetry_manifest = (
            Path(str(request.get("run_dir")))
            / "analysis_sources"
            / "training_telemetry_manifest.json"
            if isinstance(request, dict) and request.get("run_dir")
            else None
        )
        if (
            telemetry_manifest is not None
            and telemetry_manifest.is_file()
            and self.artifacts is not None
        ):
            records.extend(
                stage_training_telemetry_artifacts(
                    workspace,
                    telemetry_manifest,
                    position_unit="epoch",
                    target_namespace="sft",
                )
            )
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
            "online-validation-epoch-*.json"
        ):
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            position = payload.get("artifact_position") or {}
            epoch = int(position.get("value") or 0)
            checkpoint = workspace.checkpoints / f"epoch-{epoch:03d}"
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
                        epoch,
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
            _, _, selected_epoch, selected_checkpoint = selected
            requests = config.sft.get("evaluation_requests") or {}
            offline = self._evaluate(
                command,
                config,
                checkpoint=str(selected_checkpoint),
                purpose="offline_validation",
                step=selected_epoch,
                request=dict(requests.get("offline_validation") or {}),
                workspace=workspace,
                artifact_records=records,
            )
        else:
            selected_epoch = None
            selected_checkpoint = None
            offline = {
                "status": "skipped",
                "score": None,
                "ranking_score": None,
                "payload": {},
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
        checkpoints = sorted(
            (
                path
                for path in workspace.checkpoints.glob("epoch-*")
                if path.is_dir()
            ),
            key=lambda path: path.name,
        )
        for checkpoint in checkpoints:
            epoch = _checkpoint_step(str(checkpoint), fallback=0)
            records.append(
                ArtifactRecord(
                    artifact_id=f"checkpoint-epoch-{epoch:03d}",
                    category="checkpoint",
                    kind="hf_checkpoint",
                    path=checkpoint.relative_to(workspace.root).as_posix(),
                    metadata={
                        "artifact_position": {
                            "unit": "epoch",
                            "value": epoch,
                        }
                    },
                )
            )
        remove_generated_wandb_symlinks(workspace.audit)
        trial_manifest = self.artifacts.publish(
            workspace,
            trial_status="failed",
            selected_checkpoint_id=(
                f"checkpoint-epoch-{selected_epoch:03d}"
                if selected_epoch is not None
                else None
            ),
            records=tuple(records),
            metadata={
                "command_id": command.command_id,
                "position_unit": "epoch",
                "artifact_interval": int(
                    config.sft.get("artifact_interval") or 1
                ),
                "training_error": str(error),
            },
        )
        model_ref = (
            str(selected_checkpoint)
            if selected_checkpoint is not None
            else (str(checkpoints[-1]) if checkpoints else None)
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
                "model_ref": model_ref,
                "selected_position": (
                    {"unit": "epoch", "value": selected_epoch}
                    if selected_epoch is not None
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
                "analysis": analysis_profile(
                    "data_selection",
                    version=1,
                    topology={
                        "selection": None,
                        "training": {
                            "telemetry_artifact_ids": [
                                record.artifact_id
                                for record in records
                                if record.kind.startswith("training_telemetry")
                            ],
                        },
                        "evaluation": None,
                    },
                ),
            },
        )
        return result_uri, manifest_uri

    def _evaluate_checkpoints(
        self,
        command: TrainSFTCommand,
        config: SFTInput,
        checkpoints: list[str],
        online: list[dict[str, Any]],
        early_stopped: bool = False,
        early_stop_epoch: object = None,
        workspace: TrialArtifactWorkspace | None = None,
        artifact_records: list[ArtifactRecord] | None = None,
    ) -> dict[str, Any] | None:
        requests = config.sft.get("evaluation_requests")
        if (
            self.evaluations is None
            and workspace is None
        ) or not isinstance(requests, dict):
            return None
        online_request = requests.get("online_validation")
        offline_request = requests.get("offline_validation")
        run_offline = bool(config.sft.get("run_offline_validation", True))
        if not isinstance(online_request, dict) or (
            run_offline and not isinstance(offline_request, dict)
        ):
            raise ValueError(
                "sft.evaluation_requests requires online_validation and "
                "offline_validation"
            )
        if not early_stopped:
            evaluated = {str(item["checkpoint"]) for item in online}
            for index, checkpoint in enumerate(checkpoints, start=1):
                if checkpoint in evaluated:
                    continue
                step = _checkpoint_step(checkpoint, fallback=index)
                online.append(
                    self._dispatch_online(
                        command,
                        config,
                        checkpoint,
                        step,
                        workspace=workspace,
                        artifact_records=artifact_records,
                    )
                )
        horizon = int(early_stop_epoch) if early_stop_epoch is not None else None
        rankable = [
            item
            for item in online
            if item.get("status") == "complete"
            and (horizon is None or int(item["step"]) <= horizon)
        ]
        if not rankable:
            skipped = {
                "status": "skipped",
                "score": None,
                "ranking_score": None,
                "payload": {},
                "command_id": None,
                "reason": "no_complete_online_validation",
            }
            if workspace is not None and artifact_records is not None:
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
            return {
                "selected_step": None,
                "selected_checkpoint": None,
                "selected_online": None,
                "early_stop_epoch": horizon,
                "online": online,
                "offline": skipped,
            }
        selected = min(
            rankable,
            key=lambda item: (
                *score_pair_sort_key(
                    float(item["ranking_score"]),
                    (
                        float(item["secondary_score"])
                        if item.get("secondary_score") is not None
                        else None
                    ),
                    direction="maximize",
                ),
                int(item["step"]),
            ),
        )
        offline = (
            self._evaluate(
                command,
                config,
                checkpoint=str(selected["checkpoint"]),
                purpose="offline_validation",
                step=int(selected["step"]),
                request=offline_request,
                workspace=workspace,
                artifact_records=artifact_records,
            )
            if run_offline
            else {
                "status": "skipped",
                "score": None,
                "ranking_score": None,
                "payload": {},
                "command_id": None,
                "reason": "calibration_online_only",
            }
        )
        return {
            "selected_step": int(selected["step"]),
            "selected_checkpoint": str(selected["checkpoint"]),
            "selected_online": selected,
            "early_stop_epoch": horizon,
            "online": online,
            "offline": offline,
        }

    @staticmethod
    def _cleanup_checkpoints(
        checkpoints: list[str],
        evaluations: dict[str, Any],
        *,
        top_k: int,
    ) -> None:
        if top_k < -1:
            raise ValueError(
                "sft.checkpoint_retention_top_k must be -1 or non-negative"
            )
        selected = evaluations.get("selected_checkpoint")
        if top_k == -1 and selected is not None:
            return
        horizon = evaluations.get("early_stop_epoch")
        successful = [
            item
            for item in evaluations["online"]
            if item.get("status") == "complete"
            and (horizon is None or int(item["step"]) <= int(horizon))
        ]
        ranked = sorted(
            successful,
            key=lambda item: (
                *score_pair_sort_key(
                    float(item["ranking_score"]),
                    (
                        float(item["secondary_score"])
                        if item.get("secondary_score") is not None
                        else None
                    ),
                    direction="maximize",
                ),
                int(item["step"]),
            ),
        )
        retained = (
            {str(item["checkpoint"]) for item in ranked[:top_k]}
            if top_k >= 0
            else set(checkpoints)
        )
        if selected is not None:
            retained.add(str(selected))
        else:
            retained = {checkpoints[-1]}
        paths = [Path(value) for value in checkpoints]
        roots = {path.resolve().parent for path in paths}
        if len(roots) != 1:
            raise ValueError("SFT checkpoints must share one directory")
        cleanup_checkpoint_directories(
            [
                path
                for path in paths
                if str(path) not in retained
            ],
            allowed_root=roots.pop(),
            name_pattern=r"(?:checkpoint-|epoch-)\d+",
        )

    def _dispatch_online(
        self,
        command: TrainSFTCommand,
        config: SFTInput,
        checkpoint: str,
        step: int,
        *,
        workspace: TrialArtifactWorkspace | None = None,
        artifact_records: list[ArtifactRecord] | None = None,
    ) -> dict[str, Any]:
        requests = config.sft.get("evaluation_requests")
        if not isinstance(requests, dict) or not isinstance(
            requests.get("online_validation"), dict
        ):
            raise ValueError(
                "sft.evaluation_requests.online_validation is required"
            )
        evaluation_request = dict(requests["online_validation"])
        training_request = config.sft.get("request")
        if isinstance(training_request, dict) and training_request.get("stop_file"):
            evaluation_request["cancellation_file"] = str(
                training_request["stop_file"]
            )
        result = self._evaluate(
            command,
            config,
            checkpoint=checkpoint,
            purpose="online_validation",
            step=step,
            request=evaluation_request,
            workspace=workspace,
            artifact_records=artifact_records,
        )
        return {
            "step": step,
            "checkpoint": checkpoint,
            **result,
        }

    def _localize_input(
        self,
        config: SFTInput,
        *,
        command: TrainSFTCommand,
        workspace: TrialArtifactWorkspace | None = None,
    ) -> SFTInput:
        dataset = config.sft.get("dataset")
        request = config.sft.get("request")
        if not isinstance(dataset, dict) or not isinstance(request, dict):
            raise ValueError("SFT input requires bound sft.dataset and sft.request")
        dataset = self._materialize_selection(config, dataset, command=command)
        data_ref = str(dataset.get("data_ref") or "")
        info_ref = str(dataset.get("info_ref") or "")
        dataset_name = str(dataset.get("dataset_name") or "")
        if not data_ref or not info_ref or not dataset_name:
            raise ValueError("SFT bound dataset refs and name are required")
        data_path = self.io.path_for(data_ref)
        info_path = self.io.path_for(info_ref)
        if data_path.parent != info_path.parent:
            raise ValueError("SFT dataset data and info must share an Engine directory")
        localized_request = copy.deepcopy(request)
        localized_request.update(
            {
                "data_file": str(data_path),
                "dataset_dir": str(info_path.parent),
                "dataset_name": dataset_name,
                "coordinator_resource_owner": (
                    f"{command.run_id}/{command.coordinator_id}"
                ),
            }
        )
        if workspace is not None:
            control_dir = workspace.audit / "sft-control"
            localized_request.update(
                {
                    "checkpoint_output_dir": str(workspace.checkpoints),
                    "run_dir": str(workspace.audit),
                    "stop_file": str(control_dir / "STOP"),
                }
            )
        sft = copy.deepcopy(config.sft)
        sft["dataset"] = dataset
        sft["request"] = localized_request
        if workspace is not None:
            sft["trial_artifact_root"] = str(workspace.root)
        raw = copy.deepcopy(config.raw)
        raw["sft"] = sft
        return SFTInput(config.artifact_ref, sft, raw)

    def _materialize_selection(
        self,
        config: SFTInput,
        dataset: dict[str, Any],
        *,
        command: TrainSFTCommand,
    ) -> dict[str, Any]:
        inventory_ref = str(dataset.get("candidate_inventory_ref") or "")
        training_ref = str(dataset.get("training_data_ref") or "")
        if not inventory_ref and not training_ref:
            return dataset
        if dataset.get("realization_status") == "final":
            raise ValueError(
                "final SFT training binding cannot retain selector source refs"
            )
        if not inventory_ref or not training_ref:
            raise ValueError("SFT selection source refs must be provided together")
        select_size = dataset.get("select_size")
        script = self.io.read_bytes(config.artifact_ref)
        inventory = self.io.read_bytes(inventory_ref)
        training = self.io.read_bytes(training_ref)
        enrichment = config.sft.get("request", {}).get("judge_enrichment", {})
        enabled = enrichment.get("enabled") if isinstance(enrichment, dict) else None
        if type(enabled) is not bool:
            raise ValueError("SFT request judge_enrichment.enabled must be explicit")
        if not enabled:
            realized = asyncio.run(
                self.selection_realizer.realize(
                    script,
                    inventory,
                    training,
                    select_size=select_size,
                )
            )
            selection = realized.selection
            selected_rows = list(realized.selected_rows)
        else:
            selection, selected_rows = self._execute_judged_selection(
                config, command, script, inventory, training,
                select_size=select_size,
            )
        dataset_name = str(dataset.get("dataset_name") or "")
        return self.selection_realizer.publish_final_binding(
            self.io,
            SelectionRealization(selection, tuple(selected_rows)),
            root_uri=f"{config.artifact_ref.rsplit('/', 1)[0]}/dataset",
            dataset_name=dataset_name,
            candidate_pool_ref=training_ref,
            pool_stats=copy.deepcopy(dataset.get("pool_stats") or {}),
        )

    def _execute_judged_selection(
        self,
        config: SFTInput,
        command: TrainSFTCommand,
        script: bytes,
        inventory: bytes,
        training: bytes,
        *,
        select_size: int,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        request = config.sft.get("request")
        local = request.get("local_judge") if isinstance(request, dict) else None
        if not isinstance(local, dict):
            raise ValueError("enabled SFT Judge enrichment requires request.local_judge")
        auth_env = str(local.get("authorization_env") or "")
        authorization = os.environ.get(auth_env) or read_dotenv_value(auth_env)
        if not auth_env or not authorization:
            raise ValueError("SFT Judge authorization environment is unavailable")
        request_timeout = float(
            (local.get("timeout_policy") or {}).get("request_timeout_seconds") or 0
        )
        if request_timeout <= 0:
            raise ValueError("SFT Judge timeout_policy.request_timeout_seconds is required")
        client = LocalRubricJudgeClient(HttpRubricJobGateway(
            str(local.get("gateway_url") or ""),
            authorization=authorization,
            timeout_seconds=request_timeout,
        ))
        metadata = {
            "run_id": command.run_id,
            "engine_command_id": command.command_id,
            "scope": {"run_id": command.run_id, "trial_id": command.trial_id},
            "subject_ref": f"{command.run_id}/{command.trial_id}",
            "phase": "sft_selection_pretraining",
        }

        async def run() -> tuple[dict[str, object], list[dict[str, object]]]:
            vllm = local.get("vllm") if isinstance(local.get("vllm"), dict) else {}
            max_batch_size = int(vllm.get("max_num_seqs") or 1)
            dispatcher = SelectionJudgeBatchDispatcher(
                client,
                max_batch_size=max_batch_size,
                submission_id=f"{command.run_id}:{command.command_id}:selection",
                job_metadata=metadata,
            )

            async def judge_batch(
                requests: list[dict[str, object]],
            ) -> list[dict[str, object]]:
                normalized = []
                for request in requests:
                    if not isinstance(request, dict):
                        raise ValueError("selection Judge request must be an object")
                    normalized.append(
                        {
                            "question": str(request["question"]),
                            "response": str(request["response"]),
                            "rubric": _selection_rubric_json(request["rubric"]),
                        }
                    )
                return await dispatcher.evaluate(normalized)

            try:
                realized = await self.selection_realizer.realize(
                    script,
                    inventory,
                    training,
                    select_size=select_size,
                    judge_batch=judge_batch,
                )
                selection = realized.selection
                selection["judge_enrichment"] = dispatcher.stats
                return selection, list(realized.selected_rows)
            finally:
                await dispatcher.close()

        return asyncio.run(run())

    def _evaluate(
        self,
        command: TrainSFTCommand,
        config: SFTInput,
        *,
        checkpoint: str,
        purpose: str,
        step: int,
        request: dict[str, Any],
        workspace: TrialArtifactWorkspace | None,
        artifact_records: list[ArtifactRecord] | None,
    ) -> dict[str, Any]:
        try:
            if self.evaluations is None:
                evaluate = getattr(self.backend, "evaluate", None)
                if not callable(evaluate):
                    raise RuntimeError(
                        "SFT checkpoint evaluation dispatcher is unavailable"
                    )
                payload = evaluate(
                    command,
                    config,
                    checkpoint,
                    purpose,
                    step,
                )
                score = (
                    None
                    if payload.get("status") == "partial"
                    else float(payload["score"])
                )
                command_id = None
            else:
                evaluation_request = dict(request)
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
                    position_unit="epoch",
                    request=evaluation_request,
                )
                if dispatched.status == "cancelled":
                    return {
                        "status": "cancelled",
                        "score": None,
                        "ranking_score": None,
                        "payload": {},
                        "command_id": dispatched.command_id,
                        "reason": "early_stopping",
                    }
                if dispatched.status not in {"complete", "partial"}:
                    raise RuntimeError(
                        dispatched.error
                        or f"{purpose} evaluation failed at epoch {step}"
                    )
                payload = dict(dispatched.payload)
                score = (
                    float(dispatched.score)
                    if dispatched.status == "complete"
                    and dispatched.score is not None
                    else None
                )
                command_id = dispatched.command_id
                payload["status"] = dispatched.status
            status = str(payload.get("status") or "complete")
            if status not in {"complete", "partial"}:
                raise ValueError(f"unsupported evaluation status: {status}")
            if status == "complete" and score is None:
                score = float(payload["score"])
            if status == "partial":
                score = None
            if workspace is not None and artifact_records is not None:
                artifact_records.extend(
                    write_evaluation_artifacts(
                        workspace,
                        purpose=purpose,
                        position_unit="epoch",
                        position_value=step,
                        checkpoint_artifact_id=(
                            f"checkpoint-epoch-{step:03d}"
                        ),
                        ranking_score=score,
                        payload=payload,
                        status=status,
                    )
                )
            return {
                "status": status,
                "score": score,
                "ranking_score": score,
                "secondary_score": (
                    float(payload["secondary_score"])
                    if status == "complete"
                    and payload.get("secondary_score") is not None
                    else None
                ),
                "payload": payload,
                "command_id": command_id,
            }
        except Exception as error:
            if workspace is None:
                raise
            failure = {
                "category": "evaluation_runtime_error",
                "message": str(error),
            }
            if artifact_records is not None:
                artifact_records.append(
                    ArtifactRecord(
                        artifact_id=(
                            f"{purpose.replace('_', '-')}-epoch-{step:03d}"
                        ),
                        category="eval_result",
                        kind=purpose,
                        status="failed",
                        metadata={"error": failure},
                    )
                )
            return {
                "status": "failed",
                "score": None,
                "ranking_score": None,
                "payload": {},
                "command_id": None,
                "error": failure,
            }


def _checkpoint_step(value: str, *, fallback: int) -> int:
    match = re.fullmatch(r"(?:checkpoint-|epoch-)(\d+)", Path(value).name)
    return int(match.group(1)) if match is not None else fallback
