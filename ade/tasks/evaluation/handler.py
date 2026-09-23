"""Evaluation execution from a typed EvaluateCommand."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ade.core.engine import EvaluateCommand
from ade.engine.storage.object_store import FileEngineObjectStore
from ade.engine.requests import EvaluationInput, decode_evaluation_input
from ade.engine.telemetry.tracking import publish_evaluation_tracking

_PURPOSES = {
    "base_reference",
    "online_validation",
    "offline_validation",
    "operator_test",
}


class EvaluationBackend(Protocol):
    def evaluate(
        self,
        command: EvaluateCommand,
        config: EvaluationInput,
    ) -> dict[str, Any]: ...


class EvaluationExecutor:
    def __init__(
        self,
        *,
        io: FileEngineObjectStore,
        backend: EvaluationBackend,
        tracking_publisher: Callable[..., dict[str, Any]] = publish_evaluation_tracking,
    ) -> None:
        self.io = io
        self.backend = backend
        self.tracking_publisher = tracking_publisher

    def execute(self, command: EvaluateCommand) -> tuple[str, ...]:
        if type(command) is not EvaluateCommand:
            raise TypeError("EvaluationExecutor requires EvaluateCommand")
        config = decode_evaluation_input(self.io.read_json(command.input_ref))
        purpose = config.purpose
        if purpose not in _PURPOSES:
            raise ValueError(f"unsupported evaluation purpose: {purpose}")
        if not command.output_uri:
            raise ValueError("EvaluateCommand output_uri is required")
        payload = self.backend.evaluate(command, config)
        backend_status = str(payload.get("status") or "complete")
        status = "complete" if backend_status == "completed" else backend_status
        if status not in {"complete", "partial"}:
            raise ValueError("Evaluation backend status must be complete or partial")
        score: float | None = None
        if "score" in payload:
            try:
                score = float(payload["score"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Evaluation backend requires numeric score") from error
        elif status == "complete" and purpose != "operator_test":
            raise ValueError("Evaluation backend requires numeric score")
        result_uri = f"{command.output_uri}/result.json"
        manifest_uri = f"{command.output_uri}/raw/manifest.json"
        unit_uri = f"{command.output_uri}/raw/units/{purpose}.json"
        self.io.put_json(
            result_uri,
            {
                "schema_version": "1",
                "command_id": command.command_id,
                "coordinator_id": command.coordinator_id,
                "plan_id": command.plan_id,
                "trial_id": command.trial_id,
                "purpose": purpose,
                "status": status,
                "score": score,
            },
        )
        self.io.put_json(unit_uri, payload)
        self.io.put_json(
            manifest_uri,
            {
                "schema_version": "1",
                "command_id": command.command_id,
                "coordinator_id": command.coordinator_id,
                "plan_id": command.plan_id,
                "trial_id": command.trial_id,
                "units": [
                    {
                        "unit_id": purpose,
                        "kind": purpose,
                        "uri": unit_uri,
                        "visibility": "operator" if purpose == "operator_test" else "agent",
                    }
                ],
            },
        )
        refs = [result_uri, manifest_uri]
        tracking = config.request.get("evaluation_tracking")
        if isinstance(tracking, dict) and tracking.get("enabled"):
            tracking_result = self.tracking_publisher(
                settings=tracking,
                command_id=command.command_id,
                run_id=command.run_id,
                coordinator_id=command.coordinator_id,
                plan_id=command.plan_id,
                trial_id=command.trial_id,
                subject_kind=str(config.request["evaluation_subject_kind"]),
                purpose=purpose,
                result=payload,
                local_root=Path(str(config.request["run_dir"])) / "wandb-evaluations",
                artifact_position=config.request.get("artifact_position"),
                position_order=config.request.get(
                    "evaluation_tracking_position_order"
                ),
                fork_lineage=(
                    config.request.get("fork_lineage")
                    if isinstance(config.request.get("fork_lineage"), dict)
                    else None
                ),
                defer_online=False,
            )
            tracking_uri = f"{command.output_uri}/tracking.json"
            self.io.put_json(tracking_uri, tracking_result)
            refs.append(tracking_uri)
        return tuple(refs)
