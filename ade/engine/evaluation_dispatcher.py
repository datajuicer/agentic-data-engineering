"""Engine-owned submission and execution of checkpoint evaluations."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from ade.core.engine import (
    EngineAttemptError,
    EngineReceipt,
    EngineReceiptStatus,
    EvaluateCommand,
    TrainRFTCommand,
    TrainSFTCommand,
)
from ade.engine.command_queue import FileCommandQueue
from ade.engine.requests import decode_evaluation_input
from ade.engine.storage.object_store import FileEngineObjectStore
from ade.engine.telemetry.tracking import mark_evaluation_tracking_position_terminal


@dataclass(frozen=True)
class CheckpointEvaluationResult:
    command_id: str
    status: str
    score: float | None
    payload: dict[str, Any]
    output_refs: tuple[str, ...]
    error: str | None = None


class EngineEvaluationDispatcher:
    def __init__(
        self,
        *,
        queue: FileCommandQueue,
        io: FileEngineObjectStore,
        execute: Callable[[EvaluateCommand], tuple[str, ...]],
        heartbeat_interval_seconds: float = 30.0,
    ) -> None:
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        self.queue = queue
        self.io = io
        self.execute = execute
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)

    def evaluate_checkpoint(
        self,
        *,
        parent: TrainRFTCommand | TrainSFTCommand,
        checkpoint: str,
        purpose: str,
        step: int,
        position_unit: str,
        request: dict[str, Any],
    ) -> CheckpointEvaluationResult:
        if position_unit not in {"epoch", "rl_step"}:
            raise ValueError("evaluation position_unit must be epoch or rl_step")
        purpose_id = purpose.replace("_", "-")
        position_id = f"{position_unit.replace('_', '-')}-{int(step)}"
        command_id = f"{parent.command_id}-{purpose_id}-{position_id}"
        input_ref = f"engine://inputs/{command_id}.json"
        output_uri = f"{parent.output_uri}/evaluations/{purpose_id}/{position_id}"
        effective = dict(request)
        effective.update(
            {
                "checkpoint_path": checkpoint,
                "parent_command_id": parent.command_id,
                "artifact_position": {
                    "unit": position_unit,
                    "value": int(step),
                },
                "coordinator_resource_owner": (
                    f"{parent.run_id}/{parent.coordinator_id}"
                ),
                "ade_run_id": parent.run_id,
                "ade_coordinator_id": parent.coordinator_id,
                "ade_plan_id": parent.plan_id,
                "ade_trial_id": parent.trial_id,
                "ade_engine_command_id": command_id,
                "ade_workload": purpose,
            }
        )
        if position_unit == "epoch":
            position_path = Path(checkpoint) / "checkpoint_position.json"
            if position_path.is_file():
                position = json.loads(position_path.read_text(encoding="utf-8"))
                effective["artifact_position"]["train_step"] = int(position["train_step"])
        self.io.put_json(
            input_ref,
            {
                "schema_version": 1,
                "evaluation": {
                    "purpose": purpose,
                    "request": effective,
                },
            },
        )
        command = EvaluateCommand(
            command_id=command_id,
            run_id=parent.run_id,
            coordinator_id=parent.coordinator_id,
            plan_id=parent.plan_id,
            trial_id=parent.trial_id,
            input_ref=input_ref,
            output_uri=output_uri,
        )
        if not self.queue.has_receipt(command_id):
            self.queue.submit(command)
            claimed = self.queue.claim(command_id)
            if claimed != command:
                raise RuntimeError(
                    f"evaluation command is already processing without a receipt: {command_id}"
                )
            self._execute_and_publish(command)
        receipt = self.queue.load_receipt(command_id)
        if receipt.status is not EngineReceiptStatus.SUCCEEDED:
            cancelled = receipt.failure_kind == "evaluation_cancelled"
            return CheckpointEvaluationResult(
                command_id=command_id,
                status="cancelled" if cancelled else "failed",
                score=None,
                payload={},
                output_refs=receipt.output_refs,
                error=receipt.error or f"evaluation failed: {command_id}",
            )
        result = self.io.read_json(f"{output_uri}/result.json")
        payload = self.io.read_json(
            f"{output_uri}/raw/units/{purpose}.json"
        )
        status = str(result.get("status") or "complete")
        if status not in {"complete", "partial"}:
            raise ValueError(f"evaluation command status is invalid: {command_id}")
        score_value = result.get("score")
        score = float(score_value) if score_value is not None else None
        if status == "complete" and score is None:
            raise ValueError(
                f"evaluation command did not produce a numeric score: {command_id}"
            )
        return CheckpointEvaluationResult(
            command_id=command_id,
            status=status,
            score=score,
            payload=payload,
            output_refs=receipt.output_refs,
        )

    def _execute_and_publish(self, command: EvaluateCommand) -> None:
        stopped = threading.Event()
        heartbeat_errors: list[Exception] = []

        def maintain_heartbeat() -> None:
            while not stopped.wait(self.heartbeat_interval_seconds):
                try:
                    self.queue.heartbeat(command.command_id)
                except Exception as error:
                    heartbeat_errors.append(error)
                    stopped.set()

        heartbeat = threading.Thread(
            target=maintain_heartbeat,
            name=f"ade-evaluation-heartbeat-{command.command_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            try:
                output_refs = tuple(self.execute(command))
                if heartbeat_errors:
                    raise RuntimeError("evaluation heartbeat persistence failed") from heartbeat_errors[0]
                receipt = EngineReceipt(
                    receipt_id=f"receipt-{command.command_id}",
                    command_id=command.command_id,
                    run_id=command.run_id,
                    coordinator_id=command.coordinator_id,
                    plan_id=command.plan_id,
                    trial_id=command.trial_id,
                    status=EngineReceiptStatus.SUCCEEDED,
                    output_refs=output_refs,
                    logical_command_id=command.logical_command_id or command.command_id,
                    attempt_id=command.attempt_id,
                    attempt_index=command.attempt_index,
                )
            except Exception as error:
                failure_kind = (
                    error.failure_kind
                    if isinstance(error, EngineAttemptError)
                    else "evaluation_failed"
                )
                self._mark_tracking_position_failed(command, status=failure_kind)
                receipt = EngineReceipt(
                    receipt_id=f"receipt-{command.command_id}",
                    command_id=command.command_id,
                    run_id=command.run_id,
                    coordinator_id=command.coordinator_id,
                    plan_id=command.plan_id,
                    trial_id=command.trial_id,
                    status=EngineReceiptStatus.FAILED,
                    error=str(error),
                    logical_command_id=command.logical_command_id or command.command_id,
                    attempt_id=command.attempt_id,
                    attempt_index=command.attempt_index,
                    failure_kind=failure_kind,
                    retryable=(
                        error.retryable
                        if isinstance(error, EngineAttemptError)
                        else True
                    ),
                )
        finally:
            stopped.set()
            heartbeat.join()
        self.queue.publish_receipt(receipt)

    def _mark_tracking_position_failed(
        self,
        command: EvaluateCommand,
        *,
        status: str,
    ) -> None:
        try:
            config = decode_evaluation_input(self.io.read_json(command.input_ref))
            request = config.request
            mark_evaluation_tracking_position_terminal(
                settings=request.get("evaluation_tracking"),
                run_id=command.run_id,
                coordinator_id=command.coordinator_id,
                plan_id=command.plan_id,
                trial_id=command.trial_id,
                subject_kind=str(request["evaluation_subject_kind"]),
                purpose=config.purpose,
                local_root=(
                    Path(str(request.get("run_dir") or "."))
                    / "wandb-evaluations"
                ),
                artifact_position=request.get("artifact_position"),
                position_order=request.get("evaluation_tracking_position_order"),
                status=status,
            )
        except (KeyError, TypeError, ValueError):
            # Invalid evaluation input has no reliable tracking identity to release.
            return
