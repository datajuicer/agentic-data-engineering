"""Production evaluation backend for typed EvaluateCommand execution."""

from __future__ import annotations

import os
from typing import Any

from ade.core.engine import EvaluateCommand
from ade.engine.execution.ray import run_eval_with_gpu_lease
from ade.engine.requests import EvaluationInput


class VllmEvaluationBackend:
    def evaluate(
        self,
        command: EvaluateCommand,
        config: EvaluationInput,
    ) -> dict[str, Any]:
        effective = dict(config.request)
        if "checkpoint_path" not in effective and "checkpoint" in effective:
            effective["checkpoint_path"] = effective["checkpoint"]
        worker_id = os.environ.get("ADE_GENERALIZATION_WORKER_ID")
        if worker_id and effective.get("evaluation_subject_kind") == "standalone_checkpoint":
            effective["coordinator_resource_owner"] = f"{command.run_id}/{worker_id}"
        effective["ade_run_id"] = command.run_id
        effective["ade_engine_command_id"] = command.command_id
        effective["staging_consumer_id"] = command.command_id
        effective["phase"] = config.purpose
        effective["result_suffix"] = command.command_id
        effective["gpu_lease_owner_suffix"] = (
            command.logical_command_id or command.command_id
        )
        result = run_eval_with_gpu_lease(effective)
        if config.purpose in {"offline_validation", "online_validation"} and "score" not in result:
            raise ValueError("production evaluation did not produce a ranking score")
        return result
