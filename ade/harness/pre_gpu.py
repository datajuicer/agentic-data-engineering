"""No-GPU orchestration for real-Agent Pre-GPU gates.

The Agent boundary is production Codex submit/poll. Engine commands are
executed by deterministic CPU/filesystem handlers and never reach Ray, VERL,
LlamaFactory, vLLM, or a GPU runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ade.core.engine import EvaluateCommand
from ade.core.bootstrap import BootstrapStatus
from ade.core.run import RunState, RunStatus
from ade.tasks.data_selection.handler import SFTExecutor
from ade.tasks.reward_design.contracts import RFTCheckpointResult, RFTTrainingResult
from ade.tasks.reward_design.handler import RFTExecutor
from ade.engine.storage.object_store import FileEngineObjectStore
from ade.engine.trial_artifacts import TrialArtifactPublisher
from ade.engine.worker import EngineWorker
from ade.harness.experiment_config import ExperimentConfigCompiler
from ade.harness.processes import AgentProcessRunner
from ade.harness.wiring import build_agent_process, build_control_process
from ade.tasks.registry import default_task_registry


class _ScriptedSFTBackend:
    def train(self, command, config) -> dict[str, Any]:
        checkpoint = (
            Path(config.sft["trial_artifact_root"])
            / "checkpoints"
            / "checkpoint-1"
        )
        checkpoint.mkdir(parents=True, exist_ok=True)
        return {
            "model_ref": str(checkpoint),
            "metrics": {"loss": 0.2, "scripted_pre_gpu": True},
        }

    def evaluate(self, command, config, checkpoint, purpose, step):
        return {
            "status": "complete",
            "score": 0.8 if purpose == "online_validation" else 0.7,
            "checkpoint": checkpoint,
            "step": step,
            "scripted_pre_gpu": True,
        }


class _ScriptedRFTBackend:
    """CPU-only typed RFT backend used by the same Pre-GPU control loop."""

    def train(self, command, config, steps, evaluate_checkpoint):
        checkpoints = []
        for step in steps:
            checkpoint = (
                Path(config.rft["trial_artifact_root"])
                / "checkpoints"
                / f"rl-step-{step:03d}"
            )
            checkpoint.mkdir(parents=True, exist_ok=True)
            checkpoints.append(
                RFTCheckpointResult(
                    step=step,
                    checkpoint_path=str(checkpoint),
                    online_validation=evaluate_checkpoint(step, str(checkpoint)),
                )
            )
        return RFTTrainingResult(tuple(checkpoints))

    def evaluate(self, command, config, checkpoint, purpose, step):
        return {
            "status": "complete",
            "score": 0.8 if purpose == "online_validation" else 0.7,
            "checkpoint": checkpoint,
            "step": step,
            "scripted_pre_gpu": True,
        }


class PreGpuGateRunner:
    def __init__(
        self,
        *,
        project_root: str | Path,
        experiment_config: str | Path,
        deployment_config: str | Path,
        run_id: str,
        gate_root: str | Path,
    ) -> None:
        self.project = Path(project_root).resolve()
        config = Path(experiment_config)
        if not config.is_absolute():
            config = self.project / config
        self.compiled = ExperimentConfigCompiler(
            default_task_registry(),
            project_root=self.project,
        ).compile_file(
            config,
            deployment_config=deployment_config,
            run_id=run_id,
        )
        task_id = self.compiled.run.task.task_id
        if task_id not in {
            "data_selection",
            "reward_design",
            "curriculum_learning",
        }:
            raise ValueError(f"real-Agent Pre-GPU runner does not support task {task_id}")
        self.run_id = run_id
        self.root = Path(gate_root).resolve()
        self.runs_root = self.root / "control"
        self.queue_root = self.root / "queue"
        self.object_root = self.root / "objects"
        self.control = build_control_process(
            project_root=self.project,
            runs_root=self.runs_root,
            queue_root=self.queue_root,
            object_root=self.object_root,
            engine_inputs=self.compiled.engine_inputs,
            bootstrap_contract=self.compiled.bootstrap,
            agent_config=self.compiled.control["agent"],
            artifact_builder_config=self.compiled.control["artifact_builder"],
            engine_config=self.compiled.control["engine"],
        )
        self.agent = build_agent_process(
            project_root=self.project,
            runs_root=self.runs_root,
            agent_config=self.compiled.control["agent"],
            artifact_builder_config=self.compiled.control["artifact_builder"],
        )
        self.agent_runner = AgentProcessRunner(
            process=self.agent,
            poll_interval=0.1,
        )
        objects = FileEngineObjectStore(self.object_root)
        artifacts = TrialArtifactPublisher(self.root / "trial-artifacts")
        handlers = {"evaluate": self._evaluate(objects)}
        if task_id == "data_selection":
            handlers["train_sft"] = SFTExecutor(
                io=objects,
                backend=_ScriptedSFTBackend(),
                artifacts=artifacts,
            ).execute
        else:
            handlers["train_rft"] = RFTExecutor(
                io=objects,
                backend=_ScriptedRFTBackend(),
                artifacts=artifacts,
            ).execute
        self.engine = EngineWorker(
            queue=self.control.trials.queue,
            handlers=handlers,
        )
        self.evidence_path = self.root / "gate-evidence.json"

    @staticmethod
    def _evaluate(objects: FileEngineObjectStore):
        def execute(command: EvaluateCommand) -> tuple[str, ...]:
            result_ref = f"{command.output_uri}/result.json"
            manifest_ref = f"{command.output_uri}/raw/manifest.json"
            objects.put_json(
                result_ref,
                {
                    "schema_version": "1",
                    "score": 0.5,
                    "model_ref": f"model://{command.run_id}/scripted-pre-gpu",
                    "scripted_pre_gpu": True,
                },
            )
            objects.put_json(
                manifest_ref,
                {"schema_version": "2", "units": [], "scripted_pre_gpu": True},
            )
            return result_ref, manifest_ref

        return execute

    def create(self) -> RunState:
        state_path = self.runs_root / self.run_id / "run.json"
        if state_path.exists():
            raise ValueError(f"Pre-GPU Gate Run already exists: {self.run_id}")
        return self.control.harness.create_experiment(self.compiled)

    def run(self, *, max_cycles: int = 160) -> RunState:
        state = self.control.repository.load(self.run_id)
        events: list[dict[str, object]] = []
        for cycle in range(1, max_cycles + 1):
            if state.status is RunStatus.COMPLETED:
                self._write_evidence(state, events, "completed")
                return state
            if state.bootstrap.status is BootstrapStatus.FAILED:
                self._write_evidence(state, events, "failed")
                raise RuntimeError(
                    f"Pre-GPU bootstrap failed: {state.bootstrap.error}"
                )
            before = state.revision
            result = self.control.workflow.reconcile_once(self.run_id)
            state = self.control.repository.load(self.run_id)
            events.append(
                {
                    "cycle": cycle,
                    "kind": "control",
                    "before_revision": before,
                    "after_revision": state.revision,
                    "transition": state.last_transition.kind,
                    "result": type(result).__name__,
                }
            )
            attempt_id = self.agent_runner.run_once(self.run_id)
            if attempt_id is not None:
                events.append(
                    {"cycle": cycle, "kind": "agent", "attempt_id": attempt_id}
                )
            receipt = self.engine.run_once()
            if receipt is not None:
                events.append(
                    {
                        "cycle": cycle,
                        "kind": "engine",
                        "command_id": receipt.command_id,
                        "status": receipt.status.value,
                    }
                )
            state = self.control.repository.load(self.run_id)
            self._write_evidence(state, events, "running")
        self._write_evidence(state, events, "failed")
        raise RuntimeError(
            f"Pre-GPU Gate Run did not complete within {max_cycles} cycles"
        )

    def _write_evidence(
        self,
        state: RunState,
        events: list[dict[str, object]],
        status: str,
    ) -> None:
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_path.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "gate": "D",
                    "status": status,
                    "run_id": self.run_id,
                    "revision": state.revision,
                    "run_status": state.status.value,
                    "config_digest": self.compiled.config_digest,
                    "agent_execution_mode": self.control.agent_calls.execution_mode,
                    "scripted_engine": True,
                    "events": events,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
