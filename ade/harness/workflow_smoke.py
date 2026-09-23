"""Production walking-workflow smoke orchestration and durable evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Protocol

from ade.controller.workflow_driver import WorkflowDriver
from ade.core.engine import EngineReceiptStatus
from ade.core.reconcile import Waiting
from ade.core.run import RunState, RunStatus
from ade.engine.worker import EngineWorker
from ade.harness.experiment_config import ExperimentConfigCompiler
from ade.harness.wiring import build_control_process, build_engine_process
from ade.memory.repository import FileRunRepository
from ade.tasks.registry import default_task_registry


class StateRepository(Protocol):
    def load(self, run_id: str) -> RunState: ...


class WorkflowSmokeRunner:
    def __init__(
        self,
        *,
        workflow: WorkflowDriver,
        worker: EngineWorker,
        repository: StateRepository,
        evidence_path: str | Path,
        max_ticks: int = 32,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if max_ticks < 1:
            raise ValueError("max_ticks must be positive")
        self.workflow = workflow
        self.worker = worker
        self.repository = repository
        self.evidence_path = Path(evidence_path)
        self.max_ticks = max_ticks
        self.metadata = dict(metadata or {})

    def run(self, run_id: str) -> RunState:
        state = self.repository.load(run_id)
        evidence = self._initial_evidence(run_id)
        self._write(evidence)
        try:
            for tick_number in range(1, self.max_ticks + 1):
                if state.status is RunStatus.COMPLETED:
                    evidence["status"] = "completed"
                    evidence["final_state"] = state.to_dict()
                    self._write(evidence)
                    return state
                before_revision = state.revision
                before_status = state.status.value
                result = self.workflow.reconcile_once(run_id)
                state = self.repository.load(run_id)
                evidence["events"].append(
                    {
                        "kind": "control_tick",
                        "tick": tick_number,
                        "before_revision": before_revision,
                        "after_revision": state.revision,
                        "before_status": before_status,
                        "after_status": state.status.value,
                        "result": type(result).__name__,
                    }
                )
                self._write(evidence)
                if (
                    isinstance(result, Waiting)
                    and state.status is RunStatus.RUNNING
                ):
                    receipt = self.worker.run_once()
                    if receipt is None:
                        raise RuntimeError(
                            "workflow made no durable progress and Engine queue is empty"
                        )
                    evidence["events"].append(
                        {
                            "kind": "engine_receipt",
                            "tick": tick_number,
                            "receipt": receipt.to_dict(),
                        }
                    )
                    self._write(evidence)
                    if receipt.status is EngineReceiptStatus.FAILED:
                        raise RuntimeError(receipt.error or "Engine command failed")
            raise RuntimeError(
                f"workflow did not complete within {self.max_ticks} control ticks"
            )
        except Exception as error:
            evidence["status"] = "failed"
            evidence["error"] = f"{type(error).__name__}: {error}"
            evidence["last_state"] = state.to_dict()
            self._write(evidence)
            raise

    def _initial_evidence(self, run_id: str) -> dict[str, Any]:
        if not self.evidence_path.is_file():
            return {
                "schema_version": "1",
                "run_id": run_id,
                "status": "running",
                "metadata": self.metadata,
                "events": [],
            }
        try:
            existing = json.loads(self.evidence_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("existing workflow smoke evidence is invalid") from error
        if not isinstance(existing, dict) or existing.get("run_id") != run_id:
            raise ValueError("existing workflow smoke evidence belongs to another run")
        events = existing.get("events")
        if not isinstance(events, list):
            raise ValueError("existing workflow smoke evidence events are invalid")
        existing["schema_version"] = "1"
        existing["status"] = "running"
        existing["metadata"] = self.metadata
        existing.pop("error", None)
        existing.pop("last_state", None)
        existing.pop("final_state", None)
        events.append({"kind": "resume"})
        return existing

    def _write(self, evidence: Mapping[str, object]) -> None:
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(evidence, indent=2, sort_keys=True, default=str) + "\n"
        ).encode()
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.evidence_path.name}.",
            dir=self.evidence_path.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.evidence_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def run_real_workflow_smoke(
    *,
    project_root: str | Path,
    experiment_config: str | Path,
    deployment_config: str | Path,
    run_id: str,
    resume: bool = False,
    max_ticks: int = 32,
) -> tuple[RunState, Path]:
    project = Path(project_root).resolve()
    config_path = Path(experiment_config)
    if not config_path.is_absolute():
        config_path = (project / config_path).resolve()
    compiled = ExperimentConfigCompiler(
        default_task_registry(),
        project_root=project,
    ).compile_file(
        config_path,
        deployment_config=deployment_config,
        run_id=run_id,
    )
    smoke_root = project / "runs" / "smoke" / run_id
    runs_root = smoke_root / "control"
    queue_root = smoke_root / "queue"
    object_root = smoke_root / "objects"
    evidence_path = smoke_root / "workflow-smoke-evidence.json"
    repository = FileRunRepository(runs_root)
    state_path = runs_root / run_id / "run.json"

    control = build_control_process(
        project_root=project,
        runs_root=runs_root,
        queue_root=queue_root,
        object_root=object_root,
        engine_inputs=compiled.engine_inputs,
        bootstrap_contract=compiled.bootstrap,
        agent_config=compiled.control["agent"],
        artifact_builder_config=compiled.control["artifact_builder"],
        engine_config=compiled.control["engine"],
    )
    engine = build_engine_process(
        queue_root=queue_root,
        object_root=object_root,
        work_root=smoke_root / "engine-work",
        engine_config=compiled.control["engine"],
    )
    engine.queue.expire_stale()

    if state_path.exists():
        if not resume:
            raise ValueError(
                f"smoke run already exists; pass --resume: {run_id}"
            )
        state = repository.load(run_id)
        expected = repository.describe_artifact(
            run_id,
            "resolved_experiment_config",
            compiled.encode(),
        )
        if state.task.config_ref != expected.artifact_id:
            raise ValueError("existing smoke run uses a different resolved config")
    else:
        control.harness.create_experiment(compiled)

    runner = WorkflowSmokeRunner(
        workflow=control.workflow,
        worker=engine.worker,
        repository=repository,
        evidence_path=evidence_path,
        max_ticks=max_ticks,
        metadata={
            "experiment_config": str(config_path),
            "experiment_id": compiled.experiment_id,
            "config_digest": compiled.config_digest,
            "source_layers": compiled.source_layers,
            "smoke_root": str(smoke_root),
        },
    )
    return runner.run(run_id), evidence_path
