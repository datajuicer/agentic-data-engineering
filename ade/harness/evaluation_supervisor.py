"""Deployment-bound supervision for standalone generalization evaluations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Protocol

from ade.engine.command_queue import FileCommandQueue
from ade.engine.storage.atomic import write_json_atomic, write_text_atomic
from ade.engine.storage.object_store import FileEngineObjectStore
from ade.harness.evaluation import (
    EvaluationConfigCompiler,
    ResolvedEvaluationConfig,
    StandaloneEvaluationService,
)
from ade.harness.generalization_operation import GeneralizationOperation
from ade.harness.generalization_results import GeneralizationResultAggregator


_SUPERVISOR_TERMINAL = {
    "complete",
    "completed_degraded",
    "blocked",
    "awaiting_full_matrix_authorization",
}
_UNIT_TERMINAL = {"succeeded", "failed"}
_SAFE_RUN_TERMINAL = {"cancelled", "completed", "failed", "paused", "suspended"}
_STATUS_FIELDS = (
    "logical_unit_id",
    "task",
    "checkpoint_id",
    "variant",
    "dataset",
    "configured_k",
    "expected_generations",
    "unit_status",
    "transport_status",
    "attempt_id",
    "attempt_index",
    "last_heartbeat_at",
    "worker_id",
    "worker_pid",
    "checkpoint_ref",
    "error",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RayInspector(Protocol):
    def inspect(
        self,
        *,
        address: str,
        required_gpus: int,
        require_idle: bool,
    ) -> dict[str, object]: ...


class WorkerManager(Protocol):
    def ensure_running(self) -> dict[str, object]: ...

    def stop(self) -> dict[str, object]: ...


class ProductionRayInspector:
    """Read Ray/GPU lease state directly from GCS without the Dashboard."""

    def inspect(
        self,
        *,
        address: str,
        required_gpus: int,
        require_idle: bool,
    ) -> dict[str, object]:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        from ade.engine.execution.ray import gpu_snapshot_task

        ray.init(
            address=address,
            namespace="ade",
            ignore_reinit_error=True,
            logging_level="ERROR",
        )
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        try:
            cluster = dict(ray.cluster_resources())
            available = dict(ray.available_resources())
            nodes = [
                {
                    "node_id": str(node.get("NodeID") or ""),
                    "node_ip": str(node.get("NodeManagerAddress") or ""),
                    "alive": bool(node.get("Alive")),
                    "gpus": float((node.get("Resources") or {}).get("GPU", 0) or 0),
                }
                for node in ray.nodes()
            ]
            gpu_nodes = [node for node in nodes if node["alive"] and node["gpus"] > 0]
            physical_gpus = sorted(
                ray.get(
                    [
                        gpu_snapshot_task.options(
                            scheduling_strategy=NodeAffinitySchedulingStrategy(
                                node_id=str(node["node_id"]),
                                soft=False,
                            )
                        ).remote(None)
                        for node in gpu_nodes
                    ]
                ),
                key=lambda item: str(item.get("node_ip") or ""),
            )
            placement_groups = [
                {
                    "placement_group_id": str(value.get("placement_group_id") or ""),
                    "name": str(value.get("name") or ""),
                    "state": str(value.get("state") or ""),
                }
                for value in ray.util.placement_group_table().values()
                if value.get("state") != "REMOVED"
            ]
            try:
                allocator = ray.get_actor("ade_gpu_lease_allocator", namespace="ade")
            except ValueError:
                leases: list[dict[str, object]] = []
                allocator_status = "absent"
            else:
                leases = [dict(item) for item in ray.get(allocator.snapshot.remote())]
                allocator_status = "present"
        finally:
            ray.shutdown()
        total_gpus = int(float(cluster.get("GPU", 0) or 0))
        available_gpus = int(float(available.get("GPU", 0) or 0))
        issues: list[str] = []
        if total_gpus < required_gpus:
            issues.append("insufficient_total_gpus")
        if require_idle and available_gpus < required_gpus:
            issues.append("insufficient_available_gpus")
        if require_idle and placement_groups:
            issues.append("non_removed_placement_groups")
        if require_idle and leases:
            issues.append("allocator_leases")
        if require_idle and any(
            gpu.get("processes")
            for snapshot in physical_gpus
            if snapshot.get("status") == "complete"
            for gpu in snapshot.get("gpus") or []
        ):
            issues.append("physical_gpu_processes")
        if require_idle and any(
            snapshot.get("status") != "complete" for snapshot in physical_gpus
        ):
            issues.append("physical_gpu_inspection_unavailable")
        return {
            "schema_version": "ade.generalization_ray_snapshot.v1",
            "checked_at": _utc_now(),
            "address": address,
            "status": "complete" if not issues else "blocked",
            "required_gpus": required_gpus,
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "nodes": nodes,
            "physical_gpus": physical_gpus,
            "allocator_actor": allocator_status,
            "allocator_leases": leases,
            "placement_groups": placement_groups,
            "issues": issues,
        }


@dataclass
class SubprocessEngineWorker:
    operation: GeneralizationOperation
    worker_id: str = "c000"

    def __post_init__(self) -> None:
        self.root = (
            self.operation.paths.evaluation_root
            / self.operation.evaluation_id
            / "supervisor"
        )
        self.pid_path = self.root / f"engine-worker-{self.worker_id}.json"
        self.log_path = self.root / f"engine-worker-{self.worker_id}.log"
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle = None

    def ensure_running(self) -> dict[str, object]:
        self.root.mkdir(parents=True, exist_ok=True)
        existing = self._load_pid()
        if existing is not None and self._owned_pid(int(existing["pid"])):
            return {**existing, "status": "running", "reused": True}
        if existing is not None:
            self._release_leases()
            write_json_atomic(
                self.pid_path,
                {**existing, "status": "exited", "observed_at": _utc_now()},
            )
        command = [
            str(self.operation.execution_environment / "bin/python"),
            "-m",
            "ade.harness.cli",
            "--project-root",
            str(self.operation.paths.deployment_root.parents[2]),
            "engine",
            "worker",
            "--queue-root",
            str(self.operation.paths.queue_root),
            "--object-root",
            str(self.operation.paths.object_root),
            "--work-root",
            str(self.operation.paths.work_root),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "RAY_ADDRESS": self.operation.ray_address,
                "ADE_GENERALIZATION_EVALUATION_ID": self.operation.evaluation_id,
                "ADE_GENERALIZATION_WORKER_ID": self.worker_id,
                "PYTHONUNBUFFERED": "1",
            }
        )
        self._close_log()
        self._log_handle = self.log_path.open("ab", buffering=0)
        self._process = subprocess.Popen(
            command,
            cwd=self.operation.paths.deployment_root.parents[2],
            env=environment,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record = {
            "schema_version": "ade.generalization_worker.v1",
            "evaluation_id": self.operation.evaluation_id,
            "pid": self._process.pid,
            "worker_id": self.worker_id,
            "status": "running",
            "started_at": _utc_now(),
            "command": command,
            "queue_root": str(self.operation.paths.queue_root),
            "log_path": str(self.log_path),
            "reused": False,
        }
        write_json_atomic(self.pid_path, record)
        return record

    def stop(self) -> dict[str, object]:
        record = self._load_pid()
        if record is None:
            return {"status": "absent"}
        pid = int(record["pid"])
        if not self._owned_pid(pid):
            result = {**record, "status": "already_exited", "stopped_at": _utc_now()}
            write_json_atomic(self.pid_path, result)
            self._release_leases()
            self._close_log()
            return result
        errors: list[str] = []
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(f"SIGTERM: {type(error).__name__}: {error}")
        deadline = time.monotonic() + 30
        while self._owned_pid(pid) and time.monotonic() < deadline:
            time.sleep(0.25)
        if self._owned_pid(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f"SIGKILL: {type(error).__name__}: {error}")
        if self._process is not None:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            self._process = None
        still_running = self._owned_pid(pid)
        if not still_running:
            self._release_leases()
        result = {
            **record,
            "status": "cleanup_failed" if still_running else "stopped",
            "stopped_at": _utc_now(),
            "errors": errors,
        }
        write_json_atomic(self.pid_path, result)
        self._close_log()
        return result

    def _release_leases(self) -> None:
        """Release only this stopped worker's coordinator leases, before reuse."""
        import ray

        ray.init(address=self.operation.ray_address, namespace="ade",
                 ignore_reinit_error=True, logging_level="ERROR")
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        try:
            try:
                allocator = ray.get_actor("ade_gpu_lease_allocator", namespace="ade")
            except ValueError:
                return
            owner = f"{self.operation.evaluation_id}/{self.worker_id}"
            for lease in ray.get(allocator.snapshot.remote()):
                if lease.get("coordinator_owner") == owner:
                    ray.get(allocator.release.remote(lease["lease_id"]))
        finally:
            ray.shutdown()

    def _load_pid(self) -> dict[str, Any] | None:
        if not self.pid_path.is_file():
            return None
        payload = json.loads(self.pid_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) and payload.get("pid") else None

    def _owned_pid(self, pid: int) -> bool:
        command_line = Path(f"/proc/{pid}/cmdline")
        if not command_line.is_file():
            return False
        try:
            text = (
                command_line.read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except OSError:
            return False
        try:
            environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except OSError:
            return False
        return (
            f"ADE_GENERALIZATION_WORKER_ID={self.worker_id}".encode() in environment
            and "ade.harness.cli" in text
            and "engine worker" in text
            and str(self.operation.paths.queue_root) in text
        )

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


class SubprocessEngineWorkerPool:
    """Independent processes sharing one evaluation's atomic command queue."""

    def __init__(self, operation: GeneralizationOperation) -> None:
        self.workers = {
            f"c{index:03d}": SubprocessEngineWorker(operation, f"c{index:03d}")
            for index in range(1, operation.max_in_flight + 1)
        }

    def ensure_running(self) -> dict[str, object]:
        return {"status": "running", "workers": {
            key: worker.ensure_running() for key, worker in self.workers.items()
        }}

    def stop(self) -> dict[str, object]:
        records = {}
        for key, worker in self.workers.items():
            try:
                records[key] = worker.stop()
            except Exception as error:
                records[key] = {"status": "cleanup_failed", "error": str(error)}
        clean = all(item["status"] in {"stopped", "already_exited", "absent"}
                    for item in records.values())
        return {"status": "stopped" if clean else "cleanup_failed", "workers": records}


class GeneralizationSupervisor:
    """Drive admission, bounded worker scheduling, recovery, monitoring, and reports."""

    def __init__(
        self,
        *,
        project_root: Path,
        operation: GeneralizationOperation,
        config: ResolvedEvaluationConfig,
        queue: FileCommandQueue,
        objects: FileEngineObjectStore,
        ray_inspector: RayInspector,
        worker: WorkerManager,
        artifact_validator: Callable[[tuple[str, ...]], dict[str, object]]
        | None = None,
        wait: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.project_root = project_root.resolve()
        self.operation = operation
        self.config = config
        self.queue = queue
        self.objects = objects
        self.service = StandaloneEvaluationService(
            root=operation.paths.evaluation_root,
            queue=queue,
            objects=objects,
        )
        self.ray_inspector = ray_inspector
        self.worker = worker
        self.artifact_validator = artifact_validator or self._validate_artifacts
        self.wait = wait
        self.clock = clock
        self.directory = operation.paths.evaluation_root / operation.evaluation_id
        self.supervisor_root = self.directory / "supervisor"
        self.supervisor_state_path = self.supervisor_root / "state.json"
        self.monitor_path = self.directory / "monitor.md"
        self.snapshot_root = self.directory / "snapshots"

    def _required_gpus(self) -> int:
        return self.operation.max_in_flight * max(
            int(unit["request"]["data_parallel_shards"])
            for unit in self.config.units.values()
        )

    def preflight(self) -> dict[str, object]:
        issues: list[str] = []
        if not self.operation.authorized:
            issues.append("resource_authorization_missing")
        unavailable = sorted(
            unit_id
            for unit_id, unit in self.config.units.items()
            if not bool(unit["checkpoint_available"])
        )
        if unavailable:
            issues.append("checkpoint_restore_required")
        datasets = tuple(
            dict.fromkeys(
                str(unit["dataset_id"]) for unit in self.config.units.values()
            )
        )
        try:
            artifact_receipt = self.artifact_validator(datasets)
        except Exception as error:
            artifact_receipt = {"status": "blocked", "error": str(error)}
            issues.append("benchmark_artifact_validation_failed")
        if (
            not isinstance(artifact_receipt, dict)
            or artifact_receipt.get("status") != "complete"
        ):
            artifact_receipt = {
                "status": "blocked",
                "error": "artifact validator returned no complete receipt",
            }
            issues.append("benchmark_artifact_validation_failed")
        active_runs = self._active_deployment_runs()
        if active_runs:
            issues.append("active_exclusive_ade_run")
        required_gpus = self._required_gpus()
        if issues:
            ray = {
                "status": "not_checked",
                "reason": "local or authorization admission failed first",
                "issues": [],
            }
        else:
            try:
                ray = self.ray_inspector.inspect(
                    address=self.operation.ray_address,
                    required_gpus=required_gpus,
                    require_idle=True,
                )
            except Exception as error:
                ray = {
                    "status": "blocked",
                    "error": str(error),
                    "issues": ["unreachable"],
                }
                issues.append("ray_inspection_failed")
            if ray.get("status") != "complete":
                issues.append("ray_admission_blocked")
        receipt = {
            "schema_version": "ade.generalization_preflight.v1",
            "checked_at": _utc_now(),
            "evaluation_id": self.operation.evaluation_id,
            "operation_digest": self.operation.operation_digest,
            "config_digest": self.config.config_digest,
            "status": "complete" if not issues else "blocked",
            "issues": list(dict.fromkeys(issues)),
            "checkpoint_unavailable_units": unavailable,
            "benchmark_artifacts": artifact_receipt,
            "active_deployment_runs": active_runs,
            "ray": ray,
            "required_gpus": required_gpus,
            "unit_count": len(self.config.units),
            "expected_generations": self.config.expected_generations,
        }
        target = self.operation.paths.request_root / "preflight.json"
        write_json_atomic(target, receipt)
        if receipt["status"] != "complete":
            raise ValueError(
                "generalization preflight blocked: "
                + ", ".join(str(item) for item in receipt["issues"])
            )
        return receipt

    def initialize(self) -> dict[str, object]:
        self.operation.materialize()
        evaluation_exists = self.directory.is_dir()
        if self.operation.run_mode == "NEW" and evaluation_exists:
            raise ValueError("NEW operation evaluation already exists")
        if self.operation.run_mode == "RESUME" and not evaluation_exists:
            raise ValueError("RESUME operation evaluation does not exist")
        if not evaluation_exists:
            self.preflight()
            evaluation = self.service.create(self.config)
            self.supervisor_root.mkdir(parents=True, exist_ok=True)
            state = self._new_supervisor_state()
            write_json_atomic(self.supervisor_state_path, state)
            self._snapshot(state, evaluation, force=True)
            return state
        resolved_path = self.directory / "config/resolved.json"
        if not resolved_path.is_file():
            raise ValueError("existing evaluation has no resolved config")
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(resolved, dict):
            raise ValueError("existing evaluation resolved config is invalid")
        if resolved.get("config_digest") != self.config.config_digest:
            raise ValueError("RESUME config digest does not match existing evaluation")
        self._verify_existing_preflight()
        if not self.supervisor_state_path.is_file():
            # Recover the narrow crash window between service.create() and the
            # first supervisor-state write. The evaluation state and queue are
            # idempotent authorities, so no scientific unit is duplicated.
            evaluation = self.service.status(self.operation.evaluation_id)
            state = self._new_supervisor_state()
            state["events"] = [
                {
                    "at": _utc_now(),
                    "event": "supervisor_state_recovered",
                    "reason": "evaluation existed without supervisor state",
                }
            ]
            write_json_atomic(self.supervisor_state_path, state)
            self._snapshot(state, evaluation, force=True)
            return state
        state = self._load_supervisor_state()
        if state.get("operation_digest") != self.operation.operation_digest:
            raise ValueError(
                "RESUME operation digest does not match existing evaluation"
            )
        if state.get("config_digest") != self.config.config_digest:
            raise ValueError("RESUME config digest does not match existing evaluation")
        return state

    def _retry_after_repair(
        self, state: dict[str, object], unit_id: str | None
    ) -> dict[str, object]:
        pending = state.get("repair_retry")
        if pending is None:
            if unit_id is None:
                return state
            if state["status"] != "completed_degraded":
                raise ValueError("repair retry requires a completed_degraded matrix")
            evaluation = self.service.status(self.operation.evaluation_id)
            unit = self._units(evaluation).get(unit_id)
            if unit is None or unit["status"] != "failed":
                raise ValueError(f"unit is not failed: {unit_id}")
            if len(unit["attempts"]) >= int(state["max_attempts"]):
                raise ValueError(f"unit attempt budget exhausted: {unit_id}")
            if (
                state.get("worker_cleanup", {}).get("status") != "stopped"
                or state.get("terminal_ray", {}).get("status") != "complete"
            ):
                raise ValueError("repair retry requires completed worker and Ray cleanup")
            self.preflight()
            pending = {"unit_id": unit_id, "attempt_index": len(unit["attempts"]) + 1}
            state["repair_retry"] = pending
            # Persist intent before the service write so AUTO can finish either
            # side of an interrupted transition without duplicating an attempt.
            write_json_atomic(self.supervisor_state_path, state)
        unit_id = str(pending["unit_id"])
        evaluation = self.service.status(self.operation.evaluation_id)
        unit = self._units(evaluation)[unit_id]
        if len(unit["attempts"]) < int(pending["attempt_index"]):
            self.service.retry(
                self.operation.evaluation_id, unit_id=unit_id, after_repair=True
            )
        state.setdefault("events", []).append(
            {"at": _utc_now(), "event": "unit_retried_after_repair", **pending}
        )
        state["status"] = "running"
        for key in ("repair_retry", "completed_at", "worker_cleanup", "terminal_ray"):
            state.pop(key, None)
        state["updated_at"] = _utc_now()
        write_json_atomic(self.supervisor_state_path, state)
        return state

    def run_until_terminal(
        self, *, retry_repaired_unit: str | None = None
    ) -> dict[str, object]:
        lock_path = self.operation.paths.request_root / "supervisor.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(
                    "another generalization supervisor owns this operation"
                ) from error
            state = self.initialize()
            state = self._retry_after_repair(state, retry_repaired_unit)
            write_json_atomic(
                self.operation.paths.request_root / "supervisor-lease.json",
                {
                    "schema_version": "ade.generalization_supervisor_lease.v1",
                    "evaluation_id": self.operation.evaluation_id,
                    "pid": os.getpid(),
                    "acquired_at": _utc_now(),
                },
            )
            try:
                if state["status"] not in _SUPERVISOR_TERMINAL:
                    state = self._ensure_worker(state)
                while state["status"] not in _SUPERVISOR_TERMINAL:
                    state = self._ensure_worker(state)
                    if state["status"] in _SUPERVISOR_TERMINAL:
                        break
                    state = self.reconcile_once()
                    if state["status"] not in _SUPERVISOR_TERMINAL:
                        self.wait(self.operation.poll_interval_seconds)
            except Exception as error:
                # Keep the evaluation-owned worker alive: it may finish an
                # already claimed unit while an AUTO supervisor is restarted.
                self._record_process_error(error)
                raise
            try:
                cleanup = self.worker.stop()
            except Exception as error:
                cleanup = {
                    "status": "cleanup_failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            state["worker_cleanup"] = cleanup
            try:
                terminal_ray = self.ray_inspector.inspect(
                    address=self.operation.ray_address,
                    required_gpus=self._required_gpus(),
                    require_idle=True,
                )
            except Exception as error:
                terminal_ray = {
                    "status": "blocked",
                    "issues": ["ray_cleanup_inspection_failed"],
                    "error": str(error),
                }
            state["terminal_ray"] = terminal_ray
            if state["status"] in {"complete", "completed_degraded"} and (
                cleanup.get("status") not in {"stopped", "already_exited"}
                or terminal_ray.get("status") != "complete"
            ):
                state["evaluation_terminal_status"] = state["status"]
                state["status"] = "blocked"
                state["blocker"] = (
                    "evaluation worker was not stopped"
                    if cleanup.get("status") not in {"stopped", "already_exited"}
                    else "terminal Ray resources were not released"
                )
            state["updated_at"] = _utc_now()
            write_json_atomic(self.supervisor_state_path, state)
            self._snapshot(
                state, self.service.status(self.operation.evaluation_id), force=True
            )
            return state

    def reconcile_once(self) -> dict[str, object]:
        state = self._load_supervisor_state()
        if state["status"] in _SUPERVISOR_TERMINAL:
            return state
        self.queue.expire_stale()
        evaluation = self.service.status(self.operation.evaluation_id)
        units = self._units(evaluation)
        eligible = (
            list(state["admission_units"])
            if state["phase"] == "admission"
            else list(evaluation["unit_order"])
        )
        events = list(state.get("events") or [])

        for unit_id in eligible:
            unit = units[unit_id]
            if unit["status"] != "failed":
                continue
            attempt = self._active_attempt(unit)
            receipt = attempt.get("receipt")
            retryable = isinstance(receipt, dict) and receipt.get("retryable") is True
            if retryable and len(unit["attempts"]) < int(state["max_attempts"]):
                self.service.retry(self.operation.evaluation_id, unit_id=unit_id)
                self.service.run(self.operation.evaluation_id, unit_id=unit_id)
                events.append(
                    {
                        "at": _utc_now(),
                        "event": "unit_retried",
                        "unit_id": unit_id,
                        "attempt_index": len(unit["attempts"]) + 1,
                    }
                )
                evaluation = self.service.status(self.operation.evaluation_id)
                units = self._units(evaluation)
            elif state["phase"] == "admission":
                state["status"] = "blocked"
                state["blocker"] = f"admission unit failed: {unit_id}"

        if state["status"] != "blocked" and state["phase"] == "admission":
            admission_statuses = [
                units[item]["status"] for item in state["admission_units"]
            ]
            if all(status == "succeeded" for status in admission_statuses):
                if bool(state["full_matrix_after_admission"]):
                    state["phase"] = "full_matrix"
                    events.append({"at": _utc_now(), "event": "admission_completed"})
                    eligible = list(evaluation["unit_order"])
                else:
                    state["status"] = "awaiting_full_matrix_authorization"

        if state["status"] not in _SUPERVISOR_TERMINAL:
            in_flight = sum(
                unit["status"] not in _UNIT_TERMINAL and unit["status"] != "created"
                for unit in units.values()
            )
            available_slots = max(0, int(state["max_in_flight"]) - in_flight)
            for unit_id in eligible:
                if available_slots <= 0:
                    break
                if units[unit_id]["status"] != "created":
                    continue
                self.service.run(self.operation.evaluation_id, unit_id=unit_id)
                events.append(
                    {"at": _utc_now(), "event": "unit_submitted", "unit_id": unit_id}
                )
                available_slots -= 1
            evaluation = self.service.status(self.operation.evaluation_id)
            units = self._units(evaluation)

        if state["phase"] == "full_matrix" and all(
            unit["status"] in _UNIT_TERMINAL for unit in units.values()
        ):
            state["status"] = (
                "complete"
                if all(unit["status"] == "succeeded" for unit in units.values())
                else "completed_degraded"
            )
            state["completed_at"] = _utc_now()
        state["events"] = events
        state["updated_at"] = _utc_now()
        write_json_atomic(self.supervisor_state_path, state)
        self._snapshot(state, evaluation)
        return state

    def _ensure_worker(self, state: dict[str, object]) -> dict[str, object]:
        record = self.worker.ensure_running()
        if "workers" in record:
            previous_workers = state.get("workers", {})
            restart_counts = state.setdefault("worker_restart_counts", {})
            for worker_id, current in record["workers"].items():
                previous = previous_workers.get(worker_id, {}).get("pid")
                if previous is not None and previous != current["pid"]:
                    lost = self.queue.terminalize_claimed(
                        run_id=self.operation.evaluation_id, worker_pid=int(previous)
                    )
                    restart_counts[worker_id] = restart_counts.get(worker_id, 0) + 1
                    state["worker_restarts"] += 1
                    state.setdefault("events", []).append({
                        "at": _utc_now(), "event": "engine_worker_restarted",
                        "worker_id": worker_id, "old_pid": previous,
                        "new_pid": current["pid"], "lost_commands": list(lost),
                    })
                    if restart_counts[worker_id] > int(state["max_worker_restarts"]):
                        state["status"] = "blocked"
                        state["blocker"] = f"Engine worker {worker_id} restart budget exhausted"
            state["workers"] = record["workers"]
            state["updated_at"] = _utc_now()
            write_json_atomic(self.supervisor_state_path, state)
            return state
        current_pid = record.get("pid")
        previous = state.get("worker_pid")
        if previous is not None and current_pid != previous:
            self.queue.terminalize_claimed(run_id=self.operation.evaluation_id)
            restarts = int(state.get("worker_restarts") or 0) + 1
            state["worker_restarts"] = restarts
            state.setdefault("events", []).append(
                {
                    "at": _utc_now(),
                    "event": "engine_worker_restarted",
                    "old_pid": previous,
                    "new_pid": current_pid,
                    "restart": restarts,
                }
            )
            if restarts > int(state["max_worker_restarts"]):
                self.worker.stop()
                state["status"] = "blocked"
                state["blocker"] = "Engine worker restart budget exhausted"
        state["worker_pid"] = current_pid
        state["worker"] = record
        state["updated_at"] = _utc_now()
        write_json_atomic(self.supervisor_state_path, state)
        return state

    def _snapshot(
        self,
        supervisor: dict[str, object],
        evaluation: dict[str, object],
        *,
        force: bool = False,
    ) -> None:
        units = self._units(evaluation)
        counts = Counter(str(unit["status"]) for unit in units.values())
        signature = json.dumps(
            {
                "phase": supervisor["phase"],
                "status": supervisor["status"],
                "counts": counts,
                "attempts": {
                    unit_id: len(unit["attempts"]) for unit_id, unit in units.items()
                },
            },
            sort_keys=True,
        )
        now = self.clock()
        last_snapshot = supervisor.get("last_snapshot_at")
        due = (
            last_snapshot is None
            or now - float(last_snapshot) >= self.operation.snapshot_interval_seconds
        )
        if not force and signature == supervisor.get("last_signature") and not due:
            return
        index = int(supervisor.get("snapshot_index") or 0) + 1
        active = []
        for unit_id, unit in units.items():
            if unit["status"] in _UNIT_TERMINAL or unit["status"] == "created":
                continue
            attempt = self._active_attempt(unit)
            command_id = str(attempt["command"]["command_id"])
            try:
                liveness = self.queue.load_liveness(command_id)
            except (FileNotFoundError, ValueError):
                liveness = {"status": unit["status"]}
            active.append(
                {
                    "unit_id": unit_id,
                    "attempt_id": attempt["attempt_id"],
                    "command_id": command_id,
                    "transport_status": liveness.get("status"),
                    "claimed_at": liveness.get("claimed_at"),
                    "last_heartbeat_at": liveness.get("last_heartbeat_at"),
                    "worker_id": liveness.get("worker_id"),
                    "worker_pid": liveness.get("worker_pid"),
                }
            )
        try:
            ray = self.ray_inspector.inspect(
                address=self.operation.ray_address,
                required_gpus=self._required_gpus(),
                require_idle=False,
            )
        except Exception as error:
            ray = {
                "status": "unavailable",
                "error": str(error),
                "issues": ["ray_monitor_inspection_failed"],
            }
        report = self._refresh_reports(evaluation)
        supervisor["report"] = report
        if (
            supervisor["status"] in {"complete", "completed_degraded"}
            and report["status"] != "complete"
        ):
            supervisor["evaluation_terminal_status"] = supervisor["status"]
            supervisor["status"] = "blocked"
            supervisor["blocker"] = "terminal report generation or validation failed"
        snapshot = {
            "schema_version": "ade.generalization_snapshot.v1",
            "snapshot_index": index,
            "observed_at": _utc_now(),
            "evaluation_id": self.operation.evaluation_id,
            "supervisor_status": supervisor["status"],
            "phase": supervisor["phase"],
            "unit_counts": dict(sorted(counts.items())),
            "active_units": active,
            "ray": ray,
            "report": report,
        }
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.snapshot_root / f"snapshot-{index:06d}.json", snapshot)
        self._append_monitor(snapshot)
        supervisor["snapshot_index"] = index
        supervisor["last_snapshot_at"] = now
        supervisor["last_signature"] = signature
        write_json_atomic(self.supervisor_state_path, supervisor)

    def _refresh_reports(self, evaluation: dict[str, object]) -> dict[str, object]:
        try:
            self._write_status_csv(evaluation)
            result = GeneralizationResultAggregator(
                evaluation_root=self.operation.paths.evaluation_root,
                objects=self.objects,
            ).aggregate(
                self.operation.evaluation_id,
                output_dir=self.operation.paths.output_root,
            )
            expected_comparisons = len(
                {
                    (str(unit["suite_id"]), str(unit["dataset_id"]))
                    for unit in self.config.units.values()
                }
            )
            if int(result["row_count"]) != len(self.config.units):
                raise ValueError("results.csv row count does not match resolved units")
            if int(result["comparison_rows"]) != expected_comparisons:
                raise ValueError(
                    "comparison.csv row count does not match task/dataset pairs"
                )
            return {**result, "status": "complete"}
        except Exception as error:
            return {
                "status": "blocked",
                "checked_at": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }

    def _append_monitor(self, snapshot: dict[str, object]) -> None:
        ray = dict(snapshot["ray"])
        counts = dict(snapshot["unit_counts"])
        active = list(snapshot["active_units"])
        lines = [
            f"## Snapshot {int(snapshot['snapshot_index']):06d} — {snapshot['observed_at']}",
            "",
            f"- Supervisor: `{snapshot['supervisor_status']}`; phase: `{snapshot['phase']}`",
            f"- Units: `{json.dumps(counts, sort_keys=True)}`",
            f"- Ray GPUs: available `{ray.get('available_gpus', 'unknown')}` / total `{ray.get('total_gpus', 'unknown')}`",
            "- Active: "
            + (
                "; ".join(
                    f"`{item['unit_id']}`/{item['attempt_id']} transport={item.get('transport_status')} heartbeat={item.get('last_heartbeat_at')}"
                    for item in active
                )
                if active
                else "none"
            ),
            "",
        ]
        existing = (
            self.monitor_path.read_text(encoding="utf-8")
            if self.monitor_path.is_file()
            else f"# Generalization monitor: {self.operation.evaluation_id}\n\n"
        )
        write_text_atomic(self.monitor_path, existing + "\n".join(lines))

    def _write_status_csv(self, evaluation: dict[str, object]) -> None:
        rows = []
        for unit_id, state in self._units(evaluation).items():
            resolved = self.config.units[unit_id]
            attempt = self._active_attempt(state)
            command_id = str(attempt["command"]["command_id"])
            try:
                liveness = self.queue.load_liveness(command_id)
            except (FileNotFoundError, ValueError):
                liveness = {}
            receipt = attempt.get("receipt")
            rows.append(
                {
                    "logical_unit_id": unit_id,
                    "task": resolved["suite_id"],
                    "checkpoint_id": resolved["checkpoint_id"],
                    "variant": resolved["variant"],
                    "dataset": resolved["dataset_id"],
                    "configured_k": resolved["samples_per_input"],
                    "expected_generations": resolved["expected_generations"],
                    "unit_status": state["status"],
                    "transport_status": liveness.get("status", ""),
                    "attempt_id": attempt["attempt_id"],
                    "attempt_index": attempt["attempt_index"],
                    "last_heartbeat_at": liveness.get("last_heartbeat_at", ""),
                    "worker_id": liveness.get("worker_id", ""),
                    "worker_pid": liveness.get("worker_pid", ""),
                    "checkpoint_ref": resolved["checkpoint_ref"],
                    "error": (
                        receipt.get("error", "") if isinstance(receipt, dict) else ""
                    ),
                }
            )
        target = self.operation.paths.output_root / "status.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_STATUS_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def _admission_units(self) -> list[str]:
        grouped: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
        for order, (unit_id, unit) in enumerate(self.config.units.items()):
            key = (str(unit["suite_id"]), str(unit["checkpoint_id"]))
            grouped.setdefault(key, []).append(
                (int(unit["expected_generations"]), order, unit_id)
            )
        return [min(values)[2] for _key, values in sorted(grouped.items())]

    def _new_supervisor_state(self) -> dict[str, object]:
        return {
            "schema_version": "ade.generalization_supervisor.v1",
            "evaluation_id": self.operation.evaluation_id,
            "operation_digest": self.operation.operation_digest,
            "config_digest": self.config.config_digest,
            "status": "running",
            "phase": "admission",
            "admission_units": self._admission_units(),
            "full_matrix_after_admission": self.operation.full_matrix_after_admission,
            "max_attempts": self.operation.max_attempts,
            "max_worker_restarts": self.operation.max_worker_restarts,
            "max_in_flight": self.operation.max_in_flight,
            "worker_restarts": 0,
            "snapshot_index": 0,
            "last_snapshot_at": None,
            "last_signature": None,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "events": [],
        }

    def _verify_existing_preflight(self) -> None:
        path = self.operation.paths.request_root / "preflight.json"
        if not path.is_file():
            raise ValueError("existing evaluation has no successful preflight receipt")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("existing evaluation preflight receipt is invalid")
        expected = {
            "status": "complete",
            "operation_digest": self.operation.operation_digest,
            "config_digest": self.config.config_digest,
        }
        mismatched = [
            key
            for key, expected_value in expected.items()
            if value.get(key) != expected_value
        ]
        if mismatched:
            raise ValueError(
                "existing evaluation preflight receipt mismatch: "
                + ", ".join(mismatched)
            )

    def _record_process_error(self, error: Exception) -> None:
        try:
            state = self._load_supervisor_state()
        except (OSError, ValueError, json.JSONDecodeError):
            return
        events = list(state.get("events") or [])
        events.append(
            {
                "at": _utc_now(),
                "event": "supervisor_process_error",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        state["events"] = events
        state["last_process_error"] = events[-1]
        state["updated_at"] = _utc_now()
        write_json_atomic(self.supervisor_state_path, state)

    def _validate_artifacts(self, dataset_ids: tuple[str, ...]) -> dict[str, object]:
        command = [
            str(self.operation.execution_environment / "bin/python"),
            str(self.project_root / "scripts/build_benchmark_catalog.py"),
            "--check",
            *dataset_ids,
        ]
        result = subprocess.run(
            command,
            cwd=self.project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        return {
            "status": "complete",
            "datasets": list(dataset_ids),
            "output": [line for line in result.stdout.splitlines() if line.strip()],
        }

    def _active_deployment_runs(self) -> list[dict[str, str]]:
        control_root = self.operation.paths.deployment_root / "control"
        result = []
        if not control_root.is_dir():
            return result
        for run_path in sorted(control_root.glob("*/run.json")):
            try:
                payload = json.loads(run_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            status = str(payload.get("status") or "unknown")
            if status not in _SAFE_RUN_TERMINAL:
                result.append(
                    {
                        "run_id": str(payload.get("run_id") or run_path.parent.name),
                        "status": status,
                        "path": str(run_path),
                    }
                )
        return result

    def _load_supervisor_state(self) -> dict[str, object]:
        value = json.loads(self.supervisor_state_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("generalization supervisor state must be a mapping")
        return value

    @staticmethod
    def _units(evaluation: dict[str, object]) -> dict[str, dict[str, Any]]:
        units = evaluation.get("units")
        if not isinstance(units, dict) or any(
            not isinstance(value, dict) for value in units.values()
        ):
            raise ValueError("evaluation units are invalid")
        return units

    @staticmethod
    def _active_attempt(unit: dict[str, Any]) -> dict[str, Any]:
        attempts = unit.get("attempts")
        if (
            not isinstance(attempts, list)
            or not attempts
            or not isinstance(attempts[-1], dict)
        ):
            raise ValueError("evaluation unit attempts are invalid")
        return attempts[-1]


def compile_operation_evaluation(
    *,
    project_root: Path,
    operation: GeneralizationOperation,
) -> ResolvedEvaluationConfig:
    if not operation.authorized:
        raise ValueError(
            "explicit resource_authorization is required before materialization"
        )
    operation.materialize()
    config = EvaluationConfigCompiler(project_root=project_root).compile_file(
        operation.paths.evaluation_config,
        deployment_config=operation.deployment_config,
        evaluation_root=operation.paths.evaluation_root,
    )
    for unit in config.units.values():
        policy = unit["request"]["coordinator_resource_policy"]
        if operation.max_in_flight * int(policy["capacity_gpus"]) > int(policy["total_gpus"]):
            raise ValueError("worker resource groups exceed configured cluster GPU capacity")
    return config
