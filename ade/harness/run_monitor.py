"""Run-owned Ray/NVML resource monitor and W&B projection."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Mapping

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from ade.engine.execution.gpu import get_gpu_lease_allocator
from ade.engine.execution.ray import gpu_snapshot_task
from ade.harness.run_cleanup import cleanup_cancelled_run
from ade.engine.telemetry.tracking import (
    WANDB_API_TIMEOUT_SECONDS,
    _wandb_no_proxy_environment,
    bounded_wandb_settings,
    reconcile_evaluation_tracking,
)
from ade.engine.telemetry.seed_import import reconcile_seed_wandb_history
from ade.harness.experiment_config import ExperimentConfigCompiler
from ade.harness.runtime_roots import deployment_runtime_roots
from ade.memory.repository import FileRunRepository
from ade.tasks.registry import default_task_registry


_TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled", "suspended"}
_WANDB_INITIAL_REMOTE_GRACE_SECONDS = 600.0
_WANDB_REMOTE_STALL_SECONDS = 1200.0
_WANDB_REMOTE_STARTUP_CHECK_SECONDS = 30.0
_WANDB_REMOTE_STEADY_CHECK_SECONDS = 600.0


def _git_head(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(project_root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _append_source_provenance(
    run_dir: Path,
    *,
    project_root: Path,
    run_id: str,
    config_digest: str,
    source_commit: str | None,
    event: str,
) -> int:
    path = run_dir / "reports" / "source-provenance.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    index = 0
    if path.is_file():
        index = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    payload = {
        "schema_version": "ade.source_provenance.v1",
        "index": index,
        "event": event,
        "run_id": run_id,
        "git_commit": source_commit,
        "config_digest": config_digest,
        "project_root": str(project_root),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return index


class RunResourceMonitor:
    def __init__(
        self,
        *,
        project_root: str | Path,
        runs_root: str | Path,
        experiment_config: str | Path,
        deployment_config: str | Path,
        run_id: str,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.runs_root = Path(runs_root).resolve()
        self.experiment_config = Path(experiment_config).resolve()
        deployment = Path(deployment_config)
        if not deployment.is_absolute():
            deployment = self.project_root / deployment
        self.deployment_config = deployment.resolve()
        self.run_id = run_id
        self.stop_requested = False

    def run(self) -> int:
        compiled = ExperimentConfigCompiler(
            default_task_registry(), project_root=self.project_root
        ).compile_file(
            self.experiment_config,
            deployment_config=self.deployment_config,
            run_id=self.run_id,
        )
        resolved = compiled.resolved
        deployment = _mapping(resolved.get("deployment"), "resolved deployment")
        resources = _mapping(deployment.get("run_resources"), "deployment run_resources")
        cluster = _mapping(resources.get("ray_cluster"), "deployment ray_cluster")
        runtime = _mapping(resolved.get("runtime"), "resolved runtime")
        monitoring = _mapping(runtime.get("monitoring"), "runtime monitoring")
        interval = float(monitoring["gpu_sample_interval_seconds"])
        if interval <= 0:
            raise ValueError("gpu_sample_interval_seconds must be positive")
        ray.init(
            address=str(cluster["address"]),
            namespace="ade",
            ignore_reinit_error=True,
        )
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, self._request_stop)
        signal.signal(signal.SIGINT, self._request_stop)
        run_dir = self.runs_root / self.run_id
        while not self.stop_requested and not (run_dir / "run.json").is_file():
            time.sleep(min(1.0, interval))
        if self.stop_requested:
            ray.shutdown()
            return 0

        usage_root = run_dir / "usage"
        usage_root.mkdir(parents=True, exist_ok=True)
        source_commit = _git_head(self.project_root)
        provenance_index = _append_source_provenance(
            run_dir,
            project_root=self.project_root,
            run_id=self.run_id,
            config_digest=compiled.config_digest,
            source_commit=source_commit,
            event="monitor_started",
        )
        sample_path = usage_root / "gpu-samples.jsonl"
        compressed_path = usage_root / "gpu-samples.jsonl.gz"
        _restore_compressed_samples(sample_path, compressed_path)
        event_path = usage_root / "ray-pool-events.jsonl"
        known_events = _event_ids(event_path)
        tracking = _RunMonitorTracking.start(
            settings=_mapping(resolved.get("tracking"), "resolved tracking"),
            run_id=self.run_id,
            local_root=run_dir / "tracking" / "run-monitor",
            fork_lineage=_read_json(run_dir / "run.json").get("forked_from"),
            source_commit=source_commit,
            provenance_index=provenance_index,
        )
        evaluation_tracking_roots = _evaluation_tracking_roots(
            resolved,
            fallback_root=(
                deployment_runtime_roots(self.project_root, resolved)["engine_work"]
                / self.run_id
                / "evaluation"
            ),
        )
        evaluation_tracking_roots = tuple(
            sorted(
                {
                    *evaluation_tracking_roots,
                    (run_dir / "tracking" / "seed-import").resolve(),
                },
                key=str,
            )
        )
        sample_index = len(_read_jsonl(sample_path))
        failure: str | None = None
        last_evaluation_reconciliation = 0.0
        evaluation_root_cursor = 0
        last_monitor_remote_check = 0.0
        last_seed_reconciliation = 0.0
        repository = FileRunRepository(self.runs_root)
        _write_json(
            run_dir / "tracking" / "run-monitor-health.json",
            tracking.health(process_status="running", sample_count=sample_index),
        )
        try:
            while not self.stop_requested:
                sample = self._sample(run_dir)
                _append_jsonl(sample_path, sample)
                tracking.log(sample, step=sample_index)
                sample_index += 1
                _write_json(
                    run_dir / "tracking" / "run-monitor-health.json",
                    tracking.health(
                        process_status="running",
                        sample_count=sample_index,
                        last_sampled_at=float(sample["sampled_at"]),
                    ),
                )
                now = time.monotonic()
                remote_interval = (
                    _WANDB_REMOTE_STEADY_CHECK_SECONDS
                    if tracking.remote_confirmed
                    else _WANDB_REMOTE_STARTUP_CHECK_SECONDS
                )
                if now - last_monitor_remote_check >= remote_interval:
                    tracking.reconcile_remote()
                    last_monitor_remote_check = now
                    _write_json(
                        run_dir / "tracking" / "run-monitor-health.json",
                        tracking.health(
                            process_status="running",
                            sample_count=sample_index,
                            last_sampled_at=float(sample["sampled_at"]),
                        ),
                    )
                if now - last_evaluation_reconciliation >= 600.0:
                    root, evaluation_root_cursor = _next_pending_evaluation_root(
                        evaluation_tracking_roots,
                        start=evaluation_root_cursor,
                    )
                    if root is not None:
                        reconcile_evaluation_tracking(
                            root,
                            max_requests=1,
                            verify_remote=False,
                        )
                    last_evaluation_reconciliation = now
                if now - last_seed_reconciliation >= 600.0:
                    current = repository.load(self.run_id)
                    if current.seeded_from is not None:
                        reconcile_seed_wandb_history(
                            target_repository=repository,
                            target=current,
                            target_tracking=dict(resolved.get("tracking", {})),
                        )
                    last_seed_reconciliation = now
                for event in sample["allocation_snapshot"]["events"]:
                    event_id = str(event["event_id"])
                    if event_id not in known_events:
                        _append_jsonl(event_path, event)
                        known_events.add(event_id)
                if sample.get("run_status") in _TERMINAL_RUN_STATUSES:
                    break
                self._wait(interval)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            summary = _summarize_samples(sample_path)
            summary["ray_cluster_id"] = cluster["cluster_id"]
            summary_path = usage_root / "gpu-summary.json"
            _write_json(summary_path, summary)
            _compress_samples(sample_path, compressed_path)
            tracking_result = tracking.finish(summary)
            _write_json(run_dir / "tracking" / "run-monitor.json", tracking_result)
            _write_json(
                run_dir / "tracking" / "run-monitor-health.json",
                tracking.health(
                    process_status="failed" if failure else "stopped",
                    sample_count=sample_index,
                    failure=failure,
                ),
            )
            terminal_status = _read_json(run_dir / "run.json").get("status")
            if terminal_status in _TERMINAL_RUN_STATUSES:
                cleanup_receipt = cleanup_cancelled_run(
                    run_id=self.run_id,
                    run_dir=run_dir,
                    queue_root=run_dir.parent.parent / "queue",
                    review_root=run_dir.parent.parent / "review",
                    terminate_processes=False,
                    artifact_cache_dirs=(
                        str(resources["artifact_staging"]["cache_dir"]),
                    ),
                )
                _write_json(
                    usage_root / "artifact-cache-cleanup.json",
                    cleanup_receipt["artifact_cache_cleanup"],
                )
            ray.shutdown()
        return 0

    def _sample(self, run_dir: Path) -> dict[str, Any]:
        nodes = [
            node
            for node in ray.nodes()
            if node.get("Alive")
            and float((node.get("Resources") or {}).get("GPU", 0) or 0) > 0
        ]
        refs = [
            gpu_snapshot_task.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=str(node["NodeID"]),
                    soft=False,
                )
            ).remote(self.run_id)
            for node in nodes
        ]
        node_samples = sorted(ray.get(refs), key=lambda item: str(item["node_ip"]))
        allocation_snapshot = ray.get(
            get_gpu_lease_allocator().monitor_snapshot.remote(run_id=self.run_id)
        )
        state = _read_json(run_dir / "run.json")
        leases = allocation_snapshot["leases"]
        external = allocation_snapshot["external_allocations"]
        allocated_node_ids = {
            str(node_id)
            for lease in leases
            for node_id in (lease.get("placement_node_ids") or [])
        }
        for node_sample in node_samples:
            if str(node_sample.get("node_id")) not in allocated_node_ids:
                continue
            for gpu in node_sample.get("gpus") or []:
                gpu["state"] = "this_run_allocated"
                gpu["allocation_source"] = "ray_placement_group"
        allocated_gpus = sum(int(item["allocated_gpus"]) for item in leases) + sum(
            len(item["gpu_slots"]) for item in external
        )
        judge_gpus = 0
        resources = state.get("run_resources") or {}
        judge = resources.get("local_judge") or {}
        attachment_path = judge.get("state_path")
        if attachment_path and Path(attachment_path).is_file():
            attachment = _read_json(Path(attachment_path))
            handle = attachment.get("service_handle") or {}
            if attachment.get("status") == "attached" and handle.get("actor_name"):
                try:
                    actor = ray.get_actor(handle["actor_name"], namespace="ade")
                    live = ray.get(actor.describe.remote(), timeout=10)["handle"]
                    if live["launch_id"] == handle["launch_id"]:
                        judge_gpus = _attribute_judge_gpus(node_samples, live)
                except (ValueError, ray.exceptions.RayError):
                    pass
        allocated_gpus += judge_gpus
        return {
            "schema_version": "1",
            "sampled_at": time.time(),
            "run_id": self.run_id,
            "run_status": state.get("status"),
            "revision": state.get("revision"),
            "last_transition": _mapping_or_none(state.get("last_transition")),
            "active_agent_calls": state.get("active_agent_calls", []),
            "active_engine_commands": state.get("active_engine_commands", []),
            "active_review_commands": state.get("active_review_commands", []),
            "usage": _usage_projection(run_dir),
            "available_gpus": sum(len(item.get("gpus") or []) for item in node_samples),
            "allocated_gpus": allocated_gpus,
            "judge_allocated_gpus": judge_gpus,
            "nodes": node_samples,
            "allocation_snapshot": allocation_snapshot,
        }

    def _request_stop(self, _signum, _frame) -> None:
        self.stop_requested = True

    def _wait(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self.stop_requested and time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def _attribute_judge_gpus(node_samples, handle):
    """Project the live actor reservation onto the same pool's physical GPUs."""
    gpu_ids = {str(value) for value in handle["gpu_ids"]}
    for node in node_samples:
        if node.get("node_id") != handle["node_id"]:
            continue
        for gpu in node.get("gpus") or []:
            if str(gpu["index"]) in gpu_ids or gpu.get("uuid") in gpu_ids:
                gpu["state"] = "this_run_allocated"
                gpu["allocation_source"] = "ray_judge_actor"
    return len(gpu_ids)


def _evaluation_tracking_roots(
    resolved: Mapping[str, Any],
    *,
    fallback_root: Path | None = None,
) -> tuple[Path, ...]:
    """Find compiler-resolved evaluation roots, including the task run root."""
    roots: set[Path] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            tracking = value.get("evaluation_tracking")
            run_dir = value.get("run_dir")
            if (
                isinstance(tracking, Mapping)
                and bool(tracking.get("enabled"))
                and isinstance(run_dir, str)
                and run_dir.strip()
            ):
                roots.add(Path(run_dir).resolve() / "wandb-evaluations")
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(resolved)
    if (
        fallback_root is not None
        and isinstance(resolved.get("tracking"), Mapping)
        and bool(resolved["tracking"].get("enabled"))
    ):
        roots.add(fallback_root.resolve() / "wandb-evaluations")
    return tuple(sorted(roots, key=str))


def _next_pending_evaluation_root(
    roots: tuple[Path, ...],
    *,
    start: int,
) -> tuple[Path | None, int]:
    """Select one pending root per cycle and rotate fairly across roots."""
    if not roots:
        return None, 0
    for offset in range(len(roots)):
        index = (start + offset) % len(roots)
        root = roots[index]
        for request_path in root.glob("*/records/*/tracking-request.json"):
            status_path = request_path.with_name("tracking-status.json")
            try:
                status = _read_json(status_path) if status_path.is_file() else {}
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                status = {}
            if status.get("status") != "published":
                return root, (index + 1) % len(roots)
    return None, start % len(roots)


class _RunMonitorTracking:
    def __init__(
        self,
        *,
        enabled: bool,
        project: str = "",
        group: str = "",
        tracking_id: str = "",
        local_dir: Path | None = None,
        run: Any = None,
        previous_environment: Mapping[str, str | None] | None = None,
        error: str | None = None,
        settings: Mapping[str, Any] | None = None,
        source_run_id: str = "",
        fork_lineage: object = None,
        mode: str = "offline",
    ) -> None:
        self.enabled = enabled
        self.project = project
        self.group = group
        self.tracking_id = tracking_id
        self.local_dir = local_dir
        self.run = run
        self.previous_environment = dict(previous_environment or {})
        self.error = error
        self.settings = dict(settings or {})
        self.source_run_id = source_run_id
        self.fork_lineage = fork_lineage
        self.mode = mode
        self._retry_attempts = 0
        self._retry_after = 0.0
        self._first_local_log_at: float | None = None
        self._last_local_step: int | None = None
        self._last_remote_checked_at: str | None = None
        self._last_remote_step: int | None = None
        self._last_remote_advanced_at: float | None = None
        self._remote_error: str | None = None

    @classmethod
    def start(
        cls,
        *,
        settings: Mapping[str, Any],
        run_id: str,
        local_root: Path,
        fork_lineage: object = None,
        source_commit: str | None = None,
        provenance_index: int | None = None,
    ) -> "_RunMonitorTracking":
        if not bool(settings.get("enabled")):
            return cls(
                enabled=False,
                settings=settings,
                source_run_id=run_id,
                fork_lineage=fork_lineage,
                mode=str(settings.get("mode") or "offline"),
            )
        # Monitoring follows the same root-project/fork-lineage identity as
        # training and evaluation tracking.
        from ade.engine.telemetry.tracking import _project_for_run

        project = _project_for_run(
            settings,
            current_run_id=run_id,
            fork_lineage=(
                fork_lineage if isinstance(fork_lineage, Mapping) else None
            ),
        )
        mode = str(settings.get("mode") or "offline")
        entity = str(
            os.environ.get(str(settings.get("entity_env") or "WANDB_ENTITY")) or ""
        )
        base_url = str(
            os.environ.get(str(settings.get("base_url_env") or "WANDB_BASE_URL")) or ""
        )
        api_key = str(
            os.environ.get(str(settings.get("api_key_env") or "WANDB_API_KEY")) or ""
        )
        tracking_id = f"{run_id}--run-monitor"
        local_dir = local_root / tracking_id
        local_dir.mkdir(parents=True, exist_ok=True)
        environment = {
            "WANDB_MODE": mode,
            "WANDB_PROJECT": project,
            "WANDB_RUN_GROUP": run_id,
            "WANDB_RUN_ID": tracking_id,
            "WANDB_RESUME": "allow",
            "WANDB_DIR": str(local_dir),
            "WANDB_DISABLE_STATS": "true",
        }
        if isinstance(fork_lineage, Mapping):
            environment["WANDB_TAGS"] = ",".join(
                f"{name}:{fork_lineage[name]}"
                for name in (
                    "lineage_root_run_id",
                    "source_run_id",
                    "source_revision",
                    "generation",
                )
                if fork_lineage.get(name) is not None
            )
        if entity:
            environment["WANDB_ENTITY"] = entity
        if base_url:
            environment["WANDB_BASE_URL"] = base_url
        if api_key:
            environment["WANDB_API_KEY"] = api_key
        if mode == "online":
            environment.update(_wandb_no_proxy_environment(base_url))
        previous = {name: os.environ.get(name) for name in environment}
        os.environ.update(environment)
        try:
            import wandb

            bounded_settings = bounded_wandb_settings(wandb)
            run = wandb.init(
                project=project,
                entity=entity or None,
                group=run_id,
                job_type="run-monitor",
                name=f"{run_id}/run-monitor",
                id=tracking_id,
                resume="allow",
                dir=str(local_dir),
                config={
                    "ade_run_id": run_id,
                    "ade": {
                        "source_commit": source_commit,
                        "source_provenance_index": provenance_index,
                    },
                    "sample_interval_source": "runtime.monitoring",
                    "fork_lineage": (
                        dict(fork_lineage)
                        if isinstance(fork_lineage, Mapping)
                        else None
                    ),
                },
                **(
                    {"settings": bounded_settings}
                    if bounded_settings is not None
                    else {}
                ),
            )
        except Exception as error:
            _restore_environment(previous)
            return cls(
                enabled=True,
                project=project,
                group=run_id,
                tracking_id=tracking_id,
                local_dir=local_dir,
                error=f"{type(error).__name__}: {error}",
                settings=settings,
                source_run_id=run_id,
                fork_lineage=fork_lineage,
                mode=mode,
            )
        return cls(
            enabled=True,
            project=project,
            group=run_id,
            tracking_id=tracking_id,
            local_dir=local_dir,
            run=run,
            previous_environment=previous,
            settings=settings,
            source_run_id=run_id,
            fork_lineage=fork_lineage,
            mode=mode,
        )

    def log(self, sample: Mapping[str, Any], *, step: int) -> None:
        if not self.enabled:
            return
        if self.run is None or self.error is not None:
            if time.monotonic() < self._retry_after:
                return
            retry_attempts = self._retry_attempts
            first_local_log_at = self._first_local_log_at
            last_local_step = self._last_local_step
            last_remote_checked_at = self._last_remote_checked_at
            last_remote_step = self._last_remote_step
            last_remote_advanced_at = self._last_remote_advanced_at
            remote_error = self._remote_error
            self._close_failed_run()
            replacement = type(self).start(
                settings=self.settings,
                run_id=self.source_run_id,
                local_root=self.local_dir.parent if self.local_dir else Path("."),
                fork_lineage=self.fork_lineage,
            )
            self.__dict__.update(replacement.__dict__)
            self._retry_attempts = retry_attempts
            self._first_local_log_at = first_local_log_at
            self._last_local_step = last_local_step
            self._last_remote_checked_at = last_remote_checked_at
            self._last_remote_step = last_remote_step
            self._last_remote_advanced_at = last_remote_advanced_at
            self._remote_error = remote_error
            if self.run is None or self.error is not None:
                self._retry_attempts += 1
                delay = min(
                    600.0,
                    float(self.settings.get("wandb_retry_backoff_seconds") or 30.0)
                    * (2 ** min(self._retry_attempts - 1, 4)),
                )
                self._retry_after = time.monotonic() + delay
        if self.run is None or self.error is not None:
            return
        try:
            self.run.log(_monitor_metrics(sample, step=step), step=step)
            now = time.monotonic()
            if self._first_local_log_at is None:
                self._first_local_log_at = now
            self._last_local_step = step
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    @property
    def remote_confirmed(self) -> bool:
        return self.mode != "online" or self._last_remote_step is not None

    def reconcile_remote(self, *, wandb_module: Any = None) -> None:
        if (
            not self.enabled
            or self.mode != "online"
            or self._last_local_step is None
        ):
            return
        checked_at = datetime.now(timezone.utc).isoformat()
        now = time.monotonic()
        try:
            if wandb_module is None:
                import wandb as wandb_module

            base_url = str(
                os.environ.get(
                    str(self.settings.get("base_url_env") or "WANDB_BASE_URL")
                )
                or ""
            ).strip()
            api_kwargs = (
                {"overrides": {"base_url": base_url}} if base_url else {}
            )
            api = wandb_module.Api(
                timeout=WANDB_API_TIMEOUT_SECONDS,
                **api_kwargs,
            )
            flush = getattr(api, "flush", None)
            if callable(flush):
                flush()
            entity = str(
                os.environ.get(
                    str(self.settings.get("entity_env") or "WANDB_ENTITY")
                )
                or ""
            ).strip()
            path = (
                f"{entity}/{self.project}/{self.tracking_id}"
                if entity
                else f"{self.project}/{self.tracking_id}"
            )
            remote = api.run(path)
            remote_step = getattr(remote, "lastHistoryStep", None)
            remote_group = getattr(remote, "group", None)
            remote_job_type = getattr(remote, "job_type", None)
            remote_config = dict(getattr(remote, "config", {}) or {})
            if (
                type(remote_step) is not int
                or remote_step < 0
                or remote_group != self.group
                or remote_job_type != "run-monitor"
                or remote_config.get("ade_run_id") != self.source_run_id
            ):
                raise ValueError("remote Run Monitor identity/history is incomplete")
            if self._last_remote_step is None or remote_step > self._last_remote_step:
                self._last_remote_advanced_at = now
            self._last_remote_step = max(
                remote_step,
                self._last_remote_step if self._last_remote_step is not None else -1,
            )
            self._remote_error = None
        except Exception as error:
            self._remote_error = f"{type(error).__name__}: {error}"
        self._last_remote_checked_at = checked_at

    def _close_failed_run(self) -> None:
        if self.run is not None:
            try:
                self.run.finish(exit_code=1)
            except Exception:
                pass
        _restore_environment(self.previous_environment)
        self.run = None
        self.previous_environment = {}

    def health(
        self,
        *,
        process_status: str,
        sample_count: int,
        last_sampled_at: float | None = None,
        failure: str | None = None,
    ) -> dict[str, Any]:
        now = time.monotonic()
        if not self.enabled:
            tracking_status = "disabled"
        elif self.error:
            tracking_status = "unhealthy"
        elif self.mode != "online":
            tracking_status = "healthy"
        elif self._last_remote_step is None:
            elapsed = (
                now - self._first_local_log_at
                if self._first_local_log_at is not None
                else 0.0
            )
            tracking_status = (
                "unhealthy"
                if elapsed >= _WANDB_INITIAL_REMOTE_GRACE_SECONDS
                else "pending_remote"
            )
        elif (
            self._last_remote_advanced_at is None
            or now - self._last_remote_advanced_at
            >= _WANDB_REMOTE_STALL_SECONDS
        ):
            tracking_status = "unhealthy"
        else:
            tracking_status = "healthy"
        result: dict[str, Any] = {
            "schema_version": "1",
            "process_status": process_status,
            "sample_count": sample_count,
            "last_sampled_at": last_sampled_at,
            "tracking": {
                "enabled": self.enabled,
                "status": tracking_status,
                "project": self.project or None,
                "group": self.group or None,
                "run_id": self.tracking_id or None,
                "mode": self.mode,
                "last_local_step": self._last_local_step,
                "last_remote_step": self._last_remote_step,
                "last_remote_checked_at": self._last_remote_checked_at,
            },
        }
        if self.error:
            result["tracking"]["error"] = self.error
        if self._remote_error:
            result["tracking"]["remote_error"] = self._remote_error
        if failure:
            result["failure"] = failure
        return result

    def finish(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "status": "disabled"}
        if self.run is not None:
            try:
                self.run.summary.update(dict(summary))
                self.run.finish(exit_code=0 if self.error is None else 1)
            except Exception as error:
                self.error = self.error or f"{type(error).__name__}: {error}"
            finally:
                _restore_environment(self.previous_environment)
        terminal_health = self.health(
            process_status="stopped",
            sample_count=int(summary.get("sample_count") or 0),
        )
        result = {
            "enabled": True,
            "status": (
                "complete"
                if terminal_health["tracking"]["status"] == "healthy"
                else "partial"
            ),
            "project": self.project,
            "group": self.group,
            "run_id": self.tracking_id,
            "job_type": "run-monitor",
            "local_run_dir": str(self.local_dir),
        }
        if self.error:
            result["error"] = self.error
        return result


def _monitor_metrics(sample: Mapping[str, Any], *, step: int) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {
        "ade/monitor_step": step,
        "ade/revision": int(sample.get("revision") or 0),
        "ade/gpu/available": int(sample.get("available_gpus") or 0),
        "ade/gpu/allocated": int(sample.get("allocated_gpus") or 0),
    }
    utilizations = []
    for node in sample.get("nodes") or []:
        for gpu in node.get("gpus") or []:
            utilization = float(gpu.get("utilization_gpu_percent") or 0.0)
            utilizations.append(utilization)
            key = _metric_key(str(node.get("node_ip")), str(gpu.get("uuid")))
            metrics[f"ade/gpu/{key}/utilization_percent"] = utilization
            metrics[f"ade/gpu/{key}/memory_used_bytes"] = int(
                gpu.get("memory_used_bytes") or 0
            )
            metrics[f"ade/gpu/{key}/power_watts"] = float(
                gpu.get("power_watts") or 0.0
            )
    metrics["ade/gpu/mean_utilization_percent"] = (
        sum(utilizations) / len(utilizations) if utilizations else 0.0
    )
    usage = sample.get("usage")
    if isinstance(usage, dict):
        _usage_metrics(metrics, "ade/tokens/run", usage.get("incremental"))
        for dimension, prefix in (
            ("by_category", "ade/tokens/category"),
            ("by_transition", "ade/tokens/transition"),
        ):
            groups = usage.get(dimension)
            if not isinstance(groups, dict):
                continue
            for name, counters in groups.items():
                _usage_metrics(
                    metrics,
                    f"{prefix}/{_metric_segment(str(name))}",
                    counters,
                )
    return metrics


def _usage_metrics(
    metrics: dict[str, int | float],
    prefix: str,
    counters: object,
) -> None:
    if not isinstance(counters, dict):
        return
    for source, target in (
        ("prompt_tokens", "prompt"),
        ("completion_tokens", "completion"),
        ("total_tokens", "total"),
        ("cached_tokens", "cached"),
        ("reasoning_tokens", "reasoning"),
        ("requests", "requests"),
    ):
        value = counters.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"{prefix}/{target}"] = value


def _usage_projection(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "usage" / "usage-summary.json"
    transition_path = run_dir / "usage" / "by-transition.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    transitions = _read_json(transition_path) if transition_path.is_file() else {}
    return {
        "incremental": _mapping_or_empty(summary.get("incremental_usage")),
        "by_category": _mapping_or_empty(summary.get("by_category")),
        "by_transition": _mapping_or_empty(transitions.get("by_transition")),
    }


def _restore_environment(previous: Mapping[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _summarize_samples(path: Path) -> dict[str, Any]:
    samples = _read_jsonl(path)
    allocated_seconds = available_seconds = busy_equivalent_seconds = 0.0
    per_gpu: dict[str, dict[str, float]] = {}
    for index, sample in enumerate(samples[:-1]):
        elapsed = max(0.0, float(samples[index + 1]["sampled_at"]) - float(sample["sampled_at"]))
        available = int(sample.get("available_gpus") or 0)
        allocated = int(sample.get("allocated_gpus") or 0)
        available_seconds += available * elapsed
        allocated_seconds += allocated * elapsed
        for node in sample.get("nodes") or []:
            for gpu in node.get("gpus") or []:
                utilization = float(gpu.get("utilization_gpu_percent") or 0.0)
                busy_equivalent_seconds += utilization / 100.0 * elapsed
                key = f"{node.get('node_id')}+{gpu.get('uuid')}"
                counters = per_gpu.setdefault(key, {"seconds": 0.0, "utilization_integral": 0.0})
                counters["seconds"] += elapsed
                counters["utilization_integral"] += utilization * elapsed
    gpu_summary = {
        key: {
            "sampled_seconds": round(value["seconds"], 3),
            "mean_utilization_percent": round(
                value["utilization_integral"] / value["seconds"], 3
            ) if value["seconds"] else 0.0,
        }
        for key, value in sorted(per_gpu.items())
    }
    return {
        "schema_version": "1",
        "status": "complete" if samples else "unavailable",
        "sample_count": len(samples),
        "pool_gpu_seconds": round(available_seconds, 3),
        "allocated_gpu_seconds": round(allocated_seconds, 3),
        "busy_gpu_equivalent_seconds": round(busy_equivalent_seconds, 3),
        "pool_occupancy": round(allocated_seconds / available_seconds, 6) if available_seconds else 0.0,
        "hardware_utilization": round(busy_equivalent_seconds / available_seconds, 6) if available_seconds else 0.0,
        "allocation_efficiency": round(busy_equivalent_seconds / allocated_seconds, 6) if allocated_seconds else 0.0,
        "per_gpu": gpu_summary,
    }


def _compress_samples(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with source.open("rb") as reader, gzip.open(temporary, "wb") as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
    os.replace(temporary, target)


def _restore_compressed_samples(source: Path, compressed: Path) -> None:
    if source.is_file() or not compressed.is_file():
        return
    source.parent.mkdir(parents=True, exist_ok=True)
    temporary = source.with_suffix(source.suffix + ".tmp")
    with gzip.open(compressed, "rb") as reader, temporary.open("wb") as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
    os.replace(temporary, source)


def _event_ids(path: Path) -> set[str]:
    return {str(row.get("event_id")) for row in _read_jsonl(path) if row.get("event_id")}


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for value in [json.loads(line)]
        if isinstance(value, dict)
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _mapping(value: Any, owner: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a mapping")
    return value


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _metric_key(node_ip: str, gpu_uuid: str) -> str:
    return "_".join(
        "".join(character if character.isalnum() else "_" for character in item)
        for item in (node_ip, gpu_uuid)
    )


def _metric_segment(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)
