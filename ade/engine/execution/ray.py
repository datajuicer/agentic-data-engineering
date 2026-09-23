from __future__ import annotations

import ctypes
import json
import hashlib
import math
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import ray
from ray._private.state import available_resources_per_node
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)

from ade.core.engine import EngineAttemptError
from ade.engine.eval.metrics import merge_data_parallel_details, resolve_primary_metric, selected_score, summarize_details
from ade.engine.eval.logs import build_eval_log_payload
from ade.engine.eval.profiles import aggregate_validation_metrics
from ade.engine.eval.registry import get_task, task_type_from_config
from ade.engine.execution.gpu import (
    acquire_gpu_lease,
    lease_metadata,
    release_gpu_lease,
    release_gpu_leases_for_owner,
)
from ade.engine.execution.paths import PROJECT_ROOT
from ade.engine.telemetry.tracking import audit_ref_from_dict, build_training_tracking
from ade.engine.telemetry.training import (
    LLAMAFACTORY_METRIC_PROFILE,
    MetricProfile,
    TelemetryExportSpec,
    export_exact_wandb_telemetry,
)
from ade.engine.execution.coordinator_resources import validate_coordinator_workload_request


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def safe_log_part(value: Any) -> str:
    if value is None or value == "":
        return "null"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "null"


def bounded_log_part(value: Any, max_length: int = 180) -> str:
    text = safe_log_part(value)
    if len(text) <= max_length:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    keep = max_length - len(digest) - 3
    head = max(1, keep // 2)
    tail = max(1, keep - head)
    return f"{text[:head]}__{digest}_{text[-tail:]}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_record_count(path: str | Path) -> int:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0
    if text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, list):
            return len(data)
        return 1
    return sum(1 for line in text.splitlines() if line.strip())


def checkpoint_step(path: str | Path) -> int | None:
    match = CHECKPOINT_RE.match(Path(path).name)
    return int(match.group(1)) if match else None


def infer_epoch(step: int, steps_per_epoch: int) -> int:
    return int(math.ceil(step / max(1, steps_per_epoch)))


def _sft_checkpoint_save_steps(*, steps_per_epoch: int, artifact_interval: int) -> int:
    if steps_per_epoch < 1 or artifact_interval < 1:
        raise ValueError("SFT steps_per_epoch and artifact_interval must be positive")
    return steps_per_epoch * artifact_interval


def list_checkpoints(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []
    checkpoints: list[tuple[int, Path]] = []
    for child in root.iterdir():
        step = checkpoint_step(child)
        if child.is_dir() and step is not None:
            checkpoints.append((step, child))
    return [path for _, path in sorted(checkpoints)]


def checkpoint_ready(path: str | Path, stable_seconds: float = 8.0) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    if checkpoint_step(path) is not None and not (path / "trainer_state.json").exists():
        return False
    required_any = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    ]
    if not any((path / name).exists() for name in required_any):
        if not list(path.glob("*.safetensors")) and not list(path.glob("pytorch_model*.bin")):
            return False
    try:
        newest = max(p.stat().st_mtime for p in path.rglob("*") if p.is_file())
    except ValueError:
        return False
    return time.time() - newest >= stable_seconds


def _add_baseline_path(project_root: str | Path) -> Path:
    root = Path(project_root)
    if not (root / "third_party/llamafactory").exists():
        root = PROJECT_ROOT
    for path in (root, root / "third_party/llamafactory" / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root


def _upsert_scalar_config(content: str, key: str, value: Any) -> str:
    if value is None:
        rendered = "null"
    elif isinstance(value, bool):
        value = "true" if value else "false"
        rendered = value
    elif isinstance(value, (dict, list)):
        rendered = json.dumps(value)
    elif isinstance(value, str):
        rendered = json.dumps(value, ensure_ascii=False)
    else:
        rendered = value
    pattern = rf"^{re.escape(key)}:.*$"
    replacement = f"{key}: {rendered}"
    if re.search(pattern, content, flags=re.MULTILINE):
        return re.sub(
            pattern,
            lambda _match: replacement,
            content,
            count=1,
            flags=re.MULTILINE,
        )
    return f"{content.rstrip()}\n{replacement}\n"


def rewrite_training_config(base_config: str, updates: dict[str, Any]) -> str:
    content = re.sub(r"^resume_from_checkpoint:.*\n?", "", base_config, flags=re.MULTILINE)
    for key in (
        "eval_dataset",
        "per_device_eval_batch_size",
        "eval_strategy",
        "save_total_limit",
    ):
        content = re.sub(rf"^\s*{re.escape(key)}:.*\n?", "", content, flags=re.MULTILINE)
    for key, value in updates.items():
        content = _upsert_scalar_config(content, key, value)
    return f"{content.rstrip()}\n"


def _llamafactory_train_command(config_path: str | Path, project_root: str | Path) -> list[str]:
    project_root = str(Path(project_root).resolve())
    lf_src = str((Path(project_root) / "third_party/llamafactory" / "src").resolve())
    launcher = (
        "import sys; "
        f"sys.path[:0] = [{lf_src!r}, {project_root!r}]; "
        "from llamafactory.cli import main; "
        "main()"
    )
    return [sys.executable, "-c", launcher, "train", str(config_path)]


def _resolve_llamafactory_path(project_root: Path, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    candidates = [
        project_root / value,
        project_root / "third_party/llamafactory" / value,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str((project_root / value).resolve())


def kill_process_group(process: subprocess.Popen) -> None:
    try:
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _prepare_training_child() -> None:
    """Put the child in its own group and kill it if its Engine worker dies."""

    os.setsid()
    if sys.platform != "linux":
        return
    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    # Close the fork/parent-death race: if the parent died before prctl was
    # installed, do not continue as a reparented orphan.
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def _descendant_pids(root_pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    proc_root = Path("/proc")
    for stat_path in proc_root.glob("[0-9]*/stat"):
        try:
            text = stat_path.read_text(encoding="utf-8")
            end = text.rfind(")")
            if end < 0:
                continue
            fields = text[end + 2 :].split()
            pid = int(stat_path.parent.name)
            ppid = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(pid)

    result: list[int] = []
    stack = list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        if pid == root_pid or pid in result:
            continue
        result.append(pid)
        stack.extend(children.get(pid, []))
    return result


def kill_descendant_processes(root_pid: int | None = None) -> list[int]:
    """Best-effort cleanup for subprocesses spawned inside a Ray worker."""
    root_pid = root_pid or os.getpid()
    killed: list[int] = []
    pids = _descendant_pids(root_pid)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
                if pid not in killed:
                    killed.append(pid)
            except ProcessLookupError:
                pass
            except OSError:
                pass
        if sig == signal.SIGTERM and killed:
            time.sleep(2)
    return killed


@ray.remote(num_gpus=0)
def gpu_snapshot_task(run_id: str | None = None) -> dict[str, Any]:
    def process_scope(pid: int) -> dict[str, Any]:
        values: dict[str, str] = {}
        process_name = ""
        current_pid = pid
        for _ in range(12):
            try:
                raw = Path(f"/proc/{current_pid}/environ").read_bytes()
                environment = {
                    item.partition(b"=")[0].decode("utf-8", errors="replace"): item.partition(b"=")[2].decode(
                        "utf-8", errors="replace"
                    )
                    for item in raw.split(b"\0")
                    if b"=" in item
                }
            except OSError:
                environment = {}
            for name in (
                "ADE_RUN_ID",
                "ADE_COORDINATOR_ID",
                "ADE_PLAN_ID",
                "ADE_TRIAL_ID",
                "ADE_ENGINE_COMMAND_ID",
                "ADE_WORKLOAD",
            ):
                if environment.get(name):
                    values[name.removeprefix("ADE_").lower()] = environment[name]
            if current_pid == pid:
                try:
                    process_name = Path(f"/proc/{pid}/comm").read_text(
                        encoding="utf-8", errors="replace"
                    ).strip()
                except OSError:
                    pass
            if values.get("run_id"):
                break
            try:
                status = Path(f"/proc/{current_pid}/status").read_text(
                    encoding="utf-8", errors="replace"
                )
                parent_line = next(
                    line for line in status.splitlines() if line.startswith("PPid:")
                )
                parent_pid = int(parent_line.split(":", 1)[1].strip())
            except (OSError, StopIteration, ValueError):
                break
            if parent_pid <= 1 or parent_pid == current_pid:
                break
            current_pid = parent_pid
        values["process_name"] = process_name
        if current_pid != pid and values.get("run_id"):
            values["scope_source_pid"] = str(current_pid)
        return values

    snapshot = {
        "sampled_at": time.time(),
        "node_id": str(ray.get_runtime_context().get_node_id()),
        "node_ip": ray.util.get_node_ip_address(),
        "hostname": socket.gethostname(),
        "status": "complete",
        "gpus": [],
    }
    try:
        import pynvml

        pynvml.nvmlInit()
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            processes = []
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                scope = process_scope(int(process.pid))
                processes.append(
                    {
                        "pid": int(process.pid),
                        "used_memory_bytes": int(process.usedGpuMemory or 0),
                        **scope,
                    }
                )
            matching = [
                process
                for process in processes
                if run_id and process.get("run_id") == run_id
            ]
            state = (
                "this_run_allocated"
                if matching
                else "external_or_unattributed"
                if processes
                else "idle"
            )
            uuid_value = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid_value, bytes):
                uuid_value = uuid_value.decode("utf-8")
            snapshot["gpus"].append(
                {
                    "index": index,
                    "uuid": str(uuid_value),
                    "state": state,
                    "utilization_gpu_percent": int(utilization.gpu),
                    "utilization_memory_percent": int(utilization.memory),
                    "memory_used_bytes": int(memory.used),
                    "memory_total_bytes": int(memory.total),
                    "power_watts": round(
                        pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 3
                    ),
                    "temperature_c": int(
                        pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    ),
                    "processes": processes,
                }
            )
        pynvml.nvmlShutdown()
    except Exception as error:
        snapshot["status"] = "unavailable"
        snapshot["error"] = f"{type(error).__name__}: {error}"
    return snapshot


@ray.remote(num_cpus=0)
def model_stage_task(stage_request: dict[str, Any]) -> dict[str, Any]:
    from ade.engine.checkpoints.cache import stage_checkpoint_if_enabled

    staged = stage_checkpoint_if_enabled(stage_request)
    return {
        "node_id": str(ray.get_runtime_context().get_node_id()),
        "node_ip": ray.util.get_node_ip_address(),
        "hostname": socket.gethostname(),
        "source": staged.get("source_checkpoint_path", stage_request["checkpoint_path"]),
        "target": staged["checkpoint_path"],
        "cache_hit": bool(staged.get("checkpoint_staging_cache_hit")),
        "elapsed_seconds": float(staged.get("checkpoint_staging_elapsed_seconds") or 0.0),
    }


@ray.remote(num_cpus=0)
def artifact_cache_release_task(
    cache_dir: str, owner_id: str, owner_kind: str = "run"
) -> dict[str, Any]:
    from ade.engine.checkpoints.cache import (
        release_staging_for_consumer,
        release_staging_for_run,
    )

    release = (
        release_staging_for_consumer
        if owner_kind == "consumer"
        else release_staging_for_run
    )

    return {
        "node_id": str(ray.get_runtime_context().get_node_id()),
        "node_ip": ray.util.get_node_ip_address(),
        **release(cache_dir, owner_id),
    }


def stage_model_on_gpu_nodes(
    request: dict[str, Any],
    *,
    model_field: str = "model",
    node_ids: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not bool(request.get("model_staging")):
        return dict(request), {"status": "disabled", "nodes": []}
    model_path = str(request.get(model_field) or "").strip()
    if not model_path:
        raise ValueError(f"{model_field} is required for model staging")
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before cluster model staging")
    requested_node_ids = set(node_ids or ())
    nodes = [
        node
        for node in ray.nodes()
        if node.get("Alive")
        and float((node.get("Resources") or {}).get("GPU", 0) or 0) > 0
        and (not requested_node_ids or str(node["NodeID"]) in requested_node_ids)
    ]
    if not nodes:
        raise RuntimeError("artifact cache cleanup found no alive Ray GPU nodes")
    if not nodes:
        raise RuntimeError("model staging found no alive Ray GPU nodes")
    stage_request = {
        "checkpoint_path": model_path,
        "checkpoint_staging": True,
        "checkpoint_cache_dir": request.get("model_cache_dir"),
        "checkpoint_cache_max_gb": request.get("model_cache_max_gb"),
        "checkpoint_cache_lock_stale_seconds": request.get(
            "model_cache_lock_stale_seconds"
        ),
        "ade_run_id": request.get("ade_run_id")
        or request.get("run_id")
        or request.get("agent_task_id"),
        "staging_consumer_id": request.get("staging_consumer_id"),
    }
    refs = [
        model_stage_task.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=str(node["NodeID"]),
                soft=False,
            )
        ).remote(stage_request)
        for node in nodes
    ]
    staged_nodes = sorted(ray.get(refs), key=lambda item: str(item["node_ip"]))
    targets = {str(item["target"]) for item in staged_nodes}
    if len(targets) != 1:
        raise RuntimeError(f"model staging produced inconsistent node-local paths: {targets}")
    staged = dict(request)
    staged[model_field] = targets.pop()
    staged["source_model_path"] = model_path
    staged["model_staged"] = True
    return staged, {
        "schema_version": "1",
        "status": "complete",
        "source": model_path,
        "target": staged[model_field],
        "nodes": staged_nodes,
    }


def placement_group_node_ids(placement_group_handle) -> tuple[str, ...]:
    table = ray.util.placement_group_table(placement_group_handle)
    bundle_to_node = table.get("bundles_to_node_id") or {}
    node_ids = tuple(sorted({str(node_id) for node_id in bundle_to_node.values()}))
    if not node_ids:
        raise RuntimeError("GPU placement group has no resolved nodes")
    return node_ids


class GpuMemoryAdmissionError(RuntimeError):
    """No Ray GPU node has enough physical free memory for a workload."""


def _select_gpu_memory_candidate(
    snapshots: list[dict[str, Any]],
    *,
    required_gpus: int,
    minimum_free_fraction: float,
) -> dict[str, Any]:
    required_gpus = int(required_gpus)
    minimum_free_fraction = float(minimum_free_fraction)
    candidates = []
    for snapshot in snapshots:
        if snapshot.get("status") != "complete":
            continue
        qualifying = []
        for gpu in snapshot.get("gpus") or []:
            total = int(gpu.get("memory_total_bytes") or 0)
            used = int(gpu.get("memory_used_bytes") or 0)
            free_fraction = (total - used) / total if total > 0 else 0.0
            if free_fraction >= minimum_free_fraction:
                qualifying.append(free_fraction)
        if len(qualifying) >= required_gpus:
            candidates.append(
                (
                    min(qualifying),
                    str(snapshot.get("node_ip") or ""),
                    snapshot,
                )
            )
    if not candidates:
        details = [
            {
                "node_ip": snapshot.get("node_ip"),
                "status": snapshot.get("status"),
                "free_fractions": [
                    round(
                        (
                            int(gpu.get("memory_total_bytes") or 0)
                            - int(gpu.get("memory_used_bytes") or 0)
                        )
                        / int(gpu.get("memory_total_bytes") or 1),
                        4,
                    )
                    for gpu in snapshot.get("gpus") or []
                    if int(gpu.get("memory_total_bytes") or 0) > 0
                ],
            }
            for snapshot in snapshots
        ]
        raise GpuMemoryAdmissionError(
            "no Ray node has "
            f"{required_gpus} GPUs with free-memory fraction >= "
            f"{minimum_free_fraction:.3f}: {details}"
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def select_gpu_node_for_memory_admission(
    *,
    required_gpus: int,
    minimum_free_fraction: float,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized for GPU memory admission")
    # Physical VRAM can be free while a placement group reserves the GPUs.
    # Use unreserved Ray resources, including the training bundle CPU shape.
    available = available_resources_per_node()
    nodes = [
        node
        for node in ray.nodes()
        if node.get("Alive")
        and available.get(str(node["NodeID"]), {}).get("GPU", 0) >= required_gpus
        and available.get(str(node["NodeID"]), {}).get("CPU", 0) >= 10 * required_gpus
    ]
    refs = [
        gpu_snapshot_task.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=str(node["NodeID"]), soft=False
            )
        ).remote(None)
        for node in nodes
    ]
    try:
        snapshots = list(ray.get(refs, timeout=timeout_seconds))
    except ray.exceptions.GetTimeoutError as error:
        raise GpuMemoryAdmissionError("timed out sampling GPU memory for admission") from error
    # Refresh after sampling; reservations can change while snapshots run.
    available = available_resources_per_node()
    snapshots = [
        snapshot for snapshot in snapshots
        if available.get(str(snapshot["node_id"]), {}).get("GPU", 0) >= required_gpus
        and available.get(str(snapshot["node_id"]), {}).get("CPU", 0) >= 10 * required_gpus
    ]
    selected = _select_gpu_memory_candidate(
        snapshots,
        required_gpus=required_gpus,
        minimum_free_fraction=minimum_free_fraction,
    )
    node_ip = str(selected["node_ip"])
    return {
        "node_id": str(selected["node_id"]),
        "node_ip": node_ip,
        "node_resource": f"node:{node_ip}",
        "minimum_free_fraction": float(minimum_free_fraction),
    }


def _release_artifact_cache_on_gpu_nodes(
    *,
    cache_dir: str,
    owner_id: str,
    owner_kind: str,
    node_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before artifact cache cleanup")
    requested_node_ids = set(node_ids or ())
    nodes = [
        node
        for node in ray.nodes()
        if node.get("Alive")
        and float((node.get("Resources") or {}).get("GPU", 0) or 0) > 0
        and (not requested_node_ids or str(node["NodeID"]) in requested_node_ids)
    ]
    refs = [
        artifact_cache_release_task.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=str(node["NodeID"]), soft=False
            )
        ).remote(cache_dir, owner_id, owner_kind)
        for node in nodes
    ]
    return {
        "schema_version": "1",
        f"{owner_kind}_id": owner_id,
        "nodes": sorted(ray.get(refs), key=lambda item: str(item["node_ip"])),
    }


def release_artifact_cache_consumer_on_gpu_nodes(
    *,
    cache_dir: str,
    consumer_id: str,
    node_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return _release_artifact_cache_on_gpu_nodes(
        cache_dir=cache_dir,
        owner_id=consumer_id,
        owner_kind="consumer",
        node_ids=node_ids,
    )


def cleanup_artifact_cache_on_gpu_nodes(
    *, cache_dir: str, run_id: str
) -> dict[str, Any]:
    return _release_artifact_cache_on_gpu_nodes(
        cache_dir=cache_dir,
        owner_id=run_id,
        owner_kind="run",
    )


def cleanup_output_root(output_dir: str | Path) -> list[str]:
    """Remove duplicate final-model files saved at output_dir root.

    Checkpoint directories are kept. LLaMA-Factory/Trainer also saves a full
    model at the output root after training, which is redundant for this engine
    and can double disk usage for full finetune runs.
    """
    root = Path(output_dir)
    removed: list[str] = []
    removable_names = {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    }
    for child in (root.iterdir() if root.exists() else []):
        if child.is_dir():
            continue
        if child.name in removable_names or re.match(r"^(model|pytorch_model)-\d+-of-\d+\.(safetensors|bin)$", child.name):
            try:
                child.unlink()
                removed.append(str(child))
            except OSError:
                pass
    for child in (root.iterdir() if root.exists() else []):
        if child.is_dir() and child.name.startswith(".tmp"):
            shutil.rmtree(child, ignore_errors=True)
            removed.append(str(child))
    return removed


def _materialize_final_sft_checkpoint(
    output_dir: str | Path,
    *,
    final_step: int,
    final_epoch: int,
) -> Path:
    root = Path(output_dir)
    normalized_epoch = root / f"epoch-{final_epoch:03d}"
    if normalized_epoch.is_dir():
        return normalized_epoch
    target = root / f"checkpoint-{final_step}"
    if target.is_dir():
        return target
    model_names = {
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
    candidates = [
        path
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name in model_names
            or re.fullmatch(
                r"(?:model|pytorch_model)-\d+-of-\d+\.(?:safetensors|bin)",
                path.name,
            )
        )
    ]
    if not any(
        path.name in {"adapter_model.bin", "adapter_model.safetensors", "model.safetensors", "pytorch_model.bin"}
        or re.fullmatch(
            r"(?:model|pytorch_model)-\d+-of-\d+\.(?:safetensors|bin)",
            path.name,
        )
        for path in candidates
    ):
        raise RuntimeError(f"SFT final model weights are missing from {root}")
    target.mkdir(parents=False)
    for path in candidates:
        path.replace(target / path.name)
    state_path = root / "trainer_state.json"
    if state_path.is_file():
        state_path.replace(target / state_path.name)
    else:
        write_json(target / "trainer_state.json", {"epoch": final_epoch})
    return target


def build_training_subprocess_env(
    request: dict[str, Any],
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    python_bin = str(Path(sys.executable).parent)
    current_path = env.get("PATH", "")
    env["PATH"] = (
        f"{python_bin}{os.pathsep}{current_path}" if current_path else python_bin
    )
    requested_gpus = int(request.get("train_gpus", 1))
    if requested_gpus < 1:
        raise ValueError("train_gpus must be positive")
    declared = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible = (
        [item.strip() for item in declared.split(",") if item.strip()]
        if declared
        else [str(index) for index in range(requested_gpus)]
    )
    if len(visible) < requested_gpus:
        raise ValueError(
            f"train_gpus={requested_gpus} exceeds available visible GPUs "
            f"({len(visible)})"
        )
    env["CUDA_VISIBLE_DEVICES"] = ",".join(visible[:requested_gpus])
    return env


def _stage_model_locally(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from ade.engine.checkpoints.cache import stage_checkpoint_if_enabled

    staged_checkpoint = stage_checkpoint_if_enabled(
        {
            "checkpoint_path": request["model"],
            "checkpoint_staging": True,
            "checkpoint_cache_dir": request.get("model_cache_dir"),
            "checkpoint_cache_max_gb": request.get("model_cache_max_gb"),
            "checkpoint_cache_lock_stale_seconds": request.get(
                "model_cache_lock_stale_seconds"
            ),
            "ade_run_id": request.get("agent_task_id") or request.get("run_id"),
            "staging_consumer_id": request.get("staging_consumer_id"),
        }
    )
    staged = dict(request)
    staged["source_model_path"] = request["model"]
    staged["model"] = staged_checkpoint["checkpoint_path"]
    staged["model_staged"] = True
    node = {
        "node_ip": socket.gethostbyname(socket.gethostname()),
        "hostname": socket.gethostname(),
        "source": staged_checkpoint.get("source_checkpoint_path", request["model"]),
        "target": staged_checkpoint["checkpoint_path"],
        "cache_hit": bool(staged_checkpoint.get("checkpoint_staging_cache_hit")),
        "elapsed_seconds": float(
            staged_checkpoint.get("checkpoint_staging_elapsed_seconds") or 0.0
        ),
    }
    return staged, {
        "schema_version": "1",
        "status": "complete",
        "source": request["model"],
        "target": staged["model"],
        "nodes": [node],
    }


def _run_train_task(request: dict[str, Any]) -> dict[str, Any]:
    # The Engine entrypoint restores SIGCHLD on its main thread after Ray init.
    # Checkpoint-aware SFT invokes this function from a training thread.
    from ade.tasks.data_selection.llamafactory_prompt_protocol import validate_sft_prompt_contract

    if request.get("model_staging") and not request.get("model_staged"):
        request, staging_receipt = _stage_model_locally(request)
        write_json(
            Path(request["run_dir"]) / "receipts" / "model-staging.json",
            staging_receipt,
        )
    # The staged model path is node-local.  Prompt validation runs in the
    # Engine process before LlamaFactory starts, so validate against the
    # shared source path while retaining the staged path for training workers.
    validation_request = request
    source_model_path = request.get("source_model_path")
    if source_model_path:
        validation_request = dict(request)
        validation_request["model"] = source_model_path
    validate_sft_prompt_contract(validation_request)
    project_root = Path(request["project_root"]).resolve()
    project_root = _add_baseline_path(project_root)

    task_id = request["task_id"]
    output_dir = Path(request["checkpoint_output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(request["run_dir"]).resolve()
    log_dir = run_dir / "logs" / "train"
    log_dir.mkdir(parents=True, exist_ok=True)
    train_stage = safe_log_part(request.get("train_stage") or "train")
    log_prefix = request.get("log_prefix") or f"train__run_{safe_log_part(task_id)}__stage_{train_stage}"
    log_path = log_dir / f"{safe_log_part(log_prefix)}.log"
    stop_file = Path(request["stop_file"]).resolve()

    data_file = Path(request["data_file"]).resolve()
    train_count = read_record_count(data_file)
    per_device_batch_size = int(request["per_device_train_batch_size"])
    grad_accum = int(request["gradient_accumulation_steps"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    world_size = len([part for part in visible.split(",") if part.strip()]) or int(request.get("train_gpus", 8))
    effective_batch_size = per_device_batch_size * grad_accum * world_size
    steps_per_epoch = max(1, math.ceil(train_count / effective_batch_size))

    recipe_path = Path(request["recipe_path"]).resolve()
    base_config = recipe_path.read_text(encoding="utf-8")
    project_root = _add_baseline_path(request.get("project_root", Path.cwd()))
    tracking_settings = request.get("training_telemetry")
    tracking_settings = tracking_settings if isinstance(tracking_settings, dict) else {}
    tracking_env, tracking_audit = build_training_tracking(
        settings=tracking_settings,
        backend="llamafactory",
        run_dir=run_dir,
        run_group=str(request.get("agent_task_id") or task_id),
        coordinator_id=str(request.get("coordinator_id") or "c000"),
        plan_id=str(request.get("plan_id") or "p000"),
        trial_id=str(request.get("trial_id") or task_id),
        logical_command_id=str(request.get("logical_command_id") or task_id),
        attempt_id=str(request.get("attempt_id") or "attempt-001"),
        fork_lineage=(
            request.get("fork_lineage")
            if isinstance(request.get("fork_lineage"), dict)
            else None
        ),
    )
    updates = {
            "model_name_or_path": request["model"],
            "dataset": request["dataset_name"],
            "dataset_dir": request["dataset_dir"],
            "output_dir": str(output_dir),
            "cutoff_len": int(request["cutoff_len"]),
            "num_train_epochs": int(request["num_train_epochs"]),
            "max_steps": int(request["max_steps"]),
            "per_device_train_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "save_steps": _sft_checkpoint_save_steps(
                steps_per_epoch=steps_per_epoch,
                artifact_interval=int(request.get("artifact_interval", 1)),
            ),
            "save_strategy": "steps",
            "eval_strategy": request.get("eval_strategy", "no"),
            "report_to": "wandb" if tracking_audit.get("enabled") else request.get("report_to", "none"),
            "logging_steps": request.get("logging_steps", 1),
            "learning_rate": request.get("learning_rate", 5.0e-6),
            "lr_scheduler_type": request.get("lr_scheduler_type", "cosine"),
            "warmup_ratio": request.get("warmup_ratio", 0.0),
            "seed": int(request["seed"]),
            "bf16": request.get("bf16", True),
            "save_total_limit": (
                None
                if request.get("save_total_limit") is None
                else int(request["save_total_limit"])
            ),
            "run_name": tracking_audit.get("name") or task_id,
            "train_on_prompt": request["train_on_prompt"],
            "mask_history": request["mask_history"],
    }
    if request.get("default_system") is not None:
        updates["default_system"] = str(request["default_system"])
    deepspeed_match = re.search(r"^\s*deepspeed:\s*(\S+)\s*$", base_config, flags=re.MULTILINE)
    if deepspeed_match:
        updates["deepspeed"] = _resolve_llamafactory_path(project_root, deepspeed_match.group(1))
    if request.get("use_ray_training"):
        updates["ray_num_workers"] = int(request.get("train_gpus", 1))
        updates["ray_init_kwargs"] = {
            "address": "auto",
            "ignore_reinit_error": True,
            "namespace": request.get("ray_namespace"),
        }
        updates["master_addr"] = ray.util.get_node_ip_address()
    if request.get("add_special_tokens") is not None:
        updates["add_special_tokens"] = request["add_special_tokens"]
    if request.get("resize_vocab") is not None:
        updates["resize_vocab"] = request["resize_vocab"]
    config_text = rewrite_training_config(base_config, updates)
    temp = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix=f"{train_stage}_{safe_log_part(task_id)}_", delete=False, encoding="utf-8")
    process: subprocess.Popen | None = None
    try:
        temp.write(config_text)
        temp.close()
        env = os.environ.copy()
        env.update(tracking_env)
        # A prior W&B publication in this long-lived Engine process may leave
        # WANDB_SERVICE pointing at its exited local service socket.  Never
        # propagate that process-local token into a distributed Ray training
        # driver; the training workers must establish their own service.
        env.pop("WANDB_SERVICE", None)
        env.update(
            {
                "ADE_RUN_ID": str(request.get("agent_task_id") or ""),
                "ADE_COORDINATOR_ID": str(request.get("coordinator_id") or ""),
                "ADE_PLAN_ID": str(request.get("plan_id") or ""),
                "ADE_TRIAL_ID": str(request.get("trial_id") or ""),
                "ADE_ENGINE_COMMAND_ID": str(request.get("task_id") or ""),
                "ADE_WORKLOAD": "training",
            }
        )
        if tracking_audit.get("enabled"):
            Path(str(tracking_audit["local_run_dir"])).mkdir(parents=True, exist_ok=True)
        if request.get("use_ray_training"):
            env["USE_RAY"] = "1"
            env["LLAMAFACTORY_DEVICE_NAME"] = "gpu"
            env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            env["ADE_GPU_LEASE_PLACEMENT_GROUP_ID"] = str(
                request["gpu_lease_placement_group_id"]
            )
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env.pop("FORCE_TORCHRUN", None)
        else:
            env = build_training_subprocess_env(request, env)
            env["FORCE_TORCHRUN"] = "1"
        lf_dir = project_root / "third_party/llamafactory"
        env["PYTHONPATH"] = f"{lf_dir / 'src'}:{project_root}:{env.get('PYTHONPATH', '')}"
        start = time.time()
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                _llamafactory_train_command(temp.name, project_root),
                cwd=lf_dir,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                preexec_fn=_prepare_training_child if hasattr(os, "setsid") else None,
            )
        stopped = False
        while process.poll() is None:
            if stop_file.exists():
                stopped = True
                kill_process_group(process)
                break
            time.sleep(2)
        exit_code = process.wait()
        analysis_root = run_dir / "analysis_sources"
        total_training_steps = steps_per_epoch * int(request["num_train_epochs"])
        telemetry_metrics: dict[str, Any] = {
            "training_telemetry_available": False,
            "training_telemetry_reason": "training_tracking_disabled",
        }
        audit_payload = tracking_audit.get("audit_ref")
        if isinstance(audit_payload, dict):
            metric_profile = MetricProfile(
                profile_id=LLAMAFACTORY_METRIC_PROFILE.profile_id,
                exact_names=LLAMAFACTORY_METRIC_PROFILE.exact_names,
                anchored_patterns=LLAMAFACTORY_METRIC_PROFILE.anchored_patterns,
                expected_steps=tuple(range(1, total_training_steps + 1)),
                required=total_training_steps > 0,
            )
            telemetry_spec = TelemetryExportSpec(
                audit_ref=audit_ref_from_dict(audit_payload),
                metric_profile=metric_profile,
                telemetry_path=analysis_root / "training_telemetry.jsonl",
                manifest_path=analysis_root / "training_telemetry_manifest.json",
                supplemental_history_path=log_path,
            )
            telemetry = export_exact_wandb_telemetry(telemetry_spec)
            telemetry_metrics.update(
                {
                    "training_telemetry_manifest_path": str(telemetry_spec.manifest_path),
                    "training_telemetry_available": telemetry.get("status") == "complete",
                }
            )
            if telemetry_spec.telemetry_path.exists():
                telemetry_metrics["training_telemetry_path"] = str(telemetry_spec.telemetry_path)
            telemetry_summary_path = analysis_root / "training_telemetry_summary.json"
            if telemetry_summary_path.exists():
                telemetry_metrics["training_telemetry_summary_path"] = str(telemetry_summary_path)
        if exit_code != 0 and not stopped:
            raise RuntimeError(f"training failed with exit code {exit_code}; see {log_path}")
        final_checkpoint = None
        if not stopped:
            final_checkpoint = _materialize_final_sft_checkpoint(
                output_dir,
                final_step=steps_per_epoch * int(request["num_train_epochs"]),
                final_epoch=int(request["num_train_epochs"]),
            )
        removed_root_files = cleanup_output_root(output_dir)
        return {
            "status": "stopped" if stopped else "completed",
            "task_id": task_id,
            "exit_code": exit_code,
            "train_count": train_count,
            "effective_batch_size": effective_batch_size,
            "steps_per_epoch": steps_per_epoch,
            "checkpoint_output_dir": str(output_dir),
            "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
            "log_path": str(log_path),
            "removed_root_files": removed_root_files,
            "training_run_audit_ref": tracking_audit.get("audit_ref"),
            **telemetry_metrics,
            "elapsed_seconds": round(time.time() - start, 2),
        }
    finally:
        if process is not None:
            kill_process_group(process)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                kill_process_group(process)
        kill_descendant_processes(os.getpid())
        try:
            os.unlink(temp.name)
        except OSError:
            pass


def run_train_task(request: dict[str, Any]) -> dict[str, Any]:
    request = dict(request)
    request["staging_consumer_id"] = str(
        request.get("staging_consumer_id") or request.get("task_id") or ""
    )
    requested_gpus = int(request.get("train_gpus", 1))
    coordinator_owner, workload = validate_coordinator_workload_request(
        request, requested_gpus=requested_gpus
    )
    if coordinator_owner == "standalone":
        staged = False
        try:
            if request.get("model_staging"):
                request, staging_receipt = _stage_model_locally(request)
                staged = True
                write_json(
                    Path(request["run_dir"]) / "receipts" / "model-staging.json",
                    staging_receipt,
                )
            return _run_train_task(request)
        finally:
            if staged:
                from ade.engine.checkpoints.cache import release_staging_for_consumer

                release_staging_for_consumer(
                    str(request["model_cache_dir"]),
                    str(request["staging_consumer_id"]),
                )
    lease_owner = f"{coordinator_owner}/{workload}"
    release_gpu_leases_for_owner(lease_owner)
    lease = acquire_gpu_lease(
        owner=lease_owner,
        kind="train",
        requested_gpus=requested_gpus,
        minimum_gpus=requested_gpus,
        distributed_train=True,
        coordinator_owner=coordinator_owner,
        workload=workload,
        coordinator_policy=dict(request["coordinator_resource_policy"]),
    )
    request["gpu_lease_placement_group"] = lease["placement_group_name"]
    request["gpu_lease_placement_group_id"] = lease["placement_group"].id.hex()
    request["use_ray_training"] = True
    staged_node_ids: tuple[str, ...] = ()
    try:
        if request.get("model_staging"):
            staged_node_ids = placement_group_node_ids(lease["placement_group"])
            request, staging_receipt = stage_model_on_gpu_nodes(
                request,
                node_ids=staged_node_ids,
            )
            write_json(
                Path(request["run_dir"]) / "receipts" / "model-staging.json",
                staging_receipt,
            )
        return _run_train_task(request)
    finally:
        try:
            if staged_node_ids:
                release_artifact_cache_consumer_on_gpu_nodes(
                    cache_dir=str(request["model_cache_dir"]),
                    consumer_id=str(request["staging_consumer_id"]),
                    node_ids=staged_node_ids,
                )
        finally:
            release_gpu_lease(lease)


train_task = ray.remote(run_train_task)


def _build_eval_result_filename(epoch: int | None, timestamp: str, avg_k: int, suffix: str | None) -> str:
    if suffix:
        return f"eval_result__{bounded_log_part(suffix)}__avgk_{avg_k}__ts_{timestamp}.json"
    return f"eval_result__epoch_{safe_log_part(epoch)}__avgk_{avg_k}__ts_{timestamp}.json"


def _build_eval_shard_crash_filename(shard_suffix: str) -> str:
    return f"eval_shard_crash__{bounded_log_part(shard_suffix)}.json"


def _combine_eval_shards(request: dict[str, Any], shard_payloads: list[dict[str, Any]], elapsed_seconds: float) -> list[dict[str, Any]]:
    eval_logs_dir = Path(request["run_dir"]) / "logs" / "eval"
    eval_logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    avg_k = int(request["avg_k"])
    combined_results: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = {}

    for payload in shard_payloads:
        for item in payload.get("results", []):
            by_dataset.setdefault(item["dataset_name"], []).append(item)

    for dataset in request["datasets"]:
        dataset_name = dataset["name"]
        items = sorted(
            by_dataset.get(dataset_name, []),
            key=lambda item: (item.get("result_details") or [{}])[0].get("index", 0),
        )
        if not items:
            raise RuntimeError(f"missing shard results for dataset {dataset_name}")

        details: list[dict[str, Any]] = []
        generation_seconds = 0.0
        tokenizer_load_seconds = 0.0
        llm_init_seconds = 0.0
        shard_result_paths: list[str] = []
        for item in items:
            item_details = item.get("result_details") or []
            details.extend(item_details)
            timings = item.get("timings", {})
            generation_seconds += float(timings.get("generation_seconds") or 0.0)
            tokenizer_load_seconds += float(timings.get("tokenizer_load_seconds") or 0.0)
            llm_init_seconds += float(timings.get("llm_init_seconds") or 0.0)
            if item.get("result_path"):
                shard_result_paths.append(str(item["result_path"]))

        has_row_shards = any(
            int((item.get("row_shard") or {}).get("count") or 1) > 1
            for item in items
        )
        if has_row_shards:
            details = merge_data_parallel_details(details)
        task_type = str(dataset.get("task_type") or request.get("task_type") or items[0].get("task_type") or task_type_from_config(dataset, request))
        primary_metric = resolve_primary_metric(dataset, request, items)
        merged_metrics = summarize_details(details, primary_metric=primary_metric, configured_k=avg_k)
        average_accuracy = selected_score(merged_metrics, primary_metric)
        std_deviation = float(merged_metrics.get("accuracy_std", 0.0) or 0.0)
        evaluated_run_indices = [detail.get("index") for detail in details]
        rollout_layout = str(request.get("rollout_layout") or "question_n")
        seed_metadata: dict[str, Any] = {}
        if rollout_layout == "question_n":
            for detail in details:
                detail.pop("seed", None)
            evaluated_seeds = None
            seed_metadata = {
                "seed_strategy": "question_n",
                "seed_base": request.get("seed"),
                "seed_avg_k": avg_k,
            }
        else:
            evaluated_seeds = [
                detail.get("seed")
                for detail in details
                if detail.get("seed") is not None
            ]
            if evaluated_seeds and len(evaluated_seeds) != len(set(evaluated_seeds)):
                raise RuntimeError(f"duplicate eval seeds detected while combining {dataset_name}: {evaluated_seeds}")
        suffix = request.get("result_suffix")
        if len(request["datasets"]) != 1:
            suffix = f"{suffix}__result_dataset_{safe_log_part(dataset_name)}" if suffix else f"result_dataset_{safe_log_part(dataset_name)}"
        result_path = eval_logs_dir / _build_eval_result_filename(request.get("epoch"), timestamp, avg_k, suffix)
        base_payload = {
            "model_path": request["checkpoint_path"],
            "dataset_name": dataset_name,
            "avg_k_runs": avg_k,
            "evaluated_run_indices": evaluated_run_indices,
            "evaluated_seeds": evaluated_seeds or None,
            "seed": request.get("seed"),
            "score": average_accuracy,
            "avg_at_k_score": average_accuracy,
            "average_accuracy": average_accuracy,
            "score_standard_deviation": std_deviation,
            "standard_deviation": std_deviation,
            "metrics": merged_metrics,
            "task_type": task_type,
            "primary_metric": primary_metric,
            "answer_format": request.get("answer_format", "math"),
            "epoch": request.get("epoch"),
            "timings": {
                "tokenizer_load_seconds": tokenizer_load_seconds,
                "llm_init_seconds": llm_init_seconds,
                "generation_seconds": generation_seconds,
                "parallel_wall_seconds": elapsed_seconds,
            },
            "parallel_eval": {
                "enabled": True,
                "shard_count": len(items),
                "requested_gpus": int(request.get("data_parallel_shards") or 1),
                "allocated_gpus": int((request.get("_gpu_lease") or {}).get("allocated_gpus") or 1),
                "data_parallel_shards": min(
                    int(request.get("data_parallel_shards") or 1),
                    int((request.get("_gpu_lease") or {}).get("allocated_gpus") or 1),
                ),
                "gpus_per_shard": int(request.get("gpus_per_shard") or request.get("data_parallel_shards") or 1),
                "shard_result_paths": shard_result_paths,
                "gpu_lease": lease_metadata(request["_gpu_lease"]),
            },
        }
        base_payload.update(seed_metadata)
        if request.get("metadata"):
            base_payload.update(request["metadata"])
        result_payload = build_eval_log_payload(
            base_payload=base_payload,
            dataset_path=dataset["path"],
            details=details,
        )
        result_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        _cleanup_shard_eval_logs(eval_logs_dir, shard_result_paths)
        combined_results.append({
            "dataset_name": dataset_name,
            "score": average_accuracy,
            "avg_at_k_score": average_accuracy,
            "metrics": merged_metrics,
            "task_type": task_type,
            "primary_metric": primary_metric,
            "score_standard_deviation": std_deviation,
            "result_path": str(result_path),
            "timings": result_payload["timings"],
        })
    return combined_results


def _cleanup_shard_eval_logs(eval_logs_dir: Path, paths: list[str]) -> None:
    root = eval_logs_dir.resolve()
    for value in sorted(set(paths)):
        path = Path(value)
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved.parent != root:
            continue
        if "__shard_" not in resolved.name:
            continue
        if resolved.suffix != ".json":
            continue
        try:
            resolved.unlink()
        except FileNotFoundError:
            continue


def _merge_data_parallel_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return merge_data_parallel_details(details)


def _split_run_indices(avg_k: int, max_shards: int) -> list[list[int]]:
    shard_count = max(1, min(avg_k, max_shards))
    chunks: list[list[int]] = [[] for _ in range(shard_count)]
    for offset, run_index in enumerate(range(avg_k)):
        chunks[offset % shard_count].append(run_index)
    return [chunk for chunk in chunks if chunk]


def _eval_row_shard_count(request: dict[str, Any], data_parallel_shards: int) -> int:
    requested = max(1, int(data_parallel_shards or 1))
    row_counts = []
    for dataset in request.get("datasets") or []:
        task_type = task_type_from_config(dataset, request)
        rows = get_task(task_type).load_examples(dataset["path"], dataset_name=dataset.get("name"), dataset_config=dataset)
        row_counts.append(len(rows))
    max_rows = max(row_counts, default=0)
    if max_rows <= 0:
        return 1
    return max(1, min(requested, max_rows))


def _eval_shard_specs(avg_k: int, data_parallel_shards: int, row_shard_count: int | None = None) -> list[dict[str, Any]]:
    run_indices = list(range(max(1, avg_k)))
    row_shard_count = max(1, int(row_shard_count if row_shard_count is not None else data_parallel_shards or 1))
    specs = []
    for row_shard_index in range(row_shard_count):
        specs.append(
            {
                "run_indices": run_indices,
                "row_shard_index": row_shard_index,
                "row_shard_count": row_shard_count,
                "data_parallel_size": 1,
            }
        )
    return specs


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    size = max(1, size)
    return [items[index:index + size] for index in range(0, len(items), size)]


def _ray_max_node_gpus() -> int:
    max_gpus = 0
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        resources = node.get("Resources") or {}
        max_gpus = max(max_gpus, int(resources.get("GPU") or 0))
    return max_gpus


def _ensure_eval_cuda_env() -> None:
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return
    try:
        gpu_ids = [str(gpu_id) for gpu_id in ray.get_gpu_ids()]
    except Exception:
        gpu_ids = []
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)


@ray.remote(max_calls=1)
def eval_shard_task(request: dict[str, Any], shard_spec: dict[str, Any]) -> dict[str, Any]:
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    _ensure_eval_cuda_env()
    for source_name, target_name in (
        ("ade_run_id", "ADE_RUN_ID"),
        ("ade_coordinator_id", "ADE_COORDINATOR_ID"),
        ("ade_plan_id", "ADE_PLAN_ID"),
        ("ade_trial_id", "ADE_TRIAL_ID"),
        ("ade_engine_command_id", "ADE_ENGINE_COMMAND_ID"),
        ("ade_workload", "ADE_WORKLOAD"),
    ):
        if request.get(source_name):
            os.environ[target_name] = str(request[source_name])
    project_root = Path(request["project_root"]).resolve()
    project_root = _add_baseline_path(project_root)
    from ade.engine.checkpoints.cache import stage_checkpoint_if_enabled
    from ade.engine.eval.runner import run_evaluation_many

    if request.get("system_prompt_path") is not None or request.get("system_prompt") is not None:
        raise ValueError(
            "evaluation system prompt must be resolved per dataset; path/global prompt authority is forbidden"
        )
    shard_request = dict(request)
    shard_request["avg_k"] = int(request["avg_k"])
    run_indices = list(shard_spec["run_indices"])
    row_shard_index = int(shard_spec.get("row_shard_index", 0))
    row_shard_count = int(shard_spec.get("row_shard_count", 1))
    data_parallel_size = max(1, int(shard_spec.get("data_parallel_size") or request.get("data_parallel_shards") or 1))
    run_label = f"{run_indices[0]:02d}" if len(run_indices) == 1 else f"{run_indices[0]:02d}_{run_indices[-1]:02d}"
    shard_label = f"run_{run_label}__rows_{row_shard_index:02d}_of_{row_shard_count:02d}"
    shard_suffix = (
        f"{safe_log_part(request.get('result_suffix'))}__shard_{shard_label}"
        if request.get("result_suffix")
        else f"shard_{shard_label}"
    )
    try:
        if shard_request.get("checkpoint_staging"):
            shard_request = stage_checkpoint_if_enabled(shard_request)
        results = run_evaluation_many(
            model_path=shard_request["checkpoint_path"],
            datasets=request["datasets"],
            epoch=request.get("epoch"),
            avg_k=int(request["avg_k"]),
            run_indices=run_indices,
            seed=request.get("seed"),
            result_suffix=shard_suffix,
            extra_result_metadata=request.get("metadata"),
            project_root_dir=request["run_dir"],
            temperature=float(request.get("temperature", 0.6)),
            top_p=float(request.get("top_p", 1.0)),
            repetition_penalty=float(request.get("repetition_penalty", 1.0)),
            answer_format=request.get("answer_format", "math"),
            task_type=request.get("task_type"),
            primary_metric=request.get("primary_metric"),
            dtype=request.get("dtype", "bfloat16"),
            max_tokens=int(request.get("max_model_len") or request.get("max_new_tokens")),
            max_model_len=request.get("max_model_len"),
            max_new_tokens=request.get("max_new_tokens"),
            attention_backend=request.get("attention_backend"),
            gpu_memory_utilization=float(request.get("gpu_memory_utilization", 0.9)),
            max_num_seqs=request.get("max_num_seqs"),
            swap_space=request.get("swap_space"),
            cpu_offload_gb=request.get("cpu_offload_gb"),
            enforce_eager=request.get("enforce_eager"),
            max_seq_len_to_capture=request.get("max_seq_len_to_capture"),
            kv_cache_dtype=request.get("kv_cache_dtype"),
            calculate_kv_scales=request.get("calculate_kv_scales"),
            enable_prefix_caching=request.get("enable_prefix_caching"),
            vllm_use_v1=request.get("vllm_use_v1"),
            tokenizer_path=request.get("tokenizer_path"),
            row_shard_index=row_shard_index,
            row_shard_count=row_shard_count,
            data_parallel_size=data_parallel_size,
            thinking_budget=request.get("thinking_budget", -1),
            reasoning_parser=request.get("reasoning_parser", ""),
            model_protocol=request.get("model_protocol"),
            rollout_layout=request.get("rollout_layout", "question_n"),
        )
    except Exception as exc:
        crash_dir = Path(request["run_dir"]) / "logs" / "eval"
        crash_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            crash_dir / _build_eval_shard_crash_filename(shard_suffix),
            {
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "shard_spec": shard_spec,
                "checkpoint_path": request.get("checkpoint_path"),
                "checkpoint_staging": request.get("checkpoint_staging"),
                "env": {
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "VLLM_TARGET_DEVICE": os.environ.get("VLLM_TARGET_DEVICE"),
                },
                "worker": {
                    "node_ip": ray.util.get_node_ip_address(),
                    "gpu_ids": ray.get_gpu_ids(),
                },
            },
        )
        raise
    return {
        "status": "completed",
        "phase": request.get("phase"),
        "checkpoint_path": request["checkpoint_path"],
        "shard_checkpoint_path": shard_request["checkpoint_path"],
        "epoch": request.get("epoch"),
        "run_indices": run_indices,
        "row_shard": {"index": row_shard_index, "count": row_shard_count},
        "data_parallel_size": data_parallel_size,
        "results": results,
        "worker": {
            "node_ip": ray.util.get_node_ip_address(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def _run_eval_shards(
    request: dict[str, Any],
    shard_specs: list[dict[str, Any]],
    lease: dict[str, Any],
    *,
    max_inflight_shards: int,
    allocated_gpus: int,
    gpus_per_shard: int,
    max_retries: int,
) -> tuple[list[dict[str, Any]], int]:
    shard_payloads = []
    pending_specs = [(spec, 0) for spec in shard_specs]
    running: dict[Any, tuple[dict[str, Any], int, int]] = {}
    available_bundle_indices = list(range(allocated_gpus))
    retry_count = 0

    try:
        while pending_specs or running:
            _raise_if_eval_cancelled(request)
            while pending_specs and len(running) < max_inflight_shards and available_bundle_indices:
                _raise_if_eval_cancelled(request)
                spec, retries = pending_specs.pop(0)
                bundle_index = available_bundle_indices.pop(0)
                strategy = PlacementGroupSchedulingStrategy(
                    placement_group=lease["placement_group"],
                    placement_group_bundle_index=bundle_index,
                    placement_group_capture_child_tasks=True,
                )
                future = eval_shard_task.options(
                    num_cpus=1,
                    num_gpus=gpus_per_shard,
                    scheduling_strategy=strategy,
                ).remote(request, spec)
                running[future] = (
                    spec,
                    bundle_index,
                    retries,
                )

            if not running:
                raise RuntimeError("evaluation lease has no schedulable GPU bundle")

            ready, _ = ray.wait(
                list(running),
                timeout=5.0,
                num_returns=1,
            )
            _raise_if_eval_cancelled(request)
            for future in ready:
                spec, bundle_index, retries = running.pop(future)
                available_bundle_indices.append(bundle_index)
                available_bundle_indices.sort()
                try:
                    shard_payloads.append(ray.get(future))
                except Exception as exc:
                    if retries < max_retries:
                        pending_specs.append((spec, retries + 1))
                        retry_count += 1
                        continue
                    raise RuntimeError(
                        f"eval shard failed after {retries + 1} attempts; spec={spec!r}"
                    ) from exc
    except Exception:
        for future in running:
            ray.cancel(future, force=True)
        for future in tuple(running):
            try:
                ray.get(future)
            except Exception:
                pass
        raise
    return shard_payloads, retry_count


def _raise_if_eval_cancelled(request: dict[str, Any]) -> None:
    cancellation_file = request.get("cancellation_file")
    if cancellation_file and Path(str(cancellation_file)).is_file():
        raise EngineAttemptError(
            "online evaluation cancelled by SFT early stopping",
            failure_kind="evaluation_cancelled",
            retryable=False,
        )


def _run_eval_task_single(request: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(request["project_root"]).resolve()
    project_root = _add_baseline_path(project_root)

    request = dict(request)
    start = time.time()
    avg_k = int(request["avg_k"])
    data_parallel_shards = int(request.get("data_parallel_shards") or 1)
    lease = request.get("_gpu_lease")
    if not isinstance(lease, dict) or lease.get("placement_group") is None:
        raise RuntimeError("evaluation requires a task-level GPU lease")
    allocated_gpus = int(lease["allocated_gpus"])
    effective_data_parallel_shards = min(data_parallel_shards, allocated_gpus)
    request["rollout_layout"] = str(request.get("rollout_layout") or "question_n")
    if request.get("max_num_seqs") is not None:
        request["max_num_seqs"] = max(int(request["max_num_seqs"]), avg_k)
    else:
        request["max_num_seqs"] = avg_k
    row_shard_count = _eval_row_shard_count(request, effective_data_parallel_shards)
    shard_specs = _eval_shard_specs(avg_k, effective_data_parallel_shards, row_shard_count=row_shard_count)
    gpus_per_shard = 1
    max_node_gpus = _ray_max_node_gpus()
    if max_node_gpus and gpus_per_shard > max_node_gpus:
        raise RuntimeError(
            f"eval data_parallel_shards={gpus_per_shard} exceeds max GPUs on one Ray node ({max_node_gpus})"
        )
    request["gpus_per_shard"] = gpus_per_shard
    max_inflight_runs = int(
        request.get("max_inflight_shards")
        or request.get("eval_concurrent_shards")
        or avg_k
    )
    max_inflight_shards = max(
        1,
        min(
            max_inflight_runs * max(1, data_parallel_shards),
            allocated_gpus,
            len(shard_specs),
        ),
    )
    max_shard_retries = int(request.get("eval_shard_max_retries", 1))
    shard_payloads, shard_retry_count = _run_eval_shards(
        request,
        shard_specs,
        lease,
        max_inflight_shards=max_inflight_shards,
        allocated_gpus=allocated_gpus,
        gpus_per_shard=gpus_per_shard,
        max_retries=max_shard_retries,
    )
    results = _combine_eval_shards(request, shard_payloads, time.time() - start)
    output = {
        "status": "completed",
        "phase": request.get("phase"),
        "checkpoint_path": request["checkpoint_path"],
        "epoch": request.get("epoch"),
        "results": results,
        "parallel_eval": {
            "enabled": True,
            "shard_count": len(shard_specs),
            "max_inflight_runs": max_inflight_runs,
            "max_inflight_shards": max_inflight_shards,
            "avg_k": avg_k,
            "requested_gpus": data_parallel_shards,
            "allocated_gpus": allocated_gpus,
            "data_parallel_shards": effective_data_parallel_shards,
            "gpus_per_shard": gpus_per_shard,
            "shard_max_retries": max_shard_retries,
            "shard_retry_count": shard_retry_count,
            "rollout_layout": request["rollout_layout"],
            "gpu_lease": lease_metadata(lease),
        },
        "shards": [
            {
                "run_indices": payload.get("run_indices"),
                "row_shard": payload.get("row_shard"),
                "worker": payload.get("worker"),
            }
            for payload in shard_payloads
        ],
        "elapsed_seconds": round(time.time() - start, 2),
    }
    _add_validation_ranking(output, request)
    return output


def run_eval_task(request: dict[str, Any]) -> dict[str, Any]:
    datasets = request.get("datasets") or []
    profiled = [
        dataset
        for dataset in datasets
        if isinstance(dataset, dict) and isinstance(dataset.get("eval_profile"), dict)
    ]
    if not profiled:
        return _run_eval_task_single(request)
    if len(profiled) != len(datasets):
        raise ValueError("evaluation request cannot mix profiled and unprofiled datasets")

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for dataset in profiled:
        profile = dataset["eval_profile"]
        num_samples = int(profile.get("num_samples") or 0)
        shards = int(profile.get("data_parallel_shards") or 0)
        if num_samples <= 0 or shards <= 0:
            raise ValueError(f"invalid eval_profile for dataset {dataset.get('name')}")
        grouped.setdefault((num_samples, shards), []).append(
            {key: value for key, value in dataset.items() if key != "eval_profile"}
        )

    started = time.time()
    group_results: list[dict[str, Any]] = []
    for (num_samples, shards), group_datasets in sorted(grouped.items()):
        group_request = {
            **request,
            "datasets": group_datasets,
            "avg_k": num_samples,
            "data_parallel_shards": shards,
        }
        suffix = request.get("result_suffix")
        profile_suffix = f"profile_samples_{num_samples}_shards_{shards}"
        group_request["result_suffix"] = (
            f"{suffix}__{profile_suffix}" if suffix else profile_suffix
        )
        group_results.append(_run_eval_task_single(group_request))

    results = [
        item
        for group_result in group_results
        for item in group_result.get("results", [])
    ]
    output = {
        "status": "completed",
        "phase": request.get("phase"),
        "checkpoint_path": request["checkpoint_path"],
        "epoch": request.get("epoch"),
        "results": results,
        "parallel_eval": {
            "enabled": True,
            "profiled": True,
            "groups": [result.get("parallel_eval") for result in group_results],
        },
        "shards": [
            shard
            for group_result in group_results
            for shard in group_result.get("shards", [])
        ],
        "elapsed_seconds": round(time.time() - started, 2),
    }
    _add_validation_ranking(output, request)
    return output


def _add_validation_ranking(
    output: dict[str, Any], request: dict[str, Any]
) -> None:
    ranking_config = request.get("validation_ranking")
    if isinstance(ranking_config, dict):
        weights = ranking_config.get("weights")
        if not isinstance(weights, dict):
            raise ValueError("validation_ranking.weights is required")
        results = output.get("results")
        if not isinstance(results, list):
            raise ValueError("validation result datasets are unavailable")
        metrics_by_dataset = {
            str(item.get("dataset_name")): dict(item.get("metrics") or {})
            for item in results
            if str(item.get("dataset_name")) in weights
        }
        aggregate = aggregate_validation_metrics(metrics_by_dataset, ranking_config)
        output["aggregate_metrics"] = aggregate
        output["score"] = aggregate["ranking_score"]
        output["secondary_score"] = aggregate["secondary_score"]


def _eval_requested_gpus(request: dict[str, Any]) -> int:
    requested = max(1, int(request.get("data_parallel_shards") or 1))
    for dataset in request.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        profile = dataset.get("eval_profile")
        if isinstance(profile, dict):
            requested = max(requested, int(profile.get("data_parallel_shards") or 1))
    return requested


def run_eval_with_gpu_lease(request: dict[str, Any]) -> dict[str, Any]:
    request = dict(request)
    consumer_id = str(request.get("staging_consumer_id") or "").strip()
    if request.get("checkpoint_staging") and not consumer_id:
        raise ValueError(
            "staging_consumer_id is required for checkpoint-staged evaluation"
        )
    requested_gpus = _eval_requested_gpus(request)
    coordinator_owner, workload = validate_coordinator_workload_request(
        request, requested_gpus=requested_gpus
    )
    owner = str(
        f"{coordinator_owner}/{workload}/"
        + str(
            request.get("gpu_lease_owner_suffix")
            or request.get("result_suffix")
            or request.get("phase")
            or request.get("checkpoint_path")
            or f"eval-{uuid.uuid4().hex}"
        )
    )
    release_gpu_leases_for_owner(owner)
    lease = acquire_gpu_lease(
        owner=owner,
        kind=str(request.get("phase") or "evaluation"),
        requested_gpus=requested_gpus,
        minimum_gpus=requested_gpus,
        coordinator_owner=(
            coordinator_owner if coordinator_owner != "standalone" else None
        ),
        workload=workload if workload != "standalone" else None,
        coordinator_policy=(
            dict(request["coordinator_resource_policy"])
            if coordinator_owner != "standalone"
            else None
        ),
        timeout_seconds=float(
            request.get("evaluation_lease_timeout_seconds", 1800.0)
        ),
    )
    request["_gpu_lease"] = lease
    try:
        return run_eval_task(request)
    finally:
        try:
            if request.get("checkpoint_staging"):
                release_artifact_cache_consumer_on_gpu_nodes(
                    cache_dir=str(request["checkpoint_cache_dir"]),
                    consumer_id=consumer_id,
                    node_ids=placement_group_node_ids(lease["placement_group"]),
                )
        finally:
            release_gpu_lease(lease)


@ray.remote(num_gpus=0)
def eval_task(request: dict[str, Any]) -> dict[str, Any]:
    return run_eval_with_gpu_lease(request)
