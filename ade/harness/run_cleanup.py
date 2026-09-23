"""Run-scoped terminal cleanup.

This module deliberately operates on Run-owned identities only.  Shared Ray
services and deployment workers are never selected by this cleanup path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from typing import Any, Callable, Iterable

from ade.engine.command_queue import FileCommandQueue
from ade.engine.storage.atomic import write_json_atomic
from ade.review_labor.command_queue import FileReviewCommandQueue
from ade.core.dotenv import read_dotenv_value


def cleanup_cancelled_coordinator(
    *,
    state,
    coordinator_id: str,
    repository,
    engine_queue: FileCommandQueue,
    review_queue: FileReviewCommandQueue | None,
    calls,
    cancel_provider_job: Callable[[str], object] | None = None,
    release_resources: Callable[[str, str], tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    grace_seconds: float = 10.0,
) -> dict[str, Any]:
    """Fence and stop external work owned by one Search Coordinator."""

    if not coordinator_id.strip():
        raise ValueError("coordinator_id is required for scoped cleanup")
    run_id = state.run_id
    run_dir = repository.layout.run_dir(run_id)
    receipt_path = (
        run_dir / "reports" / "cleanup" / "coordinators" / f"{coordinator_id}.json"
    )
    try:
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        previous = None
    if isinstance(previous, dict) and previous.get("status") == "complete":
        return previous
    resources = state.run_resources if isinstance(state.run_resources, dict) else {}
    judge = resources.get("local_judge")
    if cancel_provider_job is None and isinstance(judge, dict):
        cancel_provider_job = _provider_job_canceller(judge)
    ray_cluster = resources.get("ray_cluster")
    if release_resources is None and isinstance(ray_cluster, dict):
        ray_address = ray_cluster.get("address")
        if isinstance(ray_address, str) and ray_address:
            release_resources = lambda target_run, target_coordinator: (
                _release_coordinator_resources(
                    target_run,
                    target_coordinator,
                    ray_address=ray_address,
                )
            )
    terminalized_engine: list[str] = []
    terminalized_review: list[str] = []
    terminated_calls: list[str] = []
    cancelled_jobs: list[str] = []
    released_leases: list[str] = []
    released_external: list[str] = []
    unresolved: list[str] = []

    target_engine_refs = tuple(
        item
        for item in state.active_engine_commands
        if item.coordinator_id == coordinator_id
    )
    for command_ref in target_engine_refs:
        try:
            if not engine_queue.has_receipt(command_ref.command_id):
                try:
                    command = repository.load_engine_command(
                        run_id,
                        command_ref.coordinator_id,
                        command_ref.plan_id,
                        command_ref.trial_id,
                        command_ref.command_id,
                    )
                    engine_queue.submit(command)
                except FileNotFoundError:
                    pass
                engine_queue.interrupt(
                    command_ref.command_id,
                    failure_kind="coordinator_cancelled",
                    reason=f"Coordinator {coordinator_id} cancellation requested",
                    retryable=False,
                )
            terminalized_engine.append(command_ref.command_id)
        except (OSError, TypeError, ValueError) as error:
            unresolved.append(f"engine:{command_ref.command_id}:{type(error).__name__}:{error}")
    for command_id in engine_queue.command_ids(
        run_id=run_id, coordinator_id=coordinator_id
    ):
        try:
            engine_queue.interrupt(
                command_id,
                failure_kind="coordinator_cancelled",
                reason=f"Coordinator {coordinator_id} cancellation requested",
                retryable=False,
            )
            terminalized_engine.append(command_id)
        except (OSError, TypeError, ValueError) as error:
            unresolved.append(f"engine:{command_id}:{type(error).__name__}:{error}")

    if review_queue is not None:
        target_review_refs = tuple(
            item
            for item in state.active_review_commands
            if item.coordinator_id == coordinator_id
        )
        review_ids = {
            *(item.command_id for item in target_review_refs),
            *review_queue.command_ids(run_id=run_id, coordinator_id=coordinator_id),
        }
        for command_id in sorted(review_ids):
            progress_path = review_queue.progress / f"{command_id}.json"
            if progress_path.is_file():
                try:
                    job_id = review_queue.load_progress(command_id).get("judge_job_id")
                    if isinstance(job_id, str) and job_id:
                        if cancel_provider_job is None:
                            unresolved.append(f"provider_job:{job_id}:cancel_unavailable")
                        else:
                            cancel_provider_job(job_id)
                            cancelled_jobs.append(job_id)
                except (OSError, TypeError, ValueError) as error:
                    unresolved.append(
                        f"provider_job:{command_id}:{type(error).__name__}:{error}"
                    )
            try:
                if not review_queue.has_receipt(command_id):
                    try:
                        command_ref = next(
                            item for item in target_review_refs if item.command_id == command_id
                        )
                        review_queue.submit(
                            repository.load_review_command(
                                run_id,
                                command_ref.coordinator_id,
                                command_ref.plan_id,
                                command_ref.trial_id,
                                command_ref.command_id,
                            )
                        )
                    except (FileNotFoundError, StopIteration):
                        pass
                    review_queue.interrupt(command_id)
                terminalized_review.append(command_id)
            except (OSError, TypeError, ValueError) as error:
                unresolved.append(f"review:{command_id}:{type(error).__name__}:{error}")

    for active in state.active_agent_calls:
        if active.coordinator_id != coordinator_id:
            continue
        if (
            active.role == "run_summarizer"
            and active.target_subject_ref in state.rm_merge_queue
        ):
            continue
        try:
            audit = calls.interrupt(active)
            if audit.get("unresolved"):
                unresolved.append(f"agent:{active.call_id}:{audit['unresolved']}")
            else:
                terminated_calls.append(active.call_id)
        except (OSError, TypeError, ValueError) as error:
            unresolved.append(f"agent:{active.call_id}:{type(error).__name__}:{error}")

    if release_resources is not None:
        try:
            leases, external = release_resources(run_id, coordinator_id)
            released_leases.extend(leases)
            released_external.extend(external)
        except Exception as error:
            unresolved.append(f"resources:{type(error).__name__}:{error}")

    expected_processes, process_pids = _coordinator_worker_processes(
        run_dir / "reports" / "worker-provenance.json", coordinator_id
    )
    write_json_atomic(
        receipt_path,
        {
            "schema_version": "ade.coordinator_cleanup.v1",
            "run_id": run_id,
            "coordinator_id": coordinator_id,
            "status": "in_progress",
            "expected_processes": list(expected_processes),
            "unresolved_items": [],
            "requested_at": time.time(),
        },
    )
    terminated_processes: list[str] = []
    for name, pid in zip(expected_processes, process_pids, strict=True):
        if _terminate_process_group(pid, grace_seconds=grace_seconds):
            terminated_processes.append(name)
        else:
            unresolved.append(f"process:{name}:{pid}")

    receipt = {
        "schema_version": "ade.coordinator_cleanup.v1",
        "run_id": run_id,
        "coordinator_id": coordinator_id,
        "status": "complete" if not unresolved else "incomplete",
        "requested_at": time.time(),
        "completed_at": time.time(),
        "terminalized_engine_commands": sorted(set(terminalized_engine)),
        "terminalized_review_commands": sorted(set(terminalized_review)),
        "terminated_agent_calls": sorted(set(terminated_calls)),
        "cancelled_provider_jobs": sorted(set(cancelled_jobs)),
        "released_gpu_leases": sorted(set(released_leases)),
        "released_external_allocations": sorted(set(released_external)),
        "expected_processes": list(expected_processes),
        "terminated_processes": sorted(set(terminated_processes)),
        "unresolved_items": sorted(set(unresolved)),
        "shared_services_touched": False,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def _coordinator_worker_processes(
    path: Path,
    coordinator_id: str,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return (), ()
    selected: list[tuple[str, int]] = []
    expected = {
        f"engine-{coordinator_id}",
        f"agent-{coordinator_id}",
        f"review-{coordinator_id}",
    }
    for worker in payload.get("workers", ()) if isinstance(payload, dict) else ():
        if not isinstance(worker, dict) or worker.get("name") not in expected:
            continue
        try:
            selected.append((str(worker["name"]), int(worker["pid"])))
        except (KeyError, TypeError, ValueError):
            continue
    selected.sort()
    return (
        tuple(name for name, _pid in selected),
        tuple(pid for _name, pid in selected),
    )


def _provider_job_canceller(judge: dict[str, Any]) -> Callable[[str], object]:
    authorization_env = str(judge.get("authorization_env") or "")
    authorization = os.environ.get(authorization_env) or read_dotenv_value(
        authorization_env
    )
    if not authorization:
        def unavailable(_job_id: str) -> object:
            raise ValueError("Local Judge authorization is unavailable")

        return unavailable
    from ade.local_rubric_judge.http_gateway import HttpRubricJobGateway

    timeout = judge.get("timeout_policy")
    request_timeout = (
        float(timeout.get("request_timeout_seconds", 30.0))
        if isinstance(timeout, dict)
        else 30.0
    )
    gateway = HttpRubricJobGateway(
        str(judge["gateway_url"]),
        authorization=authorization,
        timeout_seconds=request_timeout,
    )
    return gateway.cancel


def _release_coordinator_resources(
    run_id: str,
    coordinator_id: str,
    *,
    ray_address: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    import ray

    started = False
    if not ray.is_initialized():
        ray.init(address=ray_address, namespace="ade", ignore_reinit_error=True)
        started = True
    released_leases: list[str] = []
    released_external: list[str] = []
    try:
        from ade.engine.execution.gpu import get_gpu_lease_allocator

        allocator = get_gpu_lease_allocator()
        snapshot = ray.get(allocator.monitor_snapshot.remote(run_id=run_id))
        owner = f"{run_id}/{coordinator_id}"
        for lease in snapshot.get("leases", ()):
            if lease.get("coordinator_owner") != owner:
                continue
            lease_id = str(lease.get("lease_id") or "")
            if lease_id:
                ray.get(allocator.release.remote(lease_id=lease_id))
                released_leases.append(lease_id)
        for allocation in snapshot.get("external_allocations", ()):
            if allocation.get("coordinator_id") != coordinator_id:
                continue
            token = str(allocation.get("token") or "")
            if token:
                ray.get(allocator.release_external.remote(token=token))
                released_external.append(token)
    finally:
        if started:
            ray.shutdown()
    return tuple(sorted(released_leases)), tuple(sorted(released_external))


def cleanup_cancelled_run(
    *,
    run_id: str,
    run_dir: str | Path,
    queue_root: str | Path,
    review_root: str | Path,
    grace_seconds: float = 10.0,
    process_pids: Iterable[int] = (),
    terminate_processes: bool = True,
    ray_address: str | None = None,
    artifact_cache_dirs: Iterable[str] = (),
) -> dict[str, Any]:
    """Stop Run-owned workers and release Run-owned queues/leases.

    The caller should invoke this after a durable final terminal boundary has
    been written and after the supervisor has stopped normal dispatch.
    """
    if not run_id.strip():
        raise ValueError("run_id is required for cancellation cleanup")
    if grace_seconds <= 0:
        raise ValueError("grace_seconds must be positive")
    root = Path(run_dir).resolve()
    terminated: list[int] = []
    unresolved: list[int] = []
    provenance = root / "reports" / "worker-provenance.json"
    candidate_pids = set()
    if terminate_processes:
        candidate_pids.update(_owned_worker_pids(provenance, run_id))
        candidate_pids.update(
            pid for pid in process_pids if _process_belongs_to_run(int(pid), run_id)
        )
    for pid in sorted(candidate_pids):
        if _terminate_process_group(pid, grace_seconds=grace_seconds):
            terminated.append(pid)
        else:
            unresolved.append(pid)

    engine_queue = FileCommandQueue(queue_root)
    engine_terminalized = engine_queue.command_ids(run_id=run_id)
    for command_id in engine_terminalized:
        engine_queue.interrupt(
            command_id, reason="Run reached terminal cleanup", retryable=False,
        )
    review_queue = FileReviewCommandQueue(
        review_root, recover_claimed=False
    )
    review_terminalized = review_queue.command_ids(run_id=run_id)
    for command_id in review_terminalized:
        review_queue.interrupt(command_id, reason="Run reached terminal cleanup")
    released_leases: list[str] = []
    released_external: list[str] = []
    temporary_artifacts_removed: list[str] = []
    temporary_artifact_errors: list[str] = []
    resource_error: str | None = None
    cache_roots = tuple(
        sorted({str(Path(value).expanduser().resolve()) for value in artifact_cache_dirs})
    )
    cache_results: list[dict[str, Any]] = []
    cache_error: str | None = None
    ray_started = False
    ray = None
    try:
        import ray

        if not ray.is_initialized() and ray_address:
            ray.init(address=ray_address, namespace="ade", ignore_reinit_error=True)
            ray_started = True
    except Exception as error:
        resource_error = f"{type(error).__name__}: {error}"

    if ray is not None and ray.is_initialized():
        try:
            from ade.engine.execution.gpu import get_gpu_lease_allocator

            allocator = get_gpu_lease_allocator()
            snapshot = ray.get(allocator.monitor_snapshot.remote(run_id=run_id))
            for lease in snapshot.get("leases", ()):
                lease_id = str(lease.get("lease_id") or "")
                if lease_id:
                    ray.get(allocator.release.remote(lease_id=lease_id))
                    released_leases.append(lease_id)
            for token in snapshot.get("external_allocations", ()):
                token_id = str(token.get("token") or "")
                if token_id:
                    ray.get(allocator.release_external.remote(token=token_id))
                    released_external.append(token_id)
        except Exception as error:  # retain cleanup evidence for operator repair
            resource_error = f"{type(error).__name__}: {error}"
        if cache_roots:
            try:
                from ade.engine.execution.ray import cleanup_artifact_cache_on_gpu_nodes

                cache_results = [
                    cleanup_artifact_cache_on_gpu_nodes(
                        cache_dir=cache_dir,
                        run_id=run_id,
                    )
                    for cache_dir in cache_roots
                ]
            except Exception as error:
                cache_error = f"{type(error).__name__}: {error}"
    elif cache_roots:
        cache_error = "Ray is unavailable for requested artifact cache cleanup"

    if ray_started:
        try:
            ray.shutdown()
        except Exception:
            pass

    for path in _run_temp_artifacts(run_id):
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            temporary_artifacts_removed.append(str(path))
        except OSError as error:
            temporary_artifact_errors.append(f"{path}: {error}")

    status = (
        "complete"
        if not unresolved
        and resource_error is None
        and cache_error is None
        and not temporary_artifact_errors
        else "incomplete"
    )
    receipt = {
        "schema_version": "ade.run_cleanup.v1",
        "run_id": run_id,
        "status": status,
        "requested_at": time.time(),
        "completed_at": time.time(),
        "terminated_process_groups": terminated,
        "unresolved_processes": unresolved,
        "terminalized_engine_commands": list(engine_terminalized),
        "terminalized_review_commands": list(review_terminalized),
        "released_gpu_leases": released_leases,
        "released_external_allocations": released_external,
        "temporary_artifacts_removed": temporary_artifacts_removed,
        "temporary_artifact_errors": temporary_artifact_errors,
        "artifact_cache_cleanup": {
            "status": (
                "not_required"
                if not cache_roots
                else "complete"
                if cache_error is None
                else "incomplete"
            ),
            "cache_dirs": list(cache_roots),
            "results": cache_results,
            **({"error": cache_error} if cache_error is not None else {}),
        },
        "shared_cluster_touched": False,
    }
    if resource_error is not None:
        receipt["resource_error"] = resource_error
    path = root / "reports" / "cleanup" / "run-cleanup.json"
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        previous = {}
    if isinstance(previous, dict):
        for key in (
            "terminated_process_groups",
            "terminalized_engine_commands",
            "terminalized_review_commands",
            "released_gpu_leases",
            "released_external_allocations",
            "temporary_artifacts_removed",
        ):
            previous_values = previous.get(key, ())
            receipt[key] = sorted(set(previous_values) | set(receipt[key]))
        receipt["unresolved_processes"] = sorted(
            set(receipt["unresolved_processes"])
            | set(_live_owned_pids(previous.get("unresolved_processes", ()), run_id))
        )
        receipt["status"] = (
            "complete"
            if not receipt["unresolved_processes"]
            and "resource_error" not in receipt
            and receipt["artifact_cache_cleanup"]["status"]
            in {"complete", "not_required"}
            and not receipt["temporary_artifact_errors"]
            else "incomplete"
        )
    write_json_atomic(path, receipt)
    return receipt


def _live_owned_pids(values: Iterable[object], run_id: str) -> tuple[int, ...]:
    live: set[int] = set()
    for value in values:
        try:
            pid = int(value)
        except (TypeError, ValueError):
            continue
        if _process_belongs_to_run(pid, run_id):
            live.add(pid)
    return tuple(sorted(live))


def _run_temp_artifacts(run_id: str) -> tuple[Path, ...]:
    """Return only exact Run-prefixed temporary artifacts.

    Ray and provider workers may place Run-specific temporary directories in
    the system temp directory or one user-scoped directory below it.  Shared
    Ray session roots and unrelated temporary files are intentionally ignored.
    """
    prefix = f"{run_id}--"
    roots = {Path(tempfile.gettempdir()).resolve()}
    candidates: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        first_level = tuple(root.iterdir())
        for candidate in first_level:
            if candidate.name.startswith(prefix):
                candidates.add(candidate)
            if not candidate.is_dir() or candidate.is_symlink():
                continue
            second_level = tuple(candidate.iterdir())
            for nested in second_level:
                if nested.name.startswith(prefix):
                    candidates.add(nested)
                if not nested.is_dir() or nested.is_symlink():
                    continue
                for leaf in nested.iterdir():
                    if leaf.name.startswith(prefix):
                        candidates.add(leaf)
    return tuple(sorted(candidates))


def _owned_worker_pids(path: Path, run_id: str) -> tuple[int, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return ()
    pids: set[int] = set()
    for worker in payload.get("workers", ()) if isinstance(payload, dict) else ():
        if not isinstance(worker, dict):
            continue
        try:
            pid = int(worker.get("pid"))
        except (TypeError, ValueError):
            continue
        if _process_belongs_to_run(pid, run_id):
            pids.add(pid)
    return tuple(sorted(pids))


def _process_belongs_to_run(pid: int, run_id: str) -> bool:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except (OSError, ValueError):
        return False
    argv = tuple(item.decode("utf-8", "replace") for item in raw.split(b"\0") if item)
    return (
        "ade.harness.cli" in argv
        and run_id in argv
        and any(item in {"monitor", "control", "agent", "engine", "review"} for item in argv)
    )


def _terminate_process_group(pid: int, *, grace_seconds: float) -> bool:
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return True
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False
