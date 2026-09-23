"""Deployment-owned Local Rubric Judge health admission and Run attachment."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import json
from pathlib import Path
from typing import Any, Protocol

from ade.engine.storage.atomic import write_json_atomic


@dataclass(frozen=True)
class LocalJudgeBinding:
    cluster_id: str
    ray_address: str
    gpu_count: int
    model_path: str
    model_digest: str
    gateway_port: int
    protocol: str = "ade.rubric_jobs.v1"

    @classmethod
    def from_dict(cls, value: object) -> "LocalJudgeBinding":
        if not isinstance(value, dict) or set(value) != {
            "cluster_id", "ray_address", "gpu_count", "model_path",
            "model_digest", "gateway_port", "protocol",
        }:
            raise ValueError("resolved Local Judge binding fields are invalid")
        gpu_count = value["gpu_count"]
        if type(gpu_count) is not int or gpu_count != 8:
            raise ValueError("Local Judge requires eight Ray-allocated GPUs")
        for field in (
            "cluster_id", "ray_address", "model_path",
            "model_digest",
        ):
            if not isinstance(value[field], str) or not value[field].strip():
                raise ValueError(f"Local Judge {field} is required")
        if value["protocol"] != "ade.rubric_jobs.v1":
            raise ValueError("Local Judge protocol identity is invalid")
        port = value["gateway_port"]
        if type(port) is not int or not 1 <= port <= 65535 or port in range(8901, 8909):
            raise ValueError("Local Judge gateway_port must be valid and separate from vLLM ports")
        return cls(**value)


class JudgeLauncher(Protocol):
    def start(self, *, run_id: str, binding: LocalJudgeBinding) -> dict[str, Any]: ...
    def health(self, handle: dict[str, Any]) -> dict[str, Any]: ...
    def cancel_jobs(self, handle: dict[str, Any]) -> None: ...
    def stop(self, handle: dict[str, Any]) -> None: ...
    def released(self, handle: dict[str, Any]) -> bool: ...
    def cleanup_stopped(self, handle: dict[str, Any]) -> tuple[str, ...]: ...


class RunResourceAdmissionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LocalJudgeReadyOutcome:
    run_id: str
    launch_id: str
    state_path: str
    host: str
    gateway_url: str
    node_id: str
    status: str = "ready"


@dataclass(frozen=True)
class LocalJudgeDetachedOutcome:
    run_id: str
    launch_id: str
    state_path: str
    status: str = "detached"


class RunResourceAdmission:
    """Attach one exclusive Run to a deployment-owned Judge launch.

    A healthy service is retained across same-Run recovery and direct handoff,
    but a final Run terminal boundary stops its exact persisted launch. Tests
    use FakeJudgeLauncher; this module never discovers or kills processes by
    name.
    """

    def __init__(self, state_root: str | Path, launcher: JudgeLauncher) -> None:
        self.state_root = Path(state_root)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.launcher = launcher
        self._clusters: dict[str, str] = {}
        self._handles: dict[str, dict[str, Any]] = {}
        self._service_path = self.state_root / "service.json"
        self._lock_path = self.state_root / ".admission.lock"
        self._service: dict[str, Any] | None = None
        self._reload_persisted_state()

    def _reload_persisted_state(self) -> None:
        self._clusters = {}
        self._handles = {}
        self._service = None
        if self._service_path.is_file():
            value = json.loads(self._service_path.read_text(encoding="utf-8"))
            # A failed lifecycle transition can persist ``stopped`` after the
            # exact launch is still alive.  Keep the recorded handle available
            # for health/release reconciliation; _ensure_service will reuse a
            # healthy launch or stop that exact handle before replacing it.
            if value.get("status") in {"ready", "stopped"}:
                self._service = value
        for path in sorted(self.state_root.glob("*.json")):
            if path == self._service_path:
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") != "attached":
                continue
            run_id = str(value["run_id"])
            cluster = str(value["binding"]["cluster_id"])
            if cluster in self._clusters and self._clusters[cluster] != run_id:
                raise RunResourceAdmissionError(
                    "persisted_lease_conflict",
                    f"cluster has conflicting persisted Run leases: {cluster}",
                )
            self._clusters[cluster] = run_id
            self._handles[run_id] = value

    def admit(
        self,
        *,
        run_id: str,
        binding: LocalJudgeBinding,
        supersedes_run_id: str | None = None,
    ) -> LocalJudgeReadyOutcome:
        with self._locked():
            self._reload_persisted_state()
            expected_binding = {
                **asdict(binding),
                "gpu_count": binding.gpu_count,
            }
            owner = self._clusters.get(binding.cluster_id)
            if owner is not None and owner != run_id:
                if not supersedes_run_id or owner != supersedes_run_id:
                    raise RunResourceAdmissionError(
                        "cluster_already_leased",
                        f"cluster already leased by Run {owner}",
                    )
                source = self._handles[owner]
                if source.get("binding") != expected_binding:
                    raise RunResourceAdmissionError(
                        "fork_binding_changed",
                        "fork cannot hand off a different Local Judge binding",
                    )
                handle = self._ensure_service(binding)
                self.launcher.cancel_jobs(handle)
                detached = {
                    **source,
                    "status": "detached",
                    "service_handle": handle,
                    "superseded_by": run_id,
                }
                write_json_atomic(self.state_root / f"{owner}.json", detached)
                self._clusters.pop(binding.cluster_id, None)
                self._handles.pop(owner, None)

            if run_id in self._handles:
                state = self._handles[run_id]
                if state.get("binding") != expected_binding:
                    raise RunResourceAdmissionError(
                        "run_binding_changed",
                        "attached Run requested a different Local Judge binding",
                    )
                handle = self._ensure_service(binding)
                if state.get("service_handle") != handle:
                    state = {**state, "service_handle": handle}
                    self._handles[run_id] = state
                    write_json_atomic(self.state_root / f"{run_id}.json", state)
                return self._ready_outcome(state)

            handle = self._ensure_service(binding)
            state = {
                "run_id": run_id,
                "status": "attached",
                "binding": expected_binding,
                "service_handle": handle,
            }
            self._clusters[binding.cluster_id] = run_id
            self._handles[run_id] = state
            write_json_atomic(self.state_root / f"{run_id}.json", state)
            return self._ready_outcome(state)

    def terminal(self, run_id: str) -> LocalJudgeDetachedOutcome | None:
        with self._locked():
            self._reload_persisted_state()
            state = self._handles.get(run_id)
            if state is None:
                path = self.state_root / f"{run_id}.json"
                if not path.is_file():
                    return None
                persisted = json.loads(path.read_text())
                return LocalJudgeDetachedOutcome(
                    run_id,
                    str(persisted["service_handle"]["launch_id"]),
                    str(path),
                )
            handle = state["service_handle"]
            self.launcher.cancel_jobs(handle)
            self.launcher.stop(handle)
            if not self.launcher.released(handle):
                raise RunResourceAdmissionError(
                    "judge_release_incomplete",
                    f"Local Judge launch did not stop for terminal Run {run_id}",
                )
            self.launcher.cleanup_stopped(handle)
            if (
                self._service is not None
                and self._service.get("service_handle", {}).get("launch_id")
                == handle.get("launch_id")
            ):
                self._service = {**self._service, "status": "stopped"}
                write_json_atomic(self._service_path, self._service)
            cluster = state["binding"]["cluster_id"]
            self._clusters.pop(cluster, None)
            terminal = {**state, "status": "detached"}
            self._handles.pop(run_id, None)
            write_json_atomic(self.state_root / f"{run_id}.json", terminal)
            return LocalJudgeDetachedOutcome(
                run_id,
                str(handle["launch_id"]),
                str(self.state_root / f"{run_id}.json"),
            )

    @contextmanager
    def _locked(self):
        with self._lock_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _ensure_service(self, binding: LocalJudgeBinding) -> dict[str, Any]:
        expected_binding = {**asdict(binding), "gpu_count": binding.gpu_count}
        if self._service is not None:
            handle = self._service["service_handle"]
            health = self.launcher.health(handle)
            expected_health = {
                "ready": True,
                "protocol": binding.protocol,
                "model_digest": binding.model_digest,
                "launch_id": handle.get("launch_id"),
            }
            if (
                self._service.get("status") == "ready"
                and self._service.get("binding") == expected_binding
                and handle.get("vllm") == getattr(self.launcher, "vllm", handle.get("vllm"))
                and handle.get("generation")
                == getattr(self.launcher, "generation", handle.get("generation"))
                and handle.get("implementation_revision")
                == getattr(
                    self.launcher,
                    "implementation_revision",
                    handle.get("implementation_revision"),
                )
                and all(health.get(key) == value for key, value in expected_health.items())
            ):
                return handle
            self.launcher.cancel_jobs(handle)
            self.launcher.stop(handle)
            if self.launcher.released(handle):
                self.launcher.cleanup_stopped(handle)
            stopped = {**self._service, "status": "stopped"}
            write_json_atomic(self._service_path, stopped)
            self._service = None

        handle = self.launcher.start(
            run_id=f"deployment-{binding.cluster_id}", binding=binding
        )
        health = self.launcher.health(handle)
        expected_health = {
            "ready": True,
            "protocol": binding.protocol,
            "model_digest": binding.model_digest,
            "launch_id": handle.get("launch_id"),
        }
        if any(health.get(key) != value for key, value in expected_health.items()):
            self.launcher.cancel_jobs(handle)
            self.launcher.stop(handle)
            if self.launcher.released(handle):
                self.launcher.cleanup_stopped(handle)
            raise RunResourceAdmissionError(
                "readiness_identity_mismatch",
                "Local Judge readiness identity mismatch",
            )
        self._service = {
            "status": "ready",
            "binding": expected_binding,
            "service_handle": handle,
        }
        write_json_atomic(self._service_path, self._service)
        return handle

    def _ready_outcome(self, state: dict[str, Any]) -> LocalJudgeReadyOutcome:
        return LocalJudgeReadyOutcome(
            str(state["run_id"]),
            str(state["service_handle"]["launch_id"]),
            str(self.state_root / f"{state['run_id']}.json"),
            str(state["service_handle"]["host"]),
            str(state["service_handle"]["gateway_url"]),
            str(state["service_handle"]["node_id"]),
        )


class FakeJudgeLauncher:
    def __init__(self, *, health_override: dict[str, Any] | None = None) -> None:
        self.health_override = health_override
        self.starts: list[str] = []
        self.cancelled: list[str] = []
        self.stopped: list[str] = []

    def start(self, *, run_id: str, binding: LocalJudgeBinding) -> dict[str, Any]:
        self.starts.append(run_id)
        return {
            "launch_id": f"fake-{run_id}",
            "host": "192.0.2.1",
            "gateway_url": f"http://192.0.2.1:{binding.gateway_port}",
            "node_id": "fake-node",
            "run_id": run_id,
            "protocol": binding.protocol,
            "model_digest": binding.model_digest,
        }

    def health(self, handle: dict[str, Any]) -> dict[str, Any]:
        return self.health_override or {
            "ready": True,
            "protocol": handle["protocol"],
            "model_digest": handle["model_digest"],
            "launch_id": handle["launch_id"],
        }

    def cancel_jobs(self, handle: dict[str, Any]) -> None:
        self.cancelled.append(str(handle["launch_id"]))

    def stop(self, handle: dict[str, Any]) -> None:
        self.stopped.append(str(handle["launch_id"]))

    def released(self, handle: dict[str, Any]) -> bool:
        return str(handle["launch_id"]) in self.stopped

    def cleanup_stopped(self, handle: dict[str, Any]) -> tuple[str, ...]:
        return ()
