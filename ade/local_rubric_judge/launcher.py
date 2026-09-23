"""Ray-owned deployment Judge and its node-local process group."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from typing import Any, Callable

from ade.local_rubric_judge.lifecycle import LocalJudgeBinding, RunResourceAdmissionError


class JudgeProcessGroup:
    def __init__(
        self,
        root: str | Path,
        *,
        authorization_env: str,
        readiness_seconds: float,
        request_timeout_seconds: float,
        vllm: dict[str, Any],
        generation: dict[str, Any],
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        if not authorization_env or readiness_seconds <= 0 or request_timeout_seconds <= 0:
            raise ValueError("Local Judge launcher settings are invalid")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.authorization_env = authorization_env
        self.readiness_seconds = float(readiness_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.vllm = dict(vllm)
        self.generation = dict(generation)
        self.process_factory = process_factory
        self._processes: dict[str, tuple[subprocess.Popen, ...]] = {}
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.implementation_revision = self._current_implementation_revision()

    @staticmethod
    def _current_implementation_revision() -> str | None:
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        project_root = Path(__file__).resolve().parents[2]
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
        revision = result.stdout.strip()
        return revision or None

    def start(self, *, run_id: str, binding: LocalJudgeBinding) -> dict[str, Any]:
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        authorization = os.environ.get(self.authorization_env)
        if not authorization:
            raise ValueError(
                f"Local Judge authorization environment is missing: {self.authorization_env}"
            )
        from ray.util import get_node_ip_address

        host = get_node_ip_address()
        gateway_url = f"http://{host}:{binding.gateway_port}"
        launch_id = f"judge-{run_id}-{uuid.uuid4().hex[:12]}"
        launch_root = self.root / run_id / launch_id
        launch_root.mkdir(parents=True, exist_ok=False)
        processes: list[subprocess.Popen] = []
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(gpu_ids) != binding.gpu_count or not all(gpu_ids):
            raise ValueError("Judge must run inside its Ray GPU allocation")
        endpoint_urls = tuple(f"http://127.0.0.1:{8901 + index}" for index in range(binding.gpu_count))
        try:
            for index, gpu_id in enumerate(gpu_ids):
                log = (launch_root / f"vllm-{index}.log").open("ab")
                env = os.environ.copy()
                env.update(self.vllm["environment"])
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                command = [
                    str(self.vllm["executable"]),
                    "serve",
                    binding.model_path,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(8901 + index),
                    "--served-model-name",
                    binding.model_digest,
                    "--dtype",
                    "auto",
                    "--trust-remote-code",
                    "--gpu-memory-utilization",
                    str(self.vllm["gpu_memory_utilization"]),
                    "--max-model-len",
                    str(self.vllm["max_model_len"]),
                    "--max-num-seqs",
                    str(self.vllm["max_num_seqs"]),
                    "--max-num-batched-tokens",
                    str(self.vllm["max_num_batched_tokens"]),
                    "--reasoning-parser",
                    str(self.vllm["reasoning_parser"]),
                    "--gdn-prefill-backend",
                    str(self.vllm["gdn_prefill_backend"]),
                    "--no-enable-log-requests",
                ]
                if bool(self.vllm["enable_prefix_caching"]):
                    command.append("--enable-prefix-caching")
                process = self.process_factory(
                    tuple(command),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                log.close()
                processes.append(process)
                if index + 1 < len(gpu_ids):
                    time.sleep(float(self.vllm["launch_stagger_seconds"]))
            endpoint_deadline = time.monotonic() + self.readiness_seconds
            while time.monotonic() < endpoint_deadline:
                if any(process.poll() is not None for process in processes):
                    raise RuntimeError("Local Judge child process exited before readiness")
                try:
                    for endpoint in endpoint_urls:
                        with self._opener.open(f"{endpoint}/health", timeout=1.0) as response:
                            if response.status != 200:
                                raise RuntimeError("vLLM endpoint is not ready")
                    break
                except Exception:
                    time.sleep(0.5)
            else:
                raise TimeoutError("vLLM endpoints did not become ready before gateway startup")
            gateway_log = (launch_root / "gateway.log").open("ab")
            gateway_command = [
                sys.executable,
                "-m",
                "ade.local_rubric_judge.gateway_process",
                "--state",
                str(launch_root / "jobs"),
                "--host",
                "0.0.0.0",
                "--port",
                str(binding.gateway_port),
                "--authorization-env",
                self.authorization_env,
                "--launch-id",
                launch_id,
                "--protocol",
                binding.protocol,
                "--model",
                binding.model_digest,
                "--model-digest",
                binding.model_digest,
                "--request-timeout-seconds",
                str(self.request_timeout_seconds),
                "--per-endpoint-concurrency",
                str(self.generation["per_endpoint_concurrency"]),
                "--max-tokens",
                str(self.generation["max_tokens"]),
                "--temperature",
                str(self.generation["temperature"]),
                "--top-p",
                str(self.generation["top_p"]),
                "--seed",
                str(self.generation["seed"]),
            ]
            if bool(self.generation["enable_thinking"]):
                gateway_command.append("--enable-thinking")
            for endpoint in endpoint_urls:
                gateway_command.extend(("--endpoint", endpoint))
            processes.append(
                self.process_factory(
                    tuple(gateway_command),
                    env=os.environ.copy(),
                    stdout=gateway_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            gateway_log.close()
        except BaseException:
            self._stop_processes(tuple(processes))
            if launch_root.exists():
                shutil.rmtree(launch_root)
            raise
        self._processes[launch_id] = tuple(processes)
        handle = {
            "launch_id": launch_id,
            "run_id": run_id,
            "protocol": binding.protocol,
            "model_digest": binding.model_digest,
            "gateway_url": gateway_url,
            "host": host,
            "vllm_urls": list(endpoint_urls),
            "pids": [process.pid for process in processes],
            "vllm_ports": list(range(8901, 8901 + binding.gpu_count)),
            "gpu_ids": gpu_ids,
            "state_path": str(launch_root / "jobs"),
            "log_root": str(launch_root),
            "vllm": dict(self.vllm),
            "generation": dict(self.generation),
            "implementation_revision": self.implementation_revision,
        }
        try:
            deadline = time.monotonic() + self.readiness_seconds
            while time.monotonic() < deadline:
                exited = [
                    {"pid": process.pid, "returncode": process.poll()}
                    for process in processes
                    if process.poll() is not None
                ]
                if exited:
                    raise RuntimeError(
                        f"Local Judge child process exited before readiness: {exited}"
                    )
                if self.health(handle)["ready"]:
                    return handle
                time.sleep(0.5)
            raise TimeoutError("Local Judge did not become ready before its deadline")
        except BaseException:
            self.stop(handle)
            self.cleanup_stopped(handle)
            raise

    def health(self, handle: dict[str, Any]) -> dict[str, Any]:
        processes_alive = all(self._pid_alive(int(pid)) for pid in handle["pids"])
        gateway_ready = False
        endpoints_ready = False
        gateway_identity: dict[str, Any] = {}
        if processes_alive:
            try:
                for endpoint in handle["vllm_urls"]:
                    with self._opener.open(f"{endpoint}/health", timeout=1.0) as response:
                        if response.status != 200:
                            raise RuntimeError("vLLM endpoint is not ready")
                endpoints_ready = True
                request = urllib.request.Request(
                    f"{str(handle['gateway_url']).rstrip('/')}/v1",
                    headers={"Authorization": str(os.environ[self.authorization_env])},
                )
                with self._opener.open(request, timeout=1.0) as response:
                    gateway_identity = json.loads(response.read())
                gateway_ready = all(
                    gateway_identity.get(key) == handle[key]
                    for key in ("launch_id", "protocol", "model_digest")
                )
            except Exception:
                gateway_ready = False
        return {
            "ready": processes_alive and endpoints_ready and gateway_ready,
            "protocol": gateway_identity.get("protocol"),
            "model_digest": gateway_identity.get("model_digest"),
            "launch_id": gateway_identity.get("launch_id"),
        }

    def cancel_jobs(self, handle: dict[str, Any]) -> None:
        try:
            request = urllib.request.Request(
                f"{str(handle['gateway_url']).rstrip('/')}/v1/jobs/cancel-all",
                data=b"{}",
                headers={
                    "Authorization": str(os.environ[self.authorization_env]),
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with self._opener.open(request, timeout=5):
                pass
        except Exception:
            pass

    def stop(self, handle: dict[str, Any]) -> None:
        processes = self._processes.pop(str(handle["launch_id"]), ())
        if processes:
            self._stop_processes(processes)
            return
        for pid in handle.get("pids", ()):
            try:
                os.killpg(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                continue
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(
            self._pid_alive(int(pid)) for pid in handle.get("pids", ())
        ):
            time.sleep(0.1)
        for pid in handle.get("pids", ()):
            if self._pid_alive(int(pid)):
                try:
                    os.killpg(int(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def released(self, handle: dict[str, Any]) -> bool:
        return not any(self._pid_alive(int(pid)) for pid in handle.get("pids", ()))

    def cleanup_stopped(self, handle: dict[str, Any]) -> tuple[str, ...]:
        """Remove the exact stopped launch's transient jobs and log root."""
        if not self.released(handle):
            raise RuntimeError("cannot clean a live Local Judge launch")
        raw = handle.get("log_root")
        if not isinstance(raw, str) or not raw:
            return ()
        launch_root = Path(raw).resolve()
        if launch_root == self.root or not launch_root.is_relative_to(self.root):
            raise ValueError("Local Judge log root is outside the configured launcher root")
        if not launch_root.exists():
            return ()
        shutil.rmtree(launch_root)
        return (str(launch_root),)

    def cleanup_stale_log_roots(self, *, keep_log_root: str | Path) -> tuple[str, ...]:
        """Remove stopped Judge launch logs while retaining the resident launch."""
        keep = Path(keep_log_root).resolve()
        removed: list[str] = []
        for deployment_root in self.root.glob("deployment-*"):
            if not deployment_root.is_dir():
                continue
            for launch_root in deployment_root.glob("judge-*"):
                if not launch_root.is_dir() or launch_root.resolve() == keep:
                    continue
                shutil.rmtree(launch_root)
                removed.append(str(launch_root))
        return tuple(sorted(removed))

    @staticmethod
    def _stop_processes(processes: tuple[subprocess.Popen, ...]) -> None:
        for process in reversed(processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 10
        for process in reversed(processes):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False


class _JudgeActor:
    """Own the GPU reservation until every child has stopped."""

    def __init__(self, root, settings, binding, run_id, authorization):
        import ray

        os.environ[settings["authorization_env"]] = authorization
        self.launcher = JudgeProcessGroup(root, **settings)
        self.binding = LocalJudgeBinding.from_dict(binding)
        self.handle = self.launcher.start(run_id=run_id, binding=self.binding)
        self.handle["node_id"] = ray.get_runtime_context().get_node_id()

    def describe(self):
        from dataclasses import asdict

        return {"binding": asdict(self.binding), "handle": self.handle}

    def health(self):
        return self.launcher.health(self.handle)

    def cancel_jobs(self):
        self.launcher.cancel_jobs(self.handle)

    def stop(self):
        self.launcher.cancel_jobs(self.handle)
        self.launcher.stop(self.handle)
        if not self.launcher.released(self.handle):
            raise RuntimeError("Judge child processes have not exited")
        return self.launcher.cleanup_stopped(self.handle)


def _cleanup_dead_actor(root, settings, handle):
    """Reconcile the persisted process group on its original Ray node."""
    launcher = JudgeProcessGroup(root, **settings)
    launcher.stop(handle)
    if not launcher.released(handle):
        raise RuntimeError("Dead Judge actor left live child processes")
    return launcher.cleanup_stopped(handle)


class ProductionJudgeLauncher:
    """Attach to one named, detached Judge actor in the experiment's cluster."""

    namespace = "ade"

    def __init__(self, root, *, authorization_env, readiness_seconds,
                 request_timeout_seconds, vllm, generation):
        self.root = Path(root).resolve()
        self.settings = {
            "authorization_env": authorization_env,
            "readiness_seconds": readiness_seconds,
            "request_timeout_seconds": request_timeout_seconds,
            "vllm": dict(vllm),
            "generation": dict(generation),
        }
        self.vllm = dict(vllm)
        self.generation = dict(generation)
        self.readiness_seconds = float(readiness_seconds)
        self.implementation_revision = JudgeProcessGroup._current_implementation_revision()
        self._cleaned: dict[str, tuple[str, ...]] = {}

    def _connect(self, address):
        import ray

        if not ray.is_initialized():
            ray.init(address=address, namespace=self.namespace, logging_level="ERROR")
        # Ray's subreaper sets SIGCHLD=SIG_IGN even in the driver; ADE's
        # subprocess supervision needs real waitpid exit statuses.
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        expected_host, expected_port = address.rsplit(":", 1)
        actual_host, actual_port = ray.get_runtime_context().gcs_address.rsplit(":", 1)
        if (socket.gethostbyname(expected_host), expected_port) != (socket.gethostbyname(actual_host), actual_port):
            raise RuntimeError("Judge launcher is connected to a different Ray cluster")
        return ray

    def _actor(self, handle):
        ray = self._connect(handle["ray_address"])
        try:
            actor = ray.get_actor(handle["actor_name"], namespace=self.namespace)
            if actor._actor_id.hex() != handle["actor_id"]:
                raise RuntimeError("Judge actor identity changed")
            return actor
        except ValueError:
            return None

    def start(self, *, run_id, binding):
        from dataclasses import asdict

        authorization = os.environ.get(self.settings["authorization_env"])
        if not authorization:
            raise ValueError(
                "Local Judge authorization environment is missing: "
                + self.settings["authorization_env"]
            )
        ray = self._connect(binding.ray_address)
        if not any(node["Alive"] and node["Resources"].get("GPU", 0) >= binding.gpu_count
                   for node in ray.nodes()):
            raise RunResourceAdmissionError(
                "ray_judge_capacity_missing", "Judge requires a Ray node advertising eight GPUs"
            )
        name = f"ade-judge-{binding.cluster_id}"
        try:
            actor = ray.get_actor(name, namespace=self.namespace)
        except ValueError:
            actor = ray.remote(_JudgeActor).options(
                name=name, namespace=self.namespace, lifetime="detached",
                num_cpus=1, num_gpus=binding.gpu_count,
                max_restarts=0, max_task_retries=0,
            ).remote(str(self.root), self.settings, asdict(binding), run_id, authorization)
        try:
            description = ray.get(actor.describe.remote(), timeout=2 * self.readiness_seconds + 60)
        except BaseException:
            # Failed constructors clean their partial process groups. For a stuck
            # constructor Ray's configured subreaper owns descendant cleanup.
            ray.kill(actor, no_restart=True)
            raise
        handle = {**description["handle"], "actor_name": name,
                  "ray_address": binding.ray_address, "actor_id": actor._actor_id.hex()}
        if (description["binding"] != asdict(binding)
                or handle["vllm"] != self.vllm
                or handle["generation"] != self.generation
                or handle["implementation_revision"] != self.implementation_revision):
            raise RuntimeError("Existing Judge actor has a different deployment binding")
        return handle

    def health(self, handle):
        ray = self._connect(handle["ray_address"])
        actor = self._actor(handle)
        if actor is None:
            return {"ready": False}
        try:
            description = ray.get(actor.describe.remote(), timeout=10)
            if description["handle"]["launch_id"] != handle["launch_id"]:
                return {"ready": False}
            return ray.get(actor.health.remote(), timeout=30)
        except (ray.exceptions.RayError, TimeoutError):
            return {"ready": False}

    def cancel_jobs(self, handle):
        ray = self._connect(handle["ray_address"])
        actor = self._actor(handle)
        if actor is not None:
            try:
                ray.get(actor.cancel_jobs.remote(), timeout=10)
            except ray.exceptions.RayActorError:
                pass

    def stop(self, handle):
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        ray = self._connect(handle["ray_address"])
        actor = self._actor(handle)
        removed = None
        if actor is not None:
            try:
                description = ray.get(actor.describe.remote(), timeout=10)
                if description["handle"]["launch_id"] != handle["launch_id"]:
                    raise RuntimeError("Refusing to stop a different Judge launch")
                removed = ray.get(actor.stop.remote(), timeout=60)
            except ray.exceptions.RayActorError:
                pass
            if removed is not None:
                ray.kill(actor, no_restart=True)
        if removed is None:
            alive = any(node["Alive"] and node["NodeID"] == handle["node_id"]
                        for node in ray.nodes())
            if alive:
                removed = ray.get(ray.remote(_cleanup_dead_actor).options(
                    num_cpus=0, num_gpus=0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(handle["node_id"], soft=False),
                ).remote(str(self.root), self.settings, handle), timeout=60)
            else:
                raise RuntimeError("Judge node is unavailable; child cleanup is unverified")
        self._cleaned[handle["launch_id"]] = tuple(removed)

    def released(self, handle):
        return handle["launch_id"] in self._cleaned

    def cleanup_stopped(self, handle):
        if not self.released(handle):
            raise RuntimeError("Judge actor cleanup has not completed")
        return self._cleaned[handle["launch_id"]]
