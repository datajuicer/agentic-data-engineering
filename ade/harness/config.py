"""Strict compilation of the new ADE run config."""

from dataclasses import dataclass
from typing import Any

from ade.core.run import PortfolioState, ResolvedTask
from ade.tasks.registry import TaskRegistry

_FIELDS = {"schema_version", "run", "task", "analysis_policy", "run_resources"}
_REQUIRED_FIELDS = {"schema_version", "run", "task"}
_RUN_FIELDS = {"id", "max_plans", "max_trials"}
_TASK_FIELDS = {"id", "config"}


@dataclass(frozen=True)
class ResolvedRunConfig:
    run_id: str
    task: ResolvedTask
    portfolio: PortfolioState
    analysis_policy: dict[str, Any] | None = None
    run_resources: dict[str, Any] | None = None


class ConfigCompiler:
    def __init__(
        self,
        tasks: TaskRegistry,
    ) -> None:
        self.tasks = tasks

    def compile(self, payload: dict[str, Any]) -> ResolvedRunConfig:
        unknown = set(payload) - _FIELDS
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        missing = _REQUIRED_FIELDS - set(payload)
        if missing:
            raise ValueError(f"missing config fields: {sorted(missing)}")
        if payload["schema_version"] != 1:
            raise ValueError("schema_version must be 1")
        run = payload["run"]
        task = payload["task"]
        if not isinstance(run, dict) or not isinstance(task, dict):
            raise ValueError("run and task must be mappings")
        self._validate_fields(run, _RUN_FIELDS, _RUN_FIELDS, "run")
        self._validate_fields(task, _TASK_FIELDS, {"id"}, "task")
        run_id = str(run["id"])
        task_id = str(task["id"])
        self.tasks.get(task_id)
        task_config = task.get("config", {})
        if not isinstance(task_config, dict):
            raise ValueError("task.config must be a mapping")
        resources = _compile_run_resources(payload.get("run_resources"))
        analysis_policy = payload.get("analysis_policy")
        if analysis_policy is not None and not isinstance(analysis_policy, dict):
            raise ValueError("analysis_policy must be a mapping")
        return ResolvedRunConfig(
            run_id=run_id,
            task=ResolvedTask(
                task_id=task_id,
                plugin_id=task_id,
                config=task_config,
            ),
            portfolio=PortfolioState(
                max_plans=int(run["max_plans"]),
                max_trials=int(run["max_trials"]),
            ),
            analysis_policy=(
                dict(analysis_policy) if analysis_policy is not None else None
            ),
            run_resources=resources,
        )

    @staticmethod
    def _validate_fields(
        payload: dict[str, Any],
        allowed: set[str],
        required: set[str],
        label: str,
    ) -> None:
        unknown = set(payload) - allowed
        missing = required - set(payload)
        if unknown:
            raise ValueError(f"unknown {label} fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing {label} fields: {sorted(missing)}")


def _compile_run_resources(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"ray_cluster", "local_judge"}:
        raise ValueError("run_resources must contain ray_cluster and local_judge")
    ray = value["ray_cluster"]
    judge = value["local_judge"]
    if not isinstance(ray, dict) or set(ray) != {
        "cluster_id", "address", "exclusive_ade_run"
    }:
        raise ValueError("run_resources.ray_cluster fields are invalid")
    if ray.get("exclusive_ade_run") is not True:
        raise ValueError("ADE Run requires an exclusive Ray Cluster")
    if not isinstance(judge, dict) or set(judge) != {
        "enabled", "gpu_count", "model_path", "model_digest",
        "gateway_port", "protocol", "authorization_env", "failure_policy", "timeout_policy",
        "circuit_policy", "vllm", "generation",
    }:
        raise ValueError("run_resources.local_judge fields are invalid")
    if judge.get("enabled") is not True:
        raise ValueError("GPU-capable run_resources require Local Judge enabled")
    if not isinstance(judge.get("authorization_env"), str) or not judge["authorization_env"].strip():
        raise ValueError("run_resources.local_judge.authorization_env is required")
    if judge.get("failure_policy") != {
        "row_failure": "deterministic_fallback",
        "job_failure": "command_failed",
        "engine_retry": False,
        "provider_fallback": False,
    }:
        raise ValueError("run_resources.local_judge.failure_policy is invalid")
    timeout = judge.get("timeout_policy")
    circuit = judge.get("circuit_policy")
    vllm = judge.get("vllm")
    generation = judge.get("generation")
    if (
        not isinstance(timeout, dict)
        or set(timeout) != {"readiness_seconds", "request_timeout_seconds"}
        or any(type(timeout[name]) not in {int, float} or timeout[name] <= 0 for name in timeout)
    ):
        raise ValueError("run_resources.local_judge.timeout_policy is invalid")
    if (
        not isinstance(circuit, dict)
        or set(circuit) != {"max_failed_row_ratio", "max_consecutive_unhealthy_jobs"}
        or type(circuit["max_failed_row_ratio"]) not in {int, float}
        or not 0 <= circuit["max_failed_row_ratio"] <= 1
        or type(circuit["max_consecutive_unhealthy_jobs"]) is not int
        or circuit["max_consecutive_unhealthy_jobs"] < 1
    ):
        raise ValueError("run_resources.local_judge.circuit_policy is invalid")
    if (
        not isinstance(vllm, dict)
        or set(vllm) != {
            "executable", "environment", "gpu_memory_utilization", "max_model_len",
            "max_num_seqs", "max_num_batched_tokens", "enable_prefix_caching",
            "reasoning_parser", "gdn_prefill_backend", "launch_stagger_seconds",
        }
        or not isinstance(vllm["executable"], str)
        or not vllm["executable"].strip()
        or not isinstance(vllm["environment"], dict)
        or any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(value, str)
            or not value.strip()
            for name, value in vllm["environment"].items()
        )
        or type(vllm["gpu_memory_utilization"]) not in {int, float}
        or not 0 < vllm["gpu_memory_utilization"] <= 1
        or any(
            type(vllm[name]) is not int or vllm[name] < 1
            for name in ("max_model_len", "max_num_seqs", "max_num_batched_tokens")
        )
        or type(vllm["enable_prefix_caching"]) is not bool
        or not isinstance(vllm["reasoning_parser"], str)
        or not vllm["reasoning_parser"].strip()
        or not isinstance(vllm["gdn_prefill_backend"], str)
        or not vllm["gdn_prefill_backend"].strip()
        or type(vllm["launch_stagger_seconds"]) not in {int, float}
        or vllm["launch_stagger_seconds"] < 0
    ):
        raise ValueError("run_resources.local_judge.vllm is invalid")
    if (
        not isinstance(generation, dict)
        or set(generation) != {
            "per_endpoint_concurrency", "max_tokens", "temperature", "top_p",
            "enable_thinking", "seed",
        }
        or type(generation["per_endpoint_concurrency"]) is not int
        or generation["per_endpoint_concurrency"] < 1
        or type(generation["max_tokens"]) is not int
        or generation["max_tokens"] < 1
        or type(generation["temperature"]) not in {int, float}
        or generation["temperature"] < 0
        or type(generation["top_p"]) not in {int, float}
        or not 0 < generation["top_p"] <= 1
        or type(generation["enable_thinking"]) is not bool
        or type(generation["seed"]) is not int
        or generation["seed"] < 0
    ):
        raise ValueError("run_resources.local_judge.generation is invalid")
    from ade.local_rubric_judge.lifecycle import LocalJudgeBinding

    binding = LocalJudgeBinding.from_dict(
        {
            "cluster_id": ray.get("cluster_id"),
            "ray_address": ray.get("address"),

            "gpu_count": judge.get("gpu_count"),
            "model_path": judge.get("model_path"),
            "model_digest": judge.get("model_digest"),
            "gateway_port": judge.get("gateway_port"),
            "protocol": judge.get("protocol"),
        }
    )
    return {
        "ray_cluster": dict(ray),
        "local_judge": {
            **dict(judge),
            "gpu_count": binding.gpu_count,
        },
    }
