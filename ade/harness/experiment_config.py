"""Compile layered operator experiment configs into the current ADE boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import secrets
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ade.core.run import PortfolioState
from ade.engine.execution.coordinator_resources import resolved_resource_policy
from ade.tasks.data_selection.llamafactory_prompt_protocol import (
    native_tokenizer_prompt_metadata,
    resolve_sft_prompt_contract,
)
from ade.harness.config import ConfigCompiler, ResolvedRunConfig
from ade.harness.runtime_roots import deployment_runtime_roots
from ade.harness.yaml_config import load_yaml_mapping
from ade.tasks.contracts import BaselineArtifactRequest
from ade.tasks.registry import TaskRegistry

_WORKFLOWS = {
    "data_selection": "sft",
    "reward_design": "rft",
    "curriculum_learning": "rft",
    "evaluation": "evaluation",
}


def generated_experiment_run_id(
    experiment_id: str,
    *,
    now: datetime | None = None,
    suffix: str | None = None,
) -> str:
    """Return the canonical ID for a newly created experiment Run."""
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    timestamp_text = timestamp.strftime("%Y%m%dT%H%M%SZ")
    random_suffix = suffix or secrets.token_hex(3)
    if len(random_suffix) != 6 or any(
        character not in "0123456789abcdef" for character in random_suffix
    ):
        raise ValueError("run ID suffix must be six lowercase hexadecimal characters")
    return f"{experiment_id}-{timestamp_text}-{random_suffix}"
_CONTROL_FIELDS = {
    "agent",
    "artifact_builder",
    "artifact_judge",
    "evaluation",
    "model_review",
    "monitor",
    "resources",
    "runtime",
    "search",
}

_EXPERIMENT_FIELDS = {
    "schema_version",
    "experiment",
    "seed",
    "task",
    "train",
    "eval",
    "analysis",
    "runtime",
    "agent",
    "artifact_builder",
    "search",
    "engine",
    "operator_evaluation",
    "tracking",
    "bootstrap",
    "judge_enrichment",
}
_SEARCH_FIELDS = {
    "coordinator_count",
    "plans_per_coordinator",
    "trials_per_plan",
}
_AGENT_FIELDS = {
    "backend",
    "model",
    "reasoning_effort",
    "sandbox_mode",
    "approval_policy",
    "max_retries",
    "heartbeat_timeout_seconds",
}
_ARTIFACT_BUILDER_FIELDS = {"max_reflections", "reward_replay"}
_REWARD_REPLAY_FIELDS = {"prompt_groups"}
_ENGINE_FIELDS = {
    "command_transport",
    "claim_timeout_seconds",
    "heartbeat_timeout_seconds",
    "automatic_recovery",
}
_AUTOMATIC_RECOVERY_FIELDS = {
    "max_attempts",
    "dependency_readiness_timeout_seconds",
}
_BOOTSTRAP_REFERENCE_FIELDS = {"enabled"}
_COMPONENT_META_FIELDS = {"schema_version", "name"}


@dataclass(frozen=True)
class ResolvedExperimentConfig:
    experiment_id: str
    config_digest: str
    run: ResolvedRunConfig
    engine_inputs: dict[str, dict[str, object]]
    control: dict[str, object]
    source_layers: tuple[str, ...]
    bootstrap: dict[str, object]
    resolved: dict[str, object]
    source_files: tuple[tuple[str, str, str, bytes], ...]

    def to_dict(self) -> dict[str, object]:
        return copy.deepcopy(self.resolved)

    def encode(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()

    def publication_files(self) -> tuple[tuple[str, bytes], ...]:
        encode_json = lambda value: (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        files: list[tuple[str, bytes]] = [
            ("resolved.json", self.encode()),
            (
                "task-mutation-contract.json",
                encode_json(self.resolved["task_mutation_contract"]),
            ),
            ("source-manifest.json", encode_json(self.resolved["source_manifest"])),
        ]
        files.extend(
            (copy_path, content)
            for _role, _ref, copy_path, content in self.source_files
        )
        return tuple(files)


class AgentSearchStrategyCompiler:
    """Named compiler boundary for Agent Search's implicit bootstrap."""

    def __init__(self, compiler: "ExperimentConfigCompiler") -> None:
        self.compiler = compiler

    def compile(self, **kwargs: object) -> dict[str, object]:
        return self.compiler._bootstrap_contract(**kwargs)


class ExperimentConfigCompiler:
    def __init__(
        self,
        tasks: TaskRegistry,
        *,
        project_root: str | Path,
    ) -> None:
        self.tasks = tasks
        self.project_root = Path(project_root).resolve()
        self.run_compiler = ConfigCompiler(tasks)
        self.strategy_compiler = AgentSearchStrategyCompiler(self)
        self._path_digest_cache: dict[str, str] = {}
        self._chat_template_cache: dict[str, dict[str, str]] = {}

    def resolve_standalone_evaluation_contract(
        self,
        *,
        task_type: str,
        task_config: str,
        dataset_ids: list[str],
    ) -> dict[str, object]:
        """Resolve task prompts and benchmark bindings through the Run compiler path."""

        if task_type not in _WORKFLOWS or task_type == "evaluation":
            raise ValueError(f"unsupported standalone task type: {task_type}")
        if not dataset_ids or len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("standalone dataset IDs must be non-empty and unique")
        configs_root = self.project_root / "configs"
        task_path = self._resolve_config(configs_root, task_config)
        task_document = self._load_component(task_path, "task")
        benchmark_catalog_path = configs_root / "benchmarks/catalog.yaml"
        prompt_catalog_path = configs_root / "prompts/evaluation.yaml"
        benchmark_catalog = self._load_catalog(
            benchmark_catalog_path,
            "benchmark catalog",
            "benchmarks",
        )
        prompt_catalog = self._load_catalog(
            prompt_catalog_path,
            "evaluation prompt catalog",
            "prompts",
        )
        resolved_task = self._resolved_task_config(
            task_type,
            self._without_meta(task_document),
            benchmark_catalog=benchmark_catalog,
            prompt_catalog=prompt_catalog,
        )
        prompt_protocol = self._mapping(
            resolved_task.get("prompt_protocol"),
            "resolved task prompt_protocol",
        )
        datasets: dict[str, dict[str, Any]] = {}
        for dataset_id in dataset_ids:
            entry = self._mapping(
                benchmark_catalog.get(dataset_id),
                f"benchmark {dataset_id}",
            )
            usage = entry.get("usage")
            if usage not in {"validation", "test"}:
                raise ValueError(
                    f"benchmark {dataset_id} usage must be validation or test"
                )
            binding = self._dataset_binding(
                dataset_id=dataset_id,
                usage=str(usage),
                benchmark_catalog=benchmark_catalog,
                prompt_protocol=prompt_protocol,
            )
            artifact = self._mapping(
                entry.get("artifact"),
                f"benchmark {dataset_id}.artifact",
            )
            binding["expected_rows"] = self._positive_int(
                artifact,
                "expected_rows",
                f"benchmark {dataset_id}.artifact",
            )
            datasets[dataset_id] = binding
        return {
            "task_type": task_type,
            "task": resolved_task,
            "datasets": datasets,
            "source_paths": (
                task_path,
                benchmark_catalog_path,
                prompt_catalog_path,
                *(path for _role, path, _copy in self._prompt_source_items(prompt_catalog)),
            ),
        }

    def resolve_model_protocol(
        self,
        value: object,
        *,
        owner: str = "model_protocol",
    ) -> dict[str, object]:
        """Expose the canonical model protocol validator to standalone evals."""

        return self._model_protocol(value, owner)

    def compile_file(
        self,
        config_path: str | Path,
        *,
        deployment_config: str | Path,
        run_id: str | None = None,
    ) -> ResolvedExperimentConfig:
        path = Path(config_path).resolve()
        deployment_path = Path(deployment_config)
        if not deployment_path.is_absolute():
            deployment_path = self.project_root / deployment_path
        deployment_path = deployment_path.resolve()
        configs_root = self._configs_root(path)
        benchmark_catalog_path = configs_root / "benchmarks/catalog.yaml"
        prompt_catalog_path = configs_root / "prompts/evaluation.yaml"
        benchmark_catalog = self._load_catalog(
            benchmark_catalog_path,
            "benchmark catalog",
            "benchmarks",
        )
        prompt_catalog = self._load_catalog(
            prompt_catalog_path,
            "evaluation prompt catalog",
            "prompts",
        )
        source = self._load_mapping(path)
        self._strict_fields(
            source,
            _EXPERIMENT_FIELDS,
            _EXPERIMENT_FIELDS,
            "experiment",
        )
        self._schema_one(source, "experiment")
        experiment_id = self._required_text(source, "experiment", "experiment")
        selected_run_id = run_id or experiment_id
        seed = self._non_negative_int(source, "seed", "experiment")
        judge_enrichment = self._mapping(
            source.get("judge_enrichment"), "judge_enrichment"
        )
        self._strict_fields(
            judge_enrichment, {"enabled"}, {"enabled"}, "judge_enrichment"
        )
        if type(judge_enrichment["enabled"]) is not bool:
            raise ValueError("judge_enrichment.enabled must be boolean")
        task_ref = self._mapping(source.get("task"), "task")
        self._strict_fields(task_ref, {"type", "spec"}, {"type", "spec"}, "task")
        task_type = self._required_text(task_ref, "type", "task")
        if task_type not in _WORKFLOWS or task_type == "evaluation":
            raise ValueError(f"unsupported task type: {task_type}")
        task_path = self._resolve_config(
            configs_root,
            self._required_text(task_ref, "spec", "task"),
        )
        task_document = self._load_component(task_path, "task")
        task_payload = self._without_meta(task_document)

        components: dict[str, tuple[Path, dict[str, Any]]] = {}
        for role, section in (
            ("train", "train"),
            ("eval", "eval"),
            ("analysis", "analysis"),
            ("runtime", "runtime"),
        ):
            component_path = self._resolve_config(
                configs_root,
                self._required_text(source, role, "experiment"),
            )
            document = self._load_component(component_path, role)
            components[role] = (component_path, self._section(document, section))
        train_payload = components["train"][1]
        eval_payload = components["eval"][1]
        analysis_payload = components["analysis"][1]
        runtime_payload = components["runtime"][1]
        deployment_document = self._load_component(deployment_path, "deployment")
        deployment_id = self._required_text(
            deployment_document, "name", "deployment"
        )
        run_resources = self._section(deployment_document, "run_resources")
        run_resources["local_judge"]["model_path"] = self._project_path(
            str(run_resources["local_judge"]["model_path"])
        )
        ray_cluster = self._mapping(
            run_resources.get("ray_cluster"),
            "deployment.run_resources.ray_cluster",
        )
        if ray_cluster.get("cluster_id") != deployment_id:
            raise ValueError(
                "deployment name must equal run_resources.ray_cluster.cluster_id"
            )
        components["deployment"] = (deployment_path, run_resources)
        analysis_policy = self._resolved_analysis_policy(analysis_payload)

        search = self._mapping(source.get("search"), "search")
        self._strict_fields(search, _SEARCH_FIELDS, _SEARCH_FIELDS, "search")
        coordinator_count = self._positive_int(search, "coordinator_count", "search")
        plans_per_coordinator = self._positive_int(
            search,
            "plans_per_coordinator",
            "search",
        )
        max_plans = coordinator_count * plans_per_coordinator
        trials_per_plan = self._positive_int(search, "trials_per_plan", "search")
        max_trials = max_plans * trials_per_plan

        agent = self._mapping(source.get("agent"), "agent")
        self._strict_fields(agent, _AGENT_FIELDS, _AGENT_FIELDS, "agent")
        self._required_text(agent, "backend", "agent")
        self._required_text(agent, "model", "agent")
        self._required_text(agent, "reasoning_effort", "agent")
        if agent["sandbox_mode"] != "danger-full-access":
            raise ValueError("agent.sandbox_mode must be danger-full-access")
        if agent["approval_policy"] != "never":
            raise ValueError("agent.approval_policy must be never")
        self._non_negative_int(agent, "max_retries", "agent")
        self._positive_int(
            agent, "heartbeat_timeout_seconds", "agent"
        )

        artifact_builder = self._mapping(
            source.get("artifact_builder"), "artifact_builder"
        )
        required_builder_fields = (
            _ARTIFACT_BUILDER_FIELDS
            if task_type == "reward_design"
            else {"max_reflections"}
        )
        self._strict_fields(
            artifact_builder,
            _ARTIFACT_BUILDER_FIELDS,
            required_builder_fields,
            "artifact_builder",
        )
        if self._positive_int(
            artifact_builder,
            "max_reflections",
            "artifact_builder",
        ) != 2:
            raise ValueError("artifact_builder.max_reflections must be 2")
        if task_type == "reward_design":
            reward_replay = self._mapping(
                artifact_builder.get("reward_replay"),
                "artifact_builder.reward_replay",
            )
            self._strict_fields(
                reward_replay,
                _REWARD_REPLAY_FIELDS,
                _REWARD_REPLAY_FIELDS,
                "artifact_builder.reward_replay",
            )
            if self._positive_int(
                reward_replay,
                "prompt_groups",
                "artifact_builder.reward_replay",
            ) != 32:
                raise ValueError(
                    "artifact_builder.reward_replay.prompt_groups must be 32"
                )
        elif "reward_replay" in artifact_builder:
            raise ValueError(
                "artifact_builder.reward_replay only applies to Reward Design"
            )

        engine = self._mapping(source.get("engine"), "engine")
        self._strict_fields(
            engine,
            _ENGINE_FIELDS,
            {"claim_timeout_seconds", "heartbeat_timeout_seconds"},
            "engine",
        )
        self._positive_int(engine, "claim_timeout_seconds", "engine")
        self._positive_int(engine, "heartbeat_timeout_seconds", "engine")
        automatic_recovery = self._mapping(
            engine.get("automatic_recovery"),
            "engine.automatic_recovery",
        )
        self._strict_fields(
            automatic_recovery,
            _AUTOMATIC_RECOVERY_FIELDS,
            _AUTOMATIC_RECOVERY_FIELDS,
            "engine.automatic_recovery",
        )
        self._positive_int(
            automatic_recovery,
            "max_attempts",
            "engine.automatic_recovery",
        )
        self._positive_int(
            automatic_recovery,
            "dependency_readiness_timeout_seconds",
            "engine.automatic_recovery",
        )
        tracking = self._resolved_tracking_policy(source.get("tracking"))
        # A concrete W&B project is derived at runtime from the root Run
        # identity.  Do not encode the task name in the resolved policy.
        artifact_staging = self._resolved_artifact_staging(run_resources)

        self._validate_runtime(runtime_payload, coordinator_count)
        judge_gpus = self._positive_int(
            self._mapping(run_resources.get("local_judge"), "local_judge"),
            "gpu_count", "local_judge",
        )
        workload_gpus = coordinator_count * int(runtime_payload["coordinator_capacity_gpus"])
        if int(runtime_payload["cluster"]["total_gpus"]) < workload_gpus + judge_gpus:
            raise ValueError("runtime.cluster.total_gpus must cover Coordinators plus Ray Judge GPUs")

        self._validate_eval(eval_payload)
        self._validate_canonical_lengths(task_type, train_payload, eval_payload)
        self._validate_standard_math_grpo(train_payload)
        if task_type == "curriculum_learning":
            self._validate_curriculum_rft(train_payload)
        self._reject_duplicate_authority(train_payload, eval_payload)
        task_config = self._resolved_task_config(
            task_type,
            task_payload,
            benchmark_catalog=benchmark_catalog,
            prompt_catalog=prompt_catalog,
        )
        # Reward Design Agents consume the resolved task view.  The Group
        # Credit mechanism and semantic cadence are train-profile controls,
        # but they are part of the Builder's contract and must be projected
        # into that view as well as the Engine input.
        if task_type == "reward_design":
            task_rft = self._mapping(task_config.get("rft"), "resolved task rft")
            train_rft = self._mapping(
                self._mapping(train_payload, "train").get("rft"),
                "train.rft",
            )
            for field in ("group_credit", "semantic_evidence_interval"):
                if field in train_rft:
                    task_rft[field] = copy.deepcopy(train_rft[field])
            task_config["rft"] = task_rft
        if task_type == "curriculum_learning":
            task_rft = self._mapping(task_config.get("rft"), "resolved task rft")
            train_rft = self._mapping(
                self._mapping(train_payload, "train").get("rft"),
                "train.rft",
            )
            for field in (
                "total_training_steps",
                "train_batch_size",
                "gen_batch_size",
                "rollout_n",
                "max_prompt_length",
                "data_shuffle",
                "group_credit",
                "semantic_evidence_interval",
            ):
                if field in train_rft:
                    task_rft[field] = copy.deepcopy(train_rft[field])
            task_config["rft"] = task_rft
            task_config["judge_enrichment"] = {
                "enabled": judge_enrichment["enabled"],
                "owner": "curriculum_realization",
            }
        else:
            task_config["judge_enrichment"] = copy.deepcopy(judge_enrichment)
        run_payload = {
            "schema_version": 1,
            "run": {
                "id": selected_run_id,
                "max_plans": max_plans,
                "max_trials": max_trials,
            },
            "task": {"id": task_type, "config": task_config},
            "analysis_policy": analysis_policy,
        }
        resolved_resources = copy.deepcopy(run_resources)
        resolved_resources.pop("artifact_staging", None)
        resolved_resources["local_judge"]["generation"]["seed"] = seed
        run_payload["run_resources"] = resolved_resources
        run = self.run_compiler.compile(run_payload)
        run = replace(
            run,
            portfolio=PortfolioState(
                max_plans=max_plans,
                max_trials=max_trials,
                min_trials_per_plan=trials_per_plan,
                max_trials_per_plan=trials_per_plan,
            ),
        )
        engine_input = self._engine_input(
            task_type=task_type,
            run_id=selected_run_id,
            run_root=deployment_runtime_roots(
                self.project_root, {"deployment": {"id": deployment_id}}
            )["engine_work"] / selected_run_id,
            task=task_config,
            train=train_payload,
            evaluation=eval_payload,
            runtime=runtime_payload,
            seed=seed,
            tracking=tracking,
            artifact_staging=artifact_staging,
            run_resources=run_resources,
        )
        operator_policy = self._mapping(
            source.get("operator_evaluation"), "operator_evaluation"
        )
        self._strict_fields(
            operator_policy,
            {"enabled"},
            {"enabled"},
            "operator_evaluation",
        )
        if operator_policy["enabled"] is not True:
            raise ValueError("operator_evaluation.enabled must be true")
        control = {
            "agent": copy.deepcopy(agent),
            "artifact_builder": copy.deepcopy(artifact_builder),
            "search": {
                "coordinator_count": coordinator_count,
                "plans_per_coordinator": plans_per_coordinator,
                "max_plans": max_plans,
                "trials_per_plan": trials_per_plan,
                "max_search_trials": max_trials,
            },
            "engine": copy.deepcopy(engine),
            "operator_evaluation": copy.deepcopy(operator_policy),
            "tracking": copy.deepcopy(tracking),
            "runtime": copy.deepcopy(runtime_payload),
            "analysis": copy.deepcopy(analysis_policy),
            "judge_enrichment": copy.deepcopy(judge_enrichment),
        }
        bootstrap_policy = self._mapping(source.get("bootstrap"), "bootstrap")
        self._strict_fields(
            bootstrap_policy,
            {"enabled", "stop_after_baseline", "reference"},
            {"enabled"},
            "bootstrap",
        )
        if bootstrap_policy["enabled"] is not True:
            raise ValueError("bootstrap.enabled must be true for an ADE Run")
        bootstrap = self.strategy_compiler.compile(
            strategy={"name": "agent_search"},
            run_id=selected_run_id,
            task_type=task_type,
            task=task_config,
            task_config=task_config,
            train=train_payload,
            evaluation=eval_payload,
            total_trial_budget=max_trials,
            engine_input=engine_input,
            seed=seed,
            tracking=tracking,
            artifact_staging=artifact_staging,
        )
        # This is a shared lifecycle control: baseline-only runs terminate at
        # the normal bootstrap completion boundary without entering search.
        bootstrap["stop_after_baseline"] = bool(
            bootstrap_policy.get("stop_after_baseline", False)
        )
        reference_policy = self._mapping(
            bootstrap_policy.get("reference") or {"enabled": False},
            "bootstrap.reference",
        )
        self._strict_fields(
            reference_policy,
            _BOOTSTRAP_REFERENCE_FIELDS,
            _BOOTSTRAP_REFERENCE_FIELDS,
            "bootstrap.reference",
        )
        if type(reference_policy["enabled"]) is not bool:
            raise ValueError("bootstrap.reference.enabled must be boolean")
        bootstrap["reference"] = {"enabled": reference_policy["enabled"]}
        bootstrap["reference_enabled"] = bool(reference_policy["enabled"])
        source_items = [
            ("experiment", path, "sources/experiment.yaml"),
            ("task", task_path, "sources/task.yaml"),
            *(
                (role, components[role][0], f"sources/{role}.yaml")
                for role in ("train", "eval", "analysis", "runtime")
            ),
            ("deployment", deployment_path, "sources/deployment.yaml"),
            (
                "benchmark_catalog",
                benchmark_catalog_path,
                "sources/benchmark_catalog.yaml",
            ),
            (
                "prompt_catalog",
                prompt_catalog_path,
                "sources/prompt_catalog.yaml",
            ),
            *self._prompt_source_items(prompt_catalog),
        ]
        source_files = tuple(
            (
                role,
                self._portable(item_path, configs_root=configs_root),
                copy_path,
                item_path.read_bytes(),
            )
            for role, item_path, copy_path in source_items
        )
        source_manifest = [
            {
                "role": role,
                "ref": ref,
                "copy": copy_path,
                "digest": hashlib.sha256(content).hexdigest(),
            }
            for role, ref, copy_path, content in source_files
        ]
        mutation_contract = {
            "schema_version": 1,
            "task_type": task_type,
            "mutable_artifact": {
                "data_selection": "selection.py",
                "reward_design": "reward.py",
                "curriculum_learning": "curriculum.py",
            }[task_type],
        }
        resolved_evaluation = copy.deepcopy(eval_payload)
        resolved_data = self._mapping(task_config.get("data"), "resolved task data")
        resolved_evaluation["dataset_bindings"] = {
            purpose: copy.deepcopy(resolved_data[purpose])
            for purpose in ("validation", "test")
        }
        published_task_config = copy.deepcopy(task_payload)
        published_task_config["prompt_protocol"] = copy.deepcopy(
            task_config["prompt_protocol"]
        )
        if task_type in {"reward_design", "curriculum_learning"}:
            published_task_config["rft"] = copy.deepcopy(task_config["rft"])
        if task_type == "curriculum_learning":
            published_task_config["judge_enrichment"] = copy.deepcopy(
                task_config["judge_enrichment"]
            )
        resolved_payload = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "seed": seed,
            "task": {"type": task_type, "config": published_task_config},
            "agent": control["agent"],
            "artifact_builder": control["artifact_builder"],
            "search": control["search"],
            "engine": control["engine"],
            "operator_evaluation": control["operator_evaluation"],
            "analysis": analysis_policy,
            "judge_enrichment": copy.deepcopy(judge_enrichment),
            "training": copy.deepcopy(train_payload),
            "evaluation": resolved_evaluation,
            "runtime": runtime_payload,
            "tracking": control["tracking"],
            "bootstrap": copy.deepcopy(bootstrap_policy),
            "task_mutation_contract": mutation_contract,
            "source_manifest": source_manifest,
        }
        if task_type == "data_selection":
            resolved_payload["training"]["prompt_contract"] = copy.deepcopy(
                engine_input["sft"]["request"]["training_prompt_contract"]
            )
        resolved_payload["deployment"] = {
            "id": deployment_id,
            "run_resources": copy.deepcopy(run_resources),
        }
        if run.run_resources is not None:
            resolved_payload["run_resources"] = copy.deepcopy(run.run_resources)
        digest_payload = copy.deepcopy(resolved_payload)
        digest_payload["evaluation"] = self._digest_evaluation(
            resolved_evaluation
        )
        digest = hashlib.sha256(
            json.dumps(
                digest_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        resolved = copy.deepcopy(resolved_payload)
        resolved["config_digest"] = digest
        return ResolvedExperimentConfig(
            experiment_id=experiment_id,
            config_digest=digest,
            run=run,
            engine_inputs={task_type: engine_input},
            control=control,
            source_layers=tuple(ref for _, ref, _copy_path, _content in source_files),
            bootstrap=bootstrap,
            resolved=resolved,
            source_files=source_files,
        )

    def _resolved_analysis_policy(
        self,
        analysis: dict[str, Any],
    ) -> dict[str, object]:
        self._strict_fields(
            analysis,
            {
                "profile_id",
                "review_required",
                "evidence_view",
                "default_on_exhaustion",
                "provider",
            },
            {
                "profile_id",
                "review_required",
                "evidence_view",
                "default_on_exhaustion",
                "provider",
            },
            "analysis",
        )
        self._required_text(analysis, "profile_id", "analysis")
        if analysis["review_required"] is not True:
            raise ValueError("analysis.review_required must be true")
        if analysis["evidence_view"] != "canonical":
            raise ValueError("analysis.evidence_view must be canonical")
        if analysis["default_on_exhaustion"] is not True:
            raise ValueError("analysis.default_on_exhaustion must be true")
        provider = self._mapping(analysis["provider"], "analysis.provider")
        provider_fields = {
            "type",
            "temperature",
            "enable_thinking",
            "thinking_budget",
            "max_completion_tokens",
            "max_concurrent_requests",
            "request_timeout_seconds",
            "batch_timeout_seconds",
            "max_attempts",
            "retry_delay_seconds",
        }
        self._strict_fields(
            provider,
            provider_fields,
            provider_fields,
            "analysis.provider",
        )
        self._required_text(provider, "type", "analysis.provider")
        if provider["type"] != "local_analyzer":
            raise ValueError("analysis.provider.type must be local_analyzer")
        if type(provider["temperature"]) not in {int, float} or provider["temperature"] < 0:
            raise ValueError("analysis.provider.temperature must be non-negative")
        if type(provider["enable_thinking"]) is not bool:
            raise ValueError("analysis.provider.enable_thinking must be boolean")
        for field in (
            "thinking_budget",
            "max_completion_tokens",
            "max_concurrent_requests",
            "request_timeout_seconds",
            "max_attempts",
        ):
            self._positive_int(provider, field, "analysis.provider")
        batch_timeout = provider["batch_timeout_seconds"]
        if batch_timeout is not None and (
            type(batch_timeout) is not int or batch_timeout < 1
        ):
            raise ValueError(
                "analysis.provider.batch_timeout_seconds must be null or a positive integer"
            )
        if (
            type(provider["retry_delay_seconds"]) not in {int, float}
            or provider["retry_delay_seconds"] < 0
        ):
            raise ValueError("analysis.provider.retry_delay_seconds must be non-negative")
        resolved = copy.deepcopy(analysis)
        digest = hashlib.sha256(
            json.dumps(
                resolved,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        resolved["policy_digest"] = digest
        return resolved

    def _resolved_tracking_policy(self, value: object) -> dict[str, object]:
        tracking = self._mapping(value, "tracking")
        fields = {
            "provider",
            "enabled",
            "mode",
            "entity_env",
            "base_url_env",
            "api_key_env",
        }
        self._strict_fields(tracking, fields, fields, "tracking")
        if tracking["provider"] != "wandb":
            raise ValueError("tracking.provider must be wandb")
        if tracking["enabled"] is not True:
            raise ValueError("tracking.enabled must be true")
        if tracking["mode"] not in {"online", "offline"}:
            raise ValueError("tracking.mode must be online or offline")
        for field in ("entity_env", "base_url_env", "api_key_env"):
            self._required_text(tracking, field, "tracking")
        return copy.deepcopy(tracking)

    def _resolved_task_config(
        self,
        task_type: str,
        task: dict[str, Any],
        *,
        benchmark_catalog: dict[str, Any],
        prompt_catalog: dict[str, Any],
    ) -> dict[str, object]:
        if "material" in task:
            raise ValueError("task.material is no longer supported; use task.data")
        resolved = copy.deepcopy(task)
        resolved["base_model"] = self._existing_project_path(
            self._required_text(resolved, "base_model", "task"),
            "task.base_model",
        )
        resolved["base_model_protocol"] = self._model_protocol(
            resolved.get("base_model_protocol"),
            "task.base_model_protocol",
        )
        prompt_protocol = self._resolved_prompt_protocol(
            resolved.get("prompt_protocol"),
            model_path=resolved["base_model"],
            prompt_catalog=prompt_catalog,
        )
        resolved["prompt_protocol"] = prompt_protocol
        data = self._mapping(resolved.get("data"), "task data")
        allowed_data_fields = (
            {
                "fixed_training_data",
                "pool_size",
                "dataset_name",
                "validation",
                "test",
            }
            if task_type == "data_selection"
            else {"name", "train", "validation", "test"}
        )
        required_data_fields = (
            allowed_data_fields
            if task_type == "data_selection"
            else {"name", "train", "validation", "test"}
        )
        self._strict_fields(data, allowed_data_fields, required_data_fields, "task.data")
        fields = (
            ("fixed_training_data",)
            if task_type == "data_selection"
            else ("train",)
        )
        for field in fields:
            data[field] = self._existing_project_path(
                self._required_text(data, field, "task.data"),
                f"task.data.{field}",
            )
        if task_type == "data_selection":
            pool_size = self._positive_int(data, "pool_size", "task.data")
            if pool_size < 1:
                raise ValueError("task.data.pool_size must be positive")
        for collection in ("validation", "test"):
            values = data.get(collection)
            if not isinstance(values, list) or not values or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise ValueError(
                    f"task.data.{collection} must be a non-empty list of canonical dataset IDs"
                )
            if len(values) != len(set(values)):
                raise ValueError(f"task.data.{collection} contains duplicate dataset IDs")
            data[collection] = [
                self._dataset_binding(
                    dataset_id=value.strip(),
                    usage=collection,
                    benchmark_catalog=benchmark_catalog,
                    prompt_protocol=prompt_protocol,
                )
                for value in values
            ]
        resolved["data"] = data
        return resolved

    def _dataset_binding(
        self,
        *,
        dataset_id: str,
        usage: str,
        benchmark_catalog: dict[str, Any],
        prompt_protocol: dict[str, Any],
    ) -> dict[str, Any]:
        entry = benchmark_catalog.get(dataset_id)
        if not isinstance(entry, dict):
            raise ValueError(f"unknown canonical benchmark ID: {dataset_id}")
        required = {"domain", "usage", "artifact", "task", "protocol"}
        missing = required - set(entry)
        if missing:
            raise ValueError(
                f"benchmark {dataset_id} is missing binding fields: {sorted(missing)}"
            )
        expected_usage = "validation" if usage == "validation" else "test"
        if entry.get("usage") != expected_usage:
            raise ValueError(
                f"benchmark {dataset_id} usage must be {expected_usage!r}"
            )
        domain = self._required_text(entry, "domain", f"benchmark {dataset_id}")
        task_id = self._required_text(entry, "task", f"benchmark {dataset_id}")
        if task_id != dataset_id:
            raise ValueError(
                f"benchmark {dataset_id} must bind its canonical same-name task"
            )
        protocol_id = self._required_text(
            entry,
            "protocol",
            f"benchmark {dataset_id}",
        )
        profiles = self._mapping(
            entry.get("profiles"),
            f"benchmark {dataset_id}.profiles",
        )
        evaluation_k: dict[str, int] = {}
        for purpose, profile_name in (
            ("online_validation", "online_eval"),
            ("offline_validation", "offline_validation"),
            ("operator_test", "operator_test"),
        ):
            profile = self._mapping(
                profiles.get(profile_name),
                f"benchmark {dataset_id}.profiles.{profile_name}",
            )
            evaluation_k[purpose] = self._positive_int(
                profile,
                "num_samples",
                f"benchmark {dataset_id}.profiles.{profile_name}",
            )
        artifact = self._mapping(
            entry.get("artifact"),
            f"benchmark {dataset_id}.artifact",
        )
        artifact_path = self._existing_project_path(
            self._required_text(
                artifact,
                "path",
                f"benchmark {dataset_id}.artifact",
            ),
            f"benchmark {dataset_id}.artifact.path",
        )
        split = self._required_text(
            artifact,
            "split",
            f"benchmark {dataset_id}.artifact",
        )
        artifact_digest = artifact.get("digest")
        if artifact_digest is None:
            artifact_digest = self._path_digest(Path(artifact_path))
        else:
            artifact_digest = self._required_text(
                artifact,
                "digest",
                f"benchmark {dataset_id}.artifact",
            )
            if len(artifact_digest) != 64 or any(
                character not in "0123456789abcdef"
                for character in artifact_digest
            ):
                raise ValueError(
                    f"benchmark {dataset_id}.artifact.digest must be a SHA-256 digest"
                )
        return {
            "name": dataset_id,
            "ranking_name": str(entry.get("derived_from") or dataset_id),
            "path": artifact_path,
            "split": split,
            "domain": domain,
            "task_type": task_id,
            "protocol": {
                "id": protocol_id,
                "user_prompt_builder": f"{task_id}.user_prompt.v1",
                "reference_extractor": f"{task_id}.reference.v1",
                "prediction_extractor": f"{task_id}.prediction.v1",
                "grader": f"{task_id}.grader.v1",
                "metric": "accuracy_avg",
            },
            "evaluation_k": evaluation_k,
            "artifact_digest": artifact_digest,
            "prompt_protocol": copy.deepcopy(prompt_protocol),
        }

    def _resolved_prompt_protocol(
        self,
        value: object,
        *,
        model_path: object,
        prompt_catalog: dict[str, Any],
    ) -> dict[str, Any]:
        protocol = self._mapping(value, "task.prompt_protocol")
        mode = self._required_text(protocol, "mode", "task.prompt_protocol")
        if mode == "raw_completion":
            self._strict_fields(
                protocol,
                {"mode"},
                {"mode"},
                "task.prompt_protocol",
            )
            return {"mode": "raw_completion"}
        if mode != "chat_template":
            raise ValueError(
                "task.prompt_protocol.mode must be raw_completion or chat_template"
            )
        self._strict_fields(
            protocol,
            {"mode", "system_prompt"},
            {"mode", "system_prompt"},
            "task.prompt_protocol",
        )
        prompt_id = self._required_text(
            protocol,
            "system_prompt",
            "task.prompt_protocol",
        )
        return {
            "mode": "chat_template",
            "system_prompt": self._prompt_binding_by_id(
                prompt_id=prompt_id,
                prompt_catalog=prompt_catalog,
            ),
            "chat_template": self._chat_template_binding(model_path),
        }

    def _prompt_binding_by_id(
        self,
        *,
        prompt_id: str,
        prompt_catalog: dict[str, Any],
    ) -> dict[str, str]:
        matches: list[tuple[str, str, dict[str, Any]]] = []
        for workflow, raw_entries in sorted(prompt_catalog.items()):
            entries = self._mapping(raw_entries, f"prompt catalog.{workflow}")
            for name, raw_entry in sorted(entries.items()):
                entry = self._mapping(
                    raw_entry,
                    f"prompt catalog.{workflow}.{name}",
                )
                if entry.get("id") == prompt_id:
                    matches.append((workflow, name, entry))
        if len(matches) != 1:
            raise ValueError(
                f"task.prompt_protocol.system_prompt must identify exactly one "
                f"catalog prompt; {prompt_id!r} matched {len(matches)}"
            )
        workflow, name, entry = matches[0]
        owner = f"prompt catalog.{workflow}.{name}"
        self._strict_fields(entry, {"id", "path"}, {"id", "path"}, owner)
        prompt_path = Path(
            self._existing_project_path(
                self._required_text(entry, "path", owner),
                f"{owner}.path",
            )
        )
        content = prompt_path.read_text(encoding="utf-8").rstrip("\n")
        if not content:
            raise ValueError(f"{owner} must resolve to non-empty content")
        return {
            "id": prompt_id,
            "content": content,
            "digest": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }

    def _prompt_source_items(
        self,
        prompt_catalog: dict[str, Any],
    ) -> tuple[tuple[str, Path, str], ...]:
        result: list[tuple[str, Path, str]] = []
        for workflow, raw_entries in sorted(prompt_catalog.items()):
            entries = self._mapping(
                raw_entries,
                f"prompt catalog.{workflow}",
            )
            for name, raw_entry in sorted(entries.items()):
                entry = self._mapping(
                    raw_entry,
                    f"prompt catalog.{workflow}.{name}",
                )
                relative = self._required_text(
                    entry,
                    "path",
                    f"prompt catalog.{workflow}.{name}",
                )
                path = Path(
                    self._existing_project_path(
                        relative,
                        f"prompt catalog.{workflow}.{name}.path",
                    )
                )
                result.append(
                    (
                        f"prompt_asset_{workflow}_{name}",
                        path,
                        f"sources/{Path(relative).as_posix()}",
                    )
                )
        return tuple(result)

    def _digest_evaluation(
        self,
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        portable = copy.deepcopy(evaluation)
        bindings = portable.get("dataset_bindings")
        if not isinstance(bindings, dict):
            return portable
        for datasets in bindings.values():
            if not isinstance(datasets, list):
                continue
            for dataset in datasets:
                if not isinstance(dataset, dict) or "path" not in dataset:
                    continue
                digest = dataset.get("artifact_digest")
                dataset["path"] = (
                    f"artifact:{digest}"
                    if isinstance(digest, str) and digest
                    else Path(str(dataset["path"])).name
                )
        return portable

    def _chat_template_binding(self, model_path: object) -> dict[str, str]:
        path = str(model_path)
        cached = self._chat_template_cache.get(path)
        if cached is not None:
            return copy.deepcopy(cached)
        template = native_tokenizer_prompt_metadata(path)["chat_template"]
        binding = {
            "id": "native_chat_template.v1",
            "digest": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        }
        self._chat_template_cache[path] = binding
        return copy.deepcopy(binding)

    def _path_digest(self, path: Path) -> str:
        resolved = str(path.resolve())
        cached = self._path_digest_cache.get(resolved)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        if path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        elif path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.is_file())
            if not files:
                raise ValueError(f"benchmark artifact directory is empty: {path}")
            for item in files:
                digest.update(str(item.relative_to(path)).encode("utf-8"))
                digest.update(b"\0")
                with item.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        else:
            raise ValueError(f"benchmark artifact does not exist: {path}")
        value = digest.hexdigest()
        self._path_digest_cache[resolved] = value
        return value

    def _validate_runtime(
        self,
        runtime: dict[str, Any],
        coordinator_count: int,
    ) -> None:
        required = {
            "engine_transport",
            "cluster",
            "coordinator_capacity_gpus",
            "monitoring",
            "allocations",
            "concurrency",
            "backends",
        }
        self._strict_fields(runtime, required, required, "runtime")
        cluster = self._mapping(runtime["cluster"], "runtime.cluster")
        self._strict_fields(
            cluster,
            {"total_gpus", "topology_source", "exclusive_ade_run"},
            {"total_gpus", "topology_source", "exclusive_ade_run"},
            "runtime.cluster",
        )
        total_gpus = self._positive_int(cluster, "total_gpus", "runtime.cluster")
        if cluster["topology_source"] != "ray_preflight":
            raise ValueError("runtime.cluster.topology_source must be ray_preflight")
        if cluster["exclusive_ade_run"] is not True:
            raise ValueError("runtime.cluster.exclusive_ade_run must be true")
        capacity = self._positive_int(
            runtime, "coordinator_capacity_gpus", "runtime"
        )
        allocations = self._mapping(runtime["allocations"], "runtime.allocations")
        allocation_names = {
            "training",
            "online_validation",
            "offline_validation",
            "operator_evaluation",
        }
        self._strict_fields(
            allocations,
            allocation_names,
            allocation_names,
            "runtime.allocations",
        )
        resolved_allocations = {
            name: self._allocation_gpus(runtime, name)
            for name in allocation_names
        }
        oversized = {
            name: gpus
            for name, gpus in resolved_allocations.items()
            if gpus > capacity
        }
        if oversized:
            raise ValueError(
                "runtime allocations exceed coordinator capacity: "
                f"{oversized} > {capacity}"
            )
        concurrency = self._mapping(runtime["concurrency"], "runtime.concurrency")
        concurrency_fields = {
            "training_with_online_validation",
            "offline_with_operator_evaluation",
        }
        self._strict_fields(
            concurrency,
            concurrency_fields,
            concurrency_fields,
            "runtime.concurrency",
        )
        if any(type(concurrency[name]) is not bool for name in concurrency_fields):
            raise ValueError("runtime.concurrency fields must be booleans")
        if (
            concurrency["training_with_online_validation"]
            and resolved_allocations["training"]
            + resolved_allocations["online_validation"]
            > capacity
        ):
            raise ValueError(
                "concurrent training and online validation exceed "
                "coordinator capacity"
            )
        if (
            concurrency["offline_with_operator_evaluation"]
            and resolved_allocations["offline_validation"]
            + resolved_allocations["operator_evaluation"]
            > capacity
        ):
            raise ValueError(
                "concurrent offline validation and operator evaluation exceed "
                "coordinator capacity"
            )
        if total_gpus < coordinator_count * capacity:
            raise ValueError(
                "runtime cluster cannot admit coordinator_count * "
                "coordinator_capacity_gpus"
            )

    def _resolved_artifact_staging(
        self,
        run_resources: dict[str, Any] | None,
    ) -> dict[str, object]:
        if run_resources is None or "artifact_staging" not in run_resources:
            return {"enabled": False}
        staging = self._mapping(
            run_resources.get("artifact_staging"),
            "deployment.run_resources.artifact_staging",
        )
        fields = {
            "enabled",
            "cache_dir",
            "cache_max_gb",
            "lock_stale_seconds",
        }
        self._strict_fields(
            staging,
            fields,
            fields,
            "deployment.run_resources.artifact_staging",
        )
        if staging["enabled"] is not True:
            raise ValueError("deployment artifact staging must be enabled")
        cache_dir = Path(
            self._required_text(
                staging,
                "cache_dir",
                "deployment.run_resources.artifact_staging",
            )
        )
        if not cache_dir.is_absolute():
            raise ValueError("deployment artifact staging cache_dir must be absolute")
        return {
            "enabled": True,
            "cache_dir": str(cache_dir),
            "cache_max_gb": self._positive_int(
                staging,
                "cache_max_gb",
                "deployment.run_resources.artifact_staging",
            ),
            "lock_stale_seconds": self._positive_int(
                staging,
                "lock_stale_seconds",
                "deployment.run_resources.artifact_staging",
            ),
        }

    def _validate_eval(self, evaluation: dict[str, Any]) -> None:
        allowed = {
            "enabled",
            "purposes",
            "max_model_len",
            "max_new_tokens",
            "thinking_budget",
            "reasoning_parser",
            "gpu_memory_utilization",
            "max_num_seqs",
        }
        self._strict_fields(evaluation, allowed, {"enabled", "purposes"}, "eval")
        purposes = self._mapping(evaluation["purposes"], "eval.purposes")
        self._strict_fields(
            purposes,
            {"online", "offline", "operator"},
            {"online", "offline", "operator"},
            "eval.purposes",
        )
        for name in ("online", "offline", "operator"):
            profile = self._mapping(purposes[name], f"eval.purposes.{name}")
            self._strict_fields(
                profile,
                {"samples_per_input", "decoding", "ranking"},
                {"samples_per_input", "decoding"},
                f"eval.purposes.{name}",
            )
            samples = self._positive_int(
                profile, "samples_per_input", f"eval.purposes.{name}"
            )
            decoding = self._mapping(
                profile["decoding"], f"eval.purposes.{name}.decoding"
            )
            mode = self._required_text(
                decoding, "mode", f"eval.purposes.{name}.decoding"
            )
            if mode == "greedy":
                self._strict_fields(
                    decoding,
                    {"mode"},
                    {"mode"},
                    f"eval.purposes.{name}.decoding",
                )
                if samples != 1:
                    raise ValueError("greedy decoding requires samples_per_input=1")
            elif mode == "sampling":
                self._strict_fields(
                    decoding,
                    {"mode", "temperature", "top_p"},
                    {"mode", "temperature", "top_p"},
                    f"eval.purposes.{name}.decoding",
                )
                if float(decoding["temperature"]) <= 0:
                    raise ValueError("sampling temperature must be positive")
                if not 0 < float(decoding["top_p"]) <= 1:
                    raise ValueError("sampling top_p must be in (0, 1]")
            else:
                raise ValueError(f"unsupported decoding mode: {mode}")

    def _validate_canonical_lengths(
        self,
        task_type: str,
        train: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> None:
        if self._positive_int(
            evaluation,
            "max_new_tokens",
            "eval",
        ) != 16_384:
            raise ValueError("canonical evaluation max_new_tokens must be 16384")
        max_model_len = self._positive_int(
            evaluation,
            "max_model_len",
            "eval",
        )
        if task_type == "data_selection":
            if self._positive_int(train, "cutoff_len", "train") != 16_384:
                raise ValueError("canonical SFT cutoff_len must be 16384")
            if max_model_len != 32_768:
                raise ValueError("canonical SFT eval max_model_len must be 32768")
            return
        rft = self._mapping(train.get("rft"), "train.rft")
        if self._positive_int(rft, "max_prompt_length", "train.rft") != 1_024:
            raise ValueError("canonical RFT max_prompt_length must be 1024")
        if self._positive_int(rft, "max_response_length", "train.rft") != 8_192:
            raise ValueError("canonical RFT max_response_length must be 8192")
        if max_model_len < 1_024 + 16_384:
            raise ValueError(
                "RFT eval max_model_len must cover prompt plus 16384 generated tokens"
            )

    def _validate_standard_math_grpo(self, train: dict[str, Any]) -> None:
        rft_value = train.get("rft")
        if not isinstance(rft_value, dict):
            return
        rft = rft_value
        profile = rft.get("profile")
        expected_profile = {
            "simplelr_math_grpo_standard.v1": (32, True, 5.0e-7, 0.0),
            "simplelr_math_grpo_standard_100_step.v1": (100, True, 5.0e-7, 0.0),
            "simplelr_math_grpo_curriculum_100_step.v1": (100, False, 5.0e-7, 0.0),
        }.get(profile)
        if expected_profile is None:
            return
        (
            expected_training_steps,
            expected_data_shuffle,
            expected_learning_rate,
            expected_warmup_ratio,
        ) = expected_profile
        expected = {
            "algorithm": "grpo",
            "gamma": 1.0,
            "lambda": 1.0,
            "rollout_backend": "vllm",
            "total_training_steps": expected_training_steps,
            "total_epochs": 20,
            "train_batch_size": 256,
            "gen_batch_size": 256,
            "data_shuffle": expected_data_shuffle,
            "rollout_n": 8,
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 8,
            "log_prob_micro_batch_size_per_gpu": 16,
            "max_prompt_length": 1024,
            "max_response_length": 8192,
            "learning_rate": expected_learning_rate,
            "lr_scheduler_type": "constant",
            "lr_warmup_steps_ratio": expected_warmup_ratio,
            "ppo_epochs": 1,
            "actor_shuffle": False,
            "actor_use_dynamic_bsz": False,
            "actor_max_token_len_per_gpu": 16384,
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
            "ulysses_sequence_parallel_size": 1,
            "actor_param_offload": False,
            "actor_grad_offload": False,
            "actor_optimizer_offload": False,
            "reference_param_offload": True,
            "log_prob_max_token_len_per_gpu": 16384,
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "entropy_coeff": 0.001,
            "use_kl_loss": True,
            "kl_loss_coef": 0.0001,
            "kl_loss_type": "low_var_kl",
            "kl_controller": "fixed",
            "use_kl_in_reward": False,
            "kl_coef": 0.001,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "gpu_memory_utilization": 0.75,
            "rollout_dtype": "bfloat16",
            "ignore_eos": False,
            "max_num_batched_tokens": 10216,
            "max_num_seqs": 1024,
            "enable_chunked_prefill": False,
            "enforce_eager": True,
            "free_cache_engine": True,
            "artifact_interval": 4,
            "test_freq": -1,
        }
        mismatches = {
            key: {"expected": value, "actual": rft.get(key)}
            for key, value in expected.items()
            if rft.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"{profile} drift: "
                f"{mismatches}"
            )

    def _validate_curriculum_rft(self, train: dict[str, Any]) -> None:
        rft = self._mapping(train.get("rft"), "train.rft")
        profile = self._required_text(rft, "profile", "train.rft")
        if profile not in {
            "simplelr_math_grpo_curriculum_100_step.v1",
            "simplelr_math_grpo_curriculum_probe.v1",
        }:
            raise ValueError("Curriculum Learning requires a Curriculum RFT profile")
        if rft.get("data_shuffle") is not False:
            raise ValueError("Curriculum Learning requires train.rft.data_shuffle=false")
        train_batch = self._positive_int(rft, "train_batch_size", "train.rft")
        gen_batch = self._positive_int(rft, "gen_batch_size", "train.rft")
        if train_batch != gen_batch:
            raise ValueError(
                "Curriculum Learning requires train_batch_size == gen_batch_size"
            )
        group_credit = self._mapping(rft.get("group_credit"), "train.rft.group_credit")
        if (
            group_credit.get("enabled") is not True
            or group_credit.get("entrypoint") != "assign_group_credit"
            or group_credit.get("schema_version") != "ade.group_credit.v1"
        ):
            raise ValueError("Curriculum Learning requires identity Group Credit binding")

    @staticmethod
    def _reject_duplicate_authority(
        train: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> None:
        forbidden_train = {"seed", "train_gpus", "nodes", "gpus_per_node"}
        nested = train.get("rft") if isinstance(train.get("rft"), dict) else {}
        duplicates = (set(train) | set(nested)) & forbidden_train
        if duplicates:
            raise ValueError(f"training contains Runtime/root authority: {sorted(duplicates)}")
        forbidden_eval = {
            "seed",
            "temperature",
            "top_p",
            "online_validation_avg_k",
            "offline_validation_avg_k",
            "operator_test_avg_k",
            "dataset_profiles",
            "data_parallel_shards",
            "rollout_layout",
            "reasoning_parser",
        }
        duplicates = set(evaluation) & forbidden_eval
        if duplicates:
            raise ValueError(f"evaluation contains legacy/duplicate authority: {sorted(duplicates)}")

    def _bootstrap_contract(
        self,
        *,
        strategy: object,
        run_id: str,
        task_type: str,
        task: dict[str, Any],
        task_config: dict[str, object],
        train: dict[str, Any],
        evaluation: dict[str, Any],
        total_trial_budget: int,
        engine_input: dict[str, object],
        seed: int,
        tracking: dict[str, object],
        artifact_staging: dict[str, object],
    ) -> dict[str, object]:
        del train
        if (
            not isinstance(strategy, dict)
            or strategy.get("name") != "agent_search"
        ):
            return {"enabled": False, "status": "completed"}
        if evaluation.get("enabled") is False:
            raise ValueError("Agent Search requires evaluation to be enabled")
        model, requests = self._bootstrap_runtime(
            task_type,
            engine_input,
        )
        data = self._mapping(task.get("data"), "task data")
        profiles: dict[str, object] = {
            "offline": {
                "profile": "standalone_evaluation",
                "visibility": "research",
                "request": copy.deepcopy(requests["offline_validation"]),
            },
            "operator": {
                "profile": "standalone_evaluation",
                "visibility": "human_only",
                "request": self._evaluation_request(
                    purpose="operator_test",
                    task_data=data,
                    evaluation=evaluation,
                    run_root=Path(str(requests["offline_validation"]["run_dir"])).parent,
                    runtime=self._mapping(engine_input.get("runtime"), "engine runtime"),
                    seed=seed,
                    artifact_staging=artifact_staging,
                ),
            },
        }
        if task_type in {"reward_design", "curriculum_learning"}:
            profiles["online"] = {
                "profile": "standalone_evaluation",
                "visibility": "research",
                "request": copy.deepcopy(requests["online_validation"]),
            }
        for profile in profiles.values():
            profile["request"]["evaluation_tracking"] = copy.deepcopy(tracking)
        base_model_protocol = self._model_protocol(
            task_config.get("base_model_protocol"),
            "task.base_model_protocol",
        )
        checkpoint_request = self._mapping(
            requests["offline_validation"],
            "resolved checkpoint evaluation request",
        )
        # Operator evaluation covers Base, P000, and Search Trials.  Keep the
        # shared request on the checkpoint protocol; the operator driver
        # replaces it with the explicit base protocol only for Base.
        profiles["operator"]["request"]["model_protocol"] = copy.deepcopy(
            checkpoint_request["model_protocol"]
        )
        profiles["operator"]["request"]["reasoning_parser"] = copy.deepcopy(
            checkpoint_request["reasoning_parser"]
        )
        baseline = self.tasks.get(task_type).build_baseline_artifact(
            BaselineArtifactRequest(task_config=task_config, seed=seed)
        )
        try:
            baseline_content = baseline.content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("P000 baseline artifact must be UTF-8") from error
        return {
            "enabled": True,
            "status": "pending",
            "search_total_trial_budget": total_trial_budget,
            "base": {
                "id": "base",
                "reference_only": True,
                "model": model,
                "model_protocol": copy.deepcopy(base_model_protocol),
                "evaluation_profiles": profiles,
            },
            "p000": {
                "coordinator_id": "c000",
                "plan_id": "p000",
                "trial_id": "p000-t000-baseline",
                "rankable": True,
                "counts_toward_search_budget": False,
                "artifact_builder": False,
                "artifact_judge": False,
                "artifact": {
                    "path": baseline.path,
                    "kind": baseline.kind,
                    "content": baseline_content,
                    "digest": hashlib.sha256(baseline.content).hexdigest(),
                },
            },
        }

    def _bootstrap_runtime(
        self,
        task_type: str,
        engine_input: dict[str, object],
    ) -> tuple[str, dict[str, Any]]:
        if task_type == "data_selection":
            sft = self._mapping(engine_input.get("sft"), "resolved SFT config")
            request = self._mapping(sft.get("request"), "resolved SFT request")
            evaluations = sft.get("evaluation_requests")
            if not isinstance(evaluations, dict):
                raise ValueError(
                    "Agent Search requires online and offline evaluation requests"
                )
            return (
                self._required_text(request, "model", "resolved SFT request"),
                evaluations,
            )
        if task_type in {"reward_design", "curriculum_learning"}:
            rft = self._mapping(engine_input.get("rft"), "resolved RFT config")
            runtime = self._mapping(
                rft.get("verl_config"),
                "resolved RFT runtime",
            )
            runtime_rft = self._mapping(
                runtime.get("rft"),
                "resolved RFT config",
            )
            evaluations = self._mapping(
                rft.get("evaluation_requests"),
                "resolved RFT evaluation requests",
            )
            return (
                self._required_text(runtime, "base_model", "resolved RFT runtime"),
                evaluations,
            )
        raise ValueError(f"task {task_type} does not support Agent Search bootstrap")

    def _engine_input(
        self,
        *,
        task_type: str,
        run_id: str,
        run_root: Path,
        task: dict[str, Any],
        train: dict[str, Any],
        evaluation: dict[str, Any],
        runtime: dict[str, Any],
        seed: int,
        tracking: dict[str, object],
        artifact_staging: dict[str, object],
        run_resources: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if task_type == "data_selection":
            return self._sft_input(
                run_id, task, train, evaluation, runtime, seed, tracking,
                artifact_staging, run_root, run_resources,
            )
        if task_type in {"reward_design", "curriculum_learning"}:
            return self._rft_input(
                task_type,
                run_id,
                task,
                train,
                evaluation,
                runtime,
                seed,
                tracking,
                artifact_staging,
                run_root,
            )
        raise ValueError("experiment compiler currently requires a training task")

    def _sft_input(
        self,
        run_id: str,
        task: dict[str, Any],
        train: dict[str, Any],
        evaluation: dict[str, Any],
        runtime: dict[str, Any],
        seed: int,
        tracking: dict[str, object],
        artifact_staging: dict[str, object],
        run_root: Path,
        run_resources: dict[str, object] | None = None,
    ) -> dict[str, object]:
        data = self._mapping(task.get("data"), "task data")
        prompt_protocol = self._mapping(
            task.get("prompt_protocol"), "task.prompt_protocol"
        )
        if prompt_protocol.get("mode") != "chat_template":
            raise ValueError("SFT requires task.prompt_protocol.mode=chat_template")
        sft_task = self._mapping(task.get("sft"), "task sft")
        output_model_protocol = self._model_protocol(
            train.get("output_model_protocol"),
            "train.output_model_protocol",
        )
        self._validate_training_protocol(
            train,
            output_model_protocol,
            "train.output_model_protocol",
        )
        if output_model_protocol["thinking"]["reasoning_parser"] != "qwen3":
            raise ValueError(
                "SFT output_model_protocol requires reasoning_parser=qwen3"
            )
        if evaluation.get("thinking_budget") != 8192:
            raise ValueError("SFT evaluation requires thinking_budget=8192")
        num_train_epochs = self._positive_int(
            train,
            "num_train_epochs",
            "train",
        )
        train_gpus = self._allocation_gpus(runtime, "training")
        per_device_batch = self._positive_int(
            train,
            "per_device_train_batch_size",
            "train",
        )
        gradient_accumulation = self._positive_int(
            train,
            "gradient_accumulation_steps",
            "train",
        )
        selected_rows = self._positive_int(task, "select_size", "task")
        steps_per_epoch = math.ceil(
            selected_rows
            / (train_gpus * per_device_batch * gradient_accumulation)
        )
        complete_epoch_steps = steps_per_epoch * num_train_epochs
        max_steps = (
            self._positive_int(train, "max_steps", "train")
            if "max_steps" in train
            else complete_epoch_steps
        )
        if max_steps != complete_epoch_steps:
            raise ValueError(
                "train.max_steps must equal the configured complete-epoch "
                f"boundary ({complete_epoch_steps})"
            )
        if "early_stopping_patience" in train:
            self._positive_int(
                train,
                "early_stopping_patience",
                "train",
            )
        artifact_interval = self._positive_int(
            (
                train
                if "artifact_interval" in train
                else {**train, "artifact_interval": 1}
            ),
            "artifact_interval",
            "train",
        )
        checkpoint_retention_top_k = int(
            train.get("checkpoint_retention_top_k", -1)
        )
        if checkpoint_retention_top_k < -1:
            raise ValueError(
                "train.checkpoint_retention_top_k must be -1 or non-negative"
            )
        resource_policy = resolved_resource_policy(runtime)
        recipe_path = self._project_path(
            self._required_text(sft_task, "recipe_path", "task.sft")
        )
        model_path = self._project_path(
            self._required_text(task, "base_model", "task")
        )
        training_prompt_contract = resolve_sft_prompt_contract(
            project_root=self.project_root,
            recipe_path=recipe_path,
            model_path=model_path,
            prompt_protocol=prompt_protocol,
        )
        request = {
            "project_root": str(self.project_root),
            "model": model_path,
            "recipe_path": recipe_path,
            "checkpoint_output_dir": str(run_root / "checkpoints"),
            "run_dir": str(run_root),
            "stop_file": str(run_root / "STOP"),
            "default_system": self._required_text(
                self._mapping(
                    prompt_protocol.get("system_prompt"),
                    "task.prompt_protocol.system_prompt",
                ),
                "content",
                "task.prompt_protocol.system_prompt",
            ),
            "prompt_protocol": copy.deepcopy(prompt_protocol),
            "training_prompt_contract": training_prompt_contract,
            "train_on_prompt": False,
            "mask_history": False,
            "artifact_interval": artifact_interval,
            "max_steps": max_steps,
            "seed": seed,
            "train_gpus": train_gpus,
            "output_model_protocol": copy.deepcopy(output_model_protocol),
            "training_telemetry": copy.deepcopy(tracking),
            "coordinator_resource_policy": copy.deepcopy(resource_policy),
            "workload_allocation": "training",
            "use_ray_training": True,
            "ray_namespace": "ade",
            "model_staging": bool(artifact_staging.get("enabled")),
            "model_cache_dir": artifact_staging.get("cache_dir"),
            "model_cache_max_gb": artifact_staging.get("cache_max_gb"),
            "model_cache_lock_stale_seconds": artifact_staging.get("lock_stale_seconds"),
        }
        enrichment = task.get("judge_enrichment")
        if not isinstance(enrichment, dict) or type(enrichment.get("enabled")) is not bool:
            raise ValueError("task.judge_enrichment.enabled must be explicit boolean")
        request["judge_enrichment"] = copy.deepcopy(enrichment)
        if run_resources is not None and isinstance(run_resources.get("local_judge"), dict):
            request["local_judge"] = copy.deepcopy(run_resources["local_judge"])
        for field in (
            "add_special_tokens",
            "bf16",
            "checkpoint_ready_stable_seconds",
            "cutoff_len",
            "early_stopping_patience",
            "eval_strategy",
            "gradient_accumulation_steps",
            "learning_rate",
            "logging_steps",
            "lr_scheduler_type",
            "max_steps",
            "num_train_epochs",
            "per_device_train_batch_size",
            "report_to",
            "resize_vocab",
            "save_total_limit",
            "warmup_ratio",
        ):
            if field in train:
                request[field] = copy.deepcopy(train[field])
        for required in (
            "cutoff_len",
            "gradient_accumulation_steps",
            "num_train_epochs",
            "per_device_train_batch_size",
            "train_gpus",
        ):
            if required not in request:
                raise ValueError(f"train requires `{required}` for SFT")
        evaluation_requests = (
            {
                purpose: self._evaluation_request(
                    purpose=purpose,
                    task_data=data,
                    evaluation=evaluation,
                    run_root=run_root,
                    runtime=runtime,
                    seed=seed,
                    artifact_staging=artifact_staging,
                )
                for purpose in ("online_validation", "offline_validation")
            }
            if bool(evaluation.get("enabled", True))
            else None
        )
        if evaluation_requests is not None:
            for evaluation_request in evaluation_requests.values():
                evaluation_request["evaluation_tracking"] = copy.deepcopy(tracking)
                evaluation_request["model_protocol"] = copy.deepcopy(
                    output_model_protocol
                )
                evaluation_request["reasoning_parser"] = (
                    ""
                    if output_model_protocol["thinking"]["reasoning_parser"]
                    == "none"
                    else output_model_protocol["thinking"]["reasoning_parser"]
                )
            checkpoint_epochs = list(
                range(artifact_interval, num_train_epochs + 1, artifact_interval)
            )
            if not checkpoint_epochs or checkpoint_epochs[-1] != num_train_epochs:
                checkpoint_epochs.append(num_train_epochs)
            evaluation_requests["online_validation"][
                "evaluation_tracking_position_order"
            ] = [0, *checkpoint_epochs]
        return {
            "schema_version": 1,
            "runtime": copy.deepcopy(runtime),
            "sft": {
                "max_steps": max_steps,
                "steps_per_epoch": steps_per_epoch,
                "artifact_interval": artifact_interval,
                "checkpoint_retention_top_k": checkpoint_retention_top_k,
                "output_model_protocol": copy.deepcopy(output_model_protocol),
                "dataset": {
                    "dataset_name": self._required_text(
                        data,
                        "dataset_name",
                        "task.data",
                    ),
                },
                "request": request,
                "evaluation_requests": evaluation_requests,
            },
        }

    def _rft_input(
        self,
        task_type: str,
        run_id: str,
        task: dict[str, Any],
        train: dict[str, Any],
        evaluation: dict[str, Any],
        runtime: dict[str, Any],
        seed: int,
        tracking: dict[str, object],
        artifact_staging: dict[str, object],
        run_root: Path,
    ) -> dict[str, object]:
        data = self._mapping(task.get("data"), "task data")
        prompt_protocol = self._mapping(
            task.get("prompt_protocol"), "task.prompt_protocol"
        )
        output_model_protocol = self._model_protocol(
            train.get("output_model_protocol"),
            "train.output_model_protocol",
        )
        self._validate_training_protocol(
            train,
            output_model_protocol,
            "train.output_model_protocol",
        )
        train_rft = self._mapping(train.get("rft"), "train.rft")
        task_rft = task.get("rft")
        runtime_rft = (
            copy.deepcopy(task_rft) if isinstance(task_rft, dict) else {}
        )
        self._merge(runtime_rft, train_rft)
        runtime_rft["wandb"] = copy.deepcopy(tracking)
        enrichment = task.get("judge_enrichment")
        if not isinstance(enrichment, dict) or type(enrichment.get("enabled")) is not bool:
            raise ValueError("task.judge_enrichment.enabled must be explicit boolean")
        if task_type == "curriculum_learning":
            if enrichment.get("owner") != "curriculum_realization":
                raise ValueError("Curriculum Judge owner must be curriculum_realization")
            runtime_rft["judge_enrichment"] = {
                "enabled": False,
                "owner": "rft_reward",
            }
            runtime_rft["analysis_profile_id"] = "curriculum_learning"
        else:
            runtime_rft["judge_enrichment"] = copy.deepcopy(enrichment)
        runtime_rft["coordinator_resource_policy"] = resolved_resource_policy(
            runtime
        )
        runtime_rft["workload_allocation"] = "training"
        runtime_rft["model_staging"] = bool(artifact_staging.get("enabled"))
        runtime_rft["model_cache_dir"] = artifact_staging.get("cache_dir")
        runtime_rft["model_cache_max_gb"] = artifact_staging.get("cache_max_gb")
        runtime_rft["model_cache_lock_stale_seconds"] = artifact_staging.get("lock_stale_seconds")
        retired_fields = {
            "checkpoint_interval",
            "online_eval_interval",
            "keep_online_top_k",
        } & runtime_rft.keys()
        if retired_fields:
            raise ValueError(
                f"train.rft contains retired fields: {sorted(retired_fields)}"
            )
        runtime_rft["prompt_protocol"] = copy.deepcopy(prompt_protocol)
        from ade.tasks.reward_design.training_outcome_config import (
            parquet_data_sources,
            validate_adapter_binding,
        )

        train_path = self._project_path(
            self._required_text(data, "train", "task.data")
        )
        runtime_rft["training_outcome_adapter"] = validate_adapter_binding(
            runtime_rft.get("training_outcome_adapter"),
            data_sources=set(parquet_data_sources(train_path)),
        )
        total_steps = self._positive_int(
            runtime_rft,
            "total_training_steps",
            "train.rft",
        )
        artifact_interval = self._positive_int(
            (
                runtime_rft
                if "artifact_interval" in runtime_rft
                else {**runtime_rft, "artifact_interval": 4}
            ),
            "artifact_interval",
            "train.rft",
        )
        checkpoint_retention_top_k = int(
            runtime_rft.get("checkpoint_retention_top_k", -1)
        )
        if checkpoint_retention_top_k < -1:
            raise ValueError(
                "train.rft.checkpoint_retention_top_k must be -1 or non-negative"
            )
        runtime_rft["artifact_interval"] = artifact_interval
        backend = self._mapping(
            self._mapping(runtime.get("backends"), "runtime.backends").get("rft"),
            "runtime.backends.rft",
        )
        cluster = self._mapping(runtime.get("cluster"), "runtime.cluster")
        train_gpus = self._allocation_gpus(runtime, "training")
        gpus_per_node = self._positive_int(
            backend,
            "training_gpus_per_node",
            "runtime.backends.rft",
        )
        if train_gpus % gpus_per_node:
            raise ValueError("runtime training GPUs must divide evenly by cluster.gpus_per_node")
        runtime_rft.update(
            {
                "environment": self._required_text(backend, "environment", "runtime.backends.rft"),
                "verl_source": self._required_text(backend, "source", "runtime.backends.rft"),
                "tensor_model_parallel_size": self._positive_int(
                    backend, "tensor_model_parallel_size", "runtime.backends.rft"
                ),
                "nodes": train_gpus // gpus_per_node,
                "gpus_per_node": gpus_per_node,
                "seed": seed,
            }
        )
        requests = {
            purpose: self._evaluation_request(
                purpose=purpose,
                task_data=data,
                evaluation=evaluation,
                run_root=run_root,
                runtime=runtime,
                seed=seed,
                artifact_staging=artifact_staging,
            )
            for purpose in ("online_validation", "offline_validation")
        }
        for evaluation_request in requests.values():
            evaluation_request["evaluation_tracking"] = copy.deepcopy(tracking)
            evaluation_request["model_protocol"] = copy.deepcopy(
                output_model_protocol
            )
            evaluation_request["reasoning_parser"] = (
                ""
                if output_model_protocol["thinking"]["reasoning_parser"]
                == "none"
                else output_model_protocol["thinking"]["reasoning_parser"]
            )
        checkpoint_steps = list(range(artifact_interval, total_steps + 1, artifact_interval))
        if not checkpoint_steps or checkpoint_steps[-1] != total_steps:
            checkpoint_steps.append(total_steps)
        group_credit = runtime_rft.get("group_credit")
        group_credit_enabled = False
        if group_credit is not None:
            if not isinstance(group_credit, dict) or type(group_credit.get("enabled")) is not bool:
                raise ValueError("train.rft.group_credit.enabled must be explicit boolean")
            if group_credit["enabled"] is False:
                self._strict_fields(
                    group_credit,
                    {"enabled"},
                    {"enabled"},
                    "train.rft.group_credit",
                )
                if "semantic_evidence_interval" in runtime_rft:
                    raise ValueError(
                        "disabled group credit cannot declare semantic_evidence_interval"
                    )
                runtime_rft["group_credit"] = {"enabled": False}
            else:
                self._strict_fields(
                    group_credit,
                    {"enabled", "entrypoint", "schema_version"},
                    {"enabled", "entrypoint", "schema_version"},
                    "train.rft.group_credit",
                )
                if group_credit.get("entrypoint") != "assign_group_credit":
                    raise ValueError(
                        "train.rft.group_credit.entrypoint must be assign_group_credit"
                    )
                if group_credit.get("schema_version") != "ade.group_credit.v1":
                    raise ValueError(
                        "train.rft.group_credit.schema_version must be ade.group_credit.v1"
                    )
                semantic_interval = self._positive_int(
                    runtime_rft,
                    "semantic_evidence_interval",
                    "train.rft",
                )
                semantic_steps = [
                    step
                    for step in checkpoint_steps
                    if step > 0 and step % semantic_interval == 0
                ]
                if not semantic_steps:
                    raise ValueError(
                        "train.rft.semantic_evidence_interval matches no artifact position"
                    )
                group_credit_enabled = True
                runtime_rft["group_credit"] = {
                    "enabled": True,
                    "entrypoint": "assign_group_credit",
                    "schema_version": "ade.group_credit.v1",
                }
                runtime_rft["semantic_evidence_interval"] = semantic_interval
                runtime_rft["semantic_evidence_steps"] = semantic_steps
        requests["online_validation"][
            "evaluation_tracking_position_order"
        ] = [0, *checkpoint_steps]
        verl_config = {
            "task_id": run_id,
            "project_root": str(self.project_root),
            "base_model": self._project_path(
                self._required_text(task, "base_model", "task")
            ),
            "data": {"train": train_path},
            "rft": copy.deepcopy(runtime_rft),
            "train": {
                "backend": "verl",
                "rft": copy.deepcopy(runtime_rft),
            },
        }
        rft_input = {
            "total_training_steps": total_steps,
            "artifact_interval": artifact_interval,
            "checkpoint_retention_top_k": checkpoint_retention_top_k,
            "output_model_protocol": copy.deepcopy(output_model_protocol),
            "run_offline_validation": True,
            "evaluation_requests": requests,
            "verl_config": verl_config,
        }
        if group_credit is not None:
            rft_input["group_credit"] = copy.deepcopy(runtime_rft["group_credit"])
        if group_credit_enabled:
            rft_input["semantic_evidence_interval"] = runtime_rft[
                "semantic_evidence_interval"
            ]
            rft_input["semantic_evidence_steps"] = copy.deepcopy(
                runtime_rft["semantic_evidence_steps"]
            )
        return {
            "schema_version": 1,
            "runtime": copy.deepcopy(runtime),
            "rft": rft_input,
        }

    def _evaluation_request(
        self,
        *,
        purpose: str,
        task_data: dict[str, Any],
        evaluation: dict[str, Any],
        run_root: Path,
        runtime: dict[str, Any],
        seed: int,
        artifact_staging: dict[str, object],
    ) -> dict[str, Any]:
        if purpose == "operator_test":
            datasets = task_data.get("test")
            dataset_field = "test"
        else:
            datasets = task_data.get("validation")
            dataset_field = "validation"
        if not isinstance(datasets, list) or not datasets:
            raise ValueError(
                f"task.data.{dataset_field} must contain evaluation datasets"
            )
        purpose_key = {
            "online_validation": "online",
            "offline_validation": "offline",
            "operator_test": "operator",
        }[purpose]
        purposes = self._mapping(evaluation.get("purposes"), "eval.purposes")
        profile = self._mapping(
            purposes.get(purpose_key), f"eval.purposes.{purpose_key}"
        )
        configured_k = self._positive_int(
            profile, "samples_per_input", f"eval.purposes.{purpose_key}"
        )
        decoding = self._mapping(
            profile.get("decoding"), f"eval.purposes.{purpose_key}.decoding"
        )
        allocation_name = (
            "operator_evaluation" if purpose == "operator_test" else purpose
        )
        shards = self._allocation_gpus(runtime, allocation_name)
        resolved_datasets: list[dict[str, Any]] = []
        for index, value in enumerate(datasets):
            owner = f"task.data.{dataset_field}[{index}]"
            dataset = self._mapping(value, owner)
            name = self._required_text(
                dataset,
                "name",
                owner,
            )
            resolved = copy.deepcopy(dataset)
            resolved["path"] = self._project_path(
                self._required_text(
                    dataset,
                    "path",
                    owner,
                )
            )
            resolved_datasets.append(resolved)
        dataset_ks = {
            int(
                self._mapping(
                    dataset.get("evaluation_k"),
                    f"task.data.{dataset_field}[{index}].evaluation_k",
                )[purpose]
            )
            for index, dataset in enumerate(resolved_datasets)
        }
        if len(dataset_ks) != 1:
            raise ValueError(
                f"task.data.{dataset_field} mixes dataset-specific K values for "
                f"{purpose}: {sorted(dataset_ks)}; bind datasets with one K per request"
            )
        avg_k = next(iter(dataset_ks))
        if configured_k != avg_k:
            raise ValueError(
                f"eval.purposes.{purpose_key}.samples_per_input={configured_k} "
                f"does not match dataset-specific K={avg_k} for {dataset_field}"
            )
        request: dict[str, Any] = {
            "project_root": str(self.project_root),
            "run_dir": str(run_root / "evaluation"),
            "datasets": resolved_datasets,
            "avg_k": avg_k,
            "samples_per_input": avg_k,
            "data_parallel_shards": shards,
            "seed": seed,
            "decoding": copy.deepcopy(decoding),
            "coordinator_resource_policy": resolved_resource_policy(runtime),
            "workload_allocation": allocation_name,
        }
        ranking = profile.get("ranking")
        if isinstance(ranking, dict):
            resolved_ranking = copy.deepcopy(ranking)
            weights = self._mapping(
                resolved_ranking.get("weights"),
                f"eval.purposes.{purpose_key}.ranking.weights",
            )
            resolved_ranking["weights"] = {
                str(dataset["name"]): weights[str(dataset["ranking_name"])]
                for dataset in resolved_datasets
            }
            request["validation_ranking"] = resolved_ranking
        for field in (
            "answer_format",
            "gpu_memory_utilization",
            "max_model_len",
            "max_new_tokens",
            "max_num_seqs",
            "thinking_budget",
        ):
            if field in evaluation:
                request[field] = copy.deepcopy(evaluation[field])
        request.update(
            {
                "checkpoint_staging": bool(artifact_staging.get("enabled")),
                "checkpoint_cache_dir": artifact_staging.get("cache_dir"),
                "checkpoint_cache_max_gb": artifact_staging.get("cache_max_gb"),
                "checkpoint_cache_lock_stale_seconds": artifact_staging.get("lock_stale_seconds"),
            }
        )
        request.update(copy.deepcopy(decoding))
        return request

    def _model_protocol(
        self,
        value: object,
        owner: str,
    ) -> dict[str, object]:
        protocol = self._mapping(value, owner)
        self._strict_fields(protocol, {"thinking"}, {"thinking"}, owner)
        thinking = self._mapping(protocol.get("thinking"), f"{owner}.thinking")
        mode = self._required_text(thinking, "mode", f"{owner}.thinking")
        if mode == "disabled":
            self._strict_fields(
                thinking,
                {"mode", "reasoning_parser"},
                {"mode", "reasoning_parser"},
                f"{owner}.thinking",
            )
            reasoning_parser = self._required_text(
                thinking,
                "reasoning_parser",
                f"{owner}.thinking",
            )
            if reasoning_parser != "none":
                raise ValueError(
                    f"{owner}.thinking disabled mode requires reasoning_parser=none"
                )
            return {
                "thinking": {
                    "mode": "disabled",
                    "reasoning_parser": "none",
                }
            }
        if mode != "tagged":
            raise ValueError(f"{owner}.thinking.mode must be disabled or tagged")
        fields = {
            "mode",
            "tag_encoding",
            "activation",
            "open_tag",
            "close_tag",
            "reasoning_parser",
        }
        self._strict_fields(thinking, fields, fields, f"{owner}.thinking")
        tag_encoding = self._required_text(
            thinking,
            "tag_encoding",
            f"{owner}.thinking",
        )
        activation = self._required_text(
            thinking,
            "activation",
            f"{owner}.thinking",
        )
        if (tag_encoding, activation) not in {
            ("special_tokens", "learned"),
            ("text", "chat_template"),
        }:
            raise ValueError(
                f"{owner}.thinking requires special_tokens+learned or "
                "text+chat_template"
            )
        open_tag = self._required_text(
            thinking,
            "open_tag",
            f"{owner}.thinking",
        )
        close_tag = self._required_text(
            thinking,
            "close_tag",
            f"{owner}.thinking",
        )
        if open_tag == close_tag:
            raise ValueError(f"{owner}.thinking tags must be distinct")
        reasoning_parser = self._required_text(
            thinking,
            "reasoning_parser",
            f"{owner}.thinking",
        )
        if reasoning_parser not in {"none", "qwen3"}:
            raise ValueError(
                f"{owner}.thinking.reasoning_parser must be none or qwen3"
            )
        if tag_encoding == "text" and reasoning_parser != "none":
            raise ValueError(
                f"{owner}.thinking text tags require reasoning_parser=none"
            )
        return {
            "thinking": {
                "mode": "tagged",
                "tag_encoding": tag_encoding,
                "activation": activation,
                "open_tag": open_tag,
                "close_tag": close_tag,
                "reasoning_parser": reasoning_parser,
            }
        }

    @staticmethod
    def _validate_training_protocol(
        train: dict[str, Any],
        protocol: dict[str, object],
        owner: str,
    ) -> None:
        thinking = protocol["thinking"]
        if not isinstance(thinking, dict):
            raise TypeError(f"{owner}.thinking must be a mapping")
        if thinking["mode"] != "tagged":
            return
        if thinking["tag_encoding"] != "special_tokens":
            return
        raw_tokens = train.get("add_special_tokens")
        tokens = (
            [item.strip() for item in raw_tokens.split(",")]
            if isinstance(raw_tokens, str)
            else list(raw_tokens or ())
        )
        expected = [thinking["open_tag"], thinking["close_tag"]]
        if tokens != expected:
            raise ValueError(
                f"{owner} requires train.add_special_tokens={expected!r}"
            )
        if train.get("resize_vocab") is not True:
            raise ValueError(
                f"{owner} requires train.resize_vocab=true"
            )

    def _configs_root(self, path: Path) -> Path:
        for parent in path.parents:
            if parent.name == "configs":
                return parent
        raise ValueError("experiment config must be under a configs directory")

    def _resolve_config(self, configs_root: Path, reference: str) -> Path:
        path = Path(reference)
        if path.is_absolute():
            raise ValueError("config references must be relative to configs")
        resolved = (configs_root / path).resolve()
        try:
            resolved.relative_to(configs_root.resolve())
        except ValueError as error:
            raise ValueError("config reference escapes configs root") from error
        if not resolved.is_file():
            raise ValueError(f"config reference does not exist: {reference}")
        return resolved

    def _project_path(self, value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else (self.project_root / path).resolve())

    def _existing_project_path(self, value: str, owner: str) -> str:
        path = Path(self._project_path(value))
        if not path.exists():
            raise ValueError(f"{owner} does not exist: {value}")
        return str(path)

    @staticmethod
    def _allocation_gpus(runtime: dict[str, Any], name: str) -> int:
        allocations = runtime.get("allocations")
        if not isinstance(allocations, dict):
            raise ValueError("runtime.allocations must be a mapping")
        allocation = allocations.get(name)
        if not isinstance(allocation, dict):
            raise ValueError(f"runtime.allocations.{name} must be a mapping")
        value = allocation.get("gpus")
        if type(value) is not int or value < 1:
            raise ValueError(f"runtime.allocations.{name}.gpus must be positive")
        return value

    def _portable(self, path: Path, *, configs_root: Path | None = None) -> str:
        if configs_root is not None:
            workspace_root = configs_root.resolve().parent
            try:
                return str(path.resolve().relative_to(workspace_root))
            except ValueError:
                pass
        try:
            return str(path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(path.resolve())

    @staticmethod
    def _load_mapping(path: Path) -> dict[str, Any]:
        return load_yaml_mapping(path)

    def _load_component(self, path: Path, role: str) -> dict[str, Any]:
        value = self._load_mapping(path)
        self._schema_one(value, role)
        if "name" not in value:
            raise ValueError(f"{role} component requires `name`")
        return value

    def _load_catalog(
        self,
        path: Path,
        owner: str,
        section: str,
    ) -> dict[str, Any]:
        if not path.is_file():
            raise ValueError(f"{owner} does not exist: {path}")
        value = self._load_mapping(path)
        self._schema_one(value, owner)
        return self._mapping(value.get(section), f"{owner}.{section}")

    @staticmethod
    def _without_meta(value: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(
            {key: item for key, item in value.items() if key not in _COMPONENT_META_FIELDS}
        )

    @staticmethod
    def _schema_one(value: dict[str, Any], owner: str) -> None:
        if value.get("schema_version") != 1:
            raise ValueError(f"{owner}.schema_version must be 1")

    @staticmethod
    def _strict_fields(
        value: dict[str, Any],
        allowed: set[str],
        required: set[str],
        owner: str,
    ) -> None:
        unknown = set(value) - allowed
        missing = required - set(value)
        if unknown:
            raise ValueError(f"unknown {owner} fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing {owner} fields: {sorted(missing)}")

    @staticmethod
    def _mapping(value: object, owner: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{owner} must be a mapping")
        return value

    @staticmethod
    def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
        value = payload.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"config module requires `{name}` mapping")
        return copy.deepcopy(value)

    @staticmethod
    def _required_text(
        payload: dict[str, Any],
        key: str,
        owner: str,
    ) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{owner} requires `{key}`")
        return value.strip()

    @staticmethod
    def _positive_int(
        payload: dict[str, Any],
        key: str,
        owner: str,
    ) -> int:
        value = payload.get(key)
        if isinstance(value, bool):
            raise ValueError(f"{owner}.{key} must be a positive integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{owner}.{key} must be a positive integer"
            ) from error
        if parsed < 1:
            raise ValueError(f"{owner}.{key} must be a positive integer")
        return parsed

    @staticmethod
    def _non_negative_int(
        payload: dict[str, Any],
        key: str,
        owner: str,
    ) -> int:
        value = payload.get(key)
        if isinstance(value, bool):
            raise ValueError(f"{owner}.{key} must be a non-negative integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{owner}.{key} must be a non-negative integer"
            ) from error
        if parsed < 0:
            raise ValueError(f"{owner}.{key} must be a non-negative integer")
        return parsed

    @staticmethod
    def _references(value: object, owner: str) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, list) and all(
            isinstance(item, str) for item in value
        ):
            return tuple(value)
        raise ValueError(f"{owner} must be a string or list of strings")

    @classmethod
    def _merge(
        cls,
        base: dict[str, Any],
        extra: dict[str, Any],
    ) -> None:
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                cls._merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
