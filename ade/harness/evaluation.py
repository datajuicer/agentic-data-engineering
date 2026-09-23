"""Standalone evaluation matrix compilation and unit-scoped lifecycle."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from ade.core.engine import EvaluateCommand
from ade.engine.command_queue import FileCommandQueue
from ade.engine.execution.coordinator_resources import resolved_resource_policy
from ade.engine.protocol import decode_command, encode_command
from ade.engine.storage.atomic import write_json_atomic
from ade.engine.storage.object_store import FileEngineObjectStore
from ade.harness.experiment_config import ExperimentConfigCompiler
from ade.harness.yaml_config import load_yaml_mapping
from ade.tasks.data_selection.llamafactory_prompt_protocol import (
    native_tokenizer_prompt_metadata,
)
from ade.tasks.registry import default_task_registry


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SCHEMA_FIELDS = {
    "schema_version", "evaluation_id", "seed", "evaluation_role", "runtime", "suites",
}
_SUITE_FIELDS = {
    "id", "task_type", "task", "eval", "prompt_contract", "metrics", "checkpoints", "datasets",
}


@dataclass(frozen=True)
class ResolvedEvaluationConfig:
    evaluation_id: str
    config_digest: str
    evaluation_role: str
    deployment_id: str
    units: dict[str, dict[str, object]]
    expected_generations: int
    source_manifest: tuple[dict[str, str], ...]
    source_files: tuple[tuple[str, bytes], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "evaluation_id": self.evaluation_id,
            "config_digest": self.config_digest,
            "evaluation_role": self.evaluation_role,
            "deployment_id": self.deployment_id,
            "unit_count": len(self.units),
            "expected_generations": self.expected_generations,
            "units": copy.deepcopy(self.units),
            "source_manifest": copy.deepcopy(list(self.source_manifest)),
        }


class EvaluationConfigCompiler:
    """Compile one model/checkpoint x dataset matrix into Engine requests."""

    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.contracts = ExperimentConfigCompiler(
            default_task_registry(), project_root=self.project_root
        )

    def compile_file(
        self,
        path: Path,
        *,
        deployment_config: Path,
        evaluation_id: str | None = None,
        evaluation_root: Path | None = None,
    ) -> ResolvedEvaluationConfig:
        source_path = path.resolve()
        source = load_yaml_mapping(source_path)
        self._strict(source, _SCHEMA_FIELDS, _SCHEMA_FIELDS, "evaluation")
        if source.get("schema_version") != 2:
            raise ValueError("standalone evaluation schema_version must be 2")
        selected_id = evaluation_id or self._text(
            source.get("evaluation_id"), "evaluation.evaluation_id"
        )
        self._safe_id(selected_id, "evaluation_id")
        seed = self._non_negative_int(source.get("seed"), "evaluation.seed")
        evaluation_role = self._text(
            source.get("evaluation_role"), "evaluation.evaluation_role"
        )
        if evaluation_role not in {"generalization_test", "benchmark_test"}:
            raise ValueError("evaluation_role must be generalization_test or benchmark_test")

        runtime_path = self._component_path(source.get("runtime"))
        runtime_document = self._component(runtime_path, "runtime")
        runtime = self._mapping(runtime_document.get("runtime"), "runtime.runtime")
        allocation = self._mapping(
            self._mapping(runtime.get("allocations"), "runtime.allocations").get(
                "operator_evaluation"
            ),
            "runtime.allocations.operator_evaluation",
        )
        shards = self._positive_int(
            allocation.get("gpus"), "runtime.allocations.operator_evaluation.gpus"
        )

        deployment_path = deployment_config.resolve()
        deployment = self._component(deployment_path, "deployment")
        deployment_id = self._text(deployment.get("name"), "deployment.name")
        run_resources = self._mapping(
            deployment.get("run_resources"), "deployment.run_resources"
        )
        staging = self._mapping(
            run_resources.get("artifact_staging"),
            "deployment.run_resources.artifact_staging",
        )
        if type(staging.get("enabled")) is not bool:
            raise ValueError("deployment artifact_staging.enabled must be boolean")

        raw_suites = source.get("suites")
        if not isinstance(raw_suites, list) or not raw_suites:
            raise ValueError("evaluation.suites must be a non-empty list")
        suite_ids: set[str] = set()
        units: dict[str, dict[str, object]] = {}
        source_paths: set[Path] = {source_path, runtime_path, deployment_path}
        root = (
            evaluation_root.resolve()
            if evaluation_root is not None
            else (self.project_root / "runs/evaluations").resolve()
        )

        for suite_index, raw_suite in enumerate(raw_suites):
            suite = self._mapping(raw_suite, f"evaluation.suites[{suite_index}]")
            self._strict(suite, _SUITE_FIELDS, _SUITE_FIELDS, f"suite[{suite_index}]")
            suite_id = self._text(suite.get("id"), f"suite[{suite_index}].id")
            self._safe_id(suite_id, f"suite[{suite_index}].id")
            if suite_id in suite_ids:
                raise ValueError(f"duplicate suite ID: {suite_id}")
            suite_ids.add(suite_id)
            task_type = self._text(
                suite.get("task_type"), f"suite {suite_id}.task_type"
            )
            task_ref = self._text(suite.get("task"), f"suite {suite_id}.task")
            eval_path = self._component_path(suite.get("eval"))
            evaluation = self._mapping(
                self._component(eval_path, "eval").get("eval"), "eval.eval"
            )
            source_paths.add(eval_path)
            dataset_specs = self._dataset_specs(suite.get("datasets"), suite_id)
            contract = self.contracts.resolve_standalone_evaluation_contract(
                task_type=task_type,
                task_config=task_ref,
                dataset_ids=[item["id"] for item in dataset_specs],
            )
            source_paths.update(Path(item) for item in contract["source_paths"])
            resolved_task = self._mapping(contract.get("task"), "resolved task")
            resolved_datasets = self._mapping(
                contract.get("datasets"), "resolved datasets"
            )
            prompt_contract = self._prompt_contract(
                suite.get("prompt_contract"), resolved_task, suite_id
            )
            metrics = self._metrics(suite.get("metrics"), suite_id)
            checkpoints = self._checkpoints(
                suite.get("checkpoints"), suite_id, prompt_contract
            )
            if evaluation_role == "generalization_test":
                original_data = self._mapping(
                    resolved_task.get("data"), "resolved task.data"
                )
                excluded = {
                    str(item["name"])
                    for name in ("validation", "test")
                    for item in self._list_of_mappings(
                        original_data.get(name), f"resolved task.data.{name}"
                    )
                }
                overlap = sorted(
                    {item["id"] for item in dataset_specs} & excluded
                )
                if overlap:
                    raise ValueError(
                        f"suite {suite_id} generalization datasets overlap task val/test: {overlap}"
                    )

            for checkpoint in checkpoints:
                for dataset_spec in dataset_specs:
                    dataset_id = dataset_spec["id"]
                    unit_id = f"{suite_id}.{checkpoint['id']}.{dataset_id}"
                    self._safe_id(unit_id, "unit_id")
                    if unit_id in units:
                        raise ValueError(f"duplicate unit ID: {unit_id}")
                    dataset = copy.deepcopy(resolved_datasets[dataset_id])
                    dataset_prompt = self._mapping(
                        dataset.get("prompt_protocol"),
                        f"dataset {dataset_id}.prompt_protocol",
                    )
                    dataset_prompt["chat_template"] = copy.deepcopy(
                        checkpoint["chat_template"]
                    )
                    dataset["prompt_protocol"] = dataset_prompt
                    expected_rows = self._positive_int(
                        dataset.pop("expected_rows"),
                        f"dataset {dataset_id}.expected_rows",
                    )
                    samples = int(dataset_spec["samples_per_input"])
                    request = self._request(
                        evaluation_id=selected_id,
                        evaluation_role=evaluation_role,
                        unit_id=unit_id,
                        suite_id=suite_id,
                        task_type=task_type,
                        checkpoint=checkpoint,
                        dataset=dataset,
                        evaluation=evaluation,
                        runtime=runtime,
                        staging=staging,
                        root=root,
                        seed=seed,
                        shards=shards,
                        samples=samples,
                    )
                    units[unit_id] = {
                        "unit_id": unit_id,
                        "suite_id": suite_id,
                        "task_type": task_type,
                        "checkpoint_id": checkpoint["id"],
                        "variant": checkpoint["variant"],
                        "subject_ref": checkpoint["subject_ref"],
                        "checkpoint_ref": checkpoint["path"],
                        "checkpoint_available": checkpoint["available"],
                        "artifact_state": checkpoint["artifact_state"],
                        "dataset_id": dataset_id,
                        "family": dataset_spec["family"],
                        "expected_rows": expected_rows,
                        "samples_per_input": samples,
                        "expected_generations": expected_rows * samples,
                        "metrics": copy.deepcopy(metrics),
                        "request": request,
                    }

        source_files, source_manifest = self._source_files(source_paths)
        expected_generations = sum(
            int(unit["expected_generations"]) for unit in units.values()
        )
        digest_payload = {
            "schema_version": 2,
            "evaluation_id": selected_id,
            "evaluation_role": evaluation_role,
            "deployment_id": deployment_id,
            "units": units,
            "source_manifest": source_manifest,
        }
        digest = hashlib.sha256(
            json.dumps(
                digest_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return ResolvedEvaluationConfig(
            evaluation_id=selected_id,
            config_digest=digest,
            evaluation_role=evaluation_role,
            deployment_id=deployment_id,
            units=units,
            expected_generations=expected_generations,
            source_manifest=tuple(source_manifest),
            source_files=source_files,
        )

    def _request(
        self,
        *,
        evaluation_id: str,
        evaluation_role: str,
        unit_id: str,
        suite_id: str,
        task_type: str,
        checkpoint: dict[str, Any],
        dataset: dict[str, Any],
        evaluation: dict[str, Any],
        runtime: dict[str, Any],
        staging: dict[str, Any],
        root: Path,
        seed: int,
        shards: int,
        samples: int,
    ) -> dict[str, object]:
        operator = self._mapping(
            self._mapping(evaluation.get("purposes"), "eval.purposes").get(
                "operator"
            ),
            "eval.purposes.operator",
        )
        decoding = copy.deepcopy(
            self._mapping(operator.get("decoding"), "eval.purposes.operator.decoding")
        )
        model_protocol = copy.deepcopy(checkpoint["model_protocol"])
        parser = str(model_protocol["thinking"]["reasoning_parser"])
        request: dict[str, object] = {
            "project_root": str(self.project_root),
            "run_dir": str((root / evaluation_id / "units" / unit_id).resolve()),
            "datasets": [dataset],
            "avg_k": samples,
            "samples_per_input": samples,
            "data_parallel_shards": shards,
            "seed": seed,
            "model": checkpoint["path"],
            "checkpoint": checkpoint["path"],
            "decoding": decoding,
            "model_protocol": model_protocol,
            "reasoning_parser": "" if parser == "none" else parser,
            "thinking_budget": checkpoint["thinking_budget"],
            "checkpoint_staging": bool(staging["enabled"]),
            "checkpoint_cache_dir": staging.get("cache_dir"),
            "checkpoint_cache_max_gb": staging.get("cache_max_gb"),
            "checkpoint_cache_lock_stale_seconds": staging.get("lock_stale_seconds"),
            "coordinator_resource_policy": resolved_resource_policy(runtime),
            "workload_allocation": "operator_evaluation",
            "coordinator_resource_owner": f"{evaluation_id}/c000",
            "evaluation_subject_kind": "standalone_checkpoint",
            "evaluation_role": evaluation_role,
            "standalone_unit_id": unit_id,
            "task_workflow": task_type,
        }
        request.update(copy.deepcopy(decoding))
        for field in (
            "answer_format", "gpu_memory_utilization", "max_model_len", "max_new_tokens", "max_num_seqs",
        ):
            if field in evaluation:
                request[field] = copy.deepcopy(evaluation[field])
        return request

    def _prompt_contract(
        self, value: object, task: dict[str, Any], suite_id: str
    ) -> dict[str, dict[str, str]]:
        expected = self._mapping(value, f"suite {suite_id}.prompt_contract")
        self._strict(
            expected,
            {"system_prompt", "chat_template"},
            {"system_prompt", "chat_template"},
            f"suite {suite_id}.prompt_contract",
        )
        expected_system = self._binding(
            expected.get("system_prompt"),
            f"suite {suite_id}.prompt_contract.system_prompt",
        )
        expected_template = self._binding(
            expected.get("chat_template"),
            f"suite {suite_id}.prompt_contract.chat_template",
        )
        actual = self._mapping(task.get("prompt_protocol"), "task.prompt_protocol")
        if actual.get("mode") != "chat_template":
            raise ValueError(f"suite {suite_id} requires chat_template prompt mode")
        actual_system = self._mapping(
            actual.get("system_prompt"), "task.prompt_protocol.system_prompt"
        )
        if {
            "id": actual_system.get("id"),
            "digest": actual_system.get("digest"),
        } != expected_system:
            raise ValueError(f"suite {suite_id} frozen system prompt changed")
        return {"system_prompt": expected_system, "chat_template": expected_template}

    def _checkpoints(
        self,
        value: object,
        suite_id: str,
        prompt_contract: dict[str, dict[str, str]],
    ) -> list[dict[str, Any]]:
        entries = self._list_of_mappings(value, f"suite {suite_id}.checkpoints")
        result: list[dict[str, Any]] = []
        ids: set[str] = set()
        for index, entry in enumerate(entries):
            owner = f"suite {suite_id}.checkpoints[{index}]"
            self._strict(
                entry,
                {
                    "id",
                    "variant",
                    "subject_ref",
                    "path",
                    "model_protocol",
                    "thinking_budget",
                    "artifact_state",
                },
                {"id", "subject_ref", "path", "model_protocol", "thinking_budget"},
                owner,
            )
            checkpoint_id = self._text(entry.get("id"), f"{owner}.id")
            self._safe_id(checkpoint_id, f"{owner}.id")
            if checkpoint_id in ids:
                raise ValueError(f"suite {suite_id} has duplicate checkpoint ID")
            ids.add(checkpoint_id)
            variant = self._text(
                entry.get("variant", checkpoint_id), f"{owner}.variant"
            )
            self._safe_id(variant, f"{owner}.variant")
            subject_ref = self._text(entry.get("subject_ref"), f"{owner}.subject_ref")
            raw_path = Path(self._text(entry.get("path"), f"{owner}.path"))
            path = (raw_path if raw_path.is_absolute() else self.project_root / raw_path).resolve()
            artifact_state = str(entry.get("artifact_state") or "available")
            if artifact_state not in {"available", "restore_required"}:
                raise ValueError(f"{owner}.artifact_state is invalid")
            available = path.is_dir()
            if not available and artifact_state != "restore_required":
                raise ValueError(f"{owner}.path does not exist: {path}")
            expected_template = prompt_contract["chat_template"]
            if available:
                metadata = native_tokenizer_prompt_metadata(path)
                actual_digest = hashlib.sha256(
                    metadata["chat_template"].encode("utf-8")
                ).hexdigest()
                if actual_digest != expected_template["digest"]:
                    raise ValueError(f"{owner} native chat template changed")
            model_protocol = self.contracts.resolve_model_protocol(
                entry.get("model_protocol"), owner=f"{owner}.model_protocol"
            )
            thinking_budget = self._integer(
                entry.get("thinking_budget"), f"{owner}.thinking_budget"
            )
            thinking_mode = str(model_protocol["thinking"]["mode"])
            if thinking_mode == "disabled" and thinking_budget >= 0:
                raise ValueError(f"{owner} disabled thinking requires a negative budget")
            if thinking_mode == "tagged" and thinking_budget < 0:
                raise ValueError(f"{owner} tagged thinking requires a non-negative budget")
            result.append(
                {
                    "id": checkpoint_id,
                    "variant": variant,
                    "subject_ref": subject_ref,
                    "path": str(path),
                    "artifact_state": artifact_state,
                    "available": available,
                    "chat_template": copy.deepcopy(expected_template),
                    "model_protocol": model_protocol,
                    "thinking_budget": thinking_budget,
                }
            )
        return result

    def _dataset_specs(self, value: object, suite_id: str) -> list[dict[str, Any]]:
        entries = self._list_of_mappings(value, f"suite {suite_id}.datasets")
        result: list[dict[str, Any]] = []
        ids: set[str] = set()
        for index, entry in enumerate(entries):
            owner = f"suite {suite_id}.datasets[{index}]"
            self._strict(
                entry,
                {"id", "samples_per_input", "family"},
                {"id", "samples_per_input", "family"},
                owner,
            )
            dataset_id = self._text(entry.get("id"), f"{owner}.id")
            self._safe_id(dataset_id, f"{owner}.id")
            if dataset_id in ids:
                raise ValueError(f"suite {suite_id} has duplicate dataset ID")
            ids.add(dataset_id)
            result.append(
                {
                    "id": dataset_id,
                    "samples_per_input": self._positive_int(
                        entry.get("samples_per_input"), f"{owner}.samples_per_input"
                    ),
                    "family": self._text(entry.get("family"), f"{owner}.family"),
                }
            )
        return result

    def _metrics(self, value: object, suite_id: str) -> dict[str, str]:
        metrics = self._mapping(value, f"suite {suite_id}.metrics")
        self._strict(
            metrics,
            {"primary", "secondary"},
            {"primary", "secondary"},
            f"suite {suite_id}.metrics",
        )
        result = {
            name: self._text(metrics.get(name), f"suite {suite_id}.metrics.{name}")
            for name in ("primary", "secondary")
        }
        if set(result.values()) != {"pass_at_k", "avg_at_k"}:
            raise ValueError(f"suite {suite_id} metrics must be pass_at_k and avg_at_k")
        return result

    def _source_files(
        self, paths: set[Path]
    ) -> tuple[tuple[tuple[str, bytes], ...], list[dict[str, str]]]:
        files: list[tuple[str, bytes]] = []
        manifest: list[dict[str, str]] = []
        for path in sorted({item.resolve() for item in paths}, key=str):
            content = path.read_bytes()
            try:
                relative = path.relative_to(self.project_root).as_posix()
            except ValueError:
                relative = path.name
            copy_path = f"sources/{relative}"
            digest = hashlib.sha256(content).hexdigest()
            files.append((copy_path, content))
            manifest.append({"path": relative, "copy": copy_path, "digest": digest})
        return tuple(files), manifest

    def _component_path(self, value: object) -> Path:
        relative = Path(self._text(value, "component reference"))
        if relative.is_absolute():
            raise ValueError("component references must be relative to configs")
        path = (self.project_root / "configs" / relative).resolve()
        if not path.is_relative_to((self.project_root / "configs").resolve()):
            raise ValueError("component reference escapes configs")
        if not path.is_file():
            raise ValueError(f"component does not exist: {value}")
        return path

    @staticmethod
    def _component(path: Path, role: str) -> dict[str, Any]:
        value = load_yaml_mapping(path)
        if value.get("schema_version") != 1 or not isinstance(value.get("name"), str):
            raise ValueError(f"{role} component schema is invalid: {path}")
        return value

    @staticmethod
    def _binding(value: object, owner: str) -> dict[str, str]:
        binding = EvaluationConfigCompiler._mapping(value, owner)
        EvaluationConfigCompiler._strict(
            binding, {"id", "digest"}, {"id", "digest"}, owner
        )
        result = {
            "id": EvaluationConfigCompiler._text(binding.get("id"), f"{owner}.id"),
            "digest": EvaluationConfigCompiler._text(
                binding.get("digest"), f"{owner}.digest"
            ),
        }
        if len(result["digest"]) != 64 or any(
            character not in "0123456789abcdef" for character in result["digest"]
        ):
            raise ValueError(f"{owner}.digest must be a SHA-256 digest")
        return result

    @staticmethod
    def _strict(
        value: dict[str, Any], allowed: set[str], required: set[str], owner: str
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
    def _list_of_mappings(value: object, owner: str) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not value or any(
            not isinstance(item, dict) for item in value
        ):
            raise ValueError(f"{owner} must be a non-empty list of mappings")
        return value

    @staticmethod
    def _text(value: object, owner: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{owner} must be non-empty text")
        return value.strip()

    @staticmethod
    def _integer(value: object, owner: str) -> int:
        if type(value) is not int:
            raise ValueError(f"{owner} must be an integer")
        return value

    @classmethod
    def _positive_int(cls, value: object, owner: str) -> int:
        result = cls._integer(value, owner)
        if result < 1:
            raise ValueError(f"{owner} must be positive")
        return result

    @classmethod
    def _non_negative_int(cls, value: object, owner: str) -> int:
        result = cls._integer(value, owner)
        if result < 0:
            raise ValueError(f"{owner} must be non-negative")
        return result

    @staticmethod
    def _safe_id(value: str, owner: str) -> None:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError(f"{owner} is invalid")


class StandaloneEvaluationService:
    """Materialize, submit, observe, and retry standalone evaluation units."""

    def __init__(
        self,
        *,
        root: Path,
        queue: FileCommandQueue,
        objects: FileEngineObjectStore,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue = queue
        self.objects = objects

    def create(self, config: ResolvedEvaluationConfig) -> dict[str, object]:
        directory = self._directory(config.evaluation_id)
        if directory.exists():
            raise ValueError(f"evaluation already exists: {config.evaluation_id}")
        (directory / "config").mkdir(parents=True)
        write_json_atomic(directory / "config/resolved.json", config.to_dict())
        for relative, content in config.source_files:
            target = directory / "config" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        units: dict[str, dict[str, object]] = {}
        for unit_id, resolved in config.units.items():
            units[unit_id] = {
                "unit_id": unit_id,
                "status": "created",
                "checkpoint_ref": resolved["checkpoint_ref"],
                "checkpoint_available": resolved["checkpoint_available"],
                "artifact_state": resolved["artifact_state"],
                "attempts": [self._attempt(config.evaluation_id, unit_id, resolved, 1)],
            }
        state: dict[str, object] = {
            "schema_version": 2,
            "evaluation_id": config.evaluation_id,
            "config_digest": config.config_digest,
            "status": "created",
            "unit_count": len(units),
            "expected_generations": config.expected_generations,
            "unit_order": list(units),
            "units": units,
        }
        self._write_state(config.evaluation_id, state)
        return copy.deepcopy(state)

    def run(
        self, evaluation_id: str, *, unit_id: str | None = None
    ) -> dict[str, object]:
        state = self.status(evaluation_id)
        units = self._state_units(state)
        selected = self._selected_units(units, unit_id)
        not_ready = [
            name for name in selected
            if not Path(str(units[name]["checkpoint_ref"])).is_dir()
        ]
        if not_ready:
            raise ValueError(
                "checkpoint restore required before submission: " + ", ".join(not_ready)
            )
        for name in selected:
            unit = units[name]
            if unit["status"] in {"succeeded", "failed"}:
                continue
            attempt = self._active_attempt(unit)
            self.queue.submit(decode_command(dict(attempt["command"])))
            attempt["status"] = "submitted"
            unit["status"] = "submitted"
        self._write_state(evaluation_id, state)
        return self.status(evaluation_id)

    def retry(
        self, evaluation_id: str, *, unit_id: str, after_repair: bool = False
    ) -> dict[str, object]:
        state = self.status(evaluation_id)
        units = self._state_units(state)
        selected = self._selected_units(units, unit_id)
        unit = units[selected[0]]
        if unit["status"] != "failed":
            raise ValueError(f"unit is not failed: {unit_id}")
        prior = self._active_attempt(unit)
        receipt = prior.get("receipt")
        if not isinstance(receipt, dict) or (
            receipt.get("retryable") is not True and not after_repair
        ):
            raise ValueError(f"unit failure is not retryable: {unit_id}")
        resolved = self._resolved_unit(evaluation_id, unit_id)
        unit["attempts"].append(
            self._attempt(evaluation_id, unit_id, resolved, len(unit["attempts"]) + 1)
        )
        unit["status"] = "created"
        state["status"] = "in_progress"
        self._write_state(evaluation_id, state)
        return copy.deepcopy(state)

    def status(self, evaluation_id: str) -> dict[str, object]:
        state = self._load_state(evaluation_id)
        units = self._state_units(state)
        for unit in units.values():
            attempt = self._active_attempt(unit)
            command_id = str(attempt["command"]["command_id"])
            if not self.queue.has_receipt(command_id):
                continue
            receipt = self.queue.load_receipt(command_id)
            attempt["status"] = receipt.status.value
            attempt["receipt"] = receipt.to_dict()
            unit["status"] = receipt.status.value
        statuses = [str(unit["status"]) for unit in units.values()]
        if statuses and all(status == "succeeded" for status in statuses):
            state["status"] = "complete"
        elif statuses and all(status in {"succeeded", "failed"} for status in statuses):
            state["status"] = "completed_degraded"
        elif any(status in {"submitted", "succeeded", "failed"} for status in statuses):
            state["status"] = "in_progress"
        else:
            state["status"] = "created"
        self._write_state(evaluation_id, state)
        return copy.deepcopy(state)

    def _attempt(
        self,
        evaluation_id: str,
        unit_id: str,
        resolved: dict[str, Any],
        index: int,
    ) -> dict[str, object]:
        attempt_id = f"attempt-{index:03d}"
        logical_id = f"{evaluation_id}-{unit_id}"
        command_id = f"{logical_id}-{attempt_id}"
        input_ref = f"engine://inputs/{command_id}.json"
        output_uri = f"engine://outputs/{command_id}"
        self.objects.put_json(
            input_ref,
            {
                "schema_version": 1,
                "evaluation": {
                    "purpose": "operator_test",
                    "request": copy.deepcopy(resolved["request"]),
                },
            },
        )
        command = EvaluateCommand(
            command_id=command_id,
            logical_command_id=logical_id,
            attempt_id=attempt_id,
            attempt_index=index,
            run_id=evaluation_id,
            coordinator_id="c000",
            plan_id="p000",
            trial_id=unit_id,
            input_ref=input_ref,
            output_uri=output_uri,
        )
        return {
            "attempt_id": attempt_id,
            "attempt_index": index,
            "status": "created",
            "command": encode_command(command),
            "receipt": None,
        }

    def _resolved_unit(self, evaluation_id: str, unit_id: str) -> dict[str, Any]:
        payload = json.loads(
            (self._directory(evaluation_id) / "config/resolved.json").read_text(
                encoding="utf-8"
            )
        )
        units = payload.get("units")
        if not isinstance(units, dict) or not isinstance(units.get(unit_id), dict):
            raise ValueError(f"unknown unit_id: {unit_id}")
        return units[unit_id]

    @staticmethod
    def _state_units(state: dict[str, object]) -> dict[str, dict[str, Any]]:
        units = state.get("units")
        if not isinstance(units, dict) or any(
            not isinstance(value, dict) for value in units.values()
        ):
            raise ValueError("evaluation state units are invalid")
        return units

    @staticmethod
    def _selected_units(
        units: dict[str, dict[str, Any]], unit_id: str | None
    ) -> list[str]:
        if unit_id is None:
            return list(units)
        if unit_id not in units:
            raise ValueError(f"unknown unit_id: {unit_id}")
        return [unit_id]

    @staticmethod
    def _active_attempt(unit: dict[str, Any]) -> dict[str, Any]:
        attempts = unit.get("attempts")
        if not isinstance(attempts, list) or not attempts or not isinstance(attempts[-1], dict):
            raise ValueError("evaluation unit attempts are invalid")
        return attempts[-1]

    def _directory(self, evaluation_id: str) -> Path:
        if not _SAFE_ID.fullmatch(evaluation_id):
            raise ValueError("evaluation_id is invalid")
        return self.root / evaluation_id

    def _load_state(self, evaluation_id: str) -> dict[str, object]:
        return json.loads(
            (self._directory(evaluation_id) / "state.json").read_text(encoding="utf-8")
        )

    def _write_state(self, evaluation_id: str, state: dict[str, object]) -> None:
        write_json_atomic(self._directory(evaluation_id) / "state.json", state)
