"""Compile a concise Operator request into the standalone evaluation schema."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

from ade.engine.storage.atomic import write_text_atomic
from ade.harness.yaml_config import load_yaml_mapping


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_OPERATION_FIELDS = {
    "schema_version",
    "kind",
    "evaluation_id",
    "run_mode",
    "execution_environment",
    "deployment",
    "expected_master_ip",
    "resource_authorization",
    "contract_catalog",
    "full_matrix_after_admission",
    "execution",
    "suites",
}
_EXECUTION_FIELDS = {
    "poll_interval_seconds",
    "snapshot_interval_seconds",
    "max_attempts",
    "max_worker_restarts",
    "max_in_flight",
}
_SUITE_FIELDS = {"contract_profile", "checkpoints", "datasets"}
_CHECKPOINT_FIELDS = {
    "id",
    "variant",
    "subject_ref",
    "path",
    "protocol_role",
    "artifact_state",
}
_DATASET_FIELDS = {"id", "samples_per_input"}
_PROFILE_FIELDS = {
    "suite_id",
    "task_type",
    "task",
    "eval",
    "prompt_contract",
    "metrics",
    "protocol_roles",
    "dataset_families",
}
_TERMINAL_AUTHORIZATION_PREFIX = "AUTHORIZED"


@dataclass(frozen=True)
class GeneralizationOperationPaths:
    deployment_root: Path
    request_root: Path
    evaluation_root: Path
    queue_root: Path
    object_root: Path
    work_root: Path
    output_root: Path
    request_snapshot: Path
    evaluation_config: Path

    def to_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class GeneralizationOperation:
    evaluation_id: str
    run_mode: str
    execution_environment: Path
    deployment_config: Path
    deployment_id: str
    ray_address: str
    expected_master_ip: str
    resource_authorization: str
    full_matrix_after_admission: bool
    poll_interval_seconds: float
    snapshot_interval_seconds: float
    max_attempts: int
    max_worker_restarts: int
    max_in_flight: int
    operation_digest: str
    paths: GeneralizationOperationPaths
    expanded_config: dict[str, object]
    source_bytes: bytes

    @property
    def authorized(self) -> bool:
        return bool(
            re.fullmatch(
                rf"{_TERMINAL_AUTHORIZATION_PREFIX}(?:\s+|:\s*|-\s*)\S.*",
                self.resource_authorization,
            )
        )

    def summary(self) -> dict[str, object]:
        checkpoint_count = sum(
            len(suite["checkpoints"]) for suite in self.expanded_config["suites"]
        )
        unit_count = sum(
            len(suite["checkpoints"]) * len(suite["datasets"])
            for suite in self.expanded_config["suites"]
        )
        return {
            "schema_version": 1,
            "evaluation_id": self.evaluation_id,
            "run_mode": self.run_mode,
            "deployment_id": self.deployment_id,
            "ray_address": self.ray_address,
            "expected_master_ip": self.expected_master_ip,
            "authorized": self.authorized,
            "full_matrix_after_admission": self.full_matrix_after_admission,
            "max_in_flight": self.max_in_flight,
            "suite_count": len(self.expanded_config["suites"]),
            "checkpoint_count": checkpoint_count,
            "unit_count": unit_count,
            "operation_digest": self.operation_digest,
            "paths": self.paths.to_dict(),
        }

    def materialize(self) -> None:
        self.paths.request_root.mkdir(parents=True, exist_ok=True)
        _write_immutable(self.paths.request_snapshot, self.source_bytes)
        encoded = yaml.safe_dump(
            self.expanded_config,
            allow_unicode=True,
            sort_keys=False,
        ).encode("utf-8")
        _write_immutable(self.paths.evaluation_config, encoded)


class GeneralizationOperationCompiler:
    """Resolve one concise, authorized operation request without using GPUs."""

    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    def compile_file(self, path: Path) -> GeneralizationOperation:
        source_path = path.resolve()
        source_bytes = source_path.read_bytes()
        source = load_yaml_mapping(source_path)
        _strict(source, _OPERATION_FIELDS, _OPERATION_FIELDS, "operation")
        if source.get("schema_version") != 1:
            raise ValueError("generalization operation schema_version must be 1")
        if source.get("kind") != "ade.generalization_operation":
            raise ValueError("operation.kind must be ade.generalization_operation")

        evaluation_id = _text(source.get("evaluation_id"), "evaluation_id")
        _safe_id(evaluation_id, "evaluation_id")
        run_mode = _text(source.get("run_mode"), "run_mode").upper()
        if run_mode not in {"NEW", "RESUME", "AUTO"}:
            raise ValueError("run_mode must be NEW, RESUME, or AUTO")

        environment = Path(
            _text(source.get("execution_environment"), "execution_environment")
        ).resolve()
        expected_environment = (
            self.project_root / ".unified-vllm-0.19.1-verl-venv"
        ).resolve()
        if environment != expected_environment:
            raise ValueError(f"execution_environment must be {expected_environment}")
        if not (environment / "bin/python").is_file():
            raise ValueError("execution environment has no bin/python")

        deployment_path = _project_file(
            self.project_root,
            source.get("deployment"),
            "deployment",
        )
        deployment = load_yaml_mapping(deployment_path)
        if deployment.get("schema_version") != 1:
            raise ValueError("deployment schema_version must be 1")
        deployment_id = _text(deployment.get("name"), "deployment.name")
        _safe_id(deployment_id, "deployment.name")
        run_resources = _mapping(
            deployment.get("run_resources"), "deployment.run_resources"
        )
        ray = _mapping(run_resources.get("ray_cluster"), "ray_cluster")
        ray_address = _text(ray.get("address"), "ray_cluster.address")
        expected_master_ip = _text(
            source.get("expected_master_ip"), "expected_master_ip"
        )
        if _address_host(ray_address) != expected_master_ip:
            raise ValueError("expected_master_ip does not match deployment Ray address")

        catalog_path = _project_file(
            self.project_root,
            source.get("contract_catalog"),
            "contract_catalog",
        )
        catalog = load_yaml_mapping(catalog_path)
        if catalog.get("schema_version") != 1:
            raise ValueError("generalization contract catalog schema_version must be 1")
        runtime = _text(catalog.get("runtime"), "contract_catalog.runtime")
        profiles = _mapping(catalog.get("profiles"), "contract_catalog.profiles")
        suites = self._expand_suites(source.get("suites"), profiles)

        execution = _mapping(source.get("execution"), "execution")
        _strict(
            execution,
            _EXECUTION_FIELDS,
            _EXECUTION_FIELDS,
            "execution",
        )
        poll_interval = _positive_number(
            execution.get("poll_interval_seconds"),
            "execution.poll_interval_seconds",
        )
        snapshot_interval = _positive_number(
            execution.get("snapshot_interval_seconds"),
            "execution.snapshot_interval_seconds",
        )
        max_attempts = _positive_int(
            execution.get("max_attempts"), "execution.max_attempts"
        )
        max_worker_restarts = _non_negative_int(
            execution.get("max_worker_restarts"),
            "execution.max_worker_restarts",
        )
        max_in_flight = _positive_int(
            execution.get("max_in_flight"), "execution.max_in_flight"
        )
        if max_in_flight not in {1, 2, 3}:
            raise ValueError(
                "generalization supervisor supports max_in_flight: 1, 2 or 3"
            )
        full_matrix = source.get("full_matrix_after_admission")
        if type(full_matrix) is not bool:
            raise ValueError("full_matrix_after_admission must be boolean")
        authorization = _text(
            source.get("resource_authorization"), "resource_authorization"
        )

        expanded: dict[str, object] = {
            "schema_version": 2,
            "evaluation_id": evaluation_id,
            "seed": 42,
            "evaluation_role": "generalization_test",
            "runtime": runtime,
            "suites": suites,
        }
        deployment_root = (
            self.project_root / "runs" / "deployments" / deployment_id
        ).resolve()
        request_root = (
            deployment_root / "evaluation-requests" / evaluation_id
        ).resolve()
        paths = GeneralizationOperationPaths(
            deployment_root=deployment_root,
            request_root=request_root,
            evaluation_root=(deployment_root / "evaluations").resolve(),
            queue_root=(
                deployment_root / "evaluation-queues" / evaluation_id
            ).resolve(),
            object_root=(deployment_root / "objects").resolve(),
            work_root=(deployment_root / "engine-work").resolve(),
            output_root=(
                self.project_root / "analysis" / "generalization" / evaluation_id
            ).resolve(),
            request_snapshot=(request_root / "operation.yaml").resolve(),
            evaluation_config=(request_root / "evaluation.yaml").resolve(),
        )
        digest = hashlib.sha256(
            json.dumps(
                {
                    "source": source,
                    "deployment": deployment,
                    "contract_catalog": catalog,
                    "expanded": expanded,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return GeneralizationOperation(
            evaluation_id=evaluation_id,
            run_mode=run_mode,
            execution_environment=environment,
            deployment_config=deployment_path,
            deployment_id=deployment_id,
            ray_address=ray_address,
            expected_master_ip=expected_master_ip,
            resource_authorization=authorization,
            full_matrix_after_admission=full_matrix,
            poll_interval_seconds=poll_interval,
            snapshot_interval_seconds=snapshot_interval,
            max_attempts=max_attempts,
            max_worker_restarts=max_worker_restarts,
            max_in_flight=max_in_flight,
            operation_digest=digest,
            paths=paths,
            expanded_config=expanded,
            source_bytes=source_bytes,
        )

    def _expand_suites(
        self,
        value: object,
        profiles: dict[str, Any],
    ) -> list[dict[str, object]]:
        source_suites = _list_of_mappings(value, "suites")
        result: list[dict[str, object]] = []
        suite_ids: set[str] = set()
        for index, suite in enumerate(source_suites):
            owner = f"suites[{index}]"
            _strict(suite, _SUITE_FIELDS, _SUITE_FIELDS, owner)
            profile_id = _text(
                suite.get("contract_profile"), f"{owner}.contract_profile"
            )
            profile = _mapping(
                profiles.get(profile_id), f"contract profile {profile_id}"
            )
            _strict(
                profile,
                _PROFILE_FIELDS,
                _PROFILE_FIELDS,
                f"contract profile {profile_id}",
            )
            suite_id = _text(profile.get("suite_id"), f"{profile_id}.suite_id")
            if suite_id in suite_ids:
                raise ValueError(f"duplicate resolved suite ID: {suite_id}")
            suite_ids.add(suite_id)
            protocol_roles = _mapping(
                profile.get("protocol_roles"), f"{profile_id}.protocol_roles"
            )
            families = _mapping(
                profile.get("dataset_families"),
                f"{profile_id}.dataset_families",
            )
            checkpoints = self._expand_checkpoints(
                suite.get("checkpoints"),
                protocol_roles,
                owner,
            )
            for control in ("base", "baseline"):
                count = sum(item["variant"] == control for item in checkpoints)
                if count > 1:
                    raise ValueError(
                        f"{owner}.checkpoints must define at most one {control} entry"
                    )
            datasets = self._expand_datasets(
                suite.get("datasets"),
                families,
                owner,
            )
            result.append(
                {
                    "id": suite_id,
                    "task_type": copy.deepcopy(profile["task_type"]),
                    "task": copy.deepcopy(profile["task"]),
                    "eval": copy.deepcopy(profile["eval"]),
                    "prompt_contract": copy.deepcopy(profile["prompt_contract"]),
                    "metrics": copy.deepcopy(profile["metrics"]),
                    "checkpoints": checkpoints,
                    "datasets": datasets,
                }
            )
        return result

    @staticmethod
    def _expand_checkpoints(
        value: object,
        protocol_roles: dict[str, Any],
        owner: str,
    ) -> list[dict[str, object]]:
        entries = _list_of_mappings(value, f"{owner}.checkpoints")
        result: list[dict[str, object]] = []
        ids: set[str] = set()
        for index, entry in enumerate(entries):
            item_owner = f"{owner}.checkpoints[{index}]"
            _strict(
                entry,
                _CHECKPOINT_FIELDS,
                _CHECKPOINT_FIELDS - {"artifact_state", "variant"},
                item_owner,
            )
            checkpoint_id = _text(entry.get("id"), f"{item_owner}.id")
            _safe_id(checkpoint_id, f"{item_owner}.id")
            if checkpoint_id in ids:
                raise ValueError(f"duplicate checkpoint ID: {checkpoint_id}")
            ids.add(checkpoint_id)
            variant = _text(
                entry.get("variant", checkpoint_id), f"{item_owner}.variant"
            )
            _safe_id(variant, f"{item_owner}.variant")
            role_id = _text(entry.get("protocol_role"), f"{item_owner}.protocol_role")
            expected_role = "base" if variant == "base" else "trained"
            if role_id != expected_role:
                raise ValueError(f"{item_owner}.protocol_role must be {expected_role}")
            role = _mapping(protocol_roles.get(role_id), f"protocol role {role_id}")
            if set(role) != {"model_protocol", "thinking_budget"}:
                raise ValueError(f"protocol role {role_id} has invalid fields")
            expanded: dict[str, object] = {
                "id": checkpoint_id,
                "variant": variant,
                "subject_ref": _text(
                    entry.get("subject_ref"), f"{item_owner}.subject_ref"
                ),
                "path": _text(entry.get("path"), f"{item_owner}.path"),
                "model_protocol": copy.deepcopy(role["model_protocol"]),
                "thinking_budget": copy.deepcopy(role["thinking_budget"]),
            }
            if "artifact_state" in entry:
                expanded["artifact_state"] = _text(
                    entry.get("artifact_state"), f"{item_owner}.artifact_state"
                )
            result.append(expanded)
        return result

    @staticmethod
    def _expand_datasets(
        value: object,
        families: dict[str, Any],
        owner: str,
    ) -> list[dict[str, object]]:
        entries = _list_of_mappings(value, f"{owner}.datasets")
        result: list[dict[str, object]] = []
        ids: set[str] = set()
        for index, entry in enumerate(entries):
            item_owner = f"{owner}.datasets[{index}]"
            _strict(entry, _DATASET_FIELDS, _DATASET_FIELDS, item_owner)
            dataset_id = _text(entry.get("id"), f"{item_owner}.id")
            _safe_id(dataset_id, f"{item_owner}.id")
            if dataset_id in ids:
                raise ValueError(f"duplicate dataset ID: {dataset_id}")
            ids.add(dataset_id)
            family = families.get(dataset_id)
            if not isinstance(family, str) or not family:
                raise ValueError(
                    f"dataset {dataset_id} is not admitted by this contract profile"
                )
            result.append(
                {
                    "id": dataset_id,
                    "samples_per_input": _positive_int(
                        entry.get("samples_per_input"),
                        f"{item_owner}.samples_per_input",
                    ),
                    "family": family,
                }
            )
        return result


def _write_immutable(path: Path, content: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != content:
            raise ValueError(f"immutable operation artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, content.decode("utf-8"))


def _project_file(project_root: Path, value: object, owner: str) -> Path:
    raw = Path(_text(value, owner))
    path = (raw if raw.is_absolute() else project_root / raw).resolve()
    if not path.is_relative_to(project_root):
        raise ValueError(f"{owner} escapes project root")
    if not path.is_file():
        raise ValueError(f"{owner} does not exist: {path}")
    return path


def _address_host(address: str) -> str:
    host, separator, port = address.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise ValueError("Ray address must be host:port")
    return host


def _strict(
    value: dict[str, Any], allowed: set[str], required: set[str], owner: str
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValueError(f"unknown {owner} fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing {owner} fields: {sorted(missing)}")


def _mapping(value: object, owner: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a mapping")
    return value


def _list_of_mappings(value: object, owner: str) -> list[dict[str, Any]]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, dict) for item in value)
    ):
        raise ValueError(f"{owner} must be a non-empty list of mappings")
    return value


def _text(value: object, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be non-empty text")
    return value.strip()


def _safe_id(value: str, owner: str) -> None:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{owner} is not path-safe")


def _positive_int(value: object, owner: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{owner} must be a positive integer")
    return value


def _non_negative_int(value: object, owner: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{owner} must be a non-negative integer")
    return value


def _positive_number(value: object, owner: str) -> float:
    if type(value) not in {int, float} or float(value) <= 0:
        raise ValueError(f"{owner} must be positive")
    return float(value)
