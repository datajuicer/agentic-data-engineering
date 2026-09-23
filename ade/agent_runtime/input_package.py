"""Versioned immutable input packages for Agent Calls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
from typing import Mapping

from ade.core.agent import AgentRole
from ade.tasks.contracts import (
    AgentContextReference,
    AgentInputRequest,
    AgentInputSpec,
    AgentRoleContract,
)


@dataclass(frozen=True)
class AgentInputPackage:
    schema_version: str
    task_id: str
    skill_id: str
    role: AgentRole
    run_id: str
    subject_id: str
    basis_revision: int
    files: tuple[tuple[str, bytes], ...]
    references: tuple[AgentContextReference, ...] = ()

    def workspace_inputs(self) -> dict[str, bytes]:
        return dict(self.files)

    def with_materialized_overlay(
        self,
        overlay: Mapping[str, tuple[bytes, str]],
    ) -> "AgentInputPackage":
        """Return an immutable Attempt package whose manifest closes over overlay."""

        self.validate()
        if not overlay:
            return self
        files = self.workspace_inputs()
        manifest = _object(files["manifest.json"], "manifest.json")
        declarations = manifest.get("inputs")
        assert isinstance(declarations, list)
        declared_paths = {
            str(item["path"])
            for item in declarations
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        additions = []
        for path, value in sorted(overlay.items()):
            if path == "manifest.json" or path in files or path in declared_paths:
                raise ValueError(f"Agent input overlay conflicts with base package: {path}")
            relative = PurePosixPath(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe Agent input overlay path: {path}")
            content, source_ref = value
            if not isinstance(content, bytes) or not isinstance(source_ref, str) or not source_ref:
                raise ValueError(f"Agent input overlay binding is invalid: {path}")
            files[path] = content
            additions.append(
                {
                    "path": path,
                    "mode": "materialized",
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "source_ref": source_ref,
                }
            )
        manifest["inputs"] = sorted(
            [*declarations, *additions],
            key=lambda item: str(item["path"]),
        )
        files["manifest.json"] = _encode(manifest)
        package = AgentInputPackage(
            schema_version=self.schema_version,
            task_id=self.task_id,
            skill_id=self.skill_id,
            role=self.role,
            run_id=self.run_id,
            subject_id=self.subject_id,
            basis_revision=self.basis_revision,
            files=tuple(sorted(files.items())),
            references=self.references,
        )
        package.validate()
        return package

    def validate(self) -> None:
        if self.schema_version != "1":
            raise ValueError("Agent input package schema_version must be 1")
        if (
            not self.task_id
            or not self.skill_id
            or not self.run_id
            or not self.subject_id
            or self.basis_revision < 0
        ):
            raise ValueError("Agent input package identity is invalid")
        names = [name for name, _ in self.files]
        reference_names = [item.path for item in self.references]
        if len([*names, *reference_names]) != len(set([*names, *reference_names])):
            raise ValueError("Agent input package paths must be unique")
        required = {"manifest.json", "action.json", "task.json"}
        if not required.issubset(names):
            raise ValueError("Agent input package is missing required files")
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe Agent input package path: {name}")
        files = self.workspace_inputs()
        manifest = _object(files["manifest.json"], "manifest.json")
        identity = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "role": self.role.value,
            "run_id": self.run_id,
            "subject_id": self.subject_id,
            "basis_revision": self.basis_revision,
        }
        scope = manifest.get("scope")
        if (
            not isinstance(scope, dict)
            or scope.get("run_id") != self.run_id
            or manifest.get("subject_ref")
            != _canonical_subject_ref(scope)
        ):
            raise ValueError("Agent input package requires ScopeKey and SubjectRef")
        identity.update(scope=scope, subject_ref=manifest["subject_ref"])
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError("Agent input package manifest identity mismatch")
        declarations = manifest.get("inputs")
        if not isinstance(declarations, list):
            raise ValueError("Agent input package manifest requires inputs")
        declared: dict[str, dict[str, object]] = {}
        for item in declarations:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError("Agent input package declaration is invalid")
            path = item["path"]
            if path in declared:
                raise ValueError("Agent input package declarations must be unique")
            declared[path] = item
        if set(declared) != (set(files) - {"manifest.json"}) | set(reference_names):
            raise ValueError("Agent input package declarations do not match files")
        for path, item in declared.items():
            if not isinstance(item.get("source_ref"), str) or not item["source_ref"]:
                raise ValueError(f"Agent input package source_ref is missing: {path}")
            if path in files:
                digest = hashlib.sha256(files[path]).hexdigest()
                if item.get("mode") != "materialized" or item.get("sha256") != digest:
                    raise ValueError(f"Agent input package digest mismatch: {path}")
        references = {item.path: item for item in self.references}
        for path, reference in references.items():
            declaration = declared[path]
            if (
                declaration.get("mode") != "durable_hardlink"
                or declaration.get("source_ref") != reference.source_ref
                or declaration.get("sha256") != reference.sha256
                or declaration.get("size_bytes") != reference.size_bytes
            ):
                raise ValueError(f"Agent input reference binding mismatch: {path}")
        for path in ("action.json", "task.json"):
            payload = _object(files[path], path)
            if any(payload.get(key) != value for key, value in identity.items()):
                raise ValueError(f"Agent input package identity mismatch: {path}")


class AgentInputPackageBuilder:
    def build(
        self,
        spec: AgentInputSpec,
        contract: AgentRoleContract,
    ) -> AgentInputPackage:
        if spec.schema_version != "1":
            raise ValueError("Agent input package schema_version must be 1")
        if (
            spec.task_id != contract.task_id
            or spec.role is not contract.role
            or spec.skill_id != contract.skill_id
        ):
            raise ValueError("input spec does not match role contract")
        contract.validate_input(
            AgentInputRequest(
                role=spec.role,
                run_id=spec.run_id,
                subject_id=spec.subject_id,
                basis_revision=spec.basis_revision,
                action=spec.action,
                task=spec.task,
                context_files=spec.context_files,
            )
        )
        identity = {
            "schema_version": spec.schema_version,
            "task_id": spec.task_id,
            "role": spec.role.value,
            "run_id": spec.run_id,
            "subject_id": spec.subject_id,
            "basis_revision": spec.basis_revision,
            "scope": dict(spec.action["scope"]),
            "subject_ref": str(spec.action["subject_ref"]),
        }
        action = {**spec.action, **identity}
        task = {**spec.task, **identity}
        content = {
            "action.json": _encode(action),
            "task.json": _encode(task),
            **{
                item.path: item.content
                for item in spec.context_files
            },
        }
        sources = {
            item.path: item.source_ref
            for item in spec.context_files
        }
        materialized_inputs = [
            {
                "path": path,
                "mode": "materialized",
                "sha256": hashlib.sha256(value).hexdigest(),
                "source_ref": sources.get(path, "controller://generated"),
            }
            for path, value in sorted(content.items())
        ]
        referenced_inputs = [
            {
                "path": item.path,
                "mode": "durable_hardlink",
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
                "source_ref": item.source_ref,
            }
            for item in sorted(spec.context_references, key=lambda value: value.path)
        ]
        manifest = {
            **identity,
            "inputs": sorted(
                [*materialized_inputs, *referenced_inputs],
                key=lambda item: str(item["path"]),
            ),
        }
        files = {"manifest.json": _encode(manifest), **content}
        package = AgentInputPackage(
            schema_version=spec.schema_version,
            task_id=spec.task_id,
            skill_id=spec.skill_id,
            role=spec.role,
            run_id=spec.run_id,
            subject_id=spec.subject_id,
            basis_revision=spec.basis_revision,
            files=tuple(sorted(files.items())),
            references=spec.context_references,
        )
        package.validate()
        return package


def _encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _object(content: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Agent input package contains invalid JSON: {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Agent input package JSON must be an object: {label}")
    return value


def _canonical_subject_ref(scope: dict[str, object]) -> str:
    return "/".join(
        str(scope[name])
        for name in ("run_id", "coordinator_id", "plan_id", "trial_id")
        if name in scope
    )
