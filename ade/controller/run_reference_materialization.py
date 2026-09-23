"""Shared local materialization for exact-revision Run references."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from ade.controller.ports import RunRepository
from ade.core.artifacts import ArtifactRef
from ade.core.run import RunState
from ade.core.snapshot import SnapshotRef


ARTIFACT_FIELDS = (
    "accepted_plan_refs",
    "accepted_evidence_refs",
    "accepted_experiment_outcome_refs",
    "accepted_finding_refs",
    "accepted_summary_refs",
)


class RunReferenceMaterializer:
    """Copy an already-selected RunState projection into a new Run identity.

    Business code must pass the exact source projection it intends to inherit.
    This class only copies and rewrites that projection; it does not decide
    whether a fork or Seed boundary is admissible.
    """

    def __init__(
        self,
        repository: RunRepository,
        *,
        source_repository: RunRepository | None = None,
    ) -> None:
        self.repository = repository
        self.source_repository = source_repository or repository

    def copy_artifacts(
        self,
        source: RunState,
        *,
        new_run_id: str,
    ) -> tuple[tuple[tuple[ArtifactRef, bytes], ...], dict[str, ArtifactRef]]:
        refs: dict[str, ArtifactRef] = {}
        for field in ARTIFACT_FIELDS:
            for ref in getattr(source, field):
                refs.setdefault(ref.artifact_id, ref)
        if source.task.config_ref and source.task.config_ref not in refs:
            config_ref = self._find_artifact(source.run_id, source.task.config_ref)
            refs[config_ref.artifact_id] = config_ref
        rewritten: dict[str, ArtifactRef] = {}
        pairs: list[tuple[ArtifactRef, bytes]] = []
        for artifact_id, ref in sorted(refs.items()):
            content = self.source_repository.read_artifact(source.run_id, ref)
            target = self.repository.describe_artifact(new_run_id, ref.kind, content)
            if target.artifact_id != artifact_id:
                raise ValueError("referenced artifact identity changed")
            rewritten[artifact_id] = target
            pairs.append((target, content))
        return tuple(pairs), rewritten

    def _find_artifact(self, run_id: str, artifact_id: str) -> ArtifactRef:
        root = self.source_repository.layout.run_dir(run_id) / "artifacts"
        for path in sorted(root.glob("*/*")):
            if not path.is_file() or path.is_symlink():
                continue
            content = path.read_bytes()
            ref = self.source_repository.describe_artifact(
                run_id, path.parent.name, content
            )
            if ref.artifact_id == artifact_id:
                return ref
        raise ValueError(f"source artifact is missing: {artifact_id}")

    def inherited_files(
        self,
        source: RunState,
        *,
        new_run_id: str,
        accepted_transition_ids: set[str],
    ) -> tuple[tuple[tuple[str, bytes], ...], tuple[SnapshotRef, ...]]:
        source_root = self.source_repository.layout.run_dir(source.run_id)
        files: dict[str, bytes] = {}
        accepted_digests = {
            ref.digest
            for field in ARTIFACT_FIELDS
            for ref in getattr(source, field)
        }

        for trial in source.trials:
            record = self.source_repository.layout.trial_record_dir(
                source.run_id,
                trial.coordinator_id,
                trial.plan_id,
                trial.trial_id,
            )
            if record.is_dir():
                prefix = record.relative_to(source_root)
                self._add_trial_record(
                    files,
                    prefix,
                    record,
                    source_ref=source,
                    target_run_id=new_run_id,
                    accepted_digests=accepted_digests,
                    accepted_transition_ids=accepted_transition_ids,
                )

        run_head = self._memory_number(source.memory.run_head, "RM")
        run_versions = source_root / "memory" / "run" / "versions"
        if run_versions.is_dir():
            for version in sorted(run_versions.iterdir()):
                if (
                    version.is_dir()
                    and self._memory_number(version.name, "RM") <= run_head
                ):
                    self._add_tree(
                        files,
                        version.relative_to(source_root),
                        version,
                        source_ref=source,
                        target_run_id=new_run_id,
                    )
        for plan in source.plans:
            if plan.plan_memory_head is None:
                continue
            head = self._memory_number(
                plan.plan_memory_head.rsplit("/", 1)[-1], "PM"
            )
            versions = (
                source_root
                / "memory"
                / "plans"
                / plan.coordinator_id
                / plan.plan_id
                / "versions"
            )
            if not versions.is_dir():
                continue
            for version in sorted(versions.iterdir()):
                if (
                    version.is_dir()
                    and self._memory_number(version.name, "PM") <= head
                ):
                    self._add_tree(
                        files,
                        version.relative_to(source_root),
                        version,
                        source_ref=source,
                        target_run_id=new_run_id,
                    )

        rewritten_snapshots: list[SnapshotRef] = []
        for ref in source.accepted_snapshot_refs:
            root = Path(ref.root).resolve()
            if not root.is_relative_to(source_root):
                raise ValueError("source snapshot root is outside the source Run")
            relative = root.relative_to(source_root)
            manifest = rewrite_run_refs(
                json.loads((root / "manifest.json").read_text(encoding="utf-8")),
                source.run_id,
                new_run_id,
            )
            manifest["inherited_from"] = {
                "run_id": source.run_id,
                "revision": source.revision,
                "snapshot_id": ref.snapshot_id,
            }
            manifest_bytes = (
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            ).encode()
            for path, content in read_tree(root):
                if path != "manifest.json":
                    files[(relative / path).as_posix()] = content
            files[(relative / "manifest.json").as_posix()] = manifest_bytes
            target_root = self.repository.layout.run_dir(new_run_id) / relative
            rewritten_snapshots.append(
                replace(
                    ref,
                    run_id=new_run_id,
                    root=str(target_root.resolve()),
                    manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                )
            )
        return tuple(sorted(files.items())), tuple(rewritten_snapshots)

    @classmethod
    def _add_trial_record(
        cls,
        files: dict[str, bytes],
        prefix: Path,
        root: Path,
        *,
        source_ref: RunState,
        target_run_id: str,
        accepted_digests: set[str],
        accepted_transition_ids: set[str],
    ) -> None:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        entries = [
            entry
            for entry in manifest["files"]
            if entry.get("sha256") in accepted_digests
            and entry.get("accepted_by") in accepted_transition_ids
        ]
        inherited = {
            **rewrite_run_refs(
                manifest,
                source_ref.run_id,
                target_run_id,
            ),
            "files": entries,
            "inherited_from": {
                "run_id": source_ref.run_id,
                "revision": source_ref.revision,
            },
        }
        files[(prefix / "manifest.json").as_posix()] = (
            json.dumps(inherited, indent=2, sort_keys=True) + "\n"
        ).encode()
        for entry in entries:
            relative = Path(str(entry["path"]))
            target = root / relative
            if not target.is_file() or target.is_symlink():
                raise ValueError(f"accepted Trial Record file is missing: {target}")
            content = target.read_bytes()
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(
                    f"accepted Trial Record file digest changed: {target}"
                )
            files[(prefix / relative).as_posix()] = content

    @staticmethod
    def _memory_number(value: str, prefix: str) -> int:
        value = value.rsplit("/", 1)[-1]
        if not re.fullmatch(rf"{prefix}\d{{3,}}", value):
            raise ValueError(f"invalid inherited Memory identity: {value}")
        return int(value[len(prefix) :])

    @classmethod
    def _add_tree(
        cls,
        files: dict[str, bytes],
        prefix: Path,
        root: Path,
        *,
        source_ref: RunState,
        target_run_id: str,
    ) -> None:
        for relative, content in read_tree(root):
            if relative == "manifest.json":
                manifest = rewrite_run_refs(
                    json.loads(content),
                    source_ref.run_id,
                    target_run_id,
                )
                manifest["inherited_from"] = {
                    "run_id": source_ref.run_id,
                    "revision": source_ref.revision,
                }
                content = (
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                ).encode()
            files[(prefix / relative).as_posix()] = content


def read_tree(root: Path) -> tuple[tuple[str, bytes], ...]:
    if not root.is_dir():
        raise ValueError(f"required inherited directory is missing: {root}")
    result: list[tuple[str, bytes]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"inherited tree contains a symlink: {path}")
        if path.is_file():
            result.append((path.relative_to(root).as_posix(), path.read_bytes()))
    return tuple(result)


def rewrite_run_ref(
    value: str | None,
    source_run_id: str,
    new_run_id: str,
) -> str | None:
    if value is None:
        return None
    if value == source_run_id:
        return new_run_id
    if value.startswith(f"{source_run_id}/"):
        return f"{new_run_id}{value[len(source_run_id):]}"
    uri_match = re.match(
        rf"^(?P<scheme>[a-z][a-z0-9+.-]*://){re.escape(source_run_id)}(?P<rest>/.*)?$",
        value,
        flags=re.IGNORECASE,
    )
    if uri_match is not None:
        return (
            f"{uri_match.group('scheme')}{new_run_id}"
            f"{uri_match.group('rest') or ''}"
        )
    run_path = f"/runs/{source_run_id}/"
    if run_path in value:
        return value.replace(run_path, f"/runs/{new_run_id}/")
    return value


def rewrite_run_refs(
    value: Any,
    source_run_id: str,
    new_run_id: str,
) -> Any:
    if isinstance(value, str):
        return rewrite_run_ref(value, source_run_id, new_run_id)
    if isinstance(value, dict):
        return {
            key: rewrite_run_refs(item, source_run_id, new_run_id)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            rewrite_run_refs(item, source_run_id, new_run_id) for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            rewrite_run_refs(item, source_run_id, new_run_id) for item in value
        )
    return value
