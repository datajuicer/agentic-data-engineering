"""Immutable materialized Trial, Plan, and Run snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from ade.core.snapshot import SnapshotFile, SnapshotKind, SnapshotRef
from ade.memory.layout import RunLayout


class SnapshotStore:
    def __init__(self, layout: RunLayout) -> None:
        self.layout = layout

    def materialize(
        self,
        *,
        kind: SnapshotKind,
        run_id: str,
        revision: int,
        files: tuple[SnapshotFile, ...],
        coordinator_id: str | None = None,
        plan_id: str | None = None,
        trial_id: str | None = None,
    ) -> SnapshotRef:
        root = self._root(
            kind=kind,
            run_id=run_id,
            coordinator_id=coordinator_id,
            plan_id=plan_id,
            trial_id=trial_id,
            revision=revision,
        )
        paths = tuple(item.path for item in files)
        if len(paths) != len(set(paths)):
            raise ValueError("snapshot file paths must be unique")
        contents = {item.path: item.read() for item in files}
        declarations = [
            {
                "path": item.path,
                "sha256": hashlib.sha256(contents[item.path]).hexdigest(),
                "size_bytes": len(contents[item.path]),
                "source_id": item.source_id,
            }
            for item in sorted(files, key=lambda value: value.path)
        ]
        manifest = {
            "schema_version": "ade.snapshot.v1",
            "kind": kind.value,
            "run_id": run_id,
            "coordinator_id": coordinator_id,
            "plan_id": plan_id,
            "trial_id": trial_id,
            "basis_revision": revision,
            "files": declarations,
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        ref = SnapshotRef(
            snapshot_id=self._snapshot_id(
                kind=kind,
                coordinator_id=coordinator_id,
                plan_id=plan_id,
                trial_id=trial_id,
                revision=revision,
            ),
            kind=kind,
            run_id=run_id,
            revision=revision,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            root=str(root.resolve()),
            coordinator_id=coordinator_id,
            plan_id=plan_id,
            trial_id=trial_id,
        )
        if root.exists():
            self._verify_existing(root, manifest_bytes, declarations)
            return ref
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
        try:
            for path, content in contents.items():
                target = temporary.joinpath(*PurePosixPath(path).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                target.chmod(0o444)
            (temporary / "manifest.json").write_bytes(manifest_bytes)
            (temporary / "manifest.json").chmod(0o444)
            for directory in sorted(
                (path for path in temporary.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                directory.chmod(0o555)
            temporary.chmod(0o555)
            os.replace(temporary, root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return ref

    def _root(
        self,
        *,
        kind: SnapshotKind,
        run_id: str,
        coordinator_id: str | None,
        plan_id: str | None,
        trial_id: str | None,
        revision: int,
    ) -> Path:
        if kind is SnapshotKind.RUN:
            if any(value is not None for value in (coordinator_id, plan_id, trial_id)):
                raise ValueError("Run snapshot cannot identify lower scopes")
            return self.layout.run_snapshot_dir(run_id, revision)
        if kind is SnapshotKind.PLAN:
            if coordinator_id is None or plan_id is None or trial_id is not None:
                raise ValueError("Plan snapshot requires Coordinator and Plan identity")
            return self.layout.plan_snapshot_dir(
                run_id,
                coordinator_id,
                plan_id,
                revision,
            )
        if coordinator_id is None or plan_id is None or trial_id is None:
            raise ValueError("Trial snapshot requires complete Trial identity")
        return self.layout.trial_snapshot_dir(
            run_id,
            coordinator_id,
            plan_id,
            trial_id,
            revision,
        )

    @staticmethod
    def _snapshot_id(
        *,
        kind: SnapshotKind,
        coordinator_id: str | None,
        plan_id: str | None,
        trial_id: str | None,
        revision: int,
    ) -> str:
        parts = [kind.value]
        parts.extend(
            value
            for value in (coordinator_id, plan_id, trial_id)
            if value is not None
        )
        parts.append(f"r{revision:03d}")
        return "-".join(parts)

    @staticmethod
    def _verify_existing(
        root: Path,
        manifest_bytes: bytes,
        declarations: list[dict[str, object]],
    ) -> None:
        try:
            existing_manifest = (root / "manifest.json").read_bytes()
        except OSError as error:
            raise ValueError("existing snapshot is incomplete") from error
        if existing_manifest != manifest_bytes:
            raise ValueError("snapshot revision is immutable")
        for item in declarations:
            path = root.joinpath(*PurePosixPath(str(item["path"])).parts)
            try:
                content = path.read_bytes()
            except OSError as error:
                raise ValueError("existing snapshot is incomplete") from error
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError("existing snapshot content digest mismatch")
