"""Control-owned immutable Memory versions and append-only Trial Records."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping

from ade.memory.layout import RunLayout

_RM_ID = re.compile(r"^RM\d{3,}$")
_PM_ID = re.compile(r"^PM\d{3,}$")


def objective_comparison_record_path(engine_attempt_index: int) -> str:
    """Return the immutable Trial Record slot for one Engine result."""
    if engine_attempt_index < 0:
        raise ValueError("Engine attempt index cannot be negative")
    if engine_attempt_index > 1:
        return (
            f"comparisons/attempt-{engine_attempt_index:03d}/"
            "objective-comparison.json"
        )
    return "comparisons/objective-comparison.json"


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("record file path is unsafe")
    return path


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json_atomic(path: Path, payload: object) -> None:
    _write_bytes_atomic(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )


class TrialRecordStore:
    """Admit accepted natural files without allowing an accepted slot to change."""

    def __init__(self, layout: RunLayout) -> None:
        self.layout = layout

    def create(
        self,
        *,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        created_revision: int,
        plan_memory_basis: str,
        run_memory_basis: str,
    ) -> Path:
        root = self.layout.trial_record_dir(
            run_id, coordinator_id, plan_id, trial_id
        )
        manifest = {
            "schema_version": 1,
            "trial": f"{run_id}/{coordinator_id}/{plan_id}/{trial_id}",
            "created_revision": created_revision,
            "plan_memory_basis": plan_memory_basis,
            "run_memory_basis": run_memory_basis,
            "files": [],
        }
        if root.exists():
            existing = self._manifest(root)
            identity_fields = (
                "schema_version",
                "trial",
                "created_revision",
                "plan_memory_basis",
                "run_memory_basis",
            )
            if any(existing.get(key) != manifest[key] for key in identity_fields):
                raise ValueError("Trial Record identity is immutable")
            return root
        root.mkdir(parents=True)
        _write_json_atomic(root / "manifest.json", manifest)
        return root

    def admit(
        self,
        *,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        transition_id: str,
        source_ref: str,
        files: Mapping[str, bytes],
    ) -> Path:
        root = self.layout.trial_record_dir(
            run_id, coordinator_id, plan_id, trial_id
        )
        with self._lock(root.parent / ".record.lock"):
            manifest = self._manifest(root)
            entries = {item["path"]: item for item in manifest["files"]}
            additions: list[dict[str, object]] = []
            for relative, content in sorted(files.items()):
                path = _safe_relative(relative)
                if path.parts[0] == "manifest.json":
                    raise ValueError("Trial Record manifest is Control-owned")
                digest = hashlib.sha256(content).hexdigest()
                entry = {
                    "path": path.as_posix(),
                    "sha256": digest,
                    "size_bytes": len(content),
                    "source_ref": source_ref,
                    "accepted_by": transition_id,
                }
                existing = entries.get(path.as_posix())
                if existing is not None and existing != entry:
                    raise ValueError(f"accepted Trial Record slot is immutable: {relative}")
                target = root / path
                if target.exists() and target.read_bytes() != content:
                    raise ValueError(f"accepted Trial Record file is immutable: {relative}")
                if existing is None:
                    additions.append(entry)
                if not target.exists():
                    _write_bytes_atomic(target, content)
            if additions:
                manifest["files"] = [*manifest["files"], *additions]
                _write_json_atomic(root / "manifest.json", manifest)
        return root

    def read_files(
        self,
        *,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
    ) -> dict[str, bytes]:
        root = self.layout.trial_record_dir(
            run_id, coordinator_id, plan_id, trial_id
        )
        manifest = self._manifest(root)
        files = {"manifest.json": (root / "manifest.json").read_bytes()}
        for item in manifest["files"]:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError("Trial Record file declaration is invalid")
            relative = item["path"]
            path = root / _safe_relative(relative)
            content = path.read_bytes()
            if (
                path.is_symlink()
                or hashlib.sha256(content).hexdigest() != item.get("sha256")
                or len(content) != item.get("size_bytes")
            ):
                raise ValueError(f"Trial Record content is invalid: {relative}")
            files[relative] = content
        return files

    @staticmethod
    @contextmanager
    def _lock(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _manifest(root: Path) -> dict[str, object]:
        path = root / "manifest.json"
        if not path.is_file():
            raise ValueError("Trial Record has no manifest")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("files"), list):
            raise ValueError("Trial Record manifest is invalid")
        return value


class MemoryVersionStore:
    """Publish complete, immutable PM/RM directories."""

    REQUIRED_DIRS = ("outcomes", "findings")

    def __init__(self, layout: RunLayout) -> None:
        self.layout = layout

    def create_run_version(
        self,
        *,
        run_id: str,
        memory_id: str,
        parent_memory_id: str | None,
        created_by: str,
        created_revision: int,
        new_sources: tuple[str, ...],
        included_sources: tuple[str, ...],
        memory_md: str,
        files: Mapping[str, bytes] = {},
    ) -> Path:
        if not _RM_ID.fullmatch(memory_id):
            raise ValueError("Run Memory ID must use RMnnn")
        return self._create(
            self.layout.run_memory_version_dir(run_id, memory_id),
            identity=f"{run_id}/{memory_id}",
            parent=parent_memory_id,
            created_by=created_by,
            created_revision=created_revision,
            new_sources=new_sources,
            included_sources=included_sources,
            memory_md=memory_md,
            files=files,
        )

    def create_plan_version(
        self,
        *,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        memory_id: str,
        parent_memory_id: str | None,
        created_by: str,
        created_revision: int,
        new_sources: tuple[str, ...],
        included_sources: tuple[str, ...],
        memory_md: str,
        plan_md: str,
        files: Mapping[str, bytes] = {},
    ) -> Path:
        if not _PM_ID.fullmatch(memory_id):
            raise ValueError("Plan Memory ID must use PMnnn")
        return self._create(
            self.layout.plan_memory_version_dir(
                run_id, coordinator_id, plan_id, memory_id
            ),
            identity=f"{run_id}/{coordinator_id}/{plan_id}/{memory_id}",
            parent=parent_memory_id,
            created_by=created_by,
            created_revision=created_revision,
            new_sources=new_sources,
            included_sources=included_sources,
            memory_md=memory_md,
            files={"plan.md": plan_md.encode(), **files},
        )

    def read_run_version(self, *, run_id: str, memory_id: str) -> dict[str, bytes]:
        if not _RM_ID.fullmatch(memory_id):
            raise ValueError("Run Memory ID must use RMnnn")
        return self._read(
            self.layout.run_memory_version_dir(run_id, memory_id),
            identity=f"{run_id}/{memory_id}",
        )

    def read_plan_version(
        self,
        *,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        memory_id: str,
    ) -> dict[str, bytes]:
        if not _PM_ID.fullmatch(memory_id):
            raise ValueError("Plan Memory ID must use PMnnn")
        return self._read(
            self.layout.plan_memory_version_dir(
                run_id, coordinator_id, plan_id, memory_id
            ),
            identity=f"{run_id}/{coordinator_id}/{plan_id}/{memory_id}",
        )

    @staticmethod
    def _read(root: Path, *, identity: str) -> dict[str, bytes]:
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError(f"Memory version has no manifest: {identity}")
        manifest_content = manifest_path.read_bytes()
        manifest = json.loads(manifest_content)
        if (
            not isinstance(manifest, dict)
            or manifest.get("identity") != identity
            or not isinstance(manifest.get("digests"), dict)
        ):
            raise ValueError(f"Memory version manifest is invalid: {identity}")
        files = {"manifest.json": manifest_content}
        for relative, digest in manifest["digests"].items():
            if not isinstance(relative, str) or not isinstance(digest, str):
                raise ValueError(f"Memory version manifest is invalid: {identity}")
            path = root / _safe_relative(relative)
            content = path.read_bytes()
            if path.is_symlink() or hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Memory version content is invalid: {identity}")
            files[relative] = content
        return files

    def _create(
        self,
        target: Path,
        *,
        identity: str,
        parent: str | None,
        created_by: str,
        created_revision: int,
        new_sources: tuple[str, ...],
        included_sources: tuple[str, ...],
        memory_md: str,
        files: Mapping[str, bytes],
    ) -> Path:
        contents = {"MEMORY.md": memory_md.encode(), **files}
        for relative in contents:
            path = _safe_relative(relative)
            if path.name == "manifest.json":
                raise ValueError("Memory manifest is Control-owned")
        manifest = {
            "schema_version": 1,
            "identity": identity,
            "parent": parent,
            "created_by": created_by,
            "created_revision": created_revision,
            "new_sources": list(new_sources),
            "included_sources": list(included_sources),
            "required_files": sorted(contents),
            "digests": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in sorted(contents.items())
            },
        }
        if target.exists():
            if json.loads((target / "manifest.json").read_text()) != manifest:
                raise ValueError(f"Memory version is immutable: {identity}")
            for name, digest in manifest["digests"].items():
                path = target / name
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digest
                ):
                    raise ValueError(f"Memory version content is invalid: {identity}")
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}."))
        try:
            for name in self.REQUIRED_DIRS:
                (temporary / name).mkdir()
            for relative, content in contents.items():
                destination = temporary / _safe_relative(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        for path in sorted(target.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        target.chmod(0o555)
        return target
