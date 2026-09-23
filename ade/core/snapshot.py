"""Accepted immutable snapshot identities."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
import re


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SnapshotKind(StrEnum):
    TRIAL = "trial"
    PLAN = "plan"
    RUN = "run"


@dataclass(frozen=True)
class SnapshotFile:
    path: str
    source_id: str
    content: bytes | None = None
    source_path: Path | None = None

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe snapshot path: {self.path}")
        if self.path == "manifest.json":
            raise ValueError("snapshot manifest is Harness-owned")
        if not self.source_id:
            raise ValueError("snapshot source_id is required")
        if (self.content is None) == (self.source_path is None):
            raise ValueError("snapshot file requires exactly one source")

    @classmethod
    def generated(cls, path: str, content: bytes, *, source_id: str) -> "SnapshotFile":
        return cls(path=path, source_id=source_id, content=content)

    @classmethod
    def inherited(
        cls,
        path: str,
        source_path: str | Path,
        *,
        source_id: str,
    ) -> "SnapshotFile":
        return cls(
            path=path,
            source_id=source_id,
            source_path=Path(source_path).resolve(),
        )

    def read(self) -> bytes:
        if self.content is not None:
            return self.content
        assert self.source_path is not None
        if self.source_path.is_symlink() or not self.source_path.is_file():
            raise ValueError(f"snapshot source must be a regular file: {self.source_path}")
        return self.source_path.read_bytes()


@dataclass(frozen=True)
class SnapshotRef:
    snapshot_id: str
    kind: SnapshotKind
    run_id: str
    revision: int
    manifest_sha256: str
    root: str
    coordinator_id: str | None = None
    plan_id: str | None = None
    trial_id: str | None = None

    def __post_init__(self) -> None:
        if not self.snapshot_id or not self.run_id:
            raise ValueError("Snapshot identity is required")
        if self.revision < 0:
            raise ValueError("Snapshot revision must be non-negative")
        if not _SHA256.fullmatch(self.manifest_sha256):
            raise ValueError("Snapshot manifest_sha256 is invalid")
        if not Path(self.root).is_absolute():
            raise ValueError("Snapshot root must be absolute")
        if self.kind is SnapshotKind.RUN:
            if any(
                value is not None
                for value in (self.coordinator_id, self.plan_id, self.trial_id)
            ):
                raise ValueError("Run snapshot cannot identify lower scopes")
        elif self.kind is SnapshotKind.PLAN:
            if self.coordinator_id is None or self.plan_id is None or self.trial_id is not None:
                raise ValueError("Plan snapshot requires Coordinator and Plan identity")
        elif any(
            value is None
            for value in (self.coordinator_id, self.plan_id, self.trial_id)
        ):
            raise ValueError("Trial snapshot requires complete Trial identity")
