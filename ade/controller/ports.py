"""Infrastructure ports used by Controller application services."""

from __future__ import annotations

from typing import Any, Protocol

from ade.core.artifacts import ArtifactRef
from ade.core.engine import EngineCommand, EngineReceipt
from ade.core.run import RunState
from ade.core.snapshot import SnapshotFile, SnapshotKind, SnapshotRef
from ade.review_labor.protocol import ReviewCommand, ReviewReceipt


class SnapshotPort(Protocol):
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
    ) -> SnapshotRef: ...


class RunRepository(Protocol):
    layout: Any
    snapshots: SnapshotPort

    def create(
        self,
        state: RunState,
        *,
        initial_artifacts: tuple[tuple[ArtifactRef, bytes], ...] = (),
        initial_config_files: tuple[tuple[str, bytes], ...] = (),
        inherited_files: tuple[tuple[str, bytes], ...] = (),
        initialize_memory: bool = True,
    ) -> RunState: ...
    def load(self, run_id: str) -> RunState: ...
    def load_revision(self, run_id: str, revision: int) -> RunState: ...
    def commit(
        self,
        state: RunState,
        *,
        expected_revision: int,
        event_type: str = "state_committed",
    ) -> RunState: ...
    def describe_artifact(
        self, run_id: str, kind: str, content: bytes
    ) -> ArtifactRef: ...
    def put_artifact(self, run_id: str, kind: str, content: bytes) -> ArtifactRef: ...
    def read_artifact(self, run_id: str, ref: ArtifactRef) -> bytes: ...
    def store_engine_command(self, command: EngineCommand): ...
    def load_engine_command(
        self,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        command_id: str,
    ) -> EngineCommand: ...
    def store_engine_receipt(self, receipt: EngineReceipt): ...
    def store_review_command(self, command: ReviewCommand): ...
    def load_review_command(
        self,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        command_id: str,
    ) -> ReviewCommand: ...
    def append_usage(self, event: dict[str, Any]) -> None: ...


class EngineCommandPort(Protocol):
    def submit(self, command: EngineCommand): ...
    def interrupt(self, command_id: str) -> EngineReceipt: ...
    def has_receipt(self, command_id: str) -> bool: ...
    def load_receipt(self, command_id: str) -> EngineReceipt: ...


class EngineObjectPort(Protocol):
    def read_json(self, uri: str) -> dict[str, Any]: ...
    def read_bytes(self, uri: str) -> bytes: ...
    def put_json(self, uri: str, payload: dict[str, Any]) -> str: ...
    def put_bytes(self, uri: str, content: bytes) -> str: ...


class ReviewCommandPort(Protocol):
    def submit(self, command: ReviewCommand): ...
    def has_receipt(self, command_id: str) -> bool: ...
    def load_receipt(self, command_id: str) -> ReviewReceipt: ...
