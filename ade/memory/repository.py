"""File-backed RunState authority and immutable object storage."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Iterator

from ade.core.artifacts import ArtifactRef
from ade.core.engine import EngineCommand, EngineReceipt
from ade.core.run import RunState
from ade.engine.protocol import decode_command, encode_command
from ade.memory.layout import RunLayout
from ade.memory.records import MemoryVersionStore, TrialRecordStore
from ade.memory.snapshots import SnapshotStore
from ade.memory.usage import RunUsageLedger
from ade.review_labor.protocol import ReviewCommand, decode_command as decode_review_command, encode_command as encode_review_command

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RunAlreadyExistsError(RuntimeError):
    pass


class RunNotFoundError(RuntimeError):
    pass


class StaleRevisionError(RuntimeError):
    pass


class FileRunRepository:
    def __init__(
        self,
        runs_root: str | Path,
        *,
        run_resource_admission=None,
        results_projector=None,
    ) -> None:
        self.runs_root = Path(runs_root).resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.layout = RunLayout(self.runs_root)
        self.snapshots = SnapshotStore(self.layout)
        self.memory_versions = MemoryVersionStore(self.layout)
        self.trial_records = TrialRecordStore(self.layout)
        self.run_resource_admission = run_resource_admission
        self.results_projector = results_projector

    def create(
        self,
        state: RunState,
        *,
        initial_artifacts: tuple[tuple[ArtifactRef, bytes], ...] = (),
        initial_config_files: tuple[tuple[str, bytes], ...] = (),
        inherited_files: tuple[tuple[str, bytes], ...] = (),
        initialize_memory: bool = True,
    ) -> RunState:
        if state.revision != 0:
            raise ValueError("new RunState revision must be zero")
        self._validate_scope_ids(state)
        for ref, content in initial_artifacts:
            expected = self.describe_artifact(state.run_id, ref.kind, content)
            if ref != expected:
                raise ValueError("initial artifact reference does not match content")
        for relative, _content in initial_config_files:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("initial config path is unsafe")
        for relative, _content in inherited_files:
            path = Path(relative)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] in {"config", "artifacts", "state"}
                or (len(path.parts) == 1 and path.name in {"run.json", "manifest.json"})
            ):
                raise ValueError("inherited Run file path is unsafe")
        run_dir = self._run_dir(state.run_id)
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError as error:
            raise RunAlreadyExistsError(state.run_id) from error
        for name in (
            "artifacts",
            "config",
            "coordinators",
            "memory",
            "reports",
            "state",
            "tracking",
            "usage",
        ):
            (run_dir / name).mkdir()
        for name in ("events.jsonl", "performance-series.jsonl"):
            self._write_bytes_atomic(run_dir / "usage" / name, b"")
        zero_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }
        lineage_usage = (
            {
                "source": state.forked_from.source_revision_ref,
                "summary_ref": (
                    f"run://{state.forked_from.source_run_id}/"
                    "usage/usage-summary.json"
                ),
            }
            if state.forked_from is not None
            else None
        )
        self._write_json_atomic(
            run_dir / "usage" / "usage-summary.json",
            {
                "schema_version": "1",
                "incremental_usage": zero_usage,
                "by_category": {},
                "lineage_usage": lineage_usage,
                "event_count": 0,
            },
        )
        self._write_json_atomic(
            run_dir / "usage" / "token-summary.json",
            {"schema_version": "1", **zero_usage},
        )
        self._write_json_atomic(
            run_dir / "tracking" / "mirror-contract.json",
            {
                "schema_version": "1",
                "provider": "wandb",
                "authority": "engine_local_evidence",
                "group": state.run_id,
                "status": "not_started",
                "job_types": [
                    "base-evaluation",
                    "bootstrap-evaluation",
                    "trial-training",
                    "trial-evaluation",
                    "operator-evaluation",
                    "run-monitor",
                ],
                "upload_failure_impact": "tracking_only",
            },
        )
        self._write_json_atomic(
            run_dir / "manifest.json",
            {
                "run_id": state.run_id,
                "created_at": time.time(),
                "task_id": state.task.task_id,
                "plugin_id": state.task.plugin_id,
                "config_ref": state.task.config_ref,
                "task_config": state.task.config,
            },
        )
        if initialize_memory:
            self.memory_versions.create_run_version(
                run_id=state.run_id,
                memory_id=state.memory.run_head.rsplit("/", 1)[-1],
                parent_memory_id=None,
                created_by=state.last_transition.transition_id,
                created_revision=state.revision,
                new_sources=(state.memory.collection_basis,),
                included_sources=(state.memory.collection_basis,),
                memory_md=(
                    f"# Run Memory {state.memory.run_head}\n\n"
                    f"Collection basis: `{state.memory.collection_basis}`.\n\n"
                    "No accepted Trial outcomes yet.\n"
                ),
            )
            for plan in state.plans:
                if plan.plan_memory_head is None:
                    continue
                memory_id = plan.plan_memory_head.rsplit("/", 1)[-1]
                self.memory_versions.create_plan_version(
                    run_id=state.run_id,
                    coordinator_id=plan.coordinator_id,
                    plan_id=plan.plan_id,
                    memory_id=memory_id,
                    parent_memory_id=None,
                    created_by=state.last_transition.transition_id,
                    created_revision=state.revision,
                    new_sources=(state.memory.run_head,),
                    included_sources=(state.memory.run_head,),
                    memory_md=f"# Plan Memory {memory_id}\n\nInitial Plan memory.\n",
                    plan_md=f"# Plan {plan.coordinator_id}/{plan.plan_id}\n",
                )
        for ref, content in initial_artifacts:
            self._store_described_artifact(state.run_id, ref, content)
        for relative, content in initial_config_files:
            target = run_dir / "config" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            self._write_bytes_atomic(target, content)
        for relative, content in inherited_files:
            self._write_bytes_atomic(run_dir / relative, content)
        self._write_revision_snapshot(run_dir, state)
        self._write_json_atomic(run_dir / "run.json", state.to_dict())
        self._safe_materialize_scope_indexes(state)
        self._safe_append_event(
            run_dir,
            "run_created",
            run_id=state.run_id,
            basis_revision=None,
            result_revision=state.revision,
        )
        self._safe_materialize_timelines(state)
        self._safe_materialize_results(state)
        return state

    def load_revision(self, run_id: str, revision: int) -> RunState:
        """Load one exact immutable committed revision, never the current head."""

        self._validate_name(run_id, "run_id")
        if revision < 0:
            raise ValueError("revision must be non-negative")
        path = self.layout.revision_snapshot_path(run_id, revision)
        if not path.is_file():
            raise RunNotFoundError(f"{run_id}@rev-{revision:06d}")
        state = RunState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if state.run_id != run_id or state.revision != revision:
            raise ValueError("revision snapshot identity mismatch")
        return state

    def load(self, run_id: str) -> RunState:
        path = self._run_dir(run_id) / "run.json"
        if not path.is_file():
            raise RunNotFoundError(run_id)
        state = RunState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self._write_revision_snapshot(self._run_dir(run_id), state)
        return state

    def commit(
        self,
        state: RunState,
        *,
        expected_revision: int,
        event_type: str = "state_committed",
    ) -> RunState:
        self._validate_scope_ids(state)
        run_dir = self._run_dir(state.run_id)
        with self._lock(run_dir):
            current = self.load(state.run_id)
            if current.revision != expected_revision:
                raise StaleRevisionError(
                    f"expected revision {expected_revision}, found {current.revision}"
                )
            if state.revision != expected_revision + 1:
                raise ValueError("committed state must advance revision by one")
            transition = state.last_transition
            if (
                transition is None
                or transition.kind != event_type
                or transition.from_revision != expected_revision
                or transition.to_revision != state.revision
            ):
                raise ValueError("commit event must match last_transition")
            self._write_revision_snapshot(run_dir, state)
            self._write_json_atomic(run_dir / "run.json", state.to_dict())
            self._safe_materialize_scope_indexes(state)
            self._safe_append_event(
                run_dir,
                event_type,
                run_id=state.run_id,
                basis_revision=expected_revision,
                result_revision=state.revision,
            )
            self._safe_materialize_timelines(state)
            self._safe_materialize_results(state)
        return state

    def put_artifact(self, run_id: str, kind: str, content: bytes) -> ArtifactRef:
        ref = self.describe_artifact(run_id, kind, content)
        self._store_described_artifact(run_id, ref, content)
        return ref

    def describe_artifact(
        self,
        run_id: str,
        kind: str,
        content: bytes,
    ) -> ArtifactRef:
        self._validate_name(run_id, "run_id")
        self._validate_name(kind, "artifact kind")
        digest = hashlib.sha256(content).hexdigest()
        artifact_id = f"{kind}-{digest[:16]}"
        relative = Path("artifacts") / kind / digest
        return ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            uri=f"run://{run_id}/{relative.as_posix()}",
            digest=digest,
            size_bytes=len(content),
        )

    def _store_described_artifact(
        self,
        run_id: str,
        ref: ArtifactRef,
        content: bytes,
    ) -> None:
        relative = Path(ref.uri.removeprefix(f"run://{run_id}/"))
        target = self._run_dir(run_id) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return
        self._write_bytes_atomic(target, content)

    def read_artifact(self, run_id: str, ref: ArtifactRef) -> bytes:
        prefix = f"run://{run_id}/"
        if not ref.uri.startswith(prefix):
            raise ValueError("artifact URI is outside this run")
        relative = Path(ref.uri.removeprefix(prefix))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact URI is unsafe")
        content = (self._run_dir(run_id) / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != ref.digest:
            raise ValueError("artifact digest mismatch")
        return content

    def store_engine_command(self, command: EngineCommand) -> Path:
        path = self.layout.command_dir(
            command.run_id,
            command.coordinator_id,
            command.plan_id,
            command.trial_id,
            command.command_id,
        ) / "command.json"
        payload = encode_command(command)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise ValueError(f"command {command.command_id} is immutable")
            return path
        self._write_json_atomic(path, payload)
        return path

    def load_engine_command(
        self,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        command_id: str,
    ) -> EngineCommand:
        path = self.layout.command_dir(
            run_id, coordinator_id, plan_id, trial_id, command_id
        ) / "command.json"
        if not path.is_file():
            raise ValueError(f"persisted Engine command is missing: {command_id}")
        command = decode_command(json.loads(path.read_text(encoding="utf-8")))
        if (
            command.run_id,
            command.coordinator_id,
            command.plan_id,
            command.trial_id,
            command.command_id,
        ) != (run_id, coordinator_id, plan_id, trial_id, command_id):
            raise ValueError("persisted Engine command identity mismatch")
        return command

    def store_engine_receipt(self, receipt: EngineReceipt) -> Path:
        self._validate_name(receipt.receipt_id, "receipt_id")
        command_dir = self.layout.command_dir(
            receipt.run_id,
            receipt.coordinator_id,
            receipt.plan_id,
            receipt.trial_id,
            receipt.command_id,
        )
        command_path = command_dir / "command.json"
        if not command_path.is_file():
            raise ValueError(
                f"receipt has no persisted command: {receipt.command_id}"
            )
        command = decode_command(
            json.loads(command_path.read_text(encoding="utf-8"))
        )
        if (
            command.command_id,
            command.run_id,
            command.coordinator_id,
            command.plan_id,
            command.trial_id,
            command.logical_command_id or command.command_id,
            command.attempt_id,
            command.attempt_index,
        ) != (
            receipt.command_id,
            receipt.run_id,
            receipt.coordinator_id,
            receipt.plan_id,
            receipt.trial_id,
            receipt.logical_command_id or receipt.command_id,
            receipt.attempt_id,
            receipt.attempt_index,
        ):
            raise ValueError("Engine receipt does not match persisted command")
        path = command_dir / "receipt.json"
        payload = json.loads(json.dumps(asdict(receipt)))
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise ValueError(f"receipt {receipt.receipt_id} is immutable")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(path, payload)
        return path

    def append_usage(self, event: dict[str, object]) -> None:
        run_id = event.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("usage event requires run_id")
        RunUsageLedger(self.layout.run_dir(run_id)).append_or_enrich(event)

    def store_review_command(self, command: ReviewCommand) -> Path:
        path = self.layout.review_command_dir(
            command.run_id,
            command.coordinator_id,
            command.plan_id,
            command.trial_id,
            command.command_id,
        ) / "command.json"
        payload = encode_review_command(command)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise ValueError(f"Review command {command.command_id} is immutable")
            return path
        self._write_json_atomic(path, payload)
        return path

    def load_review_command(
        self,
        run_id: str,
        coordinator_id: str,
        plan_id: str,
        trial_id: str,
        command_id: str,
    ) -> ReviewCommand:
        path = self.layout.review_command_dir(
            run_id, coordinator_id, plan_id, trial_id, command_id
        ) / "command.json"
        if not path.is_file():
            raise ValueError(f"persisted Review command is missing: {command_id}")
        command = decode_review_command(json.loads(path.read_text(encoding="utf-8")))
        if (
            command.run_id,
            command.coordinator_id,
            command.plan_id,
            command.trial_id,
            command.command_id,
        ) != (run_id, coordinator_id, plan_id, trial_id, command_id):
            raise ValueError("persisted Review command identity mismatch")
        return command

    def _run_dir(self, run_id: str) -> Path:
        return self.layout.run_dir(run_id)

    def _write_revision_snapshot(self, run_dir: Path, state: RunState) -> None:
        path = self.layout.revision_snapshot_path(state.run_id, state.revision)
        payload = json.loads(json.dumps(state.to_dict()))
        if path.exists():
            existing = RunState.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
            existing_payload = json.loads(json.dumps(existing.to_dict()))
            if existing_payload != payload:
                raise ValueError(f"revision {state.revision} snapshot is immutable")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(path, payload)

    @staticmethod
    def _validate_name(value: str, label: str) -> None:
        if not _SAFE_NAME.fullmatch(value):
            raise ValueError(f"{label} contains unsafe characters")

    @contextmanager
    def _lock(self, run_dir: Path) -> Iterator[None]:
        lock_path = run_dir / ".run.lock"
        with lock_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _safe_append_event(
        self,
        run_dir: Path,
        event_type: str,
        *,
        run_id: str,
        basis_revision: int | None,
        result_revision: int,
    ) -> None:
        try:
            self._append_event(
                run_dir,
                event_type,
                run_id=run_id,
                basis_revision=basis_revision,
                result_revision=result_revision,
            )
        except OSError:
            return

    def _append_event(
        self,
        run_dir: Path,
        event_type: str,
        *,
        run_id: str,
        basis_revision: int | None,
        result_revision: int,
    ) -> None:
        path = run_dir / "events.jsonl"
        now = time.time()
        identity = (
            f"{run_id}\0{event_type}\0{basis_revision}\0"
            f"{result_revision}\0{now}"
        )
        record = {
            "event_id": f"event-{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
            "at": now,
            "event": event_type,
            "run_id": run_id,
            "basis_revision": basis_revision,
            "result_revision": result_revision,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _materialize_scope_indexes(self, state: RunState) -> None:
        for coordinator in state.coordinators:
            coordinator_dir = self.layout.coordinator_dir(
                state.run_id,
                coordinator.coordinator_id,
            )
            coordinator_dir.mkdir(parents=True, exist_ok=True)
            self._write_json_atomic(
                coordinator_dir / "coordinator.json",
                asdict(coordinator),
            )
        for plan in state.plans:
            plan_dir = self.layout.plan_dir(
                state.run_id,
                plan.coordinator_id,
                plan.plan_id,
            )
            plan_dir.mkdir(parents=True, exist_ok=True)
            self._write_json_atomic(plan_dir / "plan.json", asdict(plan))
        for trial in state.trials:
            trial_dir = self.layout.trial_dir(
                state.run_id,
                trial.coordinator_id,
                trial.plan_id,
                trial.trial_id,
            )
            trial_dir.mkdir(parents=True, exist_ok=True)
            self._write_json_atomic(trial_dir / "trial.json", asdict(trial))

    def _safe_materialize_scope_indexes(self, state: RunState) -> None:
        try:
            self._materialize_scope_indexes(state)
        except OSError:
            return

    def _safe_materialize_timelines(self, state: RunState) -> None:
        try:
            self._materialize_timelines(state)
        except OSError:
            return

    def materialize_results(self, state: RunState) -> None:
        """Refresh the local accepted-result projection for the current state."""

        self._safe_materialize_results(state)

    def _safe_materialize_results(self, state: RunState) -> None:
        if self.results_projector is None:
            try:
                self._refresh_results_revision(state)
            except OSError:
                return
            return
        try:
            self.results_projector.materialize(state, self._run_dir(state.run_id))
        except OSError:
            return

    def _refresh_results_revision(self, state: RunState) -> None:
        """Keep the projection revision current in processes without Engine I/O."""

        path = self._run_dir(state.run_id) / "reports" / "results.csv"
        if not path.is_file():
            return
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if fieldnames is None or "revision" not in fieldnames:
                return
            rows = list(reader)
        for row in rows:
            row["revision"] = str(state.revision)
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        self._write_bytes_atomic(path, output.getvalue().encode("utf-8"))

    def _materialize_timelines(self, state: RunState) -> None:
        run_dir = self._run_dir(state.run_id)
        event_path = run_dir / "events.jsonl"
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        coordinator_ids = tuple(
            coordinator.coordinator_id for coordinator in state.coordinators
        )
        bootstrap_ids = tuple(
            coordinator.coordinator_id
            for coordinator in state.coordinators
            if coordinator.kind.value == "bootstrap"
        )
        rows: list[dict[str, object]] = []
        previous_run_at: float | None = None
        previous_coordinator_at: dict[str, float] = {}
        for event in events:
            revision = int(event["result_revision"])
            snapshot = json.loads(
                self.layout.revision_snapshot_path(
                    state.run_id, revision
                ).read_text(encoding="utf-8")
            )
            previous_snapshot = (
                json.loads(
                    self.layout.revision_snapshot_path(
                        state.run_id, revision - 1
                    ).read_text(encoding="utf-8")
                )
                if revision > 0
                else None
            )
            transition = snapshot.get("last_transition") or {}
            subject_ref = str(transition.get("subject_ref") or "")
            coordinator_id, plan_id, trial_id = self._timeline_scope(
                subject_id=subject_ref,
                event_type=str(event["event"]),
                snapshot=snapshot,
                coordinator_ids=coordinator_ids,
                bootstrap_ids=bootstrap_ids,
                previous_snapshot=previous_snapshot,
            )
            plan = next(
                (
                    item
                    for item in snapshot.get("plans", [])
                    if item.get("coordinator_id") == coordinator_id
                    and item.get("plan_id") == plan_id
                ),
                {},
            )
            trial = next(
                (
                    item
                    for item in snapshot.get("trials", [])
                    if item.get("coordinator_id") == coordinator_id
                    and item.get("plan_id") == plan_id
                    and item.get("trial_id") == trial_id
                ),
                {},
            )
            at = float(event["at"])
            row = {
                "timestamp_utc": datetime.fromtimestamp(
                    at, tz=timezone.utc
                ).isoformat(),
                "timestamp_unix": at,
                "revision": revision,
                "transition": event["event"],
                "subject_ref": subject_ref,
                "coordinator_id": coordinator_id or "",
                "plan_id": plan_id or "",
                "trial_id": trial_id or "",
                "run_status": snapshot.get("status", ""),
                "bootstrap_status": (snapshot.get("bootstrap") or {}).get(
                    "status", ""
                ),
                "plan_status": plan.get("status", ""),
                "trial_phase": trial.get("phase", ""),
                "plan_catalog_revision": (snapshot.get("plan_catalog") or {}).get(
                    "revision", ""
                ),
                "seconds_since_run_event": (
                    "" if previous_run_at is None else round(at - previous_run_at, 6)
                ),
                "seconds_since_coordinator_event": (
                    ""
                    if coordinator_id is None
                    or coordinator_id not in previous_coordinator_at
                    else round(at - previous_coordinator_at[coordinator_id], 6)
                ),
            }
            rows.append(row)
            previous_run_at = at
            if coordinator_id is not None:
                previous_coordinator_at[coordinator_id] = at

        fieldnames = (
            "timestamp_utc",
            "timestamp_unix",
            "revision",
            "transition",
            "subject_ref",
            "coordinator_id",
            "plan_id",
            "trial_id",
            "run_status",
            "bootstrap_status",
            "plan_status",
            "trial_phase",
            "plan_catalog_revision",
            "seconds_since_run_event",
            "seconds_since_coordinator_event",
        )
        self._write_timeline_csv(run_dir / "reports" / "timeline.csv", fieldnames, rows)
        for coordinator_id in coordinator_ids:
            self._write_timeline_csv(
                run_dir
                / "reports"
                / "coordinators"
                / coordinator_id
                / "timeline.csv",
                fieldnames,
                [row for row in rows if row["coordinator_id"] == coordinator_id],
            )

    @staticmethod
    def _timeline_scope(
        *,
        subject_id: str,
        event_type: str,
        snapshot: dict[str, object],
        coordinator_ids: tuple[str, ...],
        bootstrap_ids: tuple[str, ...],
        previous_snapshot: dict[str, object] | None = None,
    ) -> tuple[str | None, str | None, str | None]:
        parts = subject_id.split("/")
        run_id = snapshot.get("run_id")
        if parts and run_id is not None and parts[0] == str(run_id):
            parts = parts[1:]
        if parts and parts[0] in coordinator_ids:
            return (
                parts[0],
                parts[1] if len(parts) > 1 else None,
                parts[2] if len(parts) > 2 else None,
            )
        trials = [
            trial
            for trial in snapshot.get("trials", [])
            if trial.get("trial_id") == subject_id
        ]
        if len(trials) == 1:
            trial = trials[0]
            return (
                str(trial["coordinator_id"]),
                str(trial["plan_id"]),
                subject_id,
            )
        plans = [
            plan
            for plan in snapshot.get("plans", [])
            if plan.get("plan_id") == subject_id
        ]
        if len(plans) == 1:
            plan = plans[0]
            return str(plan["coordinator_id"]), subject_id, None
        if event_type in {
            "summary_accepted",
            "plan_summary_defaulted",
            "run_summary_defaulted",
        } and previous_snapshot is not None:
            previous_trials = {
                (
                    trial.get("coordinator_id"),
                    trial.get("plan_id"),
                    trial.get("trial_id"),
                ): trial
                for trial in previous_snapshot.get("trials", [])
            }
            changed = []
            for trial in snapshot.get("trials", []):
                key = (
                    trial.get("coordinator_id"),
                    trial.get("plan_id"),
                    trial.get("trial_id"),
                )
                previous = previous_trials.get(key)
                if previous is not None and trial.get("phase") != previous.get("phase"):
                    changed.append(trial)
            if len(changed) == 1:
                trial = changed[0]
                return (
                    str(trial["coordinator_id"]),
                    str(trial["plan_id"]),
                    str(trial["trial_id"]),
                )
        if event_type.startswith("bootstrap_") and len(bootstrap_ids) == 1:
            return bootstrap_ids[0], None, None
        return None, None, None

    @staticmethod
    def _write_timeline_csv(
        path: Path,
        fieldnames: tuple[str, ...],
        rows: list[dict[str, object]],
    ) -> None:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        FileRunRepository._write_bytes_atomic(path, output.getvalue().encode())

    def _validate_scope_ids(self, state: RunState) -> None:
        self._validate_name(state.run_id, "run_id")
        coordinator_ids = {
            coordinator.coordinator_id for coordinator in state.coordinators
        }
        if not coordinator_ids:
            raise ValueError("Run requires at least one Coordinator")
        if len(coordinator_ids) != len(state.coordinators):
            raise ValueError("duplicate coordinator_id")
        for coordinator_id in coordinator_ids:
            self._validate_name(coordinator_id, "coordinator_id")
        plan_keys = {
            (plan.coordinator_id, plan.plan_id) for plan in state.plans
        }
        if len(plan_keys) != len(state.plans):
            raise ValueError("duplicate Plan scope")
        for coordinator_id, plan_id in plan_keys:
            self._validate_name(plan_id, "plan_id")
            if coordinator_id not in coordinator_ids:
                raise ValueError(
                    f"Plan belongs to unknown Coordinator: {coordinator_id}"
                )
        trial_keys: set[tuple[str, str, str]] = set()
        for trial in state.trials:
            self._validate_name(trial.trial_id, "trial_id")
            key = (trial.coordinator_id, trial.plan_id, trial.trial_id)
            if key in trial_keys:
                raise ValueError("duplicate Trial scope")
            trial_keys.add(key)
            if (trial.coordinator_id, trial.plan_id) not in plan_keys:
                raise ValueError(
                    "Trial belongs to unknown Plan: "
                    f"{trial.coordinator_id}/{trial.plan_id}"
                )

    @staticmethod
    def _write_json_atomic(path: Path, payload: object) -> None:
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        FileRunRepository._write_bytes_atomic(path, encoded)

    @staticmethod
    def _write_bytes_atomic(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
