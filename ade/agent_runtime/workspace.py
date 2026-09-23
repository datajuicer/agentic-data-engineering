"""Creation of immutable Agent Call attempt workspaces."""

from __future__ import annotations

import json
import errno
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from ade.core.agent import (
    AgentCall,
    AgentCallAttempt,
    AgentRole,
    AgentSession,
    AgentSessionStatus,
    AttemptKind,
)
from ade.core.scope import subject_ref
from ade.core.validation import ValidationReport
from ade.memory.layout import RunLayout
from ade.tasks.contracts import AgentContextReference

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REQUIRED_INPUTS = {"manifest.json", "action.json", "task.json"}


class WorkspaceManager:
    def __init__(
        self,
        calls_root: str | Path,
        *,
        run_layout: RunLayout | None = None,
        trial_artifacts_root: str | Path | None = None,
    ) -> None:
        self.calls_root = Path(calls_root).resolve()
        self.run_layout = run_layout
        self.trial_artifacts_root = (
            Path(trial_artifacts_root).resolve()
            if trial_artifacts_root is not None
            else None
        )

    @classmethod
    def for_runs_root(
        cls,
        runs_root: str | Path,
        *,
        trial_artifacts_root: str | Path | None = None,
    ) -> "WorkspaceManager":
        layout = RunLayout(runs_root)
        durable_root = (
            Path(trial_artifacts_root)
            if trial_artifacts_root is not None
            else layout.runs_root.parent / "engine-work" / "trial_artifacts"
        )
        return cls(
            layout.runs_root,
            run_layout=layout,
            trial_artifacts_root=durable_root,
        )

    def call_dir(self, call: AgentCall) -> Path:
        if self.run_layout is not None:
            return self.run_layout.agent_call_dir(call)
        if not _SAFE_ID.fullmatch(call.call_id):
            raise ValueError("call_id contains unsafe characters")
        return self.calls_root / call.call_id

    def session_dir(self, session: AgentSession) -> Path:
        if self.run_layout is not None:
            return self.run_layout.agent_session_dir(session)
        if not _SAFE_ID.fullmatch(session.session_id):
            raise ValueError("session_id contains unsafe characters")
        return self.calls_root / session.session_id

    def ensure_session(
        self,
        session: AgentSession,
        call: AgentCall,
    ) -> AgentSession:
        if call.session_id != session.session_id:
            raise ValueError("Agent Call belongs to another logical Session")
        path = self.session_dir(session) / "session.json"
        identity = {
            "session_id": session.session_id,
            "run_id": session.run_id,
            "role": session.role.value,
            "subject_id": session.subject_id,
            "coordinator_id": session.coordinator_id,
            "plan_id": session.plan_id,
            "trial_id": session.trial_id,
            "scope": _scope(
                session.run_id,
                session.coordinator_id,
                session.plan_id,
                session.trial_id,
            ),
            "subject_ref": subject_ref(
                session.run_id,
                session.coordinator_id,
                session.plan_id,
                session.trial_id,
            ),
        }
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("persisted Agent Session is invalid") from error
            if not isinstance(payload, dict) or any(
                payload.get(key) != value for key, value in identity.items()
            ):
                raise ValueError("persisted Agent Session identity mismatch")
        else:
            payload = {
                "schema_version": "1",
                **identity,
                "status": session.status.value,
                "resume_handle": session.resume_handle,
                "current_session_id": session.resume_handle,
                "call_ids": list(session.call_ids),
                "attempts": [],
            }
        call_ids = payload.get("call_ids")
        if not isinstance(call_ids, list) or any(
            not isinstance(value, str) or not value for value in call_ids
        ):
            raise ValueError("persisted Agent Session call IDs are invalid")
        if call.call_id not in call_ids:
            call_ids.append(call.call_id)
        payload["call_ids"] = call_ids
        self._write_json(path, payload)
        return self.load_session(session)

    def load_session(self, expected: AgentSession) -> AgentSession:
        path = self.session_dir(expected) / "session.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("persisted Agent Session is invalid") from error
        resume_handle = payload.get("current_session_id")
        if not isinstance(resume_handle, str) or not resume_handle:
            resume_handle = payload.get("resume_handle")
        if not isinstance(resume_handle, str) or not resume_handle:
            resume_handle = None
        return AgentSession(
            session_id=str(payload["session_id"]),
            run_id=str(payload["run_id"]),
            role=AgentRole(str(payload["role"])),
            subject_id=str(payload["subject_id"]),
            status=AgentSessionStatus(str(payload.get("status", "active"))),
            coordinator_id=payload.get("coordinator_id"),
            plan_id=payload.get("plan_id"),
            trial_id=payload.get("trial_id"),
            resume_handle=resume_handle,
            call_ids=tuple(payload.get("call_ids", ())),
        )

    def create_attempt(
        self,
        call: AgentCall,
        number: int,
        kind: AttemptKind,
        inputs: dict[str, bytes],
        references: tuple[AgentContextReference, ...] = (),
    ) -> Path:
        if not _SAFE_ID.fullmatch(call.call_id):
            raise ValueError("call_id contains unsafe characters")
        missing = _REQUIRED_INPUTS - set(inputs)
        if missing:
            raise ValueError(f"Agent input is missing {sorted(missing)}")
        resolved_references = tuple(
            (reference, self._resolve_reference(reference))
            for reference in references
        )
        label = self.attempt_label(call, number, kind)
        call_dir = self.call_dir(call)
        attempt = call_dir / "attempts" / label
        if attempt.exists():
            raise FileExistsError(f"Agent attempt already exists: {label}")
        input_dir = attempt / "input"
        output_dir = attempt / "output"
        scratch_dir = attempt / "scratch"
        for path in (input_dir, output_dir, scratch_dir):
            path.mkdir(parents=True)
        for relative_name, content in inputs.items():
            relative = PurePosixPath(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe Agent input path: {relative_name}")
            target = input_dir.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            target.chmod(0o444)
        for reference, source in resolved_references:
            relative = PurePosixPath(reference.path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe Agent input reference path: {reference.path}")
            target = input_dir.joinpath(*relative.parts)
            if target.exists():
                raise ValueError(f"Agent input reference path conflicts: {reference.path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError as error:
                if error.errno == errno.EXDEV:
                    raise ValueError(
                        f"durable Agent input requires one filesystem: {reference.path}"
                    ) from error
                raise
        for directory in sorted(
            (path for path in input_dir.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        input_dir.chmod(0o555)
        call_record = call_dir / "call.json"
        call_payload = self.call_record_payload(call)
        if call_record.exists():
            if json.loads(call_record.read_text(encoding="utf-8")) != call_payload:
                raise ValueError(f"Agent call {call.call_id} is immutable")
        else:
            call_record.write_text(
                json.dumps(call_payload, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        return attempt

    def _resolve_reference(self, reference: AgentContextReference) -> Path:
        prefix = "engine://trial-artifacts/"
        if self.trial_artifacts_root is None or not reference.source_ref.startswith(prefix):
            raise ValueError(f"unsupported durable Agent input: {reference.source_ref}")
        relative = PurePosixPath(reference.source_ref.removeprefix(prefix))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 6:
            raise ValueError(f"unsafe durable Agent input: {reference.source_ref}")
        root = self.trial_artifacts_root
        source = root.joinpath(*relative.parts)
        if (
            not source.is_file()
            or source.is_symlink()
            or not source.resolve().is_relative_to(root)
        ):
            raise ValueError(f"durable Agent input is unavailable: {reference.source_ref}")
        if source.stat().st_size != reference.size_bytes:
            raise ValueError(f"durable Agent input size mismatch: {reference.path}")
        return source

    @staticmethod
    def call_payload(call: AgentCall) -> dict[str, object]:
        return {
            "call_id": call.call_id,
            "run_id": call.run_id,
            "owner_scope": _scope_from_subject_ref(call.owner_subject_ref),
            "owner_subject_ref": call.owner_subject_ref,
            "target_scope_kind": call.scope.value,
            "target_scope": _scope_from_subject_ref(call.target_subject_ref),
            "target_subject_ref": call.target_subject_ref,
            "coordinator_id": call.coordinator_id,
            "plan_id": call.plan_id,
            "trial_id": call.trial_id,
            "role": call.role.value,
            "skill_id": call.skill_id,
            "subject_id": call.subject_id,
            "basis_revision": call.basis_revision,
            "max_retries": call.max_retries,
            "session_id": call.session_id,
        }

    def call_record_payload(self, call: AgentCall) -> dict[str, object]:
        payload = self.call_payload(call)
        if call.session_id is not None:
            payload["session_record_path"] = str(
                (self._session_dir_for_call(call) / "session.json").resolve()
            )
        return payload

    def _session_dir_for_call(self, call: AgentCall) -> Path:
        if self.run_layout is None:
            return self.calls_root / call.session_id
        if call.role is AgentRole.COORDINATOR:
            subject_id = str(call.coordinator_id)
            coordinator_id = call.coordinator_id
            plan_id = None
        elif call.role in {
            AgentRole.ARTIFACT_BUILDER,
            AgentRole.ANALYZER,
            AgentRole.PLAN_SUMMARIZER,
        }:
            subject_id = str(call.plan_id)
            coordinator_id = call.coordinator_id
            plan_id = call.plan_id
        else:
            subject_id = call.run_id
            coordinator_id = None
            plan_id = None
        session = AgentSession(
            session_id=call.session_id,
            run_id=call.run_id,
            role=call.role,
            subject_id=subject_id,
            coordinator_id=coordinator_id,
            plan_id=plan_id,
        )
        return self.session_dir(session)

    def attempt_contract(
        self,
        call: AgentCall,
        number: int,
        kind: AttemptKind,
    ) -> AgentCallAttempt:
        label = self.attempt_label(call, number, kind)
        if self.run_layout is not None:
            call_uri = self.run_layout.agent_call_uri(call)
        else:
            call_uri = f"run://agent-calls/{call.call_id}"
        return AgentCallAttempt(
            attempt_id=f"{call.call_id}/{label}",
            call_id=call.call_id,
            number=number,
            kind=kind,
            workspace_uri=f"{call_uri}/attempts/{label}",
            reflection_index=call.reflection_index,
        )

    @staticmethod
    def attempt_label(
        call: AgentCall,
        number: int,
        kind: AttemptKind,
    ) -> str:
        if call.reflection_index == 0:
            return f"{number:03d}-{kind.value}"
        return f"r{call.reflection_index:03d}-a{number:03d}-{kind.value}"

    def attempt_dir(
        self,
        call: AgentCall,
        number: int,
        kind: AttemptKind,
    ) -> Path:
        return self.call_dir(call) / "attempts" / self.attempt_label(
            call, number, kind
        )

    def record_attempt_result(
        self,
        attempt: Path,
        report: ValidationReport,
    ) -> None:
        payload = {
            "schema_version": "1",
            "status": "accepted" if report.ok else "validation_failed",
            "violations": [
                {
                    "code": item.code,
                    "message": item.message,
                    "path": item.path,
                    "repairable": item.repairable,
                }
                for item in report.violations
            ],
        }
        self._write_json(attempt / "gate-result.json", payload)

    def record_worker_error(self, attempt: Path, error: Exception) -> None:
        self._write_json(
            attempt / "scratch" / "worker-error.json",
            {
                "schema_version": "1",
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )

    def record_attempt_heartbeat(
        self,
        attempt: Path,
        *,
        status: str,
        observed_at: float,
    ) -> None:
        if status not in {"running", "completed", "failed"}:
            raise ValueError("invalid Agent heartbeat status")
        self._write_json(
            attempt / "scratch" / "heartbeat.json",
            {
                "schema_version": "1",
                "status": status,
                "observed_at": float(observed_at),
            },
        )

    @staticmethod
    def load_attempt_heartbeat(attempt: Path) -> float | None:
        path = attempt / "scratch" / "heartbeat.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            observed_at = float(payload["observed_at"])
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError("persisted Agent heartbeat is invalid") from error
        return observed_at

    def record_receipt(
        self,
        call: AgentCall,
        attempt: AgentCallAttempt,
        report: ValidationReport,
    ) -> None:
        payload = {
            "schema_version": "1",
            "call_id": call.call_id,
            "attempt_id": attempt.attempt_id,
            "status": "accepted" if report.ok else "rejected",
            "owner_scope": _scope_from_subject_ref(call.owner_subject_ref),
            "owner_subject_ref": call.owner_subject_ref,
            "target_scope": _scope_from_subject_ref(call.target_subject_ref),
            "target_subject_ref": call.target_subject_ref,
            "violations": [
                {
                    "code": item.code,
                    "message": item.message,
                    "path": item.path,
                    "repairable": item.repairable,
                }
                for item in report.violations
            ],
        }
        self._write_json(self.call_dir(call) / "receipt.json", payload)
    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _scope(
    run_id: str,
    coordinator_id: str | None,
    plan_id: str | None,
    trial_id: str | None,
) -> dict[str, str]:
    values = (run_id, coordinator_id, plan_id, trial_id)
    names = ("run_id", "coordinator_id", "plan_id", "trial_id")
    return {
        name: value
        for name, value in zip(names, values, strict=True)
        if value is not None
    }


def _scope_from_subject_ref(reference: str) -> dict[str, str]:
    return dict(
        zip(
            ("run_id", "coordinator_id", "plan_id", "trial_id"),
            reference.split("/"),
            strict=False,
        )
    )
