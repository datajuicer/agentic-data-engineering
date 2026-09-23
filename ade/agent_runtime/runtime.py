"""Agent Call Attempt execution, validation, recovery, and retry preparation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
import re
from pathlib import Path
import threading
import time
from typing import Protocol

from ade.agent_runtime.delivery import DeliveryGate
from ade.agent_runtime.input_package import AgentInputPackage
from ade.agent_runtime.skills import ResolvedSkill, SkillResolver
from ade.agent_runtime.workspace import WorkspaceManager
from ade.core.agent import AgentCall, AgentCallAttempt, AgentSession, AttemptKind
from ade.core.validation import DeliveryViolation, ValidationReport
from ade.tasks.contracts import (
    AgentRoleContract,
    AgentRoleOutput,
    PlanSummary,
    PlanningDecision,
    RoleOutputError,
    RunSummary,
)


class AgentBackend(Protocol):
    def invoke(
        self,
        *,
        skill: ResolvedSkill,
        attempt: Path,
        feedback_paths: tuple[str, ...],
    ) -> None: ...


@dataclass(frozen=True)
class AcceptedCall:
    call: AgentCall
    attempt: AgentCallAttempt
    workspace: Path
    validation: ValidationReport
    output: AgentRoleOutput
    session: AgentSession | None = None


@dataclass(frozen=True)
class RejectedCall:
    call: AgentCall
    attempt: AgentCallAttempt
    workspace: Path
    validation: ValidationReport
    output: AgentRoleOutput | None
    session: AgentSession | None = None


class AgentRuntime:
    def __init__(
        self,
        *,
        skills: SkillResolver,
        workspaces: WorkspaceManager,
        backend: AgentBackend,
        heartbeat_interval_seconds: float = 30.0,
    ) -> None:
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        self.skills = skills
        self.workspaces = workspaces
        self.backend = backend
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)

    def run(
        self,
        call: AgentCall,
        package: AgentInputPackage,
        gate: DeliveryGate,
        contract: AgentRoleContract,
    ) -> AcceptedCall | RejectedCall:
        if not isinstance(package, AgentInputPackage):
            raise TypeError("AgentRuntime requires AgentInputPackage")
        package.validate()
        self._validate_binding(call, package, contract)
        return self._run_attempts(
            call,
            package,
            gate,
            contract,
            start_number=0,
            previous=None,
            report=ValidationReport(),
        )

    def retry(
        self,
        call: AgentCall,
        package: AgentInputPackage,
        gate: DeliveryGate,
        contract: AgentRoleContract,
        previous: AcceptedCall,
        report: ValidationReport,
    ) -> AcceptedCall | RejectedCall:
        if not isinstance(package, AgentInputPackage):
            raise TypeError("AgentRuntime requires AgentInputPackage")
        package.validate()
        self._validate_binding(call, package, contract)
        if previous.call != call or previous.attempt.call_id != call.call_id:
            raise ValueError("retry attempt belongs to another Agent Call")
        if report.ok or any(not item.repairable for item in report.violations):
            raise ValueError("external retry requires Agent-correctable feedback")
        return self._run_attempts(
            call,
            package,
            gate,
            contract,
            start_number=previous.attempt.number + 1,
            previous=previous,
            report=report,
        )

    def prepare_retry_attempt(
        self,
        call: AgentCall,
        package: AgentInputPackage,
        contract: AgentRoleContract,
        previous: AcceptedCall | RejectedCall,
        report: ValidationReport,
    ) -> tuple[AgentCallAttempt, Path]:
        package.validate()
        self._validate_binding(call, package, contract)
        if previous.call != call or previous.attempt.call_id != call.call_id:
            raise ValueError("retry attempt belongs to another Agent Call")
        if report.ok or any(not item.repairable for item in report.violations):
            raise ValueError("Agent retry requires repairable feedback")
        number = previous.attempt.number + 1
        if number > call.max_retries:
            raise ValueError("Agent retry budget is exhausted")
        retry_package = self._retry_attempt_package(
            package,
            previous_workspace=previous.workspace,
            previous_attempt=previous.attempt,
            report=report,
        )
        path = self.workspaces.create_attempt(
            call,
            number,
            AttemptKind.RETRY,
            retry_package.workspace_inputs(),
            retry_package.references,
        )
        self._carry_attempt_audits(previous.workspace, path)
        return (
            self.workspaces.attempt_contract(call, number, AttemptKind.RETRY),
            path,
        )

    def _run_attempts(
        self,
        call: AgentCall,
        package: AgentInputPackage,
        gate: DeliveryGate,
        contract: AgentRoleContract,
        *,
        start_number: int,
        previous: AcceptedCall | RejectedCall | None,
        report: ValidationReport,
    ) -> AcceptedCall | RejectedCall:
        skill = self.skills.resolve(call.skill_id)
        if skill.allow_implicit_invocation:
            raise ValueError(f"Skill must disable implicit invocation: {call.skill_id}")
        feedback_paths = tuple(
            sorted({item.path for item in report.violations if item.path is not None})
        )
        attempt_contract = previous.attempt if previous is not None else None
        attempt_path = previous.workspace if previous is not None else None
        output = previous.output if previous is not None else None
        for number in range(start_number, call.max_retries + 1):
            kind = AttemptKind.INITIAL if number == 0 else AttemptKind.RETRY
            attempt_package = package
            if attempt_path is not None:
                assert attempt_contract is not None
                attempt_package = self._retry_attempt_package(
                    package,
                    previous_workspace=attempt_path,
                    previous_attempt=attempt_contract,
                    report=report,
                )
            previous_attempt_path = attempt_path
            attempt_path = self.workspaces.create_attempt(
                call,
                number,
                kind,
                attempt_package.workspace_inputs(),
                attempt_package.references,
            )
            if previous_attempt_path is not None:
                self._carry_attempt_audits(previous_attempt_path, attempt_path)
            attempt_contract = self.workspaces.attempt_contract(call, number, kind)
            self._invoke_backend(
                skill=skill,
                attempt=attempt_path,
                feedback_paths=feedback_paths,
            )
            report, output = self._validate_attempt(
                call,
                attempt_path,
                gate,
                contract,
            )
            self.workspaces.record_attempt_result(attempt_path, report)
            if report.ok:
                assert output is not None
                self.workspaces.record_receipt(call, attempt_contract, report)
                return AcceptedCall(call, attempt_contract, attempt_path, report, output)
            if any(not item.repairable for item in report.violations):
                self.workspaces.record_receipt(call, attempt_contract, report)
                return RejectedCall(call, attempt_contract, attempt_path, report, output)
            feedback_paths = tuple(
                sorted({item.path for item in report.violations if item.path is not None})
            )
        if attempt_contract is None or attempt_path is None:
            raise ValueError("Agent Call has no remaining retry attempts")
        self.workspaces.record_receipt(call, attempt_contract, report)
        return RejectedCall(call, attempt_contract, attempt_path, report, output)

    def recover(
        self,
        call: AgentCall,
        package: AgentInputPackage,
        gate: DeliveryGate,
        contract: AgentRoleContract,
    ) -> AcceptedCall | RejectedCall:
        if not isinstance(package, AgentInputPackage):
            raise TypeError("AgentRuntime requires AgentInputPackage")
        package.validate()
        self._validate_binding(call, package, contract)
        call_dir = self.workspaces.call_dir(call).resolve()
        self._validate_call_record(call, call_dir)
        attempts = sorted(
            (
                path
                for path in (call_dir / "attempts").glob("*")
                if path.is_dir()
                and (identity := self._attempt_identity(path.name)) is not None
                and identity[0] == call.reflection_index
            ),
            reverse=True,
        )
        if not attempts:
            raise FileNotFoundError(f"no persisted attempts for {call.call_id}")
        attempt_path = attempts[0]
        identity = self._attempt_identity(attempt_path.name)
        assert identity is not None
        _reflection_index, number, kind = identity
        if number > call.max_retries:
            raise ValueError("persisted attempt exceeds Agent Call retry budget")
        self._validate_persisted_inputs(attempt_path, package)
        attempt = self.workspaces.attempt_contract(call, number, kind)
        report, output = self._validate_attempt(
            call,
            attempt_path,
            gate,
            contract,
        )
        if report.ok:
            assert output is not None
            self.workspaces.record_attempt_result(attempt_path, report)
            self.workspaces.record_receipt(call, attempt, report)
            return AcceptedCall(call, attempt, attempt_path, report, output)
        if not self._invocation_completed(attempt_path):
            feedback_paths = self._persisted_feedback_paths(attempt_path)
            self._invoke_backend(
                skill=self.skills.resolve(call.skill_id),
                attempt=attempt_path,
                feedback_paths=feedback_paths,
                resume_submitted=True,
            )
            report, output = self._validate_attempt(
                call,
                attempt_path,
                gate,
                contract,
            )
            if report.ok:
                assert output is not None
                self.workspaces.record_attempt_result(attempt_path, report)
                self.workspaces.record_receipt(call, attempt, report)
                return AcceptedCall(call, attempt, attempt_path, report, output)
        self.workspaces.record_attempt_result(attempt_path, report)
        self.workspaces.record_receipt(call, attempt, report)
        return RejectedCall(call, attempt, attempt_path, report, output)

    def revalidate(
        self,
        call: AgentCall,
        package: AgentInputPackage,
        gate: DeliveryGate,
        contract: AgentRoleContract,
    ) -> AcceptedCall | RejectedCall:
        """Re-admit the latest persisted Attempt without invoking its backend."""
        if not isinstance(package, AgentInputPackage):
            raise TypeError("AgentRuntime requires AgentInputPackage")
        package.validate()
        self._validate_binding(call, package, contract)
        call_dir = self.workspaces.call_dir(call).resolve()
        self._validate_call_record(call, call_dir)
        attempts = sorted(
            (
                path
                for path in (call_dir / "attempts").glob("*")
                if path.is_dir()
                and (identity := self._attempt_identity(path.name)) is not None
                and identity[0] == call.reflection_index
            ),
            reverse=True,
        )
        if not attempts:
            raise FileNotFoundError(f"no persisted attempts for {call.call_id}")
        attempt_path = attempts[0]
        identity = self._attempt_identity(attempt_path.name)
        assert identity is not None
        _reflection_index, number, kind = identity
        if number > call.max_retries:
            raise ValueError("persisted attempt exceeds Agent Call retry budget")
        self._validate_persisted_inputs(attempt_path, package)
        attempt = self.workspaces.attempt_contract(call, number, kind)
        report, output = self._validate_attempt(call, attempt_path, gate, contract)
        self.workspaces.record_attempt_result(attempt_path, report)
        self.workspaces.record_receipt(call, attempt, report)
        if report.ok:
            assert output is not None
            return AcceptedCall(call, attempt, attempt_path, report, output)
        return RejectedCall(call, attempt, attempt_path, report, output)

    def inspect(
        self,
        call: AgentCall,
        package: AgentInputPackage,
        gate: DeliveryGate,
        contract: AgentRoleContract,
    ) -> AcceptedCall | RejectedCall | None:
        """Load a terminal Attempt without dispatching its backend.

        Control uses this read-only side of the Agent Port.  The independent
        Agent worker is the only production caller allowed to dispatch.
        """
        package.validate()
        self._validate_binding(call, package, contract)
        call_dir = self.workspaces.call_dir(call).resolve()
        self._validate_call_record(call, call_dir)
        attempts = sorted(
            (
                path
                for path in (call_dir / "attempts").glob("*")
                if path.is_dir()
                and (identity := self._attempt_identity(path.name)) is not None
                and identity[0] == call.reflection_index
            ),
            reverse=True,
        )
        if not attempts:
            raise FileNotFoundError(f"no persisted attempts for {call.call_id}")
        attempt_path = attempts[0]
        identity = self._attempt_identity(attempt_path.name)
        assert identity is not None
        _reflection_index, number, kind = identity
        self._validate_persisted_inputs(attempt_path, package)
        receipt = call_dir / "receipt.json"
        if not receipt.is_file():
            return None
        attempt = self.workspaces.attempt_contract(call, number, kind)
        try:
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("persisted Agent receipt is invalid") from error
        if not isinstance(receipt_payload, dict) or receipt_payload.get(
            "call_id"
        ) != call.call_id:
            raise ValueError("persisted Agent receipt identity mismatch")
        if receipt_payload.get("attempt_id") != attempt.attempt_id:
            # A terminal receipt for the preceding Attempt remains until the
            # newly committed retry is dispatched. It is not terminal evidence
            # for the active Attempt.
            return None
        if receipt_payload.get("status") == "rejected":
            violations = receipt_payload.get("violations")
            if not isinstance(violations, list) or not violations:
                raise ValueError("rejected Agent receipt has no violations")
            report = ValidationReport(
                tuple(
                    DeliveryViolation(
                        str(item["code"]),
                        str(item["message"]),
                        item.get("path"),
                        bool(item.get("repairable", False)),
                    )
                    for item in violations
                    if isinstance(item, dict)
                )
            )
            if not report.violations:
                raise ValueError("rejected Agent receipt violations are invalid")
            return RejectedCall(call, attempt, attempt_path, report, None)
        if receipt_payload.get("status") != "accepted":
            raise ValueError("persisted Agent receipt status is invalid")
        report, output = self._validate_attempt(call, attempt_path, gate, contract)
        self.workspaces.record_attempt_result(attempt_path, report)
        if report.ok:
            assert output is not None
            return AcceptedCall(call, attempt, attempt_path, report, output)
        return RejectedCall(call, attempt, attempt_path, report, output)

    def _invoke_backend(
        self,
        *,
        skill: ResolvedSkill,
        attempt: Path,
        feedback_paths: tuple[str, ...],
        resume_submitted: bool = False,
    ) -> None:
        marker = self._invocation_marker(attempt)
        if marker.exists():
            if not resume_submitted or self._invocation_completed(attempt):
                raise ValueError("Agent Attempt dispatch intent already exists")
        marker.write_text(
            '{"schema_version":"1","status":"submitted"}\n',
            encoding="utf-8",
        )
        stopped = threading.Event()
        heartbeat_errors: list[Exception] = []

        def maintain_heartbeat() -> None:
            while not stopped.wait(self.heartbeat_interval_seconds):
                try:
                    self.workspaces.record_attempt_heartbeat(
                        attempt,
                        status="running",
                        observed_at=time.time(),
                    )
                except Exception as error:  # surfaced after backend returns
                    heartbeat_errors.append(error)
                    stopped.set()

        self.workspaces.record_attempt_heartbeat(
            attempt,
            status="running",
            observed_at=time.time(),
        )
        heartbeat = threading.Thread(
            target=maintain_heartbeat,
            name=f"ade-agent-heartbeat-{attempt.name}",
            daemon=True,
        )
        heartbeat.start()
        try:
            self.backend.invoke(
                skill=skill,
                attempt=attempt,
                feedback_paths=feedback_paths,
            )
        except Exception:
            self.workspaces.record_attempt_heartbeat(
                attempt,
                status="failed",
                observed_at=time.time(),
            )
            raise
        finally:
            stopped.set()
            heartbeat.join()
        if heartbeat_errors:
            raise RuntimeError("Agent heartbeat persistence failed") from heartbeat_errors[0]
        self.workspaces.record_attempt_heartbeat(
            attempt,
            status="completed",
            observed_at=time.time(),
        )
        marker.write_text(
            '{"schema_version":"1","status":"completed"}\n',
            encoding="utf-8",
        )

    @staticmethod
    def _invocation_marker(attempt: Path) -> Path:
        return attempt / "scratch" / "invocation-completed.json"

    @classmethod
    def _invocation_completed(cls, attempt: Path) -> bool:
        """Return whether the backend invocation reached its durable boundary.

        The marker is created before dispatch with ``status=submitted``.  Its
        existence therefore only proves dispatch intent, not completion.  A
        worker that dies after creating that marker must be allowed to resume
        the same Attempt; only the terminal marker status suppresses replay.
        """
        marker = cls._invocation_marker(attempt)
        if not marker.is_file():
            return False
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return isinstance(payload, dict) and payload.get("status") == "completed"

    @staticmethod
    def _persisted_feedback_paths(attempt: Path) -> tuple[str, ...]:
        feedback = attempt / "input" / "retry.json"
        if not feedback.is_file():
            return ()
        try:
            payload = json.loads(feedback.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("persisted Agent retry feedback is invalid") from error
        paths = payload.get("feedback_paths") if isinstance(payload, dict) else None
        if not isinstance(paths, list) or any(
            not isinstance(path, str) or not path for path in paths
        ):
            raise ValueError("persisted Agent feedback paths are invalid")
        return tuple(sorted(set(paths)))

    def _validate_attempt(
        self,
        call: AgentCall,
        attempt_path: Path,
        gate: DeliveryGate,
        contract: AgentRoleContract,
    ) -> tuple[ValidationReport, AgentRoleOutput | None]:
        report = gate.validate(attempt_path)
        output: AgentRoleOutput | None = None
        if report.ok:
            try:
                output = contract.decode_output(attempt_path)
            except RoleOutputError as error:
                report = ValidationReport((error.violation,))
            else:
                report = contract.validate_output(output)
                if report.ok:
                    report = self._validate_output_binding(call, output)
                if report.ok:
                    contract.finalize_output(attempt_path, output)
        return report, output

    @staticmethod
    def _validate_binding(
        call: AgentCall,
        package: AgentInputPackage,
        contract: AgentRoleContract,
    ) -> None:
        expected = (
            call.run_id,
            call.role,
            call.skill_id,
            call.subject_id,
            call.basis_revision,
        )
        package_identity = (
            package.run_id,
            package.role,
            package.skill_id,
            package.subject_id,
            package.basis_revision,
        )
        contract_identity = (contract.role, contract.skill_id)
        if expected != package_identity:
            raise ValueError("Agent Call identity does not match input package")
        if contract.task_id != package.task_id or contract_identity != (
            call.role,
            call.skill_id,
        ):
            raise ValueError("Agent Call identity does not match role contract")

    def _validate_call_record(self, call: AgentCall, call_dir: Path) -> None:
        try:
            record = json.loads((call_dir / "call.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("persisted Agent Call record is invalid") from error
        expected = self.workspaces.call_record_payload(call)
        if record != expected:
            raise ValueError("persisted Agent Call record does not match call")

    @staticmethod
    def _attempt_identity(
        label: str,
    ) -> tuple[int, int, AttemptKind] | None:
        match = re.fullmatch(
            r"(?:r(?P<reflection>\d{3})-a)?(?P<retry>\d{3})-(?P<kind>initial|retry)",
            label,
        )
        if match is None:
            return None
        try:
            return (
                int(match.group("reflection") or 0),
                int(match.group("retry")),
                AttemptKind(match.group("kind")),
            )
        except ValueError:
            return None

    def _validate_persisted_inputs(
        self,
        attempt_path: Path,
        package: AgentInputPackage,
    ) -> None:
        input_dir = attempt_path / "input"
        actual_files: dict[str, bytes] = {}
        references = {item.path: item for item in package.references}
        for path in input_dir.rglob("*"):
            if path.is_symlink():
                raise ValueError("persisted Agent input contains a symlink")
            if path.is_file():
                relative = path.relative_to(input_dir).as_posix()
                if relative in references:
                    continue
                actual_files[relative] = path.read_bytes()
        persisted = replace(package, files=tuple(sorted(actual_files.items())))
        persisted.validate()
        identity = AgentRuntime._attempt_identity(attempt_path.name)
        if identity is None:
            raise ValueError("persisted Agent attempt identity is invalid")
        _reflection_index, _number, kind = identity
        base_files = package.workspace_inputs()
        for relative_name, reference in references.items():
            path = input_dir / relative_name
            source = self.workspaces._resolve_reference(reference)
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != reference.size_bytes
                or not os.path.samefile(path, source)
            ):
                raise ValueError(
                    f"persisted Agent input reference changed: {relative_name}"
                )
        for relative_name, expected in base_files.items():
            if relative_name == "manifest.json":
                continue
            path = input_dir / relative_name
            try:
                actual = path.read_bytes()
            except OSError as error:
                raise ValueError(
                    f"persisted Agent input is missing: {relative_name}"
                ) from error
            if actual != expected:
                raise ValueError(f"persisted Agent input changed: {relative_name}")
        extras = set(actual_files) - set(base_files)
        if kind is AttemptKind.INITIAL:
            if extras or actual_files.get("manifest.json") != base_files["manifest.json"]:
                raise ValueError("initial Agent inputs do not match the immutable package")
            return
        if "retry.json" not in extras or any(
            path != "retry.json" and not path.startswith("prior-delivery/")
            for path in extras
        ):
            raise ValueError("retry Agent inputs contain an invalid overlay")

    @staticmethod
    def _retry_attempt_package(
        package: AgentInputPackage,
        *,
        previous_workspace: Path,
        previous_attempt: AgentCallAttempt,
        report: ValidationReport,
    ) -> AgentInputPackage:
        retry_paths = tuple(
            sorted({item.path for item in report.violations if item.path is not None})
        )
        overlay: dict[str, tuple[bytes, str]] = {}
        inherited_prior_output = previous_workspace / "input" / "prior-delivery"
        for source in inherited_prior_output.rglob("*"):
            if source.is_file() and not source.is_symlink():
                relative = source.relative_to(inherited_prior_output).as_posix()
                overlay[f"prior-delivery/{relative}"] = (
                    source.read_bytes(),
                    f"{previous_attempt.workspace_uri}#input/prior-delivery/{relative}",
                )
        prior_output = previous_workspace / "output"
        for source in prior_output.rglob("*"):
            if source.is_file() and not source.is_symlink():
                relative = source.relative_to(prior_output).as_posix()
                overlay[f"prior-delivery/{relative}"] = (
                    source.read_bytes(),
                    f"{previous_attempt.workspace_uri}#output/{relative}",
                )
        retry_content = (
            json.dumps(
                {
                    "feedback_paths": list(retry_paths),
                    "violations": [
                        {
                            "code": item.code,
                            "message": item.message,
                            "path": item.path,
                        }
                        for item in report.violations
                    ],
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()
        overlay["retry.json"] = (
            retry_content,
            f"{previous_attempt.workspace_uri}#validation",
        )
        return package.with_materialized_overlay(overlay)

    @staticmethod
    def _carry_attempt_audits(previous: Path, target: Path) -> None:
        for source_root, target_root in (
            (previous / "scratch" / "llm_judge", target / "scratch" / "llm_judge"),
            (
                previous / "harness" / "prior-reviews",
                target / "harness" / "prior-reviews",
            ),
            (
                previous / "scratch" / "reviews",
                target / "harness" / "prior-reviews" / previous.name,
            ),
        ):
            if not source_root.is_dir():
                continue
            for source in source_root.rglob("*"):
                if not source.is_file() or source.is_symlink():
                    continue
                relative = source.relative_to(source_root)
                destination = target_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                content = source.read_bytes()
                if destination.exists() and destination.read_bytes() != content:
                    raise ValueError(
                        f"conflicting carried Agent audit: {relative.as_posix()}"
                    )
                destination.write_bytes(content)

    @staticmethod
    def _validate_output_binding(
        call: AgentCall,
        output: AgentRoleOutput,
    ) -> ValidationReport:
        violations: list[DeliveryViolation] = []
        if isinstance(output, (PlanningDecision, PlanSummary, RunSummary)):
            if output.basis_revision != call.basis_revision:
                violations.append(
                    DeliveryViolation(
                        "output_basis_mismatch",
                        "output basis_revision does not match Agent Call",
                        f"{'decision' if isinstance(output, PlanningDecision) else 'summary'}.json#/basis_revision",
                        repairable=True,
                    )
                )
        if isinstance(output, (PlanSummary, RunSummary)) and output.subject_id != call.subject_id:
            violations.append(
                DeliveryViolation(
                    "output_subject_mismatch",
                    "output subject_id does not match Agent Call",
                    "summary.json#/subject_id",
                    repairable=True,
                )
            )
        return ValidationReport(tuple(violations))
