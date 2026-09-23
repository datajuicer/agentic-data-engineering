"""Human-only operator evaluation ledger state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OperatorEvaluationStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class OperatorEvaluationRecord:
    target_id: str
    target_kind: str
    artifact_ref_id: str | None
    artifact_digest: str | None
    profile_digest: str
    status: OperatorEvaluationStatus
    coordinator_id: str | None = None
    plan_id: str | None = None
    trial_id: str | None = None
    command_id: str | None = None
    logical_command_id: str | None = None
    attempt_id: str | None = None
    attempt_index: int = 0
    retry_pending: bool = False
    failed_attempt_receipt_ids: tuple[str, ...] = ()
    receipt_id: str | None = None
    result_ref: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.target_kind not in {"base", "trial"}:
            raise ValueError("operator target kind is invalid")
        if not self.target_id or not self.profile_digest:
            raise ValueError("operator target identity is required")
        if self.target_kind == "trial" and (
            not self.coordinator_id or not self.plan_id or not self.trial_id
        ):
            raise ValueError("Trial operator target requires Trial identity")
        if self.status is OperatorEvaluationStatus.PENDING:
            if not self.logical_command_id or self.attempt_index < 1:
                raise ValueError(
                    "pending operator target requires logical Command and Attempt"
                )
            if self.retry_pending:
                if self.command_id is not None:
                    raise ValueError(
                        "retry-pending operator target cannot retain active command"
                    )
            elif not self.command_id or not self.attempt_id:
                raise ValueError("pending operator target requires active Attempt")
        elif self.retry_pending:
            raise ValueError("terminal operator target cannot be retry-pending")
