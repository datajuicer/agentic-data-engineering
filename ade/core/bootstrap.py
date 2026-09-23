"""Durable bootstrap state for Agent Search runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any


class BootstrapStatus(StrEnum):
    PENDING = "pending"
    BASE_EVALUATING = "base_evaluating"
    BASE_COMPLETED = "base_completed"
    P000_TRAINING = "p000_training"
    P000_EVALUATING = "p000_evaluating"
    P000_ANALYZING = "p000_analyzing"
    P000_PLAN_SUMMARIZING = "p000_plan_summarizing"
    P000_RUN_SUMMARIZING = "p000_run_summarizing"
    COMPLETED = "completed"
    FAILED = "failed"


class BaseEvaluationStatus(StrEnum):
    COMPLETE = "complete"
    COMPLETED_DEGRADED = "completed_degraded"


@dataclass(frozen=True)
class BaseEvaluationRecord:
    status: BaseEvaluationStatus
    profile_status: dict[str, str]
    receipt_ids: tuple[str, ...] = ()
    result_refs: tuple[str, ...] = ()
    online_result_ref: str | None = None
    offline_evidence_ref_id: str | None = None
    offline_score: float | None = None
    offline_secondary_score: float | None = None

    def __post_init__(self) -> None:
        if not self.profile_status:
            raise ValueError("Base evaluation record requires profile status")
        if self.offline_score is not None and not math.isfinite(self.offline_score):
            raise ValueError("Base evaluation offline score must be finite")
        if (
            self.offline_secondary_score is not None
            and not math.isfinite(self.offline_secondary_score)
        ):
            raise ValueError("Base evaluation offline secondary score must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "profile_status": dict(self.profile_status),
            "receipt_ids": list(self.receipt_ids),
            "result_refs": list(self.result_refs),
            "online_result_ref": self.online_result_ref,
            "offline_evidence_ref_id": self.offline_evidence_ref_id,
            "offline_score": self.offline_score,
            "offline_secondary_score": self.offline_secondary_score,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "BaseEvaluationRecord":
        return cls(
            status=BaseEvaluationStatus(str(value["status"])),
            profile_status={
                str(key): str(status)
                for key, status in dict(value["profile_status"]).items()
            },
            receipt_ids=tuple(str(item) for item in value.get("receipt_ids", ())),
            result_refs=tuple(str(item) for item in value.get("result_refs", ())),
            online_result_ref=(
                str(value["online_result_ref"])
                if value.get("online_result_ref") is not None
                else None
            ),
            offline_evidence_ref_id=(
                str(value["offline_evidence_ref_id"])
                if value.get("offline_evidence_ref_id") is not None
                else None
            ),
            offline_score=(
                float(value["offline_score"])
                if value.get("offline_score") is not None
                else None
            ),
            offline_secondary_score=(
                float(value["offline_secondary_score"])
                if value.get("offline_secondary_score") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class BootstrapState:
    enabled: bool = False
    reference_enabled: bool = False
    reference_run_id: str | None = None
    reference_revision: int | None = None
    stop_after_baseline: bool = False
    status: BootstrapStatus = BootstrapStatus.COMPLETED
    command_ids: tuple[str, ...] = ()
    command_status: dict[str, str] = field(default_factory=dict)
    command_logical_ids: dict[str, str] = field(default_factory=dict)
    command_attempt_ids: dict[str, str] = field(default_factory=dict)
    command_attempt_indices: dict[str, int] = field(default_factory=dict)
    base_evaluation: BaseEvaluationRecord | None = None
    baseline_revision: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.enabled and self.status is not BootstrapStatus.COMPLETED:
            raise ValueError("disabled bootstrap must be completed")
        if self.reference_enabled and not self.enabled:
            raise ValueError("bootstrap reference requires enabled bootstrap")
        if self.reference_revision is not None and self.reference_revision < 0:
            raise ValueError("reference_revision must be non-negative")
        if self.baseline_revision is not None and self.baseline_revision < 0:
            raise ValueError("baseline_revision must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "reference_enabled": self.reference_enabled,
            "reference_run_id": self.reference_run_id,
            "reference_revision": self.reference_revision,
            "stop_after_baseline": self.stop_after_baseline,
            "status": self.status.value,
            "command_ids": list(self.command_ids),
            "command_status": dict(self.command_status),
            "command_logical_ids": dict(self.command_logical_ids),
            "command_attempt_ids": dict(self.command_attempt_ids),
            "command_attempt_indices": dict(self.command_attempt_indices),
            "base_evaluation": (
                self.base_evaluation.to_dict()
                if self.base_evaluation is not None
                else None
            ),
            "baseline_revision": self.baseline_revision,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "BootstrapState":
        return cls(
            enabled=bool(value.get("enabled", False)),
            reference_enabled=bool(value.get("reference_enabled", False)),
            reference_run_id=(
                str(value["reference_run_id"])
                if value.get("reference_run_id") is not None
                else None
            ),
            reference_revision=(
                int(value["reference_revision"])
                if value.get("reference_revision") is not None
                else None
            ),
            stop_after_baseline=bool(value.get("stop_after_baseline", False)),
            status=BootstrapStatus(str(value.get("status", "completed"))),
            command_ids=tuple(str(item) for item in value.get("command_ids", ())),
            command_status={str(k): str(v) for k, v in dict(value.get("command_status", {})).items()},
            command_logical_ids={
                str(k): str(v)
                for k, v in dict(value.get("command_logical_ids", {})).items()
            },
            command_attempt_ids={
                str(k): str(v)
                for k, v in dict(value.get("command_attempt_ids", {})).items()
            },
            command_attempt_indices={
                str(k): int(v)
                for k, v in dict(value.get("command_attempt_indices", {})).items()
            },
            base_evaluation=(
                BaseEvaluationRecord.from_dict(dict(value["base_evaluation"]))
                if value.get("base_evaluation") is not None
                else None
            ),
            baseline_revision=(int(value["baseline_revision"]) if value.get("baseline_revision") is not None else None),
            error=str(value["error"]) if value.get("error") is not None else None,
            metadata=dict(value.get("metadata", {})),
        )
