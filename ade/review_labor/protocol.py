"""Durable Harness-to-Review-Worker command protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


@dataclass(frozen=True)
class ReviewBatch:
    batch_id: str
    pool: str
    investigation_purpose: str
    units: tuple[dict[str, Any], ...]
    rubrics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ReviewCommand:
    command_id: str
    logical_command_id: str
    attempt_id: str
    attempt_index: int
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    basis_revision: int
    batches: tuple[ReviewBatch, ...]
    pool_requirements: tuple[dict[str, Any], ...] = ()


class ReviewReceiptStatus(StrEnum):
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


@dataclass(frozen=True)
class ReviewReceipt:
    receipt_id: str
    command_id: str
    logical_command_id: str
    attempt_id: str
    attempt_index: int
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    status: ReviewReceiptStatus
    packet: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _with_scope(asdict(self))


def encode_command(command: ReviewCommand) -> dict[str, Any]:
    return _with_scope(asdict(command))


def decode_command(value: dict[str, Any]) -> ReviewCommand:
    _require_scope(value)
    return ReviewCommand(
        command_id=str(value["command_id"]),
        logical_command_id=str(value["logical_command_id"]),
        attempt_id=str(value["attempt_id"]),
        attempt_index=int(value["attempt_index"]),
        run_id=str(value["run_id"]),
        coordinator_id=str(value["coordinator_id"]),
        plan_id=str(value["plan_id"]),
        trial_id=str(value["trial_id"]),
        basis_revision=int(value["basis_revision"]),
        batches=tuple(
            ReviewBatch(
                batch_id=str(item["batch_id"]),
                pool=str(item["pool"]),
                investigation_purpose=str(item["investigation_purpose"]),
                units=tuple(dict(unit) for unit in item["units"]),
                rubrics=tuple(dict(rubric) for rubric in item["rubrics"]),
            )
            for item in value["batches"]
        ),
        pool_requirements=tuple(
            dict(item) for item in value.get("pool_requirements", ())
        ),
    )


def decode_receipt(value: dict[str, Any]) -> ReviewReceipt:
    _require_scope(value)
    return ReviewReceipt(
        receipt_id=str(value["receipt_id"]),
        command_id=str(value["command_id"]),
        logical_command_id=str(value["logical_command_id"]),
        attempt_id=str(value["attempt_id"]),
        attempt_index=int(value["attempt_index"]),
        run_id=str(value["run_id"]),
        coordinator_id=str(value["coordinator_id"]),
        plan_id=str(value["plan_id"]),
        trial_id=str(value["trial_id"]),
        status=ReviewReceiptStatus(str(value["status"])),
        packet=dict(value.get("packet", {})),
        error=(str(value["error"]) if value.get("error") is not None else None),
    )


def _with_scope(value: dict[str, Any]) -> dict[str, Any]:
    ref = (
        f"{value['run_id']}/{value['coordinator_id']}/"
        f"{value['plan_id']}/{value['trial_id']}"
    )
    return {
        **value,
        "scope": {
            "run_id": value["run_id"],
            "coordinator_id": value["coordinator_id"],
            "plan_id": value["plan_id"],
            "trial_id": value["trial_id"],
        },
        "subject_ref": ref,
    }


def _require_scope(value: dict[str, Any]) -> None:
    expected = _with_scope(
        {
            key: value[key]
            for key in ("run_id", "coordinator_id", "plan_id", "trial_id")
        }
    )
    if (
        value.get("scope") != expected["scope"]
        or value.get("subject_ref") != expected["subject_ref"]
    ):
        raise ValueError("Review protocol requires canonical scope and SubjectRef")
