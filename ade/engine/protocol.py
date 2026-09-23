"""Serialization boundary for typed Engine commands and receipts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ade.core.engine import (
    EngineCommand,
    EngineReceipt,
    EngineReceiptStatus,
    EvaluateCommand,
    TrainRFTCommand,
    TrainSFTCommand,
)

_COMMAND_TYPES = {
    "evaluate": EvaluateCommand,
    "train_sft": TrainSFTCommand,
    "train_rft": TrainRFTCommand,
}


def encode_command(command: EngineCommand) -> dict[str, Any]:
    if type(command) not in set(_COMMAND_TYPES.values()):
        raise ValueError(f"unsupported Engine command type: {type(command).__name__}")
    return _with_scope(asdict(command))


def decode_command(payload: dict[str, Any]) -> EngineCommand:
    _require_scope(payload)
    kind = payload.get("kind")
    command_type = _COMMAND_TYPES.get(kind)
    if command_type is None:
        raise ValueError(f"unsupported Engine command: {kind}")
    allowed = {
        "command_id",
        "run_id",
        "coordinator_id",
        "plan_id",
        "trial_id",
        "input_ref",
        "output_uri",
        "logical_command_id",
        "attempt_id",
        "attempt_index",
        "kind",
        "scope",
        "subject_ref",
    }
    unexpected = set(payload) - allowed
    if unexpected:
        raise ValueError(f"unexpected Engine command fields: {sorted(unexpected)}")
    return command_type(
        command_id=str(payload["command_id"]),
        run_id=str(payload["run_id"]),
        coordinator_id=str(payload["coordinator_id"]),
        plan_id=str(payload["plan_id"]),
        trial_id=str(payload["trial_id"]),
        input_ref=str(payload["input_ref"]),
        output_uri=str(payload.get("output_uri") or ""),
        logical_command_id=str(
            payload["logical_command_id"]
            if "logical_command_id" in payload
            else payload["command_id"]
        ),
        attempt_id=str(payload.get("attempt_id") or "attempt-001"),
        attempt_index=int(payload.get("attempt_index") or 1),
    )


def encode_receipt(receipt: EngineReceipt) -> dict[str, Any]:
    return _with_scope(asdict(receipt))


def decode_receipt(payload: dict[str, Any]) -> EngineReceipt:
    _require_scope(payload)
    return EngineReceipt(
        receipt_id=str(payload["receipt_id"]),
        command_id=str(payload["command_id"]),
        run_id=str(payload["run_id"]),
        coordinator_id=str(payload["coordinator_id"]),
        plan_id=str(payload["plan_id"]),
        trial_id=str(payload["trial_id"]),
        status=EngineReceiptStatus(str(payload["status"])),
        output_refs=tuple(payload.get("output_refs", ())),
        error=payload.get("error"),
        logical_command_id=str(
            payload["logical_command_id"]
            if "logical_command_id" in payload
            else payload["command_id"]
        ),
        attempt_id=str(payload.get("attempt_id") or "attempt-001"),
        attempt_index=int(payload.get("attempt_index") or 1),
        failure_kind=payload.get("failure_kind"),
        retryable=bool(payload.get("retryable", False)),
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
        raise ValueError("Engine protocol requires canonical scope and SubjectRef")
