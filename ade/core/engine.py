"""Typed Engine commands and immutable receipts."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Literal, TypeAlias


@dataclass(frozen=True)
class EvaluateCommand:
    command_id: str
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    input_ref: str
    output_uri: str = ""
    logical_command_id: str = ""
    attempt_id: str = "attempt-001"
    attempt_index: int = 1
    kind: Literal["evaluate"] = "evaluate"


@dataclass(frozen=True)
class TrainSFTCommand:
    command_id: str
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    input_ref: str
    output_uri: str = ""
    logical_command_id: str = ""
    attempt_id: str = "attempt-001"
    attempt_index: int = 1
    kind: Literal["train_sft"] = "train_sft"


@dataclass(frozen=True)
class TrainRFTCommand:
    command_id: str
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    input_ref: str
    output_uri: str = ""
    logical_command_id: str = ""
    attempt_id: str = "attempt-001"
    attempt_index: int = 1
    kind: Literal["train_rft"] = "train_rft"


EngineCommand: TypeAlias = EvaluateCommand | TrainSFTCommand | TrainRFTCommand


class EngineReceiptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class EngineReceipt:
    receipt_id: str
    command_id: str
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str
    status: EngineReceiptStatus
    output_refs: tuple[str, ...] = ()
    error: str | None = None
    logical_command_id: str = ""
    attempt_id: str = "attempt-001"
    attempt_index: int = 1
    failure_kind: str | None = None
    retryable: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class EngineAttemptError(RuntimeError):
    """A typed physical-attempt failure consumed by EngineWorker."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: str,
        retryable: bool,
        output_refs: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.retryable = retryable
        self.output_refs = tuple(output_refs)
