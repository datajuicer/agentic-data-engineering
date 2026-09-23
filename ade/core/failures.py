"""Typed failure state."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureState:
    code: str
    message: str
    retryable: bool = False
    evidence_ref: str | None = None
