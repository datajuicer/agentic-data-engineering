"""Validation results shared by delivery gates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeliveryViolation:
    code: str
    message: str
    path: str | None = None
    repairable: bool = False


@dataclass(frozen=True)
class ValidationReport:
    violations: tuple[DeliveryViolation, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.violations
