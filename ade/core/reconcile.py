"""Explicit result of one Control reconciliation cycle."""

from __future__ import annotations

from dataclasses import dataclass

from ade.core.run import RunStatus


@dataclass(frozen=True)
class Advanced:
    from_revision: int
    to_revision: int
    transition: str
    subject_ref: str


@dataclass(frozen=True)
class Waiting:
    revision: int
    reason: str
    wakeup_ref: str | None = None


@dataclass(frozen=True)
class Terminal:
    revision: int
    status: RunStatus


ReconcileResult = Advanced | Waiting | Terminal
