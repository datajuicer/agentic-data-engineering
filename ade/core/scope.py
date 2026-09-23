"""Structured ownership keys for ADE runtime subjects."""

from dataclasses import dataclass


def _require_id(value: str, label: str) -> None:
    if not value:
        raise ValueError(f"{label} is required")


@dataclass(frozen=True)
class CoordinatorKey:
    run_id: str
    coordinator_id: str

    def __post_init__(self) -> None:
        _require_id(self.run_id, "run_id")
        _require_id(self.coordinator_id, "coordinator_id")

    @property
    def subject_ref(self) -> str:
        return f"{self.run_id}/{self.coordinator_id}"


@dataclass(frozen=True)
class PlanKey:
    run_id: str
    coordinator_id: str
    plan_id: str

    def __post_init__(self) -> None:
        _require_id(self.run_id, "run_id")
        _require_id(self.coordinator_id, "coordinator_id")
        _require_id(self.plan_id, "plan_id")

    @property
    def coordinator(self) -> CoordinatorKey:
        return CoordinatorKey(self.run_id, self.coordinator_id)

    @property
    def subject_ref(self) -> str:
        return f"{self.run_id}/{self.coordinator_id}/{self.plan_id}"


@dataclass(frozen=True)
class TrialKey:
    run_id: str
    coordinator_id: str
    plan_id: str
    trial_id: str

    def __post_init__(self) -> None:
        _require_id(self.run_id, "run_id")
        _require_id(self.coordinator_id, "coordinator_id")
        _require_id(self.plan_id, "plan_id")
        _require_id(self.trial_id, "trial_id")

    @property
    def plan(self) -> PlanKey:
        return PlanKey(self.run_id, self.coordinator_id, self.plan_id)

    @property
    def subject_ref(self) -> str:
        return (
            f"{self.run_id}/{self.coordinator_id}/{self.plan_id}/{self.trial_id}"
        )


def subject_ref(
    run_id: str,
    coordinator_id: str | None = None,
    plan_id: str | None = None,
    trial_id: str | None = None,
) -> str:
    values = (coordinator_id, plan_id, trial_id)
    seen_missing = False
    parts = [run_id]
    _require_id(run_id, "run_id")
    for label, value in zip(
        ("coordinator_id", "plan_id", "trial_id"), values, strict=True
    ):
        if value is None:
            seen_missing = True
            continue
        if seen_missing:
            raise ValueError(f"{label} cannot appear without its parent scope")
        _require_id(value, label)
        parts.append(value)
    return "/".join(parts)


def agent_action_id(
    *,
    owner_subject_ref: str,
    target_subject_ref: str,
    role_segment: str,
    basis_revision: int,
) -> str:
    """Build the canonical owner-prefixed identity for one Agent Call action."""
    if not owner_subject_ref or not target_subject_ref or not role_segment:
        raise ValueError("Agent action owner, target, and role are required")
    if basis_revision < 0:
        raise ValueError("Agent action basis_revision must be non-negative")
    if target_subject_ref == owner_subject_ref:
        relative_target = ""
    elif target_subject_ref.startswith(f"{owner_subject_ref}/"):
        relative_target = target_subject_ref[len(owner_subject_ref) + 1 :]
    else:
        raise ValueError("Agent action target must belong to its Session owner")
    parts = [owner_subject_ref, "agent", role_segment]
    if relative_target:
        parts.append(relative_target)
    parts.append(f"rev-{basis_revision:06d}")
    return "/".join(parts)
