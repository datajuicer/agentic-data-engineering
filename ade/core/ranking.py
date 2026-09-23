"""Offline-only ranking state."""

from dataclasses import dataclass
import math


def score_pair_sort_key(
    score: float,
    secondary_score: float | None,
    *,
    direction: str,
) -> tuple[float, int, float]:
    """Canonical ordering key for one profile-matched ranking score pair."""

    if direction not in {"maximize", "minimize"}:
        raise ValueError("ranking direction must be maximize or minimize")
    if not math.isfinite(score) or (
        secondary_score is not None and not math.isfinite(secondary_score)
    ):
        raise ValueError("ranking scores must be finite")
    sign = -1.0 if direction == "maximize" else 1.0
    return (
        sign * score,
        1 if secondary_score is None else 0,
        0.0 if secondary_score is None else sign * secondary_score,
    )


def score_pair_improves(
    candidate: tuple[float, float | None],
    current: tuple[float, float | None],
    *,
    direction: str,
) -> bool:
    return score_pair_sort_key(*candidate, direction=direction) < score_pair_sort_key(
        *current, direction=direction
    )


@dataclass(frozen=True)
class RankingEntry:
    subject_id: str
    score: float
    level: str = "trial_level"
    evaluation_profile: str = "offline"
    secondary_score: float | None = None
    accepted_revision: int = 0
    artifact_ref_id: str | None = None
    artifact_digest: str | None = None
    source_eligible: bool = False
    representative_trial_id: str | None = None
    operator_status: str = "not_scheduled"
    operator_result_ref: str | None = None

    def __post_init__(self) -> None:
        if self.level not in {"reference", "trial_level", "plan_level"}:
            raise ValueError("ranking entry level is invalid")
        if self.accepted_revision < 0:
            raise ValueError("ranking accepted revision must be non-negative")
        if self.operator_status not in {
            "not_scheduled",
            "pending",
            "completed",
            "failed",
            "cancelled",
            "not_applicable",
        }:
            raise ValueError("ranking operator status is invalid")
        if self.level == "plan_level" and not self.representative_trial_id:
            raise ValueError("plan ranking entry requires representative Trial")


@dataclass(frozen=True)
class RankingState:
    metric_id: str
    direction: str = "maximize"
    entries: tuple[RankingEntry, ...] = ()
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.metric_id:
            raise ValueError("ranking metric_id is required")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("ranking direction must be maximize or minimize")
        if self.revision < 0:
            raise ValueError("ranking revision must be non-negative")
        identities = tuple((item.level, item.subject_id) for item in self.entries)
        if len(identities) != len(set(identities)):
            raise ValueError("ranking entry identities must be unique")
