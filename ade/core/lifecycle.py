"""Deterministic Plan ranking and stop policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

from ade.core.ranking import score_pair_improves, score_pair_sort_key


class PlanTerminalDecision(StrEnum):
    CONTINUE = "continue"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class TrialMetric:
    trial_id: str
    artifact_ref_id: str
    score: float | None
    secondary_score: float | None = None

    def __post_init__(self) -> None:
        if not self.trial_id or not self.artifact_ref_id:
            raise ValueError("Trial metric identity is required")

    @property
    def valid(self) -> bool:
        return (
            self.score is not None
            and math.isfinite(self.score)
            and (
                self.secondary_score is None
                or math.isfinite(self.secondary_score)
            )
        )


@dataclass(frozen=True)
class PlanProgress:
    decision: PlanTerminalDecision
    no_improvement_count: int
    best_trial_id: str | None
    best_artifact_ref_id: str | None
    best_score: float | None
    best_secondary_score: float | None = None


@dataclass(frozen=True)
class PlanStopPolicy:
    min_trials: int
    max_trials: int
    no_improvement_patience: int
    direction: str = "maximize"

    def __post_init__(self) -> None:
        if self.min_trials < 1:
            raise ValueError("min_trials must be positive")
        if self.max_trials < self.min_trials:
            raise ValueError("max_trials cannot be less than min_trials")
        if self.no_improvement_patience < 1:
            raise ValueError("no_improvement_patience must be positive")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be maximize or minimize")

    def evaluate(self, results: tuple[TrialMetric, ...]) -> PlanProgress:
        if len(results) > self.max_trials:
            raise ValueError("Trial results exceed max_trials")
        best: TrialMetric | None = None
        no_improvement_count = 0
        for result in results:
            if result.valid and (
                best is None
                or self._improves(
                    (result.score, result.secondary_score),
                    (best.score, best.secondary_score),
                )
            ):
                best = result
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if result.valid and best is not None and (
                    score_pair_sort_key(
                        result.score, result.secondary_score, direction=self.direction
                    )
                    == score_pair_sort_key(
                        best.score, best.secondary_score, direction=self.direction
                    )
                ):
                    # Refresh the representative without resetting patience.
                    best = result

        decision = PlanTerminalDecision.CONTINUE
        if len(results) >= self.max_trials:
            decision = (
                PlanTerminalDecision.COMPLETED
                if best is not None
                else PlanTerminalDecision.FAILED
            )
        elif (
            len(results) >= self.min_trials
            and no_improvement_count >= self.no_improvement_patience
        ):
            decision = (
                PlanTerminalDecision.COMPLETED
                if best is not None
                else PlanTerminalDecision.FAILED
            )
        return PlanProgress(
            decision=decision,
            no_improvement_count=no_improvement_count,
            best_trial_id=best.trial_id if best is not None else None,
            best_artifact_ref_id=best.artifact_ref_id if best is not None else None,
            best_score=best.score if best is not None else None,
            best_secondary_score=(
                best.secondary_score if best is not None else None
            ),
        )

    def _improves(
        self,
        candidate: tuple[float | None, float | None],
        current: tuple[float | None, float | None],
    ) -> bool:
        assert candidate[0] is not None and current[0] is not None
        return score_pair_improves(
            (candidate[0], candidate[1]),
            (current[0], current[1]),
            direction=self.direction,
        )
