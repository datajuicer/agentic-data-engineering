"""Durable batch service for Harness-owned Analyzer Review jobs."""

from __future__ import annotations

from pathlib import Path

from ade.review_labor.jobs import ReviewEndpoint, ReviewLaborJobService


class LocalAnalyzerReviewService(ReviewLaborJobService):
    """Analyzer Review service using the durable rubric-job execution model.

    The Analyzer-specific endpoint owns prompt construction and result
    normalization.  This service owns the shared job behavior: FIFO claim,
    bounded concurrent waves, per-wave persistence, recovery, cancellation,
    progress, and usage aggregation.
    """

    def __init__(
        self,
        state_path: str | Path,
        endpoint: ReviewEndpoint,
        *,
        auto_run: bool = True,
        max_concurrency: int = 128,
        batch_timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(
            state_path,
            endpoint,
            auto_run=auto_run,
            max_concurrency=max_concurrency,
            batch_timeout_seconds=batch_timeout_seconds,
        )
