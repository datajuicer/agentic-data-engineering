"""Analyzer client for the API-backed Review Labor Rubric Job transport."""

from __future__ import annotations

from typing import Any, Protocol

from ade.rubric_jobs import JobResult, JobStatus
from ade.memory.usage import RunUsageLedger


class ReviewLaborGateway(Protocol):
    def submit(self, *, submission_id: str, input_jsonl: bytes, job_metadata: dict[str, Any]) -> JobStatus: ...
    def get_status(self, job_id: str) -> JobStatus: ...
    def get_result(self, job_id: str) -> JobResult: ...
    def cancel(self, job_id: str) -> JobStatus: ...


class ReviewLaborClient:
    def __init__(self, gateway: ReviewLaborGateway, *, usage_ledger: RunUsageLedger | None = None) -> None:
        self.gateway = gateway
        self.usage_ledger = usage_ledger

    def submit(self, *, submission_id: str, input_jsonl: bytes, job_metadata: dict[str, Any]) -> JobStatus:
        return self.gateway.submit(
            submission_id=submission_id, input_jsonl=input_jsonl, job_metadata=job_metadata
        )

    def status(self, job_id: str) -> JobStatus:
        return self.gateway.get_status(job_id)

    def result(self, job_id: str) -> JobResult:
        result = self.gateway.get_result(job_id)
        if self.usage_ledger is not None:
            self.usage_ledger.append(
                {
                    "event_id": f"review-labor-job:{result.status.job_id}",
                    "category": "analyzer_review",
                    "provider": "review_labor",
                    **result.usage.to_dict(),
                }
            )
        return result

    def cancel(self, job_id: str) -> JobStatus:
        return self.gateway.cancel(job_id)
