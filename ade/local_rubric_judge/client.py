"""Provider-neutral ADE client bound only to a Rubric Job gateway."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from ade.rubric_jobs import JobResult, JobStatus


class JobRejected(RuntimeError):
    """The gateway rejected a submission before a job handle was created."""


class JobGateway(Protocol):
    def submit(self, *, submission_id: str, input_jsonl: bytes, job_metadata: dict[str, Any]) -> JobStatus: ...
    def get_status(self, job_id: str) -> JobStatus: ...
    def get_result(self, job_id: str) -> JobResult: ...
    def cancel(self, job_id: str) -> JobStatus: ...
    def expire(self, job_id: str) -> JobStatus: ...


class LocalRubricJudgeClient:
    def __init__(self, gateway: JobGateway, *, poll_interval_seconds: float = 0.1) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.gateway = gateway
        self.poll_interval_seconds = poll_interval_seconds
        self._handles: set[str] = set()

    async def submit(
        self,
        *,
        submission_id: str,
        input_jsonl: bytes,
        job_metadata: dict[str, Any],
    ) -> JobStatus:
        try:
            status = self.gateway.submit(
                submission_id=submission_id,
                input_jsonl=input_jsonl,
                job_metadata=job_metadata,
            )
        except ValueError as error:
            raise JobRejected(str(error)) from error
        self._handles.add(status.job_id)
        return status

    async def status(self, job_id: str) -> JobStatus:
        return self.gateway.get_status(job_id)

    async def result(self, job_id: str) -> JobResult:
        return self.gateway.get_result(job_id)

    async def cancel(self, job_id: str) -> JobStatus:
        return self.gateway.cancel(job_id)

    async def wait(self, job_id: str) -> JobResult:
        while True:
            status = self.gateway.get_status(job_id)
            if status.state in {"completed", "completed_with_errors", "failed", "cancelled"}:
                return self.gateway.get_result(job_id)
            await asyncio.sleep(self.poll_interval_seconds)

    async def cancel_all(self) -> tuple[JobStatus, ...]:
        statuses = []
        for job_id in sorted(self._handles):
            status = self.gateway.get_status(job_id)
            if status.state not in {"completed", "completed_with_errors", "failed", "cancelled"}:
                status = self.gateway.cancel(job_id)
            statuses.append(status)
        return tuple(statuses)
