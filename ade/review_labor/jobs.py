"""API-backed Review Labor implementation used by the Harness worker."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any, Callable, Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from ade.engine.storage.atomic import write_json_atomic
from ade.rubric_jobs import (
    JobError,
    JobResult,
    JobStatus,
    RubricInputRow,
    RubricOutputRow,
    TokenUsage,
    canonical_digest,
    encode_input_jsonl,
    join_results,
    parse_input_jsonl,
    render_template_once,
    submission_content_digest,
)


@dataclass(frozen=True)
class ReviewResponse:
    result: dict[str, Any] | None
    error: JobError | None
    usage: TokenUsage
    model_id: str
    model_digest: str
    raw_response: dict[str, Any] | None = None
    advisory_fallback: JobError | None = None


class ReviewEndpoint(Protocol):
    async def evaluate(
        self, row: RubricInputRow, rendered_prompt: str
    ) -> ReviewResponse: ...


class ReviewLaborJobService:
    def __init__(
        self,
        state_path: str | Path,
        endpoint: ReviewEndpoint,
        *,
        auto_run: bool = True,
        max_concurrency: int = 128,
        batch_timeout_seconds: float | None = 60.0,
    ) -> None:
        if max_concurrency < 1 or (
            batch_timeout_seconds is not None and batch_timeout_seconds <= 0
        ):
            raise ValueError(
                "Review Labor concurrency must be positive and batch timeout must be null or positive"
            )
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint
        self.auto_run = auto_run
        self.max_concurrency = max_concurrency
        self.batch_timeout_seconds = batch_timeout_seconds
        self._lock = threading.RLock()
        if not self.state_path.is_file():
            write_json_atomic(self.state_path, {"next_sequence": 1, "jobs": []})
        else:
            state = self._state()
            changed = False
            for job in state["jobs"]:
                if job["state"] == "running":
                    job["state"] = "cancelled" if job["cancel_requested"] else "queued"
                    changed = True
            if changed:
                self._write(state)

    def submit(
        self, *, submission_id: str, input_jsonl: bytes, job_metadata: dict[str, Any],
        progress_callback: Callable[[JobStatus], None] | None = None,
    ) -> JobStatus:
        rows = parse_input_jsonl(input_jsonl)
        canonical = encode_input_jsonl(rows).decode("utf-8")
        digest = submission_content_digest(rows, job_metadata)
        existing_job_id: str | None = None
        with self._lock:
            state = self._state()
            existing = next(
                (job for job in state["jobs"] if job["submission_id"] == submission_id),
                None,
            )
            if existing is not None:
                if existing["content_digest"] != digest:
                    raise ValueError(
                        "submission identity already exists with different content"
                    )
                existing_job_id = existing["job_id"]
                should_run = existing["state"] == "queued"
            else:
                sequence = state["next_sequence"]
                state["next_sequence"] += 1
                job = {
                    "job_id": f"review-job-{sequence:08d}",
                    "submission_id": submission_id,
                    "content_digest": digest,
                    "queue_sequence": sequence,
                    "state": "queued",
                    "input_jsonl": canonical,
                    "metadata": job_metadata,
                    "total_rows": len(rows),
                    "completed_rows": 0,
                    "error_rows": 0,
                    "progress_completed_rows": 0,
                    "progress_error_rows": 0,
                    "cancel_requested": False,
                    "rows": [],
                    "usage": None,
                }
                state["jobs"].append(job)
                self._write(state)
                existing_job_id = job["job_id"]
                should_run = True
        if self.auto_run and should_run:
            # FastMCP can invoke this synchronous service method from its
            # asyncio loop. Run the job loop on a worker thread so production
            # submissions do not call asyncio.run() inside that active loop.
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(
                    asyncio.run,
                    self.run_until_idle(progress_callback=progress_callback),
                ).result()
        assert existing_job_id is not None
        return self.get_status(existing_job_id)

    async def run_until_idle(
        self, *, progress_callback: Callable[[JobStatus], None] | None = None
    ) -> None:
        while await self.run_next(progress_callback=progress_callback) is not None:
            pass

    def get_status(self, job_id: str) -> JobStatus:
        with self._lock:
            return _status(self._job(job_id))

    def get_result(self, job_id: str) -> JobResult:
        with self._lock:
            job = self._job(job_id)
        if job["state"] == "cancelled":
            raise ValueError("cancelled Review Labor result is audit-only")
        if job["state"] not in {"completed", "completed_with_errors", "failed"}:
            raise ValueError("Review Labor result is not terminal")
        return JobResult(
            _status(job),
            tuple(RubricOutputRow.from_dict(row) for row in job["rows"]),
            TokenUsage.from_dict(job["usage"]),
        )

    def cancel(self, job_id: str) -> JobStatus:
        with self._lock:
            state = self._state()
            job = _find_job(state, job_id)
            if job["state"] == "queued":
                job["state"] = "cancelled"
                job["cancel_requested"] = True
                job["usage"] = _aggregate_usage(()).to_dict()
            elif job["state"] == "running":
                job["cancel_requested"] = True
            self._write(state)
            return _status(job)

    def audit_provider_responses(
        self, job_id: str
    ) -> tuple[dict[str, Any] | None, ...]:
        with self._lock:
            return tuple(self._job(job_id).get("raw_provider_responses", ()))

    async def run_next(
        self, *, progress_callback: Callable[[JobStatus], None] | None = None
    ) -> JobStatus | None:
        with self._lock:
            state = self._state()
            if any(job["state"] == "running" for job in state["jobs"]):
                return None
            queued = sorted(
                (job for job in state["jobs"] if job["state"] == "queued"),
                key=lambda job: job["queue_sequence"],
            )
            if not queued:
                return None
            job_id = queued[0]["job_id"]
            queued[0]["state"] = "running"
            queued[0]["progress_completed_rows"] = 0
            queued[0]["progress_error_rows"] = 0
            self._write(state)
            if progress_callback is not None:
                progress_callback(_status(queued[0]))
        inputs = parse_input_jsonl(queued[0]["input_jsonl"])
        with self._lock:
            completed_ids = {item["record_id"] for item in self._job(job_id)["rows"]}
        pending = tuple(row for row in inputs if row.record_id not in completed_ids)
        async def evaluate(row: RubricInputRow) -> tuple[RubricInputRow, ReviewResponse]:
            try:
                response = await self.endpoint.evaluate(row, render_template_once(row))
            except Exception as error:
                response = ReviewResponse(
                    None,
                    JobError("endpoint_error", str(error), False),
                    _unavailable_usage(),
                    "unavailable",
                    "unavailable",
                    None,
                )
            return row, response

        for offset in range(0, len(pending), self.max_concurrency):
            with self._lock:
                if self._job(job_id)["cancel_requested"]:
                    state = self._state()
                    job = _find_job(state, job_id)
                    job["state"] = "cancelled"
                    job["usage"] = _aggregate_usage(
                        tuple(RubricOutputRow.from_dict(item).usage for item in job["rows"])
                    ).to_dict()
                    self._write(state)
                    return self.get_status(job_id)
            wave = pending[offset : offset + self.max_concurrency]
            tasks = tuple(asyncio.create_task(evaluate(row)) for row in wave)
            try:
                pending_batch = asyncio.gather(*tasks)
                evaluated = (
                    await pending_batch
                    if self.batch_timeout_seconds is None
                    else await asyncio.wait_for(
                        pending_batch,
                        timeout=self.batch_timeout_seconds,
                    )
                )
            except asyncio.TimeoutError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                evaluated = tuple(
                    (
                        row,
                        ReviewResponse(
                            None,
                            JobError(
                                "timeout",
                                "Review Labor batch deadline exceeded",
                                False,
                            ),
                            _unavailable_usage(),
                            "unavailable",
                            "unavailable",
                            None,
                        ),
                    )
                    for row in wave
                )

            admitted_rows = []
            raw_responses = []
            for row, response in evaluated:
                fallback_result = (
                    {
                        "review_fallback": {
                            "error_type": response.advisory_fallback.code,
                            "message": response.advisory_fallback.message,
                        }
                    }
                    if response.advisory_fallback is not None
                    else None
                )
                output = RubricOutputRow(
                    row.record_id,
                    "error" if response.error else "ok",
                    (
                        None
                        if response.error
                        else (
                            fallback_result
                            if fallback_result is not None
                            else _project_advisory_result(
                                response.result,
                                row.rubric.output_schema,
                            )
                        )
                    ),
                    response.error,
                    response.usage,
                    {"id": response.model_id, "digest": response.model_digest},
                    canonical_digest(row.rubric.template),
                    canonical_digest(row.rubric.output_schema),
                    {**row.metadata, "raw_provider_response": response.raw_response},
                )
                # Provider response is audit data, not caller metadata.
                admitted = RubricOutputRow(
                    output.record_id,
                    output.status,
                    output.result,
                    output.error,
                    output.usage,
                    output.model,
                    output.rubric_digest,
                    output.output_schema_digest,
                    row.metadata,
                )
                if admitted.status == "ok" and response.advisory_fallback is None:
                    try:
                        join_results((row,), (admitted,))
                    except ValueError as error:
                        admitted = RubricOutputRow(
                            row.record_id,
                            "error",
                            None,
                            JobError("schema_invalid", str(error), False),
                            response.usage,
                            {"id": response.model_id, "digest": response.model_digest},
                            canonical_digest(row.rubric.template),
                            canonical_digest(row.rubric.output_schema),
                            row.metadata,
                        )
                admitted_rows.append(admitted)
                raw_responses.append(response.raw_response)
            with self._lock:
                state = self._state()
                job = _find_job(state, job_id)
                if job["cancel_requested"]:
                    job["state"] = "cancelled"
                    job["usage"] = _aggregate_usage(
                        tuple(RubricOutputRow.from_dict(item).usage for item in job["rows"])
                    ).to_dict()
                    self._write(state)
                    return _status(job)
                job["rows"].extend(item.to_dict() for item in admitted_rows)
                job.setdefault("raw_provider_responses", []).extend(raw_responses)
                job["completed_rows"] += len(admitted_rows)
                job["error_rows"] += sum(item.status == "error" for item in admitted_rows)
                usages = tuple(RubricOutputRow.from_dict(item).usage for item in job["rows"])
                job["usage"] = _aggregate_usage(usages).to_dict()
                job["progress_completed_rows"] = job["completed_rows"]
                job["progress_error_rows"] = job["error_rows"]
                job["state"] = "running"
                self._write(state)
                status = _status(job)
            if progress_callback is not None:
                progress_callback(status)

        with self._lock:
            state = self._state()
            job = _find_job(state, job_id)
            job["state"] = "completed_with_errors" if job["error_rows"] else "completed"
            self._write(state)
            status = _status(job)
        if progress_callback is not None:
            progress_callback(status)
        return status

    def _state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _job(self, job_id: str) -> dict[str, Any]:
        return _find_job(self._state(), job_id)

    def _write(self, state: dict[str, Any]) -> None:
        write_json_atomic(self.state_path, state)


def _find_job(state: dict[str, Any], job_id: str) -> dict[str, Any]:
    job = next((item for item in state["jobs"] if item["job_id"] == job_id), None)
    if job is None:
        raise KeyError(job_id)
    return job


def _project_advisory_result(
    result: dict[str, Any] | None,
    output_schema: dict[str, Any],
) -> dict[str, Any] | None:
    """Ignore provider-added wrappers while preserving schema-required feedback."""

    if result is None:
        return None
    projected = _project_schema_value(result, output_schema)
    if not isinstance(projected, dict):
        return result
    try:
        Draft202012Validator(output_schema).validate(projected)
    except ValidationError:
        return result
    return projected


def _project_schema_value(value: Any, schema: dict[str, Any]) -> Any:
    if schema.get("type") == "object" and isinstance(value, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return value
        return {
            key: _project_schema_value(value[key], property_schema)
            for key, property_schema in properties.items()
            if key in value and isinstance(property_schema, dict)
        }
    if schema.get("type") == "array" and isinstance(value, list):
        prefix_items = schema.get("prefixItems")
        if isinstance(prefix_items, list):
            return [
                _project_schema_value(
                    item,
                    prefix_items[index]
                    if index < len(prefix_items)
                    and isinstance(prefix_items[index], dict)
                    else {},
                )
                for index, item in enumerate(value)
            ]
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            return [_project_schema_value(item, item_schema) for item in value]
    return value


def _status(job: dict[str, Any]) -> JobStatus:
    running = job["state"] == "running"
    return JobStatus(
        job["job_id"],
        job["submission_id"],
        job["content_digest"],
        job["state"],
        job["queue_sequence"],
        job["total_rows"],
        job.get("progress_completed_rows", job["completed_rows"])
        if running else job["completed_rows"],
        job.get("progress_error_rows", job["error_rows"])
        if running else job["error_rows"],
        None,
    )


def _aggregate_usage(usages: tuple[TokenUsage, ...]) -> TokenUsage:
    if not usages:
        return _unavailable_usage(attempts=0, requests=0)
    core = ("prompt_tokens", "completion_tokens", "total_tokens")
    known_all = all(
        all(getattr(item, name) is not None for name in core) for item in usages
    )
    known_any = any(
        any(getattr(item, name) is not None for name in core) for item in usages
    )

    def total(name: str):
        values = [getattr(item, name) for item in usages]
        return (
            sum(value for value in values if value is not None)
            if any(value is not None for value in values)
            else None
        )

    return TokenUsage.from_dict(
        {
            "usage_status": "complete"
            if known_all
            else "partial"
            if known_any
            else "unavailable",
            **{
                name: total(name)
                for name in (*core, "cached_tokens", "reasoning_tokens")
            },
            "attempts": sum(item.attempts for item in usages),
            "retries": sum(item.retries for item in usages),
            "requests": sum(item.requests for item in usages),
        }
    )


def _unavailable_usage(*, attempts: int = 1, requests: int = 1) -> TokenUsage:
    return TokenUsage.from_dict(
        {
            "usage_status": "unavailable",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cached_tokens": None,
            "reasoning_tokens": None,
            "attempts": attempts,
            "retries": 0,
            "requests": requests,
        }
    )
