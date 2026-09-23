"""Persistent single-job FIFO Local Rubric Judge service core.

The service keeps its live queue index in memory and persists each job in an
independent directory.  Large request and result payloads never share a mutable
database file with the frequently-read job status.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Protocol

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


_STORE_SCHEMA = "ade.local_rubric_judge.file_store.v1"
_TERMINAL_STATES = frozenset(
    {"completed", "completed_with_errors", "failed", "cancelled"}
)


class IdempotencyConflict(ValueError):
    pass


class CancelledJobResult(ValueError):
    pass


@dataclass(frozen=True)
class ModelResponse:
    result: dict[str, Any] | None
    error: JobError | None
    usage: TokenUsage
    model_id: str = "fake-model"
    model_digest: str = "fake-model-digest"


class Endpoint(Protocol):
    async def evaluate(self, row: RubricInputRow, rendered_prompt: str) -> ModelResponse: ...


class FakeEndpoint:
    """Scriptable fake endpoint; it never opens a network or GPU service."""

    def __init__(self, responses: dict[str, ModelResponse] | None = None) -> None:
        self.responses = dict(responses or {})
        self.calls: list[str] = []
        self.block: asyncio.Event | None = None

    async def evaluate(self, row: RubricInputRow, rendered_prompt: str) -> ModelResponse:
        assert rendered_prompt
        self.calls.append(row.record_id)
        if self.block is not None:
            await self.block.wait()
        response = self.responses.get(row.record_id)
        if response is not None:
            return response
        properties = row.rubric.output_schema.get("properties", {})
        if "scores_by_dimension" in properties:
            result: dict[str, Any] = {"scores_by_dimension": {}}
        elif "label" in properties:
            result = {"label": "pass", "reason": "fake"}
        else:
            result = {}
        return ModelResponse(result, None, _fake_usage())


class LocalRubricJudgeService:
    def __init__(
        self,
        state_path: str | Path,
        endpoint: Endpoint,
        *,
        max_concurrency: int = 1,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("Local Judge max_concurrency must be positive")
        self.state_path = Path(state_path)
        self.jobs_root = self.state_path
        self.endpoint = endpoint
        self.max_concurrency = int(max_concurrency)
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._submission_ids: dict[str, str] = {}
        self._completed_indices: dict[str, set[int]] = {}
        self._next_sequence = 1
        self._healthy = True
        self._unhealthy_reason: str | None = None
        self._initialize()

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._healthy

    @property
    def unhealthy_reason(self) -> str | None:
        with self._lock:
            return self._unhealthy_reason

    def mark_unhealthy(self, error: BaseException) -> None:
        with self._lock:
            self._healthy = False
            self._unhealthy_reason = f"{type(error).__name__}: {error}"

    def close(self) -> None:
        pass

    def submit(
        self,
        *,
        submission_id: str,
        input_jsonl: bytes,
        job_metadata: dict[str, Any],
    ) -> JobStatus:
        if not isinstance(submission_id, str) or not submission_id.strip():
            raise ValueError("submission_id is required")
        rows = parse_input_jsonl(input_jsonl)
        canonical_input = encode_input_jsonl(rows)
        digest = submission_content_digest(rows, job_metadata)
        if not isinstance(job_metadata, dict):
            raise ValueError("job metadata must be an object")
        with self._lock:
            existing_id = self._submission_ids.get(submission_id)
            if existing_id is not None:
                existing = self._jobs[existing_id]
                if existing["content_digest"] != digest:
                    raise IdempotencyConflict(
                        "submission identity already exists with different content"
                    )
                return _status(existing)

            sequence = self._next_sequence
            job_id = f"job-{sequence:08d}"
            record = {
                "schema_version": _STORE_SCHEMA,
                "job_id": job_id,
                "submission_id": submission_id,
                "content_digest": digest,
                "queue_sequence": sequence,
                "state": "queued",
                "total_rows": len(rows),
                "completed_rows": 0,
                "error_rows": 0,
                "cancel_requested": False,
                "usage": None,
                "error": None,
            }
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{job_id}.", dir=self.jobs_root)
            )
            try:
                _write_bytes_atomic(temporary / "input.jsonl", canonical_input)
                _write_json_atomic(temporary / "metadata.json", job_metadata)
                _write_json_atomic(temporary / "status.json", record)
                os.replace(temporary, self._job_root(job_id))
            finally:
                if temporary.exists():
                    for path in temporary.iterdir():
                        path.unlink()
                    temporary.rmdir()
            self._jobs[job_id] = record
            self._submission_ids[submission_id] = job_id
            self._completed_indices[job_id] = set()
            self._next_sequence += 1
            return _status(record)

    def get_status(self, job_id: str) -> JobStatus:
        with self._lock:
            return _status(self._job(job_id))

    def get_result(self, job_id: str) -> JobResult:
        with self._lock:
            record = dict(self._job(job_id))
        status = _status(record)
        if status.state == "cancelled":
            raise CancelledJobResult("cancelled job results are audit-only")
        if status.state not in {"completed", "completed_with_errors", "failed"}:
            raise ValueError("Rubric Job result is not terminal")
        rows = self.get_audit_rows(job_id)
        usage_value = record.get("usage")
        if not isinstance(usage_value, dict):
            raise ValueError("terminal Rubric Job lacks usage")
        return JobResult(status, rows, TokenUsage.from_dict(usage_value))

    def get_audit_rows(self, job_id: str) -> tuple[RubricOutputRow, ...]:
        with self._lock:
            self._job(job_id)
            indexed = self._read_indexed_outputs(job_id)
        return tuple(output for _, output in sorted(indexed.items()))

    def cancel(self, job_id: str) -> JobStatus:
        with self._lock:
            record = self._job(job_id)
            if record["state"] == "queued":
                record = self._update_job(
                    job_id, state="cancelled", cancel_requested=True
                )
            elif record["state"] == "running":
                record = self._update_job(job_id, cancel_requested=True)
            return _status(record)

    def expire(self, job_id: str) -> JobStatus:
        """Stop a deadline-bound job and make only unfinished rows errors."""
        marker = {
            "code": "deadline_exceeded",
            "message": "Rubric Job execution deadline exceeded",
            "retryable": True,
        }
        with self._lock:
            record = self._job(job_id)
            if record["state"] == "queued":
                self._update_job(job_id, cancel_requested=True, error=marker)
                self._finish_expired(job_id, marker)
            elif record["state"] == "running":
                self._update_job(job_id, cancel_requested=True, error=marker)
            return _status(self._job(job_id))

    def cancel_all(self) -> tuple[JobStatus, ...]:
        with self._lock:
            job_ids = [
                record["job_id"]
                for record in sorted(
                    self._jobs.values(), key=lambda item: item["queue_sequence"]
                )
                if record["state"] in {"queued", "running"}
            ]
        return tuple(self.cancel(job_id) for job_id in job_ids)

    async def run_next(self) -> JobStatus | None:
        claimed = self._claim_next()
        if claimed is None:
            return None
        job_id, inputs = claimed
        pending = [
            (index, row)
            for index, row in enumerate(inputs)
            if not self._row_exists(job_id, index)
        ]
        for offset in range(0, len(pending), self.max_concurrency):
            if self._cancel_requested(job_id):
                self._finish_cancel(job_id)
                return self.get_status(job_id)
            wave = pending[offset : offset + self.max_concurrency]
            outputs = await asyncio.gather(
                *(self._evaluate_row(row) for _, row in wave)
            )
            self._persist_rows(
                job_id,
                tuple(
                    (index, output)
                    for (index, _row), output in zip(wave, outputs, strict=True)
                ),
            )
            if self._cancel_requested(job_id):
                self._finish_cancel(job_id)
                return self.get_status(job_id)
        self._finish_completed(job_id)
        return self.get_status(job_id)

    async def _evaluate_row(self, row: RubricInputRow) -> RubricOutputRow:
        response: ModelResponse | None = None
        try:
            response = await self.endpoint.evaluate(row, render_template_once(row))
        except Exception as error:
            output = self._output(
                row,
                ModelResponse(
                    None,
                    JobError("endpoint_error", str(error), True),
                    _unavailable_usage(),
                    "unknown",
                    "unknown",
                ),
            )
        else:
            try:
                output = self._output(row, response)
                if output.status == "ok":
                    join_results((row,), (output,))
            except Exception as error:
                output = self._output(
                    row,
                    ModelResponse(
                        None,
                        JobError("schema_invalid", str(error), False),
                        response.usage,
                        response.model_id,
                        response.model_digest,
                    ),
                )
        return output

    def _claim_next(self) -> tuple[str, tuple[RubricInputRow, ...]] | None:
        with self._lock:
            if any(record["state"] == "running" for record in self._jobs.values()):
                return None
            queued = [
                record
                for record in self._jobs.values()
                if record["state"] == "queued"
            ]
            if not queued:
                return None
            record = min(queued, key=lambda item: item["queue_sequence"])
            job_id = str(record["job_id"])
            self._update_job(job_id, state="running")
            input_jsonl = (self._job_root(job_id) / "input.jsonl").read_bytes()
        return job_id, parse_input_jsonl(input_jsonl)

    def _output(self, row: RubricInputRow, response: ModelResponse) -> RubricOutputRow:
        status = "error" if response.error is not None else "ok"
        return RubricOutputRow(
            row.record_id,
            status,
            response.result if status == "ok" else None,
            response.error,
            response.usage,
            {"id": response.model_id, "digest": response.model_digest},
            canonical_digest(row.rubric.template),
            canonical_digest(row.rubric.output_schema),
            row.metadata,
        )

    def _persist_rows(
        self,
        job_id: str,
        rows: tuple[tuple[int, RubricOutputRow], ...],
    ) -> None:
        if not rows:
            return
        with self._lock:
            record = self._job(job_id)
            completed = self._completed_for(job_id)
            indices = [index for index, _ in rows]
            if len(indices) != len(set(indices)) or any(index in completed for index in indices):
                raise RuntimeError("Rubric Job attempted to persist duplicate row indices")
            if any(index < 0 or index >= record["total_rows"] for index in indices):
                raise RuntimeError("Rubric Job row index is out of range")
            results_root = self._job_root(job_id) / "results"
            results_root.mkdir(exist_ok=True)
            batch_index = len(tuple(results_root.glob("batch-*.jsonl")))
            payload = b"".join(
                (
                    _json({"row_index": index, "output": output.to_dict()}) + "\n"
                ).encode("utf-8")
                for index, output in rows
            )
            _write_bytes_atomic(
                results_root / f"batch-{batch_index:08d}.jsonl", payload
            )
            completed.update(indices)
            error_rows = sum(
                output.status == "error"
                for output in self._read_indexed_outputs(job_id).values()
            )
            self._update_job(
                job_id,
                completed_rows=len(completed),
                error_rows=error_rows,
            )

    def _finish_completed(self, job_id: str) -> None:
        outputs = self.get_audit_rows(job_id)
        with self._lock:
            record = self._job(job_id)
            if len(outputs) != record["total_rows"]:
                raise RuntimeError("Rubric Job completed without every result row")
            aggregate = _aggregate_usage(tuple(row.usage for row in outputs))
            state = (
                "completed_with_errors"
                if any(row.status == "error" for row in outputs)
                else "completed"
            )
            self._update_job(job_id, state=state, usage=aggregate.to_dict())

    def _finish_cancel(self, job_id: str) -> None:
        with self._lock:
            record = self._job(job_id)
            marker = record.get("error")
            if isinstance(marker, dict) and marker.get("code") == "deadline_exceeded":
                self._finish_expired(job_id, marker)
                return
        outputs = self.get_audit_rows(job_id)
        usage = _aggregate_usage(tuple(row.usage for row in outputs))
        with self._lock:
            self._update_job(job_id, state="cancelled", usage=usage.to_dict())

    def _finish_expired(self, job_id: str, marker: dict[str, Any]) -> None:
        inputs = parse_input_jsonl((self._job_root(job_id) / "input.jsonl").read_bytes())
        with self._lock:
            existing = set(self._completed_for(job_id))
        self._persist_rows(
            job_id,
            tuple(
                (
                    index,
                    self._output(
                        row,
                        ModelResponse(
                            None,
                            JobError(
                                str(marker["code"]),
                                str(marker["message"]),
                                bool(marker["retryable"]),
                            ),
                            _unavailable_usage(attempts=0),
                            "unavailable",
                            "unavailable",
                        ),
                    ),
                )
                for index, row in enumerate(inputs)
                if index not in existing
            ),
        )
        self._finish_completed(job_id)

    def _row_exists(self, job_id: str, index: int) -> bool:
        with self._lock:
            return index in self._completed_for(job_id)

    def _cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            return bool(self._job(job_id)["cancel_requested"])

    def _initialize(self) -> None:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        changed: list[str] = []
        for job_root in sorted(self.jobs_root.glob("job-*")):
            if not job_root.is_dir():
                continue
            record = _read_json(job_root / "status.json")
            status = _status(record)
            if job_root.name != status.job_id:
                raise ValueError("Local Judge job directory identity mismatch")
            inputs = parse_input_jsonl((job_root / "input.jsonl").read_bytes())
            if len(inputs) != status.total_rows:
                raise ValueError("Local Judge input and status row counts differ")
            metadata = _read_json(job_root / "metadata.json")
            if not isinstance(metadata, dict):
                raise ValueError("Local Judge job metadata is invalid")
            if status.job_id in self._jobs or status.submission_id in self._submission_ids:
                raise ValueError("Local Judge file store contains duplicate job identity")
            self._jobs[status.job_id] = record
            self._submission_ids[status.submission_id] = status.job_id
            self._next_sequence = max(self._next_sequence, status.queue_sequence + 1)
            if status.state == "running":
                indexed = self._read_indexed_outputs(status.job_id)
                self._completed_indices[status.job_id] = set(indexed)
                error = record.get("error")
                deadline_expired = (
                    isinstance(error, dict)
                    and error.get("code") == "deadline_exceeded"
                )
                record = {
                    **record,
                    "completed_rows": len(indexed),
                    "error_rows": sum(row.status == "error" for row in indexed.values()),
                    "state": (
                        "queued"
                        if not record["cancel_requested"] or deadline_expired
                        else "cancelled"
                    ),
                }
                self._jobs[status.job_id] = record
                changed.append(status.job_id)
        for job_id in changed:
            _write_json_atomic(self._job_root(job_id) / "status.json", self._jobs[job_id])

    def _job(self, job_id: str) -> dict[str, Any]:
        try:
            return self._jobs[job_id]
        except KeyError as error:
            raise KeyError(job_id) from error

    def _job_root(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def _update_job(self, job_id: str, **changes: Any) -> dict[str, Any]:
        record = {**self._job(job_id), **changes}
        _status(record)
        _write_json_atomic(self._job_root(job_id) / "status.json", record)
        self._jobs[job_id] = record
        return record

    def _completed_for(self, job_id: str) -> set[int]:
        completed = self._completed_indices.get(job_id)
        if completed is None:
            completed = set(self._read_indexed_outputs(job_id))
            self._completed_indices[job_id] = completed
        return completed

    def _read_indexed_outputs(self, job_id: str) -> dict[int, RubricOutputRow]:
        results_root = self._job_root(job_id) / "results"
        if not results_root.exists():
            return {}
        indexed: dict[int, RubricOutputRow] = {}
        for path in sorted(results_root.glob("batch-*.jsonl")):
            for line in path.read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict) or set(value) != {"row_index", "output"}:
                    raise ValueError("Local Judge result batch row is invalid")
                index = value["row_index"]
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError("Local Judge result row index is invalid")
                if index in indexed:
                    raise ValueError("Local Judge result row index is duplicated")
                indexed[index] = RubricOutputRow.from_dict(value["output"])
        return indexed


def _status(record: dict[str, Any]) -> JobStatus:
    if not isinstance(record, dict) or record.get("schema_version") != _STORE_SCHEMA:
        raise ValueError("Local Judge status file schema is invalid")
    error = record.get("error")
    return JobStatus(
        str(record.get("job_id") or ""),
        str(record.get("submission_id") or ""),
        str(record.get("content_digest") or ""),
        str(record.get("state") or ""),
        record.get("queue_sequence"),
        record.get("total_rows"),
        record.get("completed_rows"),
        record.get("error_rows"),
        JobError.from_dict(error) if error is not None else None,
    )


def _aggregate_usage(usages: tuple[TokenUsage, ...]) -> TokenUsage:
    if not usages:
        return _unavailable_usage(attempts=0)
    required = ("prompt_tokens", "completion_tokens", "total_tokens")
    known_all = all(
        all(getattr(usage, name) is not None for name in required) for usage in usages
    )
    known_any = any(
        any(getattr(usage, name) is not None for name in required) for usage in usages
    )
    status = "complete" if known_all else "partial" if known_any else "unavailable"

    def total(name: str) -> int | None:
        values = [getattr(usage, name) for usage in usages]
        return sum(int(value) for value in values if value is not None) if any(
            value is not None for value in values
        ) else None

    return TokenUsage.from_dict(
        {
            "usage_status": status,
            "prompt_tokens": total("prompt_tokens"),
            "completion_tokens": total("completion_tokens"),
            "total_tokens": total("total_tokens"),
            "cached_tokens": total("cached_tokens"),
            "reasoning_tokens": total("reasoning_tokens"),
            "attempts": sum(usage.attempts for usage in usages),
            "retries": sum(usage.retries for usage in usages),
            "requests": sum(usage.requests for usage in usages),
        }
    )


def _fake_usage() -> TokenUsage:
    return TokenUsage.from_dict(
        {
            "usage_status": "complete",
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "total_tokens": 12,
            "cached_tokens": None,
            "reasoning_tokens": None,
            "attempts": 1,
            "retries": 0,
            "requests": 1,
        }
    )


def _unavailable_usage(*, attempts: int = 1) -> TokenUsage:
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
            "requests": attempts,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Local Judge file must contain an object: {path.name}")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    _write_bytes_atomic(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
