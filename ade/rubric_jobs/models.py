"""Typed models for the provider-neutral Rubric Job protocol."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal


JOB_STATES = frozenset(
    {
        "queued",
        "running",
        "completed",
        "completed_with_errors",
        "failed",
        "cancelled",
    }
)
TERMINAL_JOB_STATES = frozenset(
    {"completed", "completed_with_errors", "failed", "cancelled"}
)
UsageStatus = Literal["complete", "partial", "unavailable"]


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_count(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer or null")
    return value


@dataclass(frozen=True)
class TokenUsage:
    usage_status: UsageStatus
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    attempts: int
    retries: int
    requests: int

    @classmethod
    def from_dict(cls, value: object) -> "TokenUsage":
        if not isinstance(value, dict):
            raise ValueError("usage must be an object")
        if set(value) != {
            "usage_status",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "attempts",
            "retries",
            "requests",
        }:
            raise ValueError("usage fields do not match ade.rubric_jobs.v1")
        status = value["usage_status"]
        if status not in {"complete", "partial", "unavailable"}:
            raise ValueError("usage_status is invalid")
        counts = {
            name: _optional_count(value[name], f"usage.{name}")
            for name in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cached_tokens",
                "reasoning_tokens",
            )
        }
        attempts = _optional_count(value["attempts"], "usage.attempts")
        retries = _optional_count(value["retries"], "usage.retries")
        requests = _optional_count(value["requests"], "usage.requests")
        assert attempts is not None and retries is not None and requests is not None
        if retries > attempts or requests != attempts:
            raise ValueError("usage attempts/retries/requests are inconsistent")
        required = ("prompt_tokens", "completion_tokens", "total_tokens")
        if status == "complete" and any(counts[name] is None for name in required):
            raise ValueError("complete usage requires prompt/completion/total tokens")
        if status == "unavailable" and any(counts[name] is not None for name in counts):
            raise ValueError("unavailable usage must not fabricate token counts")
        if counts["total_tokens"] is not None:
            if counts["prompt_tokens"] is None or counts["completion_tokens"] is None:
                raise ValueError("known total_tokens requires prompt and completion tokens")
            if counts["total_tokens"] != (
                counts["prompt_tokens"] + counts["completion_tokens"]
            ):
                raise ValueError("usage total_tokens is inconsistent")
        return cls(status, **counts, attempts=attempts, retries=retries, requests=requests)

    def to_dict(self) -> dict[str, object]:
        return {
            "usage_status": self.usage_status,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "attempts": self.attempts,
            "retries": self.retries,
            "requests": self.requests,
        }


@dataclass(frozen=True)
class JobError:
    code: str
    message: str
    retryable: bool = False

    @classmethod
    def from_dict(cls, value: object) -> "JobError":
        if not isinstance(value, dict) or set(value) != {"code", "message", "retryable"}:
            raise ValueError("error fields do not match ade.rubric_jobs.v1")
        if not isinstance(value["retryable"], bool):
            raise ValueError("error.retryable must be boolean")
        return cls(
            _required_string(value["code"], "error.code"),
            _required_string(value["message"], "error.message"),
            value["retryable"],
        )

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


@dataclass(frozen=True)
class Rubric:
    template: str
    required_variables: tuple[str, ...]
    output_schema: dict[str, Any]

    @classmethod
    def from_dict(cls, value: object) -> "Rubric":
        if not isinstance(value, dict) or set(value) != {
            "template",
            "required_variables",
            "output_schema",
        }:
            raise ValueError("rubric fields do not match ade.rubric_jobs.v1")
        template = _required_string(value["template"], "rubric.template")
        variables = value["required_variables"]
        if (
            not isinstance(variables, list)
            or not variables
            or any(not isinstance(item, str) or not item for item in variables)
            or len(set(variables)) != len(variables)
        ):
            raise ValueError("rubric.required_variables must be unique non-empty strings")
        schema = value["output_schema"]
        if not isinstance(schema, dict):
            raise ValueError("rubric.output_schema must be an object")
        return cls(template, tuple(variables), schema)

    def to_dict(self) -> dict[str, object]:
        return {
            "template": self.template,
            "required_variables": list(self.required_variables),
            "output_schema": self.output_schema,
        }


@dataclass(frozen=True)
class RubricInputRow:
    record_id: str
    rubric: Rubric
    inputs: dict[str, str]
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, value: object) -> "RubricInputRow":
        if not isinstance(value, dict) or set(value) != {
            "record_id",
            "rubric",
            "inputs",
            "metadata",
        }:
            raise ValueError("input row fields do not match ade.rubric_jobs.v1")
        inputs = value["inputs"]
        if not isinstance(inputs, dict) or any(
            not isinstance(name, str) or not isinstance(item, str)
            for name, item in inputs.items()
        ):
            raise ValueError("inputs must map variable names to strings")
        metadata = value["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        return cls(
            _required_string(value["record_id"], "record_id"),
            Rubric.from_dict(value["rubric"]),
            dict(inputs),
            dict(metadata),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "rubric": self.rubric.to_dict(),
            "inputs": self.inputs,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RubricOutputRow:
    record_id: str
    status: Literal["ok", "error"]
    result: dict[str, Any] | None
    error: JobError | None
    usage: TokenUsage
    model: dict[str, str]
    rubric_digest: str
    output_schema_digest: str
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, value: object) -> "RubricOutputRow":
        if not isinstance(value, dict) or set(value) != {
            "record_id",
            "status",
            "result",
            "error",
            "usage",
            "model",
            "rubric_digest",
            "output_schema_digest",
            "metadata",
        }:
            raise ValueError("output row fields do not match ade.rubric_jobs.v1")
        status = value["status"]
        if status not in {"ok", "error"}:
            raise ValueError("output row status is invalid")
        result = value["result"]
        error = value["error"]
        if status == "ok" and (not isinstance(result, dict) or error is not None):
            raise ValueError("ok output row requires result and no error")
        if status == "error" and (result is not None or not isinstance(error, dict)):
            raise ValueError("error output row requires error and no result")
        model = value["model"]
        if (
            not isinstance(model, dict)
            or set(model) != {"id", "digest"}
            or any(not isinstance(item, str) or not item for item in model.values())
        ):
            raise ValueError("output row model identity is invalid")
        metadata = value["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("output row metadata must be an object")
        return cls(
            _required_string(value["record_id"], "record_id"),
            status,
            dict(result) if isinstance(result, dict) else None,
            JobError.from_dict(error) if isinstance(error, dict) else None,
            TokenUsage.from_dict(value["usage"]),
            dict(model),
            _required_string(value["rubric_digest"], "rubric_digest"),
            _required_string(value["output_schema_digest"], "output_schema_digest"),
            dict(metadata),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "status": self.status,
            "result": self.result,
            "error": self.error.to_dict() if self.error else None,
            "usage": self.usage.to_dict(),
            "model": self.model,
            "rubric_digest": self.rubric_digest,
            "output_schema_digest": self.output_schema_digest,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    submission_id: str
    content_digest: str
    state: str
    queue_sequence: int
    total_rows: int
    completed_rows: int
    error_rows: int
    error: JobError | None = None

    def __post_init__(self) -> None:
        _required_string(self.job_id, "job_id")
        _required_string(self.submission_id, "submission_id")
        _required_string(self.content_digest, "content_digest")
        if self.state not in JOB_STATES:
            raise ValueError("job state is invalid")
        for name in ("queue_sequence", "total_rows", "completed_rows", "error_rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.completed_rows > self.total_rows or self.error_rows > self.completed_rows:
            raise ValueError("job progress exceeds total rows")

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": "ade.rubric_jobs.v1",
            "job_id": self.job_id,
            "submission_id": self.submission_id,
            "content_digest": self.content_digest,
            "state": self.state,
            "queue_sequence": self.queue_sequence,
            "total_rows": self.total_rows,
            "completed_rows": self.completed_rows,
            "error_rows": self.error_rows,
            "error": self.error.to_dict() if self.error else None,
        }


@dataclass(frozen=True)
class JobResult:
    status: JobStatus
    rows: tuple[RubricOutputRow, ...]
    usage: TokenUsage

    def __post_init__(self) -> None:
        if self.status.state not in TERMINAL_JOB_STATES:
            raise ValueError("job result requires a terminal job")
