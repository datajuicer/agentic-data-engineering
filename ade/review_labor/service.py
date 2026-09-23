"""Validation, persistence, and usage accounting for Review Labor."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
import uuid

from ade.rubric_jobs import Rubric, RubricInputRow, TokenUsage, encode_input_jsonl
from ade.review_labor.usage import OPTIONAL_USAGE_FIELDS, aggregate_usage
from ade.review_labor.jobs import ReviewLaborJobService

Reviewer = Callable[..., dict[str, Any]]

_PHASES = frozenset({"initial_coverage", "follow_up"})
_CORE_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
_OPTIONAL_USAGE_FIELDS = OPTIONAL_USAGE_FIELDS


class ReviewLaborService:
    def __init__(
        self,
        *,
        reviewer: Reviewer,
        model: str,
        output_root: str | Path | None = None,
        workspace_root: str | Path | None = None,
        rubric_jobs: ReviewLaborJobService | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Review Labor model is required")
        self.reviewer = reviewer
        self.model = model.strip()
        self.rubric_jobs = rubric_jobs
        self.output_root = (
            Path(output_root).resolve() if output_root is not None else None
        )
        self.workspace_root = (
            Path(workspace_root).resolve() if workspace_root is not None else None
        )
        if self.output_root is not None and self.workspace_root is None:
            raise ValueError("Review Labor output root requires a workspace root")

    def review_batch_to_file(
        self,
        units: tuple[dict[str, Any], ...],
        rubrics: tuple[dict[str, Any], ...],
        output_root: str | Path | None = None,
        *,
        submission_id: str,
        phase: str,
        round_id: str,
        investigation_purpose: str,
        identity_metadata: dict[str, Any] | None = None,
        progress_callback: Callable[[Any], None] | None = None,
    ) -> dict[str, Any]:
        resolved_output_root = self._resolve_output_root(output_root)
        if resolved_output_root is None:
            raise RuntimeError("Review Labor artifact output is not configured")
        if not submission_id.strip():
            raise ValueError("Review Labor submission_id is required")
        _validate_investigation(phase, round_id, investigation_purpose)
        unit_ids = _validate_units(units)
        _validate_review_unit_provenance(units)
        rubric_labels = _validate_rubrics(rubrics)
        reviewed = (
            self._review_with_rubric_job(
                units,
                rubrics,
                submission_id=submission_id.strip(),
                phase=phase,
                round_id=round_id,
                investigation_purpose=investigation_purpose,
                identity_metadata=identity_metadata,
                progress_callback=progress_callback,
            )
            if self.rubric_jobs is not None
            else self.reviewer(units=list(units), rubrics=list(rubrics))
        )
        normalized = _normalize_reviewer_result(reviewed, unit_ids, rubric_labels)

        review_id = f"review-{uuid.uuid4().hex}"
        review_dir = resolved_output_root / review_id
        review_dir.mkdir(parents=True, exist_ok=False)
        results_path = review_dir / "results.jsonl"
        usage_by_id = {item["unit_id"]: item for item in normalized["unit_usage"]}
        provider_responses_by_id = {
            item["unit_id"]: item for item in normalized["provider_responses"]
        }
        result_rows = [
            {
                **row,
                **(
                    {
                        "provider_response": _public_provider_response(
                            provider_responses_by_id[str(row["unit_id"])]
                        )
                    }
                    if str(row["unit_id"]) in provider_responses_by_id
                    else {}
                ),
                "usage": _public_unit_usage(usage_by_id[str(row["unit_id"])]),
            }
            for row in normalized["feedback"]
        ]
        encoded_rows = b"".join(
            (
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            ).encode("utf-8")
            for row in result_rows
        )
        _write_bytes_atomic(results_path, encoded_rows)
        manifest = {
            "schema_version": "3",
            "review_id": review_id,
            "status": (
                "completed"
                if not normalized["failed_unit_ids"]
                and not normalized["fallback_unit_ids"]
                else "partial"
            ),
            "model": self.model,
            "submission_id": submission_id.strip(),
            "phase": phase,
            "round_id": round_id.strip(),
            "investigation_purpose": investigation_purpose.strip(),
            **(identity_metadata or {}),
            "source_artifact_ids": sorted(
                {str(unit["source_artifact_id"]) for unit in units}
            ),
            "requested_unit_ids": list(unit_ids),
            "units": [
                {
                    "unit_id": str(unit["unit_id"]),
                    "source_artifact_id": str(unit["source_artifact_id"]),
                    "source_record_id": str(unit["source_record_id"]),
                    "locator": unit["locator"],
                    "question_sha256": hashlib.sha256(
                        str(unit["question"]).encode("utf-8")
                    ).hexdigest(),
                    "response_sha256": hashlib.sha256(
                        str(unit["response"]).encode("utf-8")
                    ).hexdigest(),
                }
                for unit in units
            ],
            "completed_unit_ids": normalized["completed_unit_ids"],
            "failed_unit_ids": normalized["failed_unit_ids"],
            "fallback_unit_ids": normalized["fallback_unit_ids"],
            "failures": normalized["failures"],
            "rubrics": list(rubrics),
            "unit_usage": normalized["unit_usage"],
            "usage": normalized["usage"],
            "results_sha256": hashlib.sha256(encoded_rows).hexdigest(),
            "results_size_bytes": len(encoded_rows),
        }
        manifest_path = review_dir / "manifest.json"
        _write_bytes_atomic(manifest_path, _encode(manifest))
        for path in (results_path, manifest_path):
            path.chmod(0o444)
        review_dir.chmod(0o555)
        return {
            "review_id": review_id,
            "status": manifest["status"],
            "results_path": results_path.relative_to(self.workspace_root).as_posix(),
            "manifest_path": manifest_path.relative_to(self.workspace_root).as_posix(),
            "completed_units": len(normalized["completed_unit_ids"]),
            "failed_units": len(normalized["failed_unit_ids"]),
            "fallback_units": len(normalized["fallback_unit_ids"]),
            "unit_usage": normalized["unit_usage"],
            "usage": normalized["usage"],
        }

    def _review_with_rubric_job(
        self,
        units: tuple[dict[str, Any], ...],
        rubrics: tuple[dict[str, Any], ...],
        *,
        submission_id: str,
        phase: str,
        round_id: str,
        investigation_purpose: str,
        identity_metadata: dict[str, Any] | None,
        progress_callback: Callable[[Any], None] | None,
    ) -> dict[str, Any]:
        assert self.rubric_jobs is not None
        row_rubric = Rubric.from_dict(
            {
                "template": _review_prompt_template(rubrics),
                "required_variables": ["question", "response", "context"],
                "output_schema": _review_output_schema(rubrics),
            }
        )
        rows = tuple(
            RubricInputRow(
                str(unit["unit_id"]),
                row_rubric,
                {
                    "question": str(unit["question"]),
                    "response": str(unit["response"]),
                    "context": json.dumps(
                        unit.get("context") or {}, sort_keys=True
                    ),
                },
                {
                    "source_artifact_id": unit["source_artifact_id"],
                    "source_record_id": unit["source_record_id"],
                    "locator": unit["locator"],
                    "context": dict(unit.get("context") or {}),
                },
            )
            for unit in units
        )
        status = self.rubric_jobs.submit(
            submission_id=submission_id,
            input_jsonl=encode_input_jsonl(rows),
            job_metadata={
                "phase": phase,
                "round_id": round_id.strip(),
                "investigation_purpose": investigation_purpose.strip(),
                **(identity_metadata or {}),
            },
            progress_callback=progress_callback,
        )
        result = self.rubric_jobs.get_result(status.job_id)
        raw_responses = self.rubric_jobs.audit_provider_responses(status.job_id)
        feedback = []
        usage = []
        provider_responses = []
        for output, raw in zip(result.rows, raw_responses, strict=True):
            if output.status == "ok":
                assert output.result is not None
                feedback.append({"unit_id": output.record_id, **output.result})
            else:
                assert output.error is not None
                feedback.append(
                    {
                        "unit_id": output.record_id,
                        "review_error": {
                            "type": output.error.code,
                            "message": output.error.message,
                        },
                    }
                )
            usage.append({"unit_id": output.record_id, **output.usage.to_dict()})
            if raw is not None:
                reasoning_content, content = _provider_response_text(raw)
                provider_responses.append(
                    {
                        "unit_id": output.record_id,
                        "reasoning_content": reasoning_content,
                        "content": content,
                        "raw_response": raw,
                    }
                )
        return {
            "feedback": feedback,
            "unit_usage": usage,
            "provider_responses": provider_responses,
        }

    def _resolve_output_root(self, output_root: str | Path | None) -> Path | None:
        requested = output_root if output_root is not None else self.output_root
        if requested is None or self.workspace_root is None:
            return None
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(
                "Review Labor output root must be inside the workspace"
            ) from exc
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate


def _validate_investigation(phase: str, round_id: str, purpose: str) -> None:
    if phase not in _PHASES:
        raise ValueError(f"Review Labor phase must be one of {sorted(_PHASES)}")
    for name, value in (("round_id", round_id), ("investigation_purpose", purpose)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Review Labor {name} must be non-empty")


def _review_output_schema(rubrics: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    items = []
    for rubric in rubrics:
        items.append(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "rubric_id": {"const": rubric["rubric_id"]},
                    "label": {"enum": rubric["labels"]},
                    "observations": {"type": "array", "items": {"type": "string"}},
                    "evidence_spans": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "rubric_id",
                    "label",
                    "observations",
                    "evidence_spans",
                    "confidence",
                ],
            }
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rubric_results": {
                "type": "array",
                "prefixItems": items,
                "items": False,
                "minItems": len(items),
                "maxItems": len(items),
            }
        },
        "required": ["rubric_results"],
    }


def _review_prompt_template(rubrics: tuple[dict[str, Any], ...]) -> str:
    ordered = "\n".join(
        f"{index}. {rubric['rubric_id']}: {rubric['instruction']} "
        f"Allowed labels: {' | '.join(str(label) for label in rubric['labels'])}"
        for index, rubric in enumerate(rubrics)
    )
    return (
        "Review the response against every ordered rubric below.\n"
        f"{ordered}\n"
        "Question:\n{{question}}\n"
        "Response:\n{{response}}\n"
        "Task context (JSON):\n{{context}}"
    )


def _validate_units(units: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    if not units:
        raise ValueError("units must be non-empty")
    unit_ids: list[str] = []
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("each unit must be an object")
        unit_id = unit.get("unit_id")
        question = unit.get("question")
        response = unit.get("response")
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise ValueError("unit ID must be non-empty")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("unit question must be non-empty")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("unit response must be non-empty")
        unit_ids.append(unit_id)
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("unit IDs must be unique")
    return tuple(unit_ids)


def _validate_review_unit_provenance(units: tuple[dict[str, Any], ...]) -> None:
    for unit in units:
        artifact_id = unit.get("source_artifact_id")
        record_id = unit.get("source_record_id")
        locator = unit.get("locator")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ValueError("review units require source_artifact_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("review units require source_record_id")
        expected = f"{artifact_id}::{record_id}"
        if unit.get("unit_id") != expected:
            raise ValueError(
                "review unit_id must be <source_artifact_id>::<source_record_id>"
            )
        if not isinstance(locator, dict):
            raise ValueError("review units require locator")


def _validate_rubrics(
    rubrics: tuple[dict[str, Any], ...],
) -> dict[str, tuple[str, ...]]:
    if not rubrics:
        raise ValueError("rubrics must be non-empty")
    result: dict[str, tuple[str, ...]] = {}
    for rubric in rubrics:
        if not isinstance(rubric, dict):
            raise ValueError("each rubric must be an object")
        rubric_id = rubric.get("rubric_id")
        instruction = rubric.get("instruction")
        labels = rubric.get("labels")
        if not isinstance(rubric_id, str) or not rubric_id.strip():
            raise ValueError("rubric ID must be non-empty")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("rubric instruction must be non-empty")
        if (
            not isinstance(labels, list)
            or len(labels) < 2
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(labels) != len(set(labels))
        ):
            raise ValueError("rubric labels must contain at least two unique strings")
        if rubric_id in result:
            raise ValueError(f"duplicate rubric ID: {rubric_id}")
        result[rubric_id] = tuple(labels)
    return result


def _normalize_reviewer_result(
    value: object,
    unit_ids: tuple[str, ...],
    rubric_labels: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("reviewer result must be an object")
    feedback = value.get("feedback")
    unit_usage = value.get("unit_usage")
    provider_responses = value.get("provider_responses", [])
    if (
        not isinstance(feedback, list)
        or not isinstance(unit_usage, list)
        or not isinstance(provider_responses, list)
    ):
        raise ValueError("reviewer result requires feedback and unit_usage lists")
    by_id: dict[str, dict[str, Any]] = {}
    for item in feedback:
        if not isinstance(item, dict) or not isinstance(item.get("unit_id"), str):
            raise ValueError("reviewer feedback requires unit_id")
        unit_id = item["unit_id"]
        if unit_id in by_id:
            raise ValueError(f"reviewer duplicated unit {unit_id}")
        by_id[unit_id] = item
    outside = set(by_id) - set(unit_ids)
    if outside:
        raise ValueError(f"reviewer returned units outside request: {sorted(outside)}")
    usage_by_id = _validate_unit_usage(unit_usage, unit_ids)
    normalized_provider_responses = _validate_provider_responses(
        provider_responses, unit_ids
    )
    provider_responses_by_id = {
        item["unit_id"]: item for item in normalized_provider_responses
    }
    completed: list[str] = []
    fallbacks: list[str] = []
    failures: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    for unit_id in unit_ids:
        item = by_id.get(unit_id)
        if item is None:
            failures.append(
                {
                    "unit_id": unit_id,
                    "error_type": "MissingReview",
                    "message": "reviewer returned no result",
                }
            )
            continue
        error = item.get("review_error")
        if error is not None:
            if (
                not isinstance(error, dict)
                or not isinstance(error.get("type"), str)
                or not error["type"].strip()
                or not isinstance(error.get("message"), str)
                or not error["message"].strip()
            ):
                raise ValueError("reviewer feedback review_error is malformed")
            failures.append(
                {
                    "unit_id": unit_id,
                    "error_type": error["type"],
                    "message": error["message"],
                }
            )
            rows.append(
                {
                    "unit_id": unit_id,
                    "status": "invalid",
                    "review_error": {
                        "type": error["type"],
                        "message": error["message"],
                    },
                }
            )
            continue
        fallback = item.get("review_fallback")
        if fallback is not None:
            if (
                not isinstance(fallback, dict)
                or not isinstance(fallback.get("error_type"), str)
                or not fallback["error_type"].strip()
                or not isinstance(fallback.get("message"), str)
                or not fallback["message"].strip()
                or unit_id not in provider_responses_by_id
            ):
                raise ValueError("reviewer feedback review_fallback is malformed")
            completed.append(unit_id)
            fallbacks.append(unit_id)
            rows.append(item)
            continue
        _validate_review(item, rubric_labels)
        completed.append(unit_id)
        rows.append(item)
    normalized_usage = [usage_by_id[unit_id] for unit_id in unit_ids]
    return {
        "completed_unit_ids": completed,
        "failed_unit_ids": [item["unit_id"] for item in failures],
        "fallback_unit_ids": fallbacks,
        "failures": failures,
        "feedback": rows,
        "unit_usage": normalized_usage,
        "provider_responses": normalized_provider_responses,
        "usage": aggregate_usage(normalized_usage),
    }


def _validate_unit_usage(
    values: list[Any], unit_ids: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("unit_id"), str):
            raise ValueError("unit_usage entries require unit_id")
        unit_id = str(raw["unit_id"])
        if unit_id not in unit_ids or unit_id in result:
            raise ValueError("unit_usage contains an invalid or duplicate unit_id")
        attempts = raw.get("attempts")
        usage_status = raw.get("usage_status")
        if type(attempts) is not int or attempts < 0:
            raise ValueError("unit_usage attempts must be non-negative")
        if attempts == 0 and usage_status != "unavailable":
            raise ValueError(
                "unit_usage attempts may be zero only for unavailable usage"
            )
        usage = TokenUsage.from_dict(
            {
                "usage_status": usage_status,
                **{
                    field: raw.get(field)
                    for field in (*_CORE_USAGE_FIELDS, *_OPTIONAL_USAGE_FIELDS)
                },
                "attempts": attempts,
                "retries": raw.get("retries", max(0, attempts - 1)),
                "requests": raw.get("requests", attempts),
            }
        )
        item = {"unit_id": unit_id, **usage.to_dict()}
        result[unit_id] = item
    if set(result) != set(unit_ids):
        raise ValueError("reviewer must return usage for every requested unit")
    return result


def _validate_provider_responses(
    values: list[Any], unit_ids: tuple[str, ...]
) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("unit_id"), str):
            raise ValueError("provider responses require unit_id")
        unit_id = str(raw["unit_id"])
        reasoning_content = raw.get("reasoning_content")
        content = raw.get("content")
        raw_response = raw.get("raw_response")
        if unit_id not in unit_ids or unit_id in result:
            raise ValueError(
                "provider responses contain an invalid or duplicate unit_id"
            )
        if not isinstance(reasoning_content, str):
            raise ValueError("provider reasoning_content must be a string")
        if not isinstance(content, str):
            raise ValueError("provider content must be a string")
        if not isinstance(raw_response, Mapping):
            raise ValueError("provider raw_response must be an object")
        result[unit_id] = {
            "unit_id": unit_id,
            "reasoning_content": reasoning_content,
            "content": content,
            "raw_response": dict(raw_response),
        }
    return [result[unit_id] for unit_id in unit_ids if unit_id in result]


def _public_unit_usage(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "unit_id"}


def _public_provider_response(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "unit_id"}


def _provider_response_text(raw: Mapping[str, Any]) -> tuple[str, str]:
    final_response = raw.get("final_response")
    if isinstance(final_response, Mapping):
        raw = final_response
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, Mapping) else None
        if isinstance(message, Mapping):
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            content = message.get("content", "")
            return (
                reasoning if isinstance(reasoning, str) else "",
                content if isinstance(content, str) else "",
            )
    return "", json.dumps(raw, ensure_ascii=False, sort_keys=True)


def _validate_review(
    item: dict[str, Any], rubric_labels: dict[str, tuple[str, ...]]
) -> None:
    results = item.get("rubric_results")
    if not isinstance(results, list):
        raise ValueError("reviewer feedback requires rubric_results")
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("rubric result must be an object")
        rubric_id = result.get("rubric_id")
        label = result.get("label")
        observations = result.get("observations")
        evidence_spans = result.get("evidence_spans")
        confidence = result.get("confidence")
        if rubric_id not in rubric_labels or rubric_id in seen:
            raise ValueError("reviewer returned an invalid rubric ID")
        if label not in rubric_labels[rubric_id]:
            raise ValueError("reviewer returned a label outside the rubric")
        if not isinstance(observations, list) or any(
            not isinstance(v, str) for v in observations
        ):
            raise ValueError("review observations must be strings")
        if not isinstance(evidence_spans, list) or any(
            not isinstance(v, str) for v in evidence_spans
        ):
            raise ValueError("review evidence spans must be strings")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise ValueError("review confidence must be between 0 and 1")
        seen.add(str(rubric_id))
    missing = set(rubric_labels) - seen
    if missing:
        raise ValueError(f"reviewer omitted requested rubrics: {sorted(missing)}")


def _encode(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
