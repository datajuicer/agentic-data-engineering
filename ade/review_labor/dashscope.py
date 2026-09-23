"""No-proxy DashScope provider for rubric-guided text review."""

from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
import functools
import hashlib
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import threading
import time
from typing import Any, Callable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from ade.rubric_jobs import JobError, RubricInputRow, TokenUsage
from ade.review_labor.jobs import ReviewResponse
from ade.review_labor.structured_review import (
    expand_compact_review_result as _expand_compact_review_result,
    review_schema_specs as _review_schema_specs,
    rubric_job_messages,
)


class DashScopeReviewRequestError(RuntimeError):
    def __init__(self, status_code: int, detail: str, *, retryable: bool) -> None:
        super().__init__(
            f"DashScope review request failed with HTTP {status_code}: {detail}"
        )
        self.status_code = status_code
        self.retryable = retryable


class DashScopeReviewTransport:
    def __init__(
        self,
        *,
        api_key: str,
        workspace_id: str,
        model: str,
        max_units_per_request: int = 1,
        max_concurrent_requests: int = 128,
        max_request_chars: int = 800_000,
        timeout_seconds: int = 300,
        batch_timeout_seconds: float | None = None,
        temperature: float = 0.0,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.5,
        max_completion_tokens: int = 4096,
        thinking_budget: int = 2048,
        enable_thinking: bool = True,
        score_only: bool = False,
        request_json: Callable[..., dict[str, Any]] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not api_key.strip() or not workspace_id.strip() or not model.strip():
            raise ValueError("DashScope API key, workspace ID, and model are required")
        if max_units_per_request != 1:
            raise ValueError(
                "DashScope review batch limits require exactly one unit per request"
            )
        if (
            max_concurrent_requests <= 0
            or max_request_chars <= 0
            or timeout_seconds <= 0
            or max_attempts <= 0
            or max_completion_tokens <= 0
            or (enable_thinking and thinking_budget <= 0)
        ):
            raise ValueError("DashScope review batch limits must be positive")
        if enable_thinking and thinking_budget > max_completion_tokens:
            raise ValueError(
                "DashScope thinking budget cannot exceed completion budget"
            )
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("DashScope review temperature must be between 0 and 2")
        if not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
            raise ValueError("DashScope review retry delay must be non-negative")
        if batch_timeout_seconds is not None and (
            isinstance(batch_timeout_seconds, bool)
            or not isinstance(batch_timeout_seconds, (int, float))
            or not math.isfinite(batch_timeout_seconds)
            or batch_timeout_seconds <= 0
        ):
            raise ValueError("DashScope batch timeout must be positive and finite")
        self._api_key = api_key
        self._url = _chat_completions_url(workspace_id)
        self.model = model.strip()
        self.max_units_per_request = max_units_per_request
        self.max_concurrent_requests = max_concurrent_requests
        self.max_request_chars = max_request_chars
        self.timeout_seconds = timeout_seconds
        self.batch_timeout_seconds = batch_timeout_seconds
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.max_completion_tokens = max_completion_tokens
        self.thinking_budget = thinking_budget
        self.enable_thinking = enable_thinking
        self.score_only = score_only
        self._request_json = request_json or _request_json_without_proxy
        self._sleep = sleep or time.sleep
        self._request_slots = threading.BoundedSemaphore(max_concurrent_requests)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_requests,
            thread_name_prefix="review-labor-request",
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def evaluate(
        self, row: RubricInputRow, rendered_prompt: str
    ) -> ReviewResponse:
        """Evaluate one provider-neutral Rubric Job row through the API provider."""
        import asyncio

        attempts: list[dict[str, int | None] | None] = []
        completed_responses: list[dict[str, Any]] = []
        validation_feedback: str | None = None
        for attempt in range(self.max_attempts):
            raw_response: dict[str, Any] | None = None
            try:
                body = _rubric_job_request_body(
                    model=self.model,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                    enable_thinking=self.enable_thinking,
                    thinking_budget=self.thinking_budget,
                    rendered_prompt=rendered_prompt,
                    output_schema=row.rubric.output_schema,
                    validation_feedback=validation_feedback,
                )
                raw_response = await asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    functools.partial(
                        self._request_json,
                        url=self._url,
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                            "X-DashScope-Wait-Timeout": "30",
                        },
                        body=body,
                        timeout_seconds=self.timeout_seconds,
                    ),
                )
                completed_responses.append(raw_response)
                attempts.append(_response_usage(raw_response))
                _, content = _response_contents(raw_response)
                result = _decode_response_content(content)
                if not isinstance(result, dict):
                    raise ValueError("DashScope Rubric Job result must be an object")
                specs = _review_schema_specs(row.rubric.output_schema)
                if specs is not None:
                    result = _expand_compact_review_result(result, specs)
                Draft202012Validator(row.rubric.output_schema).validate(result)
                return ReviewResponse(
                    result,
                    None,
                    _token_usage(attempts),
                    self.model,
                    self.model,
                    _audited_provider_response(completed_responses),
                )
            except (KeyError, TypeError, ValueError, ValidationError) as error:
                if raw_response is not None:
                    validation_feedback = " ".join(str(error).split())[:300]
                    if attempt + 1 == self.max_attempts:
                        return ReviewResponse(
                            None,
                            JobError("schema_invalid", validation_feedback, False),
                            _token_usage(attempts),
                            self.model,
                            self.model,
                            _audited_provider_response(completed_responses),
                        )
                else:
                    attempts.append(None)
                if raw_response is None and attempt + 1 == self.max_attempts:
                    return ReviewResponse(
                        None,
                        JobError("endpoint_error", str(error), False),
                        _token_usage(attempts),
                        self.model,
                        self.model,
                        None,
                    )
            except Exception as error:
                attempts.append(None)
                if attempt + 1 == self.max_attempts:
                    return ReviewResponse(
                        None,
                        JobError("endpoint_error", str(error), False),
                        _token_usage(attempts),
                        self.model,
                        self.model,
                        _audited_provider_response(completed_responses),
                    )
            if self.retry_delay_seconds:
                await asyncio.sleep(self.retry_delay_seconds)

    @classmethod
    def from_config(
        cls,
        *,
        project_root: str | Path,
        provider: dict[str, Any],
        request_json: Callable[..., dict[str, Any]] | None = None,
    ) -> "DashScopeReviewTransport":
        values = _read_dotenv(Path(project_root) / ".env")

        def secret(name_field: str) -> str:
            env_name = str(provider[name_field])
            return os.environ.get(env_name) or values.get(env_name, "")

        return cls(
            api_key=secret("api_key_env"),
            workspace_id=secret("workspace_id_env"),
            model=str(provider["model"]),
            max_concurrent_requests=int(provider["max_concurrent_requests"]),
            timeout_seconds=int(provider["request_timeout_seconds"]),
            batch_timeout_seconds=(
                None
                if provider["batch_timeout_seconds"] is None
                else float(provider["batch_timeout_seconds"])
            ),
            temperature=float(provider["temperature"]),
            max_attempts=int(provider["max_attempts"]),
            retry_delay_seconds=float(provider["retry_delay_seconds"]),
            max_completion_tokens=int(provider["max_completion_tokens"]),
            thinking_budget=int(provider["thinking_budget"]),
            enable_thinking=bool(provider["enable_thinking"]),
            request_json=request_json,
        )

    @classmethod
    def from_project_env(
        cls,
        *,
        project_root: str | Path,
        max_units_per_request: int = 1,
        max_concurrent_requests: int | None = None,
        max_request_chars: int = 800_000,
        timeout_seconds: int = 300,
        batch_timeout_seconds: float | None = None,
        temperature: float | None = None,
        max_attempts: int | None = None,
        retry_delay_seconds: float | None = None,
        max_completion_tokens: int | None = None,
        thinking_budget: int | None = None,
        enable_thinking: bool = True,
        score_only: bool = False,
        request_json: Callable[..., dict[str, Any]] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> "DashScopeReviewTransport":
        values = _read_dotenv(Path(project_root) / ".env")
        concurrency = (
            max_concurrent_requests
            if max_concurrent_requests is not None
            else _positive_integer(
                values.get("REVIEW_LABOR_MAX_CONCURRENT_REQUESTS", "128"),
                "REVIEW_LABOR_MAX_CONCURRENT_REQUESTS",
            )
        )
        resolved_temperature = (
            temperature
            if temperature is not None
            else _finite_float(
                values.get("REVIEW_LABOR_TEMPERATURE", "0"),
                "REVIEW_LABOR_TEMPERATURE",
            )
        )
        resolved_max_attempts = (
            max_attempts
            if max_attempts is not None
            else _positive_integer(
                values.get("REVIEW_LABOR_MAX_ATTEMPTS", "3"),
                "REVIEW_LABOR_MAX_ATTEMPTS",
            )
        )
        resolved_retry_delay = (
            retry_delay_seconds
            if retry_delay_seconds is not None
            else _nonnegative_finite_float(
                values.get("REVIEW_LABOR_RETRY_DELAY_SECONDS", "0.5"),
                "REVIEW_LABOR_RETRY_DELAY_SECONDS",
            )
        )
        resolved_max_completion_tokens = (
            max_completion_tokens
            if max_completion_tokens is not None
            else _positive_integer(
                values.get("REVIEW_LABOR_MAX_COMPLETION_TOKENS", "4096"),
                "REVIEW_LABOR_MAX_COMPLETION_TOKENS",
            )
        )
        resolved_thinking_budget = (
            thinking_budget
            if thinking_budget is not None
            else _positive_integer(
                values.get("REVIEW_LABOR_THINKING_BUDGET", "2048"),
                "REVIEW_LABOR_THINKING_BUDGET",
            )
        )
        return cls(
            api_key=values.get("DASHSCOPE_API_KEY", ""),
            workspace_id=values.get("DASHSCOPE_WORKSPACE_ID", ""),
            model=values.get("REVIEW_LABOR_MODEL", ""),
            max_units_per_request=max_units_per_request,
            max_concurrent_requests=concurrency,
            max_request_chars=max_request_chars,
            timeout_seconds=timeout_seconds,
            batch_timeout_seconds=batch_timeout_seconds,
            temperature=resolved_temperature,
            max_attempts=resolved_max_attempts,
            retry_delay_seconds=resolved_retry_delay,
            max_completion_tokens=resolved_max_completion_tokens,
            thinking_budget=resolved_thinking_budget,
            enable_thinking=enable_thinking,
            score_only=score_only,
            request_json=request_json,
            sleep=sleep,
        )

    def review_units(
        self,
        *,
        units: list[dict[str, Any]],
        rubrics: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self.review_units_with_usage(units=units, rubrics=rubrics)["feedback"]

    def review_units_with_usage(
        self,
        *,
        units: list[dict[str, Any]],
        rubrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        by_id: dict[str, dict[str, Any]] = {}
        usage_by_id: dict[str, dict[str, Any]] = {}
        provider_responses_by_id: dict[str, dict[str, Any]] = {}
        for _, response in self.iter_batches(units=units, rubrics=rubrics):
            for review in response["reviews"]:
                by_id[str(review["unit_id"])] = review
            for usage in response["unit_usage"]:
                usage_by_id[str(usage["unit_id"])] = usage
            for provider_response in response.get("provider_responses", []):
                provider_responses_by_id[str(provider_response["unit_id"])] = (
                    provider_response
                )
        ordered_ids = [str(unit["unit_id"]) for unit in units]
        return {
            "feedback": [by_id[unit_id] for unit_id in ordered_ids],
            "unit_usage": [usage_by_id[unit_id] for unit_id in ordered_ids],
            "provider_responses": [
                provider_responses_by_id[unit_id]
                for unit_id in ordered_ids
                if unit_id in provider_responses_by_id
            ],
        }

    def iter_batches(
        self,
        *,
        units: list[dict[str, Any]],
        rubrics: list[dict[str, Any]],
    ):
        batches = self._batches(units)
        if not batches:
            return
        executor = ThreadPoolExecutor(
            max_workers=min(self.max_concurrent_requests, len(batches))
        )
        futures = {
            executor.submit(self._review_batch_with_retries, batch, rubrics): tuple(
                str(unit["unit_id"]) for unit in batch
            )
            for batch in batches
        }
        completed = set()
        timed_out = False
        try:
            try:
                for future in as_completed(
                    futures,
                    timeout=self.batch_timeout_seconds,
                ):
                    completed.add(future)
                    yield future.result()
            except FuturesTimeoutError:
                timed_out = True
                for future, unit_ids in futures.items():
                    if future in completed:
                        continue
                    future.cancel()
                    yield _batch_timeout_result(
                        unit_ids,
                        self.batch_timeout_seconds,
                    )
        finally:
            executor.shutdown(wait=not timed_out, cancel_futures=True)

    def _review_batch_with_retries(
        self,
        batch: list[dict[str, Any]],
        rubrics: list[dict[str, Any]],
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        unit_ids = tuple(str(unit["unit_id"]) for unit in batch)
        last_error: Exception | None = None
        attempt_usage: list[dict[str, int | None] | None] = []
        for attempt in range(self.max_attempts):
            try:
                resolved_ids, response = self._review_batch(batch, rubrics)
                attempt_usage.append(response.pop("usage"))
                response["unit_usage"] = [
                    _unit_usage(unit_id, attempt_usage) for unit_id in resolved_ids
                ]
                return resolved_ids, response
            except (
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                urllib.error.URLError,
            ) as error:
                attempt_usage.append(None)
                last_error = error
                if (
                    isinstance(last_error, DashScopeReviewRequestError)
                    and not last_error.retryable
                ):
                    break
                if attempt + 1 < self.max_attempts and self.retry_delay_seconds:
                    base_delay = self.retry_delay_seconds * (2**attempt)
                    if (
                        isinstance(last_error, DashScopeReviewRequestError)
                        and last_error.status_code == 429
                    ):
                        base_delay *= 4
                    self._sleep(_retry_delay_seconds(base_delay, unit_ids))
        assert last_error is not None
        message = " ".join(str(last_error).split())[:500] or "provider review failed"
        return (
            unit_ids,
            {
                "reviews": [
                    {
                        "unit_id": unit_id,
                        "review_error": {
                            "type": type(last_error).__name__,
                            "message": message,
                        },
                    }
                    for unit_id in unit_ids
                ],
                "unit_usage": [
                    _unit_usage(unit_id, attempt_usage) for unit_id in unit_ids
                ],
            },
        )

    def _review_batch(
        self,
        batch: list[dict[str, Any]],
        rubrics: list[dict[str, Any]],
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        if self.score_only:
            request_payload = {
                "source_text": batch[0]["text"],
                "rubrics": [
                    {
                        "dimension_id": rubric["rubric_id"],
                        "criterion_and_standards": rubric["instruction"],
                        "allowed_scores": [float(label) for label in rubric["labels"]],
                    }
                    for rubric in rubrics
                ],
            }
            system_prompt = (
                "Score only the supplied source_text against every rubric. Return "
                'exactly one JSON object of the form {"scores":{"<dimension_id>":'
                "<allowed numeric score>}}. Include every dimension_id exactly once, "
                "use only its allowed_scores, and return no other fields or text."
            )
        else:
            request_payload = {
                "source_text": batch[0]["text"],
                "review_instructions": [
                    {
                        "instruction": rubric["instruction"],
                        "allowed_labels": rubric["labels"],
                    }
                    for rubric in rubrics
                ],
            }
            system_prompt = (
                "Review only the supplied source_text. Apply each review "
                "instruction. Return exactly one JSON object in this semantic "
                'format: {"semantic_judgments":[{"label":"<allowed '
                'label for the corresponding instruction>","observations":'
                '["<observation>"],"evidence_spans":["<exact span '
                'from source_text>"],"confidence":0.0}]}. The '
                "semantic_judgments array must have one item per "
                "review_instructions item in the same order as "
                "review_instructions. Empty observations and evidence_spans "
                "arrays are valid. Confidence must be from 0 to 1. Do not "
                "return protocol identifiers such as unit_id or rubric_id."
            )
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "enable_thinking": self.enable_thinking,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        request_payload,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
        }
        if self.enable_thinking:
            body["thinking_budget"] = self.thinking_budget
        with self._request_slots:
            response = self._request_json(
                url=self._url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "X-DashScope-Wait-Timeout": "30",
                },
                body=body,
                timeout_seconds=self.timeout_seconds,
            )
        usage = _response_usage(response)
        reasoning_content, answer_content = _best_effort_response_contents(response)
        try:
            reasoning_content, answer_content = _response_contents(response)
            payload = _decode_response_content(answer_content)
            if self.score_only:
                reviews = _score_only_reviews(payload, batch, rubrics)
            else:
                reviews = _canonical_reviews(payload, batch, rubrics)
                _retain_verbatim_evidence(reviews, batch)
                _validate_provider_reviews(reviews, batch, rubrics)
        except (KeyError, TypeError, ValueError) as error:
            reviews = [
                {
                    "unit_id": str(batch[0]["unit_id"]),
                    "review_error": {
                        "type": type(error).__name__,
                        "message": " ".join(str(error).split())[:500]
                        or "provider response could not be parsed",
                    },
                }
            ]
        return (
            tuple(str(unit["unit_id"]) for unit in batch),
            {
                "reviews": reviews,
                "usage": usage,
                "provider_responses": [
                    {
                        "unit_id": str(batch[0]["unit_id"]),
                        "reasoning_content": reasoning_content,
                        "content": answer_content,
                        "raw_response": response,
                    }
                ],
            },
        )

    def _batches(
        self,
        units: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        batches: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_chars = 0
        for unit in units:
            unit_chars = len(
                json.dumps(
                    unit,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            if unit_chars > self.max_request_chars:
                raise ValueError(
                    f"review unit {unit.get('unit_id')} exceeds max_request_chars"
                )
            if current and (
                len(current) >= self.max_units_per_request
                or current_chars + unit_chars > self.max_request_chars
            ):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(unit)
            current_chars += unit_chars
        if current:
            batches.append(current)
        return batches


def _rubric_job_request_body(
    *,
    model: str,
    temperature: float,
    max_completion_tokens: int,
    enable_thinking: bool,
    thinking_budget: int,
    rendered_prompt: str,
    output_schema: dict[str, Any],
    validation_feedback: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "enable_thinking": enable_thinking,
        "response_format": {"type": "json_object"},
        "messages": rubric_job_messages(
            rendered_prompt=rendered_prompt,
            output_schema=output_schema,
            validation_feedback=validation_feedback,
        ),
    }
    if enable_thinking:
        body["thinking_budget"] = thinking_budget
    return body


def _audited_provider_response(
    completed_responses: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not completed_responses:
        return None
    if len(completed_responses) == 1:
        return completed_responses[0]
    return {
        "attempts": completed_responses,
        "final_response": completed_responses[-1],
    }


def _response_usage(response: dict[str, Any]) -> dict[str, int | None] | None:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return None
    result: dict[str, int | None] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(field)
        if type(value) is not int or value < 0:
            return None
        result[field] = value
    if result["total_tokens"] != result["prompt_tokens"] + result["completion_tokens"]:
        return None
    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    cached = (
        prompt_details.get("cached_tokens")
        if isinstance(prompt_details, dict)
        else None
    )
    reasoning = (
        completion_details.get("reasoning_tokens")
        if isinstance(completion_details, dict)
        else None
    )
    result["cached_tokens"] = cached if type(cached) is int and cached >= 0 else None
    result["reasoning_tokens"] = (
        reasoning if type(reasoning) is int and reasoning >= 0 else None
    )
    return result


def _unit_usage(
    unit_id: str,
    attempts: list[dict[str, int | None] | None],
) -> dict[str, Any]:
    known = [item for item in attempts if item is not None]
    status = (
        "complete"
        if known and len(known) == len(attempts)
        else ("partial" if known else "unavailable")
    )
    result: dict[str, Any] = {
        "unit_id": unit_id,
        "attempts": len(attempts),
        "retries": max(0, len(attempts) - 1),
        "requests": len(attempts),
        "usage_status": status,
        "prompt_tokens": sum(int(item["prompt_tokens"]) for item in known)
        if known
        else None,
        "completion_tokens": sum(int(item["completion_tokens"]) for item in known)
        if known
        else None,
        "total_tokens": sum(int(item["total_tokens"]) for item in known)
        if known
        else None,
    }
    for field in ("cached_tokens", "reasoning_tokens"):
        values = [item.get(field) for item in known]
        result[field] = (
            sum(int(value) for value in values)
            if known
            and len(known) == len(attempts)
            and all(value is not None for value in values)
            else None
        )
    return result


def _token_usage(
    attempts: list[dict[str, int | None] | None],
) -> TokenUsage:
    item = _unit_usage("rubric-row", attempts)
    return TokenUsage.from_dict(
        {
            key: item[key]
            for key in (
                "usage_status",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cached_tokens",
                "reasoning_tokens",
                "attempts",
                "retries",
                "requests",
            )
        }
    )


def _batch_timeout_result(
    unit_ids: tuple[str, ...], timeout_seconds: float | None
) -> tuple[tuple[str, ...], dict[str, Any]]:
    message = f"DashScope review batch exceeded its {timeout_seconds:g}-second deadline"
    return (
        unit_ids,
        {
            "reviews": [
                {
                    "unit_id": unit_id,
                    "review_error": {
                        "type": "BatchTimeoutError",
                        "message": message,
                    },
                }
                for unit_id in unit_ids
            ],
            "unit_usage": [_unit_usage(unit_id, [None]) for unit_id in unit_ids],
        },
    )


def _response_payload(response: dict[str, Any]) -> Any:
    _, content = _response_contents(response)
    return _decode_response_content(content)


def _response_contents(response: dict[str, Any]) -> tuple[str, str]:
    choices = response.get("choices") if isinstance(response, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    if finish_reason not in (None, "stop", "length"):
        raise ValueError(f"DashScope review response finish_reason was {finish_reason}")
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    reasoning_content = (
        message.get("reasoning_content", "") if isinstance(message, dict) else ""
    )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("DashScope review response has no message content")
    if reasoning_content is None:
        reasoning_content = ""
    if not isinstance(reasoning_content, str):
        raise ValueError("DashScope review response has invalid reasoning_content")
    return reasoning_content, content


def _best_effort_response_contents(response: dict[str, Any]) -> tuple[str, str]:
    choices = response.get("choices") if isinstance(response, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else ""
    reasoning_content = (
        message.get("reasoning_content", "") if isinstance(message, dict) else ""
    )
    return (
        reasoning_content if isinstance(reasoning_content, str) else "",
        content if isinstance(content, str) else "",
    )


def _decode_response_content(content: str) -> Any:
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    return json.loads(text)


def _response_reviews(response: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _response_payload(response)
    reviews = _normalize_review_payload(payload)
    if not isinstance(reviews, list):
        raise ValueError("DashScope review response must contain reviews")
    return reviews


def _normalize_review_payload(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    reviews = payload.get("reviews")
    if isinstance(reviews, list):
        return reviews
    review = payload.get("review")
    if isinstance(review, dict):
        return [review]
    results = payload.get("results")
    if isinstance(results, list):
        return results
    if "unit_id" in payload and "rubric_results" in payload:
        return [payload]
    return None


def _score_only_reviews(
    payload: Any,
    units: list[dict[str, Any]],
    rubrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(units) != 1 or not isinstance(payload, dict):
        raise ValueError("DashScope score response must be one JSON object")
    scores = payload.get("scores")
    expected = {str(rubric["rubric_id"]): rubric for rubric in rubrics}
    if not isinstance(scores, dict) or set(scores) != set(expected):
        raise ValueError("DashScope score response dimensions do not match rubrics")
    results = []
    for dimension_id, rubric in expected.items():
        value = scores[dimension_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("DashScope dimension score must be numeric")
        labels = [str(label) for label in rubric["labels"]]
        label = next(
            (candidate for candidate in labels if float(candidate) == float(value)),
            None,
        )
        if label is None:
            raise ValueError("DashScope dimension score is not an allowed value")
        results.append({"rubric_id": dimension_id, "label": label})
    return [{"unit_id": str(units[0]["unit_id"]), "rubric_results": results}]


def _canonical_reviews(
    payload: Any,
    units: list[dict[str, Any]],
    rubrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(units) != 1:
        raise ValueError("DashScope semantic review requires exactly one unit")
    expected_ids = [str(rubric["rubric_id"]) for rubric in rubrics]
    semantic_results = _find_semantic_results(payload, frozenset(expected_ids))
    if semantic_results is None or len(semantic_results) != len(rubrics):
        raise ValueError(
            "DashScope review response returned invalid semantic judgments"
        )
    results_by_id = {
        result.get("rubric_id"): result
        for result in semantic_results
        if isinstance(result, dict) and isinstance(result.get("rubric_id"), str)
    }
    if len(results_by_id) == len(expected_ids) and all(
        rubric_id in results_by_id for rubric_id in expected_ids
    ):
        ordered_results = [results_by_id[rubric_id] for rubric_id in expected_ids]
    else:
        ordered_results = semantic_results
    canonical_results = []
    for rubric, result in zip(rubrics, ordered_results, strict=True):
        if not isinstance(result, dict):
            raise ValueError("DashScope semantic judgment must be an object")
        canonical_results.append(
            {
                "rubric_id": str(rubric["rubric_id"]),
                "label": result.get("label"),
                "observations": _normalize_text_list(result.get("observations")),
                "evidence_spans": _normalize_text_list(
                    result.get(
                        "evidence_spans",
                        result.get("evidence", result.get("verbatim_evidence_spans")),
                    )
                ),
                "confidence": result.get("confidence"),
            }
        )
    return [
        {
            "unit_id": str(units[0]["unit_id"]),
            "rubric_results": canonical_results,
        }
    ]


def _find_semantic_results(
    payload: Any,
    expected_rubric_ids: frozenset[str] = frozenset(),
) -> list[dict[str, Any]] | None:
    if isinstance(payload, dict):
        rubric_results = payload.get("rubric_results")
        if isinstance(rubric_results, list):
            return rubric_results
        if _is_semantic_result(payload):
            return [payload]
        collected = []
        for key, value in payload.items():
            results = _find_semantic_results(value, expected_rubric_ids)
            if results is None:
                continue
            if (
                key in expected_rubric_ids
                and len(results) == 1
                and "rubric_id" not in results[0]
            ):
                results = [{**results[0], "rubric_id": key}]
            collected.extend(results)
        return collected or None
    elif isinstance(payload, list):
        if payload and all(_is_semantic_result(item) for item in payload):
            return payload
        collected = []
        for value in payload:
            results = _find_semantic_results(value, expected_rubric_ids)
            if results is not None:
                collected.extend(results)
        return collected or None
    return None


def _is_semantic_result(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and all(field in value for field in ("label", "observations", "confidence"))
        and any(
            field in value
            for field in ("evidence_spans", "evidence", "verbatim_evidence_spans")
        )
    )


def _normalize_text_list(value: Any) -> Any:
    if isinstance(value, str) and value.strip():
        return [value]
    return value


def _validate_provider_reviews(
    reviews: list[dict[str, Any]],
    units: list[dict[str, Any]],
    rubrics: list[dict[str, Any]],
) -> None:
    expected_units = {str(unit["unit_id"]): unit for unit in units}
    expected_rubrics = {
        str(rubric["rubric_id"]): tuple(str(label) for label in rubric["labels"])
        for rubric in rubrics
    }
    if len(reviews) != len(expected_units):
        raise ValueError("DashScope review response returned the wrong review count")
    seen_units: set[str] = set()
    for review in reviews:
        if not isinstance(review, dict):
            raise ValueError("DashScope review response review must be an object")
        unit_id = review.get("unit_id")
        if unit_id not in expected_units or unit_id in seen_units:
            raise ValueError("DashScope review response returned an invalid unit_id")
        seen_units.add(unit_id)
        results = review.get("rubric_results")
        if not isinstance(results, list) or len(results) != len(expected_rubrics):
            raise ValueError(
                "DashScope review response returned invalid rubric_results"
            )
        seen_rubrics: set[str] = set()
        for result in results:
            if not isinstance(result, dict):
                raise ValueError("DashScope rubric result must be an object")
            rubric_id = result.get("rubric_id")
            if rubric_id not in expected_rubrics or rubric_id in seen_rubrics:
                raise ValueError(
                    "DashScope review response returned an invalid rubric_id"
                )
            seen_rubrics.add(rubric_id)
            if result.get("label") not in expected_rubrics[rubric_id]:
                raise ValueError("DashScope review response returned an invalid label")
            for field in ("observations", "evidence_spans"):
                value = result.get(field)
                if not isinstance(value, list) or any(
                    not isinstance(item, str) or not item.strip() for item in value
                ):
                    raise ValueError(
                        f"DashScope review response returned invalid {field}"
                    )
                if field == "evidence_spans" and any(
                    not _is_verbatim_evidence(item, expected_units[unit_id])
                    for item in value
                ):
                    raise ValueError(
                        "DashScope review response returned non-verbatim evidence_spans"
                    )
            confidence = result.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
            ):
                raise ValueError(
                    "DashScope review response returned invalid confidence"
                )


def _retain_verbatim_evidence(
    reviews: list[dict[str, Any]],
    units: list[dict[str, Any]],
) -> None:
    units_by_id = {str(unit["unit_id"]): unit for unit in units}
    for review in reviews:
        unit = units_by_id.get(str(review.get("unit_id")))
        if unit is None:
            continue
        for result in review.get("rubric_results", []):
            evidence = result.get("evidence_spans")
            if not isinstance(evidence, list):
                continue
            verbatim = [
                item
                for item in evidence
                if isinstance(item, str) and _is_verbatim_evidence(item, unit)
            ]
            result["evidence_spans"] = verbatim


def _retry_delay_seconds(base_delay: float, unit_ids: tuple[str, ...]) -> float:
    digest = hashlib.sha256("\0".join(unit_ids).encode()).digest()
    jitter = int.from_bytes(digest[:8], "big") / (1 << 64)
    return base_delay * (1 + jitter)


def _is_verbatim_evidence(evidence: str, unit: dict[str, Any]) -> bool:
    text = unit.get("text")
    if not isinstance(text, str):
        return False
    if evidence in text:
        return True
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return any(evidence in value for value in _text_values(payload))


def _text_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _text_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _text_values(item)


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"project .env is unavailable: {path}")
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _positive_integer(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _finite_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _nonnegative_finite_float(value: str, name: str) -> float:
    parsed = _finite_float(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return parsed


def _chat_completions_url(workspace_id_or_base_url: str) -> str:
    value = workspace_id_or_base_url.strip().rstrip("/")
    if "://" not in value:
        if any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            for character in value
        ):
            raise ValueError("DashScope workspace ID contains invalid characters")
        value = f"https://{value}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
        or not (
            parsed.hostname == "dashscope.aliyuncs.com"
            or parsed.hostname.endswith(".maas.aliyuncs.com")
        )
    ):
        raise ValueError("DashScope base URL must be an official HTTPS endpoint")
    if parsed.path.rstrip("/").endswith("/chat/completions"):
        return value
    return f"{value}/chat/completions"


def _request_json_without_proxy(
    *,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers=headers,
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1000]
        error_code = _provider_error_code(detail)
        retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
        if error_code == "insufficient_quota":
            retryable = False
        raise DashScopeReviewRequestError(
            exc.code,
            detail,
            retryable=retryable,
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("DashScope review HTTP response must be an object")
    return payload


def _provider_error_code(detail: str) -> str | None:
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None
