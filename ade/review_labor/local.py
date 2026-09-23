"""Deployment-local vLLM transport for Analyzer Review jobs."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import json
import math
from pathlib import Path
import threading
from typing import Any, Callable, Mapping
import urllib.error
import urllib.request

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from ade.rubric_jobs import JobError, RubricInputRow, TokenUsage
from ade.review_labor.jobs import ReviewResponse
from ade.review_labor.structured_review import (
    expand_compact_review_result,
    review_schema_specs,
    rubric_job_messages,
)


class LocalAnalyzerRequestError(RuntimeError):
    def __init__(self, status_code: int, detail: str, *, retryable: bool) -> None:
        super().__init__(
            f"Local Analyzer request failed with HTTP {status_code}: {detail}"
        )
        self.status_code = status_code
        self.retryable = retryable


class LocalAnalyzerContextExceeded(RuntimeError):
    """The rendered review request cannot fit the configured model context."""


class LocalAnalyzerTransport:
    """Run Analyzer Review against the Run-owned vLLM endpoints.

    This transport implements the existing ReviewEndpoint contract. It does not
    use the process-reward gateway, queue, client, or fallback policy.
    """

    def __init__(
        self,
        *,
        endpoint_urls: tuple[str, ...],
        model: str,
        model_digest: str,
        max_concurrent_requests: int = 128,
        timeout_seconds: int = 300,
        batch_timeout_seconds: float | None = None,
        temperature: float = 0.0,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.5,
        max_completion_tokens: int = 4096,
        thinking_budget: int = 2048,
        enable_thinking: bool = True,
        request_json: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        if not endpoint_urls or any(not value.strip() for value in endpoint_urls):
            raise ValueError("Local Analyzer requires at least one endpoint URL")
        if not model.strip() or not model_digest.strip():
            raise ValueError("Local Analyzer model identity is required")
        if (
            max_concurrent_requests <= 0
            or timeout_seconds <= 0
            or max_attempts <= 0
            or max_completion_tokens <= 0
            or (enable_thinking and thinking_budget <= 0)
        ):
            raise ValueError("Local Analyzer limits must be positive")
        if enable_thinking and thinking_budget > max_completion_tokens:
            raise ValueError(
                "Local Analyzer thinking budget cannot exceed completion budget"
            )
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("Local Analyzer temperature must be between 0 and 2")
        if not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
            raise ValueError("Local Analyzer retry delay must be non-negative")
        if batch_timeout_seconds is not None and (
            isinstance(batch_timeout_seconds, bool)
            or not isinstance(batch_timeout_seconds, (int, float))
            or not math.isfinite(batch_timeout_seconds)
            or batch_timeout_seconds <= 0
        ):
            raise ValueError("Local Analyzer batch timeout must be positive and finite")
        self.endpoint_urls = tuple(value.rstrip("/") for value in endpoint_urls)
        self.model = model.strip()
        self.model_digest = model_digest.strip()
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout_seconds = timeout_seconds
        self.batch_timeout_seconds = batch_timeout_seconds
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.max_completion_tokens = max_completion_tokens
        self.thinking_budget = thinking_budget
        self.enable_thinking = enable_thinking
        self._request_json = request_json or _request_json_without_proxy
        self._endpoint_lock = threading.Lock()
        self._next_endpoint = 0
        self._judge_state_path: Path | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_requests,
            thread_name_prefix="local-analyzer-request",
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def evaluate(
        self, row: RubricInputRow, rendered_prompt: str
    ) -> ReviewResponse:
        attempts: list[dict[str, int | None] | None] = []
        completed_responses: list[dict[str, Any]] = []
        validation_feedback: str | None = None
        for attempt in range(self.max_attempts):
            raw_response: dict[str, Any] | None = None
            try:
                raw_response = await asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    functools.partial(
                        self._request_once,
                        row=row,
                        rendered_prompt=rendered_prompt,
                        validation_feedback=validation_feedback,
                    ),
                )
                completed_responses.append(raw_response)
                attempts.append(_response_usage(raw_response))
                _reasoning, content = _response_contents(raw_response)
                result = _decode_response_content(content)
                if not isinstance(result, dict):
                    raise ValueError("Local Analyzer result must be an object")
                specs = review_schema_specs(row.rubric.output_schema)
                if specs is not None:
                    result = expand_compact_review_result(result, specs)
                Draft202012Validator(row.rubric.output_schema).validate(result)
                return ReviewResponse(
                    result,
                    None,
                    _token_usage(attempts),
                    self.model,
                    self.model_digest,
                    _audited_response(completed_responses),
                )
            except LocalAnalyzerContextExceeded as error:
                return ReviewResponse(
                    None,
                    JobError("context_exceeded", str(error), False),
                    _unavailable_usage(),
                    self.model,
                    self.model_digest,
                    None,
                )
            except (KeyError, TypeError, ValueError, ValidationError) as error:
                if raw_response is not None:
                    validation_feedback = " ".join(str(error).split())[:300]
                    if _has_readable_response(raw_response):
                        return ReviewResponse(
                            None,
                            None,
                            _token_usage(attempts),
                            self.model,
                            self.model_digest,
                            _audited_response(completed_responses),
                            JobError(
                                "schema_invalid",
                                validation_feedback,
                                False,
                            ),
                        )
                    if attempt + 1 == self.max_attempts:
                        return ReviewResponse(
                            None,
                            JobError("schema_invalid", validation_feedback, False),
                            _token_usage(attempts),
                            self.model,
                            self.model_digest,
                            _audited_response(completed_responses),
                        )
                else:
                    attempts.append(None)
                    if attempt + 1 == self.max_attempts:
                        return ReviewResponse(
                            None,
                            JobError("endpoint_error", str(error), False),
                            _token_usage(attempts),
                            self.model,
                            self.model_digest,
                            None,
                        )
            except Exception as error:
                attempts.append(None)
                retryable = not isinstance(error, LocalAnalyzerRequestError) or error.retryable
                if not retryable or attempt + 1 == self.max_attempts:
                    return ReviewResponse(
                        None,
                        JobError("endpoint_error", str(error), False),
                        _token_usage(attempts),
                        self.model,
                        self.model_digest,
                        _audited_response(completed_responses),
                    )
            if self.retry_delay_seconds:
                await asyncio.sleep(self.retry_delay_seconds)
        raise AssertionError("Local Analyzer retry loop did not return")

    def _request_once(
        self,
        *,
        row: RubricInputRow,
        rendered_prompt: str,
        validation_feedback: str | None,
    ) -> dict[str, Any]:
        """Build and submit one request on a worker thread.

        Like the RFT reward path, the client sends text and a fixed generation
        budget; vLLM owns tokenization and context checks.
        """
        return self._request_json(
            url=self._take_endpoint(),
            headers={"Content-Type": "application/json"},
            body=self._request_body(
                row=row,
                rendered_prompt=rendered_prompt,
                validation_feedback=validation_feedback,
            ),
            timeout_seconds=self.timeout_seconds,
        )

    @classmethod
    def from_config(
        cls,
        *,
        provider: Mapping[str, object],
        local_judge: Mapping[str, object],
        request_json: Callable[..., dict[str, Any]] | None = None,
    ) -> "LocalAnalyzerTransport":
        if provider.get("type") != "local_analyzer":
            raise ValueError("Review provider must be local_analyzer")
        if local_judge.get("enabled") is not True:
            raise ValueError("Local Analyzer requires an enabled Local Judge resource")
        gpu_count = local_judge.get("gpu_count")
        if type(gpu_count) is not int or gpu_count < 1:
            raise ValueError("Local Analyzer requires Local Judge GPU endpoints")
        model_path = local_judge.get("model_path")
        model_digest = local_judge.get("model_digest")
        if not isinstance(model_path, str) or not isinstance(model_digest, str):
            raise ValueError("Local Analyzer requires Local Judge model identity")
        transport = cls(
            endpoint_urls=tuple(
                f"http://{local_judge['host']}:{8901 + index}/v1/chat/completions"
                for index in range(gpu_count)
            ),
            model=model_digest,
            model_digest=model_digest,
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

        if local_judge.get("state_path"):
            transport._judge_state_path = Path(str(local_judge["state_path"]))
        return transport

    def _take_endpoint(self) -> str:
        with self._endpoint_lock:
            if self._judge_state_path is not None:
                handle = json.loads(self._judge_state_path.read_text())["service_handle"]
                self.endpoint_urls = tuple(
                    f"http://{handle['host']}:{port}/v1/chat/completions"
                    for port in handle["vllm_ports"]
                )
            endpoint = self.endpoint_urls[self._next_endpoint]
            self._next_endpoint = (self._next_endpoint + 1) % len(self.endpoint_urls)
        return endpoint

    def _request_body(
        self,
        *,
        row: RubricInputRow,
        rendered_prompt: str,
        validation_feedback: str | None,
    ) -> dict[str, Any]:
        messages = rubric_job_messages(
            rendered_prompt=rendered_prompt,
            output_schema=row.rubric.output_schema,
            validation_feedback=validation_feedback,
        )
        max_completion_tokens = self.max_completion_tokens
        thinking_budget = min(self.thinking_budget, max_completion_tokens)
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_completion_tokens": max_completion_tokens,
            "enable_thinking": self.enable_thinking,
            "chat_template_kwargs": {
                "enable_thinking": self.enable_thinking,
                **(
                    {"thinking_budget": thinking_budget}
                    if self.enable_thinking
                    else {}
                ),
            },
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        if self.enable_thinking:
            body["thinking_budget"] = thinking_budget
        return body


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
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:1000]
        if _is_context_exceeded_detail(detail):
            raise LocalAnalyzerContextExceeded(detail) from error
        raise LocalAnalyzerRequestError(
            error.code,
            detail,
            retryable=error.code in {408, 409, 425, 429, 500, 502, 503, 504},
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Local Analyzer HTTP response must be an object")
    return payload


def _is_context_exceeded_detail(detail: str) -> bool:
    normalized = " ".join(detail.lower().split())
    return (
        ("context" in normalized and ("length" in normalized or "limit" in normalized))
        or "maximum context" in normalized
        or "prompt is too long" in normalized
        or "prompt too long" in normalized
    )


def _response_contents(response: dict[str, Any]) -> tuple[str, str]:
    choices = response.get("choices") if isinstance(response, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    if finish_reason not in (None, "stop", "length"):
        raise ValueError(f"Local Analyzer finish_reason was {finish_reason}")
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    reasoning = ""
    if isinstance(message, dict):
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Local Analyzer response has no message content")
    if not isinstance(reasoning, str):
        raise ValueError("Local Analyzer response has invalid reasoning content")
    return reasoning, content


def _decode_response_content(content: str) -> Any:
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    return json.loads(text)


def _has_readable_response(response: dict[str, Any]) -> bool:
    """Return whether a schema-invalid response still contains review prose."""

    choices = response.get("choices") if isinstance(response, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        return False
    values = (
        message.get("reasoning_content"),
        message.get("reasoning"),
        message.get("content"),
    )
    text = "\n".join(value for value in values if isinstance(value, str))
    return any(character.isalnum() for character in text)


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
    cached = prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
    reasoning = (
        completion_details.get("reasoning_tokens")
        if isinstance(completion_details, dict)
        else None
    )
    result["cached_tokens"] = cached if type(cached) is int and cached >= 0 else None
    result["reasoning_tokens"] = reasoning if type(reasoning) is int and reasoning >= 0 else None
    return result


def _token_usage(attempts: list[dict[str, int | None] | None]) -> TokenUsage:
    known = [item for item in attempts if item is not None]
    status = (
        "complete"
        if known and len(known) == len(attempts)
        else ("partial" if known else "unavailable")
    )
    values: dict[str, Any] = {
        "usage_status": status,
        "attempts": len(attempts),
        "retries": max(0, len(attempts) - 1),
        "requests": len(attempts),
    }
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values[field] = sum(int(item[field]) for item in known) if known else None
    for field in ("cached_tokens", "reasoning_tokens"):
        field_values = [item.get(field) for item in known]
        values[field] = (
            sum(int(value) for value in field_values)
            if known
            and len(known) == len(attempts)
            and all(value is not None for value in field_values)
            else None
        )
    return TokenUsage.from_dict(values)


def _unavailable_usage() -> TokenUsage:
    return TokenUsage.from_dict(
        {
            "usage_status": "unavailable",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cached_tokens": None,
            "reasoning_tokens": None,
            "attempts": 0,
            "retries": 0,
            "requests": 0,
        }
    )


def _audited_response(
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
