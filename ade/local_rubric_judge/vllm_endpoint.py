"""OpenAI-compatible endpoint pool for the Run-owned Local Judge."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import urllib.request

from ade.local_rubric_judge.service import ModelResponse
from ade.rubric_jobs import JobError, RubricInputRow, TokenUsage


class VllmEndpointPool:
    def __init__(
        self,
        base_urls: tuple[str, ...],
        *,
        model: str,
        model_digest: str,
        timeout_seconds: float,
        per_endpoint_concurrency: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
        seed: int,
    ) -> None:
        if not base_urls or timeout_seconds <= 0 or per_endpoint_concurrency < 1:
            raise ValueError("Local Judge endpoints and timeout are required")
        self.base_urls = tuple(item.rstrip("/") for item in base_urls)
        self.model = model
        self.model_digest = model_digest
        self.timeout_seconds = float(timeout_seconds)
        self.per_endpoint_concurrency = int(per_endpoint_concurrency)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.enable_thinking = bool(enable_thinking)
        self.seed = int(seed)
        self._next = 0
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=len(self.base_urls) * self.per_endpoint_concurrency,
            thread_name_prefix="local-judge-request",
        )
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def evaluate(
        self,
        row: RubricInputRow,
        rendered_prompt: str,
    ) -> ModelResponse:
        with self._lock:
            base_url = self.base_urls[self._next % len(self.base_urls)]
            self._next += 1
        return await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self._evaluate_sync,
            base_url,
            row,
            rendered_prompt,
        )

    def _evaluate_sync(
        self,
        base_url: str,
        row: RubricInputRow,
        rendered_prompt: str,
    ) -> ModelResponse:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": rendered_prompt}],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "rubric_result",
                        "strict": True,
                        "schema": row.rubric.output_schema,
                    },
                },
            }
        ).encode()
        request = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw = None
        try:
            with self._opener.open(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                raw = json.loads(response.read())
            content = raw["choices"][0]["message"]["content"]
            result = json.loads(content)
            usage = raw.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            return ModelResponse(
                result,
                None,
                TokenUsage.from_dict(
                    {
                        "usage_status": "complete",
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "cached_tokens": None,
                        "reasoning_tokens": None,
                        "attempts": 1,
                        "retries": 0,
                        "requests": 1,
                    }
                ),
                self.model,
                self.model_digest,
            )
        except Exception as error:
            return ModelResponse(
                None,
                JobError("endpoint_error", str(error), True),
                TokenUsage.from_dict(
                    {
                        "usage_status": "unavailable",
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "cached_tokens": None,
                        "reasoning_tokens": None,
                        "attempts": 1,
                        "retries": 0,
                        "requests": 1,
                    }
                ),
                self.model,
                self.model_digest,
            )
