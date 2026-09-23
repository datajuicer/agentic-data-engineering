"""Single production HTTP boundary for the Run-local Rubric Job gateway."""

from __future__ import annotations

import base64
import json
from typing import Any
import urllib.request
import urllib.error

from ade.local_rubric_judge.client import JobRejected
from ade.rubric_jobs import JobResult, JobStatus, RubricOutputRow, TokenUsage


class HttpRubricJobGateway:
    def __init__(
        self,
        gateway_url: str,
        *,
        authorization: str,
        timeout_seconds: float,
    ) -> None:
        if not gateway_url.startswith(("http://", "https://")):
            raise ValueError("Local Judge gateway_url must be HTTP(S)")
        if not authorization:
            raise ValueError("Run-scoped Local Judge authorization is required")
        self.gateway_url = gateway_url.rstrip("/")
        self.authorization = authorization
        self.timeout_seconds = timeout_seconds
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def submit(
        self, *, submission_id: str, input_jsonl: bytes, job_metadata: dict[str, Any]
    ) -> JobStatus:
        value = self._request(
            "POST",
            "/v1/jobs",
            {
                "protocol": "ade.rubric_jobs.v1",
                "submission_id": submission_id,
                "input_jsonl_base64": base64.b64encode(input_jsonl).decode("ascii"),
                "job_metadata": job_metadata,
            },
        )
        return _status(value)

    def get_status(self, job_id: str) -> JobStatus:
        return _status(self._request("GET", f"/v1/jobs/{job_id}", None))

    def get_result(self, job_id: str) -> JobResult:
        value = self._request("GET", f"/v1/jobs/{job_id}/result", None)
        if not isinstance(value, dict) or set(value) != {"status", "rows", "usage"}:
            raise ValueError("Local Judge result response is invalid")
        rows = value["rows"]
        if not isinstance(rows, list):
            raise ValueError("Local Judge result rows are invalid")
        return JobResult(
            _status(value["status"]),
            tuple(RubricOutputRow.from_dict(row) for row in rows),
            TokenUsage.from_dict(value["usage"]),
        )

    def cancel(self, job_id: str) -> JobStatus:
        return _status(self._request("POST", f"/v1/jobs/{job_id}/cancel", {}))

    def expire(self, job_id: str) -> JobStatus:
        return _status(self._request("POST", f"/v1/jobs/{job_id}/expire", {}))

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None
    ) -> object:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.gateway_url + path,
            method=method,
            data=data,
            headers={
                "Authorization": self.authorization,
                "Content-Type": "application/json",
            },
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if method == "POST" and path == "/v1/jobs" and 400 <= error.code < 500:
                detail = ""
                try:
                    value = json.loads(error.read().decode("utf-8"))
                    if isinstance(value, dict):
                        detail = str(value.get("message") or value.get("error") or "").strip()
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
                suffix = f": {detail}" if detail else ""
                raise JobRejected(
                    f"Local Judge rejected submission: HTTP {error.code}{suffix}"
                ) from error
            raise


def _status(value: object) -> JobStatus:
    if not isinstance(value, dict) or value.get("protocol") != "ade.rubric_jobs.v1":
        raise ValueError("Local Judge status response is invalid")
    return JobStatus(
        str(value.get("job_id") or ""),
        str(value.get("submission_id") or ""),
        str(value.get("content_digest") or ""),
        str(value.get("state") or ""),
        value.get("queue_sequence"),
        value.get("total_rows"),
        value.get("completed_rows"),
        value.get("error_rows"),
        None,
    )
