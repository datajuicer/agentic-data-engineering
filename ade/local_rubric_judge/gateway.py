"""HTTP job gateway for a Run-owned Local Rubric Judge service."""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any

from ade.local_rubric_judge.service import LocalRubricJudgeService


def build_gateway(
    address: tuple[str, int],
    *,
    service: LocalRubricJudgeService,
    authorization: str,
    launch_id: str,
    protocol: str,
    model_digest: str,
) -> ThreadingHTTPServer:
    if not authorization or not launch_id or not protocol or not model_digest:
        raise ValueError("Local Judge gateway identity is required")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def _handle(self, method: str) -> None:
            if self.headers.get("Authorization") != authorization:
                self._send(401, {"error": "unauthorized"})
                return
            if not service.healthy:
                self._send(
                    503,
                    {
                        "error": "local_judge_unhealthy",
                        "message": service.unhealthy_reason,
                    },
                )
                return
            parts = self.path.strip("/").split("/")
            try:
                if method == "GET" and parts == ["v1"]:
                    self._send(
                        200,
                        {
                            "launch_id": launch_id,
                            "protocol": protocol,
                            "model_digest": model_digest,
                        },
                    )
                    return
                if method == "POST" and parts == ["v1", "jobs"]:
                    payload = self._payload()
                    if payload.get("protocol") != "ade.rubric_jobs.v1":
                        raise ValueError("Rubric Job protocol identity mismatch")
                    status = service.submit(
                        submission_id=str(payload["submission_id"]),
                        input_jsonl=base64.b64decode(payload["input_jsonl_base64"], validate=True),
                        job_metadata=payload["job_metadata"],
                    )
                    self._send(200, status.to_dict())
                    return
                if method == "POST" and parts == ["v1", "jobs", "cancel-all"]:
                    self._send(
                        200,
                        {
                            "statuses": [
                                item.to_dict() for item in service.cancel_all()
                            ]
                        },
                    )
                    return
                if len(parts) >= 3 and parts[:2] == ["v1", "jobs"]:
                    job_id = parts[2]
                    if method == "GET" and len(parts) == 3:
                        self._send(200, service.get_status(job_id).to_dict())
                        return
                    if method == "GET" and parts[3:] == ["result"]:
                        result = service.get_result(job_id)
                        self._send(
                            200,
                            {
                                "status": result.status.to_dict(),
                                "rows": [row.to_dict() for row in result.rows],
                                "usage": result.usage.to_dict(),
                            },
                        )
                        return
                    if method == "POST" and parts[3:] == ["cancel"]:
                        self._send(200, service.cancel(job_id).to_dict())
                        return
                    if method == "POST" and parts[3:] == ["expire"]:
                        self._send(200, service.expire(job_id).to_dict())
                        return
                self._send(404, {"error": "not_found"})
            except (KeyError, TypeError, ValueError) as error:
                self._send(400, {"error": type(error).__name__, "message": str(error)})

        def _payload(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("request body must be an object")
            return value

        def _send(self, status: int, value: object) -> None:
            content = json.dumps(value, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer(address, Handler)
