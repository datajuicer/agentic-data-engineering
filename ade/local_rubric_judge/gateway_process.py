"""Process entrypoint for the Run-owned Local Judge gateway and FIFO worker."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import time

from ade.local_rubric_judge.gateway import build_gateway
from ade.local_rubric_judge.service import LocalRubricJudgeService
from ade.local_rubric_judge.vllm_endpoint import VllmEndpointPool


logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--authorization-env", required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--request-timeout-seconds", required=True, type=float)
    parser.add_argument("--per-endpoint-concurrency", required=True, type=int)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--top-p", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()
    authorization = os.environ.get(args.authorization_env)
    if not authorization:
        raise ValueError("Local Judge authorization environment is missing")
    endpoint = VllmEndpointPool(
        tuple(args.endpoint),
        model=args.model,
        model_digest=args.model_digest,
        timeout_seconds=args.request_timeout_seconds,
        per_endpoint_concurrency=args.per_endpoint_concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
        seed=args.seed,
    )
    service = LocalRubricJudgeService(
        args.state,
        endpoint,
        max_concurrency=len(args.endpoint) * args.per_endpoint_concurrency,
    )
    stop = threading.Event()

    def work() -> None:
        try:
            while not stop.is_set():
                if asyncio.run(service.run_next()) is None:
                    stop.wait(0.1)
        except Exception as error:
            service.mark_unhealthy(error)
            logger.exception("Local Judge worker stopped after an unrecoverable error")

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    server = build_gateway(
        (args.host, args.port),
        service=service,
        authorization=authorization,
        launch_id=args.launch_id,
        protocol=args.protocol,
        model_digest=args.model_digest,
    )
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()
        worker.join(timeout=5)
        endpoint.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
