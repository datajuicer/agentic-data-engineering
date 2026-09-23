#!/usr/bin/env python3
"""Run the focused production Local Judge deployment probe."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import threading
import time
import uuid

import yaml

from ade.harness.environment import load_project_environment
from ade.local_rubric_judge.client import LocalRubricJudgeClient
from ade.local_rubric_judge.http_gateway import HttpRubricJobGateway
from ade.local_rubric_judge.launcher import ProductionJudgeLauncher
from ade.local_rubric_judge.lifecycle import LocalJudgeBinding, RunResourceAdmission
from ade.rubric_jobs import RubricInputRow, encode_input_jsonl, join_results


def _score_row(
    record_id: str,
    *,
    dimension: str,
    values: list[float],
    question: str,
    response: str,
    run_id: str,
) -> RubricInputRow:
    return RubricInputRow.from_dict(
        {
            "record_id": record_id,
            "rubric": {
                "template": (
                    f"Score only the response's {dimension}. Return only the required JSON. "
                    "Question: {{question}}\nResponse: {{response}}"
                ),
                "required_variables": ["question", "response"],
                "output_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "scores_by_dimension": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {dimension: {"enum": values}},
                            "required": [dimension],
                        }
                    },
                    "required": ["scores_by_dimension"],
                },
            },
            "inputs": {"question": question, "response": response},
            "metadata": {"run_id": run_id, "purpose": "training"},
        }
    )


async def _execute_job(
    client: LocalRubricJudgeClient,
    *,
    run_id: str,
    name: str,
    rows: tuple[RubricInputRow, ...],
    model_digest: str,
) -> dict[str, object]:
    started = time.time()
    submitted = await client.submit(
        submission_id=f"{run_id}:{name}",
        input_jsonl=encode_input_jsonl(rows),
        job_metadata={"run_id": run_id, "probe": name},
    )
    result = await client.wait(submitted.job_id)
    joined = join_results(rows, result.rows)
    if result.status.state != "completed" or result.status.error_rows != 0:
        errors = [row.error.to_dict() for row in joined if row.error is not None]
        raise RuntimeError(
            f"{name} failed: state={result.status.state} errors={errors[:3]}"
        )
    expected_model = {"id": model_digest, "digest": model_digest}
    if any(row.status != "ok" or row.model != expected_model for row in joined):
        raise RuntimeError(f"{name} returned invalid row or model identity")
    if result.usage.usage_status != "complete" or result.usage.requests != len(rows):
        raise RuntimeError(f"{name} returned incomplete usage")
    return {
        "job_id": result.status.job_id,
        "queue_sequence": result.status.queue_sequence,
        "state": result.status.state,
        "total_rows": result.status.total_rows,
        "completed_rows": result.status.completed_rows,
        "error_rows": result.status.error_rows,
        "elapsed_seconds": time.time() - started,
        "usage": result.usage.to_dict(),
        "sample_results": [row.result for row in joined[:3]],
        "model": expected_model,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-step-rows", type=int, default=64)
    parser.add_argument("--run-id")
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--retain-for-followup", action="store_true")
    lifecycle.add_argument("--release-run-id")
    args = parser.parse_args()
    if args.seed < 0 or args.full_step_rows < 1:
        raise ValueError("probe seed and full-step rows must be valid")
    if args.retain_for_followup and not args.run_id:
        parser.error("--retain-for-followup requires an explicit --run-id")

    project = Path(args.project_root).resolve()
    deployment_path = Path(args.deployment).resolve()
    load_project_environment(project)
    resources = yaml.safe_load(deployment_path.read_text(encoding="utf-8"))["run_resources"]
    ray_config = resources["ray_cluster"]
    judge = resources["local_judge"]
    generation = {**judge["generation"], "seed": args.seed}
    authorization = os.environ.get(str(judge["authorization_env"]))
    if not authorization:
        raise RuntimeError("configured Local Judge authorization is unavailable")
    binding = LocalJudgeBinding.from_dict(
        {
            "cluster_id": ray_config["cluster_id"],
            "ray_address": ray_config["address"],
            "gpu_count": judge["gpu_count"],
            "model_path": str((project / judge["model_path"]).resolve()),
            "model_digest": judge["model_digest"],
            "gateway_port": judge["gateway_port"],
            "protocol": judge["protocol"],
        }
    )
    run_id = args.run_id or (
        f"local-judge-preflight-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    run_root = project / "runs" / run_id
    deployment_root = project / "runs" / "deployments" / str(ray_config["cluster_id"])
    receipt_path = run_root / "receipts" / "local-judge-probe.json"
    launcher = ProductionJudgeLauncher(
        deployment_root / "local-judge",
        authorization_env=str(judge["authorization_env"]),
        readiness_seconds=float(judge["timeout_policy"]["readiness_seconds"]),
        request_timeout_seconds=float(judge["timeout_policy"]["request_timeout_seconds"]),
        vllm=dict(judge["vllm"]),
        generation=generation,
    )
    admission = RunResourceAdmission(deployment_root / "run-services", launcher)
    if args.release_run_id:
        attached_path = admission.state_root / f"{args.release_run_id}.json"
        if not attached_path.is_file():
            raise RuntimeError(
                f"Local Judge owner attachment does not exist: {args.release_run_id}"
            )
        attached = json.loads(attached_path.read_text(encoding="utf-8"))
        handle = attached.get("service_handle")
        if not isinstance(handle, dict):
            raise RuntimeError("Local Judge owner attachment lacks its exact handle")
        first = admission.terminal(args.release_run_id)
        second = admission.terminal(args.release_run_id)
        if first is None or second is None or first.launch_id != second.launch_id:
            raise RuntimeError("Local Judge idempotent owner release failed")
        health = launcher.health(handle)
        if health.get("ready") or not launcher.released(handle):
            raise RuntimeError("Local Judge owner release did not stop its exact launch")
        release_receipt = {
            "schema_version": 1,
            "operation": "local-judge-owner-release",
            "run_id": args.release_run_id,
            "deployment": str(deployment_path),
            "launch_id": first.launch_id,
            "status": "released",
            "service_health_after_stop": health,
            "launch_root_removed": not Path(handle["log_root"]).exists(),
            "finished_at": time.time(),
        }
        release_path = (
            project
            / "runs"
            / args.release_run_id
            / "receipts"
            / "local-judge-release.json"
        )
        release_path.parent.mkdir(parents=True, exist_ok=True)
        release_path.write_text(
            json.dumps(release_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "released", "receipt": str(release_path)}))
        return 0
    receipt: dict[str, object] = {
        "schema_version": 1,
        "probe": "deployment-owned-local-judge",
        "run_id": run_id,
        "deployment": str(deployment_path),
        "seed": args.seed,
        "full_step_rows": args.full_step_rows,
        "vllm": dict(judge["vllm"]),
        "generation": generation,
        "started_at": time.time(),
    }
    stop_monitor = threading.Event()

    def monitor() -> None:
        while not stop_monitor.wait(30):
            # Judge execution belongs to a Ray-selected node, not this host.
            print(
                json.dumps({"phase": "startup_or_probe", "run_id": run_id}),
                flush=True,
            )

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    ready = None
    try:
        print(json.dumps({"phase": "starting", "run_id": run_id}), flush=True)
        ready = admission.admit(run_id=run_id, binding=binding)
        state = json.loads(Path(ready.state_path).read_text(encoding="utf-8"))
        handle = state["service_handle"]
        receipt["launch"] = {
            "launch_id": ready.launch_id,
            "health": launcher.health(handle),
            "pids": handle["pids"],
            "host": ready.host,
            "node_id": ready.node_id,
            "gateway_url": ready.gateway_url,
            "ports": handle["vllm_ports"],
            "log_root": handle["log_root"],
        }
        print(json.dumps({"phase": "ready", "launch_id": ready.launch_id}), flush=True)
        client = LocalRubricJudgeClient(
            HttpRubricJobGateway(
                ready.gateway_url,
                authorization=authorization,
                timeout_seconds=float(judge["timeout_policy"]["request_timeout_seconds"]),
            ),
            poll_interval_seconds=0.1,
        )
        single = _score_row(
            "single-00000000",
            dimension="logic",
            values=[0.0, 1.0],
            question="What is 2 + 2?",
            response="4",
            run_id=run_id,
        )
        receipt["single_row"] = asyncio.run(
            _execute_job(
                client,
                run_id=run_id,
                name="single-row",
                rows=(single,),
                model_digest=binding.model_digest,
            )
        )
        print(json.dumps({"phase": "single_row_passed"}), flush=True)
        mixed = (
            _score_row(
                "mixed-00000000",
                dimension="logic",
                values=[0.0, 1.0],
                question="What is 3 + 5?",
                response="8",
                run_id=run_id,
            ),
            _score_row(
                "mixed-00000001",
                dimension="completion",
                values=[0.0, 0.5, 1.0],
                question="What is 7 - 2?",
                response="7 - 2 = 5.",
                run_id=run_id,
            ),
        )
        receipt["mixed_score_schemas"] = asyncio.run(
            _execute_job(
                client,
                run_id=run_id,
                name="mixed-score-schemas",
                rows=mixed,
                model_digest=binding.model_digest,
            )
        )
        print(json.dumps({"phase": "mixed_score_schemas_passed"}), flush=True)
        full_rows = tuple(
            _score_row(
                f"full-step-{index:08d}",
                dimension="logic",
                values=[0.0, 1.0],
                question=f"What is {index} + 1?",
                response=str(index + 1),
                run_id=run_id,
            )
            for index in range(args.full_step_rows)
        )
        receipt["full_step"] = asyncio.run(
            _execute_job(
                client,
                run_id=run_id,
                name=f"full-step-{args.full_step_rows}",
                rows=full_rows,
                model_digest=binding.model_digest,
            )
        )
        receipt["status"] = "passed"
        print(json.dumps({"phase": "full_step_passed", "rows": args.full_step_rows}), flush=True)
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            if ready is not None:
                if args.retain_for_followup and receipt.get("status") == "passed":
                    service_health = launcher.health(handle)
                    if not service_health.get("ready") or launcher.released(handle):
                        raise RuntimeError(
                            "Local Judge could not be retained for the same owner"
                        )
                    receipt["retention"] = {
                        "status": "attached",
                        "launch_id": ready.launch_id,
                        "state_path": ready.state_path,
                        "service_health": service_health,
                    }
                else:
                    first = admission.terminal(run_id)
                    second = admission.terminal(run_id)
                    if (
                        first is None
                        or second is None
                        or first.launch_id != second.launch_id
                    ):
                        raise RuntimeError("Local Judge idempotent Run detach failed")
                    service_health = launcher.health(handle)
                    if service_health.get("ready") or not launcher.released(handle):
                        raise RuntimeError(
                            "Local Judge was not fully stopped after Run detach"
                        )
                    receipt["detach"] = {
                        "first_status": first.status,
                        "second_status": second.status,
                        "launch_id": first.launch_id,
                        "released": True,
                        "service_health_after_stop": service_health,
                        "launch_root_removed": not Path(handle["log_root"]).exists(),
                    }
        finally:
            stop_monitor.set()
            monitor_thread.join(timeout=5)
            receipt["finished_at"] = time.time()
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "phase": "finished",
                        "status": receipt["status"],
                        "receipt": str(receipt_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
