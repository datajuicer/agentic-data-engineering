#!/usr/bin/env python3
"""Reconcile final ADE Run resources and print the durable cleanup receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ade.harness.environment import load_project_environment
from ade.harness.processes import run_resource_admission_from_resources
from ade.harness.run_cleanup import cleanup_cancelled_run
from ade.memory.repository import FileRunRepository


FINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile workers, Ray resources, staging, and Judge for a final Run."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.resolve()
    control_root = args.control_root.resolve()
    queue_root = args.queue_root.resolve()
    load_project_environment(project_root)
    state = FileRunRepository(control_root).load(args.run_id)
    if state.status.value not in FINAL_RUN_STATUSES:
        raise ValueError(
            "terminal reconciliation requires completed, failed, or cancelled; "
            f"got {state.status.value}"
        )
    resources = state.run_resources if isinstance(state.run_resources, dict) else {}
    judge_result = None
    if resources.get("local_judge") and resources.get("ray_cluster"):
        admission = run_resource_admission_from_resources(
            resources,
            runs_root=control_root,
            project_root=project_root,
        )
        detached = admission.terminal(args.run_id)
        if detached is not None:
            judge_result = {
                "status": detached.status,
                "launch_id": detached.launch_id,
                "state_path": detached.state_path,
            }

    ray_cluster = resources.get("ray_cluster", {})
    artifact_staging = resources.get("artifact_staging", {})
    cleanup = cleanup_cancelled_run(
        run_id=args.run_id,
        run_dir=control_root / args.run_id,
        queue_root=queue_root,
        review_root=queue_root.parent / "review",
        ray_address=(
            str(ray_cluster.get("address"))
            if isinstance(ray_cluster, dict) and ray_cluster.get("address")
            else None
        ),
        artifact_cache_dirs=(
            (str(artifact_staging["cache_dir"]),)
            if isinstance(artifact_staging, dict)
            and artifact_staging.get("enabled")
            and artifact_staging.get("cache_dir")
            else ()
        ),
    )
    result = {
        "run_id": args.run_id,
        "run_status": state.status.value,
        "cleanup": cleanup,
        "judge": judge_result,
        "receipt_path": str(
            control_root / args.run_id / "reports" / "cleanup" / "run-cleanup.json"
        ),
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if cleanup["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
