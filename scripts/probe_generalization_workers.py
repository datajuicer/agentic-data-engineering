"""Authorized 3 x 24 GPU lease probe; no model or benchmark evaluation.

Run from the repo with PYTHONPATH=. and the shared ADE interpreter. Receipts
exercise the production allocator, per-worker cleanup and replacement on Ray.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace

import ray
import yaml
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from ade.engine.execution.coordinator_resources import resolved_resource_policy
from ade.engine.execution.gpu import acquire_gpu_lease, release_gpu_lease
from ade.harness.evaluation_supervisor import ProductionRayInspector, SubprocessEngineWorker


@ray.remote(num_gpus=1, num_cpus=1)
def identify_gpu():
    import subprocess
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    uuid = subprocess.check_output(
        ["nvidia-smi", "--id=" + visible, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True, timeout=20,
    ).strip()
    return {"node_id": ray.get_runtime_context().get_node_id(),
            "visible_device": visible, "uuid": uuid}


def worker(address, probe_id, worker_id, policy, ready, release):
    ray.init(address=address, namespace="ade", logging_level="ERROR")
    lease = None
    try:
        lease = acquire_gpu_lease(
            owner=f"{probe_id}/{worker_id}/operator_evaluation/probe",
            kind="operator_test", requested_gpus=24, minimum_gpus=24,
            coordinator_owner=f"{probe_id}/{worker_id}",
            workload="operator_evaluation", coordinator_policy=policy,
            timeout_seconds=90,
        )
        devices = ray.get([
            identify_gpu.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=lease["placement_group"], placement_group_bundle_index=i,
            )).remote() for i in range(24)
        ], timeout=120)
        ready.put({"worker_id": worker_id, "pid": os.getpid(),
                   "lease_id": lease["lease_id"], "devices": devices})
        deadline = time.monotonic() + 240
        while not Path(release).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("probe controller did not release worker")
            time.sleep(0.1)
    except Exception as error:
        ready.put({"worker_id": worker_id, "error": str(error)})
        raise
    finally:
        release_gpu_lease(lease)
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--authorized", action="store_true", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    probe_id = output.name
    policy = resolved_resource_policy(yaml.safe_load(
        (root / "configs/runtime/canonical-24gpu-coordinator-n3.yaml").read_text()
    )["runtime"])
    inspector = ProductionRayInspector()
    preflight = inspector.inspect(address=args.ray_address, required_gpus=72, require_idle=True)
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2))
    if preflight["status"] != "complete":
        raise RuntimeError(f"Ray admission blocked: {preflight['issues']}")
    context = multiprocessing.get_context("spawn")
    ready, release = context.Queue(), output / "release"
    if release.exists():
        raise ValueError("probe output already used; choose a new output directory")
    processes = []
    receipt = {"probe_id": probe_id, "address": args.ray_address,
               "scope": "resource allocation, GPU identity, worker replacement and cleanup"}

    def start(worker_id):
        process = context.Process(target=worker, args=(
            args.ray_address, probe_id, worker_id, policy, ready, release,
        ))
        process.start()
        processes.append(process)
        return process

    def cleanup_worker(worker_id):
        # Exercise the same exact-owner cleanup used by the production pool.
        manager = object.__new__(SubprocessEngineWorker)
        manager.operation = SimpleNamespace(ray_address=args.ray_address, evaluation_id=probe_id)
        manager.worker_id = worker_id
        manager._release_leases()

    try:
        initial = [start(f"c{i:03d}") for i in range(1, 4)]
        records = [ready.get(timeout=180) for _ in range(3)]
        assert not any("error" in item for item in records), records
        assert all(len(item["devices"]) == 24 for item in records)
        assert len({d["uuid"] for item in records for d in item["devices"]}) == 72
        receipt["workers"] = records
        receipt["concurrent"] = inspector.inspect(
            address=args.ray_address, required_gpus=72, require_idle=False)
        assert len(receipt["concurrent"]["allocator_leases"]) == 3
        # Kill only our second driver; the other two continue to own their GPUs.
        initial[1].kill()
        initial[1].join(10)
        cleanup_worker("c002")
        surviving = inspector.inspect(address=args.ray_address, required_gpus=72, require_idle=False)
        receipt["after_worker_loss"] = surviving
        assert {v["coordinator_owner"] for v in surviving["allocator_leases"]} == {
            f"{probe_id}/c001", f"{probe_id}/c003"}
        start("c002")
        replacement = ready.get(timeout=180)
        assert "error" not in replacement, replacement
        remaining = [item for item in records if item["worker_id"] != "c002"]
        assert len({d["uuid"] for item in remaining + [replacement] for d in item["devices"]}) == 72
        receipt["replacement"] = replacement
        receipt["status"] = "complete"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        release.touch()
        for process in processes:
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join(5)
        for worker_id in ("c001", "c002", "c003"):
            cleanup_worker(worker_id)
        for _ in range(10):
            terminal = inspector.inspect(address=args.ray_address, required_gpus=72, require_idle=True)
            if terminal["status"] == "complete":
                break
            time.sleep(2)
        receipt["terminal"] = terminal
        if terminal["status"] != "complete":
            receipt["status"] = "cleanup_failed"
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps({"status": receipt["status"], "receipt": str(output / "receipt.json")}))
    if receipt["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
