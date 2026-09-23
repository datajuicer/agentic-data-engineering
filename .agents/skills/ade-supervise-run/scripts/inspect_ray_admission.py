#!/usr/bin/env python3
"""Inspect Ray placement groups, ADE leases and Judge actors without Dashboard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import ray


ALLOCATOR_NAME = "ade_gpu_lease_allocator"
ALLOCATOR_NAMESPACE = "ade"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read the Ray placement-group table directly from GCS and compare it "
            "with the existing ADE GPU lease allocator."
        )
    )
    parser.add_argument("--address", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def _placement_group_record(value: dict[str, Any]) -> dict[str, Any]:
    bundles = value.get("bundles") or {}
    return {
        "placement_group_id": str(value.get("placement_group_id") or ""),
        "name": str(value.get("name") or ""),
        "state": str(value.get("state") or ""),
        "strategy": str(value.get("strategy") or ""),
        "bundle_count": len(bundles),
        "bundles_to_node_id": dict(value.get("bundles_to_node_id") or {}),
    }


def _allocator_snapshot() -> tuple[str, list[dict[str, Any]]]:
    try:
        allocator = ray.get_actor(ALLOCATOR_NAME, namespace=ALLOCATOR_NAMESPACE)
    except ValueError:
        return "absent", []
    leases = ray.get(allocator.snapshot.remote())
    if not isinstance(leases, list):
        raise ValueError("ADE GPU lease allocator returned a non-list snapshot")
    return "present", [dict(lease) for lease in leases]


def main() -> int:
    args = _parser().parse_args()
    ray.init(
        address=args.address,
        namespace=ALLOCATOR_NAMESPACE,
        ignore_reinit_error=True,
        logging_level="ERROR",
    )
    try:
        placement_group_table = ray.util.placement_group_table()
        non_removed = sorted(
            (
                _placement_group_record(value)
                for value in placement_group_table.values()
                if value.get("state") != "REMOVED"
            ),
            key=lambda value: (value["name"], value["placement_group_id"]),
        )
        allocator_status, leases = _allocator_snapshot()
        judge_actors = sorted(item["name"] for item in ray.util.list_named_actors(all_namespaces=True)
                              if item["namespace"] == ALLOCATOR_NAMESPACE
                              and item["name"].startswith("ade-judge-"))
    finally:
        ray.shutdown()

    issues = []
    if non_removed:
        issues.append("non_removed_placement_groups")
    if leases:
        issues.append("allocator_leases")
    if judge_actors:
        issues.append("judge_actors")
    receipt = {
        "schema_version": "ade.ray_admission.v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "ray_address": args.address,
        "status": "complete" if not issues else "blocked",
        "placement_group_records": len(placement_group_table),
        "non_removed_placement_groups": non_removed,
        "allocator_actor": allocator_status,
        "allocator_leases": leases,
        "judge_actors": judge_actors,
        "issues": issues,
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
