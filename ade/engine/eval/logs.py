from __future__ import annotations

from typing import Any


def build_eval_log_payload(
    *,
    base_payload: dict[str, Any],
    dataset_path: Any,
    details: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "online_eval.v1",
        **base_payload,
        "dataset_path": dataset_path,
        "case_details": _case_details(details),
    }


def _case_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for detail in details:
        if not isinstance(detail, dict):
            raise ValueError("eval detail must be a mapping")
        shared_detail = {
            key: detail.get(key)
            for key in ("index", "accuracy", "correct_count", "total_count", "row_indices", "seed")
            if key in detail
        }
        cases: list[dict[str, Any]] = []
        for bucket in ("errors", "successes"):
            for case in detail.get(bucket, []) or []:
                if not isinstance(case, dict):
                    raise ValueError("eval case must be a mapping")
                cases.append(
                    {
                        "case_id": case.get("case_id"),
                        "bucket": case.get("bucket") or bucket,
                        "index": case.get("index"),
                        "example_id": case.get("example_id"),
                        "input": dict(case.get("input") or {}),
                        "gold": dict(case.get("gold") or {}),
                        "prediction": dict(case.get("prediction") or {}),
                        "diagnostics": dict(case.get("diagnostics") or {}),
                        "metadata": dict(case.get("metadata") or {}),
                    }
                )
        normalized.append({**shared_detail, "cases": cases})
    return normalized
