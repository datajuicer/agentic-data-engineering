"""Canonical Review Labor usage aggregation shared by producer and validator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


OPTIONAL_USAGE_FIELDS = ("cached_tokens", "reasoning_tokens")


def aggregate_usage(unit_usage: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = [str(item["usage_status"]) for item in unit_usage]
    if statuses and all(status == "complete" for status in statuses):
        status = "complete"
    elif statuses and all(status == "unavailable" for status in statuses):
        status = "unavailable"
    elif statuses:
        status = "partial"
    else:
        status = "unavailable"

    def total(field: str) -> int | None:
        values = [item.get(field) for item in unit_usage]
        return (
            sum(int(value) for value in values if value is not None)
            if any(value is not None for value in values)
            else None
        )

    result: dict[str, Any] = {
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"),
        "attempts": sum(int(item["attempts"]) for item in unit_usage),
        "requests": sum(
            int(item.get("requests", item["attempts"])) for item in unit_usage
        ),
        "retries": sum(
            int(item.get("retries", max(0, int(item["attempts"]) - 1)))
            for item in unit_usage
        ),
        "complete_units": statuses.count("complete"),
        "partial_units": statuses.count("partial"),
        "unavailable_units": statuses.count("unavailable"),
        "usage_status": status,
    }
    for field in OPTIONAL_USAGE_FIELDS:
        values = [item.get(field) for item in unit_usage]
        result[field] = (
            sum(int(value) for value in values)
            if values and all(value is not None for value in values)
            else None
        )
    return result
