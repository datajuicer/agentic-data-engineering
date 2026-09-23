"""Task-owned math adapter for Engine-authoritative process evidence."""

from __future__ import annotations

import json
import math
from typing import Any


ADAPTER_ID = "math_process_evidence.v1"
SCHEMA_VERSION = "ade.process_evidence.v1"
DIMENSIONS = (
    "derivation_soundness",
    "conclusion_support",
    "substantive_relevance",
)

PROCESS_RUBRIC = json.dumps(
    {
        "template": (
            "Assess the mathematical response. Question: {{question}}\n"
            "Response: {{response}}"
        ),
        "required_variables": ["question", "response"],
        "output_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scores_by_dimension": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        name: {"enum": [0.0, 0.5, 1.0]} for name in DIMENSIONS
                    },
                    "required": list(DIMENSIONS),
                }
            },
            "required": ["scores_by_dimension"],
        },
        "projection": {
            "dimensions": [
                {
                    "id": name,
                    "criterion": criterion,
                    "weight": 1.0 / len(DIMENSIONS),
                    "score_levels": [
                        {"value": 0.0, "standard": "The criterion is not satisfied."},
                        {"value": 0.5, "standard": "The criterion is partially satisfied."},
                        {"value": 1.0, "standard": "The criterion is fully satisfied."},
                    ],
                }
                for name, criterion in (
                    ("derivation_soundness", "The derivation steps are mathematically sound."),
                    ("conclusion_support", "The derivation supports the stated conclusion."),
                    ("substantive_relevance", "The reasoning is substantive and relevant."),
                )
            ]
        },
    },
    sort_keys=True,
    separators=(",", ":"),
)


def available_process_evidence(evaluation: Any) -> dict[str, Any]:
    if not isinstance(evaluation, dict):
        raise ValueError("process evidence evaluation must be an object")
    scores = evaluation.get("scores_by_dimension")
    if evaluation.get("status") != "completed" or not isinstance(scores, dict):
        raise ValueError("process evidence evaluation is not available")
    if set(scores) != set(DIMENSIONS):
        raise ValueError("process evidence dimension set is invalid")
    dimensions: dict[str, float] = {}
    for name in DIMENSIONS:
        value = scores[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"process evidence dimension {name} is invalid")
        dimensions[name] = float(value)
    return {
        "status": "available",
        "adapter_id": ADAPTER_ID,
        "schema_version": SCHEMA_VERSION,
        "dimensions": dimensions,
    }


def unavailable_process_evidence(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "adapter_id": ADAPTER_ID,
        "schema_version": SCHEMA_VERSION,
        "dimensions": None,
        "reason": reason,
    }


def not_configured_process_evidence() -> dict[str, Any]:
    return {
        "status": "not_configured",
        "adapter_id": None,
        "schema_version": None,
        "dimensions": None,
    }
