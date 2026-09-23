"""Shared request and response shaping for Analyzer Review providers."""

from __future__ import annotations

import json
import math
from typing import Any


def rubric_job_messages(
    *,
    rendered_prompt: str,
    output_schema: dict[str, Any],
    validation_feedback: str | None,
) -> list[dict[str, str]]:
    """Build the provider-neutral Analyzer messages used by API and local jobs."""
    specs = review_schema_specs(output_schema)
    if specs is None:
        user_content = json.dumps(
            {
                "rendered_rubric": rendered_prompt,
                "output_schema": output_schema,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        system_content = (
            "Apply the supplied rubric and return only one JSON object that "
            "matches the supplied JSON Schema."
        )
    else:
        user_content = (
            rendered_prompt
            + "\nReturn JSON only in this compact shape: "
            + '{"judgments":[{"label":"one allowed label",'
            + '"evidence":"one short sentence","confidence":0.0}]}. '
            + f"The judgments array must contain exactly {len(specs)} items in "
            + "rubric order. Do not repeat rubric IDs or these output instructions."
        )
        if validation_feedback:
            user_content += (
                "\nThe previous attempt was invalid: "
                + validation_feedback
                + ". Generate the complete compact JSON object again."
            )
        system_content = "Return only the requested compact JSON object."
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def review_schema_specs(
    output_schema: dict[str, Any],
) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
    try:
        properties = output_schema["properties"]
        rubric_results = properties["rubric_results"]
        items = rubric_results["prefixItems"]
        if (
            set(output_schema)
            != {"type", "additionalProperties", "properties", "required"}
            or output_schema["type"] != "object"
            or output_schema["additionalProperties"] is not False
            or output_schema["required"] != ["rubric_results"]
            or rubric_results["minItems"] != len(items)
            or rubric_results["maxItems"] != len(items)
        ):
            return None
        specs = []
        for item in items:
            item_properties = item["properties"]
            rubric_id = item_properties["rubric_id"]["const"]
            labels = item_properties["label"]["enum"]
            if not isinstance(rubric_id, str) or not isinstance(labels, list):
                return None
            specs.append((rubric_id, tuple(str(label) for label in labels)))
        return tuple(specs)
    except (KeyError, TypeError):
        return None


def expand_compact_review_result(
    payload: dict[str, Any],
    specs: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, Any]:
    if set(payload) != {"judgments"} or not isinstance(payload["judgments"], list):
        raise ValueError("compact Review result must contain only judgments")
    judgments = payload["judgments"]
    if len(judgments) != len(specs):
        raise ValueError(
            f"compact Review result requires {len(specs)} judgments, received {len(judgments)}"
        )
    results = []
    for index, (item, (rubric_id, labels)) in enumerate(
        zip(judgments, specs, strict=True)
    ):
        if not isinstance(item, dict) or set(item) != {
            "label",
            "evidence",
            "confidence",
        }:
            raise ValueError(f"compact Review judgment {index} has invalid fields")
        label = item["label"]
        evidence = item["evidence"]
        confidence = item["confidence"]
        if not isinstance(label, str) or label not in labels:
            raise ValueError(f"compact Review judgment {index} has invalid label")
        if not isinstance(evidence, str):
            raise ValueError(f"compact Review judgment {index} has invalid evidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError(f"compact Review judgment {index} has invalid confidence")
        observation = evidence.strip()
        results.append(
            {
                "rubric_id": rubric_id,
                "label": label,
                "observations": [observation] if observation else [],
                "evidence_spans": [],
                "confidence": float(confidence),
            }
        )
    return {"rubric_results": results}
