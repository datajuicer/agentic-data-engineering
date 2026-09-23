"""Shared process-rubric construction and validation."""

from __future__ import annotations

import json
import math
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


def canonical_process_rubric(rubric: object) -> str:
    """Validate and return a canonical process-rubric JSON declaration."""
    if isinstance(rubric, str):
        parse_process_rubric(rubric, required=True)
        return rubric
    raise ValueError("selection judge rubric must be canonical process-rubric JSON")


def parse_process_rubric(
    value: object,
    *,
    required: bool = False,
) -> dict[str, object] | None:
    """Parse and validate the canonical Builder process-rubric declaration."""
    if not isinstance(value, str):
        if required:
            raise ValueError("process_rubric must be valid JSON")
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        if required:
            raise ValueError("process_rubric must be valid JSON")
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "template", "required_variables", "output_schema", "projection",
    }:
        if required:
            raise ValueError("process_rubric does not match ade.rubric_jobs.v1")
        return None
    if (
        not _text(payload.get("template"))
        or payload["required_variables"] != ["question", "response"]
        or "{{question}}" not in payload["template"]
        or "{{response}}" not in payload["template"]
    ):
        return _invalid(required)
    output_schema = payload["output_schema"]
    try:
        if not isinstance(output_schema, dict):
            return _invalid(required)
        Draft202012Validator.check_schema(output_schema)
    except SchemaError:
        return _invalid(required)
    if (
        output_schema.get("type") != "object"
        or output_schema.get("additionalProperties") is not False
        or output_schema.get("required") != ["scores_by_dimension"]
        or not isinstance(output_schema.get("properties"), dict)
        or "scores_by_dimension" not in output_schema["properties"]
    ):
        return _invalid(required)
    scores_schema = output_schema["properties"]["scores_by_dimension"]
    if (
        not isinstance(scores_schema, dict)
        or scores_schema.get("type") != "object"
        or scores_schema.get("additionalProperties") is not False
        or not isinstance(scores_schema.get("properties"), dict)
        or not isinstance(scores_schema.get("required"), list)
    ):
        return _invalid(required)
    projection = payload["projection"]
    if not isinstance(projection, dict) or set(projection) != {"dimensions"}:
        return _invalid(required)
    dimensions = projection["dimensions"]
    if not isinstance(dimensions, list) or not dimensions:
        return _invalid(required)
    seen: set[str] = set()
    dimension_ids: list[str] = []
    weights = 0.0
    for dimension in dimensions:
        if not isinstance(dimension, dict):
            return _invalid(required)
        dimension_id = dimension.get("id")
        criterion = dimension.get("criterion")
        weight = dimension.get("weight")
        levels = dimension.get("score_levels")
        if (
            not _text(dimension_id)
            or dimension_id in seen
            or not _text(criterion)
            or type(weight) not in {int, float}
            or not math.isfinite(weight)
            or weight < 0.0
            or not isinstance(levels, list)
            or len(levels) < 2
        ):
            return _invalid(required)
        values: set[float] = set()
        for level in levels:
            if not isinstance(level, dict):
                return _invalid(required)
            score = level.get("value")
            if type(score) not in {int, float} or not math.isfinite(score):
                return _invalid(
                    required,
                    f"dimension {dimension_id!r} score level must be a finite number",
                )
            if not 0.0 <= float(score) <= 1.0:
                return _invalid(
                    required,
                    f"dimension {dimension_id!r} score level {score!r} "
                    "must be within [0.0, 1.0]",
                )
            if float(score) in values or not _text(level.get("standard")):
                return _invalid(required)
            values.add(float(score))
        dimension_schema = scores_schema["properties"].get(dimension_id)
        if (
            not isinstance(dimension_schema, dict)
            or set(dimension_schema) != {"enum"}
            or not isinstance(dimension_schema["enum"], list)
            or {
                float(item)
                for item in dimension_schema["enum"]
                if type(item) in {int, float} and math.isfinite(item)
            } != values
            or len(dimension_schema["enum"]) != len(values)
        ):
            return _invalid(required)
        seen.add(dimension_id)
        dimension_ids.append(dimension_id)
        weights += float(weight)
    if not math.isclose(weights, 1.0, rel_tol=0.0, abs_tol=1e-9):
        return _invalid(required)
    if set(scores_schema["properties"]) != seen or scores_schema["required"] != dimension_ids:
        return _invalid(required)
    return payload


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _invalid(required: bool, detail: str | None = None) -> None:
    if required:
        message = "process_rubric does not match ade.rubric_jobs.v1"
        if detail:
            message = f"{message}: {detail}"
        raise ValueError(message)
    return None
