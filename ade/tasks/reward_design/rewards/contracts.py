"""Fixed interface shared by generated rewards and the VERL runtime."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import Any


REWARD_RESULT_SCHEMA = "ade.reward_result.v2"
REWARD_PARAMETER_NAMES = (
    "question_prompt",
    "response_content",
    "extracted_answer",
    "outcome_score",
    "response_length_tokens",
    "max_response_length_tokens",
)


def validate_reward_function(function: Any) -> Callable[..., Any]:
    if not callable(function):
        raise TypeError("compute_score must be callable")
    parameters = tuple(inspect.signature(function).parameters.values())
    if tuple(parameter.name for parameter in parameters) != REWARD_PARAMETER_NAMES or any(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameter.default is not inspect.Parameter.empty
        for parameter in parameters
    ):
        raise TypeError(
            "compute_score must have the exact signature "
            "(question_prompt, response_content, extracted_answer, outcome_score, "
            "response_length_tokens, max_response_length_tokens)"
        )
    return function


def validate_reward_score(value: Any, *, low: float = 0.0, high: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("reward must be a finite number")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("reward must be finite")
    if score < low or score > high:
        raise ValueError(f"reward {score} is outside configured reward range [{low}, {high}]")
    return score


def validate_reward_result(
    value: Any, *, fallback: bool, group_credit_enabled: bool = False
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("generated reward must return an ade.reward_result.v2 object")
    expected_fields = {
        "schema_version",
        "score",
        "outcome_score",
        "artifact_projection",
        "rule_evidence",
    }
    if set(value) != expected_fields or value.get("schema_version") != REWARD_RESULT_SCHEMA:
        raise ValueError("generated reward result fields do not match ade.reward_result.v2")
    score = validate_reward_score(value["score"])
    outcome = validate_reward_score(value["outcome_score"])
    projection_value = value["artifact_projection"]
    projection = (
        None if projection_value is None else validate_reward_score(projection_value)
    )
    rule = _validate_rule_evidence(value["rule_evidence"])
    if fallback:
        if projection is not None:
            raise ValueError("fallback reward artifact_projection must be null")
        if not math.isclose(score, outcome, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("fallback reward score must equal outcome")
    if group_credit_enabled:
        if projection is not None:
            raise ValueError(
                "enabled Group Credit reward artifact_projection must be null"
            )
        if not math.isclose(score, outcome, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "enabled Group Credit pre-group score must equal outcome"
            )
    return {
        "schema_version": REWARD_RESULT_SCHEMA,
        "score": score,
        "outcome_score": outcome,
        "artifact_projection": projection,
        "rule_evidence": rule,
    }


def _validate_rule_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"status", "value"}:
        raise ValueError("rule_evidence must contain exactly status and value")
    status = value.get("status")
    if status == "available":
        return {"status": status, "value": validate_reward_score(value.get("value"))}
    if status == "not_configured" and value.get("value") is None:
        return {"status": status, "value": None}
    raise ValueError("rule_evidence status/value is invalid")


def enforce_outcome_component(
    result: dict[str, Any], outcome_score: float
) -> dict[str, Any]:
    expected = validate_reward_score(outcome_score)
    actual = result["outcome_score"]
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("generated reward outcome_score does not match Engine outcome_score")
    return result
